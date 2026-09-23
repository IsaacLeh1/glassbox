import type { BPETokenizer } from './tokenizer';
import type { Transformer } from './transformer';
import { getCorpus } from './corpus';
import { mulberry32 } from './tensor';

/**
 * Watching a model after it has shipped.
 *
 * Everything before this point had an answer key. Training had targets,
 * evaluation had a held-out set with the right continuations written down.
 * Production has neither. Real users send text nobody has labelled, the model
 * answers, and almost nobody says whether the answer was any good.
 *
 * So monitoring is the discipline of noticing that something has gone wrong
 * using only signals that need no ground truth at all:
 *
 *  - how surprised the model is by what it is being sent,
 *  - how far the distribution of incoming text has moved from what it was
 *    trained on,
 *  - how much of that text the tokenizer cannot even represent,
 *  - and what the outputs look like in aggregate.
 *
 * None of those measures quality directly. All of them move before quality
 * collapses, which is the entire reason they are worth watching. This module
 * computes the proxies and the true quality side by side, so a reader can see
 * how well the proxies track the thing they stand in for -- and then remember
 * that in production only the left-hand column exists.
 */

/**
 * Per-position negative log probability, from one forward pass.
 *
 * Row t of the logits is the model's prediction for position t+1, so a single
 * pass already contains the score for every position. This is the same
 * quantity teacher forcing computes one token at a time, and a test asserts
 * the two agree.
 */
function positionLosses(logits: { rows: number; cols: number; data: Float64Array }, targets: number[]): number[] {
  const out: number[] = [];
  const V = logits.cols;
  for (let t = 0; t < targets.length && t < logits.rows; t++) {
    const off = t * V;
    let max = -Infinity;
    for (let j = 0; j < V; j++) if (logits.data[off + j] > max) max = logits.data[off + j];
    let sum = 0;
    for (let j = 0; j < V; j++) sum += Math.exp(logits.data[off + j] - max);
    out.push(-(logits.data[off + targets[t]] - max - Math.log(sum)));
  }
  return out;
}

/* ---------------------------------------------------------- histograms */

/** Share of each vocabulary entry in a token stream. */
export function tokenHistogram(ids: number[], vocab: number): Float64Array {
  const h = new Float64Array(vocab);
  for (const id of ids) if (id >= 0 && id < vocab) h[id]++;
  const n = Math.max(1, ids.length);
  for (let i = 0; i < vocab; i++) h[i] /= n;
  return h;
}

/**
 * Population Stability Index, the standard industry drift measure.
 *
 * Compares how much probability mass moved between two distributions. The
 * conventional reading is: below 0.1 nothing has happened, 0.1 to 0.25 is a
 * moderate shift worth investigating, and above 0.25 the input distribution
 * has genuinely changed and whatever was validated on the old one may no
 * longer hold.
 *
 * Both sides are floored away from zero, because a category that appears in
 * one window and not the other would otherwise contribute infinity.
 */
export function psi(expected: Float64Array, actual: Float64Array, floor = 1e-4): number {
  let total = 0;
  for (let i = 0; i < expected.length; i++) {
    const e = Math.max(floor, expected[i]);
    const a = Math.max(floor, actual[i]);
    total += (a - e) * Math.log(a / e);
  }
  return total;
}

/**
 * Jensen-Shannon divergence, in nats. Symmetric, bounded by ln 2.
 *
 * Reported alongside PSI because it is bounded, which makes it easier to read
 * as a fraction of the worst possible case. PSI is unbounded and its
 * thresholds are convention rather than mathematics.
 */
export function jsDivergence(p: Float64Array, q: Float64Array): number {
  let kl1 = 0;
  let kl2 = 0;
  for (let i = 0; i < p.length; i++) {
    const m = (p[i] + q[i]) / 2;
    if (m <= 0) continue;
    if (p[i] > 0) kl1 += p[i] * Math.log(p[i] / m);
    if (q[i] > 0) kl2 += q[i] * Math.log(q[i] / m);
  }
  return (kl1 + kl2) / 2;
}

export const PSI_BANDS = [
  { max: 0.1, label: 'stable', tone: 'ok' as const },
  { max: 0.25, label: 'moving', tone: 'warn' as const },
  { max: Infinity, label: 'shifted', tone: 'err' as const },
];

export function psiBand(v: number) {
  return PSI_BANDS.find((b) => v < b.max) ?? PSI_BANDS[PSI_BANDS.length - 1];
}

/* -------------------------------------------------------------- traffic */

export interface Request {
  text: string;
  /** True when this request came from the distribution the model was trained on. */
  familiar: boolean;
}

export interface Window {
  index: number;
  requests: Request[];
  /** Fraction of this window drawn from the unfamiliar source. */
  drift: number;

  /* ---- signals available in production, with no labels at all ---- */
  /** Mean per-token surprise on incoming text. Rises on unfamiliar input. */
  surprise: number;
  /** Population stability index against the reference window. */
  psi: number;
  jsd: number;
  /** Share of characters the tokenizer has no entry for. */
  oov: number;
  /** Share of generated tokens that simply repeated the previous one. */
  repetition: number;

  /* ---- the thing you cannot see in production ---- */
  /** Real quality on this traffic, which needs the answers. */
  trueQuality: number;
}

/**
 * Build one window of traffic by mixing two sources.
 *
 * Both sources are real corpora and the model really processes every request,
 * so every signal below is measured rather than posited. The only thing
 * chosen is how much of each source goes in, which is the experiment.
 */
export function buildWindow(
  model: Transformer,
  tok: BPETokenizer,
  familiarText: string,
  strangeText: string,
  drift: number,
  index: number,
  perWindow: number,
  seed: number,
  reference: Float64Array | null,
): Window {
  const rand = mulberry32(seed + index * 1013);
  const famLines = familiarText.split(/(?<=\.)\s+/).filter((l) => l.trim().length > 12);
  const strLines = strangeText.split(/(?<=\.)\s+/).filter((l) => l.trim().length > 12);

  const requests: Request[] = [];
  for (let i = 0; i < perWindow; i++) {
    const unfamiliar = rand() < drift;
    const pool = unfamiliar ? strLines : famLines;
    if (pool.length === 0) continue;
    requests.push({ text: pool[Math.floor(rand() * pool.length)], familiar: !unfamiliar });
  }

  const allIds: number[] = [];
  let oovChars = 0;
  let totalChars = 0;
  let surpriseSum = 0;
  let surpriseN = 0;
  let quality = 0;
  let qualityN = 0;
  let repeats = 0;
  let repeatN = 0;

  const block = model.cfg.blockSize;

  for (const r of requests) {
    for (const ch of Array.from(r.text)) {
      totalChars++;
      if (!tok.index.has(ch)) oovChars++;
    }

    const ids = tok.encode(r.text).slice(0, block);
    allIds.push(...ids);
    if (ids.length < 4) continue;

    // Split each request into a prompt and the continuation that really
    // followed it. The surprise uses only the prompt side, which is what a
    // production system could compute; the quality uses the continuation,
    // which it could not.
    const cut = Math.max(2, Math.floor(ids.length / 2));

    // One forward pass scores every position at once, because the logits
    // row at position t is already the prediction for t+1. Walking the
    // sequence token by token would give the identical numbers and cost a
    // whole forward pass each time -- roughly ten times the work for a
    // dashboard that has to keep up with live traffic.
    const { trace } = model.forward(ids);
    const perPos = positionLosses(trace.logits, ids.slice(1));

    // Surprise on the incoming text itself: no answer key needed, because
    // the text is its own target.
    const head = perPos.slice(0, cut - 1);
    if (head.length > 0) {
      surpriseSum += head.reduce((a, b) => a + b, 0) / head.length;
      surpriseN++;
    }

    // The honest quality measure, which production does not have.
    const tail = perPos.slice(cut - 1);
    if (tail.length > 0) {
      quality += Math.exp(-(tail.reduce((a, b) => a + b, 0) / tail.length));
      qualityN++;
    }

    for (let i = 1; i < ids.length; i++) {
      repeatN++;
      if (ids[i] === ids[i - 1]) repeats++;
    }
  }

  const hist = tokenHistogram(allIds, tok.size);

  return {
    index,
    requests,
    drift,
    surprise: surpriseN > 0 ? surpriseSum / surpriseN : NaN,
    psi: reference ? psi(reference, hist) : 0,
    jsd: reference ? jsDivergence(reference, hist) : 0,
    oov: totalChars > 0 ? oovChars / totalChars : 0,
    repetition: repeatN > 0 ? repeats / repeatN : 0,
    trueQuality: qualityN > 0 ? quality / qualityN : NaN,
  };
}

/** The reference distribution a deployment is compared against for ever after. */
export function referenceHistogram(tok: BPETokenizer, text: string): Float64Array {
  return tokenHistogram(tok.encode(text), tok.size);
}

/* ------------------------------------------------------------ scenarios */

export type Scenario = 'steady' | 'sudden' | 'gradual' | 'spike';

export const SCENARIO_INFO: Record<Scenario, { label: string; blurb: string }> = {
  steady: {
    label: 'Nothing happens',
    blurb:
      'Traffic stays exactly what the model was built for. Worth running first, because whatever your alert does here is a false alarm by definition.',
  },
  sudden: {
    label: 'Something changes overnight',
    blurb:
      'A product launches, a new customer arrives, a competitor goes down. The distribution moves in one step and stays moved. The easiest kind to detect and the rarest.',
  },
  gradual: {
    label: 'The world drifts',
    blurb:
      'No single day looks different from the one before it. This is the common case and the dangerous one, because no threshold ever gets crossed on any given day while the model quietly stops being right.',
  },
  spike: {
    label: 'A bad afternoon',
    blurb:
      'A temporary surge of unusual traffic that then goes away. Worth seeing because an alert that fires here has arguably done its job, and a retrain triggered by it would have been wasted money.',
  },
};

/** The drift fraction at each step of a scenario. */
export function schedule(scenario: Scenario, n: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const t = i / Math.max(1, n - 1);
    if (scenario === 'steady') out.push(0);
    else if (scenario === 'sudden') out.push(t < 0.45 ? 0 : 0.85);
    else if (scenario === 'gradual') out.push(t * 0.9);
    else out.push(t > 0.4 && t < 0.62 ? 0.9 : 0);
  }
  return out;
}

/* -------------------------------------------------------------- alerting */

export type SignalId = 'surprise' | 'psi' | 'oov' | 'repetition';

export const SIGNAL_INFO: Record<SignalId, { label: string; blurb: string; available: boolean }> = {
  surprise: {
    label: 'Model surprise',
    blurb:
      'How unexpected the incoming text is to the model, measured as its own loss on it. Needs no labels because the text is its own target. The most generally useful signal there is.',
    available: true,
  },
  psi: {
    label: 'Input drift (PSI)',
    blurb:
      'How far the distribution of incoming tokens has moved from the day you deployed. Says nothing about whether the model is coping, only that the question has changed.',
    available: true,
  },
  oov: {
    label: 'Unreadable characters',
    blurb:
      'Share of incoming characters the tokenizer has no entry for. A blunt signal that goes off when the traffic is a different language, a different alphabet, or not prose at all.',
    available: true,
  },
  repetition: {
    label: 'Repetition',
    blurb:
      'How often a token simply repeats the one before it. A model that has come apart tends to loop, so this catches a specific and common failure.',
    available: true,
  },
};

export interface AlertResult {
  /** Windows where the alert fired. */
  fired: number[];
  /** Fired while traffic was genuinely normal. */
  falseAlarms: number;
  /** Did not fire while traffic was genuinely drifted. */
  missed: number;
  /** Fired while traffic was genuinely drifted. */
  caught: number;
  /** Windows that were genuinely drifted at all. */
  drifted: number;
  /** Index of the first window where drift began, or -1. */
  driftBegan: number;
  /** Windows between drift starting and the alert firing, or -1 if never. */
  delay: number;
}

const valueOf = (w: Window, s: SignalId): number =>
  s === 'surprise' ? w.surprise : s === 'psi' ? w.psi : s === 'oov' ? w.oov : w.repetition;

/**
 * Score a threshold against what actually happened.
 *
 * The tradeoff is the whole subject. A threshold tight enough to catch
 * everything wakes somebody up most nights for nothing, and an alert that
 * cries wolf is an alert people learn to close without reading. A threshold
 * loose enough never to be wrong misses the thing it exists for.
 */
export function evaluateAlert(windows: Window[], signal: SignalId, threshold: number): AlertResult {
  const fired: number[] = [];
  let falseAlarms = 0;
  let missed = 0;
  let caught = 0;
  let drifted = 0;
  let driftBegan = -1;
  let delay = -1;

  windows.forEach((w, i) => {
    const isDrifted = w.drift > 0.2;
    if (isDrifted) {
      drifted++;
      if (driftBegan < 0) driftBegan = i;
    }
    const v = valueOf(w, signal);
    const alarm = Number.isFinite(v) && v >= threshold;
    if (alarm) {
      fired.push(i);
      if (isDrifted) caught++;
      else falseAlarms++;
      if (driftBegan >= 0 && delay < 0 && i >= driftBegan) delay = i - driftBegan;
    } else if (isDrifted) {
      missed++;
    }
  });

  return { fired, falseAlarms, missed, caught, drifted, driftBegan, delay };
}

/** Sweep every sensible threshold, for choosing one with eyes open. */
export function thresholdSweep(
  windows: Window[],
  signal: SignalId,
  steps = 40,
): { threshold: number; falseAlarms: number; missed: number; caught: number }[] {
  const values = windows.map((w) => valueOf(w, signal)).filter((v) => Number.isFinite(v));
  if (values.length === 0) return [];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const out = [];
  for (let i = 0; i <= steps; i++) {
    const threshold = lo + ((hi - lo) * i) / steps;
    const r = evaluateAlert(windows, signal, threshold);
    out.push({ threshold, falseAlarms: r.falseAlarms, missed: r.missed, caught: r.caught });
  }
  return out;
}

/* ------------------------------------------------------- retrain or not */

export interface RetrainCase {
  /** Quality now, against quality when it shipped. */
  degradation: number;
  /** Dollars the degradation costs per day, at the stated value per request. */
  lossPerDay: number;
  retrainCost: number;
  /** Days for the retrain to pay for itself. Infinite if it never does. */
  paybackDays: number;
  worthIt: boolean;
}

/**
 * Whether the degradation justifies the bill.
 *
 * This is where monitoring stops being a dashboard and becomes a decision.
 * Retraining is not free: it costs compute, it costs the time of the people
 * who run it, and it puts a model into production that has not been through
 * the evaluation the old one passed. The question is never "has it drifted",
 * it is "has it drifted enough to be worth the risk and the money".
 */
export function retrainCase(
  qualityNow: number,
  qualityAtLaunch: number,
  requestsPerDay: number,
  valuePerRequest: number,
  retrainCost: number,
): RetrainCase {
  const degradation = qualityAtLaunch > 0 ? Math.max(0, (qualityAtLaunch - qualityNow) / qualityAtLaunch) : 0;
  const lossPerDay = degradation * requestsPerDay * valuePerRequest;
  const paybackDays = lossPerDay > 0 ? retrainCost / lossPerDay : Infinity;
  return {
    degradation,
    lossPerDay,
    retrainCost,
    paybackDays,
    // A month is the conventional patience for this kind of spend.
    worthIt: paybackDays <= 30,
  };
}

/** The four built-in corpora, as the two sides of a drift experiment. */
export function corpusText(id: string): string {
  return getCorpus(id, 12);
}
