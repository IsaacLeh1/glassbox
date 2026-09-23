import type { BPETokenizer } from './tokenizer';
import type { Transformer } from './transformer';
import { mulberry32 } from './tensor';

/**
 * Measuring a model, and the discipline of deciding how to measure it before
 * you have anything to measure.
 *
 * The order is the whole point. A team that trains first and evaluates second
 * will, without ever meaning to, choose the metric that makes the result look
 * good, drop the cases it fails, and move the bar to wherever it landed. None
 * of that requires dishonesty; it only requires seeing the score first.
 *
 * Everything here is built so the comparison cannot quietly cheat:
 *
 *  - Baselines and the model go through one identical code path. A harness
 *    that scores them differently is the commonest way a result gets
 *    overstated, and it is invisible once it happens.
 *  - Generation for scoring is greedy by default, because a sampled eval
 *    measures the model and the dice together.
 *  - Contamination is measured, not assumed. A case the model was trained on
 *    is not a test of anything.
 */

/* ------------------------------------------------------------ predictors */

/**
 * Anything that can say what comes next.
 *
 * The transformer and every trivial baseline implement this, so `runEval`
 * cannot treat them differently even by accident.
 */
export interface Predictor {
  id: string;
  label: string;
  blurb: string;
  /** Probability over the whole vocabulary for the token after `ctx`. */
  next(ctx: number[]): Float64Array;
}

export function modelPredictor(model: Transformer): Predictor {
  return {
    id: 'model',
    label: 'Your model',
    blurb: 'The transformer you trained.',
    next(ctx) {
      // Windowed, because a model has a fixed number of positions and reading
      // past them is an error rather than something to paper over.
      const win = ctx.slice(-model.cfg.blockSize);
      const { logits } = model.predictNext(win.length > 0 ? win : [0]);
      // The model emits raw scores. A Predictor promises a distribution, so
      // the softmax happens here rather than being assumed by the caller.
      let max = -Infinity;
      for (const x of logits) if (x > max) max = x;
      const out = new Float64Array(logits.length);
      let sum = 0;
      for (let i = 0; i < logits.length; i++) {
        out[i] = Math.exp(logits[i] - max);
        sum += out[i];
      }
      for (let i = 0; i < out.length; i++) out[i] /= sum;
      return out;
    },
  };
}

/** Counts of every token, and of every token given the one before it. */
interface Counts {
  uni: Float64Array;
  bi: Map<number, Float64Array>;
}

function countFrom(ids: number[], vocab: number): Counts {
  const uni = new Float64Array(vocab);
  const bi = new Map<number, Float64Array>();
  for (let i = 0; i < ids.length; i++) {
    uni[ids[i]]++;
    if (i > 0) {
      let row = bi.get(ids[i - 1]);
      if (!row) {
        row = new Float64Array(vocab);
        bi.set(ids[i - 1], row);
      }
      row[ids[i]]++;
    }
  }
  return { uni, bi };
}

function normalise(counts: Float64Array, smoothing = 0.1): Float64Array {
  const out = new Float64Array(counts.length);
  let total = 0;
  for (let i = 0; i < counts.length; i++) total += counts[i] + smoothing;
  for (let i = 0; i < counts.length; i++) out[i] = (counts[i] + smoothing) / total;
  return out;
}

export type BaselineId = 'uniform' | 'unigram' | 'bigram' | 'repeat';

export const BASELINE_INFO: Record<BaselineId, { label: string; blurb: string }> = {
  uniform: {
    label: 'Pure chance',
    blurb:
      'Picks any vocabulary entry with equal probability. The absolute floor. A model that cannot beat this has learned nothing at all.',
  },
  unigram: {
    label: 'Most common token',
    blurb:
      'Ignores the context entirely and answers with whatever is most frequent in the training text. Surprisingly hard to beat on short outputs, because common words really are common.',
  },
  bigram: {
    label: 'Previous token only',
    blurb:
      'Looks at exactly one token of context and answers with whatever usually follows it. No attention, no layers, a lookup table. This is the baseline that embarrasses undertrained transformers.',
  },
  repeat: {
    label: 'Say it again',
    blurb:
      'Repeats the last token of the context forever. Scores zero on anything sensible, and is worth running because a broken model often produces exactly this.',
  },
};

/**
 * Build a trivial predictor from the training text.
 *
 * These are not strawmen. A bigram table is a real language model, it is what
 * the field used before neural networks, and a small transformer that has not
 * trained long enough genuinely loses to it. Finding that out is the point.
 */
export function makeBaseline(id: BaselineId, trainIds: number[], vocab: number): Predictor {
  const info = BASELINE_INFO[id];

  if (id === 'uniform') {
    const flat = new Float64Array(vocab).fill(1 / Math.max(1, vocab));
    return { id, label: info.label, blurb: info.blurb, next: () => flat };
  }

  if (id === 'repeat') {
    return {
      id,
      label: info.label,
      blurb: info.blurb,
      next(ctx) {
        const out = new Float64Array(vocab);
        const last = ctx.length > 0 ? ctx[ctx.length - 1] : 0;
        out[Math.min(Math.max(0, last), vocab - 1)] = 1;
        return out;
      },
    };
  }

  const counts = countFrom(trainIds, vocab);
  const uni = normalise(counts.uni);

  if (id === 'unigram') {
    return { id, label: info.label, blurb: info.blurb, next: () => uni };
  }

  // Bigram, backing off to the unigram distribution for an unseen context.
  const cache = new Map<number, Float64Array>();
  return {
    id,
    label: info.label,
    blurb: info.blurb,
    next(ctx) {
      const last = ctx.length > 0 ? ctx[ctx.length - 1] : -1;
      const row = counts.bi.get(last);
      if (!row) return uni;
      let p = cache.get(last);
      if (!p) {
        p = normalise(row);
        cache.set(last, p);
      }
      return p;
    },
  };
}

/* ----------------------------------------------------------- generation */

export interface GenConfig {
  /** Greedy takes the single most likely token every time, and is repeatable. */
  greedy: boolean;
  temperature: number;
  seed: number;
}

export const DEFAULT_GEN: GenConfig = { greedy: true, temperature: 0.8, seed: 1 };

function pick(probs: Float64Array, cfg: GenConfig, rand: () => number): number {
  if (cfg.greedy) {
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    return best;
  }
  const t = Math.max(1e-3, cfg.temperature);
  // Re-temper an existing distribution: raise to the power 1/t and renormalise.
  let sum = 0;
  const w = new Float64Array(probs.length);
  for (let i = 0; i < probs.length; i++) {
    w[i] = Math.pow(probs[i], 1 / t);
    sum += w[i];
  }
  let r = rand() * sum;
  for (let i = 0; i < w.length; i++) {
    r -= w[i];
    if (r <= 0) return i;
  }
  return w.length - 1;
}

/** Continue a context by n tokens, using any predictor. */
export function continueWith(pred: Predictor, ctx: number[], n: number, cfg: GenConfig): number[] {
  const rand = mulberry32(cfg.seed);
  let ids = [...ctx];
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const id = pick(pred.next(ids), cfg, rand);
    out.push(id);
    ids = [...ids, id];
  }
  return out;
}

/**
 * Mean negative log probability the predictor assigns to the expected
 * continuation, with the true tokens fed back in at every step.
 *
 * This is teacher forcing, and it is a different question from "what would it
 * have said". It asks how surprised the predictor is by the right answer,
 * which is the question training itself optimises.
 */
export function teacherForcedLoss(pred: Predictor, ctx: number[], expected: number[]): number {
  if (expected.length === 0) return NaN;
  let total = 0;
  let ids = [...ctx];
  for (const want of expected) {
    const p = pred.next(ids);
    const q = Math.max(1e-12, p[want] ?? 1e-12);
    total += -Math.log(q);
    ids = [...ids, want];
  }
  return total / expected.length;
}

/* ---------------------------------------------------------------- cases */

/** Where a case came from, which decides whether it proves anything. */
export type CaseOrigin = 'written' | 'heldout' | 'train';

export interface EvalCase {
  id: string;
  prompt: string;
  expected: string;
  origin: CaseOrigin;
}

export const ORIGIN_INFO: Record<CaseOrigin, { label: string; blurb: string; trustworthy: boolean }> = {
  written: {
    label: 'You wrote it',
    blurb: 'Written by hand. Trustworthy only if the model has genuinely never seen this text.',
    trustworthy: true,
  },
  heldout: {
    label: 'Held out',
    blurb:
      'Taken from the slice of the corpus that was deliberately kept back from training. This is the honest kind.',
    trustworthy: true,
  },
  train: {
    label: 'From the training set',
    blurb:
      'Taken from text the model was trained on. Any score here is inflated and measures memorisation, not ability.',
    trustworthy: false,
  },
};

/**
 * Cut evaluation cases out of a token stream.
 *
 * Used for both the held-out split and, deliberately, the training split, so
 * the difference between the two scores can be seen rather than described.
 */
export function casesFromIds(
  tok: BPETokenizer,
  ids: number[],
  origin: CaseOrigin,
  count: number,
  promptLen: number,
  answerLen: number,
): EvalCase[] {
  const out: EvalCase[] = [];
  const span = promptLen + answerLen;
  if (ids.length < span + 1) return out;
  for (let k = 0; k < count; k++) {
    // Evenly spaced rather than random, so the same corpus always yields the
    // same cases and two runs are comparable.
    const i = Math.floor((k / count) * (ids.length - span - 1));
    out.push({
      id: `${origin}-${k}`,
      prompt: tok.decode(ids.slice(i, i + promptLen)),
      expected: tok.decode(ids.slice(i + promptLen, i + span)),
      origin,
    });
  }
  return out;
}

/* -------------------------------------------------------------- metrics */

export type MetricId = 'exact' | 'contains' | 'firstToken' | 'topK' | 'quality';

export interface MetricInfo {
  label: string;
  blurb: string;
  /** True when each case is a pass or a fail, which is what a rate needs. */
  binary: boolean;
  /** What the metric misses, stated plainly. Every metric has one. */
  blindSpot: string;
}

export const METRIC_INFO: Record<MetricId, MetricInfo> = {
  exact: {
    label: 'Exact match',
    blurb: 'The generated continuation must equal the expected one, character for character.',
    binary: true,
    blindSpot:
      'Brutally strict. An answer that is right in every way that matters but differs by one space scores zero, so a real improvement can show up as no change at all.',
  },
  contains: {
    label: 'Contains the answer',
    blurb: 'The expected text must appear somewhere in what the model produced.',
    binary: true,
    blindSpot:
      'Easy to game. A model that produces a long rambling output containing everything will score well, and length alone can lift the number.',
  },
  firstToken: {
    label: 'First token right',
    blurb: 'Only the very first token of the continuation has to match.',
    binary: true,
    blindSpot:
      'Measures almost nothing about the answer as a whole, but it is stable and it moves early in training when everything else is still flat.',
  },
  topK: {
    label: 'Right answer in the top few',
    blurb: 'The correct next token has to be among the model\'s highest-scoring candidates.',
    binary: true,
    blindSpot:
      'Generous. It rewards a model for nearly knowing, which is useful during development and misleading in a headline.',
  },
  quality: {
    label: 'Probability of the right answer',
    blurb:
      'How much probability the model puts on the expected text, with the true tokens fed back in at each step. Reported as a per-token average.',
    binary: false,
    blindSpot:
      'Continuous and sensitive, so it improves smoothly and flatters early progress. It also never tells you whether the model would actually have said the right thing unprompted.',
  },
};

export interface MetricConfig {
  metric: MetricId;
  /** How many candidates count for the top-k metric. */
  topK: number;
}

export interface CaseResult {
  case: EvalCase;
  got: string;
  /** 1 or 0 for a binary metric; a value in [0,1] otherwise. Higher is better. */
  score: number;
  pass: boolean;
  detail: string;
}

/* --------------------------------------------------------------- the run */

export interface EvalRun {
  predictorId: string;
  predictorLabel: string;
  metric: MetricId;
  results: CaseResult[];
  /** Mean score. For a binary metric this is the pass rate. */
  mean: number;
  /** 95% interval on the mean. Wilson for a rate, normal for a mean. */
  lo: number;
  hi: number;
  total: number;
  passed: number;
  ms: number;
}

/**
 * Wilson score interval for a pass rate.
 *
 * The obvious formula, p plus or minus 1.96 root p(1-p)/n, is wrong at the
 * sizes an eval set actually has: at 8 cases and 8 passes it reports an
 * interval of zero width, claiming certainty from almost no evidence. Wilson
 * does not do that, which is exactly why it is worth using here.
 */
export function wilson(k: number, n: number, z = 1.96): { lo: number; hi: number } {
  if (n === 0) return { lo: 0, hi: 1 };
  const p = k / n;
  const z2 = z * z;
  const denom = 1 + z2 / n;
  const centre = p + z2 / (2 * n);
  const half = z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n));
  return {
    lo: Math.max(0, (centre - half) / denom),
    hi: Math.min(1, (centre + half) / denom),
  };
}

/** Run one predictor over one set of cases. The only scoring path there is. */
export function runEval(
  pred: Predictor,
  tok: BPETokenizer,
  cases: EvalCase[],
  cfg: MetricConfig,
  gen: GenConfig,
): EvalRun {
  const t0 = performance.now();
  const results: CaseResult[] = [];

  for (const c of cases) {
    const ctx = tok.encode(c.prompt);
    const want = tok.encode(c.expected);
    let score = 0;
    let got = '';
    let detail = '';

    if (cfg.metric === 'quality') {
      const loss = teacherForcedLoss(pred, ctx, want);
      // exp(-mean loss) is the geometric mean probability per token: a
      // number in [0,1] that reads as "how likely it thought the truth was".
      score = Number.isFinite(loss) ? Math.exp(-loss) : 0;
      got = `${score.toFixed(4)} per token`;
      detail = `mean loss ${loss.toFixed(3)}`;
    } else if (cfg.metric === 'topK') {
      const p = pred.next(ctx);
      const target = want[0];
      let rank = 0;
      for (let i = 0; i < p.length; i++) if (p[i] > p[target]) rank++;
      score = rank < cfg.topK ? 1 : 0;
      got = tok.display(target);
      detail = `ranked ${rank + 1} of ${p.length}`;
    } else if (cfg.metric === 'firstToken') {
      const out = continueWith(pred, ctx, 1, gen);
      score = out[0] === want[0] ? 1 : 0;
      got = tok.display(out[0]);
      detail = `wanted ${tok.display(want[0])}`;
    } else {
      const out = continueWith(pred, ctx, Math.max(1, want.length), gen);
      got = tok.decode(out);
      if (cfg.metric === 'exact') {
        score = got.trim() === c.expected.trim() ? 1 : 0;
        detail = score ? 'identical' : 'differs';
      } else {
        score = got.includes(c.expected.trim()) && c.expected.trim().length > 0 ? 1 : 0;
        detail = score ? 'found' : 'not present';
      }
    }

    results.push({ case: c, got, score, pass: score >= (METRIC_INFO[cfg.metric].binary ? 1 : 0.5), detail });
  }

  const n = results.length;
  const mean = n > 0 ? results.reduce((a, r) => a + r.score, 0) / n : 0;
  const passed = results.filter((r) => r.pass).length;

  let lo = 0;
  let hi = 0;
  if (METRIC_INFO[cfg.metric].binary) {
    ({ lo, hi } = wilson(passed, n));
  } else if (n > 0) {
    let v = 0;
    for (const r of results) v += (r.score - mean) ** 2;
    const se = Math.sqrt(v / Math.max(1, n)) / Math.sqrt(n);
    lo = Math.max(0, mean - 1.96 * se);
    hi = Math.min(1, mean + 1.96 * se);
  }

  return {
    predictorId: pred.id,
    predictorLabel: pred.label,
    metric: cfg.metric,
    results,
    mean,
    lo,
    hi,
    total: n,
    passed,
    ms: performance.now() - t0,
  };
}

/* -------------------------------------------------------- contamination */

export interface Contamination {
  /** Fraction of the case's character n-grams that also occur in the training text. */
  overlap: number;
  /** True when the whole case appears verbatim in the training text. */
  verbatim: boolean;
}

/**
 * How much of a case the model has already seen.
 *
 * A score on contaminated data is not a measurement. It is the single most
 * common way a published result turns out to mean nothing, and it is usually
 * an accident rather than a fraud: a corpus gets deduplicated badly, or the
 * test set was scraped from the same place as the training set.
 */
export function contaminationOf(text: string, trainText: string, n = 10): Contamination {
  const t = text.trim();
  if (t.length === 0) return { overlap: 0, verbatim: false };
  const verbatim = t.length >= 8 && trainText.includes(t);
  if (t.length < n) return { overlap: verbatim ? 1 : 0, verbatim };

  let hits = 0;
  let total = 0;
  for (let i = 0; i + n <= t.length; i++) {
    total++;
    if (trainText.includes(t.slice(i, i + n))) hits++;
  }
  return { overlap: total > 0 ? hits / total : 0, verbatim };
}

/** Are two runs far enough apart to be worth believing? */
export function separated(a: EvalRun, b: EvalRun): boolean {
  return a.lo > b.hi || b.lo > a.hi;
}

/**
 * How many cases you would need for an interval of roughly this width.
 *
 * Answers the question everyone asks second: how big does the eval set have
 * to be? Inverted from the normal approximation at the observed rate, which
 * is close enough at the sizes that matter and honest about being a guide.
 */
export function casesNeeded(rate: number, halfWidth: number, z = 1.96): number {
  const p = Math.min(0.99, Math.max(0.01, rate));
  return Math.ceil((z * z * p * (1 - p)) / (halfWidth * halfWidth));
}

/** One run's score broken down by where its cases came from. */
export interface OriginSlice {
  origin: CaseOrigin;
  n: number;
  mean: number;
  lo: number;
  hi: number;
}

/**
 * Split a finished run by case origin.
 *
 * This is how the contamination question gets answered rather than asserted.
 * Whether training data inflates a score depends entirely on whether there
 * was anything specific to memorise: on a corpus generated from a handful of
 * templates the held-out text is statistically identical to the training
 * text, and the gap is zero. On real scraped data it is often enormous. The
 * only way to know which case you are in is to measure it.
 */
export function byOrigin(run: EvalRun): OriginSlice[] {
  const groups = new Map<CaseOrigin, CaseResult[]>();
  for (const r of run.results) {
    const list = groups.get(r.case.origin) ?? [];
    list.push(r);
    groups.set(r.case.origin, list);
  }
  const binary = METRIC_INFO[run.metric].binary;
  const out: OriginSlice[] = [];
  for (const [origin, rs] of groups) {
    const n = rs.length;
    const mean = rs.reduce((a, r) => a + r.score, 0) / n;
    if (binary) {
      const { lo, hi } = wilson(rs.filter((r) => r.pass).length, n);
      out.push({ origin, n, mean, lo, hi });
    } else {
      let v = 0;
      for (const r of rs) v += (r.score - mean) ** 2;
      const se = Math.sqrt(v / n) / Math.sqrt(n);
      out.push({ origin, n, mean, lo: Math.max(0, mean - 1.96 * se), hi: Math.min(1, mean + 1.96 * se) });
    }
  }
  return out.sort((a, b) => b.mean - a.mean);
}
