import { contaminationOf } from './evals';
import { suggestedTokens } from './resources';

/**
 * The programme a frontier lab actually runs, as a sequence of gates.
 *
 * The thing that most distinguishes a professional model programme from
 * someone training a network is not the hardware and not the cleverness. It
 * is that the work is gated. Nobody starts a pretraining run because the data
 * looks fine; somebody signs off that it passed a defined set of checks.
 * Nobody ships because the samples read well; the model has to clear a bar
 * written down before anyone saw a number.
 *
 * Gates are unglamorous and they are the difference between a model programme
 * and a hobby. So this file is mostly checks, each returning a pass or a fail
 * and a reason, and the reasons are the teaching.
 *
 * Every check here computes for real against the user's own text and model.
 */

export type StageId =
  | 'charter'
  | 'data'
  | 'scaling'
  | 'architecture'
  | 'pretrain'
  | 'evaluate'
  | 'safety'
  | 'launch';

export interface StageDef {
  id: StageId;
  n: number;
  title: string;
  /** Who owns this at a real lab, which is a surprisingly clarifying thing to know. */
  owner: string;
  blurb: string;
}

export const STAGES: StageDef[] = [
  {
    id: 'charter',
    n: 1,
    title: 'Write the charter',
    owner: 'Research lead, with product',
    blurb:
      'What the model is for, who it is for, and what would count as it working. Written before any data is collected, because every decision after this one is judged against it.',
  },
  {
    id: 'data',
    n: 2,
    title: 'Source and clean the data',
    owner: 'Data team',
    blurb:
      'Acquire the text, check you are allowed to use it, remove the duplicates, and prove it does not contain your own test set. This is where most of the real effort of a model programme goes, and it is the least glamorous part by a wide margin.',
  },
  {
    id: 'scaling',
    n: 3,
    title: 'Set the compute budget',
    owner: 'Research lead',
    blurb:
      'Decide how much compute you are willing to spend, then let the scaling laws tell you how to split it between a bigger model and more tokens. Getting this wrong wastes the entire run.',
  },
  {
    id: 'architecture',
    n: 4,
    title: 'Freeze the architecture',
    owner: 'Research lead',
    blurb:
      'Fix the shape. After this point it does not change, because a change invalidates every measurement taken before it.',
  },
  {
    id: 'pretrain',
    n: 5,
    title: 'Run pretraining',
    owner: 'Training infrastructure',
    blurb:
      'The expensive part, and mostly a matter of watching. A loss curve, a checkpoint schedule, and somebody on call for when a node dies at three in the morning.',
  },
  {
    id: 'evaluate',
    n: 6,
    title: 'Evaluation gate',
    owner: 'Evals team',
    blurb:
      'Run the suite that was designed before training started. Beat the trivial baselines, clear the bar you committed to, or do not proceed. This gate is the one people are most tempted to negotiate with.',
  },
  {
    id: 'safety',
    n: 7,
    title: 'Safety review',
    owner: 'Safety team, independent of research',
    blurb:
      'Red-team it, decide what it must refuse, and implement those constraints. Independent of the team that built it, because nobody is a fair judge of their own work.',
  },
  {
    id: 'launch',
    n: 8,
    title: 'Launch review',
    owner: 'Everyone, in one room',
    blurb:
      'Cost per request, latency, capacity, rollback plan, and a decision. Shipping is a choice somebody makes and signs, not something that happens when the training finishes.',
  },
];

/* ------------------------------------------------------------- gates -- */

export interface Check {
  label: string;
  pass: boolean;
  /** What the check found, in the language a reviewer would use. */
  detail: string;
  /** True when this check can be waived; a hard check cannot. */
  soft?: boolean;
}

export interface Gate {
  stage: StageId;
  checks: Check[];
  passed: boolean;
  /** Checks that failed and cannot be waived. */
  blocking: number;
}

export function gate(stage: StageId, checks: Check[]): Gate {
  const blocking = checks.filter((c) => !c.pass && !c.soft).length;
  return { stage, checks, passed: blocking === 0, blocking };
}

/* -------------------------------------------------------- data hygiene */

export interface Duplicate {
  text: string;
  count: number;
}

export interface DataReport {
  /** Documents after splitting on sentence boundaries. */
  docs: number;
  chars: number;
  /** Documents that appeared more than once, exactly. */
  exactDupes: Duplicate[];
  /** Share of documents removed by exact deduplication. */
  dupeShare: number;
  /** Documents sharing most of their ten-character runs with another document. */
  nearDupes: number;
  /** Documents that overlap the held-out evaluation text. */
  contaminated: number;
  /** Distinct characters, which is what the tokenizer will have to cover. */
  alphabet: number;
  /** The cleaned corpus, which is what actually gets trained on. */
  cleaned: string;
  cleanedChars: number;
}

/**
 * Inspect and clean a corpus, the way a data team would.
 *
 * Deduplication is not housekeeping. A document that appears a hundred times
 * is trained on a hundred times, so the model memorises it instead of
 * learning from it, and published work has repeatedly found that removing
 * duplicates improves a model at no other cost. Contamination is worse: if
 * the evaluation text is inside the training text then the benchmark score is
 * measuring memory and the whole programme is flying blind.
 */
export function inspectData(raw: string, evalText: string, nearThreshold = 0.8): DataReport {
  const docs = raw
    .split(/(?<=\.)\s+/)
    .map((d) => d.trim())
    .filter((d) => d.length > 0);

  const seen = new Map<string, number>();
  for (const d of docs) seen.set(d, (seen.get(d) ?? 0) + 1);

  const exactDupes = [...seen.entries()]
    .filter(([, n]) => n > 1)
    .map(([text, count]) => ({ text, count }))
    .sort((a, b) => b.count - a.count);

  const unique = [...seen.keys()];
  const removedExact = docs.length - unique.length;

  // Near-duplicates: documents whose character n-grams are mostly shared with
  // a document already kept. Quadratic, which is why real pipelines use
  // hashing instead, but exact and legible at this size.
  const kept: string[] = [];
  const keptGrams: Set<string>[] = [];
  let nearDupes = 0;
  const grams = (s: string, n = 10) => {
    const g = new Set<string>();
    for (let i = 0; i + n <= s.length; i++) g.add(s.slice(i, i + n));
    return g;
  };
  for (const d of unique) {
    const g = grams(d);
    let dupe = false;
    for (const k of keptGrams) {
      let hit = 0;
      for (const x of g) if (k.has(x)) hit++;
      if (g.size > 0 && hit / g.size >= nearThreshold) {
        dupe = true;
        break;
      }
    }
    if (dupe) nearDupes++;
    else {
      kept.push(d);
      keptGrams.push(g);
    }
  }

  // Decontamination: anything that overlaps the held-out text must go, or the
  // evaluation afterwards means nothing.
  const clean: string[] = [];
  let contaminated = 0;
  for (const d of kept) {
    if (evalText && contaminationOf(d, evalText).overlap > 0.9) contaminated++;
    else clean.push(d);
  }

  const cleaned = clean.join(' ');
  return {
    docs: docs.length,
    chars: raw.length,
    exactDupes,
    dupeShare: docs.length > 0 ? removedExact / docs.length : 0,
    nearDupes,
    contaminated,
    alphabet: new Set(Array.from(raw)).size,
    cleaned,
    cleanedChars: cleaned.length,
  };
}

export type Licence = 'unknown' | 'permissive' | 'noncommercial' | 'proprietary';

export const LICENCE_INFO: Record<Licence, { label: string; ok: boolean; blurb: string }> = {
  unknown: {
    label: 'Not checked',
    ok: false,
    blurb:
      'Nobody has established what this text is or who owns it. No lawyer signs this off, and at a real company it stops the programme here.',
  },
  permissive: {
    label: 'Permissive, commercial use allowed',
    ok: true,
    blurb: 'Public domain or an explicitly permissive licence. The straightforward case, and rarer than you would hope.',
  },
  noncommercial: {
    label: 'Research only',
    ok: true,
    blurb:
      'Fine for a model that never ships. Becomes a serious problem the moment somebody wants to sell access to what you trained on it.',
  },
  proprietary: {
    label: 'Someone else owns it',
    ok: false,
    blurb: 'Training on it is a decision for lawyers rather than engineers, and the honest default is not to.',
  },
};

const plural = (n: number, one: string, many = one + 's') => `${n} ${n === 1 ? one : many}`;

export function dataGate(report: DataReport, licence: Licence, minChars = 2000): Gate {
  return gate('data', [
    {
      label: 'Licence established',
      pass: LICENCE_INFO[licence].ok,
      detail: LICENCE_INFO[licence].blurb,
    },
    {
      label: 'Duplicates removed',
      pass: true,
      detail:
        report.exactDupes.length > 0
          ? `${plural(report.exactDupes.length, 'repeated document')} found and collapsed, ${(report.dupeShare * 100).toFixed(1)}% of the corpus, plus ${plural(report.nearDupes, 'near-duplicate')}.`
          : `No exact duplicates, and ${plural(report.nearDupes, 'near-duplicate')} removed.`,
    },
    {
      // Passes because the overlapping documents have already been taken out.
      // Failing on a condition the cleaning step itself resolves would leave
      // the programme with a gate nobody can ever clear, which is worse than
      // no gate: the reader concludes the check is broken rather than that
      // the data was.
      label: 'No evaluation data left in the training set',
      pass: true,
      detail:
        report.contaminated === 0
          ? 'Nothing in the corpus overlapped the held-out text.'
          : `${plural(report.contaminated, 'document')} overlapped the held-out text and ${report.contaminated === 1 ? 'was' : 'were'} removed. Had ${report.contaminated === 1 ? 'it' : 'they'} stayed, every number in the evaluation afterwards would have been measuring memory.`,
    },
    {
      label: 'Enough text to train on',
      pass: report.cleanedChars >= minChars,
      detail: `${report.cleanedChars.toLocaleString()} characters after cleaning, against a floor of ${minChars.toLocaleString()}.`,
    },
  ]);
}

/* ------------------------------------------------------------ scaling */

export interface ScalingPlan {
  /** Total floating-point operations the budget buys. */
  flops: number;
  /** Parameters the budget supports, at the conventional twenty tokens each. */
  params: number;
  tokens: number;
  /** Tokens actually available in the cleaned corpus. */
  haveTokens: number;
  /** True when there is not enough text for the model this budget would buy. */
  dataLimited: boolean;
  /** The size the available data can actually support. */
  paramsFromData: number;
}

/**
 * Split a compute budget between model size and training tokens.
 *
 * The finding that reorganised the field: for a fixed amount of compute there
 * is an optimal trade, and for years everyone was on the wrong side of it,
 * training models far larger than the data they were given could support.
 * Roughly twenty tokens per parameter is the rule of thumb that came out of
 * it. Training cost is about six operations per parameter per token, so the
 * budget determines both numbers at once.
 */
export function planScaling(flops: number, availableTokens: number): ScalingPlan {
  // C = 6 * N * D and D = 20 * N, so C = 120 * N^2.
  const params = Math.max(1, Math.sqrt(flops / 120));
  const tokens = suggestedTokens(params);
  return {
    flops,
    params,
    tokens,
    haveTokens: availableTokens,
    dataLimited: availableTokens < tokens,
    paramsFromData: Math.max(1, availableTokens / 20),
  };
}

export function scalingGate(plan: ScalingPlan, chosenParams: number): Gate {
  const ratio = plan.haveTokens / Math.max(1, chosenParams);
  return gate('scaling', [
    {
      label: 'Budget and model size agree',
      pass: chosenParams <= plan.params * 4,
      detail:
        chosenParams <= plan.params * 4
          ? `The budget supports about ${Math.round(plan.params).toLocaleString()} parameters and the design is ${chosenParams.toLocaleString()}.`
          : `The design is ${chosenParams.toLocaleString()} parameters against a budget that supports about ${Math.round(plan.params).toLocaleString()}. The run will not finish.`,
    },
    {
      label: 'Enough tokens for the model size',
      pass: ratio >= 5,
      soft: true,
      detail:
        ratio >= 20
          ? `${ratio.toFixed(0)} tokens per parameter, at or above the conventional twenty.`
          : `${ratio.toFixed(1)} tokens per parameter, below the conventional twenty. The model is larger than the data can fill, so it will memorise rather than generalise. This is the mistake the whole field made for several years.`,
    },
  ]);
}

/* ------------------------------------------------- the programme state */

export interface Signoff {
  stage: StageId;
  at: number;
  /** Set when a reviewer waived a soft failure, which a real process records. */
  waived: string[];
}

export function nextStage(done: StageId[]): StageDef | null {
  return STAGES.find((s) => !done.includes(s.id)) ?? null;
}

export function canEnter(stage: StageId, done: StageId[]): boolean {
  const i = STAGES.findIndex((s) => s.id === stage);
  if (i <= 0) return true;
  return done.includes(STAGES[i - 1].id);
}
