import type { LMMetrics } from './lmTrainer';

/**
 * Analysis of a live training run.
 *
 * The purpose of this module is to answer the question a learner actually has
 * while watching samples appear: why does the output look like that, and what
 * would make it better. Every statement it produces is derived from measured
 * numbers in the run rather than from a fixed script.
 */

export type Stage = 'noise' | 'characters' | 'words' | 'grammar' | 'structure';

export const STAGES: Record<Stage, { label: string; what: string; next: string }> = {
  noise: {
    label: 'Random',
    what: 'The weights are still essentially their starting values. Output is a uniform scatter of tokens, because every token is about equally likely.',
    next: 'The first thing any language model learns is which characters and tokens are common at all. That usually takes only a few dozen steps.',
  },
  characters: {
    label: 'Character statistics',
    what: 'The model has learned which symbols are frequent and roughly how often spaces appear, so the text has the right texture but no real words. This is what people mean by gibberish that "looks like language".',
    next: 'Next it starts gluing frequent characters into the short common words of the corpus.',
  },
  words: {
    label: 'Words',
    what: 'Real words from the corpus are appearing, but the order between them is close to random. The model has learned the vocabulary without learning what follows what.',
    next: 'Next it starts using the previous token or two to choose the following one, which is where short phrases appear.',
  },
  grammar: {
    label: 'Local grammar',
    what: 'Short sequences hold together: articles land before nouns, punctuation lands where it should. Attention is now genuinely using earlier positions rather than predicting from frequency alone.',
    next: 'Beyond this the model needs a longer context window and more capacity to keep a whole sentence consistent.',
  },
  structure: {
    label: 'Sentence structure',
    what: 'Whole clauses are coherent and follow the shape of the training text. For a model this size this is close to the ceiling.',
    next: 'Further gains now come from more parameters, a longer context, and more data, not from more steps.',
  },
};

export interface SampleAnalysis {
  chars: number;
  words: number;
  /** Fraction of whitespace-separated chunks that exist in the training text. */
  realWordRatio: number;
  /** Fraction of distinct characters used, relative to the corpus alphabet. */
  alphabetCoverage: number;
  /** Longest run of a repeating substring, as a fraction of the sample. */
  repetitionRatio: number;
  looping: boolean;
  avgWordLength: number;
  spaceRatio: number;
  targetSpaceRatio: number;
}

/**
 * Length of the longest tandem repeat, e.g. "the cat the cat the cat".
 *
 * A stretch where s[i] === s[i + p] holds for `run` consecutive positions means
 * a block of period p is repeating. It only counts as a genuine loop once at
 * least two full copies are present, which is why `run` must reach `period`
 * before anything is recorded -- otherwise any coincidental alignment of a
 * common word scores as a loop.
 */
function longestRepeatRun(s: string): number {
  const n = s.length;
  if (n < 8) return 0;
  let best = 0;
  for (let period = 1; period <= Math.min(40, Math.floor(n / 2)); period++) {
    let run = 0;
    for (let i = 0; i + period < n; i++) {
      if (s[i] === s[i + period]) {
        run++;
        if (run >= period) best = Math.max(best, run + period);
      } else {
        run = 0;
      }
    }
  }
  return Math.min(best, n);
}

export function analyseSample(sample: string, corpus: string): SampleAnalysis {
  const text = sample ?? '';
  const words = text.split(/\s+/).filter(Boolean);
  const corpusWords = new Set(corpus.toLowerCase().split(/[^a-z0-9']+/i).filter(Boolean));
  const corpusAlphabet = new Set(corpus);
  const corpusSpaces = (corpus.match(/\s/g) ?? []).length / Math.max(1, corpus.length);

  const real = words.filter((w) => corpusWords.has(w.toLowerCase().replace(/[^a-z0-9']/gi, ''))).length;
  const used = new Set(text);
  let inAlphabet = 0;
  for (const c of used) if (corpusAlphabet.has(c)) inAlphabet++;

  const repeat = longestRepeatRun(text) / Math.max(1, text.length);

  return {
    chars: text.length,
    words: words.length,
    realWordRatio: words.length ? real / words.length : 0,
    alphabetCoverage: used.size ? inAlphabet / used.size : 0,
    repetitionRatio: repeat,
    looping: repeat > 0.5,
    avgWordLength: words.length ? words.reduce((a, w) => a + w.length, 0) / words.length : 0,
    spaceRatio: text.length ? (text.match(/\s/g) ?? []).length / text.length : 0,
    targetSpaceRatio: corpusSpaces,
  };
}

/**
 * Where the run has got to. Loss relative to uniform guessing is the primary
 * signal; the sample statistics break the ties.
 */
export function stageFor(loss: number, vocab: number, a: SampleAnalysis): Stage {
  const uniform = Math.log(Math.max(2, vocab));
  const progress = Number.isFinite(loss) ? 1 - loss / uniform : 0;
  if (progress < 0.08) return 'noise';
  if (a.realWordRatio > 0.75 && progress > 0.55) return 'structure';
  if (a.realWordRatio > 0.55) return 'grammar';
  if (a.realWordRatio > 0.2) return 'words';
  return 'characters';
}

export type Severity = 'good' | 'info' | 'warn' | 'bad';

export interface Finding {
  id: string;
  severity: Severity;
  title: string;
  detail: string;
  fix?: string;
}

export interface RunFacts {
  history: LMMetrics[];
  vocab: number;
  paramCount: number;
  trainTokens: number;
  blockSize: number;
  totalSteps: number;
  corpusChars: number;
}

/** Slope of the loss over the last portion of the run, per step. */
export function recentSlope(history: LMMetrics[], window = 10): number {
  const h = history.slice(-window);
  if (h.length < 3) return 0;
  const n = h.length;
  const mx = h.reduce((a, p) => a + p.step, 0) / n;
  const my = h.reduce((a, p) => a + p.loss, 0) / n;
  let num = 0;
  let den = 0;
  for (const p of h) {
    num += (p.step - mx) * (p.loss - my);
    den += (p.step - mx) ** 2;
  }
  return den === 0 ? 0 : num / den;
}

export function diagnose(facts: RunFacts, sample: string | null, corpus: string): Finding[] {
  const out: Finding[] = [];
  const h = facts.history;
  if (h.length === 0) return out;

  const last = h[h.length - 1];
  const first = h[0];
  const uniform = Math.log(Math.max(2, facts.vocab));
  const progress = 1 - last.loss / uniform;
  const slope = recentSlope(h);
  const a = sample ? analyseSample(sample, corpus) : null;

  /* ---------------------------------------------------- learning at all */

  if (!Number.isFinite(last.loss)) {
    out.push({
      id: 'diverged',
      severity: 'bad',
      title: 'The run has diverged',
      detail:
        'The loss is no longer a finite number. Weights have grown until the arithmetic overflowed, and nothing recoverable remains in this model.',
      fix: 'Reset, lower the learning rate by a factor of ten, and make sure gradient clipping is on.',
    });
    return out;
  }

  out.push({
    id: 'progress',
    severity: progress > 0.4 ? 'good' : progress > 0.15 ? 'info' : 'warn',
    title: `Loss is ${(progress * 100).toFixed(0)}% of the way down from random guessing`,
    detail:
      `Guessing uniformly among ${facts.vocab} tokens would give a loss of ${uniform.toFixed(2)}. ` +
      `The run started at ${first.loss.toFixed(2)} and is now at ${last.loss.toFixed(2)}, which is a perplexity of ` +
      `${Math.exp(last.loss).toFixed(1)} — the model is as uncertain as if it were choosing evenly between that many tokens.`,
  });

  /* ------------------------------------------------------- still moving */

  if (h.length > 6) {
    const stalled = Math.abs(slope) < 1e-5;
    if (stalled && progress < 0.5) {
      out.push({
        id: 'stalled',
        severity: 'warn',
        title: 'Loss has stopped falling',
        detail:
          'Over the last several measurements the loss is flat, but it is still a long way above what this corpus should allow. The model has run out of either capacity or learning signal.',
        fix: 'Try a wider model or another layer first. If that does not move it, raise the learning rate; if the loss then becomes unstable, you were already at the limit.',
      });
    } else if (stalled) {
      out.push({
        id: 'converged',
        severity: 'good',
        title: 'The run has converged',
        detail: 'The loss has flattened at a low value. Additional steps at this size will not buy much more.',
        fix: 'To go further, increase the model width or the number of layers rather than the step count.',
      });
    } else {
      out.push({
        id: 'improving',
        severity: 'good',
        title: 'Still improving',
        detail: `Loss is falling at roughly ${Math.abs(slope).toExponential(1)} per step. Stopping now would leave real progress unused.`,
      });
    }
  }

  /* -------------------------------------------------------- overfitting */

  const gap = last.valLoss - last.loss;
  if (Number.isFinite(gap) && h.length > 8) {
    if (gap > 0.35) {
      out.push({
        id: 'overfit',
        severity: 'warn',
        title: 'Memorising rather than learning',
        detail:
          `Training loss is ${last.loss.toFixed(2)} but held-out loss is ${last.valLoss.toFixed(2)}. ` +
          'The model is fitting specific passages instead of the pattern behind them, which is what happens when there is more capacity than data.',
        fix: `Load more text (currently ${facts.corpusChars.toLocaleString()} characters), or make the model smaller.`,
      });
    } else if (gap < 0.12) {
      out.push({
        id: 'generalising',
        severity: 'good',
        title: 'Generalising, not memorising',
        detail: `Held-out loss tracks training loss within ${gap.toFixed(3)}. Whatever the model has learned applies to text it has never seen.`,
      });
    }
  }

  /* ------------------------------------------------------ data adequacy */

  const seen = last.tokensSeen;
  const recommended = facts.paramCount * 20;
  if (seen < recommended * 0.05) {
    out.push({
      id: 'undertrained',
      severity: 'info',
      title: 'Undertrained for its size',
      detail:
        `This model has ${facts.paramCount.toLocaleString()} parameters and has seen ${seen.toLocaleString()} tokens. ` +
        `The usual guidance is about twenty tokens per parameter, which would be ${recommended.toLocaleString()}. ` +
        'It has the capacity to be considerably better than it currently is.',
      fix: 'Raise the step count, or shrink the model so its capacity matches the data you have.',
    });
  }

  if (facts.trainTokens < facts.blockSize * 200) {
    out.push({
      id: 'tinycorpus',
      severity: 'warn',
      title: 'Very little text to learn from',
      detail: `The training split is only ${facts.trainTokens.toLocaleString()} tokens, so windows of ${facts.blockSize} overlap heavily and the model sees the same passages repeatedly.`,
      fix: 'Load more characters on the data tab.',
    });
  }

  /* --------------------------------------------------- the sample text */

  if (a && a.words > 2) {
    if (a.looping) {
      out.push({
        id: 'looping',
        severity: 'warn',
        title: 'The sample is stuck in a loop',
        detail:
          `About ${(a.repetitionRatio * 100).toFixed(0)}% of the sample is one phrase repeating. This is a sampling symptom as much as a model one: ` +
          'once a token becomes overwhelmingly likely, picking the top choice every time cycles forever.',
        fix: 'Raise the sampling temperature, or train longer so the distribution is less peaked.',
      });
    }
    if (a.realWordRatio < 0.2) {
      out.push({
        id: 'nonwords',
        severity: 'info',
        title: 'The output is not made of real words yet',
        detail:
          `Only ${(a.realWordRatio * 100).toFixed(0)}% of the chunks in the sample appear in the training text. ` +
          'The model has learned which characters are common and roughly where spaces go, which is why it looks like language from a distance, but it has not yet learned which letter sequences are actual words.',
        fix: 'This is a normal early stage. Keep training; word shapes usually appear within a few hundred steps.',
      });
    } else if (a.realWordRatio > 0.7) {
      out.push({
        id: 'words-ok',
        severity: 'good',
        title: 'Producing real words',
        detail: `${(a.realWordRatio * 100).toFixed(0)}% of the sample consists of words that appear in the training text. The remaining errors are mostly in word order rather than spelling.`,
      });
    }
    const spaceOff = Math.abs(a.spaceRatio - a.targetSpaceRatio);
    if (a.targetSpaceRatio > 0 && spaceOff > 0.06) {
      out.push({
        id: 'spacing',
        severity: 'info',
        title: a.spaceRatio > a.targetSpaceRatio ? 'Too many spaces' : 'Words are running together',
        detail:
          `The sample is ${(a.spaceRatio * 100).toFixed(0)}% whitespace against ${(a.targetSpaceRatio * 100).toFixed(0)}% in the training text. ` +
          'Word length is one of the very first things a language model gets right, so a large gap here means it is still early.',
      });
    }
  }

  /* ----------------------------------------------------- stability */

  if (last.gradNorm > 0 && h.length > 4) {
    const norms = h.slice(-8).map((m) => m.gradNorm);
    const maxN = Math.max(...norms);
    const minN = Math.min(...norms);
    if (maxN / Math.max(1e-9, minN) > 25) {
      out.push({
        id: 'spiky',
        severity: 'warn',
        title: 'Gradients are spiking',
        detail: `Gradient norm has swung between ${minN.toExponential(1)} and ${maxN.toExponential(1)} recently. Individual batches are pulling the weights very differently, which makes progress erratic.`,
        fix: 'Turn on gradient clipping, increase the batch size, or lower the learning rate.',
      });
    }
  }

  return out;
}

/** Short, plain summary of the run, for the top of the diagnostics panel. */
export function summariseRun(facts: RunFacts, sample: string | null, corpus: string): {
  stage: Stage;
  progress: number;
  headline: string;
} {
  const last = facts.history[facts.history.length - 1];
  const uniform = Math.log(Math.max(2, facts.vocab));
  const progress = last ? Math.max(0, Math.min(1, 1 - last.loss / uniform)) : 0;
  const a = analyseSample(sample ?? '', corpus);
  const stage = last ? stageFor(last.loss, facts.vocab, a) : 'noise';
  const pct = (progress * 100).toFixed(0);
  return {
    stage,
    progress,
    headline: `${STAGES[stage].label} — ${pct}% of the way from random guessing toward perfect prediction on this corpus.`,
  };
}
