import { Optimizer } from './optim';
import type { BPETokenizer } from './tokenizer';
import type { Transformer } from './transformer';

/**
 * Teaching an already-trained model something new.
 *
 * This is the same arithmetic as pretraining -- predict the next token, score
 * it, backpropagate, step -- pointed at a handful of examples the user wrote
 * instead of at a corpus. What makes it worth its own file is everything
 * around the loop, because that is where the lessons are:
 *
 *  - The vocabulary is frozen. It was decided before pretraining started and
 *    cannot grow now, so any character the model has never seen is dropped on
 *    the way in. The user should be told which, not left guessing.
 *  - Teaching one thing costs something elsewhere. Loss on the original corpus
 *    is measured before and after so catastrophic forgetting is visible rather
 *    than theoretical.
 *  - It is reversible. The weights are snapshotted first, so an experiment
 *    that ruins the model can be undone.
 */

export interface TeachConfig {
  lr: number;
  /** How many times to go over the examples. */
  epochs: number;
  clipNorm: number;
}

export const DEFAULT_TEACH: TeachConfig = { lr: 0.003, epochs: 12, clipNorm: 1 };

export interface VocabCheck {
  /** Distinct characters in the examples that the vocabulary has no entry for. */
  dropped: string[];
  /**
   * The same characters written so they can be seen. A dropped space or
   * newline is invisible on screen, which makes the warning look broken.
   */
  droppedLabels: string[];
  /** Characters that survived encoding. */
  kept: number;
  total: number;
}

/** Make a character visible, the way tokenizer interfaces conventionally do. */
export function showChar(ch: string): string {
  if (ch === ' ') return '· (space)';
  if (ch === '\n') return '↵ (newline)';
  if (ch === '\t') return '→ (tab)';
  if (ch === '\r') return '(carriage return)';
  const code = ch.codePointAt(0) ?? 0;
  if (code < 32 || code === 127) return `(control ${code})`;
  return ch;
}

/**
 * Which characters of some text this tokenizer cannot represent.
 *
 * `encode` silently discards them, which is the right behaviour for a training
 * loop and the wrong behaviour for a user who typed them.
 */
export function checkVocab(tok: BPETokenizer, text: string): VocabCheck {
  const missing = new Set<string>();
  let kept = 0;
  let total = 0;
  for (const ch of Array.from(text)) {
    total++;
    // A character survives if it is in the vocabulary in its own right, since
    // that is the fallback encode() uses when a longer piece is unknown.
    if (tok.index.has(ch)) kept++;
    else missing.add(ch);
  }
  const dropped = [...missing].sort();
  return { dropped, droppedLabels: dropped.map(showChar), kept, total };
}

export interface TeachWindow {
  x: number[];
  y: number[];
  /** Which example this came from, for attributing loss back to a line. */
  from: number;
}

/** Turn examples into next-token training windows the model can actually take. */
export function buildWindows(tok: BPETokenizer, examples: string[], blockSize: number): TeachWindow[] {
  const out: TeachWindow[] = [];
  examples.forEach((text, from) => {
    const ids = tok.encode(text);
    // Two tokens is the minimum that says anything: one to read, one to predict.
    if (ids.length < 2) return;
    // A long example is cut into overlapping windows, stepping half a block so
    // every token gets a turn at being predicted with a full context.
    const stride = Math.max(1, Math.floor(blockSize / 2));
    for (let i = 0; i + 1 < ids.length; i += stride) {
      const chunk = ids.slice(i, i + blockSize + 1);
      if (chunk.length < 2) break;
      out.push({ x: chunk.slice(0, -1), y: chunk.slice(1), from });
      if (i + blockSize + 1 >= ids.length) break;
    }
  });
  return out;
}

export interface TeachPoint {
  pass: number;
  loss: number;
  /** Loss on a sample of the original training text, at the same moment. */
  oldLoss: number;
  gradNorm: number;
}

export class Finetuner {
  readonly windows: TeachWindow[];
  readonly vocab: VocabCheck;
  readonly before: Float64Array;
  history: TeachPoint[] = [];

  /** Loss on the new examples before a single step was taken. */
  baselineNew = NaN;
  /** Loss on the original corpus before a single step was taken. */
  baselineOld = NaN;

  pass = 0;
  private idx = 0;
  private opt: Optimizer;
  private oldWindows: TeachWindow[] = [];

  readonly model: Transformer;
  readonly tok: BPETokenizer;
  readonly cfg: TeachConfig;

  constructor(
    model: Transformer,
    tok: BPETokenizer,
    examples: string[],
    cfg: TeachConfig,
    /** A slice of what the model was originally trained on, for the forgetting check. */
    originalIds: number[] = [],
  ) {
    this.model = model;
    this.tok = tok;
    this.cfg = cfg;
    this.windows = buildWindows(tok, examples, model.cfg.blockSize);
    // Joining with a separator would flag a character the user never typed:
    // the examples are encoded one at a time, so that is how they are checked.
    this.vocab = checkVocab(tok, examples.join(''));
    this.before = model.flatParams().slice();
    this.opt = new Optimizer({
      name: 'adam',
      lr: cfg.lr,
      momentum: 0.9,
      beta1: 0.9,
      beta2: 0.99,
      eps: 1e-8,
      clipNorm: cfg.clipNorm,
    });

    // Fixed windows from the original corpus, evenly spaced, so the
    // forgetting measurement is comparable from one moment to the next.
    const T = model.cfg.blockSize;
    const n = 6;
    if (originalIds.length > T + 1) {
      for (let k = 0; k < n; k++) {
        const i = Math.floor((k / n) * (originalIds.length - T - 1));
        this.oldWindows.push({ x: originalIds.slice(i, i + T), y: originalIds.slice(i + 1, i + T + 1), from: -1 });
      }
    }

    this.baselineNew = this.evalNew();
    this.baselineOld = this.evalOld();
  }

  get total() {
    return this.windows.length * this.cfg.epochs;
  }

  get done() {
    return this.pass >= this.cfg.epochs || this.windows.length === 0;
  }

  get progress() {
    return this.total > 0 ? Math.min(1, (this.pass * this.windows.length + this.idx) / this.total) : 1;
  }

  private meanLoss(ws: TeachWindow[]): number {
    if (ws.length === 0) return NaN;
    let s = 0;
    for (const w of ws) s += this.model.loss(this.model.forward(w.x).trace, w.y);
    return s / ws.length;
  }

  /** Loss on what the user is teaching. Should fall. */
  evalNew(): number {
    return this.meanLoss(this.windows);
  }

  /** Loss on what the model already knew. Watch this rise. */
  evalOld(): number {
    return this.meanLoss(this.oldWindows);
  }

  /**
   * One window: forward, backward, step. Returns false when the run is over.
   *
   * Every window is its own update rather than being batched, because with a
   * handful of examples there is nothing to average over and the user wants to
   * watch the loss move.
   */
  microStep(): boolean {
    if (this.done) return false;
    const w = this.windows[this.idx];

    this.model.zeroGrad();
    const fw = this.model.forward(w.x);
    const loss = this.model.loss(fw.trace, w.y);
    this.model.backward(fw, w.y);

    const groups = this.model.params.map((q) => q.g.data);
    const { norm } = this.opt.clip(groups);
    this.opt.tick();
    for (const q of this.model.params) this.opt.step(q.name, q.M.data, q.g.data);

    this.idx++;
    if (this.idx >= this.windows.length) {
      this.idx = 0;
      this.pass++;
      this.history.push({ pass: this.pass, loss, oldLoss: this.evalOld(), gradNorm: norm });
    }
    return !this.done;
  }

  /** Run for a time budget, so the page stays responsive. */
  runSlice(budgetMs: number): number {
    const t0 = performance.now();
    let n = 0;
    while (!this.done && performance.now() - t0 < budgetMs) {
      this.microStep();
      n++;
    }
    return n;
  }

  /** Put every weight back exactly as it was before this object existed. */
  revert() {
    this.model.loadFlat(this.before);
  }

  /** How far the weights have moved from where they started, in relative terms. */
  drift(): number {
    const now = this.model.flatParams();
    let d = 0;
    let b = 0;
    for (let i = 0; i < now.length; i++) {
      const x = now[i] - this.before[i];
      d += x * x;
      b += this.before[i] * this.before[i];
    }
    return b > 1e-12 ? Math.sqrt(d / b) : 0;
  }
}
