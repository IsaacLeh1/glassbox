import { Optimizer, type OptimConfig, type ScheduleName, lrScale } from './optim';
import { BPETokenizer } from './tokenizer';
import { Transformer, type TransformerConfig } from './transformer';
import { mulberry32 } from './tensor';

export interface LMMetrics {
  step: number;
  loss: number;
  valLoss: number;
  perplexity: number;
  gradNorm: number;
  lr: number;
  tokensSeen: number;
  wallMs: number;
}

export interface LMTrainConfig {
  batchSeqs: number;
  steps: number;
  optim: OptimConfig;
  schedule: ScheduleName;
  seed: number;
  valFraction: number;
}

export const DEFAULT_LM_TRAIN: LMTrainConfig = {
  batchSeqs: 8,
  steps: 400,
  optim: {
    name: 'adam',
    lr: 0.01,
    momentum: 0.9,
    beta1: 0.9,
    beta2: 0.99,
    eps: 1e-8,
    clipNorm: 1,
  },
  schedule: 'warmup_cosine',
  seed: 7,
  valFraction: 0.1,
};

/**
 * Trains the in-browser transformer on a token stream. This is a genuine
 * language-model training loop: sample windows, predict the next token at every
 * position, backpropagate, step.
 */
export class LMTrainer {
  model: Transformer;
  tok: BPETokenizer;
  trainIds: number[] = [];
  valIds: number[] = [];
  cfg: LMTrainConfig;
  opt: Optimizer;
  history: LMMetrics[] = [];
  step = 0;
  tokensSeen = 0;
  status: 'idle' | 'running' | 'paused' | 'done' = 'idle';
  startedAt = 0;
  lastGradNorm = 0;
  private rand: () => number;

  constructor(
    text: string,
    tcfg: TransformerConfig,
    cfg: LMTrainConfig,
    merges = 120,
  ) {
    this.tok = BPETokenizer.train(text, merges);
    const ids = this.tok.encode(text);
    const cut = Math.floor(ids.length * (1 - cfg.valFraction));
    this.trainIds = ids.slice(0, cut);
    this.valIds = ids.slice(cut);
    this.model = new Transformer({ ...tcfg, vocab: this.tok.size }, cfg.seed);
    this.cfg = cfg;
    this.opt = new Optimizer(cfg.optim);
    this.rand = mulberry32(cfg.seed);
  }

  get vocabSize() {
    return this.tok.size;
  }

  get progress() {
    return this.cfg.steps > 0 ? Math.min(1, this.step / this.cfg.steps) : 0;
  }

  /** Total tokens the full run will process -- the headline number for LLM scale. */
  get totalTokens() {
    return this.cfg.steps * this.cfg.batchSeqs * this.model.cfg.blockSize;
  }

  private sampleWindow(src: number[]): { x: number[]; y: number[] } | null {
    const T = this.model.cfg.blockSize;
    if (src.length < T + 1) return null;
    const i = Math.floor(this.rand() * (src.length - T - 1));
    return { x: src.slice(i, i + T), y: src.slice(i + 1, i + T + 1) };
  }

  private evalLoss(src: number[], samples = 4): number {
    let total = 0;
    let n = 0;
    const saved = this.rand;
    const r = mulberry32(99);
    this.rand = r;
    for (let i = 0; i < samples; i++) {
      const w = this.sampleWindow(src);
      if (!w) break;
      total += this.model.loss(this.model.forward(w.x).trace, w.y);
      n++;
    }
    this.rand = saved;
    return n ? total / n : NaN;
  }

  /* --------------------------------------------------------------------
   * A batch is accumulated one sequence at a time.
   *
   * With a long context a single optimizer step can take seconds, so the loop
   * has to be interruptible at a finer grain than one step or the page locks
   * up. Gradients simply accumulate across sequences and are only applied once
   * the batch is complete, so stopping between sequences is always safe.
   * ------------------------------------------------------------------ */
  private batchLoss = 0;
  private batchCount = 0;
  private batchOpen = false;

  /** Sequences still needed before the next weight update. */
  get batchRemaining() {
    return Math.max(0, this.cfg.batchSeqs - this.batchCount);
  }

  /**
   * Process exactly one sequence. Returns 'step' when that sequence completed
   * a batch and the weights were updated, 'seq' when more sequences are still
   * needed, and 'done' when the run is over.
   */
  microStep(record = true): 'seq' | 'step' | 'done' {
    if (this.step >= this.cfg.steps) {
      this.status = 'done';
      return 'done';
    }
    if (this.startedAt === 0) this.startedAt = performance.now();

    if (!this.batchOpen) {
      this.model.zeroGrad();
      this.batchLoss = 0;
      this.batchCount = 0;
      this.batchOpen = true;
    }

    const w = this.sampleWindow(this.trainIds);
    if (!w) {
      this.status = 'done';
      return 'done';
    }
    const fw = this.model.forward(w.x);
    this.batchLoss += this.model.loss(fw.trace, w.y);
    this.model.backward(fw, w.y);
    this.batchCount++;
    this.tokensSeen += w.x.length;

    if (this.batchCount < this.cfg.batchSeqs) return 'seq';
    this.applyBatch(record);
    return 'step';
  }

  /** Average the accumulated gradients, clip, and take one optimizer step. */
  private applyBatch(record: boolean) {
    const counted = this.batchCount;
    this.batchOpen = false;
    if (counted === 0) return;
    const loss = this.batchLoss / counted;

    // Gradients were summed over the batch, so average them before stepping.
    for (const q of this.model.params) {
      for (let i = 0; i < q.g.data.length; i++) q.g.data[i] /= counted;
    }

    const groups = this.model.params.map((q) => q.g.data);
    const { norm } = this.opt.clip(groups);
    this.lastGradNorm = norm;

    const lr = this.cfg.optim.lr * lrScale(this.cfg.schedule, this.step, this.cfg.steps);
    this.opt.cfg.lr = lr;
    this.opt.tick();
    for (const q of this.model.params) this.opt.step(q.name, q.M.data, q.g.data);

    this.step++;

    if (record && (this.step % 5 === 0 || this.step === 1)) {
      const valLoss = this.evalLoss(this.valIds);
      this.history.push({
        step: this.step,
        loss,
        valLoss,
        // Perplexity is just exp(loss): roughly how many tokens the model is
        // effectively choosing between at each position.
        perplexity: Math.exp(loss),
        gradNorm: norm,
        lr,
        tokensSeen: this.tokensSeen,
        wallMs: performance.now() - this.startedAt,
      });
    }
  }

  /** One complete optimizer step, however many sequences that takes. */
  singleStep(record = true): boolean {
    let r = this.microStep(record);
    while (r === 'seq') r = this.microStep(record);
    return r === 'step';
  }

  /**
   * Run for a time budget, stopping between sequences rather than between
   * steps so a long context cannot block the page.
   *
   * Returns the number of SEQUENCES processed, not steps: with a large model a
   * whole slice may not finish even one step, and callers use a zero return to
   * mean "nothing happened, stop trying".
   */
  runSlice(budgetMs: number, maxSteps = 200): number {
    const t0 = performance.now();
    let seqs = 0;
    let steps = 0;
    for (;;) {
      const r = this.microStep();
      if (r === 'done') break;
      seqs++;
      if (r === 'step') steps++;
      if (steps >= maxSteps) break;
      if (performance.now() - t0 >= budgetMs) break;
    }
    return seqs;
  }

  latest(): LMMetrics | null {
    return this.history.length ? this.history[this.history.length - 1] : null;
  }

  reset() {
    this.model.build(this.cfg.seed);
    this.opt = new Optimizer(this.cfg.optim);
    this.rand = mulberry32(this.cfg.seed);
    this.history = [];
    this.step = 0;
    this.tokensSeen = 0;
    this.startedAt = 0;
    this.status = 'idle';
    this.batchOpen = false;
    this.batchCount = 0;
    this.batchLoss = 0;
  }
}
