import { type Dataset, batches, gatherRows } from './datasets';
import { MLP } from './mlp';
import { Optimizer, type OptimConfig, type ScheduleName, lrScale } from './optim';
import { mulberry32 } from './tensor';

export interface TrainMetrics {
  step: number;
  epoch: number;
  loss: number;
  valLoss: number;
  acc: number;
  valAcc: number;
  gradNorm: number;
  lr: number;
  examplesSeen: number;
  wallMs: number;
}

export interface TrainerConfig {
  batchSize: number;
  epochs: number;
  optim: OptimConfig;
  schedule: ScheduleName;
  l2: number;
  seed: number;
}

export const DEFAULT_TRAINER: TrainerConfig = {
  batchSize: 24,
  epochs: 220,
  optim: {
    name: 'adam',
    lr: 0.05,
    momentum: 0.9,
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-8,
    clipNorm: 0,
  },
  schedule: 'constant',
  l2: 0,
  seed: 1,
};

export type TrainerStatus = 'idle' | 'running' | 'paused' | 'done';

/**
 * Drives a real training run in slices small enough to keep the UI at 60fps.
 * Every number it reports comes out of an actual forward and backward pass.
 */
export class Trainer {
  net: MLP;
  data: { train: Dataset; test: Dataset };
  cfg: TrainerConfig;
  opt: Optimizer;
  history: TrainMetrics[] = [];
  step = 0;
  epoch = 0;
  status: TrainerStatus = 'idle';
  examplesSeen = 0;
  startedAt = 0;
  private rand: () => number;
  private queue: number[][] = [];
  /** Snapshot of the previous weights, so the UI can show what moved. */
  lastDelta: Float64Array | null = null;
  lastGradNorm = 0;

  constructor(net: MLP, data: { train: Dataset; test: Dataset }, cfg: TrainerConfig) {
    this.net = net;
    this.data = data;
    this.cfg = cfg;
    this.opt = new Optimizer(cfg.optim);
    this.rand = mulberry32(cfg.seed);
    this.net.l2 = cfg.l2;
  }

  get totalSteps() {
    const perEpoch = Math.ceil(this.data.train.points.length / this.cfg.batchSize);
    return perEpoch * this.cfg.epochs;
  }

  get progress() {
    return this.totalSteps > 0 ? Math.min(1, this.step / this.totalSteps) : 0;
  }

  reset(keepWeights = false) {
    if (!keepWeights) this.net.reset(this.cfg.seed);
    this.opt = new Optimizer(this.cfg.optim);
    this.rand = mulberry32(this.cfg.seed);
    this.history = [];
    this.queue = [];
    this.step = 0;
    this.epoch = 0;
    this.examplesSeen = 0;
    this.status = 'idle';
    this.net.l2 = this.cfg.l2;
    this.lastDelta = null;
  }

  private refillQueue() {
    this.queue = [...batches(this.data.train.points.length, this.cfg.batchSize, this.rand)];
    this.epoch++;
  }

  /** One optimizer step on one mini-batch. Returns false when training is over. */
  singleStep(record = true): boolean {
    if (this.epoch >= this.cfg.epochs && this.queue.length === 0) {
      this.status = 'done';
      return false;
    }
    if (this.queue.length === 0) {
      if (this.epoch >= this.cfg.epochs) {
        this.status = 'done';
        return false;
      }
      this.refillQueue();
    }
    if (this.startedAt === 0) this.startedAt = performance.now();

    const idx = this.queue.shift()!;
    const xb = gatherRows(this.data.train.X, idx);
    const yb = gatherRows(this.data.train.Y, idx);

    const before = this.net.flatParams();
    const pass = this.net.forward(xb);
    const g = this.net.backward(pass, yb);

    const groups = [...g.dW.map((m) => m.data), ...g.db];
    const { norm } = this.opt.clip(groups);
    this.lastGradNorm = norm;

    // Learning-rate schedule is applied per step, not per epoch.
    const baseLr = this.cfg.optim.lr;
    const scaled = baseLr * lrScale(this.cfg.schedule, this.step, this.totalSteps);
    this.opt.cfg.lr = scaled;
    this.opt.tick();
    for (let l = 0; l < this.net.layerCount; l++) {
      this.opt.step(`W${l}`, this.net.W[l].data, g.dW[l].data);
      this.opt.step(`b${l}`, this.net.b[l], g.db[l]);
    }

    const after = this.net.flatParams();
    const delta = new Float64Array(after.length);
    for (let i = 0; i < after.length; i++) delta[i] = after[i] - before[i];
    this.lastDelta = delta;

    this.step++;
    this.examplesSeen += idx.length;

    if (record) {
      // Full-dataset metrics are expensive, so sample them at a sane cadence.
      const measure = this.step % 4 === 0 || this.step === 1 || this.queue.length === 0;
      if (measure) {
        const trainPass = this.net.forward(this.data.train.X);
        const testPass = this.net.forward(this.data.test.X);
        this.history.push({
          step: this.step,
          epoch: this.epoch,
          loss: this.net.loss(trainPass, this.data.train.Y),
          valLoss: this.net.loss(testPass, this.data.test.Y),
          acc: this.net.accuracy(this.data.train.X, this.data.train.Y),
          valAcc: this.net.accuracy(this.data.test.X, this.data.test.Y),
          gradNorm: norm,
          lr: scaled,
          examplesSeen: this.examplesSeen,
          wallMs: performance.now() - this.startedAt,
        });
      }
    }
    return true;
  }

  /** Run as many steps as fit inside a time budget, in milliseconds. */
  runSlice(budgetMs: number, maxSteps = 500): number {
    const t0 = performance.now();
    let n = 0;
    while (n < maxSteps && performance.now() - t0 < budgetMs) {
      if (!this.singleStep()) break;
      n++;
    }
    return n;
  }

  latest(): TrainMetrics | null {
    return this.history.length ? this.history[this.history.length - 1] : null;
  }
}

/**
 * Decision-boundary field: run the network over a grid so the UI can paint
 * what it currently believes about every point in the plane.
 */
export function decisionField(net: MLP, res: number, domain: [number, number]): Float32Array {
  const out = new Float32Array(res * res);
  const [lo, hi] = domain;
  const grid = { rows: res * res, cols: 2, data: new Float64Array(res * res * 2) };
  let k = 0;
  for (let i = 0; i < res; i++) {
    for (let j = 0; j < res; j++) {
      grid.data[k++] = lo + ((hi - lo) * j) / (res - 1);
      grid.data[k++] = hi - ((hi - lo) * i) / (res - 1);
    }
  }
  const p = net.predict(grid);
  for (let i = 0; i < out.length; i++) out[i] = p.data[i * p.cols];
  return out;
}

/** Same idea, but for the activation of one specific hidden neuron. */
export function neuronField(
  net: MLP,
  layer: number,
  unit: number,
  res: number,
  domain: [number, number],
): Float32Array {
  const out = new Float32Array(res * res);
  const [lo, hi] = domain;
  const grid = { rows: res * res, cols: 2, data: new Float64Array(res * res * 2) };
  let k = 0;
  for (let i = 0; i < res; i++) {
    for (let j = 0; j < res; j++) {
      grid.data[k++] = lo + ((hi - lo) * j) / (res - 1);
      grid.data[k++] = hi - ((hi - lo) * i) / (res - 1);
    }
  }
  const pass = net.forward(grid);
  const A = pass.A[layer + 1];
  for (let i = 0; i < out.length; i++) out[i] = A.data[i * A.cols + unit];
  return out;
}
