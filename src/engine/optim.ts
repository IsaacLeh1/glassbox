import type { Matrix } from './tensor';

export type OptimName = 'sgd' | 'momentum' | 'rmsprop' | 'adam';

export interface OptimConfig {
  name: OptimName;
  lr: number;
  momentum: number;
  beta1: number;
  beta2: number;
  eps: number;
  clipNorm: number;
}

export const DEFAULT_OPTIM: OptimConfig = {
  name: 'adam',
  lr: 0.05,
  momentum: 0.9,
  beta1: 0.9,
  beta2: 0.999,
  eps: 1e-8,
  clipNorm: 0,
};

export const OPTIM_INFO: Record<OptimName, { label: string; blurb: string }> = {
  sgd: {
    label: 'SGD',
    blurb:
      'The plain rule: step downhill by learning-rate times gradient. Nothing is remembered between steps, so a narrow valley makes it zig-zag.',
  },
  momentum: {
    label: 'SGD + Momentum',
    blurb:
      'Keeps a running velocity, like a ball rolling downhill. It powers through small bumps and damps the zig-zag that plain SGD suffers.',
  },
  rmsprop: {
    label: 'RMSProp',
    blurb:
      'Tracks how large recent gradients have been per parameter and divides by that. Parameters with wild gradients get smaller steps automatically.',
  },
  adam: {
    label: 'Adam',
    blurb:
      'Momentum and RMSProp combined, plus a bias correction for the first few steps. The default choice for training almost every large model today.',
  },
};

interface Slot {
  m: Float64Array;
  v: Float64Array;
}

/**
 * Optimizer operating on flat parameter buffers. Each update here is the exact
 * arithmetic a GPU kernel performs, just one parameter at a time.
 */
export class Optimizer {
  cfg: OptimConfig;
  private slots = new Map<string, Slot>();
  t = 0;

  constructor(cfg: OptimConfig) {
    this.cfg = { ...cfg };
  }

  reset() {
    this.slots.clear();
    this.t = 0;
  }

  private slot(key: string, n: number): Slot {
    let s = this.slots.get(key);
    if (!s || s.m.length !== n) {
      s = { m: new Float64Array(n), v: new Float64Array(n) };
      this.slots.set(key, s);
    }
    return s;
  }

  /** Call once per optimizer step, before touching any parameter group. */
  tick() {
    this.t++;
  }

  /**
   * Global gradient-norm clipping. Large models rely on this to survive a bad
   * batch: if the whole gradient vector is longer than clipNorm, scale it down.
   */
  clip(groups: Float64Array[]): { norm: number; scaled: boolean } {
    if (this.cfg.clipNorm <= 0) return { norm: this.globalNorm(groups), scaled: false };
    const norm = this.globalNorm(groups);
    if (norm <= this.cfg.clipNorm) return { norm, scaled: false };
    const k = this.cfg.clipNorm / (norm + 1e-12);
    for (const g of groups) for (let i = 0; i < g.length; i++) g[i] *= k;
    return { norm, scaled: true };
  }

  globalNorm(groups: Float64Array[]): number {
    let s = 0;
    for (const g of groups) for (let i = 0; i < g.length; i++) s += g[i] * g[i];
    return Math.sqrt(s);
  }

  /** Applies one update in place. `p` is modified; `g` is read only. */
  step(key: string, p: Float64Array, g: Float64Array) {
    const { name, lr, momentum, beta1, beta2, eps } = this.cfg;
    const n = p.length;
    if (name === 'sgd') {
      for (let i = 0; i < n; i++) p[i] -= lr * g[i];
      return;
    }
    const s = this.slot(key, n);
    if (name === 'momentum') {
      for (let i = 0; i < n; i++) {
        s.m[i] = momentum * s.m[i] + g[i];
        p[i] -= lr * s.m[i];
      }
      return;
    }
    if (name === 'rmsprop') {
      for (let i = 0; i < n; i++) {
        s.v[i] = beta2 * s.v[i] + (1 - beta2) * g[i] * g[i];
        p[i] -= (lr * g[i]) / (Math.sqrt(s.v[i]) + eps);
      }
      return;
    }
    // Adam: first moment is the mean gradient, second is the mean squared
    // gradient. Both start at zero, so early steps are bias corrected.
    const bc1 = 1 - Math.pow(beta1, Math.max(1, this.t));
    const bc2 = 1 - Math.pow(beta2, Math.max(1, this.t));
    for (let i = 0; i < n; i++) {
      s.m[i] = beta1 * s.m[i] + (1 - beta1) * g[i];
      s.v[i] = beta2 * s.v[i] + (1 - beta2) * g[i] * g[i];
      const mh = s.m[i] / bc1;
      const vh = s.v[i] / bc2;
      p[i] -= (lr * mh) / (Math.sqrt(vh) + eps);
    }
  }

  stepMatrix(key: string, W: Matrix, dW: Matrix) {
    this.step(key, W.data, dW.data);
  }
}

export type ScheduleName = 'constant' | 'step' | 'cosine' | 'warmup_cosine';

export const SCHEDULE_INFO: Record<ScheduleName, { label: string; blurb: string }> = {
  constant: { label: 'Constant', blurb: 'Same learning rate from first step to last.' },
  step: { label: 'Step decay', blurb: 'Cut the learning rate by half at fixed intervals.' },
  cosine: {
    label: 'Cosine decay',
    blurb: 'Glide smoothly from the full rate down to nearly zero by the final step.',
  },
  warmup_cosine: {
    label: 'Warmup + cosine',
    blurb:
      'Ramp up over the first few percent of training, then cosine down. Standard for large transformers, because a cold model takes badly to full-size steps.',
  },
};

/** Learning rate at a given step, as a fraction of the base rate. */
export function lrScale(schedule: ScheduleName, step: number, total: number): number {
  const p = total > 0 ? Math.min(1, step / total) : 0;
  switch (schedule) {
    case 'step':
      return Math.pow(0.5, Math.floor(p * 4));
    case 'cosine':
      return 0.5 * (1 + Math.cos(Math.PI * p));
    case 'warmup_cosine': {
      // Worked in step space rather than as a fraction. Doing it by fraction
      // gives the very first step a learning rate of exactly zero, which costs
      // a full batch of forward and backward work and changes nothing.
      const warmSteps = Math.max(1, Math.floor(total * 0.06));
      if (step < warmSteps) return (step + 1) / warmSteps;
      const q = (step - warmSteps) / Math.max(1, total - warmSteps);
      return 0.5 * (1 + Math.cos(Math.PI * Math.min(1, q)));
    }
    default:
      return 1;
  }
}
