export type ActName = 'tanh' | 'relu' | 'sigmoid' | 'linear' | 'gelu' | 'leaky_relu';

export interface Activation {
  name: ActName;
  label: string;
  f: (x: number) => number;
  /** Derivative expressed in terms of the PRE-activation input x. */
  df: (x: number) => number;
  blurb: string;
  range: string;
}

const K = Math.sqrt(2 / Math.PI);

export const ACTIVATIONS: Record<ActName, Activation> = {
  tanh: {
    name: 'tanh',
    label: 'tanh',
    f: Math.tanh,
    df: (x) => 1 - Math.tanh(x) ** 2,
    range: '(-1, 1)',
    blurb: 'Squashes any number into -1..1. Smooth and centred on zero, which keeps gradients well behaved in small networks.',
  },
  relu: {
    name: 'relu',
    label: 'ReLU',
    f: (x) => (x > 0 ? x : 0),
    df: (x) => (x > 0 ? 1 : 0),
    range: '[0, inf)',
    blurb: 'Passes positives through untouched and flattens negatives to zero. Cheap, and the reason very deep networks became trainable.',
  },
  leaky_relu: {
    name: 'leaky_relu',
    label: 'Leaky ReLU',
    f: (x) => (x > 0 ? x : 0.01 * x),
    df: (x) => (x > 0 ? 1 : 0.01),
    range: '(-inf, inf)',
    blurb: 'ReLU with a small slope on the negative side, so a neuron that goes negative can still recover instead of dying permanently.',
  },
  sigmoid: {
    name: 'sigmoid',
    label: 'sigmoid',
    f: (x) => 1 / (1 + Math.exp(-x)),
    df: (x) => {
      const s = 1 / (1 + Math.exp(-x));
      return s * (1 - s);
    },
    range: '(0, 1)',
    blurb: 'Squashes into 0..1, so the output reads as a probability. Saturates at the extremes, where gradients nearly vanish.',
  },
  linear: {
    name: 'linear',
    label: 'linear',
    f: (x) => x,
    df: () => 1,
    range: '(-inf, inf)',
    blurb: 'No squashing at all. Stacking linear layers gains you nothing, which is precisely why nonlinearity is required.',
  },
  gelu: {
    name: 'gelu',
    label: 'GELU',
    f: (x) => 0.5 * x * (1 + Math.tanh(K * (x + 0.044715 * x ** 3))),
    df: (x) => {
      const inner = K * (x + 0.044715 * x ** 3);
      const t = Math.tanh(inner);
      const dInner = K * (1 + 3 * 0.044715 * x * x);
      return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * dInner;
    },
    range: '(-0.17, inf)',
    blurb: 'A smooth ReLU that lets slightly negative values leak through. The default inside modern transformers.',
  },
};

export const ACT_LIST = Object.values(ACTIVATIONS);
