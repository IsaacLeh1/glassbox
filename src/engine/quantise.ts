import type { Transformer } from './transformer';

/**
 * Making a model smaller by storing its weights less precisely.
 *
 * This is the first thing anyone does to a model before serving it, and the
 * reason is memory bandwidth rather than disk. Generating one token requires
 * reading every weight in the model, so how fast a model can produce text is
 * governed almost entirely by how many bytes have to move from memory to the
 * arithmetic units. Halving the bytes roughly halves the time, which is a
 * larger effect than almost anything else available.
 *
 * What happens here is genuine: each weight is snapped onto a coarse grid and
 * then read back, so the model really is restricted to the values a quantised
 * model could hold, and any loss of quality is real and measurable.
 *
 * What does not happen here is the speed-up. Doing it properly needs integer
 * kernels, and this engine computes in Float64 throughout. So the accuracy
 * cost is measured and the time saving is calculated from the byte counts
 * rather than observed. The panel says so.
 */

export type QuantBits = 8 | 6 | 4 | 3 | 2;

export type QuantScheme = 'tensor' | 'row';

export const SCHEME_INFO: Record<QuantScheme, { label: string; blurb: string }> = {
  tensor: {
    label: 'One scale for the whole table',
    blurb:
      'Find the largest weight anywhere in the tensor and size the grid to it. Cheap, and it goes badly the moment a single outlier weight is much bigger than the rest, because the grid has to stretch to reach it and every other weight gets a coarser step.',
  },
  row: {
    label: 'A scale for every row',
    blurb:
      'Size the grid separately for each row, so one row with a large weight cannot coarsen the others. This is what production quantisation does, and it costs one extra number per row.',
  },
};

export interface TensorError {
  name: string;
  /** Mean absolute error, as a fraction of the tensor's own typical size. */
  relError: number;
  /** Largest absolute weight, which is what sets the grid. */
  peak: number;
  /** Peak divided by the typical size: how much one outlier dominates. */
  outlierRatio: number;
  count: number;
}

export interface QuantResult {
  bits: number;
  scheme: QuantScheme;
  /** Total bytes for the weights plus the scales. */
  bytes: number;
  /** What the same weights cost at 32 bits each, the usual starting point. */
  baseBytes: number;
  /** Mean relative error across the whole model. */
  relError: number;
  perTensor: TensorError[];
  /** Weights that landed exactly on zero and could be skipped entirely. */
  zeroed: number;
  total: number;
}

/** The grid: symmetric, centred on zero, with 2^(bits-1)-1 steps each way. */
function levels(bits: number): number {
  return Math.pow(2, bits - 1) - 1;
}

function quantiseSpan(data: Float64Array, from: number, to: number, q: number): { err: number; zeros: number; peak: number } {
  let peak = 0;
  for (let i = from; i < to; i++) {
    const a = Math.abs(data[i]);
    if (a > peak) peak = a;
  }
  // A span of all zeros has nothing to scale; leave it alone.
  if (peak === 0) return { err: 0, zeros: to - from, peak: 0 };

  const scale = peak / q;
  let err = 0;
  let zeros = 0;
  for (let i = from; i < to; i++) {
    const step = Math.max(-q, Math.min(q, Math.round(data[i] / scale)));
    const back = step * scale;
    err += Math.abs(back - data[i]);
    if (step === 0) zeros++;
    data[i] = back;
  }
  return { err, zeros, peak };
}

/**
 * Quantise every weight in the model, in place.
 *
 * The caller is expected to have snapshotted `flatParams()` first; this is
 * destructive precisely because that is what deployment does to a model.
 */
export function quantiseModel(model: Transformer, bits: QuantBits, scheme: QuantScheme): QuantResult {
  const q = levels(bits);
  const perTensor: TensorError[] = [];
  let totalErr = 0;
  let totalAbs = 0;
  let zeroed = 0;
  let total = 0;
  let scaleCount = 0;

  for (const p of model.params) {
    const { rows, cols, data } = p.M;
    let absSum = 0;
    for (let i = 0; i < data.length; i++) absSum += Math.abs(data[i]);

    let err = 0;
    let zeros = 0;
    let peak = 0;

    if (scheme === 'row') {
      for (let r = 0; r < rows; r++) {
        const res = quantiseSpan(data, r * cols, (r + 1) * cols, q);
        err += res.err;
        zeros += res.zeros;
        if (res.peak > peak) peak = res.peak;
      }
      scaleCount += rows;
    } else {
      const res = quantiseSpan(data, 0, data.length, q);
      err = res.err;
      zeros = res.zeros;
      peak = res.peak;
      scaleCount += 1;
    }

    const mean = absSum / Math.max(1, data.length);
    perTensor.push({
      name: p.name,
      relError: mean > 0 ? err / data.length / mean : 0,
      peak,
      outlierRatio: mean > 0 ? peak / mean : 0,
      count: data.length,
    });

    totalErr += err;
    totalAbs += absSum;
    zeroed += zeros;
    total += data.length;
  }

  return {
    bits,
    scheme,
    // Scales are kept at full precision, which is what real formats do.
    bytes: Math.ceil((total * bits) / 8) + scaleCount * 4,
    baseBytes: total * 4,
    relError: totalAbs > 0 ? totalErr / totalAbs : 0,
    perTensor: perTensor.sort((a, b) => b.relError - a.relError),
    zeroed,
    total,
  };
}

/** Byte cost without touching anything, for the size projection before you commit. */
export function projectSize(paramCount: number, bits: number, scalesPerModel: number): number {
  return Math.ceil((paramCount * bits) / 8) + scalesPerModel * 4;
}

export function prettyBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  if (b < 1024 * 1024 * 1024) return `${(b / (1024 * 1024)).toFixed(1)} MB`;
  return `${(b / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

/**
 * What the same treatment would cost a real model.
 *
 * Quantisation is one of the few things that scales exactly: the byte count is
 * linear in the parameter count, so the arithmetic that applies to a 53
 * thousand parameter model in a browser tab applies unchanged to a 70 billion
 * parameter one in a datacentre.
 */
export interface ScaleUp {
  label: string;
  params: number;
  fp16: number;
  quantised: number;
  /** How many 80GB cards it takes to hold the weights alone. */
  cardsFp16: number;
  cardsQuantised: number;
}

const CARD_BYTES = 80 * 1024 * 1024 * 1024;

export function scaleUp(bits: number): ScaleUp[] {
  const models: { label: string; params: number }[] = [
    { label: 'GPT-2 small', params: 124e6 },
    { label: 'Llama 3 8B', params: 8.03e9 },
    { label: 'Llama 3 70B', params: 70.6e9 },
    { label: 'Llama 3 405B', params: 405e9 },
  ];
  return models.map((m) => {
    const fp16 = m.params * 2;
    const quantised = (m.params * bits) / 8;
    return {
      ...m,
      fp16,
      quantised,
      cardsFp16: Math.ceil(fp16 / CARD_BYTES),
      cardsQuantised: Math.ceil(quantised / CARD_BYTES),
    };
  });
}
