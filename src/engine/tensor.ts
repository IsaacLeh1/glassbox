/**
 * Minimal row-major matrix library backed by Float64Array.
 * Everything the app displays as a number comes out of these functions --
 * there are no pre-baked values anywhere in Glassbox.
 */

export interface Matrix {
  rows: number;
  cols: number;
  data: Float64Array;
}

export function mat(rows: number, cols: number, fill = 0): Matrix {
  const data = new Float64Array(rows * cols);
  if (fill !== 0) data.fill(fill);
  return { rows, cols, data };
}

export function fromRows(rows: number[][]): Matrix {
  const r = rows.length;
  const c = rows[0]?.length ?? 0;
  const m = mat(r, c);
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) m.data[i * c + j] = rows[i][j];
  return m;
}

export function toRows(m: Matrix): number[][] {
  const out: number[][] = [];
  for (let i = 0; i < m.rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < m.cols; j++) row.push(m.data[i * m.cols + j]);
    out.push(row);
  }
  return out;
}

export const at = (m: Matrix, i: number, j: number) => m.data[i * m.cols + j];
export const set = (m: Matrix, i: number, j: number, x: number) => {
  m.data[i * m.cols + j] = x;
};

export function copy(m: Matrix): Matrix {
  return { rows: m.rows, cols: m.cols, data: new Float64Array(m.data) };
}

/** C = A @ B  -- the single most expensive thing a neural network does. */
export function matmul(a: Matrix, b: Matrix): Matrix {
  if (a.cols !== b.rows) throw new Error(`shape mismatch ${a.rows}x${a.cols} @ ${b.rows}x${b.cols}`);
  const out = mat(a.rows, b.cols);
  const { rows: n, cols: k } = a;
  const p = b.cols;
  for (let i = 0; i < n; i++) {
    for (let t = 0; t < k; t++) {
      const av = a.data[i * k + t];
      if (av === 0) continue;
      const boff = t * p;
      const ooff = i * p;
      for (let j = 0; j < p; j++) out.data[ooff + j] += av * b.data[boff + j];
    }
  }
  return out;
}

/** C = A^T @ B, without materialising the transpose. */
export function matmulTA(a: Matrix, b: Matrix): Matrix {
  if (a.rows !== b.rows) throw new Error('shape mismatch in matmulTA');
  const out = mat(a.cols, b.cols);
  for (let t = 0; t < a.rows; t++) {
    for (let i = 0; i < a.cols; i++) {
      const av = a.data[t * a.cols + i];
      if (av === 0) continue;
      for (let j = 0; j < b.cols; j++) out.data[i * b.cols + j] += av * b.data[t * b.cols + j];
    }
  }
  return out;
}

/** C = A @ B^T */
export function matmulTB(a: Matrix, b: Matrix): Matrix {
  if (a.cols !== b.cols) throw new Error('shape mismatch in matmulTB');
  const out = mat(a.rows, b.rows);
  for (let i = 0; i < a.rows; i++) {
    for (let j = 0; j < b.rows; j++) {
      let s = 0;
      for (let t = 0; t < a.cols; t++) s += a.data[i * a.cols + t] * b.data[j * b.cols + t];
      out.data[i * b.rows + j] = s;
    }
  }
  return out;
}

export function transpose(m: Matrix): Matrix {
  const out = mat(m.cols, m.rows);
  for (let i = 0; i < m.rows; i++)
    for (let j = 0; j < m.cols; j++) out.data[j * m.rows + i] = m.data[i * m.cols + j];
  return out;
}

/** Broadcast-add a length-`cols` bias vector to every row. */
export function addRowVec(m: Matrix, vec: Float64Array): Matrix {
  const out = copy(m);
  for (let i = 0; i < m.rows; i++)
    for (let j = 0; j < m.cols; j++) out.data[i * m.cols + j] += vec[j];
  return out;
}

/** Sum down the rows -- this is how a bias gradient is formed. */
export function colSums(m: Matrix): Float64Array {
  const out = new Float64Array(m.cols);
  for (let i = 0; i < m.rows; i++)
    for (let j = 0; j < m.cols; j++) out[j] += m.data[i * m.cols + j];
  return out;
}

export function mapMat(m: Matrix, fn: (x: number) => number): Matrix {
  const out = mat(m.rows, m.cols);
  for (let i = 0; i < m.data.length; i++) out.data[i] = fn(m.data[i]);
  return out;
}

export function zipMat(a: Matrix, b: Matrix, fn: (x: number, y: number) => number): Matrix {
  const out = mat(a.rows, a.cols);
  for (let i = 0; i < a.data.length; i++) out.data[i] = fn(a.data[i], b.data[i]);
  return out;
}

export const hadamard = (a: Matrix, b: Matrix) => zipMat(a, b, (x, y) => x * y);
export const sub = (a: Matrix, b: Matrix) => zipMat(a, b, (x, y) => x - y);
export const addMat = (a: Matrix, b: Matrix) => zipMat(a, b, (x, y) => x + y);
export const scale = (m: Matrix, k: number) => mapMat(m, (x) => x * k);

/** Row-wise softmax, shifted by the row max for numerical stability. */
export function softmaxRows(m: Matrix): Matrix {
  const out = mat(m.rows, m.cols);
  for (let i = 0; i < m.rows; i++) {
    let max = -Infinity;
    for (let j = 0; j < m.cols; j++) max = Math.max(max, m.data[i * m.cols + j]);
    let sum = 0;
    for (let j = 0; j < m.cols; j++) {
      const e = Math.exp(m.data[i * m.cols + j] - max);
      out.data[i * m.cols + j] = e;
      sum += e;
    }
    for (let j = 0; j < m.cols; j++) out.data[i * m.cols + j] /= sum;
  }
  return out;
}

export function softmax1d(xs: number[], temperature = 1): number[] {
  const t = Math.max(temperature, 1e-6);
  const scaled = xs.map((x) => x / t);
  const max = Math.max(...scaled);
  const exps = scaled.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/* ---------------------------------------------------------------- RNG ---- */

/** Deterministic PRNG. A fixed seed means a lesson replays identically. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller: uniform noise in, normally distributed noise out. */
export function gaussian(rand: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * Xavier/Glorot initialisation. Scale is chosen so signal variance survives a
 * trip through the layer instead of exploding or vanishing.
 */
export function randMat(rows: number, cols: number, rand: () => number, gain = 1): Matrix {
  const m = mat(rows, cols);
  const limit = gain * Math.sqrt(6 / (rows + cols));
  for (let i = 0; i < m.data.length; i++) m.data[i] = (rand() * 2 - 1) * limit;
  return m;
}

/** Summary statistics of a flat buffer, for describing a weight matrix. */
export interface BufStats {
  n: number;
  mean: number;
  std: number;
  min: number;
  max: number;
  /** Share of entries within 0.01 of zero, which is how sparse a matrix looks. */
  nearZero: number;
}

export function statsOf(d: Float64Array): BufStats {
  let mean = 0;
  let min = Infinity;
  let max = -Infinity;
  let nearZero = 0;
  for (let i = 0; i < d.length; i++) {
    mean += d[i];
    if (d[i] < min) min = d[i];
    if (d[i] > max) max = d[i];
    if (Math.abs(d[i]) < 0.01) nearZero++;
  }
  mean /= Math.max(1, d.length);
  let v = 0;
  for (let i = 0; i < d.length; i++) v += (d[i] - mean) ** 2;
  return {
    n: d.length,
    mean,
    std: Math.sqrt(v / Math.max(1, d.length)),
    min,
    max,
    nearZero: d.length > 0 ? nearZero / d.length : 0,
  };
}
