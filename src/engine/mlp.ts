import { ACTIVATIONS, type ActName } from './activations';
import {
  type Matrix,
  addRowVec,
  colSums,
  hadamard,
  mapMat,
  mat,
  matmul,
  matmulTA,
  matmulTB,
  mulberry32,
  randMat,
  softmaxRows,
  sub,
} from './tensor';

export type TaskType = 'binary' | 'regression' | 'multiclass';

export interface ForwardPass {
  /** A[0] is the input itself; A[l] is the output of layer l. */
  A: Matrix[];
  /** Z[l] is the pre-activation (weighted sum + bias) of layer l. */
  Z: Matrix[];
  output: Matrix;
}

export interface Grads {
  dW: Matrix[];
  db: Float64Array[];
  dZ: Matrix[];
  dA: Matrix[];
}

export class MLP {
  sizes: number[];
  hiddenActs: ActName[];
  task: TaskType;
  W: Matrix[] = [];
  b: Float64Array[] = [];
  l2 = 0;

  constructor(sizes: number[], hiddenActs: ActName[], task: TaskType, seed = 1337) {
    this.sizes = sizes;
    this.hiddenActs = hiddenActs;
    this.task = task;
    this.reset(seed);
  }

  get layerCount() {
    return this.sizes.length - 1;
  }

  get paramCount() {
    let n = 0;
    for (let l = 0; l < this.layerCount; l++) {
      n += this.sizes[l] * this.sizes[l + 1] + this.sizes[l + 1];
    }
    return n;
  }

  /** Multiply-accumulates for one forward pass on a single example. */
  get flopsPerExample() {
    let n = 0;
    for (let l = 0; l < this.layerCount; l++) n += 2 * this.sizes[l] * this.sizes[l + 1];
    return n;
  }

  reset(seed = Date.now() & 0xffff) {
    const rand = mulberry32(seed);
    this.W = [];
    this.b = [];
    for (let l = 0; l < this.layerCount; l++) {
      // ReLU-family layers want a larger gain to keep activation variance up.
      const act = l < this.layerCount - 1 ? this.hiddenActs[l] : 'linear';
      const gain = act === 'relu' || act === 'leaky_relu' ? Math.SQRT2 : 1;
      this.W.push(randMat(this.sizes[l], this.sizes[l + 1], rand, gain));
      this.b.push(new Float64Array(this.sizes[l + 1]));
    }
  }

  actAt(l: number): ActName {
    if (l < this.layerCount - 1) return this.hiddenActs[Math.min(l, this.hiddenActs.length - 1)];
    return this.task === 'binary' ? 'sigmoid' : 'linear';
  }

  forward(X: Matrix): ForwardPass {
    const A: Matrix[] = [X];
    const Z: Matrix[] = [];
    let cur = X;
    for (let l = 0; l < this.layerCount; l++) {
      // Z = A @ W + b -- the weighted sum every neuron computes.
      const z = addRowVec(matmul(cur, this.W[l]), this.b[l]);
      Z.push(z);
      const isOutput = l === this.layerCount - 1;
      if (isOutput && this.task === 'multiclass') {
        cur = softmaxRows(z);
      } else {
        cur = mapMat(z, ACTIVATIONS[this.actAt(l)].f);
      }
      A.push(cur);
    }
    return { A, Z, output: cur };
  }

  loss(pass: ForwardPass, Y: Matrix): number {
    const P = pass.output;
    const N = P.rows;
    const eps = 1e-12;
    let total = 0;
    if (this.task === 'regression') {
      for (let i = 0; i < P.data.length; i++) total += (P.data[i] - Y.data[i]) ** 2;
      total /= N;
    } else if (this.task === 'binary') {
      for (let i = 0; i < P.data.length; i++) {
        const p = Math.min(1 - eps, Math.max(eps, P.data[i]));
        total += -(Y.data[i] * Math.log(p) + (1 - Y.data[i]) * Math.log(1 - p));
      }
      total /= N;
    } else {
      for (let i = 0; i < P.data.length; i++) {
        if (Y.data[i] > 0) total += -Y.data[i] * Math.log(Math.max(eps, P.data[i]));
      }
      total /= N;
    }
    if (this.l2 > 0) {
      let reg = 0;
      for (const w of this.W) for (let i = 0; i < w.data.length; i++) reg += w.data[i] ** 2;
      total += this.l2 * reg;
    }
    return total;
  }

  /**
   * Backpropagation. Each layer answers one question: if my output had been a
   * little different, how much would the final loss change? Then it hands that
   * answer down to the layer beneath it.
   */
  backward(pass: ForwardPass, Y: Matrix): Grads {
    const L = this.layerCount;
    const N = pass.output.rows;
    const dW: Matrix[] = new Array(L);
    const db: Float64Array[] = new Array(L);
    const dZ: Matrix[] = new Array(L);
    const dA: Matrix[] = new Array(L + 1);

    // Output layer. For all three task setups the sigmoid/softmax/linear
    // derivative cancels against the matching loss derivative, leaving
    // simply (prediction - target).
    const diff = sub(pass.output, Y);
    dZ[L - 1] = mapMat(diff, (x) => (this.task === 'regression' ? (2 * x) / N : x / N));

    for (let l = L - 1; l >= 0; l--) {
      // Weight gradient for this layer: incoming activations x outgoing error.
      dW[l] = matmulTA(pass.A[l], dZ[l]);
      if (this.l2 > 0) {
        for (let i = 0; i < dW[l].data.length; i++) {
          dW[l].data[i] += 2 * this.l2 * this.W[l].data[i];
        }
      }
      db[l] = colSums(dZ[l]);
      // Push the error back onto the activations of the previous layer.
      dA[l] = matmulTB(dZ[l], this.W[l]);
      if (l > 0) {
        const dfn = ACTIVATIONS[this.actAt(l - 1)].df;
        dZ[l - 1] = hadamard(dA[l], mapMat(pass.Z[l - 1], dfn));
      }
    }
    dA[L] = diff;
    return { dW, db, dZ, dA };
  }

  predict(X: Matrix): Matrix {
    return this.forward(X).output;
  }

  accuracy(X: Matrix, Y: Matrix): number {
    const P = this.predict(X);
    let correct = 0;
    if (this.task === 'binary') {
      for (let i = 0; i < P.rows; i++) if ((P.data[i] > 0.5 ? 1 : 0) === Y.data[i]) correct++;
    } else if (this.task === 'multiclass') {
      for (let i = 0; i < P.rows; i++) {
        let best = 0;
        let bestV = -Infinity;
        let target = 0;
        for (let j = 0; j < P.cols; j++) {
          const p = P.data[i * P.cols + j];
          if (p > bestV) {
            bestV = p;
            best = j;
          }
          if (Y.data[i * P.cols + j] > 0) target = j;
        }
        if (best === target) correct++;
      }
    } else {
      return NaN;
    }
    return correct / P.rows;
  }

  clone(): MLP {
    const m = new MLP(this.sizes, this.hiddenActs, this.task);
    m.l2 = this.l2;
    m.W = this.W.map((w) => ({ rows: w.rows, cols: w.cols, data: new Float64Array(w.data) }));
    m.b = this.b.map((x) => new Float64Array(x));
    return m;
  }

  flatParams(): Float64Array {
    const out = new Float64Array(this.paramCount);
    let k = 0;
    for (let l = 0; l < this.layerCount; l++) {
      out.set(this.W[l].data, k);
      k += this.W[l].data.length;
      out.set(this.b[l], k);
      k += this.b[l].length;
    }
    return out;
  }

  loadFlat(p: Float64Array) {
    let k = 0;
    for (let l = 0; l < this.layerCount; l++) {
      this.W[l].data.set(p.subarray(k, k + this.W[l].data.length));
      k += this.W[l].data.length;
      this.b[l].set(p.subarray(k, k + this.b[l].length));
      k += this.b[l].length;
    }
  }

  /** Numerical gradient check: proof that the analytic backward pass is right. */
  gradCheck(X: Matrix, Y: Matrix, samples = 12, eps = 1e-5) {
    const pass = this.forward(X);
    const g = this.backward(pass, Y);
    const out: {
      layer: number;
      index: number;
      analytic: number;
      numeric: number;
      relErr: number;
    }[] = [];
    for (let s = 0; s < samples; s++) {
      const layer = s % this.layerCount;
      const w = this.W[layer];
      const index = Math.floor((s / samples) * w.data.length) % w.data.length;
      const orig = w.data[index];
      w.data[index] = orig + eps;
      const lp = this.loss(this.forward(X), Y);
      w.data[index] = orig - eps;
      const lm = this.loss(this.forward(X), Y);
      w.data[index] = orig;
      const numeric = (lp - lm) / (2 * eps);
      const analytic = g.dW[layer].data[index];
      const denom = Math.max(1e-8, Math.abs(analytic) + Math.abs(numeric));
      out.push({ layer, index, analytic, numeric, relErr: Math.abs(analytic - numeric) / denom });
    }
    return out;
  }
}

export function oneHot(labels: number[], classes: number): Matrix {
  const m = mat(labels.length, classes);
  labels.forEach((l, i) => {
    m.data[i * classes + l] = 1;
  });
  return m;
}
