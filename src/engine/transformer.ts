import { ACTIVATIONS } from './activations';
import {
  type Matrix,
  addRowVec,
  colSums,
  mat,
  matmul,
  matmulTA,
  matmulTB,
  mulberry32,
  gaussian,
  softmaxRows,
} from './tensor';

/**
 * A complete decoder-only transformer, small enough to train in a browser tab
 * and instrumented so every intermediate value can be drawn on screen.
 *
 * Nothing in here is approximated for the sake of the demo: the attention
 * scores shown in the UI are the same numbers the loss is computed from, and
 * the backward pass is checked against finite differences in the test suite.
 */

export interface TransformerConfig {
  vocab: number;
  dModel: number;
  nHeads: number;
  nLayers: number;
  blockSize: number;
  dFF: number;
}

export const DEFAULT_TCONFIG: TransformerConfig = {
  vocab: 0,
  dModel: 48,
  nHeads: 4,
  nLayers: 2,
  blockSize: 24,
  dFF: 96,
};

/** A named tensor plus its gradient buffer, so the UI can enumerate weights. */
export interface Param {
  name: string;
  M: Matrix;
  g: Matrix;
}

const p = (name: string, rows: number, cols: number): Param => ({
  name,
  M: mat(rows, cols),
  g: mat(rows, cols),
});

interface LNCache {
  xhat: Matrix;
  inv: Float64Array;
}

export interface HeadTrace {
  /** Raw q.k / sqrt(dh) before masking. */
  scores: Matrix;
  /** Post-softmax attention weights: row i says where token i looked. */
  att: Matrix;
  q: Matrix;
  k: Matrix;
  v: Matrix;
  out: Matrix;
}

export interface LayerTrace {
  input: Matrix;
  ln1: Matrix;
  heads: HeadTrace[];
  attnOut: Matrix;
  afterAttn: Matrix;
  ln2: Matrix;
  ffHidden: Matrix;
  ffOut: Matrix;
  output: Matrix;
}

/**
 * An extra vector added into the residual stream during the forward pass.
 *
 * This is how activation steering works in practice: a direction is found in
 * the model's own internal representation and then pushed along, without
 * touching a single weight. `layer` of -1 applies it right after the
 * embedding; otherwise it is applied to the output of that layer.
 */
export interface Steer {
  layer: number;
  vec: Float64Array;
  scale: number;
}

export interface Trace {
  tokens: number[];
  tokenEmb: Matrix;
  posEmb: Matrix;
  embedded: Matrix;
  layers: LayerTrace[];
  finalLN: Matrix;
  logits: Matrix;
}

function layerNormFwd(x: Matrix, g: Matrix, b: Matrix): { y: Matrix; cache: LNCache } {
  const { rows, cols } = x;
  const y = mat(rows, cols);
  const xhat = mat(rows, cols);
  const inv = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    let mean = 0;
    for (let j = 0; j < cols; j++) mean += x.data[i * cols + j];
    mean /= cols;
    let varr = 0;
    for (let j = 0; j < cols; j++) {
      const d = x.data[i * cols + j] - mean;
      varr += d * d;
    }
    varr /= cols;
    const iv = 1 / Math.sqrt(varr + 1e-5);
    inv[i] = iv;
    for (let j = 0; j < cols; j++) {
      const xh = (x.data[i * cols + j] - mean) * iv;
      xhat.data[i * cols + j] = xh;
      y.data[i * cols + j] = xh * g.data[j] + b.data[j];
    }
  }
  return { y, cache: { xhat, inv } };
}

function layerNormBwd(dy: Matrix, cache: LNCache, g: Matrix, dg: Matrix, db: Matrix): Matrix {
  const { rows, cols } = dy;
  const dx = mat(rows, cols);
  for (let i = 0; i < rows; i++) {
    let sum1 = 0;
    let sum2 = 0;
    for (let j = 0; j < cols; j++) {
      const dyv = dy.data[i * cols + j];
      const xh = cache.xhat.data[i * cols + j];
      dg.data[j] += dyv * xh;
      db.data[j] += dyv;
      const dxhat = dyv * g.data[j];
      sum1 += dxhat;
      sum2 += dxhat * xh;
    }
    for (let j = 0; j < cols; j++) {
      const xh = cache.xhat.data[i * cols + j];
      const dxhat = dy.data[i * cols + j] * g.data[j];
      dx.data[i * cols + j] = (cache.inv[i] / cols) * (cols * dxhat - sum1 - xh * sum2);
    }
  }
  return dx;
}

/** Jacobian-vector product for a row-wise softmax. */
function softmaxBwd(probs: Matrix, dOut: Matrix): Matrix {
  const { rows, cols } = probs;
  const dx = mat(rows, cols);
  for (let i = 0; i < rows; i++) {
    let dot = 0;
    for (let j = 0; j < cols; j++) dot += dOut.data[i * cols + j] * probs.data[i * cols + j];
    for (let j = 0; j < cols; j++) {
      dx.data[i * cols + j] = probs.data[i * cols + j] * (dOut.data[i * cols + j] - dot);
    }
  }
  return dx;
}

function sliceCols(m: Matrix, start: number, width: number): Matrix {
  const out = mat(m.rows, width);
  for (let i = 0; i < m.rows; i++) {
    for (let j = 0; j < width; j++) out.data[i * width + j] = m.data[i * m.cols + start + j];
  }
  return out;
}

function addColsInto(dst: Matrix, src: Matrix, start: number) {
  for (let i = 0; i < src.rows; i++) {
    for (let j = 0; j < src.cols; j++) dst.data[i * dst.cols + start + j] += src.data[i * src.cols + j];
  }
}

interface BlockParams {
  ln1g: Param;
  ln1b: Param;
  wq: Param;
  wk: Param;
  wv: Param;
  wo: Param;
  bo: Param;
  ln2g: Param;
  ln2b: Param;
  w1: Param;
  b1: Param;
  w2: Param;
  b2: Param;
}

export class Transformer {
  cfg: TransformerConfig;
  tok!: Param;
  pos!: Param;
  blocks: BlockParams[] = [];
  lnfg!: Param;
  lnfb!: Param;
  head!: Param;
  headb!: Param;
  params: Param[] = [];

  constructor(cfg: TransformerConfig, seed = 1234) {
    this.cfg = cfg;
    this.build(seed);
  }

  get dHead() {
    return Math.floor(this.cfg.dModel / this.cfg.nHeads);
  }

  get paramCount() {
    return this.params.reduce((n, q) => n + q.M.data.length, 0);
  }

  build(seed = 1234) {
    const { vocab, dModel, nLayers, blockSize, dFF } = this.cfg;
    const rand = mulberry32(seed);
    // Small normal init, as used by GPT-style models.
    const init = (q: Param, std: number) => {
      for (let i = 0; i < q.M.data.length; i++) q.M.data[i] = gaussian(rand) * std;
    };
    const ones = (q: Param) => q.M.data.fill(1);

    this.tok = p('tok_emb', vocab, dModel);
    this.pos = p('pos_emb', blockSize, dModel);
    init(this.tok, 0.08);
    init(this.pos, 0.02);

    this.blocks = [];
    for (let l = 0; l < nLayers; l++) {
      const B: BlockParams = {
        ln1g: p(`L${l}.ln1.gain`, 1, dModel),
        ln1b: p(`L${l}.ln1.bias`, 1, dModel),
        wq: p(`L${l}.attn.Wq`, dModel, dModel),
        wk: p(`L${l}.attn.Wk`, dModel, dModel),
        wv: p(`L${l}.attn.Wv`, dModel, dModel),
        wo: p(`L${l}.attn.Wo`, dModel, dModel),
        bo: p(`L${l}.attn.bo`, 1, dModel),
        ln2g: p(`L${l}.ln2.gain`, 1, dModel),
        ln2b: p(`L${l}.ln2.bias`, 1, dModel),
        w1: p(`L${l}.mlp.W1`, dModel, dFF),
        b1: p(`L${l}.mlp.b1`, 1, dFF),
        w2: p(`L${l}.mlp.W2`, dFF, dModel),
        b2: p(`L${l}.mlp.b2`, 1, dModel),
      };
      ones(B.ln1g);
      ones(B.ln2g);
      const std = 0.02;
      init(B.wq, std);
      init(B.wk, std);
      init(B.wv, std);
      // Residual projections are scaled down by depth so the residual stream
      // does not blow up as layers are stacked.
      init(B.wo, std / Math.sqrt(2 * nLayers));
      init(B.w1, std);
      init(B.w2, std / Math.sqrt(2 * nLayers));
      this.blocks.push(B);
    }

    this.lnfg = p('ln_f.gain', 1, dModel);
    this.lnfb = p('ln_f.bias', 1, dModel);
    ones(this.lnfg);
    this.head = p('head.W', dModel, vocab);
    this.headb = p('head.b', 1, vocab);
    init(this.head, 0.02);

    this.params = [this.tok, this.pos];
    for (const B of this.blocks) {
      this.params.push(
        B.ln1g, B.ln1b, B.wq, B.wk, B.wv, B.wo, B.bo,
        B.ln2g, B.ln2b, B.w1, B.b1, B.w2, B.b2,
      );
    }
    this.params.push(this.lnfg, this.lnfb, this.head, this.headb);
  }

  zeroGrad() {
    for (const q of this.params) q.g.data.fill(0);
  }

  /** Forward pass over one token sequence, keeping every intermediate. */
  forward(tokens: number[], steer?: Steer): { trace: Trace; caches: unknown[] } {
    const { dModel } = this.cfg;
    const T = tokens.length;
    const dh = this.dHead;

    // There are only blockSize rows in the position table. Reading past the
    // end of it returns undefined, which turns the entire forward pass into
    // NaN without anything appearing to go wrong -- the loss becomes NaN, the
    // gradients become NaN, and the model quietly stops learning. Callers are
    // expected to window their input; failing loudly here is what makes a
    // missing window findable instead of mysterious.
    if (T > this.cfg.blockSize) {
      throw new RangeError(
        `forward() got ${T} tokens but this model has only ${this.cfg.blockSize} positions. ` +
          'Window the input to the last blockSize tokens first.',
      );
    }

    const tokenEmb = mat(T, dModel);
    const posEmb = mat(T, dModel);
    for (let t = 0; t < T; t++) {
      for (let j = 0; j < dModel; j++) {
        tokenEmb.data[t * dModel + j] = this.tok.M.data[tokens[t] * dModel + j];
        posEmb.data[t * dModel + j] = this.pos.M.data[t * dModel + j];
      }
    }
    const embedded = mat(T, dModel);
    for (let i = 0; i < embedded.data.length; i++) {
      embedded.data[i] = tokenEmb.data[i] + posEmb.data[i];
    }

    const applySteer = (m: Matrix, atLayer: number) => {
      if (!steer || steer.layer !== atLayer || steer.scale === 0) return m;
      const out = mat(m.rows, m.cols);
      for (let i = 0; i < m.rows; i++) {
        for (let j = 0; j < m.cols; j++) {
          out.data[i * m.cols + j] = m.data[i * m.cols + j] + steer.scale * (steer.vec[j] ?? 0);
        }
      }
      return out;
    };

    let x = applySteer(embedded, -1);
    const layers: LayerTrace[] = [];
    const caches: Record<string, unknown>[] = [];

    for (let l = 0; l < this.cfg.nLayers; l++) {
      const B = this.blocks[l];
      const input = x;
      const { y: ln1, cache: c1 } = layerNormFwd(x, B.ln1g.M, B.ln1b.M);
      const Q = matmul(ln1, B.wq.M);
      const K = matmul(ln1, B.wk.M);
      const V = matmul(ln1, B.wv.M);

      const heads: HeadTrace[] = [];
      const concat = mat(T, dModel);
      const scale = 1 / Math.sqrt(dh);
      for (let h = 0; h < this.cfg.nHeads; h++) {
        const off = h * dh;
        const q = sliceCols(Q, off, dh);
        const k = sliceCols(K, off, dh);
        const v = sliceCols(V, off, dh);
        // scores[i][j] = how much token i wants to read from token j.
        const raw = matmulTB(q, k);
        const scores = mat(T, T);
        for (let i = 0; i < T; i++) {
          for (let j = 0; j < T; j++) {
            // Causal mask: a token may never see the future.
            scores.data[i * T + j] = j <= i ? raw.data[i * T + j] * scale : -1e9;
          }
        }
        const att = softmaxRows(scores);
        const out = matmul(att, v);
        addColsInto(concat, out, off);
        heads.push({ scores, att, q, k, v, out });
      }

      const attnOut = addRowVec(matmul(concat, B.wo.M), B.bo.M.data);
      const afterAttn = mat(T, dModel);
      for (let i = 0; i < afterAttn.data.length; i++) {
        afterAttn.data[i] = input.data[i] + attnOut.data[i];
      }

      const { y: ln2, cache: c2 } = layerNormFwd(afterAttn, B.ln2g.M, B.ln2b.M);
      const pre = addRowVec(matmul(ln2, B.w1.M), B.b1.M.data);
      const ffHidden = mat(pre.rows, pre.cols);
      for (let i = 0; i < pre.data.length; i++) ffHidden.data[i] = ACTIVATIONS.gelu.f(pre.data[i]);
      const ffOut = addRowVec(matmul(ffHidden, B.w2.M), B.b2.M.data);
      const output = mat(T, dModel);
      for (let i = 0; i < output.data.length; i++) {
        output.data[i] = afterAttn.data[i] + ffOut.data[i];
      }

      layers.push({ input, ln1, heads, attnOut, afterAttn, ln2, ffHidden, ffOut, output });
      caches.push({ c1, c2, Q, K, V, concat, pre });
      // Steering is applied to the stream leaving this layer, not to the trace,
      // so the diagrams still show what the layer itself produced.
      x = applySteer(output, l);
    }

    const { y: finalLN, cache: cf } = layerNormFwd(x, this.lnfg.M, this.lnfb.M);
    const logits = addRowVec(matmul(finalLN, this.head.M), this.headb.M.data);
    caches.push({ cf });

    return { trace: { tokens, tokenEmb, posEmb, embedded, layers, finalLN, logits }, caches };
  }

  /** Mean cross-entropy of predicting targets[t] from position t. */
  loss(trace: Trace, targets: number[]): number {
    const probs = softmaxRows(trace.logits);
    const T = targets.length;
    let total = 0;
    for (let t = 0; t < T; t++) {
      total += -Math.log(Math.max(1e-12, probs.data[t * this.cfg.vocab + targets[t]]));
    }
    return total / T;
  }

  /**
   * Backward pass. Gradients ACCUMULATE into param.g, so a caller can sum over
   * a batch of sequences before stepping, exactly like a real training loop.
   */
  backward(fw: { trace: Trace; caches: unknown[] }, targets: number[]): void {
    const { trace, caches } = fw;
    const { dModel, vocab, nHeads } = this.cfg;
    const T = targets.length;
    const dh = this.dHead;

    // d(cross-entropy)/d(logits) = (softmax - onehot) / T
    const probs = softmaxRows(trace.logits);
    const dLogits = mat(T, vocab);
    for (let t = 0; t < T; t++) {
      for (let j = 0; j < vocab; j++) {
        const q = probs.data[t * vocab + j];
        dLogits.data[t * vocab + j] = (q - (j === targets[t] ? 1 : 0)) / T;
      }
    }

    const accum = (dst: Matrix, src: Matrix) => {
      for (let i = 0; i < dst.data.length; i++) dst.data[i] += src.data[i];
    };
    const accumVec = (dst: Matrix, src: Float64Array) => {
      for (let i = 0; i < src.length; i++) dst.data[i] += src[i];
    };

    accum(this.head.g, matmulTA(trace.finalLN, dLogits));
    accumVec(this.headb.g, colSums(dLogits));
    let dx = matmulTB(dLogits, this.head.M);

    const cf = (caches[caches.length - 1] as { cf: LNCache }).cf;
    dx = layerNormBwd(dx, cf, this.lnfg.M, this.lnfg.g, this.lnfb.g);

    for (let l = this.cfg.nLayers - 1; l >= 0; l--) {
      const B = this.blocks[l];
      const L = trace.layers[l];
      const c = caches[l] as { c1: LNCache; c2: LNCache; concat: Matrix; pre: Matrix };

      // --- residual 2: output = afterAttn + ffOut
      const dFF = dx;
      const dAfterAttn = mat(T, dModel);
      accum(dAfterAttn, dx);

      accum(B.w2.g, matmulTA(L.ffHidden, dFF));
      accumVec(B.b2.g, colSums(dFF));
      const dHidden = matmulTB(dFF, B.w2.M);
      const dPre = mat(T, this.cfg.dFF);
      for (let i = 0; i < dPre.data.length; i++) {
        dPre.data[i] = dHidden.data[i] * ACTIVATIONS.gelu.df(c.pre.data[i]);
      }
      accum(B.w1.g, matmulTA(L.ln2, dPre));
      accumVec(B.b1.g, colSums(dPre));
      const dLn2 = matmulTB(dPre, B.w1.M);
      accum(dAfterAttn, layerNormBwd(dLn2, c.c2, B.ln2g.M, B.ln2g.g, B.ln2b.g));

      // --- residual 1: afterAttn = input + attnOut
      const dInput = mat(T, dModel);
      accum(dInput, dAfterAttn);

      accum(B.wo.g, matmulTA(c.concat, dAfterAttn));
      accumVec(B.bo.g, colSums(dAfterAttn));
      const dConcat = matmulTB(dAfterAttn, B.wo.M);

      const dQ = mat(T, dModel);
      const dK = mat(T, dModel);
      const dV = mat(T, dModel);
      const scale = 1 / Math.sqrt(dh);
      for (let h = 0; h < nHeads; h++) {
        const off = h * dh;
        const H = L.heads[h];
        const dOut = sliceCols(dConcat, off, dh);
        // out = att @ v
        const dAtt = matmulTB(dOut, H.v);
        const dV_h = matmulTA(H.att, dOut);
        const dScores = softmaxBwd(H.att, dAtt);
        // Masked entries contribute nothing back.
        for (let i = 0; i < T; i++) {
          for (let j = i + 1; j < T; j++) dScores.data[i * T + j] = 0;
        }
        for (let i = 0; i < dScores.data.length; i++) dScores.data[i] *= scale;
        const dq = matmul(dScores, H.k);
        const dk = matmulTA(dScores, H.q);
        addColsInto(dQ, dq, off);
        addColsInto(dK, dk, off);
        addColsInto(dV, dV_h, off);
      }

      accum(B.wq.g, matmulTA(L.ln1, dQ));
      accum(B.wk.g, matmulTA(L.ln1, dK));
      accum(B.wv.g, matmulTA(L.ln1, dV));
      const dLn1 = mat(T, dModel);
      accum(dLn1, matmulTB(dQ, B.wq.M));
      accum(dLn1, matmulTB(dK, B.wk.M));
      accum(dLn1, matmulTB(dV, B.wv.M));
      accum(dInput, layerNormBwd(dLn1, c.c1, B.ln1g.M, B.ln1g.g, B.ln1b.g));

      dx = dInput;
    }

    // Embeddings: scatter the gradient back to the rows that were looked up.
    for (let t = 0; t < T; t++) {
      const tokenId = trace.tokens[t];
      for (let j = 0; j < dModel; j++) {
        this.tok.g.data[tokenId * dModel + j] += dx.data[t * dModel + j];
        this.pos.g.data[t * dModel + j] += dx.data[t * dModel + j];
      }
    }
  }

  /** Next-token distribution given a context, plus the full trace behind it. */
  predictNext(tokens: number[], steer?: Steer): { probs: number[]; trace: Trace } {
    const ctx = tokens.slice(-this.cfg.blockSize);
    const fw = this.forward(ctx, steer);
    const T = ctx.length;
    const row: number[] = [];
    for (let j = 0; j < this.cfg.vocab; j++) row.push(fw.trace.logits.data[(T - 1) * this.cfg.vocab + j]);
    return { probs: row, trace: fw.trace };
  }

  flatParams(): Float64Array {
    const out = new Float64Array(this.paramCount);
    let k = 0;
    for (const q of this.params) {
      out.set(q.M.data, k);
      k += q.M.data.length;
    }
    return out;
  }

  loadFlat(src: Float64Array) {
    let k = 0;
    for (const q of this.params) {
      q.M.data.set(src.subarray(k, k + q.M.data.length));
      k += q.M.data.length;
    }
  }
}

/** Sampling controls, the knobs that decide how adventurous the output is. */
export interface SampleConfig {
  temperature: number;
  topK: number;
  topP: number;
  seed?: number;
  /**
   * Token ids the sampler may never choose. This is the bluntest guardrail
   * there is: it happens after the model has already decided what it wants,
   * and the model is not consulted about it.
   */
  banned?: Set<number>;
}

/** Why a token did not survive to the draw. */
export type CutReason = 'top-k' | 'top-p' | 'blocked';

export interface SampleStep {
  candidates: {
    id: number;
    logit: number;
    prob: number;
    kept: boolean;
    blocked: boolean;
    /**
     * Which filter removed it, or undefined if it survived. Worth recording
     * rather than inferring: with top-k and top-p both on, whichever binds
     * first is the one that actually did the cutting, and guessing from the
     * settings alone gets it wrong.
     */
    cut?: CutReason;
  }[];
  chosen: number;
  /** Probability mass the ban removed, before renormalising. */
  blockedMass: number;
}

/**
 * Turn logits into one chosen token, showing every filtering stage so the UI
 * can display exactly why a token survived or was cut.
 */
export function sampleToken(logits: number[], cfg: SampleConfig, rand: () => number): SampleStep {
  const t = Math.max(1e-3, cfg.temperature);
  const scaled = logits.map((x) => x / t);
  const max = Math.max(...scaled);
  const exps = scaled.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => e / sum);

  // A banned token is struck out before any other filter runs, so the choice
  // is made from what remains rather than being corrected afterwards.
  let blockedMass = 0;
  if (cfg.banned && cfg.banned.size > 0) {
    for (const id of cfg.banned) {
      if (id >= 0 && id < probs.length) {
        blockedMass += probs[id];
        probs[id] = 0;
      }
    }
    const left = probs.reduce((a, b) => a + b, 0);
    if (left > 0) for (let i = 0; i < probs.length; i++) probs[i] /= left;
  }

  // Banned tokens are excluded from the ranking entirely, so top-k counts k
  // tokens the sampler is actually allowed to pick rather than wasting slots
  // on ones that were already struck out.
  const order = probs
    .map((pr, id) => ({ id, pr }))
    .filter((c) => !cfg.banned?.has(c.id))
    .sort((a, b) => b.pr - a.pr);
  const keep = new Set<number>();
  const cutBy = new Map<number, CutReason>();
  let cumulative = 0;
  const limit = cfg.topK > 0 ? Math.min(cfg.topK, order.length) : order.length;
  let stoppedByP = false;
  let i = 0;
  for (; i < limit; i++) {
    keep.add(order[i].id);
    cumulative += order[i].pr;
    // top-p stops as soon as the kept set covers p of the probability mass.
    if (cfg.topP > 0 && cfg.topP < 1 && cumulative >= cfg.topP) {
      stoppedByP = true;
      i++;
      break;
    }
  }
  // Everything past where the loop stopped was cut, and the reason is
  // whichever of the two limits the loop actually hit.
  for (let j = i; j < order.length; j++) {
    cutBy.set(order[j].id, stoppedByP ? 'top-p' : 'top-k');
  }

  let norm = 0;
  for (const id of keep) norm += probs[id];
  let r = rand() * norm;
  let chosen = order[0]?.id ?? 0;
  for (const item of order) {
    if (!keep.has(item.id)) continue;
    r -= probs[item.id];
    if (r <= 0) {
      chosen = item.id;
      break;
    }
  }

  return {
    candidates: logits.map((logit, id) => ({
      id,
      logit,
      prob: probs[id],
      kept: keep.has(id),
      blocked: cfg.banned?.has(id) ?? false,
      cut: cfg.banned?.has(id) ? ('blocked' as CutReason) : cutBy.get(id),
    })),
    chosen,
    blockedMass,
  };
}
