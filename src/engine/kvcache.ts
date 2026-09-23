import { ACTIVATIONS } from './activations';
import type { Transformer, TransformerConfig } from './transformer';

/**
 * The key-value cache: the optimisation that makes serving a model possible.
 *
 * Without it, producing the next token means running the whole context
 * through the network again from scratch. Producing n tokens therefore costs
 * roughly n-squared work, because token 100 re-reads all 99 that came before
 * it, and token 101 re-reads all 100, and so on for ever.
 *
 * The saving comes from noticing that almost none of that work changes. When
 * a new token arrives, every earlier token's key and value vectors are
 * exactly what they were a moment ago -- they cannot depend on a token that
 * had not been written yet, because attention is causal. So they are kept,
 * and the new token computes only its own. Total cost falls from n-squared to
 * n, and the price is memory: every layer, every head, every position held
 * for as long as the conversation lasts.
 *
 * That memory is the reason serving a model is expensive and the reason
 * context windows cost what they do. It is also why the two halves of
 * generation behave so differently:
 *
 *  - prefill, where the prompt is processed, does all its positions at once
 *    and is limited by arithmetic,
 *  - decode, one token at a time, is limited by how fast the weights can be
 *    read out of memory, and is the part that sets how fast text appears.
 *
 * Everything here is real. A test asserts the cached path produces logits
 * identical to the uncached one, because an optimisation that changes the
 * answer is not an optimisation.
 */

export interface KVCache {
  /** [layer][head] -> keys, laid out as maxLen rows of dHead. */
  k: Float64Array[][];
  v: Float64Array[][];
  /** Positions filled so far. */
  len: number;
  maxLen: number;
  dHead: number;
  /** Bytes of live cache, which is what has to be held per concurrent request. */
  bytes: number;
}

export function emptyCache(cfg: TransformerConfig): KVCache {
  const dHead = Math.floor(cfg.dModel / cfg.nHeads);
  const k: Float64Array[][] = [];
  const v: Float64Array[][] = [];
  for (let l = 0; l < cfg.nLayers; l++) {
    const kl: Float64Array[] = [];
    const vl: Float64Array[] = [];
    for (let h = 0; h < cfg.nHeads; h++) {
      kl.push(new Float64Array(cfg.blockSize * dHead));
      vl.push(new Float64Array(cfg.blockSize * dHead));
    }
    k.push(kl);
    v.push(vl);
  }
  return {
    k,
    v,
    len: 0,
    maxLen: cfg.blockSize,
    dHead,
    // Two tensors, every layer, every head, every position.
    bytes: 2 * cfg.nLayers * cfg.nHeads * cfg.blockSize * dHead * 8,
  };
}

/** How large the cache grows for a given shape, without building one. */
export function cacheBytes(cfg: TransformerConfig, contextLen: number, bytesPerNumber = 2): number {
  const dHead = Math.floor(cfg.dModel / cfg.nHeads);
  return 2 * cfg.nLayers * cfg.nHeads * contextLen * dHead * bytesPerNumber;
}

function layerNormRow(x: Float64Array, g: Float64Array, b: Float64Array): Float64Array {
  const d = x.length;
  let mean = 0;
  for (let j = 0; j < d; j++) mean += x[j];
  mean /= d;
  let varr = 0;
  for (let j = 0; j < d; j++) varr += (x[j] - mean) ** 2;
  varr /= d;
  const inv = 1 / Math.sqrt(varr + 1e-5);
  const out = new Float64Array(d);
  for (let j = 0; j < d; j++) out[j] = (x[j] - mean) * inv * g[j] + b[j];
  return out;
}

/** row (1 x n) times matrix (n x m), the only shape the decode path needs. */
function rowMat(row: Float64Array, M: { rows: number; cols: number; data: Float64Array }): Float64Array {
  const out = new Float64Array(M.cols);
  for (let k = 0; k < M.rows; k++) {
    const a = row[k];
    if (a === 0) continue;
    const off = k * M.cols;
    for (let j = 0; j < M.cols; j++) out[j] += a * M.data[off + j];
  }
  return out;
}

export interface DecodeStep {
  logits: Float64Array;
  /** Position this token occupied. */
  pos: number;
  /** Attention each head paid to each earlier position, for display. */
  attention: Float64Array[][];
}

/**
 * Process exactly one token, reusing everything already in the cache.
 *
 * This is the decode step, and it is the operation a serving system spends
 * almost all of its time in.
 */
export function decodeToken(model: Transformer, token: number, cache: KVCache): DecodeStep {
  const { dModel, nHeads, nLayers, vocab } = model.cfg;
  const dh = cache.dHead;
  const pos = cache.len;
  if (pos >= cache.maxLen) {
    throw new RangeError(
      `the cache holds ${cache.maxLen} positions and is full. A real server evicts here, or refuses the request.`,
    );
  }

  // Embedding plus position, exactly as the uncached path does it.
  let x = new Float64Array(dModel);
  for (let j = 0; j < dModel; j++) {
    x[j] = model.tok.M.data[token * dModel + j] + model.pos.M.data[pos * dModel + j];
  }

  const attention: Float64Array[][] = [];
  const scale = 1 / Math.sqrt(dh);

  for (let l = 0; l < nLayers; l++) {
    const B = model.blocks[l];
    const input = x;
    const ln1 = layerNormRow(x, B.ln1g.M.data, B.ln1b.M.data);

    const qAll = rowMat(ln1, B.wq.M);
    const kAll = rowMat(ln1, B.wk.M);
    const vAll = rowMat(ln1, B.wv.M);

    const concat = new Float64Array(dModel);
    const layerAtt: Float64Array[] = [];

    for (let h = 0; h < nHeads; h++) {
      const off = h * dh;
      const kRow = cache.k[l][h];
      const vRow = cache.v[l][h];

      // The new token's key and value join the cache and stay there.
      for (let j = 0; j < dh; j++) {
        kRow[pos * dh + j] = kAll[off + j];
        vRow[pos * dh + j] = vAll[off + j];
      }

      // Score this token against every position up to and including itself.
      const n = pos + 1;
      const scores = new Float64Array(n);
      let max = -Infinity;
      for (let t = 0; t < n; t++) {
        let s = 0;
        for (let j = 0; j < dh; j++) s += qAll[off + j] * kRow[t * dh + j];
        s *= scale;
        scores[t] = s;
        if (s > max) max = s;
      }
      let sum = 0;
      for (let t = 0; t < n; t++) {
        scores[t] = Math.exp(scores[t] - max);
        sum += scores[t];
      }
      for (let t = 0; t < n; t++) scores[t] /= sum;

      for (let t = 0; t < n; t++) {
        const w = scores[t];
        for (let j = 0; j < dh; j++) concat[off + j] += w * vRow[t * dh + j];
      }
      layerAtt.push(scores);
    }
    attention.push(layerAtt);

    const projected = rowMat(concat, B.wo.M);
    const afterAttn = new Float64Array(dModel);
    for (let j = 0; j < dModel; j++) afterAttn[j] = input[j] + projected[j] + B.bo.M.data[j];

    const ln2 = layerNormRow(afterAttn, B.ln2g.M.data, B.ln2b.M.data);
    const hidden = rowMat(ln2, B.w1.M);
    for (let j = 0; j < hidden.length; j++) hidden[j] = ACTIVATIONS.gelu.f(hidden[j] + B.b1.M.data[j]);
    const ffOut = rowMat(hidden, B.w2.M);

    const out = new Float64Array(dModel);
    for (let j = 0; j < dModel; j++) out[j] = afterAttn[j] + ffOut[j] + B.b2.M.data[j];
    x = out;
  }

  const finalLN = layerNormRow(x, model.lnfg.M.data, model.lnfb.M.data);
  const logits = rowMat(finalLN, model.head.M);
  for (let j = 0; j < vocab; j++) logits[j] += model.headb.M.data[j];

  cache.len = pos + 1;
  return { logits, pos, attention };
}

/** Fill the cache from a prompt, returning the logits after the last token. */
export function prefill(model: Transformer, tokens: number[], cache: KVCache): DecodeStep | null {
  let last: DecodeStep | null = null;
  for (const t of tokens) last = decodeToken(model, t, cache);
  return last;
}

/* ------------------------------------------------------------ measuring */

export interface Timing {
  /** Milliseconds to process the prompt. */
  prefillMs: number;
  /** Milliseconds per generated token, averaged. */
  perTokenMs: number;
  promptTokens: number;
  generated: number;
  tokensPerSecond: number;
}

/** Time a full request the way it is actually served, with a cache. */
export function timeCached(model: Transformer, prompt: number[], generate: number): Timing {
  const cache = emptyCache(model.cfg);
  const t0 = performance.now();
  let step = prefill(model, prompt, cache);
  const t1 = performance.now();

  let n = 0;
  for (let i = 0; i < generate && cache.len < cache.maxLen; i++) {
    let best = 0;
    const lg = step!.logits;
    for (let j = 1; j < lg.length; j++) if (lg[j] > lg[best]) best = j;
    step = decodeToken(model, best, cache);
    n++;
  }
  const t2 = performance.now();

  return {
    prefillMs: t1 - t0,
    perTokenMs: n > 0 ? (t2 - t1) / n : 0,
    promptTokens: prompt.length,
    generated: n,
    tokensPerSecond: t2 > t1 && n > 0 ? n / ((t2 - t1) / 1000) : 0,
  };
}

/** Time the same request without a cache, re-reading the context every step. */
export function timeUncached(model: Transformer, prompt: number[], generate: number): Timing {
  const t0 = performance.now();
  let ids = [...prompt];
  model.forward(ids.slice(-model.cfg.blockSize));
  const t1 = performance.now();

  let n = 0;
  for (let i = 0; i < generate && ids.length < model.cfg.blockSize; i++) {
    const { logits } = model.predictNext(ids);
    let best = 0;
    for (let j = 1; j < logits.length; j++) if (logits[j] > logits[best]) best = j;
    ids = [...ids, best];
    n++;
  }
  const t2 = performance.now();

  return {
    prefillMs: t1 - t0,
    perTokenMs: n > 0 ? (t2 - t1) / n : 0,
    promptTokens: prompt.length,
    generated: n,
    tokensPerSecond: t2 > t1 && n > 0 ? n / ((t2 - t1) / 1000) : 0,
  };
}
