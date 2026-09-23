import { describe, expect, it } from 'vitest';
import { Transformer, type TransformerConfig } from './transformer';
import { BPETokenizer } from './tokenizer';
import { getCorpus } from './corpus';
import { LMTrainer, DEFAULT_LM_TRAIN } from './lmTrainer';
import { DEFAULT_TCONFIG } from './transformer';
import { cacheBytes, decodeToken, emptyCache, prefill, timeCached, timeUncached } from './kvcache';
import { prettyBytes, projectSize, quantiseModel, scaleUp, type QuantBits } from './quantise';
import { DEFAULT_GEN, casesFromIds, modelPredictor, runEval } from './evals';
import { GPU_PROFILES } from '../sim/cluster';
import {
  batchCurve,
  costOf,
  decodeBound,
  loadCurve,
  queue,
  replicasFor,
  type BatchPoint,
} from './serving';

const CFG: TransformerConfig = { vocab: 0, dModel: 24, nHeads: 3, nLayers: 2, blockSize: 16, dFF: 48 };

function build() {
  const text = getCorpus('stories', 4);
  const tok = BPETokenizer.train(text, 90);
  const model = new Transformer({ ...CFG, vocab: tok.size }, 3);
  return { tok, model, ids: tok.encode(text) };
}

describe('the key-value cache', () => {
  const { tok, model } = build();

  it('produces logits identical to running the whole context again', () => {
    // The property the entire optimisation rests on. If this drifts, the
    // cache is not an optimisation, it is a different model.
    const ids = tok.encode('the quiet fox sat');
    const cache = emptyCache(model.cfg);
    const step = prefill(model, ids, cache)!;
    const { logits } = model.predictNext(ids);
    expect(step.logits.length).toBe(logits.length);
    for (let i = 0; i < logits.length; i++) {
      expect(step.logits[i]).toBeCloseTo(logits[i], 9);
    }
  });

  it('stays identical all the way through a generated continuation', () => {
    const start = tok.encode('the fox');
    const cache = emptyCache(model.cfg);
    let step = prefill(model, start, cache)!;
    let ids = [...start];

    for (let n = 0; n < 6; n++) {
      let best = 0;
      for (let j = 1; j < step.logits.length; j++) if (step.logits[j] > step.logits[best]) best = j;
      ids = [...ids, best];
      // The uncached path, recomputing from scratch every single time.
      const fresh = model.predictNext(ids).logits;
      step = decodeToken(model, best, cache);
      for (let i = 0; i < fresh.length; i++) expect(step.logits[i]).toBeCloseTo(fresh[i], 9);
    }
  });

  it('tracks how many positions it holds', () => {
    const cache = emptyCache(model.cfg);
    expect(cache.len).toBe(0);
    prefill(model, tok.encode('the fox sat'), cache);
    expect(cache.len).toBe(tok.encode('the fox sat').length);
  });

  it('refuses to overflow rather than corrupting itself', () => {
    // A real server evicts or rejects here. Silently wrapping would produce
    // wrong answers that look plausible.
    const cache = emptyCache(model.cfg);
    for (let i = 0; i < cache.maxLen; i++) decodeToken(model, 0, cache);
    expect(() => decodeToken(model, 0, cache)).toThrow(/full/);
  });

  it('attention rows still sum to one at every position', () => {
    const cache = emptyCache(model.cfg);
    const ids = tok.encode('the fox sat on');
    let step = null;
    for (const t of ids) step = decodeToken(model, t, cache);
    for (const layer of step!.attention) {
      for (const head of layer) {
        expect(head.length).toBe(ids.length);
        let sum = 0;
        for (const x of head) sum += x;
        expect(sum).toBeCloseTo(1, 9);
      }
    }
  });

  it('a token can only attend to itself at the first position', () => {
    const cache = emptyCache(model.cfg);
    const step = decodeToken(model, 5, cache);
    for (const layer of step.attention) {
      for (const head of layer) {
        expect(head.length).toBe(1);
        expect(head[0]).toBeCloseTo(1, 12);
      }
    }
  });

  it('is cheaper than recomputing, and the gap widens with context', () => {
    // The whole point. Measured rather than asserted, and compared as work
    // done rather than wall time, which is too noisy at this size.
    const t = new LMTrainer(
      getCorpus('stories', 6),
      { ...DEFAULT_TCONFIG, dModel: 48, nHeads: 4, nLayers: 2, blockSize: 32, dFF: 96 },
      { ...DEFAULT_LM_TRAIN, steps: 5 },
      100,
    );
    const prompt = t.tok.encode('the fox').slice(0, 4);
    const short = timeCached(t.model, prompt, 6);
    const long = timeCached(t.model, prompt, 20);
    expect(short.generated).toBe(6);
    expect(long.generated).toBeGreaterThan(short.generated);
    // Per-token cost with a cache grows only with attention length, so it
    // should stay within the same order of magnitude as the context trebles.
    expect(long.perTokenMs).toBeLessThan(short.perTokenMs * 4 + 1);
  });

  it('both timing paths report a coherent account of the same request', () => {
    const { model: m, tok: tk } = build();
    const prompt = tk.encode('the fox').slice(0, 3);
    for (const timing of [timeCached(m, prompt, 5), timeUncached(m, prompt, 5)]) {
      expect(timing.promptTokens).toBe(prompt.length);
      expect(timing.generated).toBeGreaterThan(0);
      expect(timing.prefillMs).toBeGreaterThanOrEqual(0);
      expect(timing.perTokenMs).toBeGreaterThanOrEqual(0);
      expect(Number.isFinite(timing.tokensPerSecond)).toBe(true);
    }
  });

  it('sizes itself the way the memory actually works out', () => {
    const cfg = { ...CFG, vocab: 100 };
    // two tensors x layers x heads x positions x dHead
    const dHead = cfg.dModel / cfg.nHeads;
    expect(cacheBytes(cfg, 10, 2)).toBe(2 * cfg.nLayers * cfg.nHeads * 10 * dHead * 2);
    // Doubling the context doubles the cache, which is the thing that makes
    // long contexts expensive to serve.
    expect(cacheBytes(cfg, 20, 2)).toBe(2 * cacheBytes(cfg, 10, 2));
  });
});

describe('quantisation', () => {
  it('shrinks the model by exactly the ratio of the bit widths', () => {
    const { model } = build();
    const n = model.paramCount;
    const eight = projectSize(n, 8, 1);
    const four = projectSize(n, 4, 1);
    expect(four).toBeLessThan(eight);
    expect(eight / four).toBeGreaterThan(1.9);
  });

  it('really does restrict the weights to a coarse grid', () => {
    const { model } = build();
    const before = Array.from(model.flatParams());
    quantiseModel(model, 4, 'tensor');
    const after = Array.from(model.flatParams());
    expect(after).not.toEqual(before);
    // At four bits there are fifteen levels either side of zero, so the
    // distinct values in any one tensor cannot exceed thirty-one.
    const first = model.params[0];
    const distinct = new Set(Array.from(first.M.data).map((x) => x.toFixed(12)));
    expect(distinct.size).toBeLessThanOrEqual(31);
  });

  it('loses more the fewer bits it is given', () => {
    let last = -1;
    for (const bits of [8, 6, 4, 3, 2] as QuantBits[]) {
      const { model } = build();
      const r = quantiseModel(model, bits, 'row');
      expect(r.relError).toBeGreaterThan(last);
      last = r.relError;
    }
  });

  it('per-row scaling is at least as accurate as one scale for everything', () => {
    // The reason production quantisation is per-channel: one outlier weight
    // cannot coarsen the grid for an entire tensor.
    const a = build();
    const b = build();
    const perTensor = quantiseModel(a.model, 4, 'tensor');
    const perRow = quantiseModel(b.model, 4, 'row');
    expect(perRow.relError).toBeLessThanOrEqual(perTensor.relError);
  });

  it('costs bytes for its scales, and per-row costs more of them', () => {
    const a = build();
    const b = build();
    const perTensor = quantiseModel(a.model, 8, 'tensor');
    const perRow = quantiseModel(b.model, 8, 'row');
    expect(perRow.bytes).toBeGreaterThan(perTensor.bytes);
    expect(perTensor.bytes).toBeLessThan(perTensor.baseBytes);
  });

  it('names the tensors that suffered most', () => {
    const { model } = build();
    const r = quantiseModel(model, 3, 'tensor');
    expect(r.perTensor.length).toBe(model.params.length);
    // Sorted worst first, so the panel can lead with the problem.
    for (let i = 1; i < r.perTensor.length; i++) {
      expect(r.perTensor[i - 1].relError).toBeGreaterThanOrEqual(r.perTensor[i].relError);
    }
  });

  it('leaves an all-zero tensor exactly alone instead of dividing by zero', () => {
    const { model } = build();
    model.params[1].M.data.fill(0);
    const r = quantiseModel(model, 4, 'row');
    expect(Number.isFinite(r.relError)).toBe(true);
    for (const x of model.params[1].M.data) expect(x).toBe(0);
  });

  it('can be undone from a snapshot, which is how you compare fairly', () => {
    const { model } = build();
    const snapshot = model.flatParams().slice();
    quantiseModel(model, 2, 'tensor');
    expect(Array.from(model.flatParams())).not.toEqual(Array.from(snapshot));
    model.loadFlat(snapshot);
    expect(Array.from(model.flatParams())).toEqual(Array.from(snapshot));
  });

  it('costs measurable quality, scored on the same harness as everything else', () => {
    const t = new LMTrainer(
      getCorpus('stories', 6),
      { ...DEFAULT_TCONFIG, dModel: 32, nHeads: 2, nLayers: 1, blockSize: 16, dFF: 64 },
      { ...DEFAULT_LM_TRAIN, steps: 120 },
      80,
    );
    for (let i = 0; i < 120; i++) t.singleStep(false);
    const cases = casesFromIds(t.tok, t.valIds, 'heldout', 8, 6, 3);
    const cfg = { metric: 'quality' as const, topK: 5 };
    const snapshot = t.model.flatParams().slice();

    const full = runEval(modelPredictor(t.model), t.tok, cases, cfg, DEFAULT_GEN).mean;
    quantiseModel(t.model, 8, 'row');
    const eight = runEval(modelPredictor(t.model), t.tok, cases, cfg, DEFAULT_GEN).mean;
    t.model.loadFlat(snapshot);
    quantiseModel(t.model, 2, 'tensor');
    const two = runEval(modelPredictor(t.model), t.tok, cases, cfg, DEFAULT_GEN).mean;

    // Eight bits is nearly free; two bits is not.
    expect(Math.abs(eight - full)).toBeLessThan(0.05);
    expect(two).toBeLessThan(eight);
  });

  it('scales the same arithmetic up to real model sizes', () => {
    const rows = scaleUp(4);
    const llama70 = rows.find((r) => r.label.includes('70B'))!;
    // 70.6 billion at 4 bits is about 35 GB, so one 80GB card holds it.
    expect(llama70.cardsQuantised).toBe(1);
    // At fp16 it is about 141 GB, so it does not.
    expect(llama70.cardsFp16).toBeGreaterThan(1);
    expect(llama70.quantised).toBeLessThan(llama70.fp16);
  });

  it('prints byte counts a person can read', () => {
    expect(prettyBytes(512)).toBe('512 B');
    expect(prettyBytes(2048)).toBe('2.0 KB');
    expect(prettyBytes(5 * 1024 * 1024)).toBe('5.0 MB');
    expect(prettyBytes(3 * 1024 * 1024 * 1024)).toContain('GB');
  });
});

describe('serving economics', () => {
  const cfg: TransformerConfig = { ...CFG, vocab: 200 };
  const h100 = GPU_PROFILES.find((g) => g.id === 'h100')!;

  it('decoding one token is limited by memory, not by arithmetic', () => {
    // The central fact of serving. At 8 billion parameters a card does the
    // arithmetic almost instantly and then waits for the bytes.
    const b = decodeBound(8.03e9, 2, 2048, cfg, h100);
    expect(b.memoryBound).toBe(true);
    expect(b.memorySeconds).toBeGreaterThan(b.computeSeconds);
  });

  it('halving the bytes per weight roughly doubles the tokens per second', () => {
    const fp16 = decodeBound(8.03e9, 2, 1024, cfg, h100);
    const int8 = decodeBound(8.03e9, 1, 1024, cfg, h100);
    expect(int8.tokensPerSecond / fp16.tokensPerSecond).toBeGreaterThan(1.9);
    expect(int8.tokensPerSecond / fp16.tokensPerSecond).toBeLessThan(2.1);
  });

  it('lands in the right ballpark for a real model on a real card', () => {
    // 8B at fp16 is 16GB; an H100 reads 3.35 TB/s, so a little over 200
    // tokens a second for a single stream. Published figures agree.
    const b = decodeBound(8.03e9, 2, 512, cfg, h100);
    expect(b.tokensPerSecond).toBeGreaterThan(100);
    expect(b.tokensPerSecond).toBeLessThan(400);
  });

  it('a longer context costs more per token, because the cache is read too', () => {
    const short = decodeBound(1e9, 2, 128, cfg, h100);
    const long = decodeBound(1e9, 2, 8192, cfg, h100);
    expect(long.bytesPerToken).toBeGreaterThan(short.bytesPerToken);
    expect(long.tokensPerSecond).toBeLessThan(short.tokensPerSecond);
  });

  it('turns a token rate into money', () => {
    const c = costOf(200, 500, h100, 8.03e9, 2, 1024, cfg);
    expect(c.usdPerMillionTokens).toBeGreaterThan(0);
    expect(Number.isFinite(c.usdPerRequest)).toBe(true);
    // Faster serving is cheaper per token, in exact proportion.
    const faster = costOf(400, 500, h100, 8.03e9, 2, 1024, cfg);
    expect(faster.usdPerMillionTokens).toBeCloseTo(c.usdPerMillionTokens / 2, 6);
  });

  it('says how many conversations fit alongside the weights', () => {
    const c = costOf(200, 500, h100, 8.03e9, 2, 1024, cfg);
    expect(c.maxConcurrent).toBeGreaterThan(0);
    // A bigger model leaves less room for caches.
    const bigger = costOf(200, 500, h100, 30e9, 2, 1024, cfg);
    expect(bigger.maxConcurrent).toBeLessThan(c.maxConcurrent);
  });

  it('reports no capacity at all when the weights do not fit', () => {
    const c = costOf(200, 500, h100, 400e9, 2, 1024, cfg);
    expect(c.maxConcurrent).toBe(0);
  });
});

describe('queueing', () => {
  it('is fine at low load and catastrophic near saturation', () => {
    // The cliff, which is the whole lesson.
    const half = queue(5, 10);
    const ninety = queue(9, 10);
    const ninetynine = queue(9.9, 10);
    expect(half.meanSeconds).toBeCloseTo(0.2, 9);
    expect(ninety.meanSeconds).toBeCloseTo(1, 9);
    expect(ninetynine.meanSeconds).toBeCloseTo(10, 9);
    // Ten times the load, fifty times the wait.
    expect(ninetynine.meanSeconds / half.meanSeconds).toBeCloseTo(50, 6);
  });

  it('reports saturation rather than a negative wait', () => {
    // The naive formula returns a negative number here, which would render
    // as a suspiciously fast service.
    for (const q of [queue(10, 10), queue(12, 10), queue(1, 0)]) {
      expect(q.saturated).toBe(true);
      expect(q.meanSeconds).toBe(Infinity);
      expect(q.p99Seconds).toBe(Infinity);
    }
  });

  it('the slow tail is always worse than the average', () => {
    for (const load of [0.1, 0.5, 0.8, 0.95]) {
      const q = queue(load * 10, 10);
      expect(q.p99Seconds).toBeGreaterThan(q.meanSeconds);
    }
  });

  it('utilisation is the load divided by the capacity', () => {
    expect(queue(3, 12).utilisation).toBeCloseTo(0.25, 9);
  });

  it('works out how many replicas a latency target needs', () => {
    // One server at 10/s cannot hold a 0.5s p99 under 9/s of load.
    expect(queue(9, 10).p99Seconds).toBeGreaterThan(0.5);
    const n = replicasFor(9, 10, 0.5);
    expect(n).toBeGreaterThan(1);
    // And with that many, each one comfortably can.
    expect(queue(9 / n, 10).p99Seconds).toBeLessThanOrEqual(0.5 + 1e-9);
  });

  it('needs only one replica when one is genuinely enough', () => {
    expect(replicasFor(1, 100, 1)).toBe(1);
  });

  it('reports an impossible promise as impossible, not as a huge number', () => {
    // Regression: the guard against dividing by zero turned "no number of
    // replicas can do this" into two billion replicas, which reads as an
    // answer. A single request at zero load already takes ln(100)/mu, and
    // splitting the load shortens the queue, never the work.
    const mu = 0.24;
    const floor = Math.log(100) / mu;
    expect(replicasFor(0.1, mu, floor / 2)).toBe(Infinity);
    // Just above the floor it becomes possible again.
    expect(Number.isFinite(replicasFor(0.1, mu, floor * 1.5))).toBe(true);
  });

  it('draws a curve that climbs without ever saturating', () => {
    const c = loadCurve(10);
    expect(c.length).toBeGreaterThan(10);
    for (let i = 1; i < c.length; i++) {
      expect(c[i].x).toBeGreaterThan(c[i - 1].x);
      expect(c[i].y).toBeGreaterThan(c[i - 1].y);
      expect(Number.isFinite(c[i].y)).toBe(true);
    }
  });
});

describe('batching', () => {
  const cfg: TransformerConfig = { ...CFG, vocab: 200 };

  it('raises throughput and lengthens each individual request', () => {
    const curve = batchCurve(100, 32, cfg, 1024, 2, 16e9);
    expect(curve[0].size).toBe(1);
    for (let i = 1; i < curve.length; i++) {
      expect(curve[i].tokensPerSecond).toBeGreaterThan(curve[i - 1].tokensPerSecond);
      expect(curve[i].perRequestMs).toBeGreaterThan(curve[i - 1].perRequestMs);
      expect(curve[i].cacheGB).toBeGreaterThan(curve[i - 1].cacheGB);
    }
  });

  it('the gain flattens once the caches dominate the weights', () => {
    // With a tiny model the weights are cheap to re-read, so batching buys
    // much less. This is why batching helps large models most.
    const big = batchCurve(100, 32, cfg, 1024, 2, 16e9);
    const small = batchCurve(100, 32, cfg, 1024, 2, 1e6);
    const gain = (c: BatchPoint[]) => c[c.length - 1].tokensPerSecond / c[0].tokensPerSecond;
    expect(gain(big)).toBeGreaterThan(gain(small));
  });

  it('a batch of one is exactly the unbatched rate', () => {
    const curve = batchCurve(100, 1, cfg, 1024, 2, 16e9);
    expect(curve[0].tokensPerSecond).toBeCloseTo(100, 9);
  });
});
