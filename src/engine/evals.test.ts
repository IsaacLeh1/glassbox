import { describe, expect, it } from 'vitest';
import { Transformer, type TransformerConfig } from './transformer';
import { BPETokenizer } from './tokenizer';
import { getCorpus } from './corpus';
import { LMTrainer, DEFAULT_LM_TRAIN } from './lmTrainer';
import { DEFAULT_TCONFIG } from './transformer';
import { Finetuner } from './finetune';
import {
  BASELINE_INFO,
  byOrigin,
  DEFAULT_GEN,
  METRIC_INFO,
  casesFromIds,
  casesNeeded,
  contaminationOf,
  continueWith,
  makeBaseline,
  modelPredictor,
  runEval,
  separated,
  teacherForcedLoss,
  wilson,
  type EvalCase,
  type MetricId,
} from './evals';

const CFG: TransformerConfig = { vocab: 0, dModel: 16, nHeads: 2, nLayers: 2, blockSize: 16, dFF: 32 };

function build() {
  const text = getCorpus('stories', 4);
  const tok = BPETokenizer.train(text, 90);
  const model = new Transformer({ ...CFG, vocab: tok.size }, 3);
  return { text, tok, model, ids: tok.encode(text) };
}

describe('the confidence interval', () => {
  it('does not claim certainty from a tiny perfect run', () => {
    // The naive interval reports zero width at 8 of 8, which is the exact
    // mistake that makes a small eval set look conclusive.
    const w = wilson(8, 8);
    expect(w.hi).toBe(1);
    expect(w.lo).toBeLessThan(0.75);
  });

  it('narrows as evidence accumulates', () => {
    const small = wilson(5, 10);
    const large = wilson(500, 1000);
    expect(large.hi - large.lo).toBeLessThan(small.hi - small.lo);
  });

  it('brackets the observed rate and stays inside zero to one', () => {
    for (const [k, n] of [[0, 4], [1, 3], [7, 9], [0, 1], [30, 30]]) {
      const w = wilson(k, n);
      expect(w.lo).toBeGreaterThanOrEqual(0);
      expect(w.hi).toBeLessThanOrEqual(1);
      expect(w.lo).toBeLessThanOrEqual(k / n);
      expect(w.hi).toBeGreaterThanOrEqual(k / n);
    }
  });

  it('says something rather than nothing when there are no cases at all', () => {
    expect(wilson(0, 0)).toEqual({ lo: 0, hi: 1 });
  });

  it('estimates how many cases a given precision needs', () => {
    // Tightening the interval by ten times costs a hundred times the cases.
    expect(casesNeeded(0.5, 0.1)).toBeLessThan(casesNeeded(0.5, 0.01));
    expect(casesNeeded(0.5, 0.05)).toBeGreaterThan(300);
  });
});

describe('baselines', () => {
  const { tok, ids } = build();

  it('every baseline returns a real distribution over the whole vocabulary', () => {
    for (const id of Object.keys(BASELINE_INFO) as (keyof typeof BASELINE_INFO)[]) {
      const b = makeBaseline(id, ids, tok.size);
      const p = b.next(tok.encode('the fox'));
      expect(p.length).toBe(tok.size);
      let sum = 0;
      for (const x of p) {
        expect(x).toBeGreaterThanOrEqual(0);
        sum += x;
      }
      expect(sum).toBeCloseTo(1, 6);
    }
  });

  it('the unigram baseline ignores its context and the bigram does not', () => {
    const uni = makeBaseline('unigram', ids, tok.size);
    const bi = makeBaseline('bigram', ids, tok.size);
    const a = tok.encode('the fox');
    const b = tok.encode('a quiet river');
    expect(Array.from(uni.next(a))).toEqual(Array.from(uni.next(b)));
    expect(Array.from(bi.next(a))).not.toEqual(Array.from(bi.next(b)));
  });

  it('the repeat baseline says exactly what it was just given', () => {
    const r = makeBaseline('repeat', ids, tok.size);
    const ctx = tok.encode('the fox');
    const out = continueWith(r, ctx, 3, DEFAULT_GEN);
    expect(out).toEqual([ctx[ctx.length - 1], ctx[ctx.length - 1], ctx[ctx.length - 1]]);
  });

  it('a bigram table really does beat an untrained transformer', () => {
    // The humbling result the module is built around. An untrained model is
    // worse than counting pairs, and that has to be demonstrable, not just
    // asserted in the copy.
    const { tok: t2, model, ids: i2 } = build();
    const cases = casesFromIds(t2, i2, 'heldout', 8, 6, 4);
    const cfg = { metric: 'quality' as MetricId, topK: 5 };
    const m = runEval(modelPredictor(model), t2, cases, cfg, DEFAULT_GEN);
    const b = runEval(makeBaseline('bigram', i2, t2.size), t2, cases, cfg, DEFAULT_GEN);
    expect(b.mean).toBeGreaterThan(m.mean);
  });

  it('a trained model beats pure chance', () => {
    const t = new LMTrainer(getCorpus('stories', 6), { ...DEFAULT_TCONFIG, dModel: 32, nHeads: 2, nLayers: 1, blockSize: 16, dFF: 64 }, { ...DEFAULT_LM_TRAIN, steps: 60 }, 80);
    for (let i = 0; i < 60; i++) t.singleStep(false);
    const cases = casesFromIds(t.tok, t.valIds, 'heldout', 6, 6, 3);
    const cfg = { metric: 'quality' as MetricId, topK: 5 };
    const m = runEval(modelPredictor(t.model), t.tok, cases, cfg, DEFAULT_GEN);
    const u = runEval(makeBaseline('uniform', t.trainIds, t.tok.size), t.tok, cases, cfg, DEFAULT_GEN);
    expect(m.mean).toBeGreaterThan(u.mean);
  });
});

describe('scoring', () => {
  const { tok, model, ids } = build();
  const pred = modelPredictor(model);

  it('the model predictor hands back a real distribution, not raw scores', () => {
    // Regression: predictNext returns a field that used to be called `probs`
    // and has always held logits. Treating them as probabilities gave
    // negative "probabilities", NaN through the sampler, and a teacher-forced
    // loss four times larger than pure chance.
    const p = pred.next(tok.encode('the fox'));
    let sum = 0;
    for (const x of p) {
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThanOrEqual(1);
      sum += x;
    }
    expect(sum).toBeCloseTo(1, 9);
  });

  it('greedy generation is repeatable and sampling is not', () => {
    const ctx = tok.encode('the fox');
    const a = continueWith(pred, ctx, 6, { greedy: true, temperature: 1, seed: 1 });
    const b = continueWith(pred, ctx, 6, { greedy: true, temperature: 1, seed: 999 });
    expect(b).toEqual(a);

    // Sampling is tested against a deliberately flat predictor. An untrained
    // transformer can be peaked enough that two seeds legitimately agree, so
    // using it here would test the model's luck rather than the sampler.
    const flat = makeBaseline('uniform', ids, tok.size);
    const c = continueWith(flat, ctx, 8, { greedy: false, temperature: 1, seed: 1 });
    const d = continueWith(flat, ctx, 8, { greedy: false, temperature: 1, seed: 2 });
    expect(d).not.toEqual(c);
  });

  it('teacher forcing scores the truth, not what the model would have said', () => {
    const ctx = tok.encode('the fox');
    const want = tok.encode(' sat');
    const loss = teacherForcedLoss(pred, ctx, want);
    expect(Number.isFinite(loss)).toBe(true);
    expect(loss).toBeGreaterThan(0);
    // An untrained model over a vocabulary of this size should be near the
    // uniform surprise of log(V).
    expect(loss).toBeLessThan(Math.log(tok.size) * 1.5);
  });

  it('teacher forcing on an empty target is not a number rather than a zero', () => {
    // Silently returning 0 would read as a perfect score.
    expect(Number.isNaN(teacherForcedLoss(pred, tok.encode('the'), []))).toBe(true);
  });

  it('every metric produces a score inside zero to one', () => {
    const cases = casesFromIds(tok, ids, 'heldout', 5, 5, 3);
    for (const metric of Object.keys(METRIC_INFO) as MetricId[]) {
      const run = runEval(pred, tok, cases, { metric, topK: 5 }, DEFAULT_GEN);
      expect(run.total).toBe(5);
      for (const r of run.results) {
        expect(r.score).toBeGreaterThanOrEqual(0);
        expect(r.score).toBeLessThanOrEqual(1);
      }
      expect(run.mean).toBeGreaterThanOrEqual(0);
      expect(run.mean).toBeLessThanOrEqual(1);
      expect(run.lo).toBeLessThanOrEqual(run.mean + 1e-9);
      expect(run.hi).toBeGreaterThanOrEqual(run.mean - 1e-9);
    }
  });

  it('exact match is stricter than contains', () => {
    const cases = casesFromIds(tok, ids, 'heldout', 10, 5, 3);
    const exact = runEval(pred, tok, cases, { metric: 'exact', topK: 5 }, DEFAULT_GEN);
    const contains = runEval(pred, tok, cases, { metric: 'contains', topK: 5 }, DEFAULT_GEN);
    expect(contains.passed).toBeGreaterThanOrEqual(exact.passed);
  });

  it('a larger k never lowers the top-k score', () => {
    const cases = casesFromIds(tok, ids, 'heldout', 10, 5, 2);
    const k1 = runEval(pred, tok, cases, { metric: 'topK', topK: 1 }, DEFAULT_GEN);
    const k10 = runEval(pred, tok, cases, { metric: 'topK', topK: 10 }, DEFAULT_GEN);
    expect(k10.passed).toBeGreaterThanOrEqual(k1.passed);
  });

  it('every metric admits in writing what it misses', () => {
    for (const m of Object.values(METRIC_INFO)) {
      expect(m.blindSpot.length).toBeGreaterThan(40);
    }
  });

  it('handles an empty case list without pretending to a result', () => {
    const run = runEval(pred, tok, [], { metric: 'exact', topK: 5 }, DEFAULT_GEN);
    expect(run.total).toBe(0);
    expect(run.mean).toBe(0);
    // No evidence at all means the full range is still possible.
    expect(run.hi).toBe(1);
  });

  it('scores the model and a baseline through the identical path', () => {
    // If these diverged the comparison would be meaningless, which is the
    // commonest way a benchmark result gets quietly overstated.
    const cases = casesFromIds(tok, ids, 'heldout', 4, 5, 2);
    const cfg = { metric: 'firstToken' as MetricId, topK: 5 };
    const a = runEval(pred, tok, cases, cfg, DEFAULT_GEN);
    const b = runEval(makeBaseline('unigram', ids, tok.size), tok, cases, cfg, DEFAULT_GEN);
    expect(a.results.map((r) => r.case.id)).toEqual(b.results.map((r) => r.case.id));
    expect(a.metric).toBe(b.metric);
  });
});

describe('contamination', () => {
  const train = 'the fox sat on the warm stone and watched the river go by all afternoon';

  it('catches a case lifted straight out of the training text', () => {
    const c = contaminationOf('the fox sat on the warm stone', train);
    expect(c.verbatim).toBe(true);
    expect(c.overlap).toBeCloseTo(1, 6);
  });

  it('reports little overlap for genuinely new text', () => {
    const c = contaminationOf('quantum chromodynamics explains hadrons', train);
    expect(c.verbatim).toBe(false);
    expect(c.overlap).toBeLessThan(0.2);
  });

  it('notices partial overlap rather than only exact copies', () => {
    const c = contaminationOf('the fox sat on the cold stone', train);
    expect(c.verbatim).toBe(false);
    expect(c.overlap).toBeGreaterThan(0.2);
    expect(c.overlap).toBeLessThan(1);
  });

  it('says nothing about empty text instead of dividing by zero', () => {
    expect(contaminationOf('   ', train)).toEqual({ overlap: 0, verbatim: false });
  });

  it('training directly on the test cases inflates the score', () => {
    // The contamination effect, produced deliberately so it can be measured.
    //
    // An earlier version of this test asserted that a normally-trained model
    // scores higher on its training split than on its held-out split. That
    // passed on one seed and is false in general here: the corpora are
    // generated from a handful of templates, so the held-out text is
    // statistically identical to the training text and there is nothing
    // specific to memorise. Whether contamination inflates a score depends on
    // the data, which is why the module measures it rather than asserting it.
    const t = new LMTrainer(
      getCorpus('stories', 6),
      { ...DEFAULT_TCONFIG, dModel: 32, nHeads: 2, nLayers: 1, blockSize: 16, dFF: 64 },
      { ...DEFAULT_LM_TRAIN, steps: 120 },
      80,
    );
    for (let i = 0; i < 120; i++) t.singleStep(false);

    const cfg = { metric: 'quality' as MetricId, topK: 5 };
    const cases = casesFromIds(t.tok, t.valIds, 'heldout', 8, 8, 4);
    const before = runEval(modelPredictor(t.model), t.tok, cases, cfg, DEFAULT_GEN);

    // Now train on the answers themselves, which is what contamination is.
    const ft = new Finetuner(
      t.model,
      t.tok,
      cases.map((c) => `${c.prompt}${c.expected}`),
      { lr: 0.01, epochs: 20, clipNorm: 1 },
    );
    while (ft.microStep()) {
      /* run */
    }
    const after = runEval(modelPredictor(t.model), t.tok, cases, cfg, DEFAULT_GEN);
    expect(after.mean).toBeGreaterThan(before.mean);

    // And it is reversible, so the demonstration can be undone.
    ft.revert();
    const reverted = runEval(modelPredictor(t.model), t.tok, cases, cfg, DEFAULT_GEN);
    expect(reverted.mean).toBeCloseTo(before.mean, 9);
  });
});

describe('splitting a result by where the cases came from', () => {
  const { tok, model, ids } = build();

  it('accounts for every case exactly once', () => {
    const mixed = [
      ...casesFromIds(tok, ids.slice(0, 400), 'train', 4, 6, 2),
      ...casesFromIds(tok, ids.slice(400), 'heldout', 6, 6, 2),
    ];
    const run = runEval(modelPredictor(model), tok, mixed, { metric: 'quality', topK: 5 }, DEFAULT_GEN);
    const slices = byOrigin(run);
    expect(slices.reduce((a, s) => a + s.n, 0)).toBe(mixed.length);
    expect(new Set(slices.map((s) => s.origin)).size).toBe(slices.length);
  });

  it('each slice carries its own interval, and one case is not certain', () => {
    const one: EvalCase[] = [{ id: 'x', prompt: 'the fox', expected: ' sat', origin: 'written' }];
    const run = runEval(modelPredictor(model), tok, one, { metric: 'exact', topK: 5 }, DEFAULT_GEN);
    const [s] = byOrigin(run);
    expect(s.n).toBe(1);
    expect(s.hi - s.lo).toBeGreaterThan(0.3);
  });

  it('returns nothing for a run with no cases', () => {
    const run = runEval(modelPredictor(model), tok, [], { metric: 'exact', topK: 5 }, DEFAULT_GEN);
    expect(byOrigin(run)).toEqual([]);
  });
});

describe('cases cut from a corpus', () => {
  const { tok, ids } = build();

  it('produces the requested number, split into prompt and answer', () => {
    const cases = casesFromIds(tok, ids, 'heldout', 6, 7, 3);
    expect(cases.length).toBe(6);
    for (const c of cases) {
      expect(c.prompt.length).toBeGreaterThan(0);
      expect(c.expected.length).toBeGreaterThan(0);
      expect(c.origin).toBe('heldout');
    }
  });

  it('is deterministic, so two runs are comparable', () => {
    expect(casesFromIds(tok, ids, 'heldout', 5, 6, 3)).toEqual(casesFromIds(tok, ids, 'heldout', 5, 6, 3));
  });

  it('returns nothing rather than garbage when there is not enough text', () => {
    expect(casesFromIds(tok, ids.slice(0, 3), 'heldout', 5, 6, 3)).toEqual([]);
  });
});

describe('comparing two runs', () => {
  const mk = (mean: number, lo: number, hi: number) =>
    ({ mean, lo, hi, predictorId: 'x', predictorLabel: 'x', metric: 'exact', results: [], total: 0, passed: 0, ms: 0 }) as never;

  it('calls overlapping intervals inseparable', () => {
    expect(separated(mk(0.6, 0.4, 0.8), mk(0.5, 0.3, 0.7))).toBe(false);
  });

  it('calls clearly disjoint intervals separated', () => {
    expect(separated(mk(0.9, 0.85, 0.95), mk(0.3, 0.2, 0.4))).toBe(true);
  });

  it('does not care which way round they are given', () => {
    const a = mk(0.9, 0.85, 0.95);
    const b = mk(0.3, 0.2, 0.4);
    expect(separated(a, b)).toBe(separated(b, a));
  });
});

describe('a written case', () => {
  it('works end to end without ever touching a corpus', () => {
    const { tok, model } = build();
    const cases: EvalCase[] = [
      { id: 'a', prompt: 'the fox', expected: ' sat', origin: 'written' },
      { id: 'b', prompt: 'the owl', expected: ' watched', origin: 'written' },
    ];
    const run = runEval(modelPredictor(model), tok, cases, { metric: 'contains', topK: 5 }, DEFAULT_GEN);
    expect(run.total).toBe(2);
    expect(run.results[0].case.origin).toBe('written');
  });
});
