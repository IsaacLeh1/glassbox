import { describe, expect, it } from 'vitest';
import {
  STAGES,
  analyseSample,
  diagnose,
  recentSlope,
  stageFor,
  summariseRun,
  type RunFacts,
} from './diagnostics';
import { prettyRenderer, crossoverSize, BENCH_SIZES, type BenchPoint } from './compute';
import type { LMMetrics } from './lmTrainer';

const CORPUS = 'the quiet fox sleeps in the garden at dawn. the red cat waits by the river at night.';

function hist(losses: number[], opts: Partial<LMMetrics> = {}): LMMetrics[] {
  return losses.map((loss, i) => ({
    step: (i + 1) * 5,
    loss,
    valLoss: opts.valLoss ?? loss + 0.02,
    perplexity: Math.exp(loss),
    gradNorm: opts.gradNorm ?? 1,
    lr: 0.01,
    tokensSeen: (i + 1) * 200,
    wallMs: (i + 1) * 100,
  }));
}

const facts = (over: Partial<RunFacts> = {}): RunFacts => ({
  history: hist([5, 4, 3.4, 3.1, 3.0, 2.95, 2.9]),
  vocab: 187,
  paramCount: 56000,
  trainTokens: 30000,
  blockSize: 24,
  totalSteps: 400,
  corpusChars: 120000,
  ...over,
});

describe('sample analysis', () => {
  it('recognises text made of real corpus words', () => {
    const a = analyseSample('the fox sleeps in the garden', CORPUS);
    expect(a.realWordRatio).toBe(1);
    expect(a.looping).toBe(false);
  });

  it('recognises gibberish as not being real words', () => {
    const a = analyseSample('thq brx zlmp ov dre gxrden', CORPUS);
    expect(a.realWordRatio).toBeLessThan(0.2);
  });

  it('detects a looping sample', () => {
    const a = analyseSample('the cat the cat the cat the cat the cat the cat', CORPUS);
    expect(a.repetitionRatio).toBeGreaterThan(0.5);
    expect(a.looping).toBe(true);
  });

  it('does not call ordinary varied text a loop', () => {
    const a = analyseSample('the quiet fox sleeps in the garden at dawn near a river', CORPUS);
    expect(a.looping).toBe(false);
  });

  it('measures spacing against the corpus', () => {
    const a = analyseSample('the fox sleeps', CORPUS);
    expect(a.targetSpaceRatio).toBeGreaterThan(0);
    expect(a.spaceRatio).toBeGreaterThan(0);
  });

  it('survives an empty sample without throwing', () => {
    const a = analyseSample('', CORPUS);
    expect(a.words).toBe(0);
    expect(a.realWordRatio).toBe(0);
    expect(a.looping).toBe(false);
  });
});

describe('learning stage', () => {
  const uniform = Math.log(187);

  it('calls an untrained model noise', () => {
    const a = analyseSample('qx zz vv', CORPUS);
    expect(stageFor(uniform, 187, a)).toBe('noise');
  });

  it('calls partially trained gibberish the character stage', () => {
    const a = analyseSample('thq brx zlmp ov dre', CORPUS);
    expect(stageFor(uniform * 0.7, 187, a)).toBe('characters');
  });

  it('calls real words with poor order the word stage', () => {
    const a = analyseSample('garden the sleeps fox dawn zzz qqq', CORPUS);
    expect(['words', 'grammar']).toContain(stageFor(uniform * 0.6, 187, a));
  });

  it('calls a well-trained coherent sample the structure stage', () => {
    const a = analyseSample('the quiet fox sleeps in the garden at dawn', CORPUS);
    expect(stageFor(uniform * 0.3, 187, a)).toBe('structure');
  });

  it('every stage has copy written for it', () => {
    for (const k of ['noise', 'characters', 'words', 'grammar', 'structure'] as const) {
      expect(STAGES[k].label.length).toBeGreaterThan(2);
      expect(STAGES[k].what.length).toBeGreaterThan(40);
      expect(STAGES[k].next.length).toBeGreaterThan(20);
    }
  });
});

describe('loss trend', () => {
  it('reports a negative slope while the loss is falling', () => {
    expect(recentSlope(hist([5, 4, 3, 2, 1]))).toBeLessThan(0);
  });

  it('reports roughly zero when the loss is flat', () => {
    expect(Math.abs(recentSlope(hist([2, 2, 2, 2, 2])))).toBeLessThan(1e-9);
  });

  it('reports a positive slope when the loss is climbing', () => {
    expect(recentSlope(hist([1, 2, 3, 4, 5]))).toBeGreaterThan(0);
  });
});

describe('diagnosis', () => {
  it('says nothing at all before there is data', () => {
    expect(diagnose(facts({ history: [] }), null, CORPUS)).toEqual([]);
  });

  it('reports divergence and stops there', () => {
    const f = facts({ history: hist([5, 20, Number.POSITIVE_INFINITY]) });
    const out = diagnose(f, null, CORPUS);
    expect(out).toHaveLength(1);
    expect(out[0].id).toBe('diverged');
    expect(out[0].severity).toBe('bad');
    expect(out[0].fix).toBeTruthy();
  });

  it('always reports progress relative to uniform guessing', () => {
    const out = diagnose(facts(), null, CORPUS);
    const p = out.find((f) => f.id === 'progress');
    expect(p).toBeDefined();
    expect(p!.detail).toContain('perplexity');
  });

  it('flags a flat loss that is still far from good', () => {
    const f = facts({ history: hist([5, 5, 5, 5, 5, 5, 5, 5]) });
    expect(diagnose(f, null, CORPUS).some((x) => x.id === 'stalled')).toBe(true);
  });

  it('calls a flat low loss converged rather than stalled', () => {
    const f = facts({ history: hist([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]) });
    const out = diagnose(f, null, CORPUS);
    expect(out.some((x) => x.id === 'converged')).toBe(true);
    expect(out.some((x) => x.id === 'stalled')).toBe(false);
  });

  it('detects overfitting from the train and validation gap', () => {
    const h = hist([1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35]).map((m) => ({ ...m, valLoss: m.loss + 1.2 }));
    const out = diagnose(facts({ history: h }), null, CORPUS);
    const of = out.find((x) => x.id === 'overfit');
    expect(of).toBeDefined();
    expect(of!.fix).toContain('more text');
  });

  it('does not cry overfitting when the gap is small', () => {
    const h = hist([1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35]).map((m) => ({ ...m, valLoss: m.loss + 0.02 }));
    const out = diagnose(facts({ history: h }), null, CORPUS);
    expect(out.some((x) => x.id === 'overfit')).toBe(false);
    expect(out.some((x) => x.id === 'generalising')).toBe(true);
  });

  it('flags a model that has seen far too few tokens for its size', () => {
    const out = diagnose(facts({ paramCount: 5_000_000 }), null, CORPUS);
    const u = out.find((x) => x.id === 'undertrained');
    expect(u).toBeDefined();
    expect(u!.detail).toContain('twenty tokens per parameter');
  });

  it('flags a corpus too small for the context window', () => {
    const out = diagnose(facts({ trainTokens: 500, blockSize: 32 }), null, CORPUS);
    expect(out.some((x) => x.id === 'tinycorpus')).toBe(true);
  });

  it('explains a looping sample as partly a sampling problem', () => {
    const out = diagnose(facts(), 'the cat the cat the cat the cat the cat the cat', CORPUS);
    const l = out.find((x) => x.id === 'looping');
    expect(l).toBeDefined();
    expect(l!.fix).toContain('temperature');
  });

  it('explains gibberish as the character stage rather than a fault', () => {
    const out = diagnose(facts(), 'thq brx zlmp ov dre gxrden aa bb', CORPUS);
    const n = out.find((x) => x.id === 'nonwords');
    expect(n).toBeDefined();
    expect(n!.severity).toBe('info');
    expect(n!.fix).toContain('normal early stage');
  });

  it('scores the sample against the real corpus, not an empty one', () => {
    // Regression: diagnose() once analysed the sample against an empty string,
    // so every sample scored 0% real words no matter how good it was.
    const good = 'the quiet fox sleeps in the garden at dawn';
    const out = diagnose(facts(), good, CORPUS);
    expect(out.some((x) => x.id === 'nonwords')).toBe(false);
    const ok = out.find((x) => x.id === 'words-ok');
    expect(ok).toBeDefined();
    expect(ok!.detail).toContain('100%');
  });

  it('agrees with the standalone sample analysis on the same text', () => {
    const text = 'the fox waits by the river';
    const direct = analyseSample(text, CORPUS);
    const out = diagnose(facts(), text, CORPUS);
    const pct = `${(direct.realWordRatio * 100).toFixed(0)}%`;
    const worded = out.find((x) => x.id === 'words-ok' || x.id === 'nonwords');
    expect(worded).toBeDefined();
    expect(worded!.detail).toContain(pct);
  });

  it('flags spiking gradients', () => {
    const h = hist([3, 3, 3, 3, 3, 3, 3, 3]).map((m, i) => ({ ...m, gradNorm: i % 2 === 0 ? 0.01 : 9 }));
    expect(diagnose(facts({ history: h }), null, CORPUS).some((x) => x.id === 'spiky')).toBe(true);
  });

  it('every finding has a non-empty title and detail', () => {
    const out = diagnose(facts(), 'the fox the fox the fox the fox', CORPUS);
    for (const f of out) {
      expect(f.title.length).toBeGreaterThan(5);
      expect(f.detail.length).toBeGreaterThan(20);
      expect(['good', 'info', 'warn', 'bad']).toContain(f.severity);
    }
  });
});

describe('run summary', () => {
  it('produces a stage, a bounded progress figure and a headline', () => {
    const s = summariseRun(facts(), 'the fox sleeps in the garden', CORPUS);
    expect(s.progress).toBeGreaterThanOrEqual(0);
    expect(s.progress).toBeLessThanOrEqual(1);
    expect(s.headline).toContain('%');
    expect(STAGES[s.stage]).toBeDefined();
  });

  it('never reports negative progress even when loss exceeds uniform', () => {
    const s = summariseRun(facts({ history: hist([99]) }), null, CORPUS);
    expect(s.progress).toBe(0);
  });
});

describe('compute helpers', () => {
  it('cleans up an ANGLE renderer string into a card name', () => {
    const raw = 'ANGLE (NVIDIA, NVIDIA GeForce RTX 5070 Ti Laptop GPU (0x00002F58) Direct3D11 vs_5_0 ps_5_0, D3D11)';
    expect(prettyRenderer(raw)).toBe('NVIDIA GeForce RTX 5070 Ti Laptop GPU');
  });

  it('passes through a plain renderer string', () => {
    expect(prettyRenderer('Apple M2 Pro')).toBe('Apple M2 Pro');
    expect(prettyRenderer(null)).toBeNull();
  });

  it('finds the first size at which the GPU wins', () => {
    const pts: BenchPoint[] = [
      { n: 32, cpuMs: 0.1, cpuGflops: 0.5, gpuMs: 3.5, gpuGflops: 0.02, gpuWins: false },
      { n: 128, cpuMs: 3.3, cpuGflops: 1.3, gpuMs: 4.5, gpuGflops: 0.9, gpuWins: false },
      { n: 256, cpuMs: 29, cpuGflops: 1.2, gpuMs: 3.6, gpuGflops: 9.3, gpuWins: true },
    ];
    expect(crossoverSize(pts)).toBe(256);
  });

  it('reports no crossover when the GPU never wins', () => {
    const pts: BenchPoint[] = [{ n: 32, cpuMs: 0.1, cpuGflops: 0.5, gpuMs: 3.5, gpuGflops: 0.02, gpuWins: false }];
    expect(crossoverSize(pts)).toBeNull();
  });

  it('benchmark sizes are ascending and span the interesting range', () => {
    expect(BENCH_SIZES[0]).toBeLessThan(64);
    expect(BENCH_SIZES[BENCH_SIZES.length - 1]).toBeGreaterThanOrEqual(1024);
    for (let i = 1; i < BENCH_SIZES.length; i++) expect(BENCH_SIZES[i]).toBeGreaterThan(BENCH_SIZES[i - 1]);
  });
});
