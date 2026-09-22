import { describe, expect, it } from 'vitest';
import { KNOWN_MODELS, pipelineStages, summarise, type ArchConfig } from './shapes';
import { activationBytes, checkBudget, compareToGpu, memoryFor, paramCountFor, suggestedTokens } from './resources';
import { project, DEFAULT_CLUSTER, GPU_PROFILES, fmtDuration, fmtUSD, fmtCount } from '../sim/cluster';
import { Transformer, type TransformerConfig } from './transformer';

const ARCH: ArchConfig = { vocab: 100, dModel: 32, nHeads: 4, nLayers: 2, dFF: 64, ctx: 16 };

describe('pipeline shapes', () => {
  const stages = pipelineStages(ARCH, 8);

  it('every matmul has matching inner dimensions', () => {
    for (const s of stages) {
      if (!s.b) continue;
      expect(s.a[1], `${s.id}: ${s.a} x ${s.b}`).toBe(s.b[0]);
      expect(s.out[0]).toBe(s.a[0]);
      expect(s.out[1]).toBe(s.b[1]);
    }
  });

  it('the chain starts at token ids and ends at a distribution over the vocabulary', () => {
    expect(stages[0].a).toEqual([8, 1]);
    const last = stages[stages.length - 1];
    expect(last.out).toEqual([8, ARCH.vocab]);
  });

  it('each stage output feeds a compatible next stage within its group', () => {
    const embed = stages.find((s) => s.id === 'embed')!;
    expect(embed.out).toEqual([8, ARCH.dModel]);
    const head = stages.find((s) => s.id === 'head')!;
    expect(head.a).toEqual([8, ARCH.dModel]);
  });

  it('attention cost grows with the square of sequence length', () => {
    const at8 = pipelineStages(ARCH, 8).find((s) => s.id === 'scores')!.flops;
    const at16 = pipelineStages(ARCH, 16).find((s) => s.id === 'scores')!.flops;
    expect(at16 / at8).toBeCloseTo(4, 6);
  });

  it('feed-forward cost grows linearly with sequence length', () => {
    const at8 = pipelineStages(ARCH, 8).find((s) => s.id === 'ff1')!.flops;
    const at16 = pipelineStages(ARCH, 16).find((s) => s.id === 'ff1')!.flops;
    expect(at16 / at8).toBeCloseTo(2, 6);
  });
});

describe('parameter accounting', () => {
  it('matches the real transformer implementation exactly', () => {
    // The shapes module and the actual model must never disagree.
    const cfg: TransformerConfig = { vocab: 100, dModel: 32, nHeads: 4, nLayers: 2, blockSize: 16, dFF: 64 };
    const model = new Transformer(cfg, 1);
    expect(paramCountFor(cfg)).toBe(model.paramCount);
  });

  it('holds across a range of shapes', () => {
    for (const [v, d, h, l, ff, ctx] of [
      [40, 16, 2, 1, 32, 8],
      [211, 48, 4, 3, 96, 24],
      [77, 64, 8, 2, 256, 32],
    ] as const) {
      const cfg: TransformerConfig = { vocab: v, dModel: d, nHeads: h, nLayers: l, blockSize: ctx, dFF: ff };
      expect(paramCountFor(cfg), `${v}/${d}/${h}/${l}`).toBe(new Transformer(cfg, 2).paramCount);
    }
  });

  it('published architectures land near their reported parameter counts', () => {
    const expected: Record<string, [number, number]> = {
      // id -> plausible range in billions, wide enough for tied embeddings and
      // gated feed-forward variants that this simplified model does not encode.
      'gpt2-small': [0.1, 0.2],
      'gpt2-medium': [0.3, 0.45],
      'gpt2-xl': [1.3, 1.9],
      'llama3-8b': [6, 10],
      'llama3-70b': [55, 85],
    };
    for (const m of KNOWN_MODELS) {
      const s = summarise(m.cfg, 128, m.tied);
      const [lo, hi] = expected[m.id];
      expect(s.params / 1e9, `${m.label} = ${fmtCount(s.params)}`).toBeGreaterThan(lo);
      expect(s.params / 1e9, `${m.label} = ${fmtCount(s.params)}`).toBeLessThan(hi);
    }
  });

  it('parameter groups sum to the total', () => {
    const s = summarise(ARCH, 8, false);
    expect(s.embedParams + s.attnParams + s.ffParams + s.headParams).toBe(s.params);
  });
});

describe('memory accounting', () => {
  const cfg: TransformerConfig = { vocab: 200, dModel: 48, nHeads: 4, nLayers: 2, blockSize: 24, dFF: 96 };

  it('adds up to the reported total', () => {
    const m = memoryFor(cfg, 8, 100000);
    const sum = m.weightsMB + m.gradsMB + m.optimiserMB + m.activationsMB + m.tokenizerMB;
    expect(m.totalMB).toBeCloseTo(sum, 9);
  });

  it('Adam costs twice the weights in optimizer state', () => {
    const m = memoryFor(cfg, 8);
    expect(m.optimiserMB).toBeCloseTo(2 * m.weightsMB, 9);
  });

  it('activation memory grows with the square of the context window', () => {
    const short = activationBytes({ ...cfg, blockSize: 16 }, 16);
    const long = activationBytes({ ...cfg, blockSize: 32 }, 32);
    // Quadratic attention term plus linear terms, so strictly between 2x and 4x.
    expect(long / short).toBeGreaterThan(2);
    expect(long / short).toBeLessThan(4);
  });

  it('a budget that is exceeded is reported as over, not silently allowed', () => {
    const m = memoryFor(cfg, 8);
    const machine = { deviceMemoryGB: 8, logicalCores: 8, jsHeapLimitMB: 4000, jsHeapUsedMB: 100, platform: 'Windows' };
    expect(checkBudget(m, 1, machine).verdict).toBe('over');
    expect(checkBudget(m, 100000, machine).verdict).toBe('ok');
  });

  it('the browser heap ceiling overrides a generous user budget', () => {
    const big = memoryFor({ ...cfg, dModel: 512, dFF: 2048, nLayers: 8, vocab: 50000 }, 8);
    const machine = { deviceMemoryGB: 32, logicalCores: 16, jsHeapLimitMB: 200, jsHeapUsedMB: 50, platform: 'Windows' };
    const check = checkBudget(big, 999999, machine);
    expect(check.verdict).toBe('over');
    expect(check.message).toContain('this tab can only address');
  });
});

describe('cluster projection', () => {
  it('doubling the GPU count roughly halves a compute-bound run', () => {
    const base = { ...DEFAULT_CLUSTER, parallel: 'ddp' as const, nodes: 1, gpusPerNode: 1 };
    const a = project(base);
    const b = project({ ...base, gpusPerNode: 2 });
    expect(b.wallSeconds).toBeLessThan(a.wallSeconds);
    expect(a.wallSeconds / b.wallSeconds).toBeGreaterThan(1.4);
  });

  it('total compute follows 6ND exactly', () => {
    const p = project(DEFAULT_CLUSTER);
    expect(p.totalFlops).toBeCloseTo(6 * p.params * p.totalTokens, -10);
  });

  it('a model too large for the device is flagged rather than projected happily', () => {
    const huge = project({ ...DEFAULT_CLUSTER, paramsB: 500, parallel: 'ddp', nodes: 1, gpusPerNode: 1 });
    expect(huge.memFits).toBe(false);
  });

  it('sharding reduces memory per device', () => {
    const ddp = project({ ...DEFAULT_CLUSTER, parallel: 'ddp' });
    const fsdp = project({ ...DEFAULT_CLUSTER, parallel: 'fsdp' });
    expect(fsdp.memPerGpuGB).toBeLessThan(ddp.memPerGpuGB);
  });

  it('lower precision reduces memory and increases throughput', () => {
    const fp32 = project({ ...DEFAULT_CLUSTER, precision: 'fp32' });
    const bf16 = project({ ...DEFAULT_CLUSTER, precision: 'bf16' });
    expect(bf16.memPerGpuGB).toBeLessThan(fp32.memPerGpuGB);
    expect(bf16.tokensPerSec).toBeGreaterThan(fp32.tokensPerSec);
  });

  it('energy in MWh is megawatts times hours, with no stray conversion', () => {
    // Regression: this was multiplied by an extra 1000, overstating the energy
    // of every projected run by three orders of magnitude.
    const p = project(DEFAULT_CLUSTER);
    expect(p.mwhTotal).toBeCloseTo(p.megawatts * (p.wallSeconds / 3600), 6);
  });

  it('power draw matches the hardware it claims to be using', () => {
    const spec = { ...DEFAULT_CLUSTER, nodes: 8, gpusPerNode: 8 };
    const p = project(spec);
    // 64 GPUs at 700W, with a 1.35x allowance for cooling and hosts.
    expect(p.megawatts).toBeCloseTo((64 * spec.gpu.tdpWatts * 1.35) / 1e6, 9);
    expect(p.megawatts).toBeLessThan(0.2);
  });

  it('a frontier-sized fleet lands in a physically sensible power range', () => {
    const p = project({ ...DEFAULT_CLUSTER, nodes: 2048, gpusPerNode: 8, paramsB: 3000, tokensB: 60000 });
    expect(p.worldSize).toBe(16384);
    // Tens of megawatts, which is a large datacenter, not a small country.
    expect(p.megawatts).toBeGreaterThan(5);
    expect(p.megawatts).toBeLessThan(100);
  });

  it('cost is positive and scales with GPU-hours', () => {
    const p = project(DEFAULT_CLUSTER);
    expect(p.usdCost).toBeGreaterThan(0);
    expect(p.usdCost).toBeCloseTo(p.gpuHours * DEFAULT_CLUSTER.gpu.usdPerHour, 4);
  });
});

describe('local versus GPU comparison', () => {
  it('reports the local machine as a small fraction of one accelerator', () => {
    const c = compareToGpu(200, 56000, 1e6, 989, 0.42, 64);
    expect(c.fractionOfOneGpu).toBeGreaterThan(0);
    expect(c.fractionOfOneGpu).toBeLessThan(1);
    expect(c.secondsLocal).toBeGreaterThan(c.secondsOneGpu);
    expect(c.secondsOneGpu).toBeGreaterThan(c.secondsCluster);
  });

  it('the cluster is exactly world-size faster than one device', () => {
    const c = compareToGpu(200, 1e6, 1e9, 989, 0.4, 16);
    expect(c.secondsOneGpu / c.secondsCluster).toBeCloseTo(16, 6);
  });

  it('suggested tokens follows the twenty-per-parameter guidance', () => {
    expect(suggestedTokens(1e9)).toBe(2e10);
  });
});

describe('formatting never produces a misleading zero', () => {
  it('sub-second durations are shown in milliseconds, not as 0.0 s', () => {
    expect(fmtDuration(0.4)).toBe('400 ms');
    expect(fmtDuration(0.0004)).toBe('under a millisecond');
    expect(fmtDuration(0)).toBe('0 s');
  });

  it('tiny costs are not shown as $0.00', () => {
    expect(fmtUSD(0.004)).toBe('under a cent');
    expect(fmtUSD(0)).toBe('$0.00');
    expect(fmtUSD(1500)).toBe('$1.5K');
  });

  it('large durations stay readable', () => {
    expect(fmtDuration(86400 * 3)).toContain('days');
    expect(fmtDuration(86400 * 800)).toContain('years');
  });
});

describe('GPU profiles are internally consistent', () => {
  it('every profile has sane published values', () => {
    for (const g of GPU_PROFILES) {
      expect(g.tflops).toBeGreaterThan(0);
      expect(g.memGB).toBeGreaterThan(0);
      expect(g.tdpWatts).toBeGreaterThan(0);
      expect(g.usdPerHour).toBeGreaterThan(0);
      expect(g.linkGBs).toBeGreaterThan(0);
    }
  });

  it('faster chips are not also cheaper per hour', () => {
    const sorted = [...GPU_PROFILES].sort((a, b) => a.tflops - b.tflops);
    for (let i = 1; i < sorted.length; i++) {
      if (sorted[i].tflops > sorted[i - 1].tflops * 1.5) {
        expect(sorted[i].usdPerHour).toBeGreaterThanOrEqual(sorted[i - 1].usdPerHour);
      }
    }
  });
});
