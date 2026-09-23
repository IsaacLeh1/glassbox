import type { TransformerConfig } from './transformer';

/**
 * Machine detection and memory accounting.
 *
 * The point of this module is honesty about what training in a browser tab
 * costs. Glassbox computes in Float64 for numerical clarity, which is eight
 * bytes per number -- four times what a real bf16 training run uses. That is a
 * deliberate trade and the figures below say so rather than hiding it.
 */

export interface MachineInfo {
  /** Browser-reported RAM in GB, coarsely rounded and capped at 8 by the spec. */
  deviceMemoryGB: number | null;
  logicalCores: number | null;
  /** Chrome only: the hard ceiling on this tab's JavaScript heap. */
  jsHeapLimitMB: number | null;
  jsHeapUsedMB: number | null;
  platform: string;
}

interface PerfMemory {
  jsHeapSizeLimit: number;
  usedJSHeapSize: number;
}

export function detectMachine(): MachineInfo {
  const nav = navigator as Navigator & { deviceMemory?: number };
  const perf = performance as Performance & { memory?: PerfMemory };
  return {
    deviceMemoryGB: typeof nav.deviceMemory === 'number' ? nav.deviceMemory : null,
    logicalCores: typeof navigator.hardwareConcurrency === 'number' ? navigator.hardwareConcurrency : null,
    jsHeapLimitMB: perf.memory ? perf.memory.jsHeapSizeLimit / 1048576 : null,
    jsHeapUsedMB: perf.memory ? perf.memory.usedJSHeapSize / 1048576 : null,
    platform: navigator.userAgent.includes('Win')
      ? 'Windows'
      : navigator.userAgent.includes('Mac')
        ? 'macOS'
        : navigator.userAgent.includes('Linux')
          ? 'Linux'
          : 'Unknown',
  };
}

export function readHeapUsedMB(): number | null {
  const perf = performance as Performance & { memory?: PerfMemory };
  return perf.memory ? perf.memory.usedJSHeapSize / 1048576 : null;
}

export const BYTES_PER_NUMBER = 8; // Float64Array

export interface MemoryBreakdown {
  paramCount: number;
  weightsMB: number;
  gradsMB: number;
  optimiserMB: number;
  activationsMB: number;
  tokenizerMB: number;
  totalMB: number;
  /** The same model trained the way a real run would, in bf16 with fp32 states. */
  realWorldBf16MB: number;
}

/** Peak live activation memory for one forward and backward on one sequence. */
export function activationBytes(cfg: TransformerConfig, T: number): number {
  const { dModel, dFF, nLayers, nHeads, vocab } = cfg;
  const perLayer =
    12 * T * dModel + // residual stream copies, ln outputs, Q/K/V, projections
    2 * T * dFF + // feed-forward pre-activation and hidden
    2 * nHeads * T * T; // scores and attention weights per head
  const inputs = 3 * T * dModel;
  const output = T * dModel + T * vocab;
  return (inputs + nLayers * perLayer + output) * BYTES_PER_NUMBER;
}

export function paramCountFor(cfg: TransformerConfig): number {
  const { vocab, dModel, nLayers, dFF, blockSize } = cfg;
  const embed = vocab * dModel + blockSize * dModel;
  const perLayer = 4 * dModel * dModel + dModel + 4 * dModel + 2 * dModel * dFF + dFF + dModel;
  const head = dModel * vocab + vocab;
  return embed + nLayers * perLayer + head + 2 * dModel;
}

export function memoryFor(cfg: TransformerConfig, batchSeqs: number, vocabChars = 0): MemoryBreakdown {
  const p = paramCountFor(cfg);
  const MB = 1048576;
  const weights = (p * BYTES_PER_NUMBER) / MB;
  const grads = weights;
  // Adam keeps a first and second moment per parameter.
  const optimiser = 2 * weights;
  // Only one sequence is live at a time; the rest is accumulated gradient.
  const activations = (activationBytes(cfg, cfg.blockSize) * Math.min(batchSeqs, 1)) / MB;
  const tokenizer = (vocabChars * 2) / MB;

  // For comparison: bf16 weights and grads, fp32 master copy and two fp32
  // optimiser states, which is the standard mixed-precision recipe.
  const realWorld = (p * (2 + 2 + 4 + 4 + 4)) / MB;

  return {
    paramCount: p,
    weightsMB: weights,
    gradsMB: grads,
    optimiserMB: optimiser,
    activationsMB: activations,
    tokenizerMB: tokenizer,
    totalMB: weights + grads + optimiser + activations + tokenizer,
    realWorldBf16MB: realWorld,
  };
}

export type BudgetVerdict = 'ok' | 'tight' | 'over';

export interface BudgetCheck {
  verdict: BudgetVerdict;
  usedMB: number;
  budgetMB: number;
  fraction: number;
  message: string;
}

export function checkBudget(mem: MemoryBreakdown, budgetMB: number, machine: MachineInfo): BudgetCheck {
  const fraction = mem.totalMB / Math.max(1, budgetMB);
  const heapCeiling = machine.jsHeapLimitMB;
  let verdict: BudgetVerdict = fraction > 1 ? 'over' : fraction > 0.75 ? 'tight' : 'ok';
  let message =
    verdict === 'over'
      ? `This configuration needs about ${mem.totalMB.toFixed(0)} MB, which is more than the ${budgetMB} MB you allowed. Reduce width, layers, context or vocabulary before training.`
      : verdict === 'tight'
        ? `About ${mem.totalMB.toFixed(0)} MB of your ${budgetMB} MB budget. It will run, but leave the machine some room.`
        : `About ${mem.totalMB.toFixed(0)} MB of your ${budgetMB} MB budget. Comfortable.`;

  if (heapCeiling !== null && mem.totalMB > heapCeiling * 0.8) {
    verdict = 'over';
    message = `This needs roughly ${mem.totalMB.toFixed(0)} MB, but this tab can only address about ${heapCeiling.toFixed(0)} MB before the browser kills it. Make the model smaller.`;
  }
  return { verdict, usedMB: mem.totalMB, budgetMB, fraction, message };
}

/**
 * Measured local throughput, converted into the terms step 05 uses so the
 * same model can be priced on real accelerators.
 */
export interface ThroughputComparison {
  localTokensPerSec: number;
  localFlopsPerSec: number;
  /** Fraction of a single accelerator this machine is currently achieving. */
  fractionOfOneGpu: number;
  gpuTokensPerSec: number;
  secondsLocal: number;
  secondsOneGpu: number;
  secondsCluster: number;
}

export function compareToGpu(
  localTokensPerSec: number,
  paramCount: number,
  targetTokens: number,
  gpuTflops: number,
  mfu: number,
  worldSize: number,
): ThroughputComparison {
  // The usual estimate: six FLOPs per parameter per token for forward plus backward.
  const flopsPerToken = 6 * paramCount;
  const localFlopsPerSec = localTokensPerSec * flopsPerToken;
  const gpuFlopsPerSec = gpuTflops * 1e12 * mfu;
  const gpuTokensPerSec = gpuFlopsPerSec / flopsPerToken;
  return {
    localTokensPerSec,
    localFlopsPerSec,
    fractionOfOneGpu: gpuFlopsPerSec > 0 ? localFlopsPerSec / gpuFlopsPerSec : 0,
    gpuTokensPerSec,
    secondsLocal: localTokensPerSec > 0 ? targetTokens / localTokensPerSec : Infinity,
    secondsOneGpu: targetTokens / gpuTokensPerSec,
    secondsCluster: targetTokens / (gpuTokensPerSec * Math.max(1, worldSize)),
  };
}

/**
 * Chinchilla-style guidance: roughly twenty training tokens per parameter is
 * where a model of a given size stops being undertrained.
 */
export function suggestedTokens(paramCount: number): number {
  return paramCount * 20;
}
