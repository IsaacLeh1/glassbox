import type { GpuProfile } from '../sim/cluster';
import type { TransformerConfig } from './transformer';
import { cacheBytes } from './kvcache';

/**
 * What it costs to keep a model answering questions.
 *
 * Training is a one-off bill. Serving is the bill that never stops, and for
 * most deployed models it is the larger of the two within months. The
 * arithmetic is unglamorous and almost entirely about two things: how many
 * bytes have to be read to produce one token, and how many requests are
 * waiting while that happens.
 *
 * Everything in this file is arithmetic over measured or published numbers.
 * Where a figure comes from a vendor datasheet rather than from a measurement
 * on this machine, the panel that uses it says so.
 */

/* ------------------------------------------------------- memory bound -- */

/**
 * Decoding one token reads every weight exactly once.
 *
 * This is the single most useful fact about serving. A forward pass for one
 * token does about two floating-point operations per parameter, which is
 * nothing, but it has to fetch every parameter from memory to do it. So the
 * speed is set by memory bandwidth, not by arithmetic, and a card with twice
 * the bandwidth produces tokens roughly twice as fast regardless of how many
 * teraflops it claims.
 */
export interface DecodeBound {
  /** Bytes read per generated token: the weights, plus the cache. */
  bytesPerToken: number;
  /** Tokens per second that bandwidth alone allows. */
  tokensPerSecond: number;
  /** Seconds of arithmetic, for comparison. Almost always far smaller. */
  computeSeconds: number;
  memorySeconds: number;
  /** True when memory is the limit, which for single-stream decode it is. */
  memoryBound: boolean;
}

export function decodeBound(
  paramCount: number,
  bytesPerWeight: number,
  contextLen: number,
  cfg: TransformerConfig,
  gpu: GpuProfile,
): DecodeBound {
  const weightBytes = paramCount * bytesPerWeight;
  const kv = cacheBytes(cfg, contextLen, bytesPerWeight);
  const bytesPerToken = weightBytes + kv;

  // memBandwidth is in TB/s on these profiles.
  const memorySeconds = bytesPerToken / (gpu.memBandwidth * 1e12);
  // Two operations per parameter for one token, against the card's peak.
  const computeSeconds = (2 * paramCount) / (gpu.tflops * 1e12);

  return {
    bytesPerToken,
    tokensPerSecond: memorySeconds > 0 ? 1 / Math.max(memorySeconds, computeSeconds) : 0,
    computeSeconds,
    memorySeconds,
    memoryBound: memorySeconds > computeSeconds,
  };
}

/* -------------------------------------------------------- what it costs */

export interface CostModel {
  /** Requests the card can serve per second at this shape. */
  requestsPerSecond: number;
  usdPerHour: number;
  usdPerRequest: number;
  usdPerMillionTokens: number;
  /** How many requests fit in memory at once, before the weights are counted. */
  maxConcurrent: number;
}

export function costOf(
  tokensPerSecond: number,
  tokensPerRequest: number,
  gpu: GpuProfile,
  paramCount: number,
  bytesPerWeight: number,
  contextLen: number,
  cfg: TransformerConfig,
): CostModel {
  const rps = tokensPerRequest > 0 ? tokensPerSecond / tokensPerRequest : 0;
  const perSecond = gpu.usdPerHour / 3600;
  const weightGB = (paramCount * bytesPerWeight) / 1e9;
  const kvGB = cacheBytes(cfg, contextLen, bytesPerWeight) / 1e9;
  // Whatever memory the weights do not occupy is available for caches, and
  // that division is what sets how many conversations a card can hold.
  const freeGB = Math.max(0, gpu.memGB - weightGB);

  return {
    requestsPerSecond: rps,
    usdPerHour: gpu.usdPerHour,
    usdPerRequest: rps > 0 ? perSecond / rps : Infinity,
    usdPerMillionTokens: tokensPerSecond > 0 ? (perSecond / tokensPerSecond) * 1e6 : Infinity,
    maxConcurrent: kvGB > 0 ? Math.floor(freeGB / kvGB) : Infinity,
  };
}

/** Published per-million-token prices, for a sense of scale. */
export const MARKET_PRICES: { label: string; usdPerMTokOut: number }[] = [
  { label: 'a frontier model, output', usdPerMTokOut: 15 },
  { label: 'a mid-tier model, output', usdPerMTokOut: 3 },
  { label: 'a small hosted model, output', usdPerMTokOut: 0.6 },
];

/* ------------------------------------------------------------- queueing */

/**
 * What happens when requests arrive faster than they can be served.
 *
 * The result is not a gentle slowdown. Queueing has a cliff in it: at 50
 * percent utilisation the wait is about the same as the service time, at 90
 * percent it is ten times worse, and at 99 percent it is a hundred times
 * worse. Capacity planning is mostly about staying off the end of that curve,
 * and it is why a service that looks fine in testing falls over at launch.
 *
 * The model here is M/M/1: one server, arrivals at random, exponential
 * service times. Real serving is more complicated, but the shape of the cliff
 * is the same and the arithmetic is exact for the model stated.
 */
export interface Queue {
  /** Arrivals per second. */
  lambda: number;
  /** Completions per second the server is capable of. */
  mu: number;
  /** Fraction of the time the server is busy. */
  utilisation: number;
  /** True once arrivals outpace service and the queue grows without bound. */
  saturated: boolean;
  /** Mean seconds from arrival to completion, including waiting. */
  meanSeconds: number;
  /** The slow tail: 99 percent of requests finish faster than this. */
  p99Seconds: number;
  /** Requests waiting, on average. */
  inQueue: number;
}

export function queue(lambda: number, mu: number): Queue {
  const utilisation = mu > 0 ? lambda / mu : Infinity;
  if (!(mu > lambda)) {
    return {
      lambda,
      mu,
      utilisation,
      saturated: true,
      meanSeconds: Infinity,
      p99Seconds: Infinity,
      inQueue: Infinity,
    };
  }
  const spare = mu - lambda;
  return {
    lambda,
    mu,
    utilisation,
    saturated: false,
    // Response time is exponentially distributed with rate (mu - lambda).
    meanSeconds: 1 / spare,
    p99Seconds: Math.log(100) / spare,
    inQueue: (utilisation * utilisation) / (1 - utilisation),
  };
}

/**
 * How many replicas are needed to hold a p99 target at a given load.
 *
 * Assumes the load is split evenly, which is what a load balancer is for.
 */
export function replicasFor(lambda: number, mu: number, p99Target: number): number {
  if (mu <= 0 || p99Target <= 0) return Infinity;
  const floor = Math.log(100) / mu;
  // Even with no queue at all, one request still takes ln(100)/mu at the 99th
  // percentile. If the promise is tighter than that, no number of replicas
  // helps: splitting the load shortens the queue, never the work. Returning a
  // huge finite number here instead would read as an answer rather than as
  // the impossibility it is.
  if (p99Target <= floor) return Infinity;
  // Each replica sees lambda/n and must satisfy ln(100)/(mu - lambda/n) <= target.
  const needed = lambda / (mu - Math.log(100) / p99Target);
  if (needed <= 0) return 1;
  return Math.max(1, Math.ceil(needed));
}

/** The utilisation-versus-latency curve, for drawing the cliff. */
export function loadCurve(mu: number, points = 40): { x: number; y: number }[] {
  const out: { x: number; y: number }[] = [];
  for (let i = 1; i <= points; i++) {
    const u = (i / (points + 1)) * 0.99;
    const q = queue(u * mu, mu);
    if (Number.isFinite(q.p99Seconds)) out.push({ x: u, y: q.p99Seconds });
  }
  return out;
}

/* -------------------------------------------------------------- batching */

/**
 * Running several requests through the weights at once.
 *
 * Since decoding is limited by reading the weights rather than by arithmetic,
 * and the same weights serve every request in a batch, a batch of eight reads
 * the weights once instead of eight times. Throughput rises almost in
 * proportion. What it costs is the time each individual request spends
 * waiting for the batch to fill, which is why an interactive product batches
 * far less aggressively than a bulk pipeline.
 */
export interface BatchPoint {
  size: number;
  tokensPerSecond: number;
  perRequestMs: number;
  cacheGB: number;
}

export function batchCurve(
  singleTokensPerSecond: number,
  maxBatch: number,
  cfg: TransformerConfig,
  contextLen: number,
  bytesPerWeight: number,
  weightBytes: number,
): BatchPoint[] {
  const kv = cacheBytes(cfg, contextLen, bytesPerWeight);
  const out: BatchPoint[] = [];
  for (let b = 1; b <= maxBatch; b *= 2) {
    // Weights are read once for the whole batch; each request still needs its
    // own cache read. That ratio is what limits the gain.
    const bytesOne = weightBytes + kv;
    const bytesBatch = weightBytes + kv * b;
    const speedup = (bytesOne * b) / bytesBatch;
    const tps = singleTokensPerSecond * speedup;
    out.push({
      size: b,
      tokensPerSecond: tps,
      perRequestMs: tps > 0 ? (1000 * b) / tps : Infinity,
      cacheGB: (kv * b) / 1e9,
    });
  }
  return out;
}

/* ------------------------------------------------- something to cost -- */

/**
 * Shapes to run the cost arithmetic against.
 *
 * A model small enough to train in a browser tab costs nothing to serve, so
 * every figure rounds to zero and the page teaches nothing. The arithmetic is
 * identical at any size -- it is driven entirely by parameter count and cache
 * shape -- so the honest fix is to apply the same formula to a model whose
 * numbers mean something, clearly labelled as somebody else's model.
 */
export interface ServeTarget {
  id: string;
  label: string;
  params: number;
  /** Enough of the architecture to size the key-value cache. */
  dModel: number;
  nHeads: number;
  nLayers: number;
  note: string;
}

export const SERVE_TARGETS: ServeTarget[] = [
  {
    id: 'gpt2',
    label: 'GPT-2 Small (124M)',
    params: 124e6,
    dModel: 768,
    nHeads: 12,
    nLayers: 12,
    note: 'Small enough to serve from a single consumer card, and still useful for narrow tasks.',
  },
  {
    id: 'llama8b',
    label: 'Llama 3 8B',
    params: 8.03e9,
    dModel: 4096,
    nHeads: 32,
    nLayers: 32,
    note: 'The size most self-hosted deployments actually run. Fits one card once quantised.',
  },
  {
    id: 'llama70b',
    label: 'Llama 3 70B',
    params: 70.6e9,
    dModel: 8192,
    nHeads: 64,
    nLayers: 80,
    note: 'Needs several cards at full precision, one or two once quantised. The usual quality tier.',
  },
  {
    id: 'llama405b',
    label: 'Llama 3 405B',
    params: 405e9,
    dModel: 16384,
    nHeads: 128,
    nLayers: 126,
    note: 'A frontier-scale open model. Serving it is a multi-card engineering problem in its own right.',
  },
];

/** Money at any magnitude, without rounding a real cost away to nothing. */
export function money(n: number): string {
  if (!Number.isFinite(n)) return '—';
  if (n === 0) return '$0';
  if (n < 1e-6) return `$${n.toExponential(1)}`;
  if (n < 0.01) return `$${n.toFixed(6)}`;
  if (n < 1) return `$${n.toFixed(4)}`;
  if (n < 1000) return `$${n.toFixed(2)}`;
  if (n < 1e6) return `$${(n / 1e3).toFixed(1)}K`;
  return `$${(n / 1e6).toFixed(2)}M`;
}
