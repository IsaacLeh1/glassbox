import { mulberry32 } from '../engine/tensor';

/**
 * Datacenter training simulator.
 *
 * Two things are happening in this module and it matters which is which:
 *
 *  1. The PROJECTION maths below is real. Step time, memory footprint, network
 *     volume, wall-clock and dollar cost are computed from published accelerator
 *     specifications and the standard 6N FLOPs-per-token estimate. Change the
 *     model size and the numbers move the way a real capacity plan moves.
 *
 *  2. The per-device telemetry (temperatures, fan-level jitter, log ordering) is
 *     a simulation. It is shaped to behave like a real cluster so the workflow
 *     is recognisable, but no hardware is being read.
 *
 * The loss values that stream through the log come from the actual model
 * training in the browser, not from a script.
 */

export interface GpuProfile {
  id: string;
  label: string;
  /** Dense BF16 matrix throughput, TFLOP/s, vendor published. */
  tflops: number;
  memGB: number;
  /** HBM bandwidth, TB/s. */
  memBandwidth: number;
  tdpWatts: number;
  /** Per-device interconnect bandwidth to peers, GB/s. */
  linkGBs: number;
  /** Approximate on-demand rental rate, USD per GPU-hour. */
  usdPerHour: number;
}

export const GPU_PROFILES: GpuProfile[] = [
  { id: 'b200', label: 'NVIDIA B200', tflops: 2250, memGB: 192, memBandwidth: 8.0, tdpWatts: 1000, linkGBs: 1800, usdPerHour: 6.0 },
  { id: 'h200', label: 'NVIDIA H200', tflops: 989, memGB: 141, memBandwidth: 4.8, tdpWatts: 700, linkGBs: 900, usdPerHour: 4.0 },
  { id: 'h100', label: 'NVIDIA H100 SXM', tflops: 989, memGB: 80, memBandwidth: 3.35, tdpWatts: 700, linkGBs: 900, usdPerHour: 3.0 },
  { id: 'a100', label: 'NVIDIA A100 80GB', tflops: 312, memGB: 80, memBandwidth: 2.0, tdpWatts: 400, linkGBs: 600, usdPerHour: 1.6 },
  { id: 'l40s', label: 'NVIDIA L40S', tflops: 362, memGB: 48, memBandwidth: 0.86, tdpWatts: 350, linkGBs: 64, usdPerHour: 1.1 },
  { id: '4090', label: 'GeForce RTX 4090', tflops: 165, memGB: 24, memBandwidth: 1.0, tdpWatts: 450, linkGBs: 32, usdPerHour: 0.4 },
];

export type Precision = 'fp32' | 'bf16' | 'fp8';

export const PRECISIONS: Record<Precision, { label: string; bytes: number; speedup: number; blurb: string }> = {
  fp32: { label: 'FP32', bytes: 4, speedup: 0.5, blurb: 'Full precision. Safest, slowest, and twice the memory of BF16.' },
  bf16: { label: 'BF16', bytes: 2, speedup: 1, blurb: 'The standard for large training runs: half the memory, same exponent range as FP32.' },
  fp8: { label: 'FP8', bytes: 1, speedup: 1.8, blurb: 'Newest tier. Roughly doubles throughput again, but needs careful loss scaling to stay stable.' },
};

export type ParallelMode = 'ddp' | 'fsdp' | 'tp_pp';

export const PARALLEL_INFO: Record<ParallelMode, { label: string; blurb: string }> = {
  ddp: {
    label: 'Data parallel (DDP)',
    blurb:
      'Every GPU holds a full copy of the model and processes a different slice of the batch. Simple and fast, but the whole model plus optimizer state must fit on one device.',
  },
  fsdp: {
    label: 'Fully sharded (FSDP / ZeRO-3)',
    blurb:
      'Parameters, gradients and optimizer state are split across all GPUs and gathered just in time. Memory per device drops roughly by the world size, at the cost of extra communication.',
  },
  tp_pp: {
    label: 'Tensor + pipeline parallel',
    blurb:
      'Individual matrices are split across GPUs and the layer stack is cut into pipeline stages. Required once a single layer stops fitting on one device.',
  },
};

export interface ClusterSpec {
  gpu: GpuProfile;
  nodes: number;
  gpusPerNode: number;
  precision: Precision;
  parallel: ParallelMode;
  /** Model FLOPs utilization: the fraction of peak actually achieved. */
  mfu: number;
  paramsB: number;
  tokensB: number;
  globalBatchTokens: number;
}

export const DEFAULT_CLUSTER: ClusterSpec = {
  gpu: GPU_PROFILES[2],
  nodes: 8,
  gpusPerNode: 8,
  precision: 'bf16',
  parallel: 'fsdp',
  mfu: 0.42,
  paramsB: 7,
  tokensB: 300,
  globalBatchTokens: 4_194_304,
};

export interface Projection {
  worldSize: number;
  params: number;
  totalTokens: number;
  /** 6 FLOPs per parameter per token: 2 forward, 4 backward. */
  flopsPerToken: number;
  totalFlops: number;
  clusterFlopsPerSec: number;
  stepTokens: number;
  stepFlops: number;
  computeMsPerStep: number;
  commsMsPerStep: number;
  msPerStep: number;
  totalSteps: number;
  wallSeconds: number;
  tokensPerSec: number;
  bytesPerParam: number;
  weightsGB: number;
  gradsGB: number;
  optimStateGB: number;
  activationsGB: number;
  memPerGpuGB: number;
  memFits: boolean;
  allReduceGB: number;
  checkpointGB: number;
  gpuHours: number;
  usdCost: number;
  megawatts: number;
  mwhTotal: number;
}

const GB = 1024 ** 3;

/**
 * Capacity plan for a training run. Every line here is arithmetic over the
 * published specs above, not a lookup table.
 */
export function project(spec: ClusterSpec): Projection {
  const worldSize = Math.max(1, spec.nodes * spec.gpusPerNode);
  const params = spec.paramsB * 1e9;
  const totalTokens = spec.tokensB * 1e9;
  const prec = PRECISIONS[spec.precision];

  // The standard estimate: forward is ~2 FLOPs per parameter per token, and
  // backward costs about twice the forward.
  const flopsPerToken = 6 * params;
  const totalFlops = flopsPerToken * totalTokens;

  const perGpuFlops = spec.gpu.tflops * 1e12 * prec.speedup * spec.mfu;
  const clusterFlopsPerSec = perGpuFlops * worldSize;

  const stepTokens = spec.globalBatchTokens;
  const stepFlops = flopsPerToken * stepTokens;
  const computeMsPerStep = (stepFlops / clusterFlopsPerSec) * 1000;

  // Ring all-reduce moves 2(n-1)/n times the gradient payload per device.
  const gradBytes = params * prec.bytes;
  const ringFactor = worldSize > 1 ? (2 * (worldSize - 1)) / worldSize : 0;
  const allReduceBytes = gradBytes * ringFactor;
  // FSDP also gathers parameters on the way forward and back.
  const commMultiplier = spec.parallel === 'fsdp' ? 2.5 : spec.parallel === 'tp_pp' ? 1.7 : 1;
  const linkBytesPerSec = spec.gpu.linkGBs * 1e9;
  const commsMsPerStep =
    worldSize > 1 ? ((allReduceBytes * commMultiplier) / linkBytesPerSec) * 1000 : 0;

  // Compute and communication overlap heavily in a tuned run, but never fully.
  const msPerStep = Math.max(computeMsPerStep, commsMsPerStep) + 0.25 * Math.min(computeMsPerStep, commsMsPerStep);

  const totalSteps = Math.max(1, Math.round(totalTokens / stepTokens));
  const wallSeconds = (msPerStep / 1000) * totalSteps;
  const tokensPerSec = stepTokens / (msPerStep / 1000);

  // Memory: weights + gradients + Adam keeps two extra states per parameter,
  // and mixed precision keeps an FP32 master copy of the weights.
  const shard = spec.parallel === 'ddp' ? 1 : worldSize;
  const weightsGB = (params * prec.bytes) / GB;
  const gradsGB = (params * prec.bytes) / GB / shard;
  const optimStateGB = (params * 4 * 3) / GB / shard;
  const weightsShardedGB = spec.parallel === 'ddp' ? weightsGB : weightsGB / shard;
  // Activation memory scales with batch per device; this is a working estimate.
  const tokensPerGpu = stepTokens / worldSize;
  const activationsGB = (tokensPerGpu * Math.sqrt(params) * prec.bytes * 0.02) / GB;

  const memPerGpuGB = weightsShardedGB + gradsGB + optimStateGB + activationsGB;
  const memFits = memPerGpuGB < spec.gpu.memGB * 0.92;

  const checkpointGB = (params * 4 * 4) / GB; // weights + 3 optimizer slots, FP32
  const gpuHours = (wallSeconds / 3600) * worldSize;
  const usdCost = gpuHours * spec.gpu.usdPerHour;
  const megawatts = (worldSize * spec.gpu.tdpWatts * 1.35) / 1e6; // 1.35x for cooling and host
  // Megawatts multiplied by hours is already megawatt-hours. There is no
  // further conversion; an earlier factor of 1000 here overstated the energy
  // of every run by three orders of magnitude.
  const mwhTotal = megawatts * (wallSeconds / 3600);

  return {
    worldSize,
    params,
    totalTokens,
    flopsPerToken,
    totalFlops,
    clusterFlopsPerSec,
    stepTokens,
    stepFlops,
    computeMsPerStep,
    commsMsPerStep,
    msPerStep,
    totalSteps,
    wallSeconds,
    tokensPerSec,
    bytesPerParam: prec.bytes,
    weightsGB: weightsShardedGB,
    gradsGB,
    optimStateGB,
    activationsGB,
    memPerGpuGB,
    memFits,
    allReduceGB: allReduceBytes / GB,
    checkpointGB,
    gpuHours,
    usdCost,
    megawatts,
    mwhTotal,
  };
}

/* ------------------------------------------------------- device telemetry -- */

export type DeviceState = 'idle' | 'compute' | 'allreduce' | 'checkpoint' | 'failed' | 'recovering';

export interface Device {
  id: number;
  node: number;
  rank: number;
  state: DeviceState;
  util: number;
  memUsedGB: number;
  tempC: number;
  watts: number;
  stepsDone: number;
}

export type LogLevel = 'info' | 'warn' | 'error' | 'metric';

export interface LogLine {
  t: number;
  ts: string;
  rank: number | null;
  level: LogLevel;
  msg: string;
}

export interface ClusterEvents {
  onLog?: (line: LogLine) => void;
}

/**
 * Animates a fleet of devices through the phases of a training step and emits
 * the log traffic a real run produces. Loss values are injected from the live
 * in-browser trainer rather than invented.
 */
export class ClusterSim {
  spec: ClusterSpec;
  proj: Projection;
  devices: Device[] = [];
  logs: LogLine[] = [];
  step = 0;
  phase: 'forward' | 'backward' | 'allreduce' | 'optimizer' | 'checkpoint' = 'forward';
  phaseProgress = 0;
  running = false;
  elapsedSimSeconds = 0;
  failedRanks = new Set<number>();
  checkpoints: { step: number; sizeGB: number; at: number }[] = [];
  private rand: () => number;
  private maxLogs = 400;
  private events: ClusterEvents;

  constructor(spec: ClusterSpec, events: ClusterEvents = {}, seed = 4) {
    this.spec = spec;
    this.proj = project(spec);
    this.rand = mulberry32(seed);
    this.events = events;
    this.buildDevices();
  }

  buildDevices() {
    this.devices = [];
    let id = 0;
    for (let n = 0; n < this.spec.nodes; n++) {
      for (let g = 0; g < this.spec.gpusPerNode; g++) {
        this.devices.push({
          id: id,
          node: n,
          rank: id,
          state: 'idle',
          util: 0,
          memUsedGB: 0,
          tempC: 34 + this.rand() * 4,
          watts: 70 + this.rand() * 30,
          stepsDone: 0,
        });
        id++;
      }
    }
  }

  reconfigure(spec: ClusterSpec) {
    this.spec = spec;
    this.proj = project(spec);
    this.buildDevices();
  }

  log(level: LogLevel, msg: string, rank: number | null = null) {
    const line: LogLine = {
      t: this.elapsedSimSeconds,
      ts: formatClock(this.elapsedSimSeconds),
      rank,
      level,
      msg,
    };
    this.logs.push(line);
    if (this.logs.length > this.maxLogs) this.logs.splice(0, this.logs.length - this.maxLogs);
    this.events.onLog?.(line);
  }

  start() {
    this.running = true;
    const p = this.proj;
    this.log('info', `initializing ${p.worldSize} ranks across ${this.spec.nodes} nodes`);
    this.log('info', `${this.spec.gpu.label} | ${PRECISIONS[this.spec.precision].label} | ${PARALLEL_INFO[this.spec.parallel].label}`);
    this.log('info', `model ${fmtCount(p.params)} params, ${fmtCount(p.totalTokens)} training tokens`);
    this.log(
      p.memFits ? 'info' : 'error',
      `memory per device ${p.memPerGpuGB.toFixed(1)} GB of ${this.spec.gpu.memGB} GB` +
        (p.memFits ? ' -- fits' : ' -- OUT OF MEMORY, reduce batch or shard further'),
    );
    this.log('info', `projected ${p.msPerStep.toFixed(0)} ms/step, ${p.totalSteps.toLocaleString()} steps, ${fmtDuration(p.wallSeconds)} wall clock`);
    this.log('info', 'NCCL ring established, all ranks reporting healthy');
  }

  /**
   * Advance the simulation. `realDtMs` is wall time since the last frame and
   * `liveLoss` is the current loss from the real model training locally.
   */
  tick(realDtMs: number, liveLoss: number | null, speedup: number) {
    if (!this.running) return;
    const simDt = (realDtMs / 1000) * speedup;
    this.elapsedSimSeconds += simDt;

    const stepSeconds = this.proj.msPerStep / 1000;
    const phaseShare = { forward: 0.22, backward: 0.42, allreduce: 0.26, optimizer: 0.1, checkpoint: 0 };
    this.phaseProgress += simDt / Math.max(1e-6, stepSeconds);

    while (this.phaseProgress >= 1) {
      this.phaseProgress -= 1;
      this.completeStep(liveLoss);
    }

    const q = this.phaseProgress;
    this.phase =
      q < phaseShare.forward
        ? 'forward'
        : q < phaseShare.forward + phaseShare.backward
          ? 'backward'
          : q < phaseShare.forward + phaseShare.backward + phaseShare.allreduce
            ? 'allreduce'
            : 'optimizer';

    for (const d of this.devices) {
      if (d.state === 'failed') {
        d.util = 0;
        d.watts = 40;
        d.tempC += (32 - d.tempC) * 0.05;
        continue;
      }
      const busy = this.phase === 'allreduce' ? 0.35 : 0.97;
      const target = busy * (0.94 + this.rand() * 0.06);
      d.util += (target * 100 - d.util) * 0.25;
      d.state = this.phase === 'allreduce' ? 'allreduce' : 'compute';
      d.memUsedGB = this.proj.memPerGpuGB * (0.9 + 0.1 * (this.phase === 'backward' ? 1 : 0.6));
      const targetTemp = 42 + (d.util / 100) * 32 + (d.node % 3) * 1.5;
      d.tempC += (targetTemp - d.tempC) * 0.04 + (this.rand() - 0.5) * 0.15;
      d.watts = this.spec.gpu.tdpWatts * (0.35 + (d.util / 100) * 0.6);
    }
  }

  private completeStep(liveLoss: number | null) {
    this.step++;
    for (const d of this.devices) if (d.state !== 'failed') d.stepsDone++;

    if (this.step % 10 === 0) {
      const p = this.proj;
      const lossStr = liveLoss !== null && Number.isFinite(liveLoss) ? liveLoss.toFixed(4) : 'n/a';
      this.log(
        'metric',
        `step ${this.step.toLocaleString()}/${p.totalSteps.toLocaleString()} | loss ${lossStr} | ` +
          `${(p.tokensPerSec / 1e3).toFixed(1)}k tok/s | mfu ${(this.spec.mfu * 100).toFixed(0)}% | ` +
          `${p.msPerStep.toFixed(0)}ms`,
      );
    }
    if (this.step % 250 === 0) {
      this.checkpoints.push({ step: this.step, sizeGB: this.proj.checkpointGB, at: this.elapsedSimSeconds });
      this.log('info', `checkpoint written: step-${this.step} (${this.proj.checkpointGB.toFixed(1)} GB across ${this.proj.worldSize} shards)`);
    }
    if (this.step % 137 === 0) {
      this.log('warn', `rank ${Math.floor(this.rand() * this.proj.worldSize)} straggling, ${(8 + this.rand() * 40).toFixed(0)}ms behind the collective`);
    }
  }

  /** Inject a hardware failure, the single most common cause of a stalled run. */
  failRandomDevice() {
    const alive = this.devices.filter((d) => d.state !== 'failed');
    if (alive.length <= 1) return;
    const victim = alive[Math.floor(this.rand() * alive.length)];
    victim.state = 'failed';
    this.failedRanks.add(victim.rank);
    this.log('error', `rank ${victim.rank} (node ${victim.node}) lost: Xid 79, GPU has fallen off the bus`, victim.rank);
    this.log('error', 'NCCL collective timed out after 1800000 ms -- all ranks blocked');
    this.log('warn', 'training halted: a synchronous data-parallel run is only as available as its least available device');
  }

  /** Recover the way a real run does: rebuild the world, reload, re-run lost steps. */
  recoverAll() {
    if (this.failedRanks.size === 0) return;
    const last = this.checkpoints[this.checkpoints.length - 1];
    for (const d of this.devices) {
      if (d.state === 'failed') d.state = 'recovering';
    }
    this.failedRanks.clear();
    this.log('info', 'replacement nodes allocated, rebuilding NCCL communicator');
    if (last) {
      const lost = this.step - last.step;
      this.log('info', `restoring checkpoint step-${last.step} (${last.sizeGB.toFixed(1)} GB)`);
      this.log('warn', `rewinding ${lost.toLocaleString()} steps of progress -- everything since the last checkpoint is gone`);
      this.step = last.step;
    } else {
      this.log('warn', 'no checkpoint exists yet, restarting from step 0');
      this.step = 0;
    }
    for (const d of this.devices) d.state = 'compute';
    this.log('info', 'all ranks healthy, resuming');
  }

  stop() {
    this.running = false;
    for (const d of this.devices) {
      d.state = 'idle';
      d.util = 0;
    }
  }
}

/* ------------------------------------------------------------ formatting -- */

export function formatClock(sec: number): string {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

export function fmtDuration(sec: number): string {
  if (!Number.isFinite(sec)) return 'n/a';
  if (sec <= 0) return '0 s';
  if (sec < 0.001) return 'under a millisecond';
  if (sec < 1) return `${(sec * 1000).toFixed(0)} ms`;
  if (sec < 90) return `${sec.toFixed(1)} s`;
  if (sec < 5400) return `${(sec / 60).toFixed(1)} min`;
  if (sec < 172800) return `${(sec / 3600).toFixed(1)} hours`;
  const days = sec / 86400;
  if (days < 400) return `${days.toFixed(1)} days`;
  return `${(days / 365).toFixed(1)} years`;
}

export function fmtCount(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toFixed(0);
}

export function fmtUSD(n: number): string {
  if (n > 0 && n < 0.01) return 'under a cent';
  if (n >= 1e9) return `$${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `$${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `$${(n / 1e3).toFixed(1)}K`;
  return `$${n.toFixed(2)}`;
}

export function fmtFlops(n: number): string {
  if (n >= 1e21) return `${(n / 1e21).toFixed(2)} ZFLOP`;
  if (n >= 1e18) return `${(n / 1e18).toFixed(2)} EFLOP`;
  if (n >= 1e15) return `${(n / 1e15).toFixed(2)} PFLOP`;
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)} TFLOP`;
  return `${(n / 1e9).toFixed(2)} GFLOP`;
}
