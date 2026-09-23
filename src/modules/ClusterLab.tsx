import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ClusterSim,
  DEFAULT_CLUSTER,
  GPU_PROFILES,
  PARALLEL_INFO,
  PRECISIONS,
  fmtCount,
  fmtDuration,
  fmtFlops,
  fmtUSD,
  project,
  type ClusterSpec,
  type Device,
  type LogLevel,
  type ParallelMode,
  type Precision,
} from '../sim/cluster';
import { buildLMTrainer } from '../engine/lmPresets';
import type { LMTrainer } from '../engine/lmTrainer';
import {
  Badge,
  BackgroundNotice,
  Btn,
  Callout,
  Depth,
  Eq,
  Field,
  Label,
  M,
  Panel,
  ProgressBar,
  Segmented,
  Select,
  Slider,
  Stat,
  fmt,
  fmtInt,
  Keep,
} from '../ui/kit';
import { V } from '../content/varInfo';
import { BarMeter, LineChart, type Series } from '../ui/viz';
import { useFrameLoop } from '../ui/loop';
import FrontierTab from './FrontierTab';

const LOG_COLOR: Record<LogLevel, string> = {
  info: 'var(--text-2)',
  warn: 'var(--warn)',
  error: 'var(--err)',
  metric: 'var(--accent)',
};

function DeviceCell({ d, size }: { d: Device; size: number }) {
  const color =
    d.state === 'failed'
      ? 'var(--err)'
      : d.state === 'allreduce'
        ? 'var(--pos)'
        : d.state === 'idle'
          ? 'var(--panel-3)'
          : 'var(--ok)';
  return (
    <div
      title={`rank ${d.rank} · node ${d.node} · ${d.state} · ${d.util.toFixed(0)}% · ${d.tempC.toFixed(0)}°C · ${d.watts.toFixed(0)}W`}
      style={{
        width: size,
        height: size,
        borderRadius: 3,
        background: 'var(--panel-3)',
        position: 'relative',
        overflow: 'hidden',
        border: d.state === 'failed' ? '1px solid var(--err)' : '1px solid transparent',
      }}
    >
      <div
        style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: `${Math.max(0, Math.min(100, d.util))}%`,
          background: color,
          opacity: d.state === 'failed' ? 0.35 : 0.85,
          transition: 'height .18s linear',
        }}
      />
    </div>
  );
}

export default function ClusterLab() {
  const [spec, setSpec] = useState<ClusterSpec>(DEFAULT_CLUSTER);
  const [speed, setSpeed] = useState(900);
  const [running, setRunning] = useState(false);
  const [, setTick] = useState(0);
  const [tab, setTab] = useState<'plan' | 'run' | 'frontier'>('plan');

  const proj = useMemo(() => project(spec), [spec]);

  // A real tiny model trains alongside, and its loss is what appears in the log.
  const localRef = useRef<LMTrainer | null>(null);
  if (!localRef.current) localRef.current = buildLMTrainer('stories', 'tiny');
  const local = localRef.current;

  const simRef = useRef<ClusterSim | null>(null);
  if (!simRef.current) simRef.current = new ClusterSim(spec);
  const sim = simRef.current;

  useEffect(() => {
    sim.reconfigure(spec);
    setTick((t) => t + 1);
  }, [spec, sim]);

  const lastT = useRef(performance.now());
  useFrameLoop(running, () => {
    const now = performance.now();
    const dt = now - lastT.current;
    lastT.current = now;
    // Keep the real model moving so the loss in the log is genuine.
    local.runSlice(4);
    sim.tick(dt, local.latest()?.loss ?? null, speed);
    setTick((t) => t + 1);
  });

  const start = () => {
    if (!sim.running) sim.start();
    lastT.current = performance.now();
    setRunning(true);
    setTab('run');
  };
  const stop = () => {
    setRunning(false);
    sim.stop();
  };

  const patch = (p: Partial<ClusterSpec>) => setSpec((s) => ({ ...s, ...p }));

  const avgTemp = sim.devices.length ? sim.devices.reduce((a, d) => a + d.tempC, 0) / sim.devices.length : 0;
  const totalW = sim.devices.reduce((a, d) => a + d.watts, 0);
  const alive = sim.devices.filter((d) => d.state !== 'failed').length;
  const cellSize = proj.worldSize > 256 ? 7 : proj.worldSize > 128 ? 10 : proj.worldSize > 64 ? 13 : 17;

  const lossSeries: Series[] = [
    {
      id: 'loss',
      label: 'loss (from the model training locally)',
      color: 'var(--accent)',
      points: local.history.map((m) => ({ x: m.step, y: m.loss })),
    },
  ];

  return (
    <div className="mx-auto max-w-[1400px] px-6 py-6">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <Segmented
          value={tab}
          onChange={setTab}
          options={[
            { id: 'plan', label: '1 · Plan the run' },
            { id: 'run', label: '2 · Watch it run' },
            { id: 'frontier', label: '3 · Frontier scale' },
          ]}
        />
        <div className="flex items-center gap-2">
          <Badge tone={proj.memFits ? 'ok' : 'err'}>{proj.memFits ? 'fits in memory' : 'out of memory'}</Badge>
          <Badge tone="accent">{proj.worldSize} GPUs</Badge>
        </div>
      </div>

      <Callout tone="info" title="What is real here and what is not">
        The capacity plan is real arithmetic: step time, memory, network volume, wall clock and cost all come
        from published accelerator specifications and the standard six-FLOPs-per-parameter-per-token estimate.
        The per-device telemetry is a simulation, because there is no GPU fleet in a browser tab. The loss
        values scrolling through the log are genuine, taken from a real transformer training on this page while
        you watch.
      </Callout>

      <div className="mt-4 grid gap-4 xl:grid-cols-[320px_minmax(0,1fr)]">
        <div className="space-y-4">
          <Panel title="The job">
            <div className="space-y-3.5">
              <Slider
                label="model size" info={V.paramsB}
                value={Math.log10(spec.paramsB)}
                min={-2}
                max={4}
                step={0.02}
                format={(x) => `${fmtCount(Math.pow(10, x) * 1e9)} params`}
                onChange={(x) => patch({ paramsB: Math.pow(10, x) })}
              />
              <Slider
                label="training tokens" info={V.trainTokens}
                value={Math.log10(spec.tokensB)}
                min={-1}
                max={4.2}
                step={0.02}
                format={(x) => `${fmtCount(Math.pow(10, x) * 1e9)} tokens`}
                onChange={(x) => patch({ tokensB: Math.pow(10, x) })}
              />
              <div className="text-[11px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                Rough guidance is about twenty training tokens per parameter. For {fmtCount(proj.params)}{' '}
                parameters that would be {fmtCount(proj.params * 20)} tokens; you have set{' '}
                {fmtCount(proj.totalTokens)}.
              </div>
              <Slider
                label="global batch" info={V.globalBatch}
                value={Math.log2(spec.globalBatchTokens)}
                min={14}
                max={25}
                step={1}
                format={(x) => `${fmtCount(Math.pow(2, x))} tokens`}
                onChange={(x) => patch({ globalBatchTokens: Math.pow(2, Math.round(x)) })}
              />
            </div>
          </Panel>

          <Panel title="The cluster">
            <div className="space-y-3.5">
              <Field label="accelerator" info={V.accelerator}>
                <Select
                  value={spec.gpu.id}
                  onChange={(id) => patch({ gpu: GPU_PROFILES.find((g) => g.id === id)! })}
                  options={GPU_PROFILES.map((g) => ({ id: g.id, label: g.label }))}
                />
              </Field>
              <div className="grid grid-cols-3 gap-1.5 text-[10.5px]" style={{ color: 'var(--text-3)' }}>
                <div>
                  {spec.gpu.tflops} TFLOP/s
                  <div>BF16 dense</div>
                </div>
                <div>
                  {spec.gpu.memGB} GB
                  <div>{spec.gpu.memBandwidth} TB/s</div>
                </div>
                <div>
                  {spec.gpu.tdpWatts} W
                  <div>${spec.gpu.usdPerHour}/hr</div>
                </div>
              </div>
              <Slider
                label="nodes" info={V.nodes}
                value={spec.nodes}
                min={1}
                max={64}
                step={1}
                format={(x) => String(Math.round(x))}
                onChange={(x) => patch({ nodes: Math.round(x) })}
              />
              <Slider
                label="GPUs per node" info={V.gpusPerNode}
                value={spec.gpusPerNode}
                min={1}
                max={8}
                step={1}
                format={(x) => String(Math.round(x))}
                onChange={(x) => patch({ gpusPerNode: Math.round(x) })}
              />
              <Field label="precision" info={V.precision} hint={PRECISIONS[spec.precision].blurb}>
                <Select
                  value={spec.precision}
                  onChange={(p) => patch({ precision: p as Precision })}
                  options={(Object.keys(PRECISIONS) as Precision[]).map((p) => ({ id: p, label: PRECISIONS[p].label }))}
                />
              </Field>
              <Field label="parallelism" info={V.parallelism} hint={PARALLEL_INFO[spec.parallel].blurb}>
                <Select
                  value={spec.parallel}
                  onChange={(p) => patch({ parallel: p as ParallelMode })}
                  options={(Object.keys(PARALLEL_INFO) as ParallelMode[]).map((p) => ({
                    id: p,
                    label: PARALLEL_INFO[p].label,
                  }))}
                />
              </Field>
              <p className="text-[11px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                {PARALLEL_INFO[spec.parallel].blurb}
              </p>
              <Slider
                label="model FLOPs utilization" info={V.mfu}
                value={spec.mfu}
                min={0.1}
                max={0.7}
                step={0.01}
                format={(x) => `${(x * 100).toFixed(0)}%`}
                onChange={(x) => patch({ mfu: x })}
                hint="The fraction of peak the run actually achieves. Well-tuned large runs land between 35 and 50 percent."
              />
            </div>
          </Panel>
        </div>

        <div className="space-y-4">
          {/* Frontier keeps its own preset and weight tweaks; the other two
              branches read only from this component's state, so they are
              safe to rebuild. */}
          <Keep when={tab === 'frontier'}>
            <FrontierTab spec={spec} onSpec={setSpec} />
          </Keep>
          {tab === 'plan' ? (
            <>
              <Panel title="Capacity plan" subtitle="Computed from the specification on the left, not looked up">
                <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                  <Stat label="wall clock" value={fmtDuration(proj.wallSeconds)} tone="accent" />
                  <Stat label="cost" value={fmtUSD(proj.usdCost)} sub={`${fmtCount(proj.gpuHours)} GPU-hours`} />
                  <Stat label="throughput" value={fmtCount(proj.tokensPerSec)} unit="tok/s" />
                  <Stat label="power draw" value={fmt(proj.megawatts, 2)} unit="MW" sub={`${fmtCount(proj.mwhTotal)} MWh total`} />
                  <Stat label="total compute" value={fmtFlops(proj.totalFlops)} />
                  <Stat label="steps" value={fmtInt(proj.totalSteps)} sub={`${proj.msPerStep.toFixed(0)} ms each`} />
                  <Stat
                    label="memory per GPU"
                    value={`${proj.memPerGpuGB.toFixed(1)} GB`}
                    tone={proj.memFits ? 'ok' : 'err'}
                    sub={`of ${spec.gpu.memGB} GB`}
                  />
                  <Stat label="checkpoint size" value={`${proj.checkpointGB.toFixed(0)} GB`} />
                </div>

                <div className="mt-5 grid gap-5 md:grid-cols-2">
                  <div>
                    <Label>where each step goes</Label>
                    <div className="mt-2 space-y-2">
                      <div>
                        <div className="mb-0.5 flex justify-between text-[11px]">
                          <span style={{ color: 'var(--text-3)' }}>compute</span>
                          <span className="mono">{proj.computeMsPerStep.toFixed(0)} ms</span>
                        </div>
                        <BarMeter
                          value={proj.computeMsPerStep}
                          max={Math.max(proj.computeMsPerStep, proj.commsMsPerStep)}
                          color="var(--ok)"
                        />
                      </div>
                      <div>
                        <div className="mb-0.5 flex justify-between text-[11px]">
                          <span style={{ color: 'var(--text-3)' }}>communication</span>
                          <span className="mono">{proj.commsMsPerStep.toFixed(0)} ms</span>
                        </div>
                        <BarMeter
                          value={proj.commsMsPerStep}
                          max={Math.max(proj.computeMsPerStep, proj.commsMsPerStep)}
                          color="var(--pos)"
                        />
                      </div>
                    </div>
                    <p className="mt-2 text-[11px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                      {proj.commsMsPerStep > proj.computeMsPerStep
                        ? 'This run is communication bound. Adding GPUs will barely help and may hurt. Increase the batch size or shard less aggressively.'
                        : 'This run is compute bound, which is where you want to be. The network is keeping up.'}{' '}
                      Each step moves {proj.allReduceGB.toFixed(1)} GB of gradients across the interconnect.
                    </p>
                  </div>

                  <div>
                    <Label>memory per device</Label>
                    <div className="mt-2 space-y-1.5">
                      {([
                        ['weights', proj.weightsGB, 'var(--accent)'],
                        ['gradients', proj.gradsGB, 'var(--pos)'],
                        ['optimizer state', proj.optimStateGB, 'var(--ok)'],
                        ['activations', proj.activationsGB, 'var(--warn)'],
                      ] as [string, number, string][]).map(([l, v, c]) => (
                        <div key={l}>
                          <div className="mb-0.5 flex justify-between text-[11px]">
                            <span style={{ color: 'var(--text-3)' }}>{l}</span>
                            <span className="mono">{v.toFixed(1)} GB</span>
                          </div>
                          <BarMeter value={v} max={spec.gpu.memGB} color={c} height={5} />
                        </div>
                      ))}
                    </div>
                    <p className="mt-2 text-[11px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                      Adam keeps two extra numbers per parameter and mixed precision keeps an FP32 master copy,
                      so optimizer state is usually the largest single item, not the weights.
                    </p>
                  </div>
                </div>

                {!proj.memFits && (
                  <div className="mt-4">
                    <Callout tone="err" title="This will not start">
                      {proj.memPerGpuGB.toFixed(1)} GB is required per device but a {spec.gpu.label} has{' '}
                      {spec.gpu.memGB} GB. Switch to fully sharded parallelism, drop to a lower precision, cut
                      the batch size, or add GPUs so each holds a smaller shard.
                    </Callout>
                  </div>
                )}
              </Panel>

              <Panel title="The arithmetic behind those numbers">
                <Depth
                  plain={
                    <>
                      <p className="mb-2">
                        Training cost comes from one estimate: a model does roughly six units of arithmetic per
                        parameter for every token it sees. Two of those are the forward pass and four are the
                        backward pass, which costs about twice as much. Multiply by the number of parameters and
                        the number of tokens and you have the total work.
                      </p>
                      <p>
                        Then divide by how fast the hardware actually goes, which is never the number on the
                        box. A well-run job achieves around forty percent of peak. Everything else here follows
                        from those two figures.
                      </p>
                    </>
                  }
                  math={
                    <>
                      <Eq note="The estimate underpinning essentially every published training cost.">
                        C ≈ 6 · N · D, for N parameters and D training tokens
                      </Eq>
                      <div className="mono space-y-1 text-[11.5px]" style={{ color: 'var(--text-2)' }}>
                        <div>
                          C = 6 × {fmtCount(proj.params)} × {fmtCount(proj.totalTokens)} = {fmtFlops(proj.totalFlops)}
                        </div>
                        <div>
                          rate = {spec.gpu.tflops} TFLOP/s × {PRECISIONS[spec.precision].speedup} ×{' '}
                          {(spec.mfu * 100).toFixed(0)}% × {proj.worldSize} = {fmtFlops(proj.clusterFlopsPerSec)}/s
                        </div>
                        <div>wall clock = C / rate = {fmtDuration(proj.wallSeconds)}</div>
                        <div>
                          cost = {fmtCount(proj.gpuHours)} GPU-hours × ${spec.gpu.usdPerHour} = {fmtUSD(proj.usdCost)}
                        </div>
                      </div>
                      <p className="mt-2">
                        Communication is a ring all-reduce: each device sends <M>2(n−1)/n</M> times the gradient
                        payload, which is {proj.allReduceGB.toFixed(1)} GB per step over a {spec.gpu.linkGBs} GB/s
                        link.
                      </p>
                    </>
                  }
                  code={
                    <Eq>
                      <div className="space-y-1 text-[11px]">
                        <div>const flopsPerToken = 6 * params;</div>
                        <div>const perGpuFlops = gpu.tflops * 1e12 * prec.speedup * mfu;</div>
                        <div>const computeMsPerStep = stepFlops / (perGpuFlops * worldSize) * 1000;</div>
                        <div>const ringFactor = 2 * (worldSize - 1) / worldSize;</div>
                        <div>const commsMsPerStep = allReduceBytes * mult / linkBytesPerSec * 1000;</div>
                      </div>
                    </Eq>
                  }
                />
              </Panel>
            </>
          ) : (
            <>
              <Panel
                title="Cluster"
                subtitle={`${spec.nodes} nodes × ${spec.gpusPerNode} ${spec.gpu.label}`}
                right={
                  <div className="flex flex-wrap items-center gap-1.5">
                    <BackgroundNotice running={running} />
                    <Btn size="sm" variant="primary" onClick={running ? stop : start}>
                      {running ? 'Pause' : sim.step > 0 ? 'Resume' : 'Launch job'}
                    </Btn>
                    <Btn
                      size="sm"
                      onClick={() => {
                        sim.failRandomDevice();
                        setTick((t) => t + 1);
                      }}
                      disabled={!sim.running}
                    >
                      Kill a GPU
                    </Btn>
                    <Btn
                      size="sm"
                      onClick={() => {
                        sim.recoverAll();
                        setTick((t) => t + 1);
                      }}
                      disabled={sim.failedRanks.size === 0}
                    >
                      Recover
                    </Btn>
                  </div>
                }
              >
                <div className="mb-3 flex flex-wrap items-center gap-4">
                  <div className="flex items-center gap-2">
                    <Label>phase</Label>
                    <Badge tone={sim.phase === 'allreduce' ? 'warn' : 'ok'}>{sim.phase}</Badge>
                  </div>
                  <div style={{ width: 170 }}>
                    <Slider
                      label="sim speed" info={V.simSpeed}
                      value={Math.log10(speed)}
                      min={0}
                      max={4}
                      step={0.05}
                      format={(x) => `${fmtCount(Math.pow(10, x))}×`}
                      onChange={(x) => setSpeed(Math.pow(10, x))}
                    />
                  </div>
                  <Stat label="sim clock" value={sim.logs.length ? sim.logs[sim.logs.length - 1].ts : '00:00:00'} />
                </div>

                <div className="flex flex-wrap gap-2">
                  {Array.from({ length: spec.nodes }, (_, n) => (
                    <div
                      key={n}
                      className="rounded p-1"
                      style={{ background: 'var(--bg-2)', border: '1px solid var(--border)' }}
                      title={`node ${n}`}
                    >
                      <div className="flex gap-[2px]">
                        {sim.devices
                          .filter((d) => d.node === n)
                          .map((d) => (
                            <DeviceCell key={d.id} d={d} size={cellSize} />
                          ))}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-5">
                  <Stat label="step" value={fmtInt(sim.step)} sub={`of ${fmtInt(proj.totalSteps)}`} />
                  <Stat
                    label="healthy ranks"
                    value={`${alive} / ${proj.worldSize}`}
                    tone={alive === proj.worldSize ? 'ok' : 'err'}
                  />
                  <Stat label="avg temp" value={`${avgTemp.toFixed(0)}°C`} tone={avgTemp > 78 ? 'warn' : undefined} />
                  <Stat label="draw" value={`${(totalW / 1000).toFixed(1)} kW`} />
                  <Stat label="checkpoints" value={sim.checkpoints.length} />
                </div>
                <div className="mt-3">
                  <ProgressBar value={proj.totalSteps > 0 ? sim.step / proj.totalSteps : 0} />
                </div>
              </Panel>

              <div className="grid gap-4 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
                <Panel title="Log stream" subtitle="Loss values come from the real model training on this page" pad={false}>
                  <div
                    className="mono h-[290px] overflow-y-auto px-3 py-2 text-[10.5px] leading-[1.7]"
                    style={{ background: 'var(--bg-2)' }}
                    ref={(el) => {
                      if (el) el.scrollTop = el.scrollHeight;
                    }}
                  >
                    {sim.logs.length === 0 ? (
                      <span style={{ color: 'var(--text-3)' }}>Launch the job to begin.</span>
                    ) : (
                      sim.logs.map((l, i) => (
                        <div key={i} style={{ color: LOG_COLOR[l.level] }}>
                          <span style={{ color: 'var(--text-3)' }}>[{l.ts}]</span>{' '}
                          {l.rank !== null && <span style={{ color: 'var(--text-3)' }}>rank{l.rank} </span>}
                          {l.msg}
                        </div>
                      ))
                    )}
                  </div>
                </Panel>

                <div className="space-y-4">
                  <Panel title="Real loss, from a real model" subtitle="A transformer training in this tab while the simulation runs">
                    <LineChart series={lossSeries} height={130} xLabel="step" logY />
                    <div className="mt-2 grid grid-cols-2 gap-3">
                      <Stat label="local step" value={local.step} />
                      <Stat label="local loss" value={fmt(local.latest()?.loss ?? NaN, 4)} tone="accent" />
                    </div>
                  </Panel>

                  <Panel title="What a failure costs">
                    <p className="text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                      Press <strong>Kill a GPU</strong>. In a synchronous data-parallel run every rank waits on
                      every other rank, so one dead device stops the entire job. Recovery means allocating a
                      replacement, rebuilding the communicator and reloading the last checkpoint, which throws
                      away every step since it was written.
                    </p>
                    <p className="mt-2 text-[12px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                      At this scale a checkpoint is {proj.checkpointGB.toFixed(0)} GB. Writing more often
                      shortens what you lose but spends more time not training. That trade is a real and
                      constant argument on large runs.
                    </p>
                    {sim.failedRanks.size > 0 && (
                      <div className="mt-3">
                        <Callout tone="err" title={`${sim.failedRanks.size} rank(s) down`}>
                          The collective is blocked. Nothing is progressing, and the meter is still running at{' '}
                          {fmtUSD((proj.usdCost / Math.max(1, proj.wallSeconds)) * 3600)} per hour.
                        </Callout>
                      </div>
                    )}
                  </Panel>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
