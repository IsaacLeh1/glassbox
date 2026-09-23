import { useState } from 'react';
import type { LMTrainer } from '../engine/lmTrainer';
import { suggestedTokens } from '../engine/resources';
import { GPU_PROFILES, fmtCount, fmtDuration, fmtFlops, fmtUSD } from '../sim/cluster';
import { V } from '../content/varInfo';
import { Callout, Code, Depth, Empty, Field, Panel, Select, Slider, Stat, fmtInt } from '../ui/kit';
import { BarMeter } from '../ui/viz';

/**
 * Puts the reader's own measured throughput next to real accelerators.
 *
 * The local number is measured, not estimated. The hardware numbers use the
 * standard 6N FLOPs-per-parameter-per-token model, which is only valid once a
 * model is large enough to keep a device busy -- so when it is not, the panel
 * says so rather than printing a confident and meaningless figure.
 */

const SCALE_POINTS = [
  { id: 'yours', label: 'Your model', params: 0 },
  { id: '10m', label: '10M params', params: 10e6 },
  { id: '124m', label: '124M (GPT-2 Small)', params: 124e6 },
  { id: '1b', label: '1B params', params: 1e9 },
  { id: '8b', label: '8B (Llama 3 8B)', params: 8e9 },
  { id: '70b', label: '70B params', params: 70e9 },
];

export default function CompareTab({
  built,
  measuredTokPerSec,
}: {
  built: LMTrainer | null;
  measuredTokPerSec: number;
}) {
  const [gpuId, setGpuId] = useState('h100');
  const [worldSize, setWorldSize] = useState(64);
  const [mfu, setMfu] = useState(0.42);
  const [scaleId, setScaleId] = useState('yours');
  const gpu = GPU_PROFILES.find((g) => g.id === gpuId)!;

  if (!built || measuredTokPerSec <= 0) {
    return (
      <Empty>
        Train your model for a few seconds first. This comparison is built from your measured throughput, not
        from a guess.
      </Empty>
    );
  }

  const yourParams = built.model.paramCount;
  const scalePoint = SCALE_POINTS.find((p) => p.id === scaleId)!;
  const params = scalePoint.params || yourParams;
  const isYours = scalePoint.id === 'yours';

  // Your machine's measured arithmetic rate. It does not change with model
  // size, so everything below extrapolates from this single measurement.
  const localFlopsPerSec = measuredTokPerSec * 6 * yourParams;

  const targetTokens = suggestedTokens(params);
  const totalFlops = 6 * params * targetTokens;
  const secondsLocal = totalFlops / Math.max(1, localFlopsPerSec);

  const gpuFlopsPerSec = gpu.tflops * 1e12 * mfu;
  const secondsOneGpu = totalFlops / gpuFlopsPerSec;
  const secondsCluster = totalFlops / (gpuFlopsPerSec * worldSize);
  const oneGpuCost = (secondsOneGpu / 3600) * gpu.usdPerHour;
  const clusterCost = (secondsCluster / 3600) * worldSize * gpu.usdPerHour;
  const fractionOfOneGpu = localFlopsPerSec / gpuFlopsPerSec;

  // The 6N estimate assumes the model is big enough to saturate the device.
  // Below roughly ten million parameters that assumption does not hold at all.
  const tooSmallForGpuMath = params < 10e6;

  const rows = [
    { label: 'This browser tab', flops: localFlopsPerSec, seconds: secondsLocal, cost: 0, color: 'var(--accent)' },
    { label: `One ${gpu.label}`, flops: gpuFlopsPerSec, seconds: secondsOneGpu, cost: oneGpuCost, color: 'var(--pos)' },
    {
      label: `${worldSize} × ${gpu.label}`,
      flops: gpuFlopsPerSec * worldSize,
      seconds: secondsCluster,
      cost: clusterCost,
      color: 'var(--ok)',
    },
  ];

  return (
    <div className="space-y-4">
      <Panel
        title="Your browser tab against real hardware"
        subtitle="Your rate is measured from the run you just did. The hardware figures are computed from published specifications."
      >
        <div className="grid gap-4 md:grid-cols-4">
          <Field label="compare at" info={V.paramsB}>
            <Select
              value={scaleId}
              onChange={setScaleId}
              options={SCALE_POINTS.map((p) => ({
                id: p.id,
                label: p.id === 'yours' ? `Your model (${fmtCount(yourParams)})` : p.label,
              }))}
            />
          </Field>
          <Field label="accelerator" info={V.accelerator}>
            <Select
              value={gpuId}
              onChange={setGpuId}
              options={GPU_PROFILES.map((g) => ({ id: g.id, label: g.label }))}
            />
          </Field>
          <Slider
            label="cluster size"
            value={Math.log2(worldSize)}
            min={0}
            max={13}
            step={1}
            format={(x) => `${Math.pow(2, Math.round(x))} GPUs`}
            onChange={(x) => setWorldSize(Math.pow(2, Math.round(x)))}
            info={V.nodes}
          />
          <Slider
            label="assumed MFU"
            value={mfu}
            min={0.1}
            max={0.7}
            step={0.01}
            format={(x) => `${(x * 100).toFixed(0)}%`}
            onChange={setMfu}
            info={V.mfu}
          />
        </div>

        <div
          className="mt-4 rounded-lg p-3 text-[12px] leading-relaxed"
          style={{ background: 'var(--bg-2)', border: '1px solid var(--border)', color: 'var(--text-2)' }}
        >
          Training {isYours ? 'your' : 'a'}{' '}
          <span className="mono" style={{ color: 'var(--text)' }}>
            {fmtCount(params)}
          </span>
          -parameter model on its recommended{' '}
          <span className="mono" style={{ color: 'var(--text)' }}>
            {fmtCount(targetTokens)}
          </span>{' '}
          tokens is{' '}
          <span className="mono" style={{ color: 'var(--text)' }}>
            {fmtFlops(totalFlops)}
          </span>{' '}
          of arithmetic.
        </div>

        <div className="mt-4 space-y-3">
          {rows.map((r) => (
            <div key={r.label}>
              <div className="mb-1 flex flex-wrap items-baseline justify-between gap-2 text-[12px]">
                <span style={{ color: 'var(--text-2)' }}>{r.label}</span>
                <span className="mono" style={{ color: r.color }}>
                  {fmtFlops(r.flops)}/s
                  <span style={{ color: 'var(--text-3)' }}>
                    {'  ·  '}
                    {fmtDuration(r.seconds)}
                    {r.cost > 0 ? `  ·  ${fmtUSD(r.cost)}` : '  ·  free'}
                  </span>
                </span>
              </div>
              <BarMeter
                value={Math.log10(Math.max(1, r.flops))}
                max={Math.log10(Math.max(10, gpuFlopsPerSec * worldSize))}
                color={r.color}
                height={8}
              />
            </div>
          ))}
        </div>
        <p className="mt-2 text-[10.5px]" style={{ color: 'var(--text-3)' }}>
          Bars are on a log scale, because a linear one would render your tab as an invisible sliver.
        </p>

        <div className="mt-5 grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Stat label="your measured rate" value={fmtInt(measuredTokPerSec)} unit="tok/s" tone="accent" />
          <Stat label="your compute rate" value={fmtFlops(localFlopsPerSec)} unit="/s" />
          <Stat
            label="share of one GPU"
            value={
              fractionOfOneGpu < 1e-6
                ? `${(fractionOfOneGpu * 1e6).toFixed(2)} ppm`
                : `${(fractionOfOneGpu * 100).toFixed(4)}%`
            }
            hint="What fraction of a single accelerator this tab matches."
          />
          <Stat label="slower by" value={`${fmtCount(1 / Math.max(1e-12, fractionOfOneGpu))}×`} tone="warn" />
        </div>

        {tooSmallForGpuMath && (
          <div className="mt-4">
            <Callout tone="warn" title="At this size the GPU figures are a ceiling, not a forecast">
              The six-FLOPs-per-parameter-per-token estimate assumes the model is large enough to keep the
              accelerator busy. A {fmtCount(params)}-parameter model is nowhere near that: its matrices are so
              small that a real GPU would spend almost all its time launching kernels and waiting on memory
              rather than multiplying, and would sit largely idle. Treat the hardware rows above as an upper
              bound. Pick a larger size in the selector to see a comparison where the estimate genuinely holds.
            </Callout>
          </div>
        )}
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Why the gap is so large">
          <Depth
            plain={
              <>
                <p className="mb-2">
                  Your tab runs one arithmetic operation at a time, in JavaScript, on a single CPU core, with
                  every number stored at double the precision anyone actually uses. A training accelerator has
                  thousands of units doing matrix multiplies at once, memory built to feed them, and arithmetic
                  at half or a quarter of that precision.
                </p>
                <p>
                  That gap is five to six orders of magnitude. It is not a flaw in what you just did: your
                  model trained on real data with real backpropagation and genuinely learned something. It is
                  simply the difference between one core and a machine built for nothing else.
                </p>
              </>
            }
            math={
              <>
                <p className="mb-2">
                  Measured here: {fmtInt(measuredTokPerSec)} tok/s on {fmtCount(yourParams)} parameters, which
                  by the 6N estimate is {fmtFlops(localFlopsPerSec)} per second.
                </p>
                <p className="mb-2">
                  One {gpu.label} at {(mfu * 100).toFixed(0)}% of {gpu.tflops} TFLOP/s reaches{' '}
                  {fmtFlops(gpuFlopsPerSec)} per second, a factor of{' '}
                  {fmtCount(1 / Math.max(1e-12, fractionOfOneGpu))}.
                </p>
                <p>
                  Note the shape of the scaling. Training compute is 6ND, and the guidance D ≈ 20N makes that
                  120N². Doubling the model size quadruples the work, which is why the selector above moves the
                  wall clock so violently.
                </p>
              </>
            }
            code={
              <Code>{`const localFlopsPerSec = measuredTokPerSec * 6 * yourParams;
const totalFlops       = 6 * params * suggestedTokens(params);  // 20 tokens per param
const secondsLocal     = totalFlops / localFlopsPerSec;`}</Code>
            }
          />
        </Panel>

        <Panel title="What is identical">
          <p className="text-[12.5px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
            Everything conceptual. Your model samples a window of tokens, runs a forward pass, computes cross
            entropy against the next token, backpropagates through attention, layer norm and the feed-forward
            block, clips the gradient, and takes an Adam step on a warmup-then-cosine schedule. A frontier run
            does exactly that, with larger matrices and more of them, across thousands of devices that then
            have to agree on the result.
          </p>
          <p className="mt-2 text-[12.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
            The genuinely new problems at scale are not mathematical. They are keeping thousands of devices in
            step, surviving hardware failures, moving gradients fast enough, and not running out of memory.
            Step 05 is about exactly those.
          </p>
          <div className="mt-3">
            <Callout tone="insight">
              {isYours ? (
                <>
                  Your {fmtCount(yourParams)}-parameter model would need {fmtDuration(secondsLocal)} in this
                  tab to see its recommended {fmtCount(targetTokens)} tokens.
                </>
              ) : (
                <>
                  A {fmtCount(params)}-parameter model would take {fmtDuration(secondsLocal)} in this tab,
                  against {fmtDuration(secondsCluster)} and about {fmtUSD(clusterCost)} on {worldSize}{' '}
                  {gpu.label} cards. That is the entire reason this work happens in datacenters.
                </>
              )}
            </Callout>
          </div>
        </Panel>
      </div>
    </div>
  );
}
