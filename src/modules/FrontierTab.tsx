import { useMemo, useState } from 'react';
import { MLP } from '../engine/mlp';
import { fromRows } from '../engine/tensor';
import { fmtCount, fmtDuration, fmtFlops, fmtUSD, project, type ClusterSpec } from '../sim/cluster';
import {
  Badge,
  Btn,
  Callout,
  Depth,
  Label,
  M,
  Panel,
  Slider,
  Stat,
  fmt,
  fmtInt,
} from '../ui/kit';
import { BarMeter } from '../ui/viz';
import { V } from '../content/varInfo';

/**
 * What a run at frontier scale actually involves.
 *
 * Everything numeric here is derived rather than asserted. The architecture is
 * inferred from the parameter count using the usual transformer scaling
 * relation, and the failure rate is calibrated against the one large published
 * figure available: Meta reported 419 unexpected interruptions across 54 days
 * on roughly 16k H100s while training Llama 3 405B.
 */

/**
 * Each preset carries the fleet a run that size is actually done on. Inheriting
 * the teaching-sized cluster from the planner gives arithmetically correct but
 * absurd answers, like a three-trillion-parameter run taking thirteen hundred
 * years.
 */
const PRESETS = [
  { id: '175', label: 'GPT-3 scale', paramsB: 175, tokensB: 300, gpus: 1024 },
  { id: '405', label: 'Llama 3 405B', paramsB: 405, tokensB: 15600, gpus: 16384 },
  { id: '1000', label: '1 trillion', paramsB: 1000, tokensB: 20000, gpus: 32768 },
  { id: '3000', label: '3 trillion', paramsB: 3000, tokensB: 60000, gpus: 65536 },
  { id: '10000', label: '10 trillion', paramsB: 10000, tokensB: 200000, gpus: 131072 },
];

/**
 * Infer a plausible shape from a parameter count.
 *
 * Transformer parameters are about 12·L·d², and large models are built at an
 * aspect ratio of roughly d ≈ 128·L. Solving both together gives L from N
 * alone. Sanity check: at 175B this returns 96 layers and a width of 12288,
 * which is exactly GPT-3.
 */
function shapeFor(params: number): { layers: number; dModel: number; heads: number; dFF: number } {
  const layers = Math.max(1, Math.round(Math.cbrt(params / 196608)));
  const dModel = Math.round((128 * layers) / 128) * 128;
  const heads = Math.max(1, Math.round(dModel / 128));
  return { layers, dModel, heads, dFF: dModel * 4 };
}

export default function FrontierTab({ spec, onSpec }: { spec: ClusterSpec; onSpec: (s: ClusterSpec) => void }) {
  const [presetId, setPresetId] = useState('3000');
  const [rank, setRank] = useState(16);
  const [tweaked, setTweaked] = useState(0);
  const [gpus, setGpus] = useState(65536);

  const preset = PRESETS.find((p) => p.id === presetId)!;
  const params = preset.paramsB * 1e9;
  const shape = useMemo(() => shapeFor(params), [params]);

  const choose = (id: string) => {
    const p = PRESETS.find((x) => x.id === id)!;
    setPresetId(id);
    setGpus(p.gpus);
  };

  const localSpec: ClusterSpec = useMemo(
    () => ({
      ...spec,
      paramsB: preset.paramsB,
      tokensB: preset.tokensB,
      // Frontier runs fill whole halls, so the fleet comes from the preset
      // rather than from the teaching-sized cluster on the planner tab.
      nodes: Math.max(1, Math.round(gpus / 8)),
      gpusPerNode: 8,
      globalBatchTokens: 16_777_216,
    }),
    [spec, preset, gpus],
  );
  const proj = useMemo(() => project(localSpec), [localSpec]);

  /* ------- what one weight is worth, demonstrated on a real small model ---- */

  const demo = useMemo(() => {
    const net = new MLP([2, 6, 1], ['tanh'], 'binary', 21);
    const X = fromRows([[0.4, -0.3]]);
    const before = net.predict(X).data[0];
    const W = net.W[0];
    W.data[0] += tweaked;
    const after = net.predict(X).data[0];
    W.data[0] -= tweaked;
    return { before, after, delta: after - before, paramCount: net.paramCount };
  }, [tweaked]);

  /* ------------------------------------ what it takes to adapt the model -- */

  const bytesBf16 = params * 2;
  const fullFinetuneBytes = params * 16; // bf16 weights+grads, fp32 master and two Adam states
  // LoRA on the query and value projections of every layer.
  const loraParams = 4 * shape.layers * rank * shape.dModel;
  const secondsToInspect = params; // one weight per second
  const failuresPerDay = (proj.worldSize / 16384) * 8; // calibrated from the Llama 3 figure
  const expectedFailures = failuresPerDay * (proj.wallSeconds / 86400);
  const lostWork = Math.min(1, (expectedFailures * (250 * (proj.msPerStep / 1000))) / Math.max(1, proj.wallSeconds));

  return (
    <div className="space-y-4">
      <Panel
        title="A run at frontier scale"
        subtitle="The same arithmetic as the planner, taken to the sizes that make headlines"
        right={<Badge tone="accent">{fmtCount(params)} parameters</Badge>}
      >
        <div className="flex flex-wrap gap-1.5">
          {PRESETS.map((p) => (
            <Btn key={p.id} size="sm" active={p.id === presetId} onClick={() => choose(p.id)}>
              {p.label}
            </Btn>
          ))}
          <Btn size="sm" onClick={() => onSpec(localSpec)} title="Load this into the planner and the live cluster">
            Send to the planner
          </Btn>
        </div>

        <div className="mt-4 max-w-[360px]">
          <Slider
            label="GPUs in the fleet"
            value={Math.log2(gpus)}
            min={6}
            max={18}
            step={1}
            format={(x) => `${fmtInt(Math.pow(2, Math.round(x)))} × ${spec.gpu.label}`}
            onChange={(x) => setGpus(Math.pow(2, Math.round(x)))}
            info={V.nodes}
          />
        </div>

        <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Stat label="wall clock" value={fmtDuration(proj.wallSeconds)} tone="accent" />
          <Stat label="cost" value={fmtUSD(proj.usdCost)} sub={`${fmtCount(proj.gpuHours)} GPU-hours`} />
          <Stat label="total compute" value={fmtFlops(proj.totalFlops)} />
          <Stat label="energy" value={`${fmtCount(proj.mwhTotal)} MWh`} sub={`${fmt(proj.megawatts, 1)} MW draw`} />
        </div>

        <div className="mt-5 grid gap-5 lg:grid-cols-2">
          <div>
            <Label>the shape this size implies</Label>
            <div className="mt-2 space-y-1.5 text-[11.5px]">
              {[
                ['layers', fmtInt(shape.layers)],
                ['width of each token vector', fmtInt(shape.dModel)],
                ['attention heads per layer', fmtInt(shape.heads)],
                ['feed-forward width', fmtInt(shape.dFF)],
                ['largest single matrix', `${fmtInt(shape.dModel)} × ${fmtInt(shape.dFF)}`],
                ['weights, stored in BF16', `${fmtCount(bytesBf16)}B`],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between">
                  <span style={{ color: 'var(--text-3)' }}>{k}</span>
                  <span className="mono" style={{ color: 'var(--text-2)' }}>
                    {v}
                  </span>
                </div>
              ))}
            </div>
            <p className="mt-2 text-[10.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
              Inferred from the parameter count using the usual transformer relation and the aspect ratio
              large models are actually built at. As a check, feeding it 175B returns 96 layers at width
              12288, which is exactly GPT-3. Above roughly a trillion parameters real models are usually
              mixture-of-experts instead, where only a fraction of the weights run on any given token, so
              treat a dense shape at that size as an upper bound rather than a blueprint.
            </p>
          </div>

          <div>
            <Label>what goes wrong, and how often</Label>
            <div className="mt-2 space-y-1.5 text-[11.5px]">
              {[
                ['GPUs in the job', fmtInt(proj.worldSize)],
                ['expected hardware failures', fmtInt(expectedFailures)],
                ['roughly one every', fmtDuration(86400 / Math.max(1e-6, failuresPerDay))],
                ['checkpoint size', `${fmtCount(proj.checkpointGB * 1e9)}B`],
                ['work lost to restarts', `${(lostWork * 100).toFixed(1)}%`],
                ['cost of that lost work', fmtUSD(proj.usdCost * lostWork)],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between">
                  <span style={{ color: 'var(--text-3)' }}>{k}</span>
                  <span className="mono" style={{ color: 'var(--text-2)' }}>
                    {v}
                  </span>
                </div>
              ))}
            </div>
            <p className="mt-2 text-[10.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
              Calibrated against the one large published figure: Meta reported 419 unexpected interruptions
              over 54 days on about 16,000 H100s while training Llama 3 405B. At this cluster size that rate
              scales to the numbers above.
            </p>
          </div>
        </div>
      </Panel>

      {/* ------------------------------------------- one weight at this scale */}

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="What changing one weight does at this scale">
          <p className="mb-3 text-[12.5px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
            You can edit one weight in step 01 and watch the answer move, because that model has a few dozen
            of them. Here is the same act, in proportion.
          </p>

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            <Stat label="one weight is" value={`1 in ${fmtCount(params)}`} />
            <Stat
              label="as a fraction"
              value={`${(100 / params).toExponential(1)}%`}
              hint="Of the model's total parameters."
            />
            <Stat
              label="to look at them all"
              value={fmtDuration(secondsToInspect)}
              hint="At one weight per second, without sleeping."
            />
          </div>

          <div className="mt-4">
            <Label>so try it on a model small enough to feel it</Label>
            <p className="mt-1 mb-2 text-[11.5px]" style={{ color: 'var(--text-3)' }}>
              This is a real {demo.paramCount}-parameter network. Nudge one of its weights.
            </p>
            <Slider
              label="change to a single weight"
              value={tweaked}
              min={-3}
              max={3}
              step={0.05}
              onChange={setTweaked}
              tone="signed"
              info={V.selectedWeight}
            />
            <div className="mt-2 grid grid-cols-3 gap-3">
              <Stat label="output before" value={fmt(demo.before, 4)} />
              <Stat label="output after" value={fmt(demo.after, 4)} tone="accent" />
              <Stat label="moved by" value={fmt(demo.delta, 4)} tone={Math.abs(demo.delta) > 0.01 ? 'warn' : undefined} />
            </div>
          </div>

          <div className="mt-3">
            <Callout tone="insight" title="Now scale that intuition">
              In a {demo.paramCount}-parameter network one weight is {(100 / demo.paramCount).toFixed(1)}% of
              the model and you can clearly see it move the answer. In a {fmtCount(params)}-parameter model
              the same single weight is {(100 / params).toExponential(1)}% — around{' '}
              {fmtCount(params / demo.paramCount)} times less influential. No individual weight means anything
              on its own at that size, which is why nobody at a frontier lab edits one. Behaviour lives in the
              pattern across billions of them, not in any particular number.
            </Callout>
          </div>
        </Panel>

        <Panel title="So how is a model that size actually changed?">
          <div className="space-y-3">
            <div className="rounded-lg p-3" style={{ background: 'var(--bg-2)' }}>
              <div className="mb-1 flex items-center justify-between">
                <span className="text-[12px] font-semibold">Full fine-tuning</span>
                <span className="mono text-[11px]" style={{ color: 'var(--warn)' }}>
                  {fmtCount(fullFinetuneBytes)}B of memory
                </span>
              </div>
              <p className="text-[11.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                Update every parameter. Needs weights, gradients, an FP32 master copy and two Adam states, so
                roughly sixteen bytes per parameter. That is{' '}
                {fmtInt(Math.ceil(fullFinetuneBytes / (80 * 1e9)))} H100s just to hold it, before any
                activations. Almost nobody does this outside the labs that trained the model.
              </p>
            </div>

            <div className="rounded-lg p-3" style={{ background: 'var(--bg-2)' }}>
              <div className="mb-1 flex items-center justify-between">
                <span className="text-[12px] font-semibold">Low-rank adaptation</span>
                <span className="mono text-[11px]" style={{ color: 'var(--ok)' }}>
                  {fmtCount(loraParams)} trainable
                </span>
              </div>
              <div className="mb-2">
                <Slider
                  label="rank"
                  info={V.loraRank}
                  value={rank}
                  min={1}
                  max={128}
                  step={1}
                  format={(x) => String(Math.round(x))}
                  onChange={(x) => setRank(Math.round(x))}
                />
              </div>
              <BarMeter value={Math.log10(Math.max(1, loraParams))} max={Math.log10(params)} color="var(--ok)" height={6} />
              <p className="mt-1.5 text-[11.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                Freeze everything and learn a small pair of thin matrices alongside the big ones. At rank{' '}
                {rank} across {fmtInt(shape.layers)} layers that is {fmtCount(loraParams)} parameters —{' '}
                <span style={{ color: 'var(--ok)' }}>{((loraParams / params) * 100).toFixed(4)}%</span> of the
                model. This is how essentially all customisation of large models is done.
              </p>
            </div>

            <div className="rounded-lg p-3" style={{ background: 'var(--bg-2)' }}>
              <div className="mb-1 flex items-center justify-between">
                <span className="text-[12px] font-semibold">Steering the activations</span>
                <span className="mono text-[11px]" style={{ color: 'var(--accent)' }}>
                  {fmtInt(shape.dModel)} numbers
                </span>
              </div>
              <p className="text-[11.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                Change no weights at all. Find a direction in the model&apos;s internal state and add it while
                the model runs. One vector the width of the model — {fmtInt(shape.dModel)} numbers against{' '}
                {fmtCount(params)} — can visibly shift behaviour. Step 08 does exactly this, for real, on
                the model in your browser.
              </p>
            </div>

            <div className="rounded-lg p-3" style={{ background: 'var(--bg-2)' }}>
              <div className="mb-1 flex items-center justify-between">
                <span className="text-[12px] font-semibold">Changing the prompt</span>
                <span className="mono text-[11px]" style={{ color: 'var(--accent)' }}>
                  0 parameters
                </span>
              </div>
              <p className="text-[11.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                Cheapest of all, reversible, and often the largest single improvement available. Step 06
                measures exactly how much it buys on a model you connect yourself.
              </p>
            </div>
          </div>
        </Panel>
      </div>

      <Panel title="What a run like this is really like">
        <Depth
          plain={
            <div className="space-y-2">
              <p>
                It is far less dramatic than people imagine and far more operational. A team spends weeks on
                data before anything starts. The run itself is launched, and then mostly watched. Someone is
                on call at all hours, because the cluster will lose hardware roughly{' '}
                {failuresPerDay < 1
                  ? `once every ${fmtDuration(86400 / Math.max(1e-6, failuresPerDay))}`
                  : `${failuresPerDay.toFixed(1)} times a day`}{' '}
                and a stalled job is burning {fmtUSD((proj.usdCost / Math.max(1, proj.wallSeconds)) * 3600)} an
                hour while it sits there.
              </p>
              <p>
                Nobody adjusts weights by hand. Nobody inspects a neuron and decides it looks wrong. The
                levers are the data, the learning rate schedule, the batch size, and when to stop. Almost all
                of the skill is in noticing early that a loss curve has a shape it should not have, because
                restarting on day two is cheap and restarting on day forty is not.
              </p>
              <p>
                At the end there is a set of numbers {fmtCount(bytesBf16)} bytes in size, and the real work of
                finding out what it can actually do is only beginning.
              </p>
            </div>
          }
          math={
            <div className="space-y-2">
              <p>
                Total compute is <M>C = 6ND = 6 × {fmtCount(params)} × {fmtCount(proj.totalTokens)}</M> ={' '}
                {fmtFlops(proj.totalFlops)}, delivered at {fmtFlops(proj.clusterFlopsPerSec)} per second by{' '}
                {fmtInt(proj.worldSize)} devices at {(localSpec.mfu * 100).toFixed(0)}% utilisation.
              </p>
              <p>
                Useful throughput is degraded by failures. With an expected{' '}
                {fmtInt(expectedFailures)} interruptions and a mean loss of half a checkpoint interval per
                event, roughly {(lostWork * 100).toFixed(1)}% of the spend buys nothing. Shortening the
                checkpoint interval reduces that term and increases fixed overhead; the optimum is where the
                two derivatives meet.
              </p>
            </div>
          }
          code={
            <div className="space-y-1">
              <p className="text-[11.5px]">The shape inference, which reproduces GPT-3 from its size alone:</p>
              <div className="mono text-[11px]">
                <div>const layers = Math.round(Math.cbrt(params / 196608)); // N ~ 12·L·d², d ~ 128·L</div>
                <div>const dModel = layers * 128;</div>
              </div>
            </div>
          }
        />
      </Panel>
    </div>
  );
}
