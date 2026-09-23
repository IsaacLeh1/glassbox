import { useCallback, useMemo, useRef, useState } from 'react';
import { ACTIVATIONS, ACT_LIST, type ActName } from '../engine/activations';
import { MLP } from '../engine/mlp';
import { DATASET_INFO, DATASET_LIST, makeDataset, type DatasetName } from '../engine/datasets';
import { decisionField, neuronField } from '../engine/trainer';
import {
  Badge,
  Btn,
  Callout,
  Code,
  Depth,
  Eq,
  Field,
  Label,
  M,
  Panel,
  Segmented,
  Select,
  Slider,
  Stat,
  fmt,
  fmtPct,
  signedTextColor,
  Keep,
} from '../ui/kit';
import { V } from '../content/varInfo';
import { FieldCanvas, HeatGrid } from '../ui/viz';
import { NetworkGraph, type UnitRef, type WeightRef } from '../ui/NetworkGraph';
import WeightPrimer from '../ui/WeightPrimer';
import SignLegend from '../ui/SignLegend';

const RES = 72;

/* ====================================================== stage 1: neuron == */

function SingleNeuron() {
  const [w1, setW1] = useState(1.2);
  const [w2, setW2] = useState(-0.8);
  const [b, setB] = useState(0.1);
  const [act, setAct] = useState<ActName>('sigmoid');
  const [probe, setProbe] = useState({ x: 0.45, y: 0.3 });

  const f = ACTIVATIONS[act];

  const field = useMemo(() => {
    const out = new Float32Array(RES * RES);
    for (let i = 0; i < RES; i++) {
      for (let j = 0; j < RES; j++) {
        const x = -1.25 + (2.5 * j) / (RES - 1);
        const y = 1.25 - (2.5 * i) / (RES - 1);
        out[i * RES + j] = f.f(w1 * x + w2 * y + b);
      }
    }
    return out;
  }, [w1, w2, b, f]);

  const z = w1 * probe.x + w2 * probe.y + b;
  const a = f.f(z);
  const norm = Math.hypot(w1, w2);

  // The field for sigmoid/tanh sits in a different range than ReLU, so the
  // canvas is told which palette to use.
  const mode = act === 'sigmoid' ? 'prob' : 'signed';

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
      <div className="space-y-4">
        <WeightPrimer />

        <SignLegend />

        <Panel
          title="One neuron, three numbers"
          subtitle="Two weights and a bias. That is the entire thing."
          right={<Badge tone="accent">3 parameters</Badge>}
        >
          <div className="grid gap-5 md:grid-cols-[300px_minmax(0,1fr)]">
            <div>
              <FieldCanvas
                field={field}
                res={RES}
                size={300}
                mode={mode}
                probe={probe}
                onClickPoint={(x, y) => setProbe({ x, y })}
              />
              <p className="mt-2 text-[11px]" style={{ color: 'var(--text-3)' }}>
                Every pixel is this neuron evaluated at that point. Click anywhere to move the probe.
              </p>
            </div>

            <div className="space-y-4">
              <div className="space-y-3">
                <Slider
                  label={<>weight on x<sub>1</sub></>} info={V.weight1}
                  value={w1}
                  min={-3}
                  max={3}
                  step={0.01}
                  onChange={setW1}
                  tone="signed"
                  hint="How strongly the first input pushes the output up or down."
                />
                <Slider
                  label={<>weight on x<sub>2</sub></>} info={V.weight2}
                  value={w2}
                  min={-3}
                  max={3}
                  step={0.01}
                  onChange={setW2}
                  tone="signed"
                  hint="Same, for the second input."
                />
                <Slider
                  label="bias" info={V.bias}
                  value={b}
                  min={-3}
                  max={3}
                  step={0.01}
                  onChange={setB}
                  tone="signed"
                  hint="Shifts the whole boundary without rotating it."
                />
                <Field label="activation" info={V.activation}>
                  <Select
                    value={act}
                    onChange={(v) => setAct(v as ActName)}
                    options={ACT_LIST.map((x) => ({ id: x.name, label: x.label }))}
                  />
                </Field>
              </div>

              <div className="flex flex-wrap gap-1.5">
                <Btn size="sm" onClick={() => { setW1(1.2); setW2(-0.8); setB(0.1); }}>
                  Reset
                </Btn>
                <Btn size="sm" onClick={() => { setW1(-w1); setW2(-w2); setB(-b); }} title="Flip every sign">
                  Flip signs
                </Btn>
                <Btn size="sm" onClick={() => { setW1(w1 * 2); setW2(w2 * 2); setB(b * 2); }}>
                  Double all
                </Btn>
                <Btn size="sm" onClick={() => { setW1(0); setW2(0); }}>
                  Zero the weights
                </Btn>
              </div>
            </div>
          </div>
        </Panel>

        <Panel title="The calculation, with your numbers in it" subtitle="Recomputed on every change">
          <Eq
            note={
              <>
                The weighted sum <M>z</M> is the only place the inputs and weights meet. Everything else in a
                neural network is this line, repeated.
              </>
            }
          >
            <div className="space-y-1.5">
              <div>
                z = w<sub>1</sub>·x<sub>1</sub> + w<sub>2</sub>·x<sub>2</sub> + b
              </div>
              <div style={{ color: 'var(--text-2)' }}>
                z = (<span style={{ color: signedTextColor(w1) }}>{fmt(w1, 2)}</span>)(
                {fmt(probe.x, 2)}) + (<span style={{ color: signedTextColor(w2) }}>{fmt(w2, 2)}</span>)(
                {fmt(probe.y, 2)}) + (<span style={{ color: signedTextColor(b) }}>{fmt(b, 2)}</span>)
              </div>
              <div>
                z = <span style={{ color: signedTextColor(z) }}>{fmt(z, 4)}</span>
              </div>
              <div style={{ color: 'var(--text-3)' }}>
                a = {f.label}(z) = <span style={{ color: 'var(--accent)' }}>{fmt(a, 4)}</span>
              </div>
            </div>
          </Eq>

          <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Stat label="weighted sum z" value={fmt(z, 3)} />
            <Stat label="output a" value={fmt(a, 3)} tone="accent" />
            <Stat label="output range" value={f.range} />
            <Stat
              label="boundary angle"
              value={`${((Math.atan2(w2, w1) * 180) / Math.PI).toFixed(0)}°`}
              hint="Direction of the weight vector."
            />
          </div>

          <div className="mt-4">
            <Depth
              plain={
                <>
                  <p className="mb-2">
                    Read the three lines above from the top. The first is the recipe. The second is the same
                    recipe with your actual numbers dropped in. The third is the answer.
                  </p>
                  <p className="mb-2">
                    Each input is multiplied by its own weight, and those results are added together. Then the
                    bias is added on at the end, which is why it moves the answer even when both inputs are
                    zero. That single total is called <strong>z</strong>, and it is the only place in the whole
                    system where the inputs and the weights ever meet.
                  </p>
                  <p>
                    The last line squashes z into a sensible range — here, a number between 0 and 1 you can
                    read as a confidence. That squashing step matters more than it looks: without it, stacking
                    a hundred of these on top of each other would give you exactly the same thing as one.
                  </p>
                </>
              }
              math={
                <>
                  <p className="mb-2">
                    The neuron computes <M>a = f(w·x + b)</M> with <M>w, x ∈ ℝ²</M>. The set{' '}
                    <M>{'{x : w·x + b = 0}'}</M> is a hyperplane: in two dimensions, a line. Its normal vector is{' '}
                    <M>w</M>, so rotating <M>w</M> rotates the boundary, and its distance from the origin is{' '}
                    <M>|b| / ‖w‖</M> = {fmt(Math.abs(b) / (norm || 1e-9), 3)} right now.
                  </p>
                  <p>
                    Scaling <M>w</M> and <M>b</M> together by the same factor leaves the boundary where it is but
                    steepens <M>f</M> across it: the neuron becomes more decisive without changing its mind about
                    anything. That is exactly what the "Double all" button does.
                  </p>
                </>
              }
              code={
                <Code>{`// engine/mlp.ts -- the weighted sum, for a whole layer at once
const z = addRowVec(matmul(cur, this.W[l]), this.b[l]);
cur = mapMat(z, ACTIVATIONS[this.actAt(l)].f);

// engine/activations.ts -- the function you selected
${act}: {
  f:  ${ACTIVATIONS[act].f.toString().replace(/\s+/g, ' ')},
  df: ${ACTIVATIONS[act].df.toString().replace(/\s+/g, ' ').slice(0, 120)}
}`}</Code>
              }
            />
          </div>
        </Panel>
      </div>

      <div className="space-y-4">
        <Panel title="What you should see" pad>
          <ol className="space-y-2.5 text-[12.5px]" style={{ color: 'var(--text-2)' }}>
            {[
              'Drag the first weight slider only. The line where the colours meet swings around like a compass needle. The weights decide which direction that line points.',
              'Now drag the bias only. The same line slides across without turning at all. Weights set the direction; the bias sets the position.',
              'Set both weights to zero. Every pixel turns the same colour. With no weights there is nothing connecting the inputs to the answer, so the inputs stop mattering entirely.',
              'Press "Double all". The line does not move, but the colours get much sharper at the edge. The neuron has become more confident without changing its mind about anything.',
              'Switch the activation to ReLU. Half the picture goes completely flat. Those are inputs the neuron now ignores altogether, which is how a neuron can quietly die during training.',
            ].map((t, i) => (
              <li key={i} className="flex gap-2.5">
                <span
                  className="mono mt-[2px] flex h-4 w-4 shrink-0 items-center justify-center rounded text-[9.5px] font-semibold"
                  style={{ background: 'var(--panel-3)', color: 'var(--accent)' }}
                >
                  {i + 1}
                </span>
                <span>{t}</span>
              </li>
            ))}
          </ol>
        </Panel>

        <Panel title={`${f.label} in detail`}>
          <ActivationPlot act={act} at={z} />
          <p className="mt-2 text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
            {f.blurb}
          </p>
        </Panel>

        <Callout tone="insight">
          A single neuron can only ever draw one straight line. That is the whole limitation, and the reason
          the next section exists.
        </Callout>
      </div>
    </div>
  );
}

function ActivationPlot({ act, at }: { act: ActName; at: number }) {
  const f = ACTIVATIONS[act];
  const W = 280;
  const H = 110;
  const xr = 4;
  const sx = (x: number) => ((x + xr) / (2 * xr)) * W;
  const ys: number[] = [];
  for (let i = 0; i <= 120; i++) ys.push(f.f(-xr + (2 * xr * i) / 120));
  const lo = Math.min(...ys, -1);
  const hi = Math.max(...ys, 1);
  const sy = (y: number) => H - 8 - ((y - lo) / (hi - lo || 1)) * (H - 16);
  const d = ys.map((y, i) => `${i === 0 ? 'M' : 'L'}${sx(-xr + (2 * xr * i) / 120).toFixed(1)},${sy(y).toFixed(1)}`).join(' ');
  const dd = ys
    .map((_, i) => {
      const x = -xr + (2 * xr * i) / 120;
      return `${i === 0 ? 'M' : 'L'}${sx(x).toFixed(1)},${sy(f.df(x)).toFixed(1)}`;
    })
    .join(' ');
  const clamped = Math.max(-xr, Math.min(xr, at));

  return (
    <div>
      <svg width={W} height={H} style={{ display: 'block', maxWidth: '100%' }}>
        <line x1={0} x2={W} y1={sy(0)} y2={sy(0)} stroke="var(--grid)" />
        <line x1={sx(0)} x2={sx(0)} y1={0} y2={H} stroke="var(--grid)" />
        <path d={dd} fill="none" stroke="var(--ok)" strokeWidth={1.2} strokeDasharray="3 3" opacity={0.75} />
        <path d={d} fill="none" stroke="var(--accent)" strokeWidth={1.8} />
        <line x1={sx(clamped)} x2={sx(clamped)} y1={0} y2={H} stroke="var(--text-3)" strokeDasharray="2 3" />
        <circle cx={sx(clamped)} cy={sy(f.f(clamped))} r={3} fill="var(--accent)" />
      </svg>
      <div className="mt-1 flex gap-3 text-[10px]" style={{ color: 'var(--text-3)' }}>
        <span style={{ color: 'var(--accent)' }}>— f(z)</span>
        <span style={{ color: 'var(--ok)' }}>-- slope f&apos;(z)</span>
        <span className="ml-auto mono">
          f&apos;({fmt(at, 2)}) = {fmt(f.df(at), 3)}
        </span>
      </div>
    </div>
  );
}

/* ===================================================== stage 2: network == */

function NetworkExplorer() {
  const [dsName, setDsName] = useState<DatasetName>('xor');
  const [hidden, setHidden] = useState<number[]>([4, 3]);
  const [act, setAct] = useState<ActName>('tanh');
  const [seed, setSeed] = useState(7);
  const [version, bump] = useReducerBump();
  const [selected, setSelected] = useState<WeightRef | null>({ layer: 0, from: 0, to: 0 });
  const [selectedUnit, setSelectedUnit] = useState<UnitRef | null>(null);
  const [probe, setProbe] = useState<{ x: number; y: number } | null>({ x: 0.5, y: 0.5 });

  const ds = useMemo(() => makeDataset(dsName, 220, 0.09, 42), [dsName]);

  const netRef = useRef<MLP | null>(null);
  const sizes = useMemo(() => [2, ...hidden, 1], [hidden]);
  const sizeKey = sizes.join('-') + act + seed;
  const lastKey = useRef('');
  if (!netRef.current || lastKey.current !== sizeKey) {
    netRef.current = new MLP(sizes, hidden.map(() => act), 'binary', seed);
    lastKey.current = sizeKey;
  }
  const net = netRef.current;

  const field = useMemo(() => decisionField(net, RES, [-1.25, 1.25]), [net, version, sizeKey]);

  const unitField = useMemo(() => {
    if (!selectedUnit || selectedUnit.layer === 0 || selectedUnit.layer === net.sizes.length - 1) return null;
    return neuronField(net, selectedUnit.layer - 1, selectedUnit.unit, 56, [-1.25, 1.25]);
  }, [net, selectedUnit, version, sizeKey]);

  const probeActs = useMemo(() => {
    if (!probe) return null;
    const X = { rows: 1, cols: 2, data: new Float64Array([probe.x, probe.y]) };
    const pass = net.forward(X);
    return pass.A.map((A) => Array.from(A.data.slice(0, A.cols)));
  }, [net, probe, version, sizeKey]);

  const acc = useMemo(() => net.accuracy(ds.X, ds.Y), [net, ds, version, sizeKey]);
  const loss = useMemo(() => net.loss(net.forward(ds.X), ds.Y), [net, ds, version, sizeKey]);

  const selW = selected ? net.W[selected.layer].data[selected.from * net.W[selected.layer].cols + selected.to] : 0;

  const setSelW = useCallback(
    (v: number) => {
      if (!selected) return;
      const W = net.W[selected.layer];
      W.data[selected.from * W.cols + selected.to] = v;
      bump();
    },
    [net, selected, bump],
  );

  const grads = useMemo(() => {
    const pass = net.forward(ds.X);
    return net.backward(pass, ds.Y);
  }, [net, ds, version, sizeKey]);

  const selGrad = selected
    ? grads.dW[selected.layer].data[selected.from * grads.dW[selected.layer].cols + selected.to]
    : 0;

  return (
    <div className="space-y-4">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_320px]">
        <Panel
          title="Every weight in the network"
          subtitle="Click any connection to select it. Blue is negative, orange is positive, thickness is magnitude."
          right={<Badge tone="accent">{net.paramCount} parameters</Badge>}
        >
          <NetworkGraph
            net={net}
            activations={probeActs}
            selected={selected}
            onSelectWeight={setSelected}
            selectedUnit={selectedUnit}
            onSelectUnit={setSelectedUnit}
            width={560}
            height={280}
            showValues
          />

          {selected && (
            <div className="mt-3 rounded-lg p-3" style={{ background: 'var(--bg-2)', border: '1px solid var(--border)' }}>
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <span className="mono text-[11.5px]" style={{ color: 'var(--text-2)' }}>
                  W<sub>{selected.layer}</sub>[{selected.from} → {selected.to}]
                  <span style={{ color: 'var(--text-3)' }}>
                    {'  '}
                    {selected.layer === 0
                      ? `input x${selected.from + 1}`
                      : `h${selected.layer} unit ${selected.from}`}
                    {' → '}
                    {selected.layer === net.layerCount - 1
                      ? 'output'
                      : `h${selected.layer + 1} unit ${selected.to}`}
                  </span>
                </span>
                <div className="flex gap-1.5">
                  <Btn size="sm" onClick={() => setSelW(0)}>Zero</Btn>
                  <Btn size="sm" onClick={() => setSelW(-selW)}>Flip</Btn>
                  <Btn size="sm" onClick={() => setSelW(selW - 0.02 * selGrad * 400)} title="Take one gradient-descent step on this single weight">
                    Nudge downhill
                  </Btn>
                </div>
              </div>
              <Slider
                label="value" info={V.selectedWeight}
                value={selW}
                min={-4}
                max={4}
                step={0.001}
                onChange={setSelW}
                tone="signed"
              />
              <div className="mt-2 grid grid-cols-3 gap-3">
                <Stat label="weight" value={fmt(selW, 4)} />
                <Stat
                  label="gradient"
                  value={fmt(selGrad, 5)}
                  hint="How much the loss would rise if this weight increased slightly."
                />
                <Stat
                  label="wants to"
                  value={Math.abs(selGrad) < 1e-7 ? 'stay' : selGrad > 0 ? 'decrease' : 'increase'}
                  tone={Math.abs(selGrad) < 1e-7 ? undefined : 'ok'}
                  hint="Gradient descent moves opposite the gradient."
                />
              </div>
            </div>
          )}
        </Panel>

        <div className="space-y-4">
          <Panel title="What the network believes" subtitle="Painted by running it over every point">
            <FieldCanvas
              field={field}
              res={RES}
              size={286}
              points={ds.points}
              probe={probe}
              onClickPoint={(x, y) => setProbe({ x, y })}
            />
            <div className="mt-3 grid grid-cols-2 gap-3">
              <Stat label="accuracy" value={fmtPct(acc)} tone={acc > 0.9 ? 'ok' : acc > 0.7 ? 'warn' : 'err'} />
              <Stat label="loss" value={fmt(loss, 4)} />
            </div>
          </Panel>

          <Panel title="Setup">
            <div className="space-y-3">
              <Field label="dataset" info={V.dataset} hint={DATASET_INFO[dsName].blurb}>
                <Select
                  value={dsName}
                  onChange={(v) => setDsName(v as DatasetName)}
                  options={DATASET_LIST.map((d) => ({ id: d, label: DATASET_INFO[d].label }))}
                />
              </Field>
              <p className="text-[11.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                {DATASET_INFO[dsName].blurb}
              </p>
              <Field label="hidden layers" info={V.hiddenLayers}>
                <div className="flex flex-wrap gap-1.5">
                  {[[4], [8], [4, 3], [8, 8], [6, 6, 4]].map((h) => (
                    <Btn
                      key={h.join()}
                      size="sm"
                      active={h.join() === hidden.join()}
                      onClick={() => setHidden(h)}
                    >
                      {h.join(' · ')}
                    </Btn>
                  ))}
                </div>
              </Field>
              <Field label="hidden activation" info={V.hiddenActivation}>
                <Select
                  value={act}
                  onChange={(v) => setAct(v as ActName)}
                  options={ACT_LIST.filter((a) => a.name !== 'linear').map((a) => ({ id: a.name, label: a.label }))}
                />
              </Field>
              <Btn size="sm" onClick={() => setSeed((s) => s + 1)}>
                Reroll random initialisation
              </Btn>
            </div>
          </Panel>
        </div>
      </div>

      <Panel title="Reading the colours" subtitle="The same rule as step 01, in brief">
        <SignLegend compact />
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel
          title="What one hidden neuron responds to"
          subtitle={
            selectedUnit && selectedUnit.layer > 0 && selectedUnit.layer < net.sizes.length - 1
              ? `Hidden layer ${selectedUnit.layer}, unit ${selectedUnit.unit}`
              : 'Click a hidden neuron in the diagram above'
          }
        >
          {unitField ? (
            <div className="flex flex-wrap items-start gap-4">
              <FieldCanvas field={unitField} res={56} size={200} mode="signed" points={ds.points} />
              <div className="min-w-[180px] flex-1 space-y-2 text-[12px]" style={{ color: 'var(--text-2)' }}>
                <p>
                  This is the output of that single neuron across the whole input plane, before anything
                  downstream sees it. Orange is where it fires positive, blue where it fires negative.
                </p>
                <p style={{ color: 'var(--text-3)' }}>
                  Each hidden neuron contributes one soft straight edge. The output layer adds those edges
                  together with its own weights, and the curve you see in the decision map is the sum. Zero out a
                  weight leaving this neuron and watch the corresponding edge disappear.
                </p>
              </div>
            </div>
          ) : (
            <p className="text-[12.5px]" style={{ color: 'var(--text-3)' }}>
              Select a hidden neuron to see the shape it carves out on its own.
            </p>
          )}
        </Panel>

        <Panel
          title="The weight matrices"
          subtitle="The same numbers as the diagram, laid out the way the code stores them"
        >
          <div className="flex flex-wrap gap-5">
            {net.W.map((W, l) => (
              <div key={l}>
                <div className="mb-1.5 flex items-baseline gap-2">
                  <Label>W{l}</Label>
                  <span className="mono text-[10px]" style={{ color: 'var(--text-3)' }}>
                    {W.rows}×{W.cols}
                  </span>
                </div>
                <HeatGrid
                  rows={W.rows}
                  cols={W.cols}
                  get={(i, j) => W.data[i * W.cols + j]}
                  cell={20}
                  highlight={selected?.layer === l ? { i: selected.from, j: selected.to } : undefined}
                  onClick={(i, j) => setSelected({ layer: l, from: i, to: j })}
                />
              </div>
            ))}
          </div>
          <div className="mt-3">
            <Depth
              plain={
                <>
                  A layer of a neural network is stored as a grid of numbers. Each row is one incoming signal,
                  each column is one neuron receiving it, and the cell where they cross is how strongly that
                  signal is weighted. Everything a trained model knows is in grids like these.
                </>
              }
              math={
                <>
                  Layer <M>l</M> holds <M>W<sub>l</sub> ∈ ℝ<sup>n×m</sup></M> and <M>b<sub>l</sub> ∈ ℝ<sup>m</sup></M>,
                  computing <M>A<sub>l</sub> = f(A<sub>l-1</sub>W<sub>l</sub> + b<sub>l</sub>)</M>. This network has{' '}
                  {net.paramCount} parameters and needs {net.flopsPerExample} multiply-accumulates per example.
                  A frontier model is the same expression with dimensions in the tens of thousands.
                </>
              }
              code={
                <Code>{`interface Matrix { rows: number; cols: number; data: Float64Array }

// A weight lives at exactly one offset in a flat array:
const value = W.data[row * W.cols + col];`}</Code>
              }
            />
          </div>
        </Panel>
      </div>

      <Callout tone="insight">
        Nothing in this network is doing anything clever yet. The weights are still the random numbers it was
        born with, which is why the accuracy is near chance on anything harder than a straight line. Step 02
        is about where good weights come from.
      </Callout>
    </div>
  );
}

function useReducerBump(): [number, () => void] {
  const [v, setV] = useState(0);
  return [v, useCallback(() => setV((x) => x + 1), [])];
}

/* ================================================================ page == */

export default function NeuronLab() {
  const [stage, setStage] = useState<'one' | 'net'>('one');

  return (
    <div className="mx-auto max-w-[1320px] px-6 py-6">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <Segmented
          value={stage}
          onChange={setStage}
          options={[
            { id: 'one', label: '1 · A single neuron' },
            { id: 'net', label: '2 · A network of them' },
          ]}
        />
        <p className="text-[11.5px]" style={{ color: 'var(--text-3)' }}>
          Nothing here is trained yet. These are the weights the model starts life with.
        </p>
      </div>
      <Keep when={stage === 'one'}><SingleNeuron /></Keep>
      <Keep when={stage === 'net'}><NetworkExplorer /></Keep>
    </div>
  );
}
