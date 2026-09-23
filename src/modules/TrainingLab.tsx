import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ACT_LIST, type ActName } from '../engine/activations';
import { MLP } from '../engine/mlp';
import { DATASET_INFO, DATASET_LIST, makeDataset, splitDataset, type DatasetName } from '../engine/datasets';
import { DEFAULT_TRAINER, Trainer, decisionField, type TrainerConfig } from '../engine/trainer';
import { OPTIM_INFO, SCHEDULE_INFO, type OptimName, type ScheduleName } from '../engine/optim';
import { Value, v as mkVal } from '../engine/autograd';
import {
  Badge,
  BackgroundNotice,
  Btn,
  Callout,
  Code,
  Depth,
  Eq,
  Field,
  M,
  Panel,
  ProgressBar,
  Segmented,
  Select,
  Slider,
  Stat,
  fmt,
  fmtPct,
  signedTextColor,
} from '../ui/kit';
import { V } from '../content/varInfo';
import { FieldCanvas, LineChart, type Series } from '../ui/viz';
import { NetworkGraph, type WeightRef } from '../ui/NetworkGraph';
import { useFrameLoop } from '../ui/loop';

const RES = 64;

/* ================================================== tab 1: live training */

function TrainTab() {
  const [dsName, setDsName] = useState<DatasetName>('circles');
  const [noise, setNoise] = useState(0.09);
  const [hidden, setHidden] = useState<number[]>([8, 6]);
  const [act, setAct] = useState<ActName>('tanh');
  const [cfg, setCfg] = useState<TrainerConfig>(DEFAULT_TRAINER);
  const [running, setRunning] = useState(false);
  const [, setTick] = useState(0);
  const [selected, setSelected] = useState<WeightRef | null>(null);
  const [showGrads, setShowGrads] = useState(true);

  const ds = useMemo(() => makeDataset(dsName, 260, noise, 42), [dsName, noise]);
  const split = useMemo(() => splitDataset(ds, 0.3, 7), [ds]);

  const key = `${dsName}|${noise}|${hidden.join()}|${act}|${JSON.stringify(cfg)}`;
  const trainerRef = useRef<Trainer | null>(null);
  const lastKey = useRef('');
  if (!trainerRef.current || lastKey.current !== key) {
    const net = new MLP([2, ...hidden, 1], hidden.map(() => act), 'binary', cfg.seed);
    trainerRef.current = new Trainer(net, split, cfg);
    lastKey.current = key;
  }
  const trainer = trainerRef.current;
  const net = trainer.net;

  // Run as much real training as fits in a frame budget, and keep going even
  // if the tab is put in the background.
  useFrameLoop(running, () => {
    const n = trainer.runSlice(9);
    if (n === 0 || trainer.status === 'done') setRunning(false);
    setTick((t) => t + 1);
  });

  const field = useMemo(
    () => decisionField(net, RES, [-1.25, 1.25]),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [net, trainer.step, key],
  );

  const grads = useMemo(() => {
    const pass = net.forward(split.train.X);
    return net.backward(pass, split.train.Y);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [net, split, trainer.step, key]);

  const h = trainer.history;
  const last = trainer.latest();

  const lossSeries: Series[] = [
    { id: 'train', label: 'train loss', color: 'var(--accent)', points: h.map((m) => ({ x: m.step, y: m.loss })) },
    { id: 'val', label: 'validation loss', color: 'var(--pos)', dashed: true, points: h.map((m) => ({ x: m.step, y: m.valLoss })) },
  ];
  const accSeries: Series[] = [
    { id: 'acc', label: 'train accuracy', color: 'var(--ok)', points: h.map((m) => ({ x: m.step, y: m.acc })) },
    { id: 'vacc', label: 'validation accuracy', color: 'var(--warn)', dashed: true, points: h.map((m) => ({ x: m.step, y: m.valAcc })) },
  ];

  const gap = last ? last.valLoss - last.loss : 0;
  const overfitting = h.length > 20 && gap > 0.12;

  const reset = useCallback(() => {
    setRunning(false);
    trainer.reset();
    setTick((t) => t + 1);
  }, [trainer]);

  return (
    <div className="grid gap-4 xl:grid-cols-[300px_minmax(0,1fr)]">
      {/* ------------------------------------------------------- controls */}
      <div className="space-y-4">
        <Panel title="Run">
          <div className="mb-3 flex items-center gap-1.5">
            <BackgroundNotice running={running} />
            <Btn variant="primary" onClick={() => setRunning((r) => !r)} disabled={trainer.status === 'done'}>
              {running ? 'Pause' : trainer.step > 0 ? 'Resume' : 'Train'}
            </Btn>
            <Btn
              onClick={() => {
                trainer.singleStep();
                setTick((t) => t + 1);
              }}
              disabled={running || trainer.status === 'done'}
              title="One mini-batch, one weight update"
            >
              Step
            </Btn>
            <Btn onClick={reset}>Reset</Btn>
          </div>
          <ProgressBar value={trainer.progress} tone={trainer.status === 'done' ? 'ok' : 'accent'} />
          <div className="mt-2 grid grid-cols-2 gap-2.5">
            <Stat label="step" value={trainer.step.toLocaleString()} sub={`epoch ${trainer.epoch}`} />
            <Stat label="examples seen" value={trainer.examplesSeen.toLocaleString()} />
          </div>
        </Panel>

        <Panel title="Hyperparameters" subtitle="Each of these changes the run for a specific reason">
          <div className="space-y-3.5">
            <Field label="optimizer" info={V.optimizer} hint={OPTIM_INFO[cfg.optim.name].blurb}>
              <Select
                value={cfg.optim.name}
                onChange={(n) => setCfg((c) => ({ ...c, optim: { ...c.optim, name: n as OptimName } }))}
                options={(Object.keys(OPTIM_INFO) as OptimName[]).map((o) => ({ id: o, label: OPTIM_INFO[o].label }))}
              />
            </Field>
            <p className="-mt-1 text-[11px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
              {OPTIM_INFO[cfg.optim.name].blurb}
            </p>

            <Slider
              label="learning rate" info={V.learningRate}
              value={Math.log10(cfg.optim.lr)}
              min={-4}
              max={0.5}
              step={0.02}
              format={(x) => Math.pow(10, x).toPrecision(2)}
              onChange={(x) => setCfg((c) => ({ ...c, optim: { ...c.optim, lr: Math.pow(10, x) } }))}
              hint="How far to move on each step. The single most consequential number in training."
            />
            <Slider
              label="batch size" info={V.batchSize}
              value={cfg.batchSize}
              min={1}
              max={128}
              step={1}
              format={(x) => String(Math.round(x))}
              onChange={(x) => setCfg((c) => ({ ...c, batchSize: Math.round(x) }))}
              hint="Examples averaged per update. Small batches are noisy but often generalise better."
            />
            <Slider
              label="epochs" info={V.epochs}
              value={cfg.epochs}
              min={20}
              max={800}
              step={10}
              format={(x) => String(Math.round(x))}
              onChange={(x) => setCfg((c) => ({ ...c, epochs: Math.round(x) }))}
            />
            <Field label="lr schedule" info={V.lrSchedule} hint={SCHEDULE_INFO[cfg.schedule].blurb}>
              <Select
                value={cfg.schedule}
                onChange={(s) => setCfg((c) => ({ ...c, schedule: s as ScheduleName }))}
                options={(Object.keys(SCHEDULE_INFO) as ScheduleName[]).map((s) => ({
                  id: s,
                  label: SCHEDULE_INFO[s].label,
                }))}
              />
            </Field>
            <Slider
              label="L2 regularisation" info={V.l2}
              value={cfg.l2}
              min={0}
              max={0.02}
              step={0.0005}
              format={(x) => x.toFixed(4)}
              onChange={(x) => setCfg((c) => ({ ...c, l2: x }))}
              hint="Penalises large weights. The standard cure for overfitting."
            />
            <Slider
              label="gradient clipping" info={V.clipNorm}
              value={cfg.optim.clipNorm}
              min={0}
              max={5}
              step={0.1}
              format={(x) => (x === 0 ? 'off' : x.toFixed(1))}
              onChange={(x) => setCfg((c) => ({ ...c, optim: { ...c.optim, clipNorm: x } }))}
              hint="Caps the length of the whole gradient vector so one bad batch cannot wreck the run."
            />
          </div>
        </Panel>

        <Panel title="Problem">
          <div className="space-y-3">
            <Field label="dataset" info={V.dataset}>
              <Select
                value={dsName}
                onChange={(d) => setDsName(d as DatasetName)}
                options={DATASET_LIST.map((d) => ({ id: d, label: DATASET_INFO[d].label }))}
              />
            </Field>
            <Slider label="label noise" info={V.labelNoise} value={noise} min={0} max={0.45} step={0.01} onChange={setNoise} />
            <Field label="hidden layers" info={V.hiddenLayers}>
              <div className="flex flex-wrap gap-1.5">
                {[[], [4], [8, 6], [16, 12], [24, 24, 16]].map((hh, i) => (
                  <Btn key={i} size="sm" active={hh.join() === hidden.join()} onClick={() => setHidden(hh)}>
                    {hh.length ? hh.join(' · ') : 'none'}
                  </Btn>
                ))}
              </div>
            </Field>
            <Field label="activation" info={V.hiddenActivation}>
              <Select
                value={act}
                onChange={(a) => setAct(a as ActName)}
                options={ACT_LIST.filter((a) => a.name !== 'linear').map((a) => ({ id: a.name, label: a.label }))}
              />
            </Field>
          </div>
        </Panel>
      </div>

      {/* --------------------------------------------------------- output */}
      <div className="space-y-4">
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_300px]">
          <Panel title="Loss" subtitle="What the optimizer is actually minimising" right={<Badge>{h.length} samples</Badge>}>
            <LineChart series={lossSeries} height={168} yLabel="loss" xLabel="step" logY />
            <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Stat label="train loss" value={fmt(last?.loss ?? NaN, 4)} tone="accent" />
              <Stat label="val loss" value={fmt(last?.valLoss ?? NaN, 4)} tone={overfitting ? 'warn' : undefined} />
              <Stat label="grad norm" value={fmt(last?.gradNorm ?? NaN, 4)} hint="Length of the full gradient vector." />
              <Stat label="learning rate" value={fmt(last?.lr ?? cfg.optim.lr, 4)} />
            </div>
          </Panel>

          <Panel title="Decision boundary" subtitle="Redrawn every frame from the live weights">
            <FieldCanvas field={field} res={RES} size={268} points={split.train.points} />
            <div className="mt-3 grid grid-cols-2 gap-3">
              <Stat
                label="train acc"
                value={fmtPct(last?.acc ?? net.accuracy(split.train.X, split.train.Y))}
                tone="ok"
              />
              <Stat
                label="val acc"
                value={fmtPct(last?.valAcc ?? net.accuracy(split.test.X, split.test.Y))}
                tone={overfitting ? 'warn' : undefined}
              />
            </div>
          </Panel>
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <Panel title="Accuracy" subtitle="Held-out data is the only honest measure">
            <LineChart series={accSeries} height={140} yLabel="accuracy" xLabel="step" yMin={0} yMax={1} />
            {overfitting && (
              <div className="mt-3">
                <Callout tone="warn" title="Overfitting">
                  Training loss keeps falling while validation loss has turned upward. The model is now
                  memorising individual points rather than learning the pattern. Raise L2 regularisation, add
                  label noise, or shrink the network and watch the two curves come back together.
                </Callout>
              </div>
            )}
          </Panel>

          <Panel
            title="Gradients on the network"
            subtitle="Dashed green shows where backpropagation wants to push"
            right={
              <Btn size="sm" active={showGrads} onClick={() => setShowGrads((s) => !s)}>
                {showGrads ? 'Hide' : 'Show'}
              </Btn>
            }
          >
            <NetworkGraph
              net={net}
              gradients={showGrads ? grads : null}
              selected={selected}
              onSelectWeight={setSelected}
              width={430}
              height={210}
              flowing={running}
            />
            {selected && (
              <div className="mono mt-2 text-[11px]" style={{ color: 'var(--text-3)' }}>
                W<sub>{selected.layer}</sub>[{selected.from}→{selected.to}] ={' '}
                <span style={{ color: signedTextColor(net.W[selected.layer].data[selected.from * net.W[selected.layer].cols + selected.to]) }}>
                  {fmt(net.W[selected.layer].data[selected.from * net.W[selected.layer].cols + selected.to], 4)}
                </span>
                {'   grad = '}
                <span style={{ color: 'var(--ok)' }}>
                  {fmt(grads.dW[selected.layer].data[selected.from * grads.dW[selected.layer].cols + selected.to], 5)}
                </span>
              </div>
            )}
          </Panel>
        </div>

        <Panel title="What is happening on every step">
          <Depth
            plain={
              <>
                <p className="mb-2">
                  Training is a loop of four moves, repeated thousands of times. Take a handful of examples.
                  Run them through the network and see what it predicts. Compare that to the right answer and
                  measure how wrong it was. Then work out, for every single weight, whether nudging it up or
                  down would have made the answer better, and nudge all of them a tiny amount in the better
                  direction.
                </p>
                <p>
                  That is the whole algorithm. There is no insight and no understanding in it. The model
                  improves because a few million tiny corrections in the right direction add up.
                </p>
              </>
            }
            math={
              <>
                <Eq note="One optimizer step, for a mini-batch B.">
                  <div className="space-y-1">
                    <div>ŷ = f(X<sub>B</sub>; θ)</div>
                    <div>L = (1/|B|) Σ loss(ŷ<sub>i</sub>, y<sub>i</sub>) + λ‖θ‖²</div>
                    <div>g = ∇<sub>θ</sub> L</div>
                    <div>θ ← θ − η · g</div>
                  </div>
                </Eq>
                <p>
                  The current gradient has norm {fmt(last?.gradNorm ?? 0, 4)} and the step size <M>η</M> is{' '}
                  {fmt(last?.lr ?? cfg.optim.lr, 4)}, so this update moved the parameter vector roughly{' '}
                  {fmt((last?.gradNorm ?? 0) * (last?.lr ?? cfg.optim.lr), 5)} in parameter space. With{' '}
                  {net.paramCount} parameters, that is a very small move in a very high-dimensional room.
                </p>
              </>
            }
            code={
              <Code>{`// engine/trainer.ts -- one real training step
const pass = this.net.forward(xb);           // predictions
const g    = this.net.backward(pass, yb);    // every dL/dw
const { norm } = this.opt.clip(groups);      // optional safety cap
this.opt.cfg.lr = baseLr * lrScale(this.cfg.schedule, this.step, this.totalSteps);
this.opt.tick();
for (let l = 0; l < this.net.layerCount; l++) {
  this.opt.step(\`W\${l}\`, this.net.W[l].data, g.dW[l].data);
  this.opt.step(\`b\${l}\`, this.net.b[l],       g.db[l]);
}`}</Code>
            }
          />
        </Panel>
      </div>
    </div>
  );
}

/* ================================================ tab 2: backprop steps */

interface GraphNode {
  v: Value;
  depth: number;
}

function buildExample(x1v: number, x2v: number, w1v: number, w2v: number, bv: number, yv: number) {
  const x1 = mkVal(x1v, 'x1');
  const x2 = mkVal(x2v, 'x2');
  const w1 = mkVal(w1v, 'w1');
  const w2 = mkVal(w2v, 'w2');
  const b = mkVal(bv, 'b');
  const y = mkVal(yv, 'y');
  const m1 = x1.mul(w1);
  m1.label = 'x1·w1';
  const m2 = x2.mul(w2);
  m2.label = 'x2·w2';
  const s = m1.add(m2);
  s.label = 'sum';
  const z = s.add(b);
  z.label = 'z';
  const a = z.tanh();
  a.label = 'a';
  const diff = a.sub(y);
  diff.label = 'a−y';
  const loss = diff.pow(2);
  loss.label = 'loss';
  return { loss, named: { x1, x2, w1, w2, b, y, m1, m2, s, z, a, diff } };
}

function BackpropTab() {
  const [x1, setX1] = useState(0.8);
  const [x2, setX2] = useState(-0.5);
  const [w1, setW1] = useState(0.9);
  const [w2, setW2] = useState(1.4);
  const [b, setB] = useState(-0.3);
  const [y, setY] = useState(1);
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);

  const built = useMemo(() => buildExample(x1, x2, w1, w2, b, y), [x1, x2, w1, w2, b, y]);
  const trace = useMemo(() => built.loss.backwardTrace(), [built]);

  useEffect(() => setStepIdx(0), [built]);

  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => {
      setStepIdx((i) => {
        if (i >= trace.length) {
          setPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, 620);
    return () => clearInterval(t);
  }, [playing, trace.length]);

  // Replay the trace up to the chosen step so displayed gradients are exact.
  const grads = useMemo(() => {
    const order = built.loss.topo();
    for (const n of order) n.grad = 0;
    built.loss.grad = 1;
    for (let i = 0; i < stepIdx; i++) {
      const node = trace[i].node;
      node.backwardStep();
    }
    return new Map(order.map((n) => [n.id, n.grad]));
  }, [built, trace, stepIdx]);

  const nodes = useMemo<GraphNode[]>(() => {
    const order = built.loss.topo();
    const depth = new Map<number, number>();
    for (const n of order) {
      const d = n.prev.length === 0 ? 0 : Math.max(...n.prev.map((p) => (depth.get(p.id) ?? 0) + 1));
      depth.set(n.id, d);
    }
    return order.map((n) => ({ v: n, depth: depth.get(n.id) ?? 0 }));
  }, [built]);

  const maxDepth = Math.max(...nodes.map((n) => n.depth));
  const byDepth = useMemo(() => {
    const m = new Map<number, GraphNode[]>();
    for (const n of nodes) {
      const arr = m.get(n.depth) ?? [];
      arr.push(n);
      m.set(n.depth, arr);
    }
    return m;
  }, [nodes]);

  const W = 760;
  const H = 300;
  const colX = (d: number) => 60 + (d * (W - 120)) / Math.max(1, maxDepth);
  const pos = new Map<number, { x: number; y: number }>();
  for (const [d, arr] of byDepth) {
    arr.forEach((n, i) => {
      const step = Math.min(58, (H - 60) / Math.max(1, arr.length - 1 || 1));
      const total = (arr.length - 1) * step;
      pos.set(n.v.id, { x: colX(d), y: H / 2 - total / 2 + i * step });
    });
  }

  const active = stepIdx > 0 && stepIdx <= trace.length ? trace[stepIdx - 1].node : null;
  const next = stepIdx < trace.length ? trace[stepIdx].node : null;
  const done = stepIdx >= trace.length;

  const explainStep = (n: Value | null) => {
    if (!n) return null;
    if (n.op === '+') return 'Addition passes the gradient through unchanged to both inputs. Each one contributed equally to the sum.';
    if (n.op === '*') return 'Multiplication sends each input the gradient scaled by the OTHER input. If one factor is large, the other one matters more.';
    if (n.op === 'tanh') return 'The gradient is multiplied by 1 − tanh(z)², which is near zero when the neuron is saturated. That is where vanishing gradients come from.';
    if (n.op === '^') return 'The power rule: multiply by the exponent times the base to one lower power.';
    if (n.op === '') return 'A leaf. Nothing feeds it, so the gradient stops here. For a weight, this number is exactly what the optimizer will use.';
    return `Applies the local derivative rule for ${n.op}.`;
  };

  return (
    <div className="space-y-4">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_320px]">
        <Panel
          title="The computation graph"
          subtitle="One neuron and a squared-error loss. Backpropagation walks this graph backwards."
          right={
            <Badge tone={done ? 'ok' : 'accent'}>
              {stepIdx} / {trace.length}
            </Badge>
          }
        >
          <svg width={W} height={H} style={{ display: 'block', maxWidth: '100%' }}>
            {nodes.map((n) =>
              n.v.prev.map((p, k) => {
                const a = pos.get(p.id)!;
                const bb = pos.get(n.v.id)!;
                const g = grads.get(p.id) ?? 0;
                const flowing = active?.id === n.v.id;
                return (
                  <g key={`${n.v.id}-${k}`}>
                    <path
                      d={`M${a.x + 22},${a.y} C${(a.x + bb.x) / 2},${a.y} ${(a.x + bb.x) / 2},${bb.y} ${bb.x - 22},${bb.y}`}
                      fill="none"
                      stroke={flowing ? 'var(--ok)' : 'var(--border-2)'}
                      strokeWidth={flowing ? 2.2 : 1}
                      className={flowing ? 'gb-flow' : undefined}
                    />
                    {Math.abs(g) > 1e-9 && (
                      <text
                        x={(a.x + bb.x) / 2}
                        y={(a.y + bb.y) / 2 - 3}
                        textAnchor="middle"
                        className="mono"
                        fontSize={8.5}
                        fill="var(--ok)"
                      >
                        {fmt(g, 3)}
                      </text>
                    )}
                  </g>
                );
              }),
            )}
            {nodes.map((n) => {
              const p = pos.get(n.v.id)!;
              const g = grads.get(n.v.id) ?? 0;
              const isActive = active?.id === n.v.id;
              const isNext = next?.id === n.v.id;
              const isLeaf = n.v.prev.length === 0;
              return (
                <g key={n.v.id}>
                  <rect
                    x={p.x - 24}
                    y={p.y - 15}
                    width={48}
                    height={30}
                    rx={7}
                    fill={isActive ? 'color-mix(in srgb, var(--ok) 22%, var(--panel-2))' : 'var(--panel-2)'}
                    stroke={isActive ? 'var(--ok)' : isNext ? 'var(--accent)' : isLeaf ? 'var(--border-2)' : 'var(--border)'}
                    strokeWidth={isActive || isNext ? 2 : 1}
                  />
                  <text x={p.x} y={p.y - 3} textAnchor="middle" className="mono" fontSize={8} fill="var(--text-3)">
                    {n.v.label || n.v.op || 'const'}
                  </text>
                  <text x={p.x} y={p.y + 8} textAnchor="middle" className="mono" fontSize={9.5} fill="var(--text)">
                    {fmt(n.v.data, 2)}
                  </text>
                  {Math.abs(g) > 1e-9 && (
                    <text x={p.x} y={p.y + 25} textAnchor="middle" className="mono" fontSize={8.5} fill="var(--ok)">
                      ∂L/∂ = {fmt(g, 3)}
                    </text>
                  )}
                </g>
              );
            })}
          </svg>

          <div className="mt-3 flex flex-wrap items-center gap-1.5">
            <Btn variant="primary" onClick={() => setPlaying((p) => !p)} disabled={done}>
              {playing ? 'Pause' : 'Play backward pass'}
            </Btn>
            <Btn onClick={() => setStepIdx((i) => Math.min(trace.length, i + 1))} disabled={done}>
              Next derivative
            </Btn>
            <Btn onClick={() => setStepIdx((i) => Math.max(0, i - 1))} disabled={stepIdx === 0}>
              Back
            </Btn>
            <Btn onClick={() => { setStepIdx(0); setPlaying(false); }}>Reset</Btn>
            <Btn onClick={() => setStepIdx(trace.length)} disabled={done}>
              Skip to end
            </Btn>
          </div>
        </Panel>

        <div className="space-y-4">
          <Panel title="Inputs and weights">
            <div className="space-y-3">
              <Slider label="x₁" info={V.probeInput} value={x1} min={-2} max={2} onChange={setX1} tone="signed" />
              <Slider label="x₂" info={V.probeInput} value={x2} min={-2} max={2} onChange={setX2} tone="signed" />
              <Slider label="w₁" info={V.weight1} value={w1} min={-2} max={2} onChange={setW1} tone="signed" />
              <Slider label="w₂" info={V.weight2} value={w2} min={-2} max={2} onChange={setW2} tone="signed" />
              <Slider label="bias" info={V.bias} value={b} min={-2} max={2} onChange={setB} tone="signed" />
              <Slider label="target y" info={V.targetY} value={y} min={-1} max={1} step={0.1} onChange={setY} tone="signed" />
            </div>
          </Panel>

          <Panel title={active ? `Step ${stepIdx}: ${active.label || active.op || 'leaf'}` : 'Not started'}>
            {active ? (
              <>
                <p className="text-[12.5px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                  {explainStep(active)}
                </p>
                <div className="mt-3 grid grid-cols-2 gap-3">
                  <Stat label="value" value={fmt(active.data, 4)} />
                  <Stat label="its gradient" value={fmt(grads.get(active.id) ?? 0, 4)} tone="ok" />
                </div>
              </>
            ) : (
              <p className="text-[12.5px]" style={{ color: 'var(--text-3)' }}>
                Press play. The pass starts at the loss, seeds its own gradient to 1, and works backwards.
              </p>
            )}
          </Panel>

          <Panel title="Final gradients" subtitle="What the optimizer would act on">
            <div className="space-y-1.5">
              {(['w1', 'w2', 'b'] as const).map((k) => {
                const node = built.named[k];
                const g = grads.get(node.id) ?? 0;
                return (
                  <div key={k} className="flex items-center justify-between">
                    <span className="mono text-[12px]" style={{ color: 'var(--text-2)' }}>
                      ∂L/∂{k}
                    </span>
                    <span className="mono tnum text-[12px]" style={{ color: done ? 'var(--ok)' : 'var(--text-3)' }}>
                      {fmt(g, 5)}
                      {done && Math.abs(g) > 1e-9 && (
                        <span style={{ color: 'var(--text-3)' }}> → {g > 0 ? 'decrease' : 'increase'}</span>
                      )}
                    </span>
                  </div>
                );
              })}
            </div>
            {done && (
              <div className="mt-3">
                <Callout tone="insight">
                  Every one of those numbers came from multiplying local derivatives along a path. No part of
                  the graph ever had to know what the others were doing. That locality is the only reason
                  training a model with billions of parameters is possible at all.
                </Callout>
              </div>
            )}
          </Panel>
        </div>
      </div>

      <Panel title="The chain rule, said three ways">
        <Depth
          plain={
            <>
              Each box asks one question of the box before it: if you had come out slightly bigger, how much
              worse would the final answer have been? The box answers, multiplies in its own little rule, and
              passes the question further back. By the time the question reaches a weight, the number attached
              to it is exactly how much that weight is to blame.
            </>
          }
          math={
            <>
              <Eq note="For a chain of operations, gradients multiply.">
                ∂L/∂w₁ = (∂L/∂a)(∂a/∂z)(∂z/∂m₁)(∂m₁/∂w₁) = 2(a−y) · (1−a²) · 1 · x₁
              </Eq>
              <p>
                Substituting the current values: 2({fmt(built.named.a.data, 3)} − {fmt(y, 2)}) ·
                (1 − {fmt(built.named.a.data, 3)}²) · {fmt(x1, 2)} ={' '}
                <span style={{ color: 'var(--ok)' }}>
                  {fmt(2 * (built.named.a.data - y) * (1 - built.named.a.data ** 2) * x1, 5)}
                </span>
                . That matches the value the graph produced, because it is the same computation.
              </p>
            </>
          }
          code={
            <Code>{`// engine/autograd.ts -- each operation stores its own local rule
mul(other) {
  const out = new Value(this.data * o.data, [this, o], '*');
  out.backwardStep = () => {
    this.grad += o.data * out.grad;   // d(a*b)/da = b
    o.grad    += this.data * out.grad;
  };
  return out;
}

backward() {
  const order = this.topo();          // dependencies first
  for (const v of order) v.grad = 0;
  this.grad = 1;                      // seed: dL/dL = 1
  for (let i = order.length - 1; i >= 0; i--) order[i].backwardStep();
}`}</Code>
          }
        />
      </Panel>
    </div>
  );
}

/* ==================================================== tab 3: failure lab */

interface FailurePreset {
  id: string;
  label: string;
  symptom: string;
  cause: string;
  fix: string;
  dataset: DatasetName;
  hidden: number[];
  act: ActName;
  cfg: Partial<TrainerConfig>;
  lr: number;
}

const FAILURES: FailurePreset[] = [
  {
    id: 'diverge',
    label: 'Learning rate far too high',
    symptom: 'Loss shoots upward, often to infinity, within a few dozen steps. The boundary thrashes.',
    cause:
      'Each step overshoots the minimum and lands somewhere worse than it started, so the next gradient is larger still. The process feeds itself.',
    fix: 'Lower the learning rate by a factor of ten, or switch on gradient clipping to cap the damage a single step can do.',
    dataset: 'circles',
    hidden: [8, 6],
    act: 'tanh',
    cfg: { epochs: 120 },
    lr: 2.5,
  },
  {
    id: 'crawl',
    label: 'Learning rate far too low',
    symptom: 'The loss curve is almost flat. Accuracy barely leaves its starting point.',
    cause: 'Every step is real and in the right direction, just microscopic. The run would converge, given millions of steps nobody has time for.',
    fix: 'Raise the learning rate until the loss falls quickly without becoming unstable. That band is usually narrower than people expect.',
    dataset: 'circles',
    hidden: [8, 6],
    act: 'tanh',
    cfg: { epochs: 200 },
    lr: 0.00015,
  },
  {
    id: 'capacity',
    label: 'No hidden layer on a curved problem',
    symptom: 'Loss falls a little, then stops. Accuracy plateaus near chance and refuses to move.',
    cause:
      'Without a hidden layer the model can only draw one straight line. The data is not separable by a straight line, so no setting of the weights can succeed. Training is working perfectly; the model simply cannot express the answer.',
    fix: 'Add a hidden layer. This is the exact limitation that stalled neural networks for years.',
    dataset: 'circles',
    hidden: [],
    act: 'tanh',
    cfg: { epochs: 260 },
    lr: 0.08,
  },
  {
    id: 'overfit',
    label: 'Memorising the noise',
    symptom: 'Training loss keeps sinking while validation loss bottoms out and then climbs.',
    cause:
      'The network has more capacity than the problem needs, so it starts fitting the random jitter in individual training points. Those points are not the pattern, so held-out performance decays.',
    fix: 'Add L2 regularisation, collect more data, or use a smaller network. Watch the two curves reconverge.',
    dataset: 'moons',
    hidden: [32, 32, 24],
    act: 'relu',
    cfg: { epochs: 500, batchSize: 8, l2: 0 },
    lr: 0.02,
  },
  {
    id: 'deadrelu',
    label: 'Dead ReLU units',
    symptom: 'Progress stalls part-way. The boundary has fewer bends than the network has neurons.',
    cause:
      'A large step pushed some neurons into the region where ReLU outputs zero for every input in the dataset. Their gradient is then exactly zero forever, so they can never come back. They are permanently switched off.',
    fix: 'Lower the learning rate, or switch to Leaky ReLU, which keeps a small slope on the negative side so a neuron can recover.',
    dataset: 'spirals',
    hidden: [12, 12],
    act: 'relu',
    cfg: { epochs: 300 },
    lr: 0.9,
  },
];

function FailureTab() {
  const [presetId, setPresetId] = useState(FAILURES[0].id);
  const [running, setRunning] = useState(false);
  const [, setTick] = useState(0);
  const preset = FAILURES.find((f) => f.id === presetId)!;

  const ds = useMemo(() => makeDataset(preset.dataset, 240, preset.id === 'overfit' ? 0.3 : 0.08, 42), [preset]);
  const split = useMemo(() => splitDataset(ds, 0.35, 7), [ds]);

  const trainerRef = useRef<Trainer | null>(null);
  const lastId = useRef('');
  if (!trainerRef.current || lastId.current !== presetId) {
    const cfg: TrainerConfig = {
      ...DEFAULT_TRAINER,
      ...preset.cfg,
      optim: { ...DEFAULT_TRAINER.optim, lr: preset.lr, name: 'sgd' },
    };
    const net = new MLP([2, ...preset.hidden, 1], preset.hidden.map(() => preset.act), 'binary', 11);
    trainerRef.current = new Trainer(net, split, cfg);
    lastId.current = presetId;
    setRunning(false);
  }
  const trainer = trainerRef.current;

  useFrameLoop(running, () => {
    const n = trainer.runSlice(9);
    if (n === 0) setRunning(false);
    setTick((t) => t + 1);
  });

  const field = useMemo(
    () => decisionField(trainer.net, RES, [-1.25, 1.25]),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [trainer.net, trainer.step, presetId],
  );
  const h = trainer.history;
  const last = trainer.latest();

  // Count neurons whose activation is zero across the entire dataset.
  const dead = useMemo(() => {
    if (preset.act !== 'relu' && preset.act !== 'leaky_relu') return null;
    const pass = trainer.net.forward(split.train.X);
    let total = 0;
    let deadCount = 0;
    for (let l = 1; l < pass.A.length - 1; l++) {
      const A = pass.A[l];
      for (let j = 0; j < A.cols; j++) {
        total++;
        let alive = false;
        for (let i = 0; i < A.rows; i++) {
          if (A.data[i * A.cols + j] > 1e-8) {
            alive = true;
            break;
          }
        }
        if (!alive) deadCount++;
      }
    }
    return { deadCount, total };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trainer.net, trainer.step, split, preset.act, presetId]);

  const diverged = !!last && (!Number.isFinite(last.loss) || last.loss > 8);

  return (
    <div className="space-y-4">
      <Panel
        title="Break it on purpose"
        subtitle="Every one of these is a real failure mode, reproduced with real training, not a mock-up."
      >
        <div className="flex flex-wrap gap-1.5">
          {FAILURES.map((f) => (
            <Btn key={f.id} size="sm" active={f.id === presetId} onClick={() => setPresetId(f.id)}>
              {f.label}
            </Btn>
          ))}
        </div>
      </Panel>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
        <div className="space-y-4">
          <Panel
            title={preset.label}
            right={
              <div className="flex gap-1.5">
                <Btn variant="primary" size="sm" onClick={() => setRunning((r) => !r)}>
                  {running ? 'Pause' : 'Run it'}
                </Btn>
                <Btn
                  size="sm"
                  onClick={() => {
                    trainer.reset();
                    setRunning(false);
                    setTick((t) => t + 1);
                  }}
                >
                  Reset
                </Btn>
              </div>
            }
          >
            <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_230px]">
              <LineChart
                series={[
                  { id: 'train', label: 'train loss', color: 'var(--accent)', points: h.map((m) => ({ x: m.step, y: m.loss })) },
                  { id: 'val', label: 'validation loss', color: 'var(--pos)', dashed: true, points: h.map((m) => ({ x: m.step, y: m.valLoss })) },
                ]}
                height={190}
                yLabel="loss"
                xLabel="step"
                logY
              />
              <FieldCanvas field={field} res={RES} size={225} points={split.train.points} />
            </div>

            <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Stat label="step" value={trainer.step.toLocaleString()} />
              <Stat
                label="train loss"
                value={last && !Number.isFinite(last.loss) ? 'NaN' : fmt(last?.loss ?? NaN, 4)}
                tone={diverged ? 'err' : undefined}
              />
              <Stat label="val loss" value={fmt(last?.valLoss ?? NaN, 4)} />
              <Stat label="val accuracy" value={fmtPct(last?.valAcc ?? NaN)} />
            </div>

            {dead && dead.deadCount > 0 && (
              <div className="mt-3">
                <Callout tone="err" title={`${dead.deadCount} of ${dead.total} hidden neurons are dead`}>
                  These units output exactly zero for every point in the training set. Their gradient is
                  therefore exactly zero too, so no amount of further training can revive them. That capacity
                  is gone for the rest of the run.
                </Callout>
              </div>
            )}
            {diverged && (
              <div className="mt-3">
                <Callout tone="err" title="The run has diverged">
                  The loss is no longer a finite useful number. Once weights reach this magnitude the model
                  produces saturated outputs everywhere and nothing recoverable remains. In a real run costing
                  real money, this is why people watch the loss curve obsessively for the first few hundred
                  steps.
                </Callout>
              </div>
            )}
          </Panel>
        </div>

        <div className="space-y-4">
          <Panel title="Symptom">
            <p className="text-[12.5px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
              {preset.symptom}
            </p>
          </Panel>
          <Panel title="Cause">
            <p className="text-[12.5px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
              {preset.cause}
            </p>
          </Panel>
          <Panel title="Fix">
            <p className="text-[12.5px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
              {preset.fix}
            </p>
          </Panel>
          <Panel title="This configuration">
            <div className="space-y-1 text-[11.5px]" style={{ color: 'var(--text-3)' }}>
              <div className="flex justify-between">
                <span>dataset</span>
                <span className="mono" style={{ color: 'var(--text-2)' }}>{DATASET_INFO[preset.dataset].label}</span>
              </div>
              <div className="flex justify-between">
                <span>hidden</span>
                <span className="mono" style={{ color: 'var(--text-2)' }}>
                  {preset.hidden.length ? preset.hidden.join(' · ') : 'none'}
                </span>
              </div>
              <div className="flex justify-between">
                <span>activation</span>
                <span className="mono" style={{ color: 'var(--text-2)' }}>{preset.act}</span>
              </div>
              <div className="flex justify-between">
                <span>learning rate</span>
                <span className="mono" style={{ color: 'var(--text-2)' }}>{preset.lr}</span>
              </div>
              <div className="flex justify-between">
                <span>parameters</span>
                <span className="mono" style={{ color: 'var(--text-2)' }}>{trainer.net.paramCount}</span>
              </div>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}

/* ================================================================ page == */

export default function TrainingLab() {
  const [tab, setTab] = useState<'train' | 'backprop' | 'fail'>('train');
  return (
    <div className="mx-auto max-w-[1400px] px-6 py-6">
      <div className="mb-5">
        <Segmented
          value={tab}
          onChange={setTab}
          options={[
            { id: 'train', label: '1 · Watch it learn' },
            { id: 'backprop', label: '2 · Backprop, one derivative at a time' },
            { id: 'fail', label: '3 · Make it fail' },
          ]}
        />
      </div>
      {tab === 'train' ? <TrainTab /> : tab === 'backprop' ? <BackpropTab /> : <FailureTab />}
    </div>
  );
}
