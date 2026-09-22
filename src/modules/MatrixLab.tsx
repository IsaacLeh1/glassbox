import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { KNOWN_MODELS, pipelineStages, summarise, type ArchConfig, type Stage } from '../engine/shapes';
import { buildLMTrainer } from '../engine/lmPresets';
import type { LMTrainer } from '../engine/lmTrainer';
import { softmax1d } from '../engine/tensor';
import { fmtCount, fmtFlops } from '../sim/cluster';
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
  ProgressBar,
  Segmented,
  Select,
  Slider,
  Stat,
  InfoDot,
  fmtInt,
  signedColor,
  signedTextColor,
} from '../ui/kit';
import { V } from '../content/varInfo';
import { BarMeter, HeatGrid } from '../ui/viz';

/* ================================================== tab 1: by hand ====== */

function EditableMatrix({
  rows,
  cols,
  data,
  onChange,
  highlightRow,
  highlightCol,
  label,
  max = 9,
}: {
  rows: number;
  cols: number;
  data: number[];
  onChange: (i: number, v: number) => void;
  highlightRow?: number;
  highlightCol?: number;
  label: string;
  max?: number;
}) {
  return (
    <div>
      <div className="mb-1 flex items-baseline gap-2">
        <Label>{label}</Label>
        <span className="mono text-[10px]" style={{ color: 'var(--text-3)' }}>
          {rows}×{cols}
        </span>
      </div>
      <div
        className="inline-grid gap-[3px] rounded-lg p-2"
        style={{ gridTemplateColumns: `repeat(${cols}, 40px)`, background: 'var(--bg-2)', border: '1px solid var(--border)' }}
      >
        {Array.from({ length: rows * cols }, (_, k) => {
          const i = Math.floor(k / cols);
          const j = k % cols;
          const on = highlightRow === i || highlightCol === j;
          return (
            <input
              key={k}
              type="number"
              value={data[k]}
              onChange={(e) => onChange(k, parseFloat(e.target.value) || 0)}
              className="mono tnum"
              style={{
                width: 40,
                padding: '3px 2px',
                textAlign: 'center',
                fontSize: 11.5,
                background: on ? 'color-mix(in srgb, var(--accent) 16%, var(--panel-2))' : 'var(--panel-2)',
                border: `1px solid ${on ? 'var(--accent)' : 'var(--border)'}`,
                color: signedTextColor(data[k]),
              }}
              min={-max}
              max={max}
            />
          );
        })}
      </div>
    </div>
  );
}

function ByHandTab() {
  const [ar, setAr] = useState(2);
  const [ac, setAc] = useState(3);
  const [bc, setBc] = useState(2);
  const [A, setA] = useState<number[]>([1, 2, 3, 4, 5, 6]);
  const [B, setB] = useState<number[]>([7, 8, 9, 10, 11, 12]);
  const [cell, setCell] = useState<{ i: number; j: number } | null>({ i: 0, j: 0 });
  const [playing, setPlaying] = useState(false);

  const resize = useCallback(
    (nar: number, nac: number, nbc: number) => {
      setA((old) => Array.from({ length: nar * nac }, (_, k) => old[k] ?? ((k % 7) + 1)));
      setB((old) => Array.from({ length: nac * nbc }, (_, k) => old[k] ?? ((k % 5) + 2)));
      setAr(nar);
      setAc(nac);
      setBc(nbc);
      setCell({ i: 0, j: 0 });
    },
    [],
  );

  const C = useMemo(() => {
    const out = new Array(ar * bc).fill(0);
    for (let i = 0; i < ar; i++) {
      for (let j = 0; j < bc; j++) {
        let s = 0;
        for (let t = 0; t < ac; t++) s += (A[i * ac + t] ?? 0) * (B[t * bc + j] ?? 0);
        out[i * bc + j] = s;
      }
    }
    return out;
  }, [A, B, ar, ac, bc]);

  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => {
      setCell((c) => {
        if (!c) return { i: 0, j: 0 };
        const next = c.j + 1 < bc ? { i: c.i, j: c.j + 1 } : { i: c.i + 1, j: 0 };
        if (next.i >= ar) {
          setPlaying(false);
          return c;
        }
        return next;
      });
    }, 700);
    return () => clearInterval(t);
  }, [playing, ar, bc]);

  const terms =
    cell != null
      ? Array.from({ length: ac }, (_, t) => ({
          a: A[cell.i * ac + t] ?? 0,
          b: B[t * bc + cell.j] ?? 0,
        }))
      : [];
  const sum = terms.reduce((s, t) => s + t.a * t.b, 0);

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
      <div className="space-y-4">
        <Panel
          title="One rule, applied everywhere"
          subtitle="Cell (i, j) of the answer is row i of the left matrix paired off against column j of the right one."
          right={
            <div className="flex gap-1.5">
              <Btn size="sm" variant="primary" onClick={() => { setCell({ i: 0, j: 0 }); setPlaying(true); }}>
                Walk through it
              </Btn>
              <Btn size="sm" onClick={() => setPlaying(false)} disabled={!playing}>
                Stop
              </Btn>
            </div>
          }
        >
          <div className="flex flex-wrap items-start gap-5">
            <EditableMatrix
              rows={ar}
              cols={ac}
              data={A}
              onChange={(k, v) => setA((a) => a.map((x, i) => (i === k ? v : x)))}
              highlightRow={cell?.i}
              label="A"
            />
            <div className="pt-7 text-[18px]" style={{ color: 'var(--text-3)' }}>
              ×
            </div>
            <EditableMatrix
              rows={ac}
              cols={bc}
              data={B}
              onChange={(k, v) => setB((b) => b.map((x, i) => (i === k ? v : x)))}
              highlightCol={cell?.j}
              label="B"
            />
            <div className="pt-7 text-[18px]" style={{ color: 'var(--text-3)' }}>
              =
            </div>
            <div>
              <div className="mb-1 flex items-baseline gap-2">
                <Label>C = A × B</Label>
                <span className="mono text-[10px]" style={{ color: 'var(--text-3)' }}>
                  {ar}×{bc}
                </span>
              </div>
              <div
                className="inline-grid gap-[3px] rounded-lg p-2"
                style={{ gridTemplateColumns: `repeat(${bc}, 52px)`, background: 'var(--bg-2)', border: '1px solid var(--border)' }}
              >
                {C.map((v, k) => {
                  const i = Math.floor(k / bc);
                  const j = k % bc;
                  const on = cell?.i === i && cell?.j === j;
                  return (
                    <button
                      key={k}
                      onClick={() => { setCell({ i, j }); setPlaying(false); }}
                      className="mono tnum focus-ring cursor-pointer rounded"
                      style={{
                        padding: '4px 2px',
                        fontSize: 11.5,
                        background: on ? 'color-mix(in srgb, var(--accent) 24%, var(--panel-2))' : 'var(--panel-2)',
                        border: `1px solid ${on ? 'var(--accent)' : 'var(--border)'}`,
                        color: 'var(--text)',
                      }}
                    >
                      {v}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {cell && (
            <Eq
              note={`Every one of the ${ar * bc} output cells is computed exactly this way, and they are all independent of each other. That independence is why a GPU can do thousands of them at once.`}
            >
              <div className="flex flex-wrap items-center gap-1.5">
                <span style={{ color: 'var(--text-3)' }}>
                  C[{cell.i}][{cell.j}] =
                </span>
                {terms.map((t, k) => (
                  <span key={k} className="flex items-center gap-1.5">
                    {k > 0 && <span style={{ color: 'var(--text-3)' }}>+</span>}
                    <span style={{ color: 'var(--accent)' }}>{t.a}</span>
                    <span style={{ color: 'var(--text-3)' }}>·</span>
                    <span style={{ color: 'var(--pos)' }}>{t.b}</span>
                  </span>
                ))}
                <span style={{ color: 'var(--text-3)' }}>=</span>
                <span style={{ color: 'var(--text)', fontWeight: 600 }}>{sum}</span>
              </div>
            </Eq>
          )}
        </Panel>

        <Panel title="Shapes are the whole discipline">
          <div className="mb-4 grid gap-3 sm:grid-cols-3">
            <Field label="rows of A" info={V.matRowsA}>
              <Slider label="" value={ar} min={1} max={5} step={1} format={String} onChange={(v) => resize(v, ac, bc)} />
            </Field>
            <Field label="shared dimension" info={V.matShared}>
              <Slider label="" value={ac} min={1} max={5} step={1} format={String} onChange={(v) => resize(ar, v, bc)} />
            </Field>
            <Field label="columns of B" info={V.matColsB}>
              <Slider label="" value={bc} min={1} max={5} step={1} format={String} onChange={(v) => resize(ar, ac, v)} />
            </Field>
          </div>

          <div
            className="mono rounded-lg p-3 text-center text-[13px]"
            style={{ background: 'var(--bg-2)', border: '1px solid var(--border)' }}
          >
            <span style={{ color: 'var(--accent)' }}>({ar} × {ac})</span>
            <span style={{ color: 'var(--text-3)' }}> × </span>
            <span style={{ color: 'var(--pos)' }}>({ac} × {bc})</span>
            <span style={{ color: 'var(--text-3)' }}> = </span>
            <span>({ar} × {bc})</span>
            <div className="mt-1.5 text-[10.5px]" style={{ color: 'var(--text-3)' }}>
              the inner numbers must match and then they vanish; the outer two become the answer
            </div>
          </div>

          <div className="mt-3 grid grid-cols-3 gap-3">
            <Stat label="multiplications" value={ar * ac * bc} />
            <Stat label="additions" value={ar * bc * (ac - 1)} />
            <Stat label="FLOPs" value={2 * ar * ac * bc} hint="Counted as one multiply plus one add per term." />
          </div>

          <div className="mt-4">
            <Depth
              plain={
                <>
                  <p className="mb-2">
                    A matrix is a grid of numbers, and multiplying two of them is one rule repeated: slide a row
                    across a column, multiply the pairs you meet, add the results. That single number goes in
                    one cell of the answer.
                  </p>
                  <p>
                    The reason this matters for AI is that neural networks are almost nothing but this. The
                    inner dimensions have to agree, which is why so much of the work of building a model is
                    making sure shapes line up. Get one wrong and nothing runs at all.
                  </p>
                </>
              }
              math={
                <>
                  <Eq>C₍ij₎ = Σ₍k₎ A₍ik₎ B₍kj₎, for A ∈ ℝ^(m×k), B ∈ ℝ^(k×n), C ∈ ℝ^(m×n)</Eq>
                  <p>
                    The cost is <M>O(mkn)</M> multiply-accumulates, counted as <M>2mkn</M> FLOPs. Matrix
                    multiplication is associative but not commutative: <M>AB ≠ BA</M> in general, and usually
                    the shapes do not even permit the reverse.
                  </p>
                </>
              }
              code={
                <Code>{`// engine/tensor.ts -- ordered so the inner loop walks memory in sequence
for (let i = 0; i < n; i++) {
  for (let t = 0; t < k; t++) {
    const av = a.data[i * k + t];
    if (av === 0) continue;          // skip work on zeros
    for (let j = 0; j < p; j++)
      out.data[i * p + j] += av * b.data[t * p + j];
  }
}`}</Code>
              }
            />
          </div>
        </Panel>
      </div>

      <div className="space-y-4">
        <Panel title="Why this operation and not another">
          <ul className="space-y-2.5 text-[12.5px]" style={{ color: 'var(--text-2)' }}>
            <li>
              <strong style={{ color: 'var(--text)' }}>Every output cell is independent.</strong> Nothing has to
              wait for anything else, so the work splits across thousands of cores perfectly.
            </li>
            <li>
              <strong style={{ color: 'var(--text)' }}>It reuses data.</strong> Each row of A is used for every
              column of B, so a value fetched once from memory is used many times. Memory is the slow part.
            </li>
            <li>
              <strong style={{ color: 'var(--text)' }}>Hardware is built for it.</strong> A modern accelerator
              has dedicated circuits that do nothing but small matrix multiplies. That is what a tensor core is.
            </li>
            <li>
              <strong style={{ color: 'var(--text)' }}>It composes.</strong> Stack matrix multiplies with a
              nonlinearity between them and you can approximate essentially any function.
            </li>
          </ul>
        </Panel>

        <Callout tone="insight">
          One caution: a stack of matrix multiplies with nothing in between collapses. A×B×C is just one matrix.
          Without a nonlinearity between layers, a hundred-layer network is mathematically identical to a
          single layer. That is why the activation functions in module 01 are not decoration.
        </Callout>
      </div>
    </div>
  );
}

/* ============================================ tab 2: words as matrices == */

/**
 * A hard cap on the sentence. Past a couple of hundred characters the matrices
 * stop fitting on screen and the attention grid becomes an unreadable smear,
 * which defeats the point of the panel.
 */
const MAX_CHARS = 180;

const EXAMPLES = [
  'the quiet fox sleeps',
  'the cat sat on the mat and then the cat left',
  'she said the red bird flew over the old bridge',
];

function WordsTab({ trainer }: { trainer: LMTrainer }) {
  const [text, setText] = useState('the quiet fox sleeps');
  const [stageIdx, setStageIdx] = useState(0);
  const [running, setRunning] = useState(false);
  const [selectedToken, setSelectedToken] = useState(0);

  const cfg = trainer.model.cfg;
  const allIds = useMemo(() => trainer.tok.encode(text), [text, trainer]);
  const ids = useMemo(() => allIds.slice(0, cfg.blockSize), [allIds, cfg.blockSize]);
  const T = ids.length;
  const overflow = allIds.length - ids.length;
  const labels = ids.map((id) => trainer.tok.display(id) || '·');

  const arch: ArchConfig = {
    vocab: cfg.vocab,
    dModel: cfg.dModel,
    nHeads: cfg.nHeads,
    nLayers: cfg.nLayers,
    dFF: cfg.dFF,
    ctx: cfg.blockSize,
  };
  const stages = useMemo(() => pipelineStages(arch, Math.max(1, T)), [arch, T]);
  const trace = useMemo(() => (T > 0 ? trainer.model.forward(ids).trace : null), [ids, trainer, T]);

  useEffect(() => {
    setStageIdx(0);
    setRunning(false);
  }, [text]);

  useEffect(() => {
    if (!running) return;
    const t = setTimeout(() => {
      setStageIdx((i) => {
        if (i >= stages.length - 1) {
          setRunning(false);
          return i;
        }
        return i + 1;
      });
    }, 520);
    return () => clearTimeout(t);
  }, [running, stageIdx, stages.length]);

  const stage: Stage | undefined = stages[Math.min(stageIdx, stages.length - 1)];
  const cumulativeFlops = stages
    .slice(0, stageIdx + 1)
    .reduce((n, s) => n + s.flops * (s.perLayer ? cfg.nLayers : 1), 0);
  const totalFlops = stages.reduce((n, s) => n + s.flops * (s.perLayer ? cfg.nLayers : 1), 0);
  const done = stageIdx >= stages.length - 1;

  const logitsRow: number[] = [];
  if (trace && T > 0) {
    for (let j = 0; j < cfg.vocab; j++) logitsRow.push(trace.logits.data[(T - 1) * cfg.vocab + j]);
  }
  const probs = logitsRow.length ? softmax1d(logitsRow, 1) : [];
  const top = probs.map((p, id) => ({ p, id })).sort((a, b) => b.p - a.p).slice(0, 6);

  const selId = ids[Math.min(selectedToken, Math.max(0, T - 1))] ?? 0;

  return (
    <div className="space-y-4">
      <Panel
        title="Your sentence, in the only language the model has"
        subtitle="Text goes in as characters and comes out the other end as a grid of numbers. Here is the whole conversion."
        right={
          <div className="flex gap-1.5">
            <Btn size="sm" variant="primary" onClick={() => { setStageIdx(0); setRunning(true); }} disabled={T === 0}>
              Run it through
            </Btn>
            <Btn size="sm" onClick={() => setRunning(false)} disabled={!running}>
              Pause
            </Btn>
            <Btn size="sm" onClick={() => { setStageIdx(stages.length - 1); setRunning(false); }}>
              Skip to output
            </Btn>
          </div>
        }
      >
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="mono"
          maxLength={MAX_CHARS}
          placeholder="type any sentence and watch it become numbers"
          aria-label="Your sentence"
        />

        <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[11px]" style={{ color: 'var(--text-3)' }}>
          <span>
            <span className="mono" style={{ color: overflow > 0 ? 'var(--warn)' : 'var(--text-2)' }}>
              {allIds.length}
            </span>{' '}
            tokens, window holds <span className="mono">{cfg.blockSize}</span>
          </span>
          <span>
            <span className="mono">{text.length}</span> / {MAX_CHARS} characters
          </span>
          <span className="flex flex-wrap items-center gap-1">
            try:
            {EXAMPLES.map((e) => (
              <button
                key={e}
                onClick={() => setText(e)}
                className="focus-ring cursor-pointer rounded px-1.5 py-0.5"
                style={{ background: 'var(--panel-2)', border: '1px solid var(--border)', color: 'var(--text-2)' }}
              >
                {e.length > 26 ? `${e.slice(0, 26)}…` : e}
              </button>
            ))}
          </span>
        </div>

        {overflow > 0 && (
          <div className="mt-3">
            <Callout tone="warn" title={`The last ${overflow} token${overflow === 1 ? '' : 's'} will not be seen`}>
              This model can only hold {cfg.blockSize} tokens at once, so everything past that is simply cut
              off before it ever reaches the network. It is not summarised or remembered; it does not exist as
              far as the model is concerned. Shorten the sentence, or choose a larger model above, and the
              tokens will reappear.
            </Callout>
          </div>
        )}

        {T > 0 && T <= 3 && (
          <div className="mt-3">
            <Callout tone="info" title="Short sentences are easier to follow here">
              With only {T} token{T === 1 ? '' : 's'} there is very little for attention to do, since each
              token can only look at the ones before it. Six to twelve words usually shows the most
              interesting pattern without the grids becoming unreadable.
            </Callout>
          </div>
        )}

        <div className="mt-3 grid gap-4 md:grid-cols-[minmax(0,1fr)_260px]">
          <div>
            <Label>step 1 — cut into tokens and count them</Label>
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {labels.map((t, i) => (
                <button
                  key={i}
                  onClick={() => setSelectedToken(i)}
                  className="focus-ring cursor-pointer rounded px-2 py-1 text-left"
                  style={{
                    background: i === selectedToken ? 'color-mix(in srgb, var(--accent) 18%, var(--panel-2))' : 'var(--panel-2)',
                    border: `1px solid ${i === selectedToken ? 'var(--accent)' : 'var(--border)'}`,
                  }}
                >
                  <div className="mono text-[11.5px]" style={{ color: 'var(--text)' }}>
                    {t}
                  </div>
                  <div className="mono text-[9.5px]" style={{ color: 'var(--text-3)' }}>
                    pos {i} · id {ids[i]}
                  </div>
                </button>
              ))}
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Stat label="tokens (T)" value={T} tone="accent" hint="Every shape below is written in terms of this number." />
            <Stat label="vocabulary (V)" value={cfg.vocab} />
            <Stat label="model width (d)" value={cfg.dModel} />
            <Stat label="layers" value={cfg.nLayers} />
          </div>
        </div>
      </Panel>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,420px)]">
        <Panel
          title="Step 2 — a token id becomes a row of numbers"
          subtitle="This is the step people usually skip, and it is the one that explains what an embedding is."
        >
          {T > 0 && trace ? (
            <div className="space-y-4">
              <div>
                <div className="mb-1 flex items-baseline gap-2">
                  <Label>one-hot matrix</Label>
                  <span className="mono text-[10px]" style={{ color: 'var(--text-3)' }}>
                    {T} × {cfg.vocab}
                  </span>
                </div>
                <div className="overflow-x-auto">
                  <HeatGrid
                    rows={T}
                    cols={cfg.vocab}
                    get={(i, j) => (ids[i] === j ? 1 : 0)}
                    mode="heat"
                    max={1}
                    cell={3}
                    gap={0}
                  />
                </div>
                <p className="mt-1 text-[11px]" style={{ color: 'var(--text-3)' }}>
                  One row per token, one column per vocabulary entry. Exactly {T} cells out of{' '}
                  {fmtInt(T * cfg.vocab)} are non-zero.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <span className="mono text-[13px]" style={{ color: 'var(--text-3)' }}>
                  ×
                </span>
                <div>
                  <div className="mb-1 flex items-baseline gap-2">
                    <Label>embedding table E</Label>
                    <span className="mono text-[10px]" style={{ color: 'var(--text-3)' }}>
                      {cfg.vocab} × {cfg.dModel}
                    </span>
                  </div>
                  <div style={{ maxHeight: 150, overflowY: 'auto' }}>
                    <HeatGrid
                      rows={cfg.vocab}
                      cols={cfg.dModel}
                      get={(i, j) => trainer.model.tok.M.data[i * cfg.dModel + j]}
                      cell={3}
                      gap={0}
                      highlight={{ i: selId, j: 0 }}
                    />
                  </div>
                </div>
              </div>

              <div>
                <div className="mb-1 flex items-baseline gap-2">
                  <Label>= the embedded sentence</Label>
                  <span className="mono text-[10px]" style={{ color: 'var(--text-3)' }}>
                    {T} × {cfg.dModel}
                  </span>
                </div>
                <div className="overflow-x-auto">
                  <HeatGrid
                    rows={T}
                    cols={cfg.dModel}
                    get={(i, j) => trace.tokenEmb.data[i * cfg.dModel + j]}
                    cell={7}
                    gap={1}
                    highlight={{ i: Math.min(selectedToken, T - 1), j: 0 }}
                  />
                </div>
              </div>

              <div>
                <Label>
                  the vector for "{labels[Math.min(selectedToken, T - 1)]}" — {cfg.dModel} numbers, and that is
                  the entire meaning of that token to this model
                </Label>
                <div className="mono mt-1.5 flex flex-wrap gap-1 text-[10px]">
                  {Array.from({ length: Math.min(cfg.dModel, 32) }, (_, j) => {
                    const v = trace.tokenEmb.data[Math.min(selectedToken, T - 1) * cfg.dModel + j];
                    return (
                      <span
                        key={j}
                        className="rounded px-1 py-[1px]"
                        style={{ background: signedColor(v, 0.4, 0.9), color: 'var(--text)' }}
                      >
                        {v.toFixed(2)}
                      </span>
                    );
                  })}
                  {cfg.dModel > 32 && <span style={{ color: 'var(--text-3)' }}>+{cfg.dModel - 32} more</span>}
                </div>
              </div>

              <Callout tone="insight" title="The trick nobody mentions">
                Multiplying a one-hot row by the embedding table just picks out one row of that table, because
                every other term is multiplied by zero. So real code never builds the one-hot matrix at all, it
                just indexes. The conceptual operation is a matrix multiply; the implementation is a lookup.
                Both give identical answers.
              </Callout>
            </div>
          ) : (
            <p className="text-[12.5px]" style={{ color: 'var(--text-3)' }}>
              Type something above.
            </p>
          )}
        </Panel>

        <div className="space-y-4">
          <Panel
            title="Step 3 — the chain of matrices"
            subtitle="Each row is one operation. Watch the shapes transform."
            right={<Badge tone={done ? 'ok' : 'accent'}>{stageIdx + 1} / {stages.length}</Badge>}
          >
            <ProgressBar value={(stageIdx + 1) / stages.length} tone={done ? 'ok' : 'accent'} />
            <div className="mt-3 max-h-[300px] space-y-0.5 overflow-y-auto pr-1">
              {stages.map((s, i) => {
                const active = i === stageIdx;
                const passed = i < stageIdx;
                return (
                  <button
                    key={s.id}
                    onClick={() => { setStageIdx(i); setRunning(false); }}
                    className="focus-ring block w-full cursor-pointer rounded px-2 py-1.5 text-left transition-colors"
                    style={{
                      background: active ? 'color-mix(in srgb, var(--accent) 12%, transparent)' : 'transparent',
                      border: `1px solid ${active ? 'var(--accent)' : 'transparent'}`,
                      opacity: passed || active ? 1 : 0.42,
                    }}
                  >
                    <div className="flex items-baseline justify-between gap-2">
                      <span className="truncate text-[11.5px]" style={{ color: active ? 'var(--text)' : 'var(--text-2)' }}>
                        {s.label}
                      </span>
                      {s.perLayer && (
                        <span className="mono shrink-0 text-[9px]" style={{ color: 'var(--text-3)' }}>
                          ×{cfg.nLayers}
                        </span>
                      )}
                    </div>
                    <div className="mono text-[9.5px]" style={{ color: 'var(--text-3)' }}>
                      ({s.a[0]}×{s.a[1]}){s.b ? ` × (${s.b[0]}×${s.b[1]})` : ''} → ({s.out[0]}×{s.out[1]})
                    </div>
                  </button>
                );
              })}
            </div>
          </Panel>

          {stage && (
            <Panel title={stage.label} subtitle={stage.group}>
              <p className="mb-3 text-[12.5px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                {stage.detail}
              </p>
              <div
                className="mono rounded-lg p-2.5 text-center text-[12px]"
                style={{ background: 'var(--bg-2)', border: '1px solid var(--border)' }}
              >
                <span style={{ color: 'var(--accent)' }}>
                  {stage.aLabel} ({stage.a[0]}×{stage.a[1]})
                </span>
                {stage.b && (
                  <>
                    <span style={{ color: 'var(--text-3)' }}> × </span>
                    <span style={{ color: 'var(--pos)' }}>
                      {stage.bLabel} ({stage.b[0]}×{stage.b[1]})
                    </span>
                  </>
                )}
                <span style={{ color: 'var(--text-3)' }}> → </span>
                <span>
                  {stage.outLabel} ({stage.out[0]}×{stage.out[1]})
                </span>
              </div>
              <div className="mt-3 grid grid-cols-3 gap-3">
                <Stat label="FLOPs here" value={fmtFlops(stage.flops * (stage.perLayer ? cfg.nLayers : 1))} />
                <Stat label="parameters" value={fmtCount(stage.params * (stage.perLayer ? cfg.nLayers : 1))} />
                <Stat label="running total" value={fmtFlops(cumulativeFlops)} tone="accent" />
              </div>
            </Panel>
          )}

          <Panel title="Output" subtitle={done ? 'The last row of the logits matrix, turned into probabilities' : 'Run the pipeline to the end'}>
            {done && top.length > 0 ? (
              <div className="space-y-1.5">
                {top.map((t) => (
                  <div key={t.id} className="flex items-center gap-2">
                    <span className="mono w-[70px] shrink-0 truncate text-[11.5px]" style={{ color: 'var(--text-2)' }}>
                      {trainer.tok.display(t.id) || '·'}
                    </span>
                    <div className="flex-1">
                      <BarMeter value={t.p} max={top[0].p} color="var(--accent)" height={7} />
                    </div>
                    <span className="mono tnum w-[46px] shrink-0 text-right text-[10.5px]" style={{ color: 'var(--text-3)' }}>
                      {(t.p * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
                <p className="pt-2 text-[11.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                  {fmtFlops(totalFlops)} of arithmetic across {stages.length} stages, for {T} tokens, and the
                  entire result is this one list of probabilities. Pick one, append it, and the whole pipeline
                  runs again from the top.
                </p>
              </div>
            ) : (
              <p className="text-[12.5px]" style={{ color: 'var(--text-3)' }}>
                {running ? 'Running…' : 'Press "Run it through" to watch the sentence move stage by stage.'}
              </p>
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}

/* ================================================== tab 3: scale ======== */

function ScaleTab({ trainer }: { trainer: LMTrainer }) {
  const [T, setT] = useState(512);
  const cfg = trainer.model.cfg;

  const mine: ArchConfig = {
    vocab: cfg.vocab,
    dModel: cfg.dModel,
    nHeads: cfg.nHeads,
    nLayers: cfg.nLayers,
    dFF: cfg.dFF,
    ctx: cfg.blockSize,
  };

  const rows = useMemo(() => {
    const all = [
      { id: 'yours', label: 'The model in this tab', note: 'Trained live in your browser.', cfg: mine, tied: false, reportedParams: fmtCount(trainer.model.paramCount) },
      ...KNOWN_MODELS,
    ];
    return all.map((m) => {
      const t = Math.min(T, m.cfg.ctx);
      const s = summarise(m.cfg, t, m.tied);
      return { ...m, t, s };
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [T, trainer, cfg.vocab, cfg.dModel, cfg.nHeads, cfg.nLayers, cfg.dFF, cfg.blockSize]);

  const maxParams = Math.max(...rows.map((r) => r.s.params));

  return (
    <div className="space-y-4">
      <Panel
        title="The same shapes, at every scale"
        subtitle="Nothing in the list below changes the operations. Only the numbers in the brackets change."
      >
        <div className="mb-4 max-w-[380px]">
          <Slider
            label="sequence length for the comparison" info={V.seqLength}
            value={T}
            min={16}
            max={4096}
            step={16}
            format={(x) => `${Math.round(x)} tokens`}
            onChange={(x) => setT(Math.round(x))}
            hint="Clamped to each model's context window."
          />
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-[11.5px]" style={{ borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ color: 'var(--text-3)' }}>
                {['model', 'd_model', 'layers', 'heads', 'vocab', 'context', 'parameters', 'biggest matrix', 'FLOPs / token', 'KV cache / token'].map((h) => (
                  <th key={h} className="px-2 py-1.5 text-left font-medium" style={{ borderBottom: '1px solid var(--border)' }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const biggest = Math.max(r.cfg.vocab * r.cfg.dModel, r.cfg.dModel * r.cfg.dFF);
                const biggestLabel =
                  r.cfg.vocab * r.cfg.dModel >= r.cfg.dModel * r.cfg.dFF
                    ? `${fmtCount(r.cfg.vocab)} × ${r.cfg.dModel}`
                    : `${r.cfg.dModel} × ${fmtCount(r.cfg.dFF)}`;
                const isMine = r.id === 'yours';
                return (
                  <tr key={r.id} style={{ borderBottom: '1px solid var(--border)' }}>
                    <td className="px-2 py-2">
                      <div style={{ color: isMine ? 'var(--accent)' : 'var(--text)', fontWeight: isMine ? 600 : 400 }}>
                        {r.label}
                      </div>
                      <div className="mt-0.5" style={{ color: 'var(--text-3)', maxWidth: 220 }}>
                        {r.note}
                      </div>
                      <div className="mt-1" style={{ width: 200 }}>
                        <BarMeter
                          value={Math.log10(r.s.params)}
                          max={Math.log10(maxParams)}
                          color={isMine ? 'var(--accent)' : 'var(--pos)'}
                          height={4}
                        />
                        <span className="text-[9px]" style={{ color: 'var(--text-3)' }}>
                          log scale
                        </span>
                      </div>
                    </td>
                    <td className="mono px-2 py-2">{r.cfg.dModel}</td>
                    <td className="mono px-2 py-2">{r.cfg.nLayers}</td>
                    <td className="mono px-2 py-2">{r.cfg.nHeads}</td>
                    <td className="mono px-2 py-2">{fmtCount(r.cfg.vocab)}</td>
                    <td className="mono px-2 py-2">{fmtCount(r.cfg.ctx)}</td>
                    <td className="mono px-2 py-2" style={{ color: 'var(--text)' }}>
                      {fmtCount(r.s.params)}
                      <div className="text-[9.5px]" style={{ color: 'var(--text-3)' }}>
                        reported {r.reportedParams}
                      </div>
                    </td>
                    <td className="mono px-2 py-2">
                      {biggestLabel}
                      <div className="text-[9.5px]" style={{ color: 'var(--text-3)' }}>
                        {fmtCount(biggest)} numbers
                      </div>
                    </td>
                    <td className="mono px-2 py-2">{fmtFlops(r.s.flopsPerToken)}</td>
                    <td className="mono px-2 py-2">{(r.s.kvCacheBytesPerToken / 1024).toFixed(1)} KB</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-[10.5px]" style={{ color: 'var(--text-3)' }}>
          Parameter counts are computed from the published dimensions, so they land near but not exactly on the
          commonly quoted figures — those differ on details like tied embeddings, gated feed-forward layers and
          grouped-query attention. The point of the table is the ratios, not the last digit.
        </p>
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Where the parameters actually sit">
          <div className="space-y-3">
            {rows.map((r) => {
              const total = r.s.params;
              const seg = [
                { label: 'embeddings', v: r.s.embedParams, c: 'var(--accent)' },
                { label: 'attention', v: r.s.attnParams, c: 'var(--pos)' },
                { label: 'feed-forward', v: r.s.ffParams, c: 'var(--ok)' },
                { label: 'unembedding', v: r.s.headParams, c: 'var(--warn)' },
              ];
              return (
                <div key={r.id}>
                  <div className="mb-1 flex justify-between text-[11px]">
                    <span style={{ color: r.id === 'yours' ? 'var(--accent)' : 'var(--text-2)' }}>{r.label}</span>
                    <span className="mono" style={{ color: 'var(--text-3)' }}>
                      {fmtCount(total)}
                    </span>
                  </div>
                  <div className="flex h-2.5 overflow-hidden rounded-full" style={{ background: 'var(--panel-3)' }}>
                    {seg.map((s) => (
                      <div key={s.label} style={{ width: `${(s.v / total) * 100}%`, background: s.c }} title={`${s.label}: ${fmtCount(s.v)}`} />
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
          <div className="mt-3 flex flex-wrap gap-3 text-[10px]" style={{ color: 'var(--text-3)' }}>
            {[
              ['embeddings', 'var(--accent)'],
              ['attention', 'var(--pos)'],
              ['feed-forward', 'var(--ok)'],
              ['unembedding', 'var(--warn)'],
            ].map(([l, c]) => (
              <span key={l} className="inline-flex items-center gap-1">
                <span style={{ width: 8, height: 8, borderRadius: 2, background: c, display: 'inline-block' }} />
                {l}
              </span>
            ))}
          </div>
          <p className="mt-3 text-[11.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
            In small models the embedding table dominates, because the vocabulary is large relative to
            everything else. As models grow, the feed-forward layers take over completely: they are typically
            about two thirds of a large model&apos;s weights.
          </p>
        </Panel>

        <Panel title="What actually changes with scale">
          <Depth
            plain={
              <>
                <p className="mb-2">
                  A large model is not a different kind of thing. It has the same tokenizer step, the same
                  embedding lookup, the same attention, the same feed-forward layers, in the same order. What
                  changes is the size of the grids: a few dozen numbers wide becomes a few thousand, two layers
                  becomes eighty, a vocabulary of a couple of hundred becomes a hundred thousand.
                </p>
                <p>
                  Those changes are multiplicative, which is why the cost explodes. Doubling the width roughly
                  quadruples the work in every weight matrix, because both dimensions grow at once.
                </p>
              </>
            }
            math={
              <>
                <p className="mb-2">
                  Per-layer parameters scale as <M>O(d²)</M>: attention contributes <M>4d²</M> and the
                  feed-forward block <M>2d·d_ff</M>, with <M>d_ff</M> conventionally <M>4d</M>, giving{' '}
                  <M>12d²</M> per layer. Total is then roughly <M>12·n_layers·d²</M> plus embeddings.
                </p>
                <p>
                  Attention adds a term the others do not have: <M>O(T²·d)</M> per layer, quadratic in sequence
                  length. At {T} tokens that term is{' '}
                  {fmtFlops(2 * T * T * (rows[0]?.cfg.dModel ?? 1) * 2 * (rows[0]?.cfg.nLayers ?? 1))} for the
                  tiny model here, but it is why long context is priced the way it is.
                </p>
              </>
            }
            code={
              <Code>{`// engine/shapes.ts
const perLayer = 4*dModel*dModel      // Wq Wk Wv Wo
               + 2*dModel*dFF         // W1 W2
               + 6*dModel + dFF;      // biases and layer-norm gains
const total = vocab*dModel + ctx*dModel
            + nLayers*perLayer
            + dModel*vocab;           // unembedding`}</Code>
            }
          />
        </Panel>
      </div>
    </div>
  );
}

/* ================================================================ page == */

export default function MatrixLab() {
  const [tab, setTab] = useState<'hand' | 'words' | 'scale'>('hand');
  const [presetId, setPresetId] = useState('small');
  const trainerRef = useRef<LMTrainer | null>(null);
  const lastKey = useRef('');
  if (!trainerRef.current || lastKey.current !== presetId) {
    trainerRef.current = buildLMTrainer('stories', presetId);
    lastKey.current = presetId;
  }

  return (
    <div className="mx-auto max-w-[1400px] px-6 py-6">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <Segmented
          value={tab}
          onChange={setTab}
          options={[
            { id: 'hand', label: '1 · Multiply by hand' },
            { id: 'words', label: '2 · Your words as matrices' },
            { id: 'scale', label: '3 · Small model vs large model' },
          ]}
        />
        {tab !== 'hand' && (
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1 text-[10.5px]" style={{ color: 'var(--text-3)' }}>
              local model
              <InfoDot info={V.localModelSize} title="local model" />
            </span>
            <div style={{ width: 100 }}>
              <Select
                value={presetId}
                onChange={setPresetId}
                options={[
                  { id: 'tiny', label: 'Tiny' },
                  { id: 'small', label: 'Small' },
                  { id: 'medium', label: 'Medium' },
                ]}
              />
            </div>
          </div>
        )}
      </div>

      {tab === 'hand' ? (
        <ByHandTab />
      ) : tab === 'words' ? (
        <WordsTab trainer={trainerRef.current} />
      ) : (
        <ScaleTab trainer={trainerRef.current} />
      )}
    </div>
  );
}
