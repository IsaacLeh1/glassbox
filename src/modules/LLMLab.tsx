import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CORPORA, getCorpus } from '../engine/corpus';
import type { LMTrainer } from '../engine/lmTrainer';
import { MODEL_PRESETS, buildLMTrainer } from '../engine/lmPresets';
import { sampleToken, type SampleConfig, type Trace } from '../engine/transformer';
import { charTokenizer } from '../engine/tokenizer';
import { mulberry32, softmax1d } from '../engine/tensor';
import {
  Badge,
  BackgroundNotice,
  Btn,
  Callout,
  Code,
  Depth,
  Empty,
  Eq,
  Field,
  Label,
  M,
  Panel,
  ProgressBar,
  Segmented,
  Select,
  Slider,
  Spinner,
  Stat,
  InfoDot,
  fmt,
  fmtInt,
} from '../ui/kit';
import { V } from '../content/varInfo';
import { BarMeter, HeatGrid, LineChart, type Series } from '../ui/viz';
import Diagnostics from '../ui/Diagnostics';
import { useFrameLoop } from '../ui/loop';
import type { RunFacts } from '../engine/diagnostics';
import { getCorpus as loadCorpus } from '../engine/corpus';

export interface LMCtx {
  trainer: LMTrainer;
  version: number;
  bump: () => void;
}

/* ============================================================ tokenizer == */

function TokenizerTab({ trainer }: LMCtx) {
  const [text, setText] = useState('the quiet fox sleeps in the garden at dawn.');
  const tok = trainer.tok;

  const chunks = useMemo(() => text.match(/\s*\S+|\s+/g) ?? [], [text]);
  const perWord = useMemo(() => chunks.map((w) => tok.encodeWordTrace(w)), [chunks, tok]);
  const ids = useMemo(() => tok.encode(text), [text, tok]);
  const chars = useMemo(() => charTokenizer(getCorpus('stories', 12)), []);

  const [openWord, setOpenWord] = useState(0);
  const steps = perWord[Math.min(openWord, perWord.length - 1)] ?? [];
  const light = typeof document !== 'undefined' && document.documentElement.getAttribute('data-theme') === 'light';

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
      <div className="space-y-4">
        <Panel
          title="Text is not what the model sees"
          subtitle="Before anything else your sentence is cut into tokens, and each becomes an integer."
          right={<Badge tone="accent">{ids.length} tokens</Badge>}
        >
          <textarea value={text} onChange={(e) => setText(e.target.value)} rows={2} className="mono" style={{ resize: 'vertical' }} />

          <div className="mt-3 flex flex-wrap gap-1">
            {ids.map((id, i) => (
              <span
                key={i}
                className="mono rounded px-1.5 py-[3px] text-[11.5px]"
                style={{
                  background: `hsl(${(id * 47) % 360} 45% ${light ? '88%' : '22%'})`,
                  color: 'var(--text)',
                  border: '1px solid var(--border)',
                }}
                title={`token id ${id}`}
              >
                {tok.display(id)}
              </span>
            ))}
          </div>

          <div
            className="mono mt-3 overflow-x-auto rounded-lg p-2.5 text-[11px]"
            style={{ background: 'var(--bg-2)', border: '1px solid var(--border)', color: 'var(--text-3)' }}
          >
            [{ids.join(', ')}]
          </div>

          <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Stat label="characters" value={text.length} />
            <Stat label="BPE tokens" value={ids.length} tone="accent" />
            <Stat label="chars per token" value={fmt(text.length / Math.max(1, ids.length), 2)} />
            <Stat label="vocabulary" value={tok.size} sub={`${tok.merges.length} merges learned`} />
          </div>
        </Panel>

        <Panel title="How one word gets merged" subtitle="Byte-pair encoding applies its learned merges in rank order">
          <div className="mb-3 flex flex-wrap gap-1">
            {chunks.map((w, i) => (
              <Btn key={i} size="sm" active={i === openWord} onClick={() => setOpenWord(i)}>
                <span className="mono">{w.replace(/ /g, '·')}</span>
              </Btn>
            ))}
          </div>

          <div className="space-y-1.5">
            {steps.map((s, i) => (
              <div key={i} className="flex flex-wrap items-center gap-3">
                <span className="mono w-14 shrink-0 text-[10px]" style={{ color: 'var(--text-3)' }}>
                  {i === 0 ? 'start' : `merge ${s.appliedMerge?.rank}`}
                </span>
                <div className="flex flex-wrap gap-1">
                  {s.tokens.map((t, j) => {
                    const isNew = s.appliedMerge && t === s.appliedMerge.result;
                    return (
                      <span
                        key={j}
                        className="mono rounded px-1.5 py-[2px] text-[11px]"
                        style={{
                          background: isNew ? 'color-mix(in srgb, var(--accent) 22%, transparent)' : 'var(--panel-2)',
                          border: `1px solid ${isNew ? 'var(--accent)' : 'var(--border)'}`,
                        }}
                      >
                        {t.replace(/ /g, '·')}
                      </span>
                    );
                  })}
                </div>
                {s.appliedMerge && (
                  <span className="mono text-[10px]" style={{ color: 'var(--text-3)' }}>
                    {s.appliedMerge.a.replace(/ /g, '·')} + {s.appliedMerge.b.replace(/ /g, '·')} seen{' '}
                    {s.appliedMerge.count}×
                  </span>
                )}
              </div>
            ))}
          </div>

          <div className="mt-4">
            <Depth
              plain={
                <>
                  The tokenizer starts out knowing only single characters. It then finds the pair of neighbours
                  that appears most often across all the training text and glues that pair into one new symbol,
                  repeating thousands of times. Common words end up as a single token; rare words stay in
                  pieces. This is why models can be oddly bad at spelling: they often never see the individual
                  letters of a common word at all.
                </>
              }
              math={
                <>
                  Greedy merging: at round <M>r</M>, pick <M>(a,b) = argmax count(a,b)</M> over adjacent symbol
                  pairs and add <M>ab</M> to the vocabulary. Encoding then applies merges in rank order, always
                  taking the lowest-rank applicable merge, which makes the result deterministic.
                </>
              }
              code={
                <Code>{`// engine/tokenizer.ts
for (const [k, c] of pairs) if (c > bestCount) { bestCount = c; bestKey = k; }
const [a, b] = bestKey.split(SEP);
t.merges.push({ rank: r, a, b, result: a + b, count: bestCount });`}</Code>
              }
            />
          </div>
        </Panel>
      </div>

      <div className="space-y-4">
        <Panel title="Merges this tokenizer learned" subtitle="In the order it discovered them">
          <div className="max-h-[320px] space-y-0.5 overflow-y-auto pr-1">
            {tok.merges.slice(0, 80).map((m) => (
              <div key={m.rank} className="mono flex items-center justify-between gap-2 text-[11px]">
                <span style={{ color: 'var(--text-3)' }}>#{m.rank}</span>
                <span className="truncate" style={{ color: 'var(--text-2)' }}>
                  {m.a.replace(/ /g, '·')} + {m.b.replace(/ /g, '·')} →{' '}
                  <span style={{ color: 'var(--accent)' }}>{m.result.replace(/ /g, '·')}</span>
                </span>
                <span style={{ color: 'var(--text-3)' }}>{m.count}</span>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="Why not just use characters?">
          <div className="space-y-2 text-[12.5px]" style={{ color: 'var(--text-2)' }}>
            <div className="flex justify-between">
              <span>character tokens</span>
              <span className="mono">{chars.encode(text).length}</span>
            </div>
            <div className="flex justify-between">
              <span>BPE tokens</span>
              <span className="mono" style={{ color: 'var(--accent)' }}>
                {ids.length}
              </span>
            </div>
            <p className="pt-1 text-[12px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
              Attention cost grows with the square of sequence length, so halving the sequence cuts the
              attention work to a quarter. That is the entire reason byte-pair encoding exists.
            </p>
          </div>
        </Panel>
      </div>
    </div>
  );
}

/* ============================================================== forward == */

function ForwardTab({ trainer, version }: LMCtx) {
  const [prompt, setPrompt] = useState('the quiet fox sleeps in the');
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [hovered, setHovered] = useState<{ i: number; j: number; v: number } | null>(null);

  const ids = useMemo(
    () => trainer.tok.encode(prompt).slice(-trainer.model.cfg.blockSize),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [prompt, trainer, version],
  );

  const result = useMemo(() => {
    if (ids.length === 0) return null;
    return trainer.model.forward(ids).trace;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ids, trainer, version]);

  const labels = ids.map((id) => trainer.tok.display(id) || '·');
  const cfg = trainer.model.cfg;
  const safeLayer = Math.min(layer, cfg.nLayers - 1);
  const safeHead = Math.min(head, cfg.nHeads - 1);

  if (!result) return <Empty>Type something for the model to read.</Empty>;

  const L: Trace['layers'][number] = result.layers[safeLayer];
  const att = L.heads[safeHead].att;
  const T = ids.length;

  const lastLogits: number[] = [];
  for (let j = 0; j < cfg.vocab; j++) lastLogits.push(result.logits.data[(T - 1) * cfg.vocab + j]);
  const probs = softmax1d(lastLogits, 1);
  const top = probs.map((p, id) => ({ p, id })).sort((a, b) => b.p - a.p).slice(0, 10);
  const entropy = -probs.reduce((a, p) => a + (p > 0 ? p * Math.log2(p) : 0), 0);

  return (
    <div className="space-y-4">
      <Panel title="A sentence, all the way through" subtitle="Every panel below is this exact input, recomputed live">
        <input value={prompt} onChange={(e) => setPrompt(e.target.value)} className="mono" />
        <div className="mt-2 flex flex-wrap gap-1">
          {labels.map((t, i) => (
            <span
              key={i}
              className="mono rounded px-1.5 py-[2px] text-[11px]"
              style={{ background: 'var(--panel-2)', border: '1px solid var(--border)', color: 'var(--text-2)' }}
            >
              <span style={{ color: 'var(--text-3)' }}>{i}</span> {t}
            </span>
          ))}
        </div>
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Step 1 — Embedding" subtitle="Each token id becomes a vector, then position is added on top">
          <div className="space-y-3">
            {[
              ['token embeddings', result.tokenEmb],
              ['+ positional embeddings', result.posEmb],
              ['= the residual stream begins', result.embedded],
            ].map(([label, mtx]) => (
              <div key={label as string}>
                <Label>
                  {label as string} ({T} × {cfg.dModel})
                </Label>
                <div className="mt-1 overflow-x-auto">
                  <HeatGrid
                    rows={T}
                    cols={cfg.dModel}
                    get={(i, j) => (mtx as typeof result.tokenEmb).data[i * cfg.dModel + j]}
                    cell={7}
                    gap={1}
                  />
                </div>
              </div>
            ))}
          </div>
          <div className="mt-3">
            <Depth
              plain={
                <>
                  A token id like 42 means nothing on its own. The model keeps a lookup table with one row of{' '}
                  {cfg.dModel} numbers for every token in its vocabulary, and that row is the token&apos;s
                  meaning as far as the model is concerned. Position has to be added separately, because
                  attention by itself has no notion of word order at all.
                </>
              }
              math={
                <>
                  <Eq>
                    x₍t₎ = E[token₍t₎] + P[t], E ∈ ℝ^({cfg.vocab}×{cfg.dModel}), P ∈ ℝ^({cfg.blockSize}×
                    {cfg.dModel})
                  </Eq>
                  Both tables are learned. Together they account for{' '}
                  {fmtInt((cfg.vocab + cfg.blockSize) * cfg.dModel)} of this model&apos;s{' '}
                  {fmtInt(trainer.model.paramCount)} parameters.
                </>
              }
              code={
                <Code>{`tokenEmb.data[t * dModel + j] = this.tok.M.data[tokens[t] * dModel + j];
posEmb.data[t * dModel + j]   = this.pos.M.data[t * dModel + j];
embedded.data[i] = tokenEmb.data[i] + posEmb.data[i];`}</Code>
              }
            />
          </div>
        </Panel>

        <Panel
          title="Step 2 — Attention"
          subtitle="Row i shows where token i looked. The upper triangle is empty because nothing sees the future."
          right={
            <div className="flex items-center gap-1.5">
              <InfoDot info={V.attentionLayer} title="layer" />
              <Select
                value={String(safeLayer)}
                onChange={(v) => setLayer(parseInt(v, 10))}
                options={Array.from({ length: cfg.nLayers }, (_, i) => ({ id: String(i), label: `layer ${i}` }))}
              />
              <InfoDot info={V.attentionHead} title="head" />
              <Select
                value={String(safeHead)}
                onChange={(v) => setHead(parseInt(v, 10))}
                options={Array.from({ length: cfg.nHeads }, (_, i) => ({ id: String(i), label: `head ${i}` }))}
              />
            </div>
          }
        >
          <div className="flex gap-2 overflow-x-auto">
            <div className="flex flex-col pt-[1px]">
              {labels.map((t, i) => (
                <span
                  key={i}
                  className="mono text-right text-[8.5px]"
                  style={{ height: 18, lineHeight: '18px', color: hovered?.i === i ? 'var(--accent)' : 'var(--text-3)' }}
                >
                  {t.slice(0, 7)}
                </span>
              ))}
            </div>
            <div>
              <HeatGrid
                rows={T}
                cols={T}
                get={(i, j) => att.data[i * T + j]}
                mode="heat"
                max={1}
                cell={17}
                gap={1}
                onHover={(i, j, v) => setHovered(i < 0 ? null : { i, j, v })}
              />
            </div>
          </div>

          <div className="mt-2 min-h-[34px]">
            {hovered ? (
              <div className="mono text-[11.5px]" style={{ color: 'var(--text-2)' }}>
                <span style={{ color: 'var(--accent)' }}>{labels[hovered.i]}</span> (pos {hovered.i}) gave{' '}
                <span style={{ color: 'var(--pos)' }}>{(hovered.v * 100).toFixed(1)}%</span> of its attention to{' '}
                <span style={{ color: 'var(--accent)' }}>{labels[hovered.j]}</span> (pos {hovered.j})
                {hovered.j > hovered.i && <span style={{ color: 'var(--text-3)' }}> — masked, always zero</span>}
              </div>
            ) : (
              <span className="text-[11.5px]" style={{ color: 'var(--text-3)' }}>
                Hover any cell to read the attention weight.
              </span>
            )}
          </div>

          <div className="mt-2">
            <Depth
              plain={
                <>
                  Every token writes three things: a query saying what it is looking for, a key advertising what
                  it offers, and a value carrying what it would contribute. Each token compares its query with
                  every earlier key, turns those scores into percentages adding to 100, and mixes the values in
                  those proportions. That is the whole mechanism, done several times in parallel by different
                  heads.
                </>
              }
              math={
                <>
                  <Eq note={`d_head = ${trainer.model.dHead}, so the scale is 1/sqrt(${trainer.model.dHead}) = ${fmt(1 / Math.sqrt(trainer.model.dHead), 3)}`}>
                    Attention(Q,K,V) = softmax( QKᵀ / √d_head + mask ) V
                  </Eq>
                  <p>
                    The mask sets every entry above the diagonal to −∞ before the softmax, making those
                    probabilities exactly zero. Without it the model could read the answer it is being asked to
                    predict.
                  </p>
                </>
              }
              code={<Code>{`const raw = matmulTB(q, k);
scores.data[i*T+j] = j <= i ? raw.data[i*T+j] * scale : -1e9;
const att = softmaxRows(scores);
const out = matmul(att, v);`}</Code>}
            />
          </div>
        </Panel>
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_380px]">
        <Panel title="Every head at once" subtitle="Different heads attend to different structure. This is specialisation, visible.">
          <div className="flex flex-wrap gap-3">
            {result.layers.map((Lr, li) =>
              Lr.heads.map((hd, hi) => (
                <button
                  key={`${li}-${hi}`}
                  onClick={() => {
                    setLayer(li);
                    setHead(hi);
                  }}
                  className="focus-ring cursor-pointer rounded-lg p-1.5 transition-colors"
                  style={{
                    background: li === safeLayer && hi === safeHead ? 'var(--panel-3)' : 'transparent',
                    border: `1px solid ${li === safeLayer && hi === safeHead ? 'var(--accent)' : 'var(--border)'}`,
                  }}
                >
                  <div className="mono mb-1 text-[9.5px]" style={{ color: 'var(--text-3)' }}>
                    L{li}·H{hi}
                  </div>
                  <HeatGrid rows={T} cols={T} get={(i, j) => hd.att.data[i * T + j]} mode="heat" max={1} cell={5} gap={0} />
                </button>
              )),
            )}
          </div>
        </Panel>

        <Panel title="Step 3 — What comes next" subtitle={`Prediction for the position after "${labels[T - 1]}"`}>
          <div className="space-y-1.5">
            {top.map((t) => (
              <div key={t.id} className="flex items-center gap-2">
                <span className="mono w-[76px] shrink-0 truncate text-[11.5px]" style={{ color: 'var(--text-2)' }}>
                  {trainer.tok.display(t.id) || '·'}
                </span>
                <div className="flex-1">
                  <BarMeter value={t.p} max={top[0].p} color="var(--accent)" height={7} />
                </div>
                <span className="mono tnum w-[48px] shrink-0 text-right text-[10.5px]" style={{ color: 'var(--text-3)' }}>
                  {(t.p * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
          <div className="mt-3 grid grid-cols-2 gap-3">
            <Stat label="vocabulary" value={cfg.vocab} />
            <Stat label="entropy" value={fmt(entropy, 2)} unit="bits" hint="How undecided the model is. Zero means certain." />
          </div>
          {trainer.step === 0 && (
            <div className="mt-3">
              <Callout tone="warn">
                This model has not been trained yet, so these probabilities are near-uniform noise. Open the
                Train tab, run it for a few seconds, then come back.
              </Callout>
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}

/* =============================================================== train == */

function TrainTab({ trainer, bump, corpusId }: LMCtx & { corpusId: string }) {
  const [running, setRunning] = useState(false);
  const [samples, setSamples] = useState<{ step: number; text: string }[]>([]);

  const generate = useCallback(
    (n = 36) => {
      const rand = mulberry32(Date.now() & 0xffff);
      let ids = trainer.tok.encode('the ');
      for (let i = 0; i < n; i++) {
        const { probs } = trainer.model.predictNext(ids);
        ids = [...ids, sampleToken(probs, { temperature: 0.8, topK: 12, topP: 0.95 }, rand).chosen];
      }
      return trainer.tok.decode(ids);
    },
    [trainer],
  );

  useFrameLoop(running, () => {
    const before = trainer.step;
    const n = trainer.runSlice(11);
    if (n === 0 || trainer.status === 'done') {
      setRunning(false);
      setSamples((s) => [...s, { step: trainer.step, text: generate() }]);
    } else if (Math.floor(trainer.step / 50) > Math.floor(before / 50)) {
      setSamples((s) => [...s, { step: trainer.step, text: generate() }]);
    }
    bump();
  });

  const h = trainer.history;
  const last = trainer.latest();
  const tokPerSec = last && last.wallMs > 0 ? last.tokensSeen / (last.wallMs / 1000) : 0;

  const series: Series[] = [
    { id: 'train', label: 'train loss', color: 'var(--accent)', points: h.map((m) => ({ x: m.step, y: m.loss })) },
    { id: 'val', label: 'validation loss', color: 'var(--pos)', dashed: true, points: h.map((m) => ({ x: m.step, y: m.valLoss })) },
  ];
  const ppl: Series[] = [
    { id: 'ppl', label: 'perplexity', color: 'var(--ok)', points: h.map((m) => ({ x: m.step, y: m.perplexity })) },
  ];

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
      <div className="space-y-4">
        <Panel
          title="Training a language model, for real"
          subtitle="Predict the next token at every position, measure the error, push every weight a little."
          right={
            <div className="flex items-center gap-1.5">
              <BackgroundNotice running={running} />
              <Btn variant="primary" size="sm" onClick={() => setRunning((r) => !r)} disabled={trainer.status === 'done'}>
                {running ? 'Pause' : trainer.step > 0 ? 'Resume' : 'Train'}
              </Btn>
              <Btn
                size="sm"
                onClick={() => {
                  setRunning(false);
                  trainer.reset();
                  setSamples([]);
                  bump();
                }}
              >
                Reset
              </Btn>
            </div>
          }
        >
          <ProgressBar value={trainer.progress} tone={trainer.status === 'done' ? 'ok' : 'accent'} />
          <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Stat label="step" value={`${trainer.step} / ${trainer.cfg.steps}`} />
            <Stat label="loss" value={fmt(last?.loss ?? NaN, 4)} tone="accent" />
            <Stat label="perplexity" value={fmt(last?.perplexity ?? NaN, 2)} hint="Roughly how many tokens it is choosing between." />
            <Stat label="tokens seen" value={fmtInt(trainer.tokensSeen)} sub={`${fmtInt(tokPerSec)} tok/s`} />
          </div>

          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <div>
              <Label>loss</Label>
              <LineChart series={series} height={150} xLabel="step" logY />
            </div>
            <div>
              <Label>perplexity</Label>
              <LineChart series={ppl} height={150} xLabel="step" yMin={1} />
            </div>
          </div>
        </Panel>

        <Diagnostics
          facts={
            {
              history: trainer.history,
              vocab: trainer.vocabSize,
              paramCount: trainer.model.paramCount,
              trainTokens: trainer.trainIds.length,
              blockSize: trainer.model.cfg.blockSize,
              totalSteps: trainer.cfg.steps,
              corpusChars: loadCorpus(corpusId, 12).length,
            } satisfies RunFacts
          }
          sample={samples.length ? samples[samples.length - 1].text : null}
          corpus={loadCorpus(corpusId, 12)}
        />

        <Panel title="What it writes as it learns" subtitle="Sampled from the live model every 50 steps">
          {samples.length === 0 ? (
            <Empty>Start training. Samples appear here as the model improves.</Empty>
          ) : (
            <div className="space-y-2">
              {[...samples]
                .reverse()
                .slice(0, 8)
                .map((s, i) => (
                  <div key={s.step} className="rounded-lg p-2.5" style={{ background: 'var(--bg-2)', border: '1px solid var(--border)' }}>
                    <div className="mono mb-1 text-[10px]" style={{ color: i === 0 ? 'var(--accent)' : 'var(--text-3)' }}>
                      step {s.step}
                    </div>
                    <div className="mono text-[11.5px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                      {s.text}
                    </div>
                  </div>
                ))}
            </div>
          )}
        </Panel>

        <Panel title="Reading the loss number">
          <Depth
            plain={
              <>
                Loss is the model&apos;s surprise. Guessing uniformly among {trainer.vocabSize} tokens would put
                it near {fmt(Math.log(trainer.vocabSize), 2)}. Perplexity turns that surprise back into a count:
                a perplexity of 8 means it is about as uncertain as picking evenly between eight options. Watch
                it fall from {trainer.vocabSize} toward single digits and you are watching grammar being learned.
              </>
            }
            math={
              <>
                <Eq note="Averaged over every position in every sequence in the batch.">
                  L = −(1/T) Σ log p(token₍t+1₎ | token₍≤t₎), perplexity = e^L
                </Eq>
                <p>
                  Uniform guessing gives L = ln({trainer.vocabSize}) = {fmt(Math.log(trainer.vocabSize), 3)}. The
                  current loss of {fmt(last?.loss ?? NaN, 3)} is the equivalent of{' '}
                  {fmt(Math.exp(last?.loss ?? Math.log(trainer.vocabSize)), 1)} equally likely choices.
                </p>
              </>
            }
            code={<Code>{`this.model.zeroGrad();
for (let s = 0; s < this.cfg.batchSeqs; s++) {
  const w  = this.sampleWindow(this.trainIds);
  const fw = this.model.forward(w.x);
  loss += this.model.loss(fw.trace, w.y);
  this.model.backward(fw, w.y);       // gradients accumulate
}
for (const q of this.model.params)
  for (let i = 0; i < q.g.data.length; i++) q.g.data[i] /= counted;`}</Code>}
          />
        </Panel>
      </div>

      <div className="space-y-4">
        <Panel title="This model">
          <div className="space-y-1.5 text-[11.5px]">
            {[
              ['parameters', fmtInt(trainer.model.paramCount)],
              ['vocabulary', String(trainer.vocabSize)],
              ['embedding width', String(trainer.model.cfg.dModel)],
              ['layers', String(trainer.model.cfg.nLayers)],
              ['heads per layer', String(trainer.model.cfg.nHeads)],
              ['head width', String(trainer.model.dHead)],
              ['context window', `${trainer.model.cfg.blockSize} tokens`],
              ['feed-forward width', String(trainer.model.cfg.dFF)],
              ['training tokens', fmtInt(trainer.trainIds.length)],
              ['held out', fmtInt(trainer.valIds.length)],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <span style={{ color: 'var(--text-3)' }}>{k}</span>
                <span className="mono" style={{ color: 'var(--text-2)' }}>
                  {v}
                </span>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="Parameter tensors" subtitle="Every named weight in the model">
          <div className="max-h-[300px] space-y-0.5 overflow-y-auto pr-1">
            {trainer.model.params.map((p) => (
              <div key={p.name} className="mono flex items-center justify-between gap-2 text-[10.5px]">
                <span className="truncate" style={{ color: 'var(--text-2)' }}>
                  {p.name}
                </span>
                <span className="shrink-0" style={{ color: 'var(--text-3)' }}>
                  {p.M.rows}×{p.M.cols}
                </span>
              </div>
            ))}
          </div>
        </Panel>

        <Callout tone="insight">
          The architecture on this page is the same one behind the large models you have used. The differences
          are entirely quantitative: more layers, wider vectors, a bigger vocabulary, vastly more text. The
          arithmetic is identical.
        </Callout>
      </div>
    </div>
  );
}

/* ============================================================ generate == */

function GenerateTab({ trainer, version }: LMCtx) {
  const [prompt, setPrompt] = useState('the quiet fox');
  const [generated, setGenerated] = useState<number[]>([]);
  const [cfg, setCfg] = useState<SampleConfig>({ temperature: 0.8, topK: 12, topP: 0.95 });
  const [auto, setAuto] = useState(false);
  const randRef = useRef(mulberry32(1234));

  const baseIds = useMemo(() => trainer.tok.encode(prompt), [prompt, trainer, version]);
  const allIds = useMemo(() => [...baseIds, ...generated], [baseIds, generated]);

  const step = useMemo(() => {
    if (allIds.length === 0) return null;
    const { probs } = trainer.model.predictNext(allIds);
    // Preview the filtering without consuming randomness.
    return sampleToken(probs, cfg, () => 0.5);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [allIds, cfg, trainer, version]);

  const advance = useCallback(() => {
    const { probs } = trainer.model.predictNext(allIds);
    const s = sampleToken(probs, cfg, randRef.current);
    setGenerated((g) => [...g, s.chosen]);
  }, [allIds, cfg, trainer]);

  useEffect(() => {
    if (!auto) return;
    if (generated.length >= 60) {
      setAuto(false);
      return;
    }
    const t = setTimeout(advance, 130);
    return () => clearTimeout(t);
  }, [auto, advance, generated.length]);

  const ranked = step ? [...step.candidates].sort((a, b) => b.prob - a.prob).slice(0, 14) : [];

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
      <div className="space-y-4">
        <Panel
          title="One token at a time"
          subtitle="The model never plans a sentence. It picks one token, appends it, and does the entire thing again."
          right={
            <div className="flex gap-1.5">
              <Btn variant="primary" size="sm" onClick={advance}>
                Next token
              </Btn>
              <Btn size="sm" active={auto} onClick={() => setAuto((a) => !a)}>
                {auto ? 'Stop' : 'Autoplay'}
              </Btn>
              <Btn size="sm" onClick={() => setGenerated([])}>
                Clear
              </Btn>
            </div>
          }
        >
          <Field label="prompt">
            <input value={prompt} onChange={(e) => setPrompt(e.target.value)} className="mono" />
          </Field>

          <div
            className="mono mt-3 min-h-[92px] rounded-lg p-3 text-[13px] leading-relaxed"
            style={{ background: 'var(--bg-2)', border: '1px solid var(--border)' }}
          >
            <span style={{ color: 'var(--text-3)' }}>{trainer.tok.decode(baseIds)}</span>
            {generated.map((id, i) => (
              <span
                key={i}
                style={{
                  color: 'var(--text)',
                  background:
                    i === generated.length - 1 ? 'color-mix(in srgb, var(--accent) 22%, transparent)' : 'transparent',
                  borderRadius: 3,
                }}
              >
                {trainer.tok.tokenString(id)}
              </span>
            ))}
            <span className="gb-pulse" style={{ color: 'var(--accent)' }}>
              ▌
            </span>
          </div>

          <div className="mt-3 grid grid-cols-3 gap-3">
            <Stat label="context used" value={`${Math.min(allIds.length, trainer.model.cfg.blockSize)} / ${trainer.model.cfg.blockSize}`} />
            <Stat label="generated" value={generated.length} />
            <Stat label="forward passes" value={generated.length} hint="One full pass through the network per token produced." />
          </div>
        </Panel>

        <Panel title="The candidates for the next token" subtitle="Faded rows were cut by top-k or top-p and cannot be chosen">
          {ranked.length === 0 ? (
            <Empty>Enter a prompt.</Empty>
          ) : (
            <div className="space-y-1.5">
              {ranked.map((c) => (
                <div key={c.id} className="flex items-center gap-2" style={{ opacity: c.kept ? 1 : 0.32 }}>
                  <span className="mono w-[84px] shrink-0 truncate text-[11.5px]" style={{ color: 'var(--text-2)' }}>
                    {trainer.tok.display(c.id) || '·'}
                  </span>
                  <span className="mono tnum w-[52px] shrink-0 text-right text-[10px]" style={{ color: 'var(--text-3)' }}>
                    {fmt(c.logit, 2)}
                  </span>
                  <div className="flex-1">
                    <BarMeter value={c.prob} max={ranked[0].prob} color={c.kept ? 'var(--accent)' : 'var(--text-3)'} height={7} />
                  </div>
                  <span
                    className="mono tnum w-[46px] shrink-0 text-right text-[10.5px]"
                    style={{ color: c.kept ? 'var(--text-2)' : 'var(--text-3)' }}
                  >
                    {(c.prob * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          )}
          <div className="mt-3 flex gap-4 text-[10px]" style={{ color: 'var(--text-3)' }}>
            <span>logit → the raw score</span>
            <span>probability → after temperature and softmax</span>
          </div>
        </Panel>
      </div>

      <div className="space-y-4">
        <Panel title="Sampling controls" subtitle="None of these change the model. They change only how its output is read.">
          <div className="space-y-4">
            <div>
              <Slider
                label="temperature" info={V.temperature}
                value={cfg.temperature}
                min={0.05}
                max={2}
                step={0.05}
                onChange={(t) => setCfg((c) => ({ ...c, temperature: t }))}
              />
              <p className="mt-1 text-[11px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                Divides every logit before the softmax. Below 1 the strong options get stronger and output turns
                repetitive; above 1 the distribution flattens and output turns erratic.
              </p>
            </div>
            <div>
              <Slider
                label="top-k" info={V.topK}
                value={cfg.topK}
                min={0}
                max={40}
                step={1}
                format={(x) => (x === 0 ? 'off' : String(Math.round(x)))}
                onChange={(k) => setCfg((c) => ({ ...c, topK: Math.round(k) }))}
              />
              <p className="mt-1 text-[11px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                Keep only the k most likely tokens, discard everything else before sampling.
              </p>
            </div>
            <div>
              <Slider
                label="top-p" info={V.topP}
                value={cfg.topP}
                min={0.1}
                max={1}
                step={0.01}
                onChange={(p) => setCfg((c) => ({ ...c, topP: p }))}
              />
              <p className="mt-1 text-[11px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                Keep the smallest set of tokens whose probabilities add to p. Narrow when the model is
                confident, wide when it is not.
              </p>
            </div>
          </div>
        </Panel>

        <Panel title="Try this">
          <ol className="space-y-2 text-[12px]" style={{ color: 'var(--text-2)' }}>
            <li>Set temperature to 0.05. The output becomes deterministic and starts looping.</li>
            <li>Set it to 1.8. Grammar falls apart, because unlikely tokens now get chosen regularly.</li>
            <li>
              Set top-k to 1. That is greedy decoding. The model has not changed at all, only the rule for
              reading it.
            </li>
          </ol>
        </Panel>

        {trainer.step === 0 && <Callout tone="warn">The model is untrained, so this will produce noise. Train it first.</Callout>}
      </div>
    </div>
  );
}

/* ================================================================ page == */

export default function LLMLab() {
  const [tab, setTab] = useState<'tok' | 'fwd' | 'train' | 'gen'>('tok');
  const [corpusId, setCorpusId] = useState('stories');
  const [presetId, setPresetId] = useState('small');
  const [version, setVersion] = useState(0);
  const [building, setBuilding] = useState(false);

  const key = `${corpusId}|${presetId}`;
  const trainerRef = useRef<LMTrainer | null>(null);
  const lastKey = useRef('');
  if (!trainerRef.current || lastKey.current !== key) {
    trainerRef.current = buildLMTrainer(corpusId, presetId);
    lastKey.current = key;
  }

  const bump = useCallback(() => setVersion((v) => v + 1), []);
  const ctx: LMCtx = { trainer: trainerRef.current, version, bump };
  const corpus = CORPORA.find((c) => c.id === corpusId)!;

  const rebuild = (fn: () => void) => {
    setBuilding(true);
    setTimeout(() => {
      fn();
      setBuilding(false);
      bump();
    }, 20);
  };

  return (
    <div className="mx-auto max-w-[1400px] px-6 py-6">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <Segmented
          value={tab}
          onChange={setTab}
          options={[
            { id: 'tok', label: '1 · Tokenize' },
            { id: 'fwd', label: '2 · Inside a forward pass' },
            { id: 'train', label: '3 · Train it' },
            { id: 'gen', label: '4 · Generate' },
          ]}
        />
        <div className="flex items-center gap-2">
          {building && <Spinner />}
          <span className="inline-flex items-center gap-1 text-[10.5px]" style={{ color: 'var(--text-3)' }}>
            corpus
            <InfoDot info={V.corpus} title="corpus" />
          </span>
          <div style={{ width: 132 }}>
            <Select value={corpusId} onChange={(v) => rebuild(() => setCorpusId(v))} options={CORPORA.map((c) => ({ id: c.id, label: c.label }))} />
          </div>
          <span className="inline-flex items-center gap-1 text-[10.5px]" style={{ color: 'var(--text-3)' }}>
            size
            <InfoDot info={V.localModelSize} title="model size" />
          </span>
          <div style={{ width: 96 }}>
            <Select value={presetId} onChange={(v) => rebuild(() => setPresetId(v))} options={MODEL_PRESETS.map((p) => ({ id: p.id, label: p.label }))} />
          </div>
          <Badge tone="accent">{fmtInt(ctx.trainer.model.paramCount)} params</Badge>
        </div>
      </div>

      <p className="mb-4 text-[11.5px]" style={{ color: 'var(--text-3)' }}>
        {corpus.blurb}
      </p>

      {tab === 'tok' ? (
        <TokenizerTab {...ctx} />
      ) : tab === 'fwd' ? (
        <ForwardTab {...ctx} />
      ) : tab === 'train' ? (
        <TrainTab {...ctx} corpusId={corpusId} />
      ) : (
        <GenerateTab {...ctx} />
      )}
    </div>
  );
}
