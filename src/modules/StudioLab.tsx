import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  SUGGESTED,
  detectTextColumn,
  getSplits,
  loadTextCorpus,
  searchDatasets,
  type HFDataset,
  type HFSplit,
} from '../data/huggingface';
import { CORPORA, getCorpus } from '../engine/corpus';
import { DEFAULT_LM_TRAIN, LMTrainer } from '../engine/lmTrainer';
import { useModel } from '../store/model';
import LabRun from './LabRun';
import { DEFAULT_TCONFIG, sampleToken, type TransformerConfig } from '../engine/transformer';
import { mulberry32 } from '../engine/tensor';
import {
  checkBudget,
  detectMachine,
  memoryFor,
  paramCountFor,
  readHeapUsedMB,
  suggestedTokens,
} from '../engine/resources';
import { fmtCount } from '../sim/cluster';
import { V } from '../content/varInfo';
import {
  Badge,
  BackgroundNotice,
  Btn,
  Callout,
  Depth,
  Empty,
  Field,
  Label,
  Panel,
  ProgressBar,
  Segmented,
  Select,
  Slider,
  Spinner,
  Stat,
  fmt,
  fmtInt,
} from '../ui/kit';
import { BarMeter, LineChart, type Series } from '../ui/viz';
import CompareTab from './CompareTab';
import DevicePanel from '../ui/DevicePanel';
import Diagnostics from '../ui/Diagnostics';
import StepCost from '../ui/StepCost';
import { useFrameLoop } from '../ui/loop';
import type { BackendId } from '../engine/compute';
import type { RunFacts } from '../engine/diagnostics';

type Tab = 'lab' | 'data' | 'design' | 'train' | 'compare';

/**
 * Opens a page on the Hugging Face hub in a new tab, so the reader can look at
 * the real dataset -- its licence, its card, and a live preview of the rows --
 * before committing to downloading any of it.
 *
 * Kept as a sibling of any button rather than nested inside one: an anchor
 * inside a button is invalid and breaks keyboard navigation.
 */
function HFLink({ path, label, title }: { path: string; label?: string; title?: string }) {
  return (
    <a
      href={`https://huggingface.co/${path}`}
      target="_blank"
      rel="noreferrer noopener"
      title={title ?? 'Open on Hugging Face'}
      className="focus-ring inline-flex shrink-0 items-center gap-1 rounded px-1.5 py-1 text-[10.5px] transition-colors"
      style={{ color: 'var(--text-3)' }}
      onClick={(e) => e.stopPropagation()}
    >
      {label}
      <svg width="9" height="9" viewBox="0 0 16 16" fill="none" aria-hidden>
        <path d="M6 3h7v7M13 3 4 12" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </a>
  );
}

interface Corpus {
  text: string;
  label: string;
  source: string;
}

export default function StudioLab() {
  const [tab, setTab] = useState<Tab>('data');

  // Whatever is built here becomes the model the rest of the walkthrough uses.
  const publish = useModel((s) => s.publish);
  const republish = useModel((s) => s.publish);

  /* ------------------------------------------------------------ machine */
  const machine = useMemo(() => detectMachine(), []);
  const [ramBudgetMB, setRamBudgetMB] = useState(() => {
    const cap = detectMachine().jsHeapLimitMB;
    return Math.round(Math.min(1200, cap ? cap * 0.45 : 700));
  });
  const [frameBudgetMs, setFrameBudgetMs] = useState(8);
  const [backend, setBackend] = useState<BackendId>('cpu');
  const [heapMB, setHeapMB] = useState<number | null>(readHeapUsedMB());

  /* --------------------------------------------------------------- data */
  const [corpus, setCorpus] = useState<Corpus>(() => ({
    text: getCorpus('stories', 12),
    label: 'Tiny stories',
    source: 'built in',
  }));
  const [query, setQuery] = useState('tiny_shakespeare');
  const [results, setResults] = useState<HFDataset[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [picked, setPicked] = useState<string | null>(null);
  const [splits, setSplits] = useState<HFSplit[] | null>(null);
  const [splitIdx, setSplitIdx] = useState(0);
  const [column, setColumn] = useState('text');
  const [maxChars, setMaxChars] = useState(120_000);
  const [loading, setLoading] = useState(false);
  const [loadMsg, setLoadMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [pasted, setPasted] = useState('');

  /* -------------------------------------------------------------- model */
  const [dModel, setDModel] = useState(48);
  const [nLayers, setNLayers] = useState(2);
  const [nHeads, setNHeads] = useState(4);
  const [dFF, setDFF] = useState(96);
  const [blockSize, setBlockSize] = useState(24);
  const [merges, setMerges] = useState(120);
  const [batchSeqs, setBatchSeqs] = useState(8);
  const [steps, setSteps] = useState(400);

  const [built, setBuilt] = useState<LMTrainer | null>(null);
  const [building, setBuilding] = useState(false);
  const [running, setRunning] = useState(false);
  const [, setTick] = useState(0);
  const [samples, setSamples] = useState<{ step: number; text: string }[]>([]);

  /* ---------------------------------------------------------- estimates */

  // Vocabulary is only known exactly after the tokenizer is trained. Before
  // that, estimate it: distinct characters plus the merges that will be learned.
  const estVocab = useMemo(() => {
    if (built) return built.vocabSize;
    const chars = new Set(corpus.text.slice(0, 20000)).size;
    return chars + merges;
  }, [built, corpus.text, merges]);

  const cfg: TransformerConfig = useMemo(
    () => ({ ...DEFAULT_TCONFIG, vocab: estVocab, dModel, nHeads, nLayers, blockSize, dFF }),
    [estVocab, dModel, nHeads, nLayers, blockSize, dFF],
  );

  const mem = useMemo(() => memoryFor(cfg, batchSeqs, corpus.text.length), [cfg, batchSeqs, corpus.text.length]);
  const budget = useMemo(() => checkBudget(mem, ramBudgetMB, machine), [mem, ramBudgetMB, machine]);
  const headsOk = dModel % nHeads === 0;
  const canBuild = budget.verdict !== 'over' && headsOk && corpus.text.length > 500;

  useEffect(() => {
    const t = setInterval(() => setHeapMB(readHeapUsedMB()), 1200);
    return () => clearInterval(t);
  }, []);

  /* ------------------------------------------------------------ actions */

  const doSearch = async () => {
    setSearching(true);
    setErr(null);
    try {
      setResults(await searchDatasets(query, 15));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setResults([]);
    } finally {
      setSearching(false);
    }
  };

  const choose = async (id: string, knownColumn?: string) => {
    setPicked(id);
    setSplits(null);
    setErr(null);
    try {
      const s = await getSplits(id);
      setSplits(s);
      const idx = Math.max(0, s.findIndex((x) => x.split === 'train'));
      setSplitIdx(idx);
      if (knownColumn) {
        setColumn(knownColumn);
      } else if (s[idx]) {
        // Read one row and guess which column actually holds the text.
        try {
          const { column: guess } = await detectTextColumn(id, s[idx].config, s[idx].split);
          setColumn(guess);
        } catch {
          /* leave whatever is in the box */
        }
      }
    } catch (e) {
      setErr(
        `Could not read ${id}: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
  };

  const doLoad = async () => {
    if (!picked || !splits || splits.length === 0) return;
    const s = splits[Math.min(splitIdx, splits.length - 1)];
    setLoading(true);
    setErr(null);
    setLoadMsg('contacting the dataset viewer…');
    try {
      const { text, rowsUsed, requests } = await loadTextCorpus(
        {
          dataset: picked,
          config: s.config,
          split: s.split,
          column,
          maxRows: 5000,
          maxChars,
        },
        (p) => setLoadMsg(`${p.rowsLoaded} rows · ${fmtInt(p.chars)} characters · ${p.requests} requests`),
      );
      if (text.trim().length < 200) {
        setErr(`Loaded ${rowsUsed} rows but column "${column}" was empty or missing. Check the column name.`);
      } else {
        setCorpus({ text, label: picked, source: `Hugging Face · ${s.config}/${s.split} · ${rowsUsed} rows` });
        setLoadMsg(`loaded ${fmtInt(text.length)} characters in ${requests} requests`);
        setBuilt(null);
      }
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const build = () => {
    setBuilding(true);
    setRunning(false);
    setSamples([]);
    // Yield a frame so the spinner paints before the tokenizer blocks.
    setTimeout(() => {
      try {
        const t = new LMTrainer(
          corpus.text,
          { ...DEFAULT_TCONFIG, vocab: 0, dModel, nHeads, nLayers, blockSize, dFF },
          { ...DEFAULT_LM_TRAIN, batchSeqs, steps },
          merges,
        );
        setBuilt(t);
        publish(t, {
          label: `${corpus.label} - ${dModel}d x ${nLayers}L`,
          source: corpus.source,
          chars: corpus.text.length,
          steps: 0,
          finalLoss: NaN,
          trainedAt: Date.now(),
        });
        setTab('train');
      } catch (e) {
        setErr(e instanceof Error ? e.message : String(e));
      } finally {
        setBuilding(false);
      }
    }, 30);
  };

  const generate = useCallback(
    (t: LMTrainer, n = 40) => {
      const rand = mulberry32(Date.now() & 0xffff);
      let ids = t.tok.encode(corpus.text.slice(0, 24).split(/\s+/)[0] || 'the');
      if (ids.length === 0) ids = [0];
      for (let i = 0; i < n; i++) {
        const { logits } = t.model.predictNext(ids);
        ids = [...ids, sampleToken(logits, { temperature: 0.85, topK: 20, topP: 0.95 }, rand).chosen];
      }
      return t.tok.decode(ids);
    },
    [corpus.text],
  );

  useFrameLoop(running && !!built, () => {
    if (!built) return;
    const before = built.step;
    const n = built.runSlice(frameBudgetMs);
    if (n === 0 || built.status === 'done') {
      setRunning(false);
      setSamples((s) => [...s, { step: built.step, text: generate(built) }]);
      // Refresh the shared copy so other modules see the trained loss, not the
      // placeholder recorded when the model was first built.
      republish(built, {
        label: `${corpus.label} - ${dModel}d x ${nLayers}L`,
        source: corpus.source,
        chars: corpus.text.length,
        steps: built.step,
        finalLoss: built.latest()?.loss ?? NaN,
        trainedAt: Date.now(),
      });
    } else if (Math.floor(built.step / 50) > Math.floor(before / 50)) {
      setSamples((s) => [...s, { step: built.step, text: generate(built) }]);
    }
    setTick((t) => t + 1);
  });

  /* ------------------------------------------------------------- render */

  const last = built?.latest() ?? null;
  const measuredTokPerSec = last && last.wallMs > 0 ? last.tokensSeen / (last.wallMs / 1000) : 0;

  return (
    <div className="mx-auto max-w-[1400px] px-6 py-6">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <Segmented
          value={tab}
          onChange={setTab}
          options={[
            { id: 'lab' as Tab, label: 'Run it like a lab' },
            { id: 'data' as Tab, label: '1 · Get data' },
            { id: 'design' as Tab, label: '2 · Design the model' },
            { id: 'train' as Tab, label: '3 · Train it' },
            { id: 'compare' as Tab, label: '4 · You vs a datacenter' },
          ]}
        />
        <div className="flex items-center gap-2">
          {(building || loading) && <Spinner />}
          <Badge tone={budget.verdict === 'ok' ? 'ok' : budget.verdict === 'tight' ? 'warn' : 'err'}>
            {mem.totalMB.toFixed(0)} / {ramBudgetMB} MB
          </Badge>
          <Badge tone="accent">{fmtCount(paramCountFor(cfg))} params</Badge>
        </div>
      </div>

      {/* ========================================================= the programme */}
      {tab === 'lab' && <LabRun />}

      {/* ================================================================ data */}
      {tab === 'data' && (
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
          <div className="space-y-4">
            <Panel
              title="Pull real data from Hugging Face"
              subtitle="Read live from the public dataset viewer. No account, no key, nothing leaves your machine except the request for rows."
            >
              <div className="flex gap-2">
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && doSearch()}
                  placeholder="search the hub, e.g. shakespeare, poems, news"
                  className="mono"
                />
                <Btn variant="primary" onClick={doSearch} disabled={searching}>
                  {searching ? 'Searching…' : 'Search'}
                </Btn>
              </div>
              <p className="mt-1.5 text-[11px]" style={{ color: 'var(--text-3)' }}>
                Searches the real Hugging Face hub. You can also{' '}
                <a
                  href="https://huggingface.co/datasets?task_categories=task_categories:text-generation&sort=trending"
                  target="_blank"
                  rel="noreferrer noopener"
                  className="focus-ring underline"
                  style={{ color: 'var(--accent)' }}
                >
                  browse the hub yourself
                </a>{' '}
                and paste any dataset name into the box above.
              </p>

              <div className="mt-3">
                <Label>or start from one that works well at this scale</Label>
                <div className="mt-1.5 flex flex-wrap gap-1.5">
                  {SUGGESTED.map((s) => (
                    <span
                      key={s.id}
                      className="inline-flex items-center rounded-lg"
                      style={{ background: 'var(--panel-2)', border: '1px solid var(--border)' }}
                    >
                      <Btn size="sm" active={picked === s.id} title={s.note} onClick={() => choose(s.id, s.column)}>
                        {s.id}
                      </Btn>
                      <HFLink path={`datasets/${s.id}`} title={`Look at ${s.id} on Hugging Face first`} />
                    </span>
                  ))}
                </div>
              </div>

              {results && (
                <div className="mt-4 max-h-[240px] space-y-1 overflow-y-auto pr-1">
                  {results.length === 0 && (
                    <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>
                      Nothing found.
                    </p>
                  )}
                  {results.map((d) => (
                    <div
                      key={d.id}
                      className="flex items-center gap-1 rounded"
                      style={{
                        background: picked === d.id ? 'color-mix(in srgb, var(--accent) 12%, transparent)' : 'transparent',
                        border: `1px solid ${picked === d.id ? 'var(--accent)' : 'var(--border)'}`,
                        opacity: d.gated ? 0.45 : 1,
                      }}
                    >
                      <button
                        onClick={() => choose(d.id)}
                        disabled={d.gated}
                        className="focus-ring min-w-0 flex-1 rounded px-2 py-1.5 text-left transition-colors"
                        style={{ background: 'transparent', cursor: d.gated ? 'not-allowed' : 'pointer' }}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="mono truncate text-[11.5px]" style={{ color: 'var(--text)' }}>
                            {d.id}
                          </span>
                          <span className="shrink-0 text-[10px]" style={{ color: 'var(--text-3)' }}>
                            {d.gated ? 'gated' : `${fmtCount(d.downloads)} downloads · ♥ ${d.likes}`}
                          </span>
                        </div>
                      </button>
                      <HFLink path={`datasets/${d.id}`} title={`Open ${d.id} on Hugging Face`} />
                    </div>
                  ))}
                </div>
              )}

              {picked && (
                <div className="mt-4 rounded-lg p-3" style={{ background: 'var(--bg-2)', border: '1px solid var(--border)' }}>
                  <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                    <span className="mono truncate text-[12px]" style={{ color: 'var(--accent)' }}>
                      {picked}
                    </span>
                    <span className="flex shrink-0 items-center gap-1">
                      <HFLink path={`datasets/${picked}`} label="dataset card" title="Licence, description and provenance" />
                      <HFLink
                        path={`datasets/${picked}/viewer`}
                        label="browse the rows live"
                        title="See the actual rows in the Hugging Face viewer before loading them"
                      />
                    </span>
                  </div>
                  {!splits ? (
                    <div className="flex items-center gap-2 text-[12px]" style={{ color: 'var(--text-3)' }}>
                      <Spinner /> reading splits…
                    </div>
                  ) : (
                    <div className="grid gap-3 md:grid-cols-3">
                      <Field label="split" info={V.hfDataset}>
                        <Select
                          value={String(splitIdx)}
                          onChange={(v) => setSplitIdx(parseInt(v, 10))}
                          options={splits.map((s, i) => ({
                            id: String(i),
                            label: `${s.config}/${s.split}${s.numRows ? ` (${fmtCount(s.numRows)})` : ''}`,
                          }))}
                        />
                      </Field>
                      <Field label="text column" info={V.hfDataset} hint="The column holding the text to train on.">
                        <input value={column} onChange={(e) => setColumn(e.target.value)} className="mono" />
                      </Field>
                      <div>
                        <Slider
                          label="how much to load"
                          value={Math.log10(maxChars)}
                          min={4}
                          max={6}
                          step={0.05}
                          format={(x) => `${fmtInt(Math.pow(10, x))} chars`}
                          onChange={(x) => setMaxChars(Math.round(Math.pow(10, x)))}
                          info={V.maxChars}
                        />
                      </div>
                    </div>
                  )}
                  <div className="mt-3 flex items-center gap-2">
                    <Btn variant="primary" onClick={doLoad} disabled={!splits || loading}>
                      {loading ? 'Loading…' : 'Load this dataset'}
                    </Btn>
                    {loadMsg && (
                      <span className="mono text-[11px]" style={{ color: 'var(--text-3)' }}>
                        {loadMsg}
                      </span>
                    )}
                  </div>
                </div>
              )}

              {err && (
                <div className="mt-3">
                  <Callout tone="err" title="Could not load that">
                    {err}
                    <p className="mt-1.5" style={{ color: 'var(--text-3)' }}>
                      Gated and private datasets cannot be read from a browser. Some datasets also need a
                      specific config name, and some have no text column at all.
                    </p>
                  </Callout>
                </div>
              )}
            </Panel>

            <Panel title="Or bring your own text">
              <div className="mb-3 flex flex-wrap gap-1.5">
                {CORPORA.map((c) => (
                  <Btn
                    key={c.id}
                    size="sm"
                    active={corpus.label === c.label}
                    onClick={() => {
                      setCorpus({ text: getCorpus(c.id, 12), label: c.label, source: 'built in' });
                      setBuilt(null);
                    }}
                  >
                    {c.label}
                  </Btn>
                ))}
              </div>
              <textarea
                value={pasted}
                onChange={(e) => setPasted(e.target.value)}
                rows={4}
                placeholder="Paste any text here: your own writing, documentation, lyrics you wrote, logs. A few tens of thousands of characters is plenty."
                className="mono"
              />
              <div className="mt-2">
                <Btn
                  onClick={() => {
                    setCorpus({ text: pasted, label: 'Pasted text', source: `${fmtInt(pasted.length)} characters` });
                    setBuilt(null);
                  }}
                  disabled={pasted.length < 500}
                >
                  Use this text {pasted.length > 0 && pasted.length < 500 ? '(needs 500+ characters)' : ''}
                </Btn>
              </div>
            </Panel>
          </div>

          <div className="space-y-4">
            <Panel title="Current corpus" right={<Badge tone="accent">{fmtInt(corpus.text.length)} chars</Badge>}>
              <div className="space-y-1 text-[11.5px]">
                <div className="flex justify-between">
                  <span style={{ color: 'var(--text-3)' }}>name</span>
                  <span className="mono truncate" style={{ color: 'var(--text-2)', maxWidth: 200 }}>
                    {corpus.label}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span style={{ color: 'var(--text-3)' }}>source</span>
                  <span className="mono truncate" style={{ color: 'var(--text-2)', maxWidth: 200 }}>
                    {corpus.source}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span style={{ color: 'var(--text-3)' }}>distinct characters</span>
                  <span className="mono" style={{ color: 'var(--text-2)' }}>
                    {new Set(corpus.text.slice(0, 20000)).size}
                  </span>
                </div>
              </div>
              <div
                className="mono mt-3 max-h-[220px] overflow-y-auto rounded p-2 text-[10.5px] leading-relaxed"
                style={{ background: 'var(--bg-2)', border: '1px solid var(--border)', color: 'var(--text-3)' }}
              >
                {corpus.text.slice(0, 1400)}
                {corpus.text.length > 1400 && '…'}
              </div>
            </Panel>

            <Callout tone="insight">
              Data is the part people underestimate. A small model on clean, repetitive text will look far more
              impressive than a larger one on messy prose, because it has less to explain. If the output later
              looks like nonsense, suspect the data before the architecture.
            </Callout>
          </div>
        </div>
      )}

      {/* ============================================================== design */}
      {tab === 'design' && (
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
          <div className="space-y-4">
            <Panel title="Design your model" subtitle="Every number here is checked against your budget before anything runs">
              <div className="grid gap-4 md:grid-cols-2">
                <Slider
                  label="model width (d_model)"
                  value={dModel}
                  min={8}
                  max={256}
                  step={8}
                  format={String}
                  onChange={(v) => setDModel(Math.round(v))}
                  info={V.dModel}
                />
                <Slider
                  label="layers"
                  value={nLayers}
                  min={1}
                  max={8}
                  step={1}
                  format={String}
                  onChange={(v) => setNLayers(Math.round(v))}
                  info={V.nLayers}
                />
                <Slider
                  label="attention heads"
                  value={nHeads}
                  min={1}
                  max={16}
                  step={1}
                  format={String}
                  onChange={(v) => setNHeads(Math.round(v))}
                  info={V.nHeads}
                />
                <Slider
                  label="feed-forward width"
                  value={dFF}
                  min={16}
                  max={1024}
                  step={16}
                  format={String}
                  onChange={(v) => setDFF(Math.round(v))}
                  info={V.dFF}
                />
                <Slider
                  label="context window"
                  value={blockSize}
                  min={8}
                  max={512}
                  step={8}
                  format={(x) => `${Math.round(x)} tokens`}
                  onChange={(v) => setBlockSize(Math.round(v))}
                  info={V.blockSize}
                />
                <Slider
                  label="tokenizer merges"
                  value={merges}
                  min={0}
                  max={600}
                  step={10}
                  format={String}
                  onChange={(v) => setMerges(Math.round(v))}
                  info={V.merges}
                />
                <Slider
                  label="sequences per batch"
                  value={batchSeqs}
                  min={1}
                  max={32}
                  step={1}
                  format={String}
                  onChange={(v) => setBatchSeqs(Math.round(v))}
                  info={V.batchSeqs}
                />
                <Slider
                  label="training steps"
                  value={steps}
                  min={50}
                  max={8000}
                  step={50}
                  format={String}
                  onChange={(v) => setSteps(Math.round(v))}
                  info={V.trainSteps}
                />
              </div>

              {!headsOk && (
                <div className="mt-3">
                  <Callout tone="err" title="Heads must divide the width">
                    {dModel} does not divide evenly by {nHeads}. Each head takes an equal slice of the model
                    width, so pick a head count that divides {dModel}.
                  </Callout>
                </div>
              )}

              <div className="mt-4">
                <StepCost cfg={cfg} batchSeqs={batchSeqs} steps={steps} frameBudgetMs={frameBudgetMs} />
              </div>

              <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                <Stat label="parameters" value={fmtCount(paramCountFor(cfg))} tone="accent" />
                <Stat label="estimated vocab" value={estVocab} sub={built ? 'measured' : 'estimate'} />
                <Stat label="head width" value={Math.floor(dModel / Math.max(1, nHeads))} />
                <Stat
                  label="tokens for a full run"
                  value={fmtCount(steps * batchSeqs * blockSize)}
                  sub={`guidance: ${fmtCount(suggestedTokens(paramCountFor(cfg)))}`}
                />
              </div>

              <div className="mt-4 flex items-center gap-2">
                <Btn variant="primary" onClick={build} disabled={!canBuild || building}>
                  {building ? 'Building tokenizer and model…' : 'Build this model'}
                </Btn>
                {!canBuild && (
                  <span className="text-[11.5px]" style={{ color: 'var(--warn)' }}>
                    {!headsOk ? 'Fix the head count first.' : budget.verdict === 'over' ? 'Over budget.' : 'Load some data first.'}
                  </span>
                )}
              </div>
            </Panel>

            <Panel title="What your choices cost">
              <Depth
                plain={
                  <>
                    <p className="mb-2">
                      Width is the expensive one. Doubling it roughly quadruples the parameters, because almost
                      every weight matrix has width on both sides. Depth is linear: twice the layers is twice
                      the parameters and twice the work. Context window is the sneaky one, because attention
                      memory grows with its square.
                    </p>
                    <p>
                      A useful habit is to change one thing, watch the parameter count and memory bar move, and
                      build an instinct for which knob is actually expensive.
                    </p>
                  </>
                }
                math={
                  <>
                    <p className="mb-2">
                      Parameters ≈ V·d + ctx·d + n_layers·(4d² + 2d·d_ff) + d·V. With your settings that is{' '}
                      {estVocab}·{dModel} + {blockSize}·{dModel} + {nLayers}·(4·{dModel}² + 2·{dModel}·{dFF}) +{' '}
                      {dModel}·{estVocab} = {fmtInt(paramCountFor(cfg))}.
                    </p>
                    <p>
                      Activation memory is dominated by n_layers · n_heads · T² for the attention maps, which at
                      T = {blockSize} and {nHeads} heads across {nLayers} layers is{' '}
                      {fmtInt(nLayers * nHeads * blockSize * blockSize * 2)} numbers held live per sequence.
                    </p>
                  </>
                }
                code={
                  <Depth
                    plain={
                      <span className="mono text-[11px]">
                        const perLayer = 4*d*d + 2*d*dFF + 6*d + dFF;
                      </span>
                    }
                  />
                }
              />
            </Panel>
          </div>

          <div className="space-y-4">
            <Panel title="Your machine" subtitle="What the browser will tell us about it">
              <div className="space-y-1.5 text-[11.5px]">
                {[
                  ['reported RAM', machine.deviceMemoryGB ? `${machine.deviceMemoryGB} GB (approx)` : 'not exposed'],
                  ['logical cores', machine.logicalCores ?? 'not exposed'],
                  ['tab heap ceiling', machine.jsHeapLimitMB ? `${machine.jsHeapLimitMB.toFixed(0)} MB` : 'not exposed'],
                  ['heap in use now', heapMB ? `${heapMB.toFixed(0)} MB` : 'not exposed'],
                  ['platform', machine.platform],
                ].map(([k, v]) => (
                  <div key={k as string} className="flex justify-between">
                    <span style={{ color: 'var(--text-3)' }}>{k as string}</span>
                    <span className="mono" style={{ color: 'var(--text-2)' }}>
                      {String(v)}
                    </span>
                  </div>
                ))}
              </div>
              <p className="mt-2 text-[10.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                Browsers deliberately report RAM coarsely and cap it at 8 GB, so treat it as a hint. The heap
                ceiling is the number that actually matters: it is the hard limit on this one tab.
              </p>
            </Panel>

            <Panel
              title="Resource budget"
              subtitle="Set the ceiling. Glassbox refuses to start a run that would exceed it."
            >
              <div className="space-y-4">
                <Slider
                  label="RAM budget for this tab"
                  value={ramBudgetMB}
                  min={100}
                  max={Math.round(machine.jsHeapLimitMB ? machine.jsHeapLimitMB * 0.85 : 3000)}
                  step={50}
                  format={(x) => `${Math.round(x)} MB`}
                  onChange={(x) => setRamBudgetMB(Math.round(x))}
                  info={V.ramBudget}
                />
                <Slider
                  label="CPU per frame"
                  value={frameBudgetMs}
                  min={2}
                  max={14}
                  step={1}
                  format={(x) => `${Math.round(x)} ms`}
                  onChange={(x) => setFrameBudgetMs(Math.round(x))}
                  info={V.frameBudget}
                />
              </div>

              <div className="mt-4 space-y-1.5">
                {([
                  ['weights', mem.weightsMB, 'var(--accent)'],
                  ['gradients', mem.gradsMB, 'var(--pos)'],
                  ['optimizer (Adam)', mem.optimiserMB, 'var(--ok)'],
                  ['activations', mem.activationsMB, 'var(--warn)'],
                  ['tokenized corpus', mem.tokenizerMB, 'var(--text-3)'],
                ] as [string, number, string][]).map(([l, v, c]) => (
                  <div key={l}>
                    <div className="mb-0.5 flex justify-between text-[11px]">
                      <span style={{ color: 'var(--text-3)' }}>{l}</span>
                      <span className="mono">{v.toFixed(1)} MB</span>
                    </div>
                    <BarMeter value={v} max={ramBudgetMB} color={c} height={5} />
                  </div>
                ))}
              </div>

              <div className="mt-3">
                <div className="mb-1 flex justify-between text-[11.5px]">
                  <span style={{ color: 'var(--text-2)' }}>total</span>
                  <span
                    className="mono font-semibold"
                    style={{
                      color:
                        budget.verdict === 'over' ? 'var(--err)' : budget.verdict === 'tight' ? 'var(--warn)' : 'var(--ok)',
                    }}
                  >
                    {mem.totalMB.toFixed(0)} MB
                  </span>
                </div>
                <ProgressBar
                  value={budget.fraction}
                  tone={budget.verdict === 'over' ? 'warn' : budget.verdict === 'tight' ? 'warn' : 'ok'}
                />
              </div>

              <div className="mt-3">
                <Callout tone={budget.verdict === 'over' ? 'err' : budget.verdict === 'tight' ? 'warn' : 'info'}>
                  {budget.message}
                </Callout>
              </div>

              <div className="mt-3 rounded-lg p-2.5 text-[11px] leading-relaxed" style={{ background: 'var(--bg-2)', border: '1px solid var(--border)', color: 'var(--text-3)' }}>
                Glassbox stores every number as a 64-bit float for clarity, which is four times what a real
                training run uses. The same model trained properly in BF16 with FP32 optimizer states would
                need about <span className="mono" style={{ color: 'var(--text-2)' }}>{mem.realWorldBf16MB.toFixed(1)} MB</span>{' '}
                instead of {mem.totalMB.toFixed(0)} MB.
              </div>
            </Panel>

            <DevicePanel
              backend={backend}
              onBackend={setBackend}
              modelMatrixSize={Math.max(dFF, dModel, estVocab)}
            />
          </div>
        </div>
      )}

      {/* =============================================================== train */}
      {tab === 'train' && (
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
          {!built ? (
            <Empty>Build a model on the previous tab first.</Empty>
          ) : (
            <>
              <div className="space-y-4">
                <Panel
                  title="Your model, training on your data"
                  subtitle={`${fmtCount(built.model.paramCount)} parameters · vocabulary ${built.vocabSize} · ${fmtInt(built.trainIds.length)} training tokens`}
                  right={
                    <div className="flex items-center gap-1.5">
                      <BackgroundNotice running={running} />
                      <Btn variant="primary" size="sm" onClick={() => setRunning((r) => !r)} disabled={built.status === 'done'}>
                        {running ? 'Pause' : built.step > 0 ? 'Resume' : 'Train'}
                      </Btn>
                      <Btn
                        size="sm"
                        onClick={() => {
                          setRunning(false);
                          built.reset();
                          setSamples([]);
                          setTick((t) => t + 1);
                        }}
                      >
                        Reset
                      </Btn>
                    </div>
                  }
                >
                  <ProgressBar value={built.progress} tone={built.status === 'done' ? 'ok' : 'accent'} />
                  <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
                    <Stat label="step" value={`${built.step} / ${built.cfg.steps}`} />
                    <Stat label="loss" value={fmt(last?.loss ?? NaN, 4)} tone="accent" />
                    <Stat label="perplexity" value={fmt(last?.perplexity ?? NaN, 2)} />
                    <Stat
                      label="throughput"
                      value={fmtInt(measuredTokPerSec)}
                      unit="tok/s"
                      sub={heapMB ? `heap ${heapMB.toFixed(0)} MB` : undefined}
                    />
                  </div>

                  <div className="mt-4 grid gap-4 md:grid-cols-2">
                    <div>
                      <Label>loss</Label>
                      <LineChart
                        series={
                          [
                            { id: 'l', label: 'train', color: 'var(--accent)', points: built.history.map((m) => ({ x: m.step, y: m.loss })) },
                            { id: 'v', label: 'validation', color: 'var(--pos)', dashed: true, points: built.history.map((m) => ({ x: m.step, y: m.valLoss })) },
                          ] as Series[]
                        }
                        height={150}
                        xLabel="step"
                        logY
                      />
                    </div>
                    <div>
                      <Label>perplexity</Label>
                      <LineChart
                        series={[{ id: 'p', label: 'perplexity', color: 'var(--ok)', points: built.history.map((m) => ({ x: m.step, y: m.perplexity })) }]}
                        height={150}
                        xLabel="step"
                        yMin={1}
                      />
                    </div>
                  </div>

                  <div className="mt-3">
                    <Slider
                      label="CPU per frame"
                      value={frameBudgetMs}
                      min={2}
                      max={14}
                      step={1}
                      format={(x) => `${Math.round(x)} ms`}
                      onChange={(x) => setFrameBudgetMs(Math.round(x))}
                      info={V.frameBudget}
                    />
                  </div>
                </Panel>

                <Diagnostics
                  facts={
                    {
                      history: built.history,
                      vocab: built.vocabSize,
                      paramCount: built.model.paramCount,
                      trainTokens: built.trainIds.length,
                      blockSize: built.model.cfg.blockSize,
                      totalSteps: built.cfg.steps,
                      corpusChars: corpus.text.length,
                    } satisfies RunFacts
                  }
                  sample={samples.length ? samples[samples.length - 1].text : null}
                  corpus={corpus.text}
                />

                <Panel title="What it writes" subtitle="Sampled from your model every 50 steps">
                  {samples.length === 0 ? (
                    <Empty>Start training to see samples.</Empty>
                  ) : (
                    <div className="space-y-2">
                      {[...samples].reverse().slice(0, 8).map((s, i) => (
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
              </div>

              <div className="space-y-4">
                <Panel title="Live resource use">
                  <div className="space-y-3">
                    <div>
                      <div className="mb-1 flex justify-between text-[11px]">
                        <span style={{ color: 'var(--text-3)' }}>heap in use</span>
                        <span className="mono">{heapMB ? `${heapMB.toFixed(0)} MB` : 'not exposed'}</span>
                      </div>
                      <BarMeter value={heapMB ?? 0} max={ramBudgetMB} color={heapMB && heapMB > ramBudgetMB ? 'var(--err)' : 'var(--ok)'} />
                    </div>
                    <div>
                      <div className="mb-1 flex justify-between text-[11px]">
                        <span style={{ color: 'var(--text-3)' }}>frame budget</span>
                        <span className="mono">{frameBudgetMs} of ~16 ms</span>
                      </div>
                      <BarMeter value={frameBudgetMs} max={16} color="var(--accent)" />
                    </div>
                  </div>
                  <p className="mt-3 text-[11px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                    Training only ever runs inside the slice you allowed, then hands the frame back to the
                    browser. That is why the interface stays responsive and why the run takes as long as it
                    does. It is the same trade a real cluster makes between throughput and everything else.
                  </p>
                </Panel>

                <Panel title="Your configuration">
                  <div className="space-y-1.5 text-[11.5px]">
                    {[
                      ['corpus', corpus.label],
                      ['characters', fmtInt(corpus.text.length)],
                      ['vocabulary', String(built.vocabSize)],
                      ['train tokens', fmtInt(built.trainIds.length)],
                      ['held out', fmtInt(built.valIds.length)],
                      ['parameters', fmtCount(built.model.paramCount)],
                      ['width', String(built.model.cfg.dModel)],
                      ['layers', String(built.model.cfg.nLayers)],
                      ['heads', String(built.model.cfg.nHeads)],
                      ['context', String(built.model.cfg.blockSize)],
                    ].map(([k, v]) => (
                      <div key={k} className="flex justify-between">
                        <span style={{ color: 'var(--text-3)' }}>{k}</span>
                        <span className="mono truncate" style={{ color: 'var(--text-2)', maxWidth: 170 }}>
                          {v}
                        </span>
                      </div>
                    ))}
                  </div>
                </Panel>
              </div>
            </>
          )}
        </div>
      )}

      {/* ============================================================= compare */}
      {tab === 'compare' && <CompareTab built={built} measuredTokPerSec={measuredTokPerSec} />}
    </div>
  );
}

