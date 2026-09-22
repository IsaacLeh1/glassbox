import { useCallback, useMemo, useRef, useState } from 'react';
import {
  METRICS,
  OPTIMIZERS,
  effectiveOutputs,
  evaluate,
  optimize,
  type EvalSummary,
  type OptimizerEvent,
  type OptimizerId,
  type OptimizeResult,
  type Program,
} from '../dspy/dspy';
import { LMClient, PRESETS, type LMCall } from '../dspy/lmClient';
import { TASKS, splitExamples } from '../dspy/tasks';
import { toJSON, toPython, toPromptText } from '../dspy/export';
import { useApp } from '../store/app';
import {
  Badge,
  Btn,
  Callout,
  Code,
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
  Toggle,
  fmt,
  fmtInt,
  fmtPct,
} from '../ui/kit';
import { V } from '../content/varInfo';
import { BarMeter } from '../ui/viz';

type Tab = 'connect' | 'task' | 'optimize' | 'compare' | 'export';

export default function DSPyLab() {
  const [tab, setTab] = useState<Tab>('connect');
  const lm = useApp((s) => s.lm);
  const setLM = useApp((s) => s.setLM);

  const clientRef = useRef<LMClient | null>(null);
  if (!clientRef.current) clientRef.current = new LMClient(lm);
  clientRef.current.cfg = lm;
  const client = clientRef.current;

  const [taskId, setTaskId] = useState(TASKS[0].id);
  const task = TASKS.find((t) => t.id === taskId)!;
  const [instructions, setInstructions] = useState(task.signature.instructions);
  const [cot, setCot] = useState(task.suggestCot);
  const [metricId, setMetricId] = useState(task.metricId);
  const [optimizerId, setOptimizerId] = useState<OptimizerId>('bootstrap');
  const [maxDemos, setMaxDemos] = useState(4);
  const [candidates, setCandidates] = useState(3);
  const [rounds, setRounds] = useState(1);

  const [conn, setConn] = useState<{ ok: boolean; detail: string; latencyMs: number } | null>(null);
  const [testing, setTesting] = useState(false);
  const [busy, setBusy] = useState<false | 'baseline' | 'optimize'>(false);
  const [events, setEvents] = useState<OptimizerEvent[]>([]);
  const [progress, setProgress] = useState(0);
  const [baseEval, setBaseEval] = useState<EvalSummary | null>(null);
  const [optEval, setOptEval] = useState<EvalSummary | null>(null);
  const [result, setResult] = useState<OptimizeResult | null>(null);
  const [calls, setCalls] = useState<LMCall[]>([]);
  const abortRef = useRef<AbortController | null>(null);

  const split = useMemo(() => splitExamples(task), [task]);
  const metricDef = METRICS.find((m) => m.id === metricId)!;
  const metric = useMemo(() => metricDef.fn(task.metricField), [metricDef, task.metricField]);

  const baseProgram: Program = useMemo(
    () => ({ signature: { ...task.signature, instructions }, demos: [], cot }),
    [task, instructions, cot],
  );

  const switchTask = (id: string) => {
    const t = TASKS.find((x) => x.id === id)!;
    setTaskId(id);
    setInstructions(t.signature.instructions);
    setCot(t.suggestCot);
    setMetricId(t.metricId);
    setBaseEval(null);
    setOptEval(null);
    setResult(null);
    setEvents([]);
  };

  const connected = conn?.ok === true;

  const runTest = async () => {
    setTesting(true);
    setConn(null);
    client.clearCache();
    const r = await client.test();
    setConn(r);
    setCalls([...client.calls]);
    setTesting(false);
  };

  const runBaseline = useCallback(async () => {
    setBusy('baseline');
    setProgress(0);
    abortRef.current = new AbortController();
    try {
      const r = await evaluate(client, baseProgram, split.valset, metric, {
        tag: 'baseline',
        signal: abortRef.current.signal,
        onProgress: (done, total) => setProgress(done / total),
      });
      setBaseEval(r);
      setOptEval(null);
      setResult(null);
    } finally {
      setCalls([...client.calls]);
      setBusy(false);
    }
  }, [client, baseProgram, split.valset, metric]);

  const runOptimize = useCallback(async () => {
    setBusy('optimize');
    setEvents([]);
    setProgress(0);
    abortRef.current = new AbortController();
    try {
      const res = await optimize(client, baseProgram, optimizerId, {
        trainset: split.trainset,
        valset: split.valset,
        metric,
        maxDemos,
        candidates,
        rounds,
        minibatch: Math.max(4, Math.floor(split.valset.length / 2)),
        signal: abortRef.current.signal,
        onEvent: (e) => setEvents((prev) => [...prev, e]),
      });
      setResult(res);
      const after = await evaluate(client, res.program, split.valset, metric, {
        tag: 'after',
        signal: abortRef.current.signal,
        onProgress: (done, total) => setProgress(done / total),
      });
      setOptEval(after);
      if (!baseEval) {
        const before = await evaluate(client, baseProgram, split.valset, metric, { tag: 'before' });
        setBaseEval(before);
      }
      setTab('compare');
    } catch (err) {
      setEvents((p) => [...p, { kind: 'error', msg: err instanceof Error ? err.message : String(err) }]);
    } finally {
      setCalls([...client.calls]);
      setBusy(false);
    }
  }, [client, baseProgram, optimizerId, split, metric, maxDemos, candidates, rounds, baseEval]);

  const cancel = () => {
    abortRef.current?.abort();
    setBusy(false);
  };

  /* ------------------------------------------------------------ render */

  return (
    <div className="mx-auto max-w-[1400px] px-6 py-6">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <Segmented
          value={tab}
          onChange={setTab}
          options={[
            { id: 'connect' as Tab, label: '1 · Connect a model' },
            { id: 'task' as Tab, label: '2 · Define the task' },
            { id: 'optimize' as Tab, label: '3 · Optimize' },
            { id: 'compare' as Tab, label: '4 · Before and after' },
            { id: 'export' as Tab, label: '5 · Export' },
          ]}
        />
        <div className="flex items-center gap-2">
          {busy && <Spinner />}
          <Badge tone={connected ? 'ok' : 'neutral'}>{connected ? lm.model : 'not connected'}</Badge>
          <Badge>{fmtInt(client.totalPromptTokens + client.totalCompletionTokens)} tokens used</Badge>
        </div>
      </div>

      {/* ============================================================ connect */}
      {tab === 'connect' && (
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
          <Panel
            title="Point Glassbox at a model you control"
            subtitle="Any OpenAI-compatible endpoint works: your UVU gateway, OpenAI, OpenRouter, or a model running locally through Ollama or LM Studio."
          >
            <div className="mb-4 flex flex-wrap gap-1.5">
              {PRESETS.map((p) => (
                <Btn
                  key={p.id}
                  size="sm"
                  active={lm.baseUrl === p.baseUrl}
                  onClick={() => setLM({ baseUrl: p.baseUrl, model: p.model })}
                  title={p.note}
                >
                  {p.label}
                </Btn>
              ))}
            </div>

            <div className="grid gap-3 md:grid-cols-2">
              <Field label="base URL" info={V.lmBaseUrl} hint="Must end at /v1. The path /chat/completions is appended.">
                <input value={lm.baseUrl} onChange={(e) => setLM({ baseUrl: e.target.value })} className="mono" />
              </Field>
              <Field label="model" info={V.lmModel}>
                <input value={lm.model} onChange={(e) => setLM({ model: e.target.value })} className="mono" />
              </Field>
              <Field label="API key" hint="Stored in this browser only and sent only to the base URL above.">
                <input
                  type="password"
                  value={lm.apiKey}
                  onChange={(e) => setLM({ apiKey: e.target.value })}
                  placeholder="paste your own key"
                  className="mono"
                  autoComplete="off"
                />
              </Field>
              <Field label="auth header style">
                <Select
                  value={lm.authHeader}
                  onChange={(v) => setLM({ authHeader: v as typeof lm.authHeader })}
                  options={[
                    { id: 'bearer', label: 'Authorization: Bearer' },
                    { id: 'x-api-key', label: 'x-api-key' },
                    { id: 'none', label: 'none (local model)' },
                  ]}
                />
              </Field>
            </div>

            <div className="mt-4 grid gap-4 md:grid-cols-2">
              <Slider
                label="temperature" info={V.lmTemperature}
                value={lm.temperature}
                min={0}
                max={1}
                step={0.05}
                onChange={(t) => setLM({ temperature: t })}
                hint="Keep at 0 while optimizing, so score differences come from the prompt and not from luck."
              />
              <Slider
                label="max tokens" info={V.lmMaxTokens}
                value={lm.maxTokens}
                min={64}
                max={2048}
                step={64}
                format={(x) => String(Math.round(x))}
                onChange={(t) => setLM({ maxTokens: Math.round(t) })}
              />
            </div>

            <div className="mt-4 flex items-center gap-2">
              <Btn variant="primary" onClick={runTest} disabled={testing}>
                {testing ? 'Testing…' : 'Test connection'}
              </Btn>
              {conn && (
                <span className="text-[12px]" style={{ color: conn.ok ? 'var(--ok)' : 'var(--err)' }}>
                  {conn.ok ? `replied in ${conn.latencyMs.toFixed(0)} ms: "${conn.detail}"` : conn.detail}
                </span>
              )}
            </div>

            {conn && !conn.ok && (
              <div className="mt-3">
                <Callout tone="err" title="Could not reach the model">
                  <p className="mb-1.5">Common causes, in the order worth checking:</p>
                  <ul className="list-inside list-disc space-y-1">
                    <li>The base URL is missing /v1, or has a trailing path it should not have.</li>
                    <li>
                      CORS. A browser can only call an endpoint that allows cross-origin requests. Gateways
                      usually do; some corporate proxies do not. Ollama needs OLLAMA_ORIGINS set.
                    </li>
                    <li>The key is wrong, expired, or lacks access to that model name.</li>
                  </ul>
                </Callout>
              </div>
            )}

            <div className="mt-4">
              <Callout tone="warn" title="About your key">
                It is kept in this browser&apos;s local storage and attached only to requests made to the base
                URL you entered. It is not sent anywhere else and never appears in exported code, which reads
                it from an environment variable instead. Even so, a key in browser storage is a key on disk:
                use a scoped one, and rotate it when you are finished.
              </Callout>
            </div>
          </Panel>

          <div className="space-y-4">
            <Panel title="What this module does">
              <Depth
                plain={
                  <>
                    <p className="mb-2">
                      Prompt engineering is usually done by hand: try a wording, read a few outputs, decide it
                      feels better, move on. DSPy replaces that with a measurement. You define what the task
                      takes in and gives out, supply some labelled examples, and choose a way of scoring an
                      answer. Then an optimizer tries variations and keeps whatever measurably scores higher.
                    </p>
                    <p>
                      The result is an ordinary prompt, but one that was selected rather than guessed. You will
                      see the before score, the after score, and exactly what changed between them.
                    </p>
                  </>
                }
                math={
                  <>
                    <p className="mb-2">
                      Treat the prompt as a parameter vector <span className="mono">φ</span> containing the
                      instruction string and the demonstration set. Given a metric{' '}
                      <span className="mono">m</span> and data <span className="mono">D</span>, the optimizer
                      searches for
                    </p>
                    <div className="mono my-2 text-center">φ* = argmax E₍x,y∼D₎ [ m(y, f(x; φ)) ]</div>
                    <p>
                      The search space is discrete and the objective is not differentiable, so the methods are
                      all forms of guided search: bootstrap from successful traces, propose-and-score for
                      instructions, and random search over combinations.
                    </p>
                  </>
                }
                code={
                  <Code>{`// dspy/dspy.ts
const baseEval = await evaluate(client, base, opts.valset, opts.metric);
// ... build candidates ...
const s = await evaluate(client, cand, opts.valset, opts.metric);
if (s > bestScore) { bestScore = s; best = cand; }   // never regress`}</Code>
                }
              />
            </Panel>

            <Panel title="How faithful is this?">
              <p className="text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                This is a TypeScript reimplementation of DSPy, not a binding to the Python package. The
                signature model, the <span className="mono">[[ ## field ## ]]</span> chat format, and the
                BootstrapFewShot, COPRO and MIPRO search procedures follow the real framework closely enough
                that what you learn transfers directly. The Export tab emits the equivalent genuine DSPy
                program so you can reproduce any run with the real library.
              </p>
            </Panel>

            {calls.length > 0 && (
              <Panel title="Request log" subtitle={`${calls.length} calls this session`}>
                <div className="max-h-[220px] space-y-1 overflow-y-auto pr-1">
                  {[...calls].reverse().slice(0, 40).map((c) => (
                    <div key={c.id} className="mono flex items-center justify-between gap-2 text-[10.5px]">
                      <span style={{ color: c.error ? 'var(--err)' : 'var(--text-3)' }}>{c.tag}</span>
                      <span style={{ color: 'var(--text-3)' }}>
                        {c.promptTokens}+{c.completionTokens} tok · {c.latencyMs.toFixed(0)} ms
                      </span>
                    </div>
                  ))}
                </div>
              </Panel>
            )}
          </div>
        </div>
      )}

      {/* =============================================================== task */}
      {tab === 'task' && (
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
          <div className="space-y-4">
            <Panel title="Pick a task" subtitle="Each one fails cold in a specific, visible way">
              <div className="flex flex-wrap gap-1.5">
                {TASKS.map((t) => (
                  <Btn key={t.id} size="sm" active={t.id === taskId} onClick={() => switchTask(t.id)}>
                    {t.label}
                  </Btn>
                ))}
              </div>
              <p className="mt-3 text-[12.5px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                {task.blurb}
              </p>
              <div className="mt-3">
                <Callout tone="insight" title="Why a cold prompt struggles here">
                  {task.whyItFails}
                </Callout>
              </div>
            </Panel>

            <Panel title="The signature" subtitle="What goes in, what must come out, and the objective">
              <div className="grid gap-3 md:grid-cols-2">
                <div>
                  <Label>inputs</Label>
                  <div className="mt-1 space-y-1">
                    {task.signature.inputs.map((f) => (
                      <div key={f.name} className="rounded p-2" style={{ background: 'var(--panel-2)', border: '1px solid var(--border)' }}>
                        <div className="mono text-[11.5px]" style={{ color: 'var(--accent)' }}>
                          {f.name}
                        </div>
                        <div className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                          {f.desc}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <Label>outputs</Label>
                  <div className="mt-1 space-y-1">
                    {effectiveOutputs(baseProgram).map((f) => (
                      <div key={f.name} className="rounded p-2" style={{ background: 'var(--panel-2)', border: '1px solid var(--border)' }}>
                        <div className="mono text-[11.5px]" style={{ color: 'var(--pos)' }}>
                          {f.name}
                          {f.name === 'reasoning' && (
                            <span className="ml-1 text-[9.5px]" style={{ color: 'var(--text-3)' }}>
                              added by chain of thought
                            </span>
                          )}
                        </div>
                        <div className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                          {f.desc}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="mt-4">
                <Field label="instructions" info={V.instructions} hint="This is what the optimizer will try to improve on.">
                  <textarea value={instructions} onChange={(e) => setInstructions(e.target.value)} rows={2} />
                </Field>
              </div>

              <div className="mt-3 grid gap-3 md:grid-cols-2">
                <div>
                  <Toggle
                    checked={cot}
                    onChange={setCot}
                    label="Chain of thought" info={V.chainOfThought}
                    hint="Adds a reasoning field the model must fill before answering."
                  />
                  <p className="mt-1 px-2 text-[11px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                    Forces the model to write its working first. Costs more tokens, and usually pays for itself
                    on anything multi-step.
                  </p>
                </div>
                <div>
                  <Field label="metric" info={V.metric} hint={metricDef.blurb}>
                    <Select
                      value={metricId}
                      onChange={setMetricId}
                      options={METRICS.map((m) => ({ id: m.id, label: m.label }))}
                    />
                  </Field>
                  <p className="mt-1 text-[11px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                    {metricDef.blurb}
                  </p>
                </div>
              </div>
            </Panel>

            <Panel title="The prompt this produces right now" subtitle="Exactly what would be sent, before any optimization">
              <Code className="max-h-[300px]">{toPromptText(baseProgram, split.valset[0]?.inputs ?? {})}</Code>
            </Panel>
          </div>

          <div className="space-y-4">
            <Panel title="Data split" subtitle="The optimizer never sees the validation set while searching">
              <div className="grid grid-cols-2 gap-3">
                <Stat label="train" value={split.trainset.length} hint="Used to build demonstrations." />
                <Stat label="validation" value={split.valset.length} hint="Used only to score candidates." />
              </div>
              <div className="mt-3 max-h-[300px] space-y-1.5 overflow-y-auto pr-1">
                {split.trainset.slice(0, 8).map((e) => (
                  <div key={e.id} className="rounded p-2 text-[11px]" style={{ background: 'var(--bg-2)', border: '1px solid var(--border)' }}>
                    {Object.entries(e.inputs).map(([k, v]) => (
                      <div key={k} className="truncate" style={{ color: 'var(--text-2)' }}>
                        <span className="mono" style={{ color: 'var(--text-3)' }}>
                          {k}:
                        </span>{' '}
                        {v}
                      </div>
                    ))}
                    {Object.entries(e.outputs).map(([k, v]) => (
                      <div key={k} className="truncate" style={{ color: 'var(--ok)' }}>
                        <span className="mono" style={{ color: 'var(--text-3)' }}>
                          {k}:
                        </span>{' '}
                        {v}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </Panel>

            <Panel title="Measure the starting point">
              <p className="mb-3 text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                Run the unoptimized program over the {split.valset.length} validation examples. This is the
                number everything else is compared against.
              </p>
              <Btn variant="primary" onClick={runBaseline} disabled={!connected || busy !== false}>
                {busy === 'baseline' ? 'Measuring…' : 'Run baseline'}
              </Btn>
              {busy === 'baseline' && (
                <div className="mt-2">
                  <ProgressBar value={progress} />
                </div>
              )}
              {!connected && (
                <p className="mt-2 text-[11.5px]" style={{ color: 'var(--warn)' }}>
                  Connect a model first.
                </p>
              )}
              {baseEval && (
                <div className="mt-3 grid grid-cols-2 gap-3">
                  <Stat label="baseline score" value={fmtPct(baseEval.score)} tone="accent" />
                  <Stat label="tokens spent" value={fmtInt(baseEval.promptTokens + baseEval.completionTokens)} />
                </div>
              )}
            </Panel>
          </div>
        </div>
      )}

      {/* =========================================================== optimize */}
      {tab === 'optimize' && (
        <div className="grid gap-4 xl:grid-cols-[360px_minmax(0,1fr)]">
          <div className="space-y-4">
            <Panel title="Choose an optimizer">
              <div className="space-y-1.5">
                {OPTIMIZERS.map((o) => (
                  <button
                    key={o.id}
                    onClick={() => setOptimizerId(o.id)}
                    className="focus-ring block w-full cursor-pointer rounded-lg p-2.5 text-left transition-colors"
                    style={{
                      background: o.id === optimizerId ? 'color-mix(in srgb, var(--accent) 10%, transparent)' : 'var(--panel-2)',
                      border: `1px solid ${o.id === optimizerId ? 'var(--accent)' : 'var(--border)'}`,
                    }}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="mono text-[12px] font-semibold" style={{ color: 'var(--text)' }}>
                        {o.label}
                      </span>
                      <Badge tone={o.id === optimizerId ? 'accent' : 'neutral'}>{o.changes}</Badge>
                    </div>
                    <p className="mt-1 text-[11.5px] leading-snug" style={{ color: 'var(--text-3)' }}>
                      {o.short}
                    </p>
                  </button>
                ))}
              </div>
              <p className="mt-3 text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                {OPTIMIZERS.find((o) => o.id === optimizerId)!.blurb}
              </p>
            </Panel>

            <Panel title="Search budget" subtitle="Every candidate costs real model calls">
              <div className="space-y-3.5">
                <Slider
                  label="max demonstrations" info={V.maxDemos}
                  value={maxDemos}
                  min={1}
                  max={8}
                  step={1}
                  format={String}
                  onChange={(v) => setMaxDemos(Math.round(v))}
                />
                <Slider
                  label="candidates per round" info={V.candidatesPerRound}
                  value={candidates}
                  min={1}
                  max={6}
                  step={1}
                  format={String}
                  onChange={(v) => setCandidates(Math.round(v))}
                />
                <Slider
                  label="rounds" info={V.rounds}
                  value={rounds}
                  min={1}
                  max={3}
                  step={1}
                  format={String}
                  onChange={(v) => setRounds(Math.round(v))}
                />
              </div>
              <div className="mt-3 rounded-lg p-2.5 text-[11.5px]" style={{ background: 'var(--bg-2)', border: '1px solid var(--border)' }}>
                <div className="flex justify-between">
                  <span style={{ color: 'var(--text-3)' }}>rough calls for this run</span>
                  <span className="mono" style={{ color: 'var(--text-2)' }}>
                    ≈{' '}
                    {optimizerId === 'labeled'
                      ? split.valset.length * 2
                      : optimizerId === 'bootstrap'
                        ? split.trainset.length + split.valset.length * 2
                        : optimizerId === 'copro'
                          ? split.valset.length * (1 + candidates * rounds) + rounds
                          : split.trainset.length * 3 + split.valset.length * 3}
                  </span>
                </div>
              </div>

              <div className="mt-3 flex gap-1.5">
                <Btn variant="primary" onClick={runOptimize} disabled={!connected || busy !== false}>
                  {busy === 'optimize' ? 'Optimizing…' : 'Run optimizer'}
                </Btn>
                {busy && <Btn onClick={cancel}>Cancel</Btn>}
              </div>
              {!connected && (
                <p className="mt-2 text-[11.5px]" style={{ color: 'var(--warn)' }}>
                  Connect a model first.
                </p>
              )}
            </Panel>
          </div>

          <div className="space-y-4">
            <Panel
              title="Search log"
              subtitle="Every candidate the optimizer built and what it actually scored"
              right={busy === 'optimize' ? <Spinner /> : undefined}
            >
              {events.length === 0 ? (
                <Empty>Run the optimizer to watch it work.</Empty>
              ) : (
                <div className="max-h-[420px] space-y-1 overflow-y-auto pr-1">
                  {events.map((e, i) => (
                    <div
                      key={i}
                      className="flex items-start gap-2 rounded px-2 py-1 text-[11.5px]"
                      style={{
                        background: e.kind === 'best' ? 'color-mix(in srgb, var(--ok) 10%, transparent)' : 'transparent',
                        color:
                          e.kind === 'error'
                            ? 'var(--err)'
                            : e.kind === 'best'
                              ? 'var(--ok)'
                              : e.kind === 'candidate'
                                ? 'var(--text-2)'
                                : 'var(--text-3)',
                      }}
                    >
                      <span className="mono shrink-0 text-[9.5px] uppercase" style={{ width: 62, opacity: 0.7 }}>
                        {e.kind}
                      </span>
                      <span className="min-w-0 flex-1">{e.msg}</span>
                      {e.score !== undefined && (
                        <span className="mono tnum shrink-0" style={{ color: 'var(--accent)' }}>
                          {fmtPct(e.score)}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </Panel>

            {result && (
              <Panel title="Everything it tried" subtitle="Including candidates that did not win">
                <div className="space-y-1.5">
                  {result.candidatePrograms.map((c, i) => (
                    <div key={i} className="flex items-center gap-2">
                      <span className="w-[210px] shrink-0 truncate text-[11.5px]" style={{ color: 'var(--text-2)' }}>
                        {c.label}
                      </span>
                      <div className="flex-1">
                        <BarMeter
                          value={c.score}
                          max={1}
                          color={c.score >= result.optimizedScore ? 'var(--ok)' : 'var(--text-3)'}
                          height={7}
                        />
                      </div>
                      <span className="mono tnum w-[48px] shrink-0 text-right text-[11px]" style={{ color: 'var(--text-2)' }}>
                        {fmtPct(c.score)}
                      </span>
                    </div>
                  ))}
                </div>
              </Panel>
            )}
          </div>
        </div>
      )}

      {/* ============================================================ compare */}
      {tab === 'compare' && (
        <div className="space-y-4">
          {!baseEval || !optEval || !result ? (
            <Empty>Run the optimizer first. The comparison appears here.</Empty>
          ) : (
            <>
              <Panel title="Before and after" subtitle={`${split.valset.length} held-out examples, same model, same data`}>
                <div className="grid gap-5 md:grid-cols-3">
                  <div className="rounded-lg p-4" style={{ background: 'var(--bg-2)', border: '1px solid var(--border)' }}>
                    <Label>before</Label>
                    <div className="mono mt-1 text-[32px] font-semibold leading-none" style={{ color: 'var(--text-3)' }}>
                      {fmtPct(baseEval.score, 0)}
                    </div>
                    <div className="mt-2 text-[11px]" style={{ color: 'var(--text-3)' }}>
                      {baseEval.results.filter((r) => r.score >= 1).length} of {baseEval.results.length} fully correct
                    </div>
                  </div>
                  <div
                    className="rounded-lg p-4"
                    style={{
                      background: 'color-mix(in srgb, var(--ok) 8%, var(--bg-2))',
                      border: '1px solid var(--ok)',
                    }}
                  >
                    <Label>after</Label>
                    <div className="mono mt-1 text-[32px] font-semibold leading-none" style={{ color: 'var(--ok)' }}>
                      {fmtPct(optEval.score, 0)}
                    </div>
                    <div className="mt-2 text-[11px]" style={{ color: 'var(--text-3)' }}>
                      {optEval.results.filter((r) => r.score >= 1).length} of {optEval.results.length} fully correct
                    </div>
                  </div>
                  <div className="space-y-3">
                    <Stat
                      label="change"
                      value={`${optEval.score >= baseEval.score ? '+' : ''}${((optEval.score - baseEval.score) * 100).toFixed(1)} pts`}
                      tone={optEval.score > baseEval.score ? 'ok' : optEval.score < baseEval.score ? 'err' : undefined}
                    />
                    <Stat
                      label="prompt tokens per call"
                      value={`${Math.round(baseEval.promptTokens / Math.max(1, baseEval.calls))} → ${Math.round(optEval.promptTokens / Math.max(1, optEval.calls))}`}
                      hint="Demonstrations make the prompt longer. That is the cost side of the trade."
                    />
                    <Stat
                      label="optimizer spent"
                      value={fmtInt(
                        client.totalPromptTokens + client.totalCompletionTokens - baseEval.promptTokens - baseEval.completionTokens,
                      )}
                      unit="tokens"
                    />
                  </div>
                </div>

                {optEval.score <= baseEval.score && (
                  <div className="mt-4">
                    <Callout tone="warn" title="No improvement, and that is a real result">
                      The optimizer could not beat the starting prompt on this validation set, so it returned
                      the original unchanged. That happens when the task is already easy for this model, when
                      the metric does not reward what the model is getting wrong, or when the validation set is
                      too small for a difference to show. Try a stricter metric, a harder task, or a different
                      optimizer.
                    </Callout>
                  </div>
                )}
              </Panel>

              <div className="grid gap-4 lg:grid-cols-2">
                <Panel title="What changed in the prompt" subtitle="Instructions">
                  <div className="space-y-2">
                    <div>
                      <Label>before</Label>
                      <div className="mono mt-1 rounded p-2 text-[11.5px]" style={{ background: 'var(--bg-2)', border: '1px solid var(--border)', color: 'var(--text-3)' }}>
                        {baseProgram.signature.instructions}
                      </div>
                    </div>
                    <div>
                      <Label>after</Label>
                      <div
                        className="mono mt-1 rounded p-2 text-[11.5px]"
                        style={{
                          background: 'color-mix(in srgb, var(--ok) 7%, var(--bg-2))',
                          border: '1px solid var(--ok)',
                          color: 'var(--text-2)',
                        }}
                      >
                        {result.program.signature.instructions}
                      </div>
                    </div>
                    {result.program.signature.instructions === baseProgram.signature.instructions && (
                      <p className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                        Unchanged. This optimizer only adjusts demonstrations.
                      </p>
                    )}
                  </div>
                </Panel>

                <Panel
                  title="Demonstrations the optimizer chose"
                  subtitle={`${result.program.demos.length} added${result.program.demos.some((d) => d.bootstrapped) ? ', generated by the model itself' : ''}`}
                >
                  {result.program.demos.length === 0 ? (
                    <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>
                      None. This run improved the instructions only.
                    </p>
                  ) : (
                    <div className="max-h-[280px] space-y-2 overflow-y-auto pr-1">
                      {result.program.demos.map((d, i) => (
                        <div key={i} className="rounded p-2 text-[11px]" style={{ background: 'var(--bg-2)', border: '1px solid var(--border)' }}>
                          <div className="mb-1">
                            <Badge tone={d.bootstrapped ? 'accent' : 'neutral'}>
                              {d.bootstrapped ? 'model generated' : 'hand labelled'}
                            </Badge>
                          </div>
                          {Object.entries(d.inputs).map(([k, v]) => (
                            <div key={k} style={{ color: 'var(--text-3)' }}>
                              <span className="mono">{k}:</span> {v}
                            </div>
                          ))}
                          {Object.entries(d.outputs).map(([k, v]) => (
                            <div key={k} style={{ color: 'var(--ok)' }}>
                              <span className="mono">{k}:</span> {String(v).slice(0, 220)}
                            </div>
                          ))}
                        </div>
                      ))}
                    </div>
                  )}
                </Panel>
              </div>

              <Panel title="Every example, side by side" subtitle="Green means the optimizer fixed it; red means it broke it">
                <div className="overflow-x-auto">
                  <table className="w-full text-[11.5px]" style={{ borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ color: 'var(--text-3)' }}>
                        {['input', 'expected', 'before', 'after', ''].map((h, i) => (
                          <th key={i} className="px-2 py-1.5 text-left font-medium" style={{ borderBottom: '1px solid var(--border)' }}>
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {baseEval.results.map((b, i) => {
                        const a = optEval.results[i];
                        if (!a) return null;
                        const delta = a.score - b.score;
                        const field = task.metricField;
                        return (
                          <tr key={b.example.id} style={{ borderBottom: '1px solid var(--border)' }}>
                            <td className="max-w-[300px] px-2 py-2" style={{ color: 'var(--text-3)' }}>
                              <div className="truncate">{Object.values(b.example.inputs)[0]}</div>
                            </td>
                            <td className="mono px-2 py-2" style={{ color: 'var(--text-2)' }}>
                              {b.example.outputs[field]}
                            </td>
                            <td className="mono max-w-[220px] px-2 py-2">
                              <div className="truncate" style={{ color: b.score >= 1 ? 'var(--ok)' : 'var(--err)' }}>
                                {b.prediction.error ? `error: ${b.prediction.error.slice(0, 40)}` : (b.prediction.outputs[field] ?? '—')}
                              </div>
                            </td>
                            <td className="mono max-w-[220px] px-2 py-2">
                              <div className="truncate" style={{ color: a.score >= 1 ? 'var(--ok)' : 'var(--err)' }}>
                                {a.prediction.error ? `error: ${a.prediction.error.slice(0, 40)}` : (a.prediction.outputs[field] ?? '—')}
                              </div>
                            </td>
                            <td className="mono px-2 py-2" style={{ color: delta > 0 ? 'var(--ok)' : delta < 0 ? 'var(--err)' : 'var(--text-3)' }}>
                              {delta > 0 ? `+${fmt(delta, 2)}` : delta < 0 ? fmt(delta, 2) : '='}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </Panel>
            </>
          )}
        </div>
      )}

      {/* ============================================================= export */}
      {tab === 'export' && (
        <div className="grid gap-4 xl:grid-cols-2">
          <Panel
            title="The equivalent real DSPy program"
            subtitle="Runs with the actual Python package, reproducing what you did here"
            right={
              <Btn
                size="sm"
                onClick={() =>
                  navigator.clipboard.writeText(
                    toPython({
                      task,
                      program: result?.program ?? baseProgram,
                      lm,
                      optimizerId,
                      metricId,
                      metricField: task.metricField,
                      trainset: split.trainset,
                      valset: split.valset,
                      maxDemos,
                      candidates,
                    }),
                  )
                }
              >
                Copy
              </Btn>
            }
          >
            <Code className="max-h-[560px]">
              {toPython({
                task,
                program: result?.program ?? baseProgram,
                lm,
                optimizerId,
                metricId,
                metricField: task.metricField,
                trainset: split.trainset,
                valset: split.valset,
                maxDemos,
                candidates,
              })}
            </Code>
          </Panel>

          <div className="space-y-4">
            <Panel
              title="The optimized prompt, as plain text"
              subtitle="Paste this anywhere. It is just a prompt in the end."
              right={
                <Btn
                  size="sm"
                  onClick={() =>
                    navigator.clipboard.writeText(toPromptText(result?.program ?? baseProgram, split.valset[0]?.inputs ?? {}))
                  }
                >
                  Copy
                </Btn>
              }
            >
              <Code className="max-h-[300px]">{toPromptText(result?.program ?? baseProgram, split.valset[0]?.inputs ?? {})}</Code>
            </Panel>

            <Panel
              title="Portable JSON"
              subtitle="Instructions and demonstrations, for loading into your own code"
              right={<Btn size="sm" onClick={() => navigator.clipboard.writeText(toJSON(result?.program ?? baseProgram))}>Copy</Btn>}
            >
              <Code className="max-h-[220px]">{toJSON(result?.program ?? baseProgram)}</Code>
            </Panel>

            <Callout tone="insight">
              The whole output of this process is a longer prompt with some examples in it. That is worth
              sitting with: the model was never touched, no weights changed, nothing was fine-tuned. All that
              changed is what gets sent, and it was chosen by measurement rather than by taste.
            </Callout>
          </div>
        </div>
      )}
    </div>
  );
}
