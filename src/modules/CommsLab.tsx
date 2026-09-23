import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Badge,
  Btn,
  Callout,
  Code,
  Depth,
  Empty,
  Field,
  Panel,
  Segmented,
  Slider,
  Stat,
  fmt,
  fmtInt,
  fmtPct,
} from '../ui/kit';
import { HeatGrid } from '../ui/viz';
import { statsOf } from '../engine/tensor';
import { roleOf } from '../content/weightRoles';
import { V } from '../content/varInfo';
import { useModel } from '../store/model';
import { buildLMTrainer } from '../engine/lmPresets';
import { DEFAULT_CONVERSE, converse, footprintMB, type Conversation } from '../engine/converse';
import { checkVocab } from '../engine/finetune';
import type { Param } from '../engine/transformer';
import CommsInside from './CommsInside';
import CommsChange, { type Intervention } from './CommsChange';

type Tab = 'model' | 'run' | 'inside' | 'change';

function Tokens({
  ids,
  display,
  from,
  onPick,
  active,
}: {
  ids: number[];
  display: (id: number) => string;
  /** Index at which the generated part starts. */
  from: number;
  onPick?: (genIndex: number) => void;
  active?: number;
}) {
  return (
    <div className="flex flex-wrap gap-1">
      {ids.map((id, i) => {
        const gen = i >= from;
        const gi = i - from;
        const on = gen && active === gi;
        return (
          <button
            key={i}
            disabled={!gen || !onPick}
            onClick={() => gen && onPick?.(gi)}
            title={gen ? `generated token ${gi}` : 'from your prompt'}
            className="mono rounded px-1.5 py-[2px] text-[11px] transition-colors"
            style={{
              background: on
                ? 'color-mix(in srgb, var(--accent) 22%, transparent)'
                : gen
                  ? 'var(--panel-3)'
                  : 'transparent',
              border: `1px solid ${on ? 'var(--accent)' : gen ? 'var(--border)' : 'var(--border-2)'}`,
              color: gen ? 'var(--text)' : 'var(--text-3)',
              cursor: gen && onPick ? 'pointer' : 'default',
            }}
          >
            {display(id) || '·'}
          </button>
        );
      })}
    </div>
  );
}

/* --------------------------------------------------------------- page -- */

export default function CommsLab() {
  const trainer = useModel((s) => s.trainer);
  const origin = useModel((s) => s.origin);
  const revision = useModel((s) => s.revision);
  const publish = useModel((s) => s.publish);

  const [tab, setTab] = useState<Tab>('model');
  const [conv, setConv] = useState<Conversation | null>(null);
  const [focusToken, setFocusToken] = useState(0);
  const [busy, setBusy] = useState(false);

  /* Sampling controls */
  const [prompt, setPrompt] = useState('the ');
  const [maxTokens, setMaxTokens] = useState(16);
  const [temperature, setTemperature] = useState(0.85);
  const [topK, setTopK] = useState(20);
  const [seed, setSeed] = useState(1);

  /* Interventions live here so the run tab can apply them and the change tab
     can edit them. Both need the same object. */
  const [iv, setIv] = useState<Intervention>({ banned: null, steer: null, edits: 0 });

  /* Weight browser */
  const [paramName, setParamName] = useState<string | null>(null);

  const loadStarter = useCallback(() => {
    setBusy(true);
    setTimeout(() => {
      const t = buildLMTrainer('stories', 'small');
      // 120 quick steps is enough to be recognisably a language model rather
      // than noise, without making the reader wait.
      for (let i = 0; i < 120; i++) t.singleStep(i % 5 === 0);
      publish(t, {
        label: 'Starter - tiny stories',
        source: 'built in, trained here for 120 steps',
        chars: 0,
        steps: t.step,
        finalLoss: t.latest()?.loss ?? NaN,
        trainedAt: Date.now(),
      });
      setBusy(false);
    }, 30);
  }, [publish]);

  const params: Param[] = useMemo(() => trainer?.model.params ?? [], [trainer]);
  const current = useMemo(
    () => params.find((p) => p.name === paramName) ?? params[0] ?? null,
    [params, paramName],
  );
  const stats = useMemo(
    () => (current ? statsOf(current.M.data) : null),
    // revision changes when a weight is edited underneath us.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [current, revision],
  );

  const vocabCheck = useMemo(
    () => (trainer ? checkVocab(trainer.tok, prompt) : null),
    [trainer, prompt],
  );

  const estMB = useMemo(
    () => (trainer ? footprintMB(trainer.model, trainer.model.cfg.blockSize, maxTokens) : 0),
    [trainer, maxTokens],
  );

  const run = useCallback(() => {
    if (!trainer) return;
    setBusy(true);
    setTimeout(() => {
      try {
        const c = converse(trainer.model, trainer.tok, prompt, {
          ...DEFAULT_CONVERSE,
          maxTokens,
          temperature,
          topK,
          topP: 0.95,
          seed,
          banned: iv.banned ?? undefined,
          steer: iv.steer ?? undefined,
        });
        setConv(c);
        setFocusToken(0);
      } finally {
        setBusy(false);
      }
    }, 20);
  }, [trainer, prompt, maxTokens, temperature, topK, seed, iv]);

  /* Any weight edit invalidates the recorded conversation: the traces on
     screen were produced by weights that no longer exist. */
  useEffect(() => {
    setConv(null);
  }, [iv.edits]);

  if (!trainer) {
    return (
      <div className="mx-auto max-w-[900px] px-6 py-10">
        <Empty>
          <div className="mb-3 text-[13px] font-medium" style={{ color: 'var(--text-2)' }}>
            This step works on a model you trained yourself.
          </div>
          <p className="mb-4">
            Everything here reads the actual weights of one model and follows one prompt through them. Build
            one in <Link to="/studio" style={{ color: 'var(--accent)' }}>Build your own</Link> and it will
            appear here automatically, or start from a small one trained on the spot.
          </p>
          <Btn onClick={loadStarter} variant="primary" disabled={busy}>
            {busy ? 'Training a starter model...' : 'Train a starter model here'}
          </Btn>
        </Empty>
      </div>
    );
  }

  const M = trainer.model;
  const cfg = M.cfg;

  return (
    <div className="mx-auto max-w-[1400px] px-6 py-6">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <Segmented
          value={tab}
          onChange={setTab}
          options={[
            { id: 'model' as Tab, label: '1 · Your model' },
            { id: 'run' as Tab, label: '2 · Say something' },
            { id: 'inside' as Tab, label: '3 · Watch it think' },
            { id: 'change' as Tab, label: '4 · Change it' },
          ]}
        />
        <div className="flex items-center gap-2">
          {iv.banned && iv.banned.size > 0 && <Badge tone="warn">{iv.banned.size} tokens blocked</Badge>}
          {iv.steer && iv.steer.scale !== 0 && <Badge tone="accent">steering {fmt(iv.steer.scale, 1)}</Badge>}
          {iv.edits > 0 && <Badge tone="warn">{iv.edits} weight edits</Badge>}
          <Badge>{fmtInt(M.paramCount)} params</Badge>
        </div>
      </div>

      {/* ------------------------------------------------ 1. your model -- */}
      {tab === 'model' && (
        <div className="grid gap-5 lg:grid-cols-[1fr_360px]">
          <div className="grid gap-5">
            <Panel
              title="These numbers are the model"
              subtitle="Not a summary of it, not a picture of it. This is the whole thing."
            >
              <Depth
                plain={
                  <>
                    <p className="mb-2">
                      A trained model is a set of tables of numbers and a fixed procedure for pushing text
                      through them. There is no list of facts in here, no rules, no sentences stored
                      anywhere. Every word this model will ever produce comes out of the{' '}
                      {fmtInt(M.paramCount)} numbers below and nothing else.
                    </p>
                    <p>
                      Pick any table on the right to see it. Each cell is one weight. Blue means positive,
                      orange means negative, and the brighter the cell the further from zero it is. What you
                      cannot do is look at a cell and say what it means, and that is not a limitation of this
                      display. Meaning in a model is spread across thousands of these at once.
                    </p>
                  </>
                }
                math={
                  <>
                    <p className="mb-2">
                      The model is the tuple{' '}
                      <span className="mono">
                        (W_emb, W_pos, {'{'}W_q, W_k, W_v, W_o, W_1, W_2, gains, biases{'}'}_l, W_U)
                      </span>{' '}
                      with {fmtInt(M.paramCount)} scalars in total, together with the fixed map
                      that composes them.
                    </p>
                    <p>
                      Nothing else is learned. The architecture, the vocabulary, and the context length were
                      all chosen before training and are not parameters.
                    </p>
                  </>
                }
                code={
                  <Code>{`// every learnable number in this model
model.params            // ${params.length} named tensors
model.paramCount        // ${fmtInt(M.paramCount)} scalars
model.flatParams()      // one Float64Array holding all of them`}</Code>
                }
              />
            </Panel>

            {current && stats && (
              <Panel
                title={<span className="mono">{current.name}</span>}
                subtitle={`${current.M.rows} x ${current.M.cols} = ${fmtInt(current.M.data.length)} weights`}
                right={<Badge>{fmtPct(current.M.data.length / M.paramCount, 1)} of the model</Badge>}
              >
                <p className="mb-3 text-[12.5px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                  {roleOf(current.name)}
                </p>
                <div className="overflow-auto">
                  <HeatGrid
                    rows={current.M.rows}
                    cols={current.M.cols}
                    get={(i, j) => current.M.data[i * current.M.cols + j]}
                    cell={current.M.rows > 40 || current.M.cols > 40 ? 5 : 11}
                    maxWidth={620}
                  />
                </div>
                <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <Stat label="Mean" value={fmt(stats.mean, 4)} />
                  <Stat label="Spread" value={fmt(stats.std, 4)} hint="Standard deviation" />
                  <Stat label="Range" value={`${fmt(stats.min, 2)} to ${fmt(stats.max, 2)}`} />
                  <Stat
                    label="Near zero"
                    value={fmtPct(stats.nearZero, 0)}
                    hint="Share of weights within 0.01 of zero"
                  />
                </div>
                {stats.nearZero > 0.5 && (
                  <div className="mt-3">
                    <Callout tone="info">
                      More than half of this matrix sits within a hair of zero. In a well-trained model that
                      is normal and is why quantising and pruning work at all: most individual weights
                      contribute almost nothing, and the behaviour lives in the minority that do not.
                    </Callout>
                  </div>
                )}
              </Panel>
            )}
          </div>

          <div className="grid content-start gap-5">
            <Panel title="Where this came from">
              {origin ? (
                <div className="grid gap-2.5">
                  <Stat label="Model" value={origin.label} />
                  <Stat label="Trained on" value={origin.source} />
                  {origin.chars > 0 && <Stat label="Text" value={`${fmtInt(origin.chars)} characters`} />}
                  <Stat
                    label="Training"
                    value={origin.steps > 0 ? `${fmtInt(origin.steps)} steps` : 'not trained yet'}
                    tone={origin.steps > 0 ? undefined : 'warn'}
                  />
                  {Number.isFinite(origin.finalLoss) && (
                    <Stat label="Final loss" value={fmt(origin.finalLoss, 3)} sub={`perplexity ${fmt(Math.exp(origin.finalLoss), 1)}`} />
                  )}
                </div>
              ) : (
                <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>
                  No origin recorded.
                </p>
              )}
              {origin && origin.steps === 0 && (
                <div className="mt-3">
                  <Callout tone="warn">
                    This model was built but never trained, so its weights are still the random numbers it
                    started with. It will produce nonsense, which is worth seeing once. Train it in{' '}
                    <Link to="/studio" style={{ color: 'var(--accent)' }}>step 07</Link> and come back.
                  </Callout>
                </div>
              )}
            </Panel>

            <Panel title="Shape">
              <div className="grid grid-cols-2 gap-2.5">
                <Stat label="Vocabulary" value={fmtInt(cfg.vocab)} hint="Distinct tokens it can read or write" />
                <Stat label="Context" value={`${cfg.blockSize} tokens`} hint="How far back it can see" />
                <Stat label="Width" value={cfg.dModel} hint="Numbers carried per token" />
                <Stat label="Blocks" value={cfg.nLayers} />
                <Stat label="Heads" value={`${cfg.nHeads} x ${M.dHead}`} />
                <Stat label="Feed-forward" value={cfg.dFF} />
              </div>
            </Panel>

            <Panel title="Every table in the model" pad={false}>
              <div className="max-h-[420px] overflow-y-auto">
                {params.map((p) => {
                  const on = current?.name === p.name;
                  return (
                    <button
                      key={p.name}
                      onClick={() => setParamName(p.name)}
                      className="flex w-full items-center justify-between gap-2 px-4 py-2 text-left transition-colors"
                      style={{
                        background: on ? 'var(--panel-2)' : 'transparent',
                        borderBottom: '1px solid var(--border)',
                        cursor: 'pointer',
                      }}
                    >
                      <span
                        className="mono truncate text-[11.5px]"
                        style={{ color: on ? 'var(--accent)' : 'var(--text-2)' }}
                      >
                        {p.name}
                      </span>
                      <span className="mono shrink-0 text-[10.5px]" style={{ color: 'var(--text-3)' }}>
                        {p.M.rows}x{p.M.cols}
                      </span>
                    </button>
                  );
                })}
              </div>
            </Panel>
          </div>
        </div>
      )}

      {/* --------------------------------------------- 2. say something -- */}
      {tab === 'run' && (
        <div className="grid gap-5 lg:grid-cols-[380px_1fr]">
          <div className="grid content-start gap-5">
            <Panel title="Your prompt">
              <Field label="Prompt" info={V.commsPrompt}>
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  rows={3}
                  spellCheck={false}
                  className="focus-ring mono w-full resize-y rounded-lg px-2.5 py-2 text-[12px]"
                  style={{
                    background: 'var(--bg-2)',
                    border: '1px solid var(--border)',
                    color: 'var(--text)',
                  }}
                />
              </Field>
              <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px]" style={{ color: 'var(--text-3)' }}>
                <span>
                  {trainer.tok.encode(prompt).length} tokens of {cfg.blockSize}
                </span>
                {trainer.tok.encode(prompt).length > cfg.blockSize && (
                  <Badge tone="warn">too long, the front will be dropped</Badge>
                )}
              </div>
              {vocabCheck && vocabCheck.dropped.length > 0 && (
                <div className="mt-3">
                  <Callout tone="warn" title="Characters this model cannot read">
                    <span className="mono">{vocabCheck.droppedLabels.join('  ')}</span> never appeared in its
                    training text, so there is no vocabulary entry for them and they are silently dropped
                    before the model sees anything. A real tokenizer has a byte-level fallback so this cannot
                    happen; this one does not, which is why you can see the failure.
                  </Callout>
                </div>
              )}
            </Panel>

            <Panel title="How it picks" subtitle="All four of these act after the model has finished.">
              <div className="grid gap-3">
                <Slider
                  label="Tokens to generate"
                  value={maxTokens}
                  min={1}
                  max={48}
                  step={1}
                  onChange={setMaxTokens}
                  format={(v) => String(v)}
                  info={V.commsTokens}
                />
                <Slider
                  label="Temperature"
                  value={temperature}
                  min={0.05}
                  max={2}
                  step={0.05}
                  onChange={setTemperature}
                  info={V.commsTemperature}
                />
                <Slider
                  label="Top-k"
                  value={topK}
                  min={0}
                  max={Math.min(60, cfg.vocab)}
                  step={1}
                  onChange={setTopK}
                  format={(v) => (v === 0 ? 'off' : String(v))}
                  info={V.commsTopK}
                />
                <Slider
                  label="Seed"
                  value={seed}
                  min={1}
                  max={200}
                  step={1}
                  onChange={setSeed}
                  format={(v) => String(v)}
                  info={V.commsSeed}
                />
              </div>
              <div className="mt-3 flex items-center justify-between gap-2">
                <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                  keeps {fmt(estMB, 1)} MB of traces
                </span>
                <Btn onClick={run} variant="primary" disabled={busy}>
                  {busy ? 'Running...' : 'Generate'}
                </Btn>
              </div>
            </Panel>

            {(iv.banned || iv.steer) && (
              <Panel title="Guardrails currently on">
                <div className="grid gap-2 text-[12px]" style={{ color: 'var(--text-2)' }}>
                  {iv.banned && iv.banned.size > 0 && (
                    <div>
                      <Badge tone="warn">{iv.banned.size} tokens</Badge> struck out at sampling time.
                    </div>
                  )}
                  {iv.steer && iv.steer.scale !== 0 && (
                    <div>
                      <Badge tone="accent">{fmt(iv.steer.scale, 1)}</Badge> pushed along a direction at
                      block {iv.steer.layer}.
                    </div>
                  )}
                  <Btn size="sm" onClick={() => setTab('change')}>
                    Edit in Change it
                  </Btn>
                </div>
              </Panel>
            )}
          </div>

          <div className="grid content-start gap-5">
            <Panel
              title="What it said"
              right={conv && <Badge>{fmt(conv.totalMs, 0)} ms</Badge>}
            >
              {!conv ? (
                <Empty>Write something and press Generate. Every token it produces is kept in full, so you can walk back through the reasoning afterwards.</Empty>
              ) : (
                <>
                  <div
                    className="rounded-lg p-3 text-[13px] leading-relaxed"
                    style={{ background: 'var(--bg-2)', border: '1px solid var(--border)' }}
                  >
                    <span style={{ color: 'var(--text-3)' }}>{conv.prompt}</span>
                    <span style={{ color: 'var(--text)' }}>{conv.reply}</span>
                  </div>

                  {conv.promptOverflowed && (
                    <div className="mt-3">
                      <Callout tone="warn">
                        Your prompt is longer than this model's {cfg.blockSize}-token context, so the
                        beginning of it was never seen. The model is not told that anything was removed. This
                        is exactly what happens to a long conversation with a large model once it passes the
                        context limit.
                      </Callout>
                    </div>
                  )}

                  <div className="mt-4">
                    <div className="mb-2 text-[11px] font-medium uppercase tracking-[0.07em]" style={{ color: 'var(--text-3)' }}>
                      Token by token &mdash; click one to inspect it
                    </div>
                    <Tokens
                      ids={[...conv.promptIds, ...conv.steps.map((s) => s.chosen)]}
                      display={(id) => trainer.tok.display(id)}
                      from={conv.promptIds.length}
                      active={focusToken}
                      onPick={(gi) => {
                        setFocusToken(gi);
                        setTab('inside');
                      }}
                    />
                  </div>
                </>
              )}
            </Panel>

            {conv && (
              <Panel title="How confident was it, token by token">
                <div className="grid gap-1.5">
                  {conv.steps.map((s) => {
                    const p = s.sample.candidates.find((c) => c.id === s.chosen)?.prob ?? 0;
                    const alts = s.sample.candidates
                      .filter((c) => c.kept && c.id !== s.chosen)
                      .sort((a, b) => b.prob - a.prob)
                      .slice(0, 3);
                    return (
                      <button
                        key={s.index}
                        onClick={() => {
                          setFocusToken(s.index);
                          setTab('inside');
                        }}
                        className="flex items-center gap-3 rounded-lg px-2 py-1.5 text-left transition-colors"
                        style={{ background: focusToken === s.index ? 'var(--panel-2)' : 'transparent', cursor: 'pointer' }}
                      >
                        <span className="mono w-7 shrink-0 text-[10.5px]" style={{ color: 'var(--text-3)' }}>
                          {s.index}
                        </span>
                        <span
                          className="mono w-20 shrink-0 truncate rounded px-1.5 py-[2px] text-[11px]"
                          style={{ background: 'var(--panel-3)', color: 'var(--text)' }}
                        >
                          {trainer.tok.display(s.chosen) || '·'}
                        </span>
                        <span className="h-1.5 flex-1 overflow-hidden rounded-full" style={{ background: 'var(--panel-3)' }}>
                          <span
                            className="block h-full rounded-full"
                            style={{ width: `${Math.max(1, p * 100)}%`, background: 'var(--accent)' }}
                          />
                        </span>
                        <span className="mono tnum w-12 shrink-0 text-right text-[10.5px]" style={{ color: 'var(--text-2)' }}>
                          {fmtPct(p, 0)}
                        </span>
                        <span className="mono hidden w-48 shrink-0 truncate text-[10.5px] sm:block" style={{ color: 'var(--text-3)' }}>
                          {alts.map((a) => trainer.tok.display(a.id)).join(' ')}
                        </span>
                      </button>
                    );
                  })}
                </div>
                <div className="mt-3">
                  <Depth
                    plain={
                      <p>
                        The bar is how sure the model was about the token it picked, and the greyed-out
                        pieces on the right are what it nearly said instead. A low bar does not mean it was
                        wrong. It means several continuations were about equally reasonable, which is the
                        normal state of affairs in the middle of a sentence. What should worry you is a very
                        high bar on a token that turns out to be nonsense.
                      </p>
                    }
                    math={
                      <p>
                        Each bar is <span className="mono">p(y_t | y_&lt;t)</span> after temperature and
                        filtering, not the raw softmax. The sequence probability is the product of these, so
                        reply likelihood falls off geometrically with length &mdash; which is why average
                        per-token loss, not total, is the meaningful score.
                      </p>
                    }
                    code={<Code>{`const p = step.sample.candidates.find(c => c.id === step.chosen).prob;`}</Code>}
                  />
                </div>
              </Panel>
            )}
          </div>
        </div>
      )}

      {/* ------------------------------------------- 3. watch it think -- */}
      {tab === 'inside' && (
        <CommsInside
          trainer={trainer}
          conv={conv}
          focusToken={focusToken}
          setFocusToken={setFocusToken}
          onNeedRun={() => setTab('run')}
        />
      )}

      {/* ------------------------------------------------- 4. change it -- */}
      {tab === 'change' && (
        <CommsChange
          trainer={trainer}
          iv={iv}
          setIv={setIv}
          prompt={prompt}
          seed={seed}
          temperature={temperature}
          topK={topK}
          maxTokens={maxTokens}
        />
      )}
    </div>
  );
}
