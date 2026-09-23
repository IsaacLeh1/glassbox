import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Badge, Btn, Callout, Code, Depth, Empty, InfoDot, Panel, Stat, fmt, fmtPct } from '../ui/kit';
import { HeatGrid } from '../ui/viz';
import { V } from '../content/varInfo';
import { bandsFor, buildStages, lastRow, logitLens, type Stage } from '../engine/replay';
import type { Conversation } from '../engine/converse';
import type { LMTrainer } from '../engine/lmTrainer';

/**
 * The scrubber.
 *
 * One generated token at a time, one stage of its forward pass at a time,
 * backwards and forwards at the reader's own pace. Nothing here recomputes
 * anything: the conversation already holds every intermediate, and this is a
 * view onto numbers that have already been produced.
 */

const PLAY_MS = 550;

function useInterval(active: boolean, ms: number, fn: () => void) {
  const cb = useRef(fn);
  cb.current = fn;
  useEffect(() => {
    if (!active) return;
    const id = window.setInterval(() => cb.current(), ms);
    return () => window.clearInterval(id);
  }, [active, ms]);
}

/* ------------------------------------------------------------ timeline -- */

function Timeline({
  stages,
  index,
  onChange,
}: {
  stages: Stage[];
  index: number;
  onChange: (i: number) => void;
}) {
  const bands = useMemo(() => bandsFor(stages), [stages]);
  const n = stages.length;

  return (
    <div>
      <div className="flex w-full overflow-hidden rounded-lg" style={{ border: '1px solid var(--border)' }}>
        {bands.map((b) => {
          const width = ((b.to - b.from + 1) / n) * 100;
          const inside = index >= b.from && index <= b.to;
          return (
            <div key={`${b.label}-${b.from}`} style={{ width: `${width}%` }} className="min-w-0">
              <div
                className="truncate px-2 py-1 text-center text-[10px] font-medium uppercase tracking-[0.07em]"
                style={{
                  background: inside ? 'color-mix(in srgb, var(--accent) 14%, transparent)' : 'var(--panel-2)',
                  color: inside ? 'var(--accent)' : 'var(--text-3)',
                  borderBottom: '1px solid var(--border)',
                }}
              >
                {b.label}
              </div>
              <div className="flex h-6">
                {stages.slice(b.from, b.to + 1).map((s, k) => {
                  const i = b.from + k;
                  const on = i === index;
                  return (
                    <button
                      key={s.id}
                      onClick={() => onChange(i)}
                      title={`${s.crumb} - ${s.title}`}
                      className="min-w-0 flex-1 transition-colors"
                      style={{
                        background: on
                          ? 'var(--accent)'
                          : i < index
                            ? 'color-mix(in srgb, var(--accent) 22%, transparent)'
                            : 'var(--panel-3)',
                        borderRight: '1px solid var(--bg-2)',
                        cursor: 'pointer',
                      }}
                    />
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
      <input
        type="range"
        min={0}
        max={n - 1}
        step={1}
        value={index}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-2 w-full"
        aria-label="Stage of the forward pass"
      />
    </div>
  );
}

/* --------------------------------------------------------------- lens -- */

function Lens({ trainer, row, label }: { trainer: LMTrainer; row: Float64Array; label: string }) {
  const top = useMemo(() => logitLens(trainer.model, row, 6), [trainer, row]);
  return (
    <div>
      <div className="mb-1.5 text-[10.5px] font-semibold uppercase tracking-[0.08em]" style={{ color: 'var(--text-3)' }}>
        {label}
      </div>
      <div className="grid gap-1">
        {top.map((e) => (
          <div key={e.id} className="flex items-center gap-2">
            <span
              className="mono w-16 shrink-0 truncate rounded px-1.5 py-[1px] text-[11px]"
              style={{ background: 'var(--panel-3)', color: 'var(--text)' }}
            >
              {trainer.tok.display(e.id) || '·'}
            </span>
            <span className="h-1.5 flex-1 overflow-hidden rounded-full" style={{ background: 'var(--panel-3)' }}>
              <span
                className="block h-full rounded-full"
                style={{ width: `${Math.max(1, e.prob * 100)}%`, background: 'var(--pos)' }}
              />
            </span>
            <span className="mono tnum w-10 shrink-0 text-right text-[10px]" style={{ color: 'var(--text-3)' }}>
              {fmtPct(e.prob, 0)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* --------------------------------------------------------------- page -- */

export default function CommsInside({
  trainer,
  conv,
  focusToken,
  setFocusToken,
  onNeedRun,
}: {
  trainer: LMTrainer;
  conv: Conversation | null;
  focusToken: number;
  setFocusToken: (i: number) => void;
  onNeedRun: () => void;
}) {
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [cell, setCell] = useState<{ i: number; j: number; v: number } | null>(null);

  const step = conv && conv.steps.length > 0 ? conv.steps[Math.min(focusToken, conv.steps.length - 1)] : null;

  const stages = useMemo(
    () => (step ? buildStages(step.trace, trainer.model.cfg) : []),
    [step, trainer.model.cfg],
  );

  // Hold position when switching tokens: the stage list is the same length
  // for every token of the same model, so the reader stays where they were.
  useEffect(() => {
    setIndex((i) => Math.min(i, Math.max(0, stages.length - 1)));
    setCell(null);
  }, [stages.length]);

  const go = useCallback(
    (i: number) => setIndex(Math.max(0, Math.min(stages.length - 1, i))),
    [stages.length],
  );

  useInterval(playing && stages.length > 0, PLAY_MS, () => {
    setIndex((i) => {
      if (i >= stages.length - 1) {
        setPlaying(false);
        return i;
      }
      return i + 1;
    });
  });

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA')) return;
      if (e.key === 'ArrowRight') {
        setPlaying(false);
        go(index + 1);
      } else if (e.key === 'ArrowLeft') {
        setPlaying(false);
        go(index - 1);
      } else if (e.key === 'Home') {
        go(0);
      } else if (e.key === 'End') {
        go(stages.length - 1);
      } else if (e.key === ' ') {
        e.preventDefault();
        setPlaying((p) => !p);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [index, go, stages.length]);

  if (!conv || !step) {
    return (
      <Empty>
        <div className="mb-3">Nothing has been said yet.</div>
        <p className="mb-4">
          This is the replay. It only exists once the model has actually run, because what you scrub through
          are the real intermediate numbers from a real forward pass, not a reconstruction.
        </p>
        <Btn onClick={onNeedRun} variant="primary">
          Go and generate something
        </Btn>
      </Empty>
    );
  }

  const stage = stages[Math.min(index, stages.length - 1)];
  const tokens = step.trace.tokens;
  const label = (id: number) => trainer.tok.display(id) || '·';

  const colLabel = (j: number) => {
    if (stage.colSpace === 'positions') return label(tokens[j]);
    if (stage.colSpace === 'vocab') return label(j);
    return String(j);
  };

  return (
    <div className="grid gap-5">
      {/* ------------------------------------------------- which token -- */}
      <Panel
        title="Which token are you watching"
        subtitle={`Each of these was produced by its own complete pass through the network. You are inside pass ${step.index + 1} of ${conv.steps.length}.`}
        right={<InfoDot info={V.commsToken} title="Token" />}
      >
        <div className="flex flex-wrap items-center gap-1.5">
          {conv.steps.map((s) => {
            const on = s.index === focusToken;
            return (
              <button
                key={s.index}
                onClick={() => setFocusToken(s.index)}
                className="mono rounded px-2 py-[3px] text-[11.5px] transition-colors"
                style={{
                  background: on ? 'color-mix(in srgb, var(--accent) 22%, transparent)' : 'var(--panel-3)',
                  border: `1px solid ${on ? 'var(--accent)' : 'var(--border)'}`,
                  color: on ? 'var(--accent)' : 'var(--text-2)',
                  cursor: 'pointer',
                }}
              >
                {label(s.chosen)}
              </button>
            );
          })}
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <Stat label="Context fed in" value={`${step.context.length} tokens`} />
          <Stat label="It chose" value={label(step.chosen)} />
          <Stat
            label="Confidence"
            value={fmtPct(step.sample.candidates.find((c) => c.id === step.chosen)?.prob ?? 0, 1)}
          />
          <Stat label="Took" value={`${fmt(step.ms, 1)} ms`} />
        </div>
        {step.truncated && (
          <div className="mt-3">
            <Callout tone="warn">
              By this point the conversation is longer than the {trainer.model.cfg.blockSize}-token window,
              so the oldest tokens have dropped off the front. The model is working from a truncated view
              and has no way of knowing it.
            </Callout>
          </div>
        )}
      </Panel>

      {/* ----------------------------------------------------- scrubber -- */}
      <Panel
        title="The forward pass, one step at a time"
        subtitle="Drag, click a band, or use the arrow keys. Space plays and pauses."
        right={
          <div className="flex items-center gap-1.5">
            <InfoDot info={V.commsStage} title="Stage" />
            <Btn size="sm" onClick={() => go(0)} title="Back to the start">
              &#124;&#9664;
            </Btn>
            <Btn size="sm" onClick={() => { setPlaying(false); go(index - 1); }} title="Previous stage">
              &#9664;
            </Btn>
            <Btn size="sm" variant={playing ? 'primary' : 'soft'} onClick={() => setPlaying((p) => !p)}>
              {playing ? 'Pause' : 'Play'}
            </Btn>
            <Btn size="sm" onClick={() => { setPlaying(false); go(index + 1); }} title="Next stage">
              &#9654;
            </Btn>
            <Btn size="sm" onClick={() => go(stages.length - 1)} title="Jump to the end">
              &#9654;&#124;
            </Btn>
          </div>
        }
      >
        <Timeline stages={stages} index={index} onChange={(i) => { setPlaying(false); go(i); }} />
        <div className="mt-2 flex items-center justify-between text-[11px]" style={{ color: 'var(--text-3)' }}>
          <span>
            stage {index + 1} of {stages.length}
          </span>
          <span className="mono">{stage.crumb}</span>
        </div>
      </Panel>

      {/* -------------------------------------------------- the display -- */}
      <div className="grid gap-5 lg:grid-cols-[1fr_340px]">
        <Panel
          title={stage.title}
          subtitle={stage.crumb}
          right={
            stage.delta !== undefined ? (
              <Badge tone={stage.delta > 0.5 ? 'warn' : 'accent'}>
                moved the stream {fmtPct(stage.delta, 0)}
              </Badge>
            ) : stage.matrix ? (
              <Badge>
                {stage.matrix.rows} x {stage.matrix.cols}
              </Badge>
            ) : null
          }
        >
          {/* --- the input stage shows tokens, not a matrix --- */}
          {stage.kind === 'tokens' && (
            <div>
              <div className="flex flex-wrap gap-1.5">
                {tokens.map((id, i) => (
                  <span
                    key={i}
                    className="rounded px-2 py-1 text-[11.5px]"
                    style={{ background: 'var(--panel-3)', border: '1px solid var(--border)' }}
                  >
                    <span className="mono" style={{ color: 'var(--text)' }}>
                      {label(id)}
                    </span>
                    <span className="mono ml-1.5 text-[10px]" style={{ color: 'var(--text-3)' }}>
                      {id}
                    </span>
                  </span>
                ))}
              </div>
              <p className="mt-3 text-[11.5px]" style={{ color: 'var(--text-3)' }}>
                {tokens.length} tokens. The small grey number is the row of the vocabulary each piece maps
                to &mdash; that is all a token is.
              </p>
            </div>
          )}

          {/* --- the sampling stage shows the candidate list --- */}
          {stage.kind === 'sample' && (
            <div>
              <div className="grid gap-1">
                {step.sample.candidates
                  .slice()
                  .sort((a, b) => b.prob - a.prob)
                  .slice(0, 12)
                  .map((c) => (
                    <div key={c.id} className="flex items-center gap-2">
                      <span
                        className="mono w-20 shrink-0 truncate rounded px-1.5 py-[2px] text-[11px]"
                        style={{
                          background: c.id === step.chosen ? 'color-mix(in srgb, var(--accent) 22%, transparent)' : 'var(--panel-3)',
                          color: c.blocked ? 'var(--text-3)' : 'var(--text)',
                          textDecoration: c.blocked ? 'line-through' : 'none',
                        }}
                      >
                        {label(c.id)}
                      </span>
                      <span className="h-2 flex-1 overflow-hidden rounded-full" style={{ background: 'var(--panel-3)' }}>
                        <span
                          className="block h-full rounded-full"
                          style={{
                            width: `${Math.max(0.5, c.prob * 100)}%`,
                            background: c.id === step.chosen ? 'var(--accent)' : c.kept ? 'var(--pos)' : 'var(--border-2)',
                          }}
                        />
                      </span>
                      <span className="mono tnum w-12 shrink-0 text-right text-[10.5px]" style={{ color: 'var(--text-3)' }}>
                        {fmtPct(c.prob, 1)}
                      </span>
                      {c.blocked && <Badge tone="err">blocked</Badge>}
                      {!c.kept && !c.blocked && c.cut && <Badge>cut by {c.cut}</Badge>}
                    </div>
                  ))}
              </div>
              {step.sample.blockedMass > 0 && (
                <div className="mt-3">
                  <Callout tone="warn">
                    A guardrail removed {fmtPct(step.sample.blockedMass, 1)} of the probability the model
                    had assigned, and the rest was rescaled to fill the gap. The model was never told. From
                    the inside it looks as though those tokens were simply never options.
                  </Callout>
                </div>
              )}
            </div>
          )}

          {/* --- everything else is a matrix --- */}
          {stage.matrix && (
            <div>
              <div className="flex gap-2">
                {/* row labels */}
                <div className="shrink-0 pt-[1px]">
                  {tokens.map((id, i) => (
                    <div
                      key={i}
                      className="mono truncate text-right text-[9.5px] leading-none"
                      style={{
                        height: stage.matrix!.rows > 20 ? 7 : 13,
                        maxWidth: 64,
                        color: cell?.i === i ? 'var(--accent)' : 'var(--text-3)',
                      }}
                      title={label(id)}
                    >
                      {stage.matrix!.rows > 20 ? '' : label(id)}
                    </div>
                  ))}
                </div>
                <div className="min-w-0 overflow-x-auto">
                  <HeatGrid
                    rows={stage.matrix.rows}
                    cols={stage.matrix.cols}
                    get={(i, j) => stage.matrix!.data[i * stage.matrix!.cols + j]}
                    mode={stage.kind === 'attention' ? 'heat' : 'signed'}
                    cell={stage.matrix.rows > 20 || stage.matrix.cols > 64 ? 6 : 12}
                    gap={stage.matrix.cols > 64 ? 0 : 1}
                    maxWidth={640}
                    highlight={cell ? { i: cell.i, j: cell.j } : undefined}
                    onHover={(i, j, v) => setCell({ i, j, v })}
                  />
                </div>
              </div>

              <div className="mt-3 flex flex-wrap items-center gap-3 text-[11.5px]" style={{ color: 'var(--text-3)' }}>
                {cell ? (
                  <>
                    <span>
                      row <span className="mono" style={{ color: 'var(--text-2)' }}>{label(tokens[cell.i])}</span>
                    </span>
                    <span>
                      column{' '}
                      <span className="mono" style={{ color: 'var(--text-2)' }}>{colLabel(cell.j)}</span>
                    </span>
                    <span className="mono tnum" style={{ color: 'var(--text)' }}>
                      {fmt(cell.v, 4)}
                    </span>
                    {stage.kind === 'attention' && (
                      <span>
                        &mdash; {fmtPct(cell.v, 1)} of the attention that{' '}
                        <span className="mono">{label(tokens[cell.i])}</span> had to spend went to{' '}
                        <span className="mono">{label(tokens[cell.j])}</span>
                      </span>
                    )}
                  </>
                ) : (
                  <span>Hover a cell to read it.</span>
                )}
              </div>

              {stage.kind === 'attention' && (
                <div className="mt-3">
                  <Callout tone="insight">
                    Read this one row at a time. The bottom row is the token doing the predicting, and the
                    bright cells in it are the earlier tokens it decided were worth looking at. This grid is
                    the closest thing a transformer has to an explanation of itself, and it is still only
                    where it looked &mdash; not why.
                  </Callout>
                </div>
              )}
            </div>
          )}

          <div className="mt-4">
            <Depth
              plain={<p>{stage.plain}</p>}
              math={
                <div>
                  <p className="mb-2">{stage.plain}</p>
                  <Code className="mono">{stage.math}</Code>
                </div>
              }
              code={<Code>{stage.code}</Code>}
            />
          </div>
        </Panel>

        {/* ------------------------------------------------- side panel -- */}
        <div className="grid content-start gap-5">
          {stage.isResidual && stage.matrix && (
            <Panel title="What it would say if it stopped here">
              <Lens
                trainer={trainer}
                row={lastRow(stage.matrix)}
                label={`after ${stage.crumb.toLowerCase()}`}
              />
              <p className="mt-3 text-[11.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                This takes the half-finished state and pushes it straight to the output, skipping every
                block that has not run yet. Scrub forward and watch the list change: that is the model
                making up its mind. The technique is called the logit lens, and the blocks in between were
                never trained to be read this way, so treat it as a leaning rather than a prediction.
              </p>
            </Panel>
          )}

          <Panel title="Where you are">
            <div className="grid gap-2">
              <Stat label="Stage" value={`${index + 1} of ${stages.length}`} />
              <Stat label="Kind" value={stage.kind} />
              {stage.layer >= 0 && <Stat label="Block" value={stage.layer} />}
              {stage.head >= 0 && <Stat label="Head" value={stage.head} />}
              {stage.matrix && (
                <Stat
                  label="Numbers on screen"
                  value={(stage.matrix.rows * stage.matrix.cols).toLocaleString()}
                />
              )}
            </div>
          </Panel>

          <Panel title="Jump to">
            <div className="flex flex-wrap gap-1.5">
              {stages
                .map((s, i) => ({ s, i }))
                .filter(({ s }) => s.isResidual || s.kind === 'attention' || s.kind === 'logits' || s.kind === 'sample')
                .map(({ s, i }) => (
                  <Btn
                    key={s.id}
                    size="sm"
                    active={i === index}
                    onClick={() => {
                      setPlaying(false);
                      go(i);
                    }}
                  >
                    {s.crumb.replace('Block ', 'B').replace('Head ', 'H').replace(' - ', ' ')}
                  </Btn>
                ))}
            </div>
            <p className="mt-2 text-[11px]" style={{ color: 'var(--text-3)' }}>
              The stages worth comparing against each other: every point the scratchpad changed, every
              attention map, and the final choice.
            </p>
          </Panel>
        </div>
      </div>
    </div>
  );
}
