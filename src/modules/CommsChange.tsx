import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Badge,
  Btn,
  Callout,
  Code,
  Depth,
  Field,
  InfoDot,
  Panel,
  Segmented,
  Slider,
  Stat,
  fmt,
  fmtInt,
  fmtPct,
} from '../ui/kit';
import { HeatGrid, LineChart } from '../ui/viz';
import { V } from '../content/varInfo';
import { useModel } from '../store/model';
import { useFrameLoop } from '../ui/loop';
import { DEFAULT_CONVERSE, converse } from '../engine/converse';
import { DEFAULT_TEACH, Finetuner } from '../engine/finetune';
import { contrastDirection, resolveBan } from '../engine/steering';
import type { LMTrainer } from '../engine/lmTrainer';
import type { Steer } from '../engine/transformer';
import { statsOf } from '../engine/tensor';
import { roleOf } from '../content/weightRoles';

/**
 * Changing the model, in the three ways there are.
 *
 * Every one of these is a real mechanism, and the point of putting them side
 * by side is that they are not the same kind of thing at all:
 *
 *  - editing weights changes what the model is,
 *  - a guardrail changes what it is allowed to say without changing what it is,
 *  - teaching changes what it is, but by the same process that made it.
 */

export interface Intervention {
  banned: Set<number> | null;
  steer: Steer | null;
  /** Counter, bumped on every weight change, so other views know to invalidate. */
  edits: number;
}

type Sub = 'weights' | 'guardrails' | 'teach';

interface RunNote {
  note: string;
  text: string;
  at: number;
}

export default function CommsChange({
  trainer,
  iv,
  setIv,
  prompt,
  seed,
  temperature,
  topK,
  maxTokens,
}: {
  trainer: LMTrainer;
  iv: Intervention;
  setIv: (f: (prev: Intervention) => Intervention) => void;
  prompt: string;
  seed: number;
  temperature: number;
  topK: number;
  maxTokens: number;
}) {
  const touch = useModel((s) => s.touch);
  const [sub, setSub] = useState<Sub>('weights');
  const [log, setLog] = useState<RunNote[]>([]);

  const M = trainer.model;
  const cfg = M.cfg;

  /* The weights exactly as they were when this tab first mounted, so any
     experiment can be walked back in one action. */
  const pristine = useRef<Float64Array | null>(null);
  if (pristine.current === null) pristine.current = M.flatParams().slice();

  const runNow = useCallback(
    (note: string) => {
      const c = converse(M, trainer.tok, prompt, {
        ...DEFAULT_CONVERSE,
        maxTokens,
        temperature,
        topK,
        topP: 0.95,
        seed,
        banned: iv.banned ?? undefined,
        steer: iv.steer ?? undefined,
      });
      setLog((l) => [{ note, text: c.reply, at: Date.now() }, ...l].slice(0, 8));
    },
    [M, trainer.tok, prompt, maxTokens, temperature, topK, seed, iv],
  );

  /* ------------------------------------------------------------ weights */

  const [paramName, setParamName] = useState(M.params[0]?.name ?? '');
  const param = useMemo(() => M.params.find((p) => p.name === paramName) ?? M.params[0], [M, paramName]);
  const [cell, setCell] = useState<{ i: number; j: number } | null>(null);
  const [draft, setDraft] = useState(0);
  const [scale, setScale] = useState(1);
  const [ablated, setAblated] = useState<Record<string, Float64Array>>({});

  const cellValue = cell && param ? param.M.data[cell.i * param.M.cols + cell.j] : 0;
  useEffect(() => {
    setDraft(cellValue);
  }, [cellValue]);

  const applyCell = () => {
    if (!cell || !param) return;
    param.M.data[cell.i * param.M.cols + cell.j] = draft;
    touch();
    setIv((p) => ({ ...p, edits: p.edits + 1 }));
    runNow(`set ${param.name}[${cell.i},${cell.j}] to ${fmt(draft, 3)}`);
  };

  const applyScale = () => {
    if (!param) return;
    for (let i = 0; i < param.M.data.length; i++) param.M.data[i] *= scale;
    touch();
    setIv((p) => ({ ...p, edits: p.edits + 1 }));
    runNow(`scaled ${param.name} by ${fmt(scale, 2)}`);
    setScale(1);
  };

  /**
   * Switch a head off by zeroing the rows of the output projection that read
   * from it. The head still does all its work; none of it is allowed through.
   */
  const toggleAblate = (layer: number, head: number) => {
    const key = `${layer}:${head}`;
    const Wo = M.blocks[layer].wo.M;
    const dh = M.dHead;
    const off = head * dh;
    const saved = ablated[key];

    if (saved) {
      for (let r = 0; r < dh; r++) {
        for (let c = 0; c < Wo.cols; c++) Wo.data[(off + r) * Wo.cols + c] = saved[r * Wo.cols + c];
      }
      setAblated((a) => {
        const next = { ...a };
        delete next[key];
        return next;
      });
      touch();
      setIv((p) => ({ ...p, edits: p.edits + 1 }));
      runNow(`restored block ${layer} head ${head}`);
      return;
    }

    const copy = new Float64Array(dh * Wo.cols);
    for (let r = 0; r < dh; r++) {
      for (let c = 0; c < Wo.cols; c++) {
        copy[r * Wo.cols + c] = Wo.data[(off + r) * Wo.cols + c];
        Wo.data[(off + r) * Wo.cols + c] = 0;
      }
    }
    setAblated((a) => ({ ...a, [key]: copy }));
    touch();
    setIv((p) => ({ ...p, edits: p.edits + 1 }));
    runNow(`switched off block ${layer} head ${head}`);
  };

  const restoreAll = () => {
    if (!pristine.current) return;
    M.loadFlat(pristine.current);
    setAblated({});
    touch();
    setIv((p) => ({ ...p, edits: p.edits + 1 }));
    runNow('restored every weight');
  };

  /* -------------------------------------------------------- guardrails */

  const [banText, setBanText] = useState('');
  const banResolved = useMemo(
    () => (banText.trim() ? resolveBan(trainer.tok, banText.split(',')) : []),
    [banText, trainer.tok],
  );
  const banCount = banResolved.reduce((n, r) => n + r.ids.length, 0);

  const applyBan = () => {
    const ids = new Set<number>();
    for (const r of banResolved) for (const id of r.ids) ids.add(id);
    setIv((p) => ({ ...p, banned: ids.size > 0 ? ids : null }));
  };

  const [groupA, setGroupA] = useState('the night was quiet and dark.\nthe owl waited in the shadows.');
  const [groupB, setGroupB] = useState('the morning was bright and loud.\nthe bee sang in the sunshine.');
  const [steerLayer, setSteerLayer] = useState(Math.max(0, cfg.nLayers - 1));
  const [steerScale, setSteerScale] = useState(0);
  const [dir, setDir] = useState<{ vec: Float64Array; separation: number; countA: number; countB: number } | null>(null);

  const findDirection = () => {
    const d = contrastDirection(
      M,
      trainer.tok,
      groupA.split('\n').filter((s) => s.trim()),
      groupB.split('\n').filter((s) => s.trim()),
      steerLayer,
    );
    setDir(d);
    if (d) setIv((p) => ({ ...p, steer: { layer: steerLayer, vec: d.vec, scale: steerScale } }));
  };

  useEffect(() => {
    if (dir) setIv((p) => ({ ...p, steer: { layer: steerLayer, vec: dir.vec, scale: steerScale } }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [steerScale]);

  /* -------------------------------------------------------------- teach */

  const [lesson, setLesson] = useState('the fox sat on the warm stone.\nthe fox watched the river go by.');
  const [teachLr, setTeachLr] = useState(DEFAULT_TEACH.lr);
  const [teachEpochs, setTeachEpochs] = useState(DEFAULT_TEACH.epochs);
  const [ft, setFt] = useState<Finetuner | null>(null);
  const [teaching, setTeaching] = useState(false);
  const [, setTick] = useState(0);

  const startTeaching = () => {
    const f = new Finetuner(
      M,
      trainer.tok,
      lesson.split('\n').filter((s) => s.trim()),
      { lr: teachLr, epochs: teachEpochs, clipNorm: 1 },
      trainer.trainIds,
    );
    setFt(f);
    setTeaching(f.windows.length > 0);
  };

  useFrameLoop(teaching && !!ft, () => {
    if (!ft) return;
    ft.runSlice(10);
    if (ft.done) {
      setTeaching(false);
      touch();
      setIv((p) => ({ ...p, edits: p.edits + 1 }));
      runNow(`taught ${ft.windows.length} windows for ${ft.cfg.epochs} passes`);
    }
    setTick((t) => t + 1);
  });

  const revertTeaching = () => {
    if (!ft) return;
    ft.revert();
    touch();
    setIv((p) => ({ ...p, edits: p.edits + 1 }));
    runNow('reverted the lesson');
    setFt(null);
  };

  const teachSeries = useMemo(() => {
    if (!ft || ft.history.length === 0) return [];
    return [
      {
        id: 'new',
        label: 'your examples',
        color: 'var(--accent)',
        points: ft.history.map((h) => ({ x: h.pass, y: h.loss })),
      },
      {
        id: 'old',
        label: 'what it already knew',
        color: 'var(--warn)',
        dashed: true,
        points: ft.history.filter((h) => Number.isFinite(h.oldLoss)).map((h) => ({ x: h.pass, y: h.oldLoss })),
      },
    ];
  }, [ft, ft?.history.length]);

  const pStats = param ? statsOf(param.M.data) : null;

  /* --------------------------------------------------------------- render */

  return (
    <div className="grid gap-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Segmented
          value={sub}
          onChange={setSub}
          options={[
            { id: 'weights' as Sub, label: 'Edit the weights' },
            { id: 'guardrails' as Sub, label: 'Guardrails' },
            { id: 'teach' as Sub, label: 'Teach it something' },
          ]}
        />
        <div className="flex items-center gap-2">
          {iv.edits > 0 && <Badge tone="warn">{iv.edits} changes made</Badge>}
          <Btn size="sm" onClick={() => runNow('no change, same prompt and seed')}>
            Re-run the prompt
          </Btn>
          <Btn size="sm" onClick={restoreAll} disabled={iv.edits === 0}>
            Restore every weight
          </Btn>
        </div>
      </div>

      <div className="grid gap-5 lg:grid-cols-[1fr_360px]">
        <div className="grid content-start gap-5">
          {/* -------------------------------------------------- weights -- */}
          {sub === 'weights' && param && pStats && (
            <>
              <Panel
                title="There is no dial in here labelled with a meaning"
                subtitle="Which is the single most important thing to understand about a trained model."
              >
                <Depth
                  plain={
                    <p>
                      You can change any number in this model right now. What you will find is that changing
                      one weight almost never does anything you can name. Meaning is not stored in
                      individual weights; it lives in patterns across thousands of them at once. This is why
                      you cannot fix a model's behaviour by finding the bad number and correcting it, and it
                      is why alignment is difficult rather than tedious. Switching off a whole attention
                      head, further down, does produce a visible effect &mdash; and even then the effect is
                      rarely something you could have predicted.
                    </p>
                  }
                  math={
                    <p>
                      A single parameter's influence is{' '}
                      <span className="mono">&part;L/&part;w_ij</span>, typically of order{' '}
                      <span className="mono">10&#8315;&#8308;</span> here, while the output depends on a sum
                      over <span className="mono">{fmtInt(M.paramCount)}</span> such terms. Coordinated
                      change across a subspace is what moves behaviour; a rank-one perturbation is almost
                      always absorbed.
                    </p>
                  }
                  code={<Code>{`param.M.data[i * param.M.cols + j] = newValue;
// then run the identical prompt with the identical seed`}</Code>}
                />
              </Panel>

              <Panel
                title={<span className="mono">{param.name}</span>}
                subtitle={roleOf(param.name)}
                right={
                  <Badge>
                    {param.M.rows} x {param.M.cols}
                  </Badge>
                }
              >
                <div className="overflow-auto">
                  <HeatGrid
                    rows={param.M.rows}
                    cols={param.M.cols}
                    get={(i, j) => param.M.data[i * param.M.cols + j]}
                    cell={param.M.rows > 40 || param.M.cols > 40 ? 5 : 11}
                    maxWidth={620}
                    highlight={cell ?? undefined}
                    onClick={(i, j) => setCell({ i, j })}
                  />
                </div>
                <p className="mt-2 text-[11.5px]" style={{ color: 'var(--text-3)' }}>
                  Click any cell to select it. Blue is positive, orange is negative.
                </p>

                {cell && (
                  <div className="mt-4 grid gap-3" style={{ borderTop: '1px solid var(--border)', paddingTop: 16 }}>
                    <div className="flex items-center gap-2">
                      <span className="mono text-[12px]" style={{ color: 'var(--text-2)' }}>
                        {param.name}[{cell.i}, {cell.j}]
                      </span>
                      <InfoDot info={V.commsCell} title="One weight" />
                    </div>
                    <Slider
                      label="Value"
                      value={draft}
                      min={-1}
                      max={1}
                      step={0.001}
                      onChange={setDraft}
                      format={(v) => fmt(v, 3)}
                    />
                    <div className="flex items-center gap-2">
                      <Btn variant="primary" size="sm" onClick={applyCell}>
                        Apply and re-run
                      </Btn>
                      <Btn size="sm" onClick={() => setDraft(cellValue)}>
                        Reset the slider
                      </Btn>
                      <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                        was {fmt(cellValue, 4)}
                      </span>
                    </div>
                  </div>
                )}

                <div className="mt-4 grid gap-3" style={{ borderTop: '1px solid var(--border)', paddingTop: 16 }}>
                  <Slider
                    label="Scale the whole matrix"
                    value={scale}
                    min={0}
                    max={3}
                    step={0.05}
                    onChange={setScale}
                    format={(v) => `x${fmt(v, 2)}`}
                    info={V.commsScale}
                  />
                  <div className="flex items-center gap-2">
                    <Btn variant="primary" size="sm" onClick={applyScale} disabled={scale === 1}>
                      Apply and re-run
                    </Btn>
                    <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                      spread is currently {fmt(pStats.std, 4)}
                    </span>
                  </div>
                </div>
              </Panel>

              <Panel
                title="Switch an attention head off"
                subtitle="The cleanest intervention there is, and a real interpretability technique."
                right={<InfoDot info={V.commsAblate} title="Ablation" />}
              >
                <div className="grid gap-2">
                  {Array.from({ length: cfg.nLayers }, (_, l) => (
                    <div key={l} className="flex flex-wrap items-center gap-1.5">
                      <span className="mono w-16 shrink-0 text-[11px]" style={{ color: 'var(--text-3)' }}>
                        block {l}
                      </span>
                      {Array.from({ length: cfg.nHeads }, (_, h) => {
                        const off = !!ablated[`${l}:${h}`];
                        return (
                          <Btn
                            key={h}
                            size="sm"
                            active={off}
                            onClick={() => toggleAblate(l, h)}
                            title={off ? 'Switch it back on' : 'Switch it off'}
                          >
                            {off ? `head ${h} off` : `head ${h}`}
                          </Btn>
                        );
                      })}
                    </div>
                  ))}
                </div>
                <div className="mt-3">
                  <Callout tone="insight">
                    Switch one off and read the run log. In a model this small most heads have not
                    specialised, so removing several changes very little &mdash; which tells you the capacity
                    is redundant, not that the heads do nothing. In a large model, removing a single head can
                    reliably break one specific behaviour, and that is how researchers work out what
                    individual heads are for.
                  </Callout>
                </div>
              </Panel>
            </>
          )}

          {/* ----------------------------------------------- guardrails -- */}
          {sub === 'guardrails' && (
            <>
              <Panel
                title="How a guardrail is actually built"
                subtitle="Nothing in the model knows a rule exists."
              >
                <Depth
                  plain={
                    <>
                      <p className="mb-2">
                        There is no list of forbidden things anywhere inside a model. A model is weights and
                        arithmetic; it has no place to keep a rule and no mechanism for consulting one. Every
                        guardrail you have ever met is one of a few things bolted on around that arithmetic.
                      </p>
                      <p className="mb-2">
                        The blunt one is below: strike tokens out at the moment of choosing. The model still
                        wants them, still assigns them probability, and is never informed. From its side
                        nothing happened at all.
                      </p>
                      <p>
                        The subtle one is also below: find a direction inside the model that corresponds to a
                        behaviour, and lean on it while the model runs. No weight is touched. This is how a
                        published refusal direction can be switched off in a real model without retraining
                        it, and it is the reason "we removed the rule" is not a thing anyone can do.
                      </p>
                    </>
                  }
                  math={
                    <>
                      <p className="mb-2">
                        Token blocking sets <span className="mono">p_i &larr; 0</span> for banned{' '}
                        <span className="mono">i</span>, then renormalises. It composes after the model, so
                        it cannot alter <span className="mono">h</span> in any way.
                      </p>
                      <p>
                        Steering adds <span className="mono">&alpha;v</span> to the residual stream at one
                        layer, with <span className="mono">v</span> the unit difference-in-means between two
                        behaviour sets. The weights are unchanged; the trajectory is not.
                      </p>
                    </>
                  }
                  code={
                    <Code>{`// blunt: after the model has finished
probs[bannedId] = 0; renormalise();

// subtle: while the model is running
h[layer] = h[layer] + scale * direction;`}</Code>
                  }
                />
              </Panel>

              <Panel title="Block words outright" subtitle="The bluntest mechanism there is.">
                <Field label="Words to block, comma separated">
                  <input
                    value={banText}
                    onChange={(e) => setBanText(e.target.value)}
                    placeholder="garden, river"
                    spellCheck={false}
                    className="focus-ring mono w-full rounded-lg px-2.5 py-2 text-[12px]"
                    style={{ background: 'var(--bg-2)', border: '1px solid var(--border)', color: 'var(--text)' }}
                  />
                </Field>

                {banResolved.length > 0 && (
                  <div className="mt-3 grid gap-2">
                    {banResolved.map((r) => (
                      <div key={r.fragment} className="text-[12px]" style={{ color: 'var(--text-2)' }}>
                        <span className="mono" style={{ color: 'var(--text)' }}>
                          {r.fragment}
                        </span>{' '}
                        &rarr; {r.ids.length} vocabulary entries:{' '}
                        <span className="mono" style={{ color: 'var(--text-3)' }}>
                          {r.pieces.slice(0, 14).join(' ')}
                          {r.pieces.length > 14 ? ' ...' : ''}
                        </span>
                        {r.split && (
                          <div className="mt-1">
                            <Badge tone="warn">split across tokens</Badge>{' '}
                            <span className="text-[11.5px]" style={{ color: 'var(--text-3)' }}>
                              this word is not stored as one piece, so it has to be blocked by its opening
                              piece &mdash; which also blocks every other word that starts the same way
                            </span>
                          </div>
                        )}
                      </div>
                    ))}
                    <div className="flex items-center gap-2">
                      <Btn variant="primary" size="sm" onClick={applyBan}>
                        Apply to the sampler
                      </Btn>
                      <Btn
                        size="sm"
                        onClick={() => {
                          setBanText('');
                          setIv((p) => ({ ...p, banned: null }));
                        }}
                      >
                        Clear
                      </Btn>
                      <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                        {banCount} of {trainer.tok.size} entries
                      </span>
                    </div>
                  </div>
                )}
              </Panel>

              <Panel
                title="Steer it from the inside"
                subtitle="Find a direction that separates two behaviours, then push along it."
              >
                <div className="grid gap-3 sm:grid-cols-2">
                  <Field label="Group A, one per line">
                    <textarea
                      value={groupA}
                      onChange={(e) => setGroupA(e.target.value)}
                      rows={4}
                      spellCheck={false}
                      className="focus-ring mono w-full resize-y rounded-lg px-2.5 py-2 text-[11.5px]"
                      style={{ background: 'var(--bg-2)', border: '1px solid var(--border)', color: 'var(--text)' }}
                    />
                  </Field>
                  <Field label="Group B, one per line">
                    <textarea
                      value={groupB}
                      onChange={(e) => setGroupB(e.target.value)}
                      rows={4}
                      spellCheck={false}
                      className="focus-ring mono w-full resize-y rounded-lg px-2.5 py-2 text-[11.5px]"
                      style={{ background: 'var(--bg-2)', border: '1px solid var(--border)', color: 'var(--text)' }}
                    />
                  </Field>
                </div>
                <div className="mt-3 grid gap-3">
                  <Slider
                    label="Read the direction at block"
                    value={steerLayer}
                    min={-1}
                    max={cfg.nLayers - 1}
                    step={1}
                    onChange={setSteerLayer}
                    format={(v) => (v < 0 ? 'the embedding' : `block ${v}`)}
                    info={V.steerLayer}
                  />
                  <Btn size="sm" variant="primary" onClick={findDirection}>
                    Find the direction
                  </Btn>
                  {dir && (
                    <>
                      <div className="grid grid-cols-3 gap-2">
                        <Stat label="Separation" value={fmt(dir.separation, 3)} hint="How far apart the two groups were" />
                        <Stat label="Group A" value={`${dir.countA} lines`} />
                        <Stat label="Group B" value={`${dir.countB} lines`} />
                      </div>
                      <Slider
                        label="Push along it"
                        value={steerScale}
                        min={-10}
                        max={10}
                        step={0.5}
                        onChange={setSteerScale}
                        info={V.steerStrength}
                      />
                      <Btn size="sm" onClick={() => runNow(`steering ${fmt(steerScale, 1)} at block ${steerLayer}`)}>
                        Re-run with this steering
                      </Btn>
                      <Callout tone="insight">
                        Set it to zero and the output is bit-for-bit the unmodified model, because adding
                        zero changes nothing. Every weight is exactly where it was before you started, and
                        can be verified as such. The behaviour changed and the model did not.
                      </Callout>
                    </>
                  )}
                </div>
              </Panel>
            </>
          )}

          {/* ---------------------------------------------------- teach -- */}
          {sub === 'teach' && (
            <>
              <Panel title="Teach it something new" subtitle="The same process that trained it, pointed at your examples.">
                <Field label="Examples, one per line" info={V.commsTeachText}>
                  <textarea
                    value={lesson}
                    onChange={(e) => setLesson(e.target.value)}
                    rows={5}
                    spellCheck={false}
                    className="focus-ring mono w-full resize-y rounded-lg px-2.5 py-2 text-[12px]"
                    style={{ background: 'var(--bg-2)', border: '1px solid var(--border)', color: 'var(--text)' }}
                  />
                </Field>
                <div className="mt-3 grid gap-3 sm:grid-cols-2">
                  <Slider
                    label="Learning rate"
                    value={teachLr}
                    min={0.0002}
                    max={0.02}
                    step={0.0002}
                    onChange={setTeachLr}
                    format={(v) => v.toExponential(1)}
                    info={V.commsTeachLr}
                  />
                  <Slider
                    label="Passes over the examples"
                    value={teachEpochs}
                    min={1}
                    max={60}
                    step={1}
                    onChange={setTeachEpochs}
                    format={(v) => String(v)}
                    info={V.commsTeachEpochs}
                  />
                </div>
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <Btn variant="primary" onClick={startTeaching} disabled={teaching}>
                    {teaching ? 'Teaching...' : 'Teach it'}
                  </Btn>
                  {ft && !teaching && (
                    <Btn onClick={revertTeaching}>Undo the lesson</Btn>
                  )}
                  {ft && (
                    <span className="text-[11.5px]" style={{ color: 'var(--text-3)' }}>
                      {ft.windows.length} training windows &middot; pass {ft.pass} of {ft.cfg.epochs}
                    </span>
                  )}
                </div>

                {ft && ft.vocab.dropped.length > 0 && (
                  <div className="mt-3">
                    <Callout tone="warn" title="Characters that had to be dropped">
                      <span className="mono">{ft.vocab.droppedLabels.join('  ')}</span> are not in this model's
                      vocabulary. The vocabulary was decided before pretraining started and cannot grow now,
                      so those characters never reached the model at all. Real fine-tuning hits exactly this
                      wall, which is why vocabulary is chosen so carefully up front.
                    </Callout>
                  </div>
                )}

                {ft && ft.windows.length === 0 && (
                  <div className="mt-3">
                    <Callout tone="err">
                      Nothing here is long enough to learn from. Each line needs at least two tokens: one to
                      read and one to predict.
                    </Callout>
                  </div>
                )}
              </Panel>

              {ft && ft.history.length > 0 && (
                <Panel title="What it is costing" subtitle="Two curves, and the gap between them is the whole story.">
                  <LineChart series={teachSeries} height={180} yLabel="loss" xLabel="pass" />
                  <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
                    <Stat label="Lesson, before" value={fmt(ft.baselineNew, 3)} />
                    <Stat
                      label="Lesson, now"
                      value={fmt(ft.evalNew(), 3)}
                      tone={ft.evalNew() < ft.baselineNew ? 'ok' : 'warn'}
                    />
                    <Stat label="Old material, before" value={fmt(ft.baselineOld, 3)} />
                    <Stat
                      label="Old material, now"
                      value={fmt(ft.evalOld(), 3)}
                      tone={ft.evalOld() > ft.baselineOld * 1.1 ? 'err' : 'ok'}
                    />
                  </div>
                  <div className="mt-3">
                    <Stat label="How far the weights have moved" value={fmtPct(ft.drift(), 2)} />
                  </div>
                  {ft.evalOld() > ft.baselineOld * 1.15 && (
                    <div className="mt-3">
                      <Callout tone="warn" title="Catastrophic forgetting, happening in front of you">
                        The loss on your examples is falling and the loss on everything the model already
                        knew is climbing. Nothing is broken; this is simply what fine-tuning does. The
                        weights that held the old behaviour are the same weights being moved to hold the
                        new one. Real labs fight this by mixing original data back into the fine-tune, by
                        using a far smaller learning rate, or by freezing most of the model and training a
                        small adapter instead.
                      </Callout>
                    </div>
                  )}
                </Panel>
              )}
            </>
          )}
        </div>

        {/* ------------------------------------------------------- log -- */}
        <div className="grid content-start gap-5">
          <Panel
            title="Run log"
            subtitle="Same prompt, same seed, every time. Any difference came from what you changed."
          >
            {log.length === 0 ? (
              <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>
                Press <span style={{ color: 'var(--text-2)' }}>Re-run the prompt</span> to take a baseline,
                then change something.
              </p>
            ) : (
              <div className="grid gap-2.5">
                {log.map((r, i) => (
                  <div
                    key={r.at + '-' + i}
                    className="rounded-lg p-2.5"
                    style={{ background: i === 0 ? 'var(--panel-2)' : 'transparent', border: '1px solid var(--border)' }}
                  >
                    <div className="mb-1 text-[10.5px] font-medium uppercase tracking-[0.07em]" style={{ color: i === 0 ? 'var(--accent)' : 'var(--text-3)' }}>
                      {i === 0 ? 'latest' : `${i} ago`} &middot; {r.note}
                    </div>
                    <div className="mono text-[11.5px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                      {r.text || '(nothing)'}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Panel>

          {sub === 'weights' && (
            <Panel title="Pick a table" pad={false}>
              <div className="max-h-[420px] overflow-y-auto">
                {M.params.map((p) => {
                  const on = param?.name === p.name;
                  return (
                    <button
                      key={p.name}
                      onClick={() => {
                        setParamName(p.name);
                        setCell(null);
                      }}
                      className="flex w-full items-center justify-between gap-2 px-4 py-2 text-left"
                      style={{
                        background: on ? 'var(--panel-2)' : 'transparent',
                        borderBottom: '1px solid var(--border)',
                        cursor: 'pointer',
                      }}
                    >
                      <span className="mono truncate text-[11.5px]" style={{ color: on ? 'var(--accent)' : 'var(--text-2)' }}>
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
          )}

          <Panel title="What this tab is for">
            <p className="text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
              Three ways to change a model, which are genuinely different things. Editing weights changes
              what the model <em>is</em>. A guardrail changes what it is <em>allowed to say</em>, while
              leaving what it is completely untouched. Teaching changes what it is, but by running the same
              process that built it in the first place. Confusing the second for the first is the single
              most common misunderstanding about how AI systems are controlled.
            </p>
          </Panel>
        </div>
      </div>
    </div>
  );
}
