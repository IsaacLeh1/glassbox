import { useCallback, useMemo, useRef, useState } from 'react';
import { buildLMTrainer } from '../engine/lmPresets';
import type { LMTrainer } from '../engine/lmTrainer';
import { sampleToken, type Steer } from '../engine/transformer';
import { contrastDirection, resolveBan, projectOnto, type Direction } from '../engine/steering';
import { mulberry32 } from '../engine/tensor';
import { useFrameLoop } from '../ui/loop';
import {
  Badge,
  BackgroundNotice,
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
  Slider,
  Stat,
  fmt,
  fmtPct,
} from '../ui/kit';
import { BarMeter } from '../ui/viz';
import { V } from '../content/varInfo';

/* ------------------------------------------------------- shared model -- */

interface Ctx {
  trainer: LMTrainer;
  trained: boolean;
  bump: () => void;
}

function TrainGate({ trainer, trained, bump }: Ctx) {
  const [running, setRunning] = useState(false);
  useFrameLoop(running, () => {
    const n = trainer.runSlice(11);
    if (n === 0 || trainer.status === 'done') setRunning(false);
    bump();
  });

  if (trained) return null;
  return (
    <Panel
      title="This module needs a trained model"
      subtitle="Guardrails only make sense on a model that has opinions to constrain"
      right={
        <div className="flex items-center gap-1.5">
          <BackgroundNotice running={running} />
          <Btn variant="primary" size="sm" onClick={() => setRunning((r) => !r)}>
            {running ? 'Pause' : trainer.step > 0 ? 'Resume' : 'Train it'}
          </Btn>
        </div>
      }
    >
      <p className="mb-3 text-[12.5px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
        An untrained model produces uniform noise, and blocking a word it was never going to say teaches you
        nothing. Train it for a few seconds first; everything below then operates on a model that genuinely
        has preferences.
      </p>
      <ProgressBar value={trainer.progress} />
      <div className="mt-2 grid grid-cols-3 gap-3">
        <Stat label="step" value={`${trainer.step} / ${trainer.cfg.steps}`} />
        <Stat label="loss" value={fmt(trainer.latest()?.loss ?? NaN, 3)} tone="accent" />
        <Stat label="ready at" value="~120 steps" />
      </div>
    </Panel>
  );
}

/* ============================================== tab 1: the layers ======= */

interface LayerDef {
  id: string;
  name: string;
  where: string;
  how: string;
  stops: string;
  misses: string;
  changesWeights: boolean;
}

const LAYERS: LayerDef[] = [
  {
    id: 'data',
    name: 'Choosing the training data',
    where: 'Before the model exists',
    how: 'Material is filtered out of the training set so the model never learns it well in the first place. This is not really a guardrail; it is prevention.',
    stops: 'Capabilities that were never in the data to begin with.',
    misses: 'Anything that can be reassembled from pieces that individually looked harmless. Filtering is never complete, and models generalise.',
    changesWeights: true,
  },
  {
    id: 'post',
    name: 'Post-training',
    where: 'Baked into the weights',
    how: 'After the model learns language, it is trained again on examples of good and bad responses, using human or AI preference judgements. The weights shift so that refusing genuinely becomes the most likely continuation.',
    stops: 'The broad shape of most misuse, because the model now prefers not to do it.',
    misses: 'Framings that look unlike the training examples. It is a learned tendency, not a rule, so it can be argued with.',
    changesWeights: true,
  },
  {
    id: 'system',
    name: 'The system prompt',
    where: 'Text at the front of the conversation',
    how: 'Instructions are placed in the context before anything the user says. Nothing about the model changes; it is simply reading more.',
    stops: 'Tone, format, scope, and anything the model was already willing to comply with.',
    misses: 'Anything later text can talk it out of. This is exactly why prompt injection works: the instruction and the attack arrive through the same channel.',
    changesWeights: false,
  },
  {
    id: 'filter',
    name: 'Input and output classifiers',
    where: 'Outside the model entirely',
    how: 'A separate, usually much smaller model reads the request, or the finished response, and scores it. Above a threshold the exchange is blocked or replaced.',
    stops: 'Clear-cut cases, reliably, and it keeps working even if the main model is fully persuaded.',
    misses: 'Anything it was not trained to recognise. It also produces false positives, which is why refusals sometimes feel arbitrary.',
    changesWeights: false,
  },
  {
    id: 'decode',
    name: 'Blocking tokens at sampling time',
    where: 'After the model has decided',
    how: 'Specific tokens are struck from the list of possible next tokens before one is chosen. Mechanical and absolute.',
    stops: 'Those exact tokens, with certainty. Nothing gets through.',
    misses: 'Almost everything else. The model simply chooses its next-favourite wording, and meaning is not made of fixed strings.',
    changesWeights: false,
  },
];

function LayersTab() {
  const [openId, setOpenId] = useState('post');
  const open = LAYERS.find((l) => l.id === openId)!;

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
      <div className="space-y-4">
        <Panel
          title="Guardrails are not one thing"
          subtitle="Five different mechanisms, at five different points, with very different properties"
        >
          <div className="space-y-1.5">
            {LAYERS.map((l, i) => {
              const on = l.id === openId;
              return (
                <button
                  key={l.id}
                  onClick={() => setOpenId(l.id)}
                  className="focus-ring block w-full cursor-pointer rounded-lg p-3 text-left transition-colors"
                  style={{
                    background: on ? 'color-mix(in srgb, var(--accent) 10%, transparent)' : 'var(--panel-2)',
                    border: `1px solid ${on ? 'var(--accent)' : 'var(--border)'}`,
                  }}
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="flex items-center gap-2">
                      <span className="mono text-[10px]" style={{ color: 'var(--text-3)' }}>
                        {i + 1}
                      </span>
                      <span className="text-[12.5px] font-semibold" style={{ color: 'var(--text)' }}>
                        {l.name}
                      </span>
                    </span>
                    <Badge tone={l.changesWeights ? 'accent' : 'neutral'}>
                      {l.changesWeights ? 'changes the weights' : 'leaves the model untouched'}
                    </Badge>
                  </div>
                  <div className="mt-0.5 text-[11px]" style={{ color: 'var(--text-3)' }}>
                    {l.where}
                  </div>
                </button>
              );
            })}
          </div>

          <div className="mt-4">
            <Callout tone="insight" title="Why there are five and not one">
              Each of these fails differently, so they are stacked deliberately. The system prompt is easy to
              change but easy to talk around. Post-training is robust but cannot be updated quickly. A
              classifier is dependable but crude and cannot see intent. Token blocking is absolute but nearly
              useless on its own. Safety comes from the overlap, not from any single layer being perfect.
            </Callout>
          </div>
        </Panel>

        <Panel title={open.name} subtitle={open.where}>
          <div className="space-y-3 text-[12.5px] leading-relaxed">
            <div>
              <Label>how it works</Label>
              <p className="mt-1" style={{ color: 'var(--text-2)' }}>
                {open.how}
              </p>
            </div>
            <div>
              <Label>what it stops</Label>
              <p className="mt-1" style={{ color: 'var(--ok)' }}>
                {open.stops}
              </p>
            </div>
            <div>
              <Label>what gets past it</Label>
              <p className="mt-1" style={{ color: 'var(--warn)' }}>
                {open.misses}
              </p>
            </div>
          </div>
        </Panel>
      </div>

      <div className="space-y-4">
        <Panel title="Where each one sits">
          <div className="space-y-2 text-[11.5px]">
            {[
              ['your message', 'var(--text-3)'],
              ['input classifier', 'var(--warn)'],
              ['system prompt added', 'var(--accent)'],
              ['the model runs', 'var(--pos)'],
              ['token blocking', 'var(--warn)'],
              ['output classifier', 'var(--warn)'],
              ['what you see', 'var(--text-3)'],
            ].map(([label, c], i) => (
              <div key={label} className="flex items-center gap-2">
                <span className="mono w-4 shrink-0 text-[9.5px]" style={{ color: 'var(--text-3)' }}>
                  {i + 1}
                </span>
                <span
                  className="flex-1 rounded px-2 py-1.5"
                  style={{ background: `color-mix(in srgb, ${c} 10%, transparent)`, color: 'var(--text-2)' }}
                >
                  {label}
                </span>
              </div>
            ))}
          </div>
          <p className="mt-3 text-[11.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
            Only step 4 is the model. Everything else is ordinary software wrapped around it, which is worth
            remembering when a product behaves in a way the model itself would not.
          </p>
        </Panel>

        <Callout tone="warn" title="The honest limitation">
          None of these are locks, and none of them are code that checks a condition. Four of the five are
          statistical systems that usually behave, and the fifth is a string match. Treat published safety
          behaviour as a strong tendency, not a guarantee, and never as the only thing standing between a
          system and a bad outcome.
        </Callout>
      </div>
    </div>
  );
}

/* ============================================ tab 2: blocking tokens ==== */

function BlockTab({ trainer, trained }: Ctx) {
  const [prompt, setPrompt] = useState('the quiet fox');
  const [banText, setBanText] = useState('garden, river');
  const [out, setOut] = useState<{ free: string; blocked: string } | null>(null);

  const resolved = useMemo(
    () => resolveBan(trainer.tok, banText.split(',')),
    [banText, trainer],
  );
  const banned = useMemo(() => {
    const s = new Set<number>();
    for (const r of resolved) for (const id of r.ids) s.add(id);
    return s;
  }, [resolved]);

  const preview = useMemo(() => {
    if (!trained) return null;
    const ids = trainer.tok.encode(prompt);
    if (ids.length === 0) return null;
    const { probs } = trainer.model.predictNext(ids);
    return sampleToken(probs, { temperature: 0.9, topK: 0, topP: 1, banned }, () => 0.5);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prompt, banned, trainer, trained, trainer.step]);

  const run = useCallback(() => {
    const gen = (ban: Set<number> | undefined) => {
      const rand = mulberry32(99);
      let ids = trainer.tok.encode(prompt);
      for (let i = 0; i < 34; i++) {
        const { probs } = trainer.model.predictNext(ids);
        ids = [...ids, sampleToken(probs, { temperature: 0.9, topK: 15, topP: 0.95, banned: ban }, rand).chosen];
      }
      return trainer.tok.decode(ids);
    };
    setOut({ free: gen(undefined), blocked: gen(banned) });
  }, [prompt, banned, trainer]);

  if (!trained) return <Empty>Train the model above first.</Empty>;

  const ranked = preview ? [...preview.candidates].sort((a, b) => b.prob - a.prob).slice(0, 12) : [];

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
      <div className="space-y-4">
        <Panel
          title="Striking tokens off the list"
          subtitle="The bluntest guardrail there is, and the only one that is absolutely reliable"
          right={
            <Btn size="sm" variant="primary" onClick={run}>
              Generate both ways
            </Btn>
          }
        >
          <div className="grid gap-3 md:grid-cols-2">
            <Field label="prompt">
              <input value={prompt} onChange={(e) => setPrompt(e.target.value)} className="mono" maxLength={80} />
            </Field>
            <Field label="forbidden words, comma separated" info={V.banList}>
              <input value={banText} onChange={(e) => setBanText(e.target.value)} className="mono" maxLength={120} />
            </Field>
          </div>
          <div className="mt-2 space-y-1.5">
            {resolved.length === 0 && (
              <p className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                Nothing listed, so nothing is blocked.
              </p>
            )}
            {resolved.map((r) => (
              <div key={r.fragment} className="flex flex-wrap items-center gap-1.5 text-[11px]">
                <span className="mono" style={{ color: 'var(--text-2)' }}>
                  {r.fragment}
                </span>
                <span style={{ color: 'var(--text-3)' }}>→</span>
                {r.pieces.map((p, i) => (
                  <span
                    key={i}
                    className="mono rounded px-1.5 py-[1px]"
                    style={{
                      background: 'color-mix(in srgb, var(--err) 14%, transparent)',
                      color: 'var(--err)',
                      textDecoration: 'line-through',
                    }}
                  >
                    {p}
                  </span>
                ))}
                {r.split && (
                  <span style={{ color: 'var(--warn)' }}>
                    not a single token, so its opening piece is blocked instead
                  </span>
                )}
              </div>
            ))}
            <p className="text-[11px]" style={{ color: 'var(--text-3)' }}>
              {banned.size} of {trainer.vocabSize} vocabulary entries struck out in total.
            </p>
          </div>

          {resolved.some((r) => r.split) && (
            <div className="mt-3">
              <Callout tone="warn" title="This is blunter than it looks">
                At least one of these words is not stored as a single token, so it cannot be blocked cleanly.
                The only way to stop it is to forbid the piece it starts with — which also silently blocks
                every other word beginning with that same piece. Guardrails built on word lists routinely
                over-block for exactly this reason.
              </Callout>
            </div>
          )}

          {out && (
            <div className="mt-4 grid gap-3 md:grid-cols-2">
              <div>
                <Label>with nothing blocked</Label>
                <div
                  className="mono mt-1 rounded-lg p-2.5 text-[11.5px] leading-relaxed"
                  style={{ background: 'var(--bg-2)', border: '1px solid var(--border)', color: 'var(--text-2)' }}
                >
                  {out.free}
                </div>
              </div>
              <div>
                <Label>with those tokens forbidden</Label>
                <div
                  className="mono mt-1 rounded-lg p-2.5 text-[11.5px] leading-relaxed"
                  style={{
                    background: 'color-mix(in srgb, var(--warn) 6%, var(--bg-2))',
                    border: '1px solid var(--border)',
                    color: 'var(--text-2)',
                  }}
                >
                  {out.blocked}
                </div>
              </div>
            </div>
          )}

          {out && (
            <div className="mt-3">
              <Callout tone="insight" title="Look at what it did instead">
                The forbidden words are genuinely absent — that part is guaranteed, because they were removed
                before the choice was made. But the sentence did not break. The model simply took its
                next-favourite option and carried on, and the meaning is usually close to what it wanted to
                say anyway. That is the whole problem with blocking words: you are constraining the spelling,
                not the intent.
              </Callout>
            </div>
          )}
        </Panel>

        <Panel title="The moment of blocking" subtitle={`Next-token candidates after "${prompt}"`}>
          {ranked.length === 0 ? (
            <Empty>Enter a prompt.</Empty>
          ) : (
            <div className="space-y-1.5">
              {ranked.map((c) => (
                <div key={c.id} className="flex items-center gap-2" style={{ opacity: c.blocked ? 0.5 : 1 }}>
                  <span
                    className="mono w-[86px] shrink-0 truncate text-[11.5px]"
                    style={{
                      color: c.blocked ? 'var(--err)' : 'var(--text-2)',
                      textDecoration: c.blocked ? 'line-through' : 'none',
                    }}
                  >
                    {trainer.tok.display(c.id) || '·'}
                  </span>
                  <div className="flex-1">
                    <BarMeter
                      value={c.prob}
                      max={ranked[0].prob || 1}
                      color={c.blocked ? 'var(--err)' : 'var(--accent)'}
                      height={7}
                    />
                  </div>
                  <span className="mono tnum w-[48px] shrink-0 text-right text-[10.5px]" style={{ color: 'var(--text-3)' }}>
                    {c.blocked ? 'blocked' : `${(c.prob * 100).toFixed(1)}%`}
                  </span>
                </div>
              ))}
            </div>
          )}
          {preview && preview.blockedMass > 0 && (
            <p className="mt-3 text-[11.5px]" style={{ color: 'var(--text-2)' }}>
              The ban removed <span className="mono" style={{ color: 'var(--warn)' }}>{fmtPct(preview.blockedMass)}</span>{' '}
              of the probability the model had assigned. That mass is not discarded — it is shared out among
              everything still allowed, which is why something always gets said.
            </p>
          )}
        </Panel>
      </div>

      <div className="space-y-4">
        <Panel title="What this teaches">
          <Depth
            plain={
              <div className="space-y-2">
                <p>
                  The model produces a score for every word it could say next. Blocking works by deleting some
                  of those options from the list before one is picked. The model is never told; it does not
                  know a rule exists.
                </p>
                <p>
                  This is completely reliable for the exact words you listed, and almost useless for anything
                  else. Block one word and the model says a synonym. Block the synonyms and it describes the
                  thing instead.
                </p>
                <p>
                  There is a subtler trap too. Words are split into tokens, so banning a word does not
                  necessarily ban every way of spelling it — a different capitalisation or a leading space is a
                  different token entirely.
                </p>
              </div>
            }
            math={
              <div className="space-y-2">
                <p>
                  Given logits <span className="mono">z</span>, banning set <span className="mono">B</span> is
                  equivalent to setting <span className="mono">z_i = −∞</span> for{' '}
                  <span className="mono">i ∈ B</span> before the softmax, which puts exactly zero mass on
                  those tokens.
                </p>
                <p>
                  The remaining distribution is renormalised by{' '}
                  <span className="mono">1 / (1 − Σ_B p_i)</span>, so the relative ordering of everything
                  permitted is unchanged. The model&apos;s preferences among allowed tokens are untouched.
                </p>
              </div>
            }
            code={
              <Code>{`for (const id of cfg.banned) {
  blockedMass += probs[id];
  probs[id] = 0;
}
const left = probs.reduce((a, b) => a + b, 0);
for (let i = 0; i < probs.length; i++) probs[i] /= left;`}</Code>
            }
          />
        </Panel>

        <Callout tone="warn" title="Try to defeat it">
          Block <span className="mono">garden</span> and generate. Then look at where the sentence goes
          instead. Now block that word too. You can keep going, and the model will keep finding somewhere to
          land, because its intent lives in the weights and you are only editing the output.
        </Callout>
      </div>
    </div>
  );
}

/* =========================================== tab 3: steering behaviour == */

const DEFAULT_A = 'the owl watches at night.\nthe moth hides until dark.\nthe crow rests at night.\nthe fox waits until dark.';
const DEFAULT_B = 'the bee sings at dawn.\nthe hare runs all morning.\nthe cat waits at dawn.\nthe dog sings all morning.';

function SteerTab({ trainer, trained }: Ctx) {
  const [aText, setAText] = useState(DEFAULT_A);
  const [bText, setBText] = useState(DEFAULT_B);
  const [layer, setLayer] = useState(Math.max(0, trainer.model.cfg.nLayers - 1));
  const [strength, setStrength] = useState(0);
  const [prompt, setPrompt] = useState('the');
  const [dir, setDir] = useState<Direction | null>(null);
  const [samples, setSamples] = useState<{ s: number; text: string }[]>([]);

  const build = useCallback(() => {
    const d = contrastDirection(
      trainer.model,
      trainer.tok,
      aText.split('\n').filter((l) => l.trim()),
      bText.split('\n').filter((l) => l.trim()),
      layer,
    );
    setDir(d);
    setSamples([]);
  }, [trainer, aText, bText, layer]);

  const gen = useCallback(
    (scale: number) => {
      const steer: Steer | undefined = dir ? { layer: dir.layer, vec: dir.vec, scale } : undefined;
      const rand = mulberry32(7);
      let ids = trainer.tok.encode(prompt);
      if (ids.length === 0) ids = [0];
      for (let i = 0; i < 28; i++) {
        const { probs } = trainer.model.predictNext(ids, steer);
        ids = [...ids, sampleToken(probs, { temperature: 0.85, topK: 15, topP: 0.95 }, rand).chosen];
      }
      return trainer.tok.decode(ids);
    },
    [trainer, prompt, dir],
  );

  const sweep = useCallback(() => {
    setSamples([-6, -3, 0, 3, 6].map((s) => ({ s, text: gen(s) })));
  }, [gen]);

  if (!trained) return <Empty>Train the model above first.</Empty>;

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
      <div className="space-y-4">
        <Panel
          title="Finding a behaviour inside the model"
          subtitle="Two groups of examples, one subtraction, and a direction you can push along"
          right={
            <div className="flex gap-1.5">
              <Btn size="sm" variant="primary" onClick={build}>
                Find the direction
              </Btn>
              <Btn size="sm" onClick={sweep} disabled={!dir}>
                Sweep strength
              </Btn>
            </div>
          }
        >
          <div className="grid gap-3 md:grid-cols-2">
            <Field label="group A — one sentence per line">
              <textarea value={aText} onChange={(e) => setAText(e.target.value)} rows={4} className="mono" />
            </Field>
            <Field label="group B — the contrast">
              <textarea value={bText} onChange={(e) => setBText(e.target.value)} rows={4} className="mono" />
            </Field>
          </div>

          <div className="mt-3 grid gap-3 md:grid-cols-2">
            <Slider
              label="layer to read and steer at"
              info={V.steerLayer}
              value={layer}
              min={-1}
              max={Math.max(0, trainer.model.cfg.nLayers - 1)}
              step={1}
              format={(x) => (Math.round(x) < 0 ? 'after embedding' : `layer ${Math.round(x)}`)}
              onChange={(x) => setLayer(Math.round(x))}
            />
            <Field label="prompt to continue">
              <input value={prompt} onChange={(e) => setPrompt(e.target.value)} className="mono" maxLength={60} />
            </Field>
          </div>

          {dir && (
            <>
              <div className="mt-4 grid grid-cols-3 gap-3">
                <Stat label="direction found" value={`${dir.vec.length} numbers`} tone="accent" />
                <Stat
                  label="separation"
                  value={fmt(dir.separation, 3)}
                  hint="How far apart the two groups sat before the direction was normalised. Larger means a cleaner distinction."
                />
                <Stat label="examples used" value={`${dir.countA} vs ${dir.countB}`} />
              </div>

              <div className="mt-4">
                <Slider
                  label="how hard to push"
                  info={V.steerStrength}
                  value={strength}
                  min={-10}
                  max={10}
                  step={0.5}
                  format={(x) => (x === 0 ? 'off' : x > 0 ? `+${x} toward A` : `${x} toward B`)}
                  onChange={setStrength}
                />
                <div className="mt-2 flex gap-1.5">
                  <Btn size="sm" onClick={() => setSamples([{ s: strength, text: gen(strength) }])}>
                    Generate at this strength
                  </Btn>
                </div>
              </div>
            </>
          )}

          {samples.length > 0 && (
            <div className="mt-4 space-y-2">
              {samples.map((s) => (
                <div key={s.s} className="flex items-start gap-3">
                  <span
                    className="mono mt-1 w-[62px] shrink-0 text-[10.5px]"
                    style={{ color: s.s > 0 ? 'var(--pos)' : s.s < 0 ? 'var(--neg)' : 'var(--text-3)' }}
                  >
                    {s.s === 0 ? 'unsteered' : s.s > 0 ? `+${s.s} → A` : `${s.s} → B`}
                  </span>
                  <div
                    className="mono flex-1 rounded p-2 text-[11.5px] leading-relaxed"
                    style={{
                      background: 'var(--bg-2)',
                      border: `1px solid ${s.s === 0 ? 'var(--accent)' : 'var(--border)'}`,
                      color: 'var(--text-2)',
                    }}
                  >
                    {s.text}
                  </div>
                </div>
              ))}
            </div>
          )}

          {!dir && (
            <div className="mt-3">
              <Callout tone="info">
                Press <strong>Find the direction</strong>. The model is run over both groups, the internal
                state is averaged for each, and one average is subtracted from the other. What remains is a
                direction in the model&apos;s own representation that separates the two ideas.
              </Callout>
            </div>
          )}
        </Panel>

        <Panel title="What just happened">
          <Depth
            plain={
              <div className="space-y-2">
                <p>
                  As the model reads, it keeps a running set of numbers for each word — its working state. If
                  you show it many night sentences and many morning sentences and average those states
                  separately, the difference between the two averages points from one idea toward the other.
                </p>
                <p>
                  Adding that difference back in while the model runs pushes it toward one side. Subtracting
                  pushes it to the other. No weight was edited, nothing was retrained, and the model has no
                  idea it happened.
                </p>
                <p>
                  This is not a toy. The same method is how researchers locate the internal direction
                  responsible for a model refusing a request — and, by subtracting it, switch the refusal off.
                  It is one of the clearest demonstrations that a behaviour is a direction in a space, not a
                  rule in a list.
                </p>
              </div>
            }
            math={
              <div className="space-y-2">
                <p>
                  For hidden states <span className="mono">h(x) ∈ ℝ^d</span> at a chosen layer, the direction
                  is the difference in means:
                </p>
                <div className="mono my-1 text-center">v = mean(h(A)) − mean(h(B)),  v̂ = v / ‖v‖</div>
                <p>
                  Steering then computes the forward pass with{' '}
                  <span className="mono">h ← h + α·v̂</span> applied at that layer. Because the residual
                  stream is additive and every later layer reads from it, a constant offset there propagates
                  through the whole remaining computation.
                </p>
                <p>
                  Normalising to unit length means <span className="mono">α</span> is measured in the same
                  units regardless of how separable the two groups were.
                </p>
              </div>
            }
            code={
              <Code>{`// engine/steering.ts
vec[j] = meanA[j] - meanB[j];        // difference in means
vec[j] /= norm;                      // unit length

// engine/transformer.ts -- applied to the residual stream
out.data[i * cols + j] = m.data[i * cols + j] + steer.scale * steer.vec[j];`}</Code>
            }
          />
        </Panel>
      </div>

      <div className="space-y-4">
        <Panel title="Where your prompt already sits">
          {dir ? (
            <>
              <p className="mb-2 text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                How strongly the current prompt already points along the direction, before any steering:
              </p>
              <Stat
                label="projection"
                value={fmt(projectOnto(trainer.model, trainer.tok, prompt, dir), 3)}
                tone="accent"
                hint="Positive means it already leans toward group A."
              />
            </>
          ) : (
            <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>
              Find a direction first.
            </p>
          )}
        </Panel>

        <Callout tone="warn" title="Why this matters for safety">
          If a refusal is a direction that can be added, it is also a direction that can be removed. Anyone
          with access to a model&apos;s weights can compute this vector and subtract it, which is why
          safety trained into open weights cannot be treated as durable. It is a real and widely discussed
          limitation, not a hypothetical one.
        </Callout>

        <Callout tone="insight" title="Try this">
          Set the strength to zero and generate, then to +6, then to −6, from the same prompt with the same
          random seed. Every difference you see comes from a single vector added to the model&apos;s working
          state. The weights are byte-for-byte identical in all three runs.
        </Callout>
      </div>
    </div>
  );
}

/* ========================================== tab 4: how the model knows == */

function KnowsTab() {
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Panel title="There is no list of rules inside the model">
        <Depth
          plain={
            <div className="space-y-3">
              <p>
                The most common assumption about AI safety is that somewhere inside the model there is a list
                of forbidden things, and the model checks its answer against that list. There is no such list.
                There is nowhere to put one. The model is a stack of matrices, and the only thing it does is
                predict the next token.
              </p>
              <p>
                So how does it refuse? During post-training it is shown many examples of requests together
                with responses that people preferred, including refusals to harmful requests. Its weights are
                nudged, again and again, until for that kind of input the highest-scoring next words genuinely
                are the opening of a refusal.
              </p>
              <p>
                It refuses for exactly the same reason it writes grammatically: that is what its weights make
                most likely. Refusing is not a separate faculty bolted on. It is the same next-token
                prediction, with the preference shaped differently.
              </p>
              <div className="rounded-lg p-3" style={{ background: 'var(--bg-2)' }}>
                <p>
                  Two consequences follow, and they explain most of what people find strange about model
                  behaviour:
                </p>
                <ul className="mt-2 space-y-1.5">
                  <li>
                    <strong>It can be talked around.</strong> A tendency learned from examples generalises to
                    things that look like those examples. Make the request look sufficiently unlike them and
                    the tendency weakens. This is not the model being tricked into breaking a rule; there was
                    never a rule to break.
                  </li>
                  <li>
                    <strong>It refuses things it should not.</strong> The same generalisation runs the other
                    way. A harmless request that resembles a harmful one gets the refusal, because the model
                    is matching shape rather than consulting a definition.
                  </li>
                </ul>
              </div>
            </div>
          }
          math={
            <div className="space-y-2">
              <p>
                Post-training optimises a preference objective rather than next-token likelihood alone. For a
                reward model <span className="mono">r</span> and reference policy{' '}
                <span className="mono">π_ref</span>, the usual objective is
              </p>
              <div className="mono my-1 text-center">
                max_π E[r(x, y)] − β · KL(π(·|x) ‖ π_ref(·|x))
              </div>
              <p>
                The KL term keeps the model close to the one that learned language, so capability is retained
                while the preference shifts. Direct preference optimisation reaches the same optimum without
                fitting <span className="mono">r</span> explicitly.
              </p>
              <p>
                The result is still a distribution over tokens. Nothing discrete is introduced, which is
                precisely why the behaviour is a tendency with a magnitude rather than a predicate with a
                truth value.
              </p>
            </div>
          }
          code={
            <Code>{`# There is no function that looks like this anywhere in a model:
if request.is_harmful():        # <- does not exist
    return refusal()

# What exists is the same loop as always:
logits = model(tokens)          # one score per possible next token
probs  = softmax(logits)        # refusal tokens simply score highest here
next   = sample(probs)`}</Code>
          }
        />
      </Panel>

      <div className="space-y-4">
        <Panel title="How each layer decides, side by side">
          <div className="overflow-x-auto">
            <table className="w-full text-[11.5px]" style={{ borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ color: 'var(--text-3)' }}>
                  {['mechanism', 'what decides', 'can it be argued with'].map((h) => (
                    <th key={h} className="px-2 py-1.5 text-left font-medium" style={{ borderBottom: '1px solid var(--border)' }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[
                  ['Post-training', 'Learned preference in the weights', 'Yes — it is a tendency'],
                  ['System prompt', 'Text the model is reading', 'Yes — very easily'],
                  ['Classifier', 'A second model scoring the text', 'Only by fooling that model too'],
                  ['Token blocking', 'A string comparison', 'No, but trivially worked around'],
                ].map(([a, b, c]) => (
                  <tr key={a} style={{ borderBottom: '1px solid var(--border)' }}>
                    <td className="px-2 py-2" style={{ color: 'var(--text)' }}>
                      {a}
                    </td>
                    <td className="px-2 py-2" style={{ color: 'var(--text-2)' }}>
                      {b}
                    </td>
                    <td className="px-2 py-2" style={{ color: c.startsWith('No') ? 'var(--ok)' : 'var(--warn)' }}>
                      {c}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>

        <Callout tone="insight" title="The useful mental model">
          Think of post-training as changing what the model <em>wants</em> to say, the system prompt as
          changing what it has just been <em>told</em>, and classifiers and token blocking as things that
          happen to the text on its way in and out. Almost every confusing incident with a deployed AI system
          becomes legible once you work out which of those four was responsible.
        </Callout>
      </div>
    </div>
  );
}

/* ================================================================ page == */

export default function GuardrailsLab() {
  const [tab, setTab] = useState<'layers' | 'block' | 'steer' | 'knows'>('layers');
  const [, setVersion] = useState(0);
  const trainerRef = useRef<LMTrainer | null>(null);
  if (!trainerRef.current) trainerRef.current = buildLMTrainer('stories', 'small');
  const trainer = trainerRef.current;
  const bump = useCallback(() => setVersion((v) => v + 1), []);
  const trained = trainer.step >= 120;
  const ctx: Ctx = { trainer, trained, bump };

  return (
    <div className="mx-auto max-w-[1400px] px-6 py-6">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <Segmented
          value={tab}
          onChange={setTab}
          options={[
            { id: 'layers', label: '1 · The five layers' },
            { id: 'block', label: '2 · Blocking words' },
            { id: 'steer', label: '3 · Steering behaviour' },
            { id: 'knows', label: '4 · How the model knows' },
          ]}
        />
        <Badge tone={trained ? 'ok' : 'warn'}>{trained ? 'model trained' : 'model untrained'}</Badge>
      </div>

      {tab !== 'layers' && tab !== 'knows' && (
        <div className="mb-4">
          <TrainGate {...ctx} />
        </div>
      )}

      {tab === 'layers' ? (
        <LayersTab />
      ) : tab === 'block' ? (
        <BlockTab {...ctx} />
      ) : tab === 'steer' ? (
        <SteerTab {...ctx} />
      ) : (
        <KnowsTab />
      )}
    </div>
  );
}
