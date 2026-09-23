import { useCallback, useMemo, useState } from 'react';
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
  Select,
  Slider,
  Stat,
  fmt,
  fmtInt,
  fmtPct,
} from '../ui/kit';
import { LineChart } from '../ui/viz';
import { V } from '../content/varInfo';
import { useModel } from '../store/model';
import { buildLMTrainer } from '../engine/lmPresets';
import { GPU_PROFILES } from '../sim/cluster';
import {
  SCHEME_INFO,
  prettyBytes,
  projectSize,
  quantiseModel,
  scaleUp,
  type QuantBits,
  type QuantResult,
  type QuantScheme,
} from '../engine/quantise';
import { cacheBytes, timeCached, timeUncached, type Timing } from '../engine/kvcache';
import {
  MARKET_PRICES,
  SERVE_TARGETS,
  batchCurve,
  money,
  costOf,
  decodeBound,
  loadCurve,
  queue,
  replicasFor,
} from '../engine/serving';
import { DEFAULT_GEN, casesFromIds, modelPredictor, runEval } from '../engine/evals';

/**
 * Step 11: putting a model into service.
 *
 * Training is a bill you pay once. Serving is the bill that never stops, and
 * within a few months it is usually the larger of the two. Almost all of it
 * comes down to two questions: how many bytes must be read to produce one
 * token, and how many requests are waiting while that happens.
 */

type Tab = 'smaller' | 'faster' | 'cost' | 'load';

const BITS: QuantBits[] = [8, 6, 4, 3, 2];

export default function DeployLab() {
  const trainer = useModel((s) => s.trainer);
  const origin = useModel((s) => s.origin);
  const publish = useModel((s) => s.publish);
  const touch = useModel((s) => s.touch);

  const [tab, setTab] = useState<Tab>('smaller');
  const [busy, setBusy] = useState(false);

  /* quantisation */
  const [bits, setBits] = useState<QuantBits>(8);
  const [scheme, setScheme] = useState<QuantScheme>('row');
  const [quant, setQuant] = useState<QuantResult | null>(null);
  const [snapshot, setSnapshot] = useState<Float64Array | null>(null);
  const [quality, setQuality] = useState<{ before: number; after: number } | null>(null);

  /* live timing */
  const [promptLen, setPromptLen] = useState(8);
  const [genLen, setGenLen] = useState(8);
  const [timing, setTiming] = useState<{ cached: Timing; uncached: Timing } | null>(null);

  /* cost model */
  const [gpuId, setGpuId] = useState('h100');
  const [targetId, setTargetId] = useState('llama8b');
  const [contextLen, setContextLen] = useState(2048);
  const [tokensPerRequest, setTokensPerRequest] = useState(400);

  /* load */
  /* Load is held as a fraction of capacity rather than an absolute rate,
     because capacity moves with every control on the cost page and a stored
     rate would silently drift out of range. */
  const [loadFraction, setLoadFraction] = useState(0.6);
  const [p99Target, setP99Target] = useState(20);

  const gpu = useMemo(() => GPU_PROFILES.find((g) => g.id === gpuId) ?? GPU_PROFILES[2], [gpuId]);

  const loadStarter = useCallback(() => {
    setBusy(true);
    setTimeout(() => {
      const t = buildLMTrainer('stories', 'small');
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

  const applyQuant = () => {
    if (!trainer) return;
    setBusy(true);
    setTimeout(() => {
      const snap = snapshot ?? trainer.model.flatParams().slice();
      // Always measure from the original weights, so switching bit widths
      // compares against the same starting point rather than compounding.
      trainer.model.loadFlat(snap);

      const cases = casesFromIds(trainer.tok, trainer.valIds, 'heldout', 12, 8, 3);
      const cfg = { metric: 'quality' as const, topK: 5 };
      const before = runEval(modelPredictor(trainer.model), trainer.tok, cases, cfg, DEFAULT_GEN).mean;

      const r = quantiseModel(trainer.model, bits, scheme);
      const after = runEval(modelPredictor(trainer.model), trainer.tok, cases, cfg, DEFAULT_GEN).mean;

      setSnapshot(snap);
      setQuant(r);
      setQuality({ before, after });
      touch();
      setBusy(false);
    }, 20);
  };

  const restore = () => {
    if (!trainer || !snapshot) return;
    trainer.model.loadFlat(snapshot);
    setSnapshot(null);
    setQuant(null);
    setQuality(null);
    touch();
  };

  const measure = () => {
    if (!trainer) return;
    setBusy(true);
    setTimeout(() => {
      const block = trainer.model.cfg.blockSize;
      const prompt = trainer.valIds.slice(0, Math.min(promptLen, block - 2));
      const n = Math.min(genLen, block - prompt.length - 1);
      setTiming({
        cached: timeCached(trainer.model, prompt, n),
        uncached: timeUncached(trainer.model, prompt, n),
      });
      setBusy(false);
    }, 20);
  };

  if (!trainer) {
    return (
      <div className="mx-auto max-w-[900px] px-6 py-10">
        <Empty>
          <div className="mb-3 text-[13px] font-medium" style={{ color: 'var(--text-2)' }}>
            You need a model to deploy.
          </div>
          <p className="mb-4">
            This step quantises the actual weights and times the actual forward pass, so it needs a real
            model. Build one in <Link to="/studio" style={{ color: 'var(--accent)' }}>Build your own</Link>,
            or start from one trained here.
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
  const bytesPerWeight = quant ? bits / 8 : 4;

  /* A 53-thousand-parameter model costs nothing to serve, so every figure on
     the cost page rounds to zero and teaches nothing. The arithmetic is driven
     entirely by parameter count and cache shape, so the same formula is
     offered against models whose numbers mean something. */
  const target = SERVE_TARGETS.find((t) => t.id === targetId);
  const costParams = target ? target.params : M.paramCount;
  const costCfg = target ? { ...cfg, dModel: target.dModel, nHeads: target.nHeads, nLayers: target.nLayers } : cfg;

  const bound = decodeBound(costParams, bytesPerWeight, contextLen, costCfg, gpu);
  const cost = costOf(bound.tokensPerSecond, tokensPerRequest, gpu, costParams, bytesPerWeight, contextLen, costCfg);
  const arrivalUsed = loadFraction * cost.requestsPerSecond;
  const q = queue(arrivalUsed, cost.requestsPerSecond);
  const replicas = replicasFor(arrivalUsed, cost.requestsPerSecond, p99Target);
  /* The best p99 any single replica can manage, at zero load. A promise
     tighter than this cannot be bought with more machines. */
  const p99Floor = cost.requestsPerSecond > 0 ? Math.log(100) / cost.requestsPerSecond : Infinity;
  // Forty iterations of arithmetic; a hook here would sit after the early
  // return above and change the hook order between renders.
  const curve = cost.requestsPerSecond > 0 && Number.isFinite(cost.requestsPerSecond)
    ? loadCurve(cost.requestsPerSecond)
    : [];
  const batches = batchCurve(bound.tokensPerSecond, 64, costCfg, contextLen, bytesPerWeight, costParams * bytesPerWeight);
  const speedup = timing && timing.cached.perTokenMs > 0 ? timing.uncached.perTokenMs / timing.cached.perTokenMs : 0;

  return (
    <div className="mx-auto max-w-[1400px] px-6 py-6">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <Segmented
          value={tab}
          onChange={setTab}
          options={[
            { id: 'smaller' as Tab, label: '1 · Make it smaller' },
            { id: 'faster' as Tab, label: '2 · Make it fast' },
            { id: 'cost' as Tab, label: '3 · What it costs' },
            { id: 'load' as Tab, label: '4 · Under load' },
          ]}
        />
        <div className="flex items-center gap-2">
          <Badge>{fmtInt(M.paramCount)} params</Badge>
          {quant && <Badge tone="warn">quantised to {quant.bits} bits</Badge>}
        </div>
      </div>

      {/* ------------------------------------------- 1. make it smaller -- */}
      {tab === 'smaller' && (
        <div className="grid gap-5 lg:grid-cols-[1fr_360px]">
          <div className="grid content-start gap-5">
            <Panel title="Why size is speed" subtitle="The one fact that explains most of serving.">
              <Depth
                plain={
                  <>
                    <p className="mb-2">
                      Producing one token means reading every single weight in the model out of memory.
                      Not most of them, not the relevant ones &mdash; all of them, once, for every token.
                      The arithmetic on top of that is almost free by comparison.
                    </p>
                    <p className="mb-2">
                      So how fast a model writes is set by how many bytes it occupies, and storing each
                      weight in eight bits instead of thirty-two makes the model roughly four times
                      faster as well as four times smaller. No other change available to you is that
                      cheap.
                    </p>
                    <p>
                      The catch is that a weight stored in fewer bits is a weight stored less exactly.
                      Below runs the real thing: the weights are snapped onto a coarse grid and the
                      damage is measured on held-out cases using the same harness as{' '}
                      <Link to="/evals" style={{ color: 'var(--accent)' }}>step 10</Link>.
                    </p>
                  </>
                }
                math={
                  <>
                    <p className="mb-2">
                      Decode is memory bound: arithmetic intensity is about{' '}
                      <span className="mono">2</span> operations per parameter byte-pair, far below the
                      ratio at which a modern accelerator becomes compute bound.
                    </p>
                    <p>
                      Symmetric linear quantisation:{' '}
                      <span className="mono">s = max|w| / (2^(b-1) - 1)</span>, then{' '}
                      <span className="mono">w' = s &middot; round(w / s)</span>, clamped.
                    </p>
                  </>
                }
                code={
                  <Code>{`const q = 2 ** (bits - 1) - 1;
const scale = peak / q;
const step = clamp(Math.round(w / scale), -q, q);
w = step * scale;   // the weight really is on the grid now`}</Code>
                }
              />
            </Panel>

            <Panel title="Quantise it">
              <div className="grid gap-3">
                <div>
                  <div className="mb-1.5 flex items-center gap-1.5">
                    <span className="text-[11px] font-medium uppercase tracking-[0.07em]" style={{ color: 'var(--text-3)' }}>
                      Bits per weight
                    </span>
                  </div>
                  <Slider
                    label="Bits"
                    value={BITS.indexOf(bits)}
                    min={0}
                    max={BITS.length - 1}
                    step={1}
                    onChange={(i) => setBits(BITS[i])}
                    format={() => `${bits} bits`}
                    info={V.deployBits}
                  />
                </div>
                <div className="grid gap-2">
                  {(Object.keys(SCHEME_INFO) as QuantScheme[]).map((id) => {
                    const on = scheme === id;
                    return (
                      <button
                        key={id}
                        onClick={() => setScheme(id)}
                        className="rounded-lg p-3 text-left transition-colors"
                        style={{
                          background: on ? 'var(--panel-2)' : 'transparent',
                          border: `1px solid ${on ? 'var(--accent)' : 'var(--border)'}`,
                          cursor: 'pointer',
                        }}
                      >
                        <div className="text-[12.5px] font-medium" style={{ color: on ? 'var(--accent)' : 'var(--text)' }}>
                          {SCHEME_INFO[id].label}
                        </div>
                        <p className="mt-1 text-[11.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                          {SCHEME_INFO[id].blurb}
                        </p>
                      </button>
                    );
                  })}
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <Btn variant="primary" onClick={applyQuant} disabled={busy}>
                    {busy ? 'Measuring...' : 'Quantise and score it'}
                  </Btn>
                  {snapshot && (
                    <Btn onClick={restore} disabled={busy}>
                      Put the full-precision weights back
                    </Btn>
                  )}
                </div>
              </div>
            </Panel>

            {quant && quality && (
              <Panel title="What it cost" subtitle="Size measured exactly; quality measured on held-out cases.">
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <Stat label="At 32 bits" value={prettyBytes(quant.baseBytes)} />
                  <Stat label={`At ${quant.bits} bits`} value={prettyBytes(quant.bytes)} tone="ok" />
                  <Stat
                    label="Smaller by"
                    value={`${fmt(quant.baseBytes / quant.bytes, 2)}x`}
                    tone="ok"
                    sub="and that much faster to read"
                  />
                  <Stat label="Weights left at zero" value={fmtPct(quant.zeroed / quant.total, 1)} />
                </div>
                <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3">
                  <Stat label="Quality, full precision" value={fmtPct(quality.before, 1)} />
                  <Stat
                    label={`Quality at ${quant.bits} bits`}
                    value={fmtPct(quality.after, 1)}
                    tone={quality.after < quality.before * 0.9 ? 'err' : quality.after < quality.before * 0.98 ? 'warn' : 'ok'}
                  />
                  <Stat
                    label="Lost"
                    value={fmtPct(Math.max(0, (quality.before - quality.after) / Math.max(1e-9, quality.before)), 1)}
                    tone={quality.after < quality.before * 0.9 ? 'err' : undefined}
                  />
                </div>
                <div className="mt-4">
                  {quality.after >= quality.before * 0.98 ? (
                    <Callout tone="insight">
                      Essentially free. The model is {fmt(quant.baseBytes / quant.bytes, 1)} times smaller
                      and behaves the same. This is why nobody serves a model at full precision, and why
                      eight-bit weights are the normal case rather than a compromise.
                    </Callout>
                  ) : quality.after >= quality.before * 0.9 ? (
                    <Callout tone="warn">
                      A real but modest loss for a large saving. Whether this trade is worth taking is a
                      product decision, and it is exactly the kind of decision the bar you committed to in{' '}
                      <Link to="/evals" style={{ color: 'var(--accent)' }}>step 10</Link> exists to
                      settle. Without that bar, the temptation is to accept whatever you happen to get.
                    </Callout>
                  ) : (
                    <Callout tone="err">
                      Past the cliff. The model has lost a serious amount of what it knew. Every model has
                      a bit width where this happens, it is not always where you expect, and the only way
                      to find it is to measure &mdash; which is what this panel just did.
                    </Callout>
                  )}
                </div>
              </Panel>
            )}

            {quant && (
              <Panel title="Which tensors suffered" subtitle="Sorted worst first. Outliers are the reason.">
                <div className="grid gap-1.5">
                  {quant.perTensor.slice(0, 10).map((t) => (
                    <div key={t.name} className="flex items-center gap-3">
                      <span className="mono w-32 shrink-0 truncate text-[11px]" style={{ color: 'var(--text-2)' }}>
                        {t.name}
                      </span>
                      <span className="h-1.5 flex-1 overflow-hidden rounded-full" style={{ background: 'var(--panel-3)' }}>
                        <span
                          className="block h-full rounded-full"
                          style={{
                            width: `${Math.min(100, t.relError * 100 * 4)}%`,
                            background: t.relError > 0.15 ? 'var(--err)' : 'var(--accent)',
                          }}
                        />
                      </span>
                      <span className="mono tnum w-14 shrink-0 text-right text-[10.5px]" style={{ color: 'var(--text-3)' }}>
                        {fmtPct(t.relError, 1)}
                      </span>
                      <span className="mono tnum w-16 shrink-0 text-right text-[10.5px]" style={{ color: 'var(--text-3)' }} title="peak divided by typical size">
                        {fmt(t.outlierRatio, 1)}x peak
                      </span>
                    </div>
                  ))}
                </div>
                <p className="mt-3 text-[11.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                  The rightmost column is how far the largest weight in that tensor sticks out above the
                  typical one. A high number there is why a tensor quantises badly: the grid has to stretch
                  to reach the outlier, and every other weight in it gets a coarser step as a result.
                  Handling those outliers separately is most of what published quantisation methods do.
                </p>
              </Panel>
            )}
          </div>

          <div className="grid content-start gap-5">
            <Panel title="The same arithmetic at real sizes" subtitle={`At ${bits} bits per weight.`}>
              <div className="grid gap-2.5">
                {scaleUp(bits).map((r) => (
                  <div key={r.label} style={{ borderBottom: '1px solid var(--border)', paddingBottom: 8 }}>
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[12px] font-medium">{r.label}</span>
                      <span className="mono text-[11px]" style={{ color: 'var(--text-3)' }}>
                        {fmtInt(r.params / 1e9 >= 1 ? Math.round(r.params / 1e9) : 0) || '<1'}B
                      </span>
                    </div>
                    <div className="mt-1 flex items-center justify-between text-[11px]" style={{ color: 'var(--text-3)' }}>
                      <span>fp16 {prettyBytes(r.fp16)}</span>
                      <span style={{ color: 'var(--ok)' }}>
                        {bits}-bit {prettyBytes(r.quantised)}
                      </span>
                    </div>
                    <div className="mt-0.5 text-[11px]" style={{ color: r.cardsQuantised < r.cardsFp16 ? 'var(--ok)' : 'var(--text-3)' }}>
                      {r.cardsFp16} card{r.cardsFp16 === 1 ? '' : 's'} &rarr; {r.cardsQuantised} card
                      {r.cardsQuantised === 1 ? '' : 's'} of 80GB
                    </div>
                  </div>
                ))}
              </div>
              <p className="mt-2 text-[11px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                Weights only; the caches are extra. Quantisation is one of the few things that scales
                exactly, so the arithmetic here is the same arithmetic applied to your own model above.
              </p>
            </Panel>

            <Panel title="An honest note">
              <p className="text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                The quality cost above is real: the weights genuinely are restricted to the coarse grid,
                and the score genuinely was re-measured. The speed benefit is not observed here, because
                doing it properly needs integer arithmetic kernels and this engine computes in 64-bit
                floating point throughout. The size and bandwidth figures are calculated from the byte
                counts, which is exactly how a deployment decision is made in practice.
              </p>
            </Panel>
          </div>
        </div>
      )}

      {/* --------------------------------------------- 2. make it fast -- */}
      {tab === 'faster' && (
        <div className="grid gap-5 lg:grid-cols-[1fr_360px]">
          <div className="grid content-start gap-5">
            <Panel title="The cache that makes serving possible">
              <Depth
                plain={
                  <>
                    <p className="mb-2">
                      Without help, producing the next token means running the entire conversation through
                      the network again from the beginning. Token one hundred re-reads all ninety-nine that
                      came before it; token one hundred and one re-reads all one hundred. Producing n
                      tokens costs roughly n squared, and a long conversation becomes impossible.
                    </p>
                    <p className="mb-2">
                      The fix follows from one observation: almost none of that work changes. What each
                      earlier token contributes to attention cannot depend on a token that had not been
                      written yet. So it is kept, and the new token computes only its own contribution.
                      The cost falls from n squared to n.
                    </p>
                    <p>
                      The price is memory. Every layer, every head, every position, held for as long as
                      the conversation lasts. That is what makes long contexts expensive and it is the
                      main thing limiting how many people one card can serve at once.
                    </p>
                  </>
                }
                math={
                  <>
                    <p className="mb-2">
                      Uncached generation of <span className="mono">n</span> tokens costs{' '}
                      <span className="mono">&Theta;(n&sup2;d)</span> in the attention path; cached costs{' '}
                      <span className="mono">&Theta;(nd)</span> plus <span className="mono">&Theta;(n&sup2;)</span>{' '}
                      in score computation alone.
                    </p>
                    <p>
                      Cache size is{' '}
                      <span className="mono">2 &middot; layers &middot; heads &middot; n &middot; d_head</span>{' '}
                      numbers, linear in context and paid per concurrent request.
                    </p>
                  </>
                }
                code={
                  <Code>{`// decode one token: only its own k and v are new
kCache[layer][head][pos] = k;
vCache[layer][head][pos] = v;
// attend against everything already stored
for (let t = 0; t <= pos; t++) score[t] = dot(q, kCache[..][t]);`}</Code>
                }
              />
            </Panel>

            <Panel title="Time it, both ways">
              <div className="grid gap-3 sm:grid-cols-2">
                <Slider
                  label="Prompt tokens"
                  value={promptLen}
                  min={1}
                  max={Math.max(2, cfg.blockSize - 4)}
                  step={1}
                  onChange={setPromptLen}
                  format={(v) => String(v)}
                  info={V.deployPromptLen}
                />
                <Slider
                  label="Tokens to generate"
                  value={genLen}
                  min={1}
                  max={Math.max(2, cfg.blockSize - promptLen - 1)}
                  step={1}
                  onChange={setGenLen}
                  format={(v) => String(v)}
                  info={V.deployGenLen}
                />
              </div>
              <div className="mt-3">
                <Btn variant="primary" onClick={measure} disabled={busy}>
                  {busy ? 'Measuring...' : 'Measure on this machine'}
                </Btn>
              </div>

              {timing && (
                <div className="mt-4 grid gap-4">
                  <div>
                    <div className="mb-1 flex items-center justify-between text-[12px]">
                      <span className="font-medium" style={{ color: 'var(--ok)' }}>
                        With a cache
                      </span>
                      <span className="mono">{fmt(timing.cached.perTokenMs, 2)} ms per token</span>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full" style={{ background: 'var(--panel-3)' }}>
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${Math.max(2, (timing.cached.perTokenMs / Math.max(timing.uncached.perTokenMs, 1e-9)) * 100)}%`,
                          background: 'var(--ok)',
                        }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="mb-1 flex items-center justify-between text-[12px]">
                      <span className="font-medium" style={{ color: 'var(--warn)' }}>
                        Recomputing every time
                      </span>
                      <span className="mono">{fmt(timing.uncached.perTokenMs, 2)} ms per token</span>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full" style={{ background: 'var(--panel-3)' }}>
                      <div className="h-full w-full rounded-full" style={{ background: 'var(--warn)' }} />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                    <Stat label="Speed-up" value={`${fmt(speedup, 2)}x`} tone={speedup > 1 ? 'ok' : 'warn'} />
                    <Stat label="Prefill" value={`${fmt(timing.cached.prefillMs, 2)} ms`} hint="time before the first word" />
                    <Stat label="Tokens per second" value={fmt(timing.cached.tokensPerSecond, 0)} />
                    <Stat label="Generated" value={timing.cached.generated} />
                  </div>

                  <Callout tone="insight">
                    Both paths produce identical logits, and a test fails if they ever diverge &mdash; an
                    optimisation that changes the answer is not an optimisation. The measured saving is{' '}
                    {fmt(speedup, 1)} times at a context of only {timing.cached.promptTokens + timing.cached.generated}{' '}
                    tokens, and it grows with the length of the conversation, because the work the cache
                    avoids is the work that scales with the square of it. At a few thousand tokens the
                    uncached path is not slow, it is unusable.
                  </Callout>
                </div>
              )}
            </Panel>
          </div>

          <div className="grid content-start gap-5">
            <Panel title="Prefill and decode are different problems">
              <p className="mb-3 text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                Processing the prompt handles every position at once, so it is limited by arithmetic and
                parallelises well. Generating is strictly one token after another, each depending on the
                last, so it is limited by how fast weights can be read from memory and cannot be
                parallelised within a single request at all.
              </p>
              <p className="text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                This is why serving systems report two numbers: time to first token, which is prefill, and
                time between tokens, which is decode. They respond to completely different fixes.
              </p>
            </Panel>

            <Panel title="What the cache costs in memory">
              <div className="grid gap-2">
                <Stat label="This model, full context" value={prettyBytes(cacheBytes(cfg, cfg.blockSize, 8))} sub="at 8 bytes per number here" />
                <Stat label="Llama 3 8B at 4k context" value={prettyBytes(cacheBytes({ ...cfg, dModel: 4096, nHeads: 32, nLayers: 32 }, 4096, 2))} sub="per conversation, at fp16" />
                <Stat label="...at 128k context" value={prettyBytes(cacheBytes({ ...cfg, dModel: 4096, nHeads: 32, nLayers: 32 }, 128000, 2))} tone="warn" />
              </div>
              <p className="mt-3 text-[11.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                Per conversation, not per server. That last figure is why very long contexts are priced the
                way they are: a single request can occupy more memory than the model itself.
              </p>
            </Panel>
          </div>
        </div>
      )}

      {/* ------------------------------------------- 3. what it costs -- */}
      {tab === 'cost' && (
        <div className="grid gap-5 lg:grid-cols-[1fr_360px]">
          <div className="grid content-start gap-5">
            <Panel title="Costing a deployment" subtitle="Published card specifications, your model's real shape.">
              <div className="grid gap-3 sm:grid-cols-2">
                <Field label="Cost this against" info={V.deployTarget}>
                  <Select
                    value={targetId}
                    onChange={setTargetId}
                    options={[
                      { id: 'mine', label: `Your model - ${fmtInt(M.paramCount)} params` },
                      ...SERVE_TARGETS.map((t) => ({ id: t.id, label: t.label })),
                    ]}
                  />
                </Field>
                <Field label="Card" info={V.deployGpu}>
                  <Select
                    value={gpuId}
                    onChange={setGpuId}
                    options={GPU_PROFILES.map((g) => ({ id: g.id, label: `${g.label} - $${g.usdPerHour}/hr` }))}
                  />
                </Field>
                <Slider
                  label="Context to plan for"
                  value={contextLen}
                  min={256}
                  max={32768}
                  step={256}
                  onChange={setContextLen}
                  format={(v) => `${fmtInt(v)} tokens`}
                  info={V.deployContext}
                />
              </div>
              <div className="mt-3">
                <Slider
                  label="Tokens in a typical answer"
                  value={tokensPerRequest}
                  min={20}
                  max={2000}
                  step={20}
                  onChange={setTokensPerRequest}
                  format={(v) => String(v)}
                  info={V.deployTokensPerRequest}
                />
              </div>

              <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                <Stat label="Bytes read per token" value={prettyBytes(bound.bytesPerToken)} />
                <Stat label="Tokens per second" value={fmt(bound.tokensPerSecond, 0)} tone="accent" />
                <Stat label="Requests per second" value={fmt(cost.requestsPerSecond, 2)} />
                <Stat
                  label="Conversations at once"
                  value={Number.isFinite(cost.maxConcurrent) ? fmtInt(cost.maxConcurrent) : '—'}
                  tone={cost.maxConcurrent === 0 ? 'err' : undefined}
                />
              </div>

              {cost.maxConcurrent === 0 && (
                <div className="mt-3">
                  <Callout tone="err">
                    The weights alone do not fit on this card at {bytesPerWeight * 8} bits. You would need
                    a bigger card, more of them, or fewer bits &mdash; which is precisely why quantisation
                    is the first thing anyone reaches for.
                  </Callout>
                </div>
              )}

              <div className="mt-4">
                <Callout tone={bound.memoryBound ? 'insight' : 'warn'}>
                  {bound.memoryBound ? (
                    <>
                      Memory bound, as single-stream decoding almost always is. The card finishes the
                      arithmetic in {fmt(bound.computeSeconds * 1e6, 1)} microseconds, then waits{' '}
                      {fmt(bound.memorySeconds * 1e6, 1)} microseconds for the bytes to arrive &mdash;{' '}
                      {fmt(bound.memorySeconds / Math.max(1e-12, bound.computeSeconds), 0)} times longer
                      doing nothing than working. Buying a card with more teraflops would change nothing
                      here; buying one with more bandwidth would help in direct proportion.
                    </>
                  ) : (
                    <>
                      Compute bound at this shape, which is unusual for decoding and normal for prefill.
                      More teraflops would help.
                    </>
                  )}
                </Callout>
              </div>
            </Panel>

            <Panel title="The bill">
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                <Stat label="Card, per hour" value={money(cost.usdPerHour)} />
                <Stat label="Per request" value={money(cost.usdPerRequest)} />
                <Stat label="Per million tokens" value={money(cost.usdPerMillionTokens)} tone="accent" />
              </div>
              <div className="mt-4">
                <div className="mb-2 text-[11px] font-medium uppercase tracking-[0.07em]" style={{ color: 'var(--text-3)' }}>
                  For comparison, published output prices
                </div>
                <div className="grid gap-1.5">
                  {MARKET_PRICES.map((m) => (
                    <div key={m.label} className="flex items-center justify-between text-[12px]">
                      <span style={{ color: 'var(--text-2)' }}>{m.label}</span>
                      <span className="mono" style={{ color: 'var(--text-3)' }}>
                        ${fmt(m.usdPerMTokOut, 2)} / M
                      </span>
                    </div>
                  ))}
                </div>
                <p className="mt-3 text-[11.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                  Your figure is raw hardware at a single stream, with the card completely dedicated to one
                  request. A real provider batches heavily, which brings the number down a great deal, and
                  then adds margin, redundancy, idle capacity and the cost of everything that is not the
                  card. Comparing the two tells you roughly how much of a hosted price is compute and how
                  much is everything else.
                </p>
              </div>
            </Panel>

            <Panel title="Batching" subtitle="The one lever that changes the economics.">
              <div className="grid gap-1.5">
                {batches.map((b) => (
                  <div key={b.size} className="flex items-center gap-3">
                    <span className="mono w-10 shrink-0 text-[11px]" style={{ color: 'var(--text-3)' }}>
                      x{b.size}
                    </span>
                    <span className="h-2 flex-1 overflow-hidden rounded-full" style={{ background: 'var(--panel-3)' }}>
                      <span
                        className="block h-full rounded-full"
                        style={{
                          width: `${Math.min(100, (b.tokensPerSecond / Math.max(1e-9, batches[batches.length - 1].tokensPerSecond)) * 100)}%`,
                          background: 'var(--accent)',
                        }}
                      />
                    </span>
                    <span className="mono tnum w-20 shrink-0 text-right text-[10.5px]" style={{ color: 'var(--text-2)' }}>
                      {fmt(b.tokensPerSecond, 0)} tok/s
                    </span>
                    <span className="mono tnum w-16 shrink-0 text-right text-[10.5px]" style={{ color: 'var(--text-3)' }}>
                      {fmt(b.perRequestMs, 0)} ms
                    </span>
                  </div>
                ))}
              </div>
              <p className="mt-3 text-[11.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                The weights are read once for the whole batch, so throughput climbs steeply at first and
                then flattens once each request's own cache costs more to read than the shared weights do.
                The third column is what one user waits. This is the entire trade: an interactive product
                sits near the top of the list, a bulk pipeline near the bottom.
              </p>
            </Panel>
          </div>

          <div className="grid content-start gap-5">
            <Panel title="What is being costed">
              <div className="grid gap-2">
                <Stat label="Model" value={target ? target.label : (origin?.label ?? 'yours')} />
                <Stat label="Parameters" value={fmtInt(costParams)} />
                <Stat label="Weight precision" value={quant ? `${quant.bits} bits` : '32 bits'} tone={quant ? 'ok' : 'warn'} />
                <Stat label="Weights occupy" value={prettyBytes(projectSize(costParams, bytesPerWeight * 8, M.params.length))} />
                <Stat label="Cache per conversation" value={prettyBytes(cacheBytes(costCfg, contextLen, bytesPerWeight))} />
              </div>
              {!quant && (
                <div className="mt-3">
                  <Callout tone="warn">
                    Costing at full precision, which nobody does. Quantise it in step one and every number
                    on this page improves at once.
                  </Callout>
                </div>
              )}
            </Panel>

            <Panel title="Where the figures come from">
              <p className="text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                Parameter counts and cache sizes are computed from your model. Memory bandwidth, card
                memory and hourly price are published specifications for the card you selected. The token
                rate is bytes divided by bandwidth, which is the standard way this estimate is made and is
                accurate to within a factor that depends mostly on how well the kernels are written.
                Nothing on this page was measured on your machine &mdash; step two is the part that was.
              </p>
            </Panel>
          </div>
        </div>
      )}

      {/* ---------------------------------------------- 4. under load -- */}
      {tab === 'load' && (
        <div className="grid gap-5 lg:grid-cols-[1fr_360px]">
          <div className="grid content-start gap-5">
            <Panel title="Queues do not degrade gracefully">
              <Depth
                plain={
                  <>
                    <p className="mb-2">
                      A server that can handle ten requests a second does not slow down gently as you
                      approach ten. At five a second, waiting adds about as much as the work itself. At
                      nine, ten times that. At nine point nine, a hundred times. The curve goes vertical.
                    </p>
                    <p>
                      This is why a service that tested perfectly falls over at launch, and why capacity
                      is planned against the peak rather than the average. Running a box at ninety-five
                      percent utilisation looks efficient on a dashboard and feels broken to everyone
                      using it.
                    </p>
                  </>
                }
                math={
                  <p>
                    For M/M/1 with arrival rate <span className="mono">&lambda;</span> and service rate{' '}
                    <span className="mono">&mu;</span>, response time is exponential with rate{' '}
                    <span className="mono">&mu; - &lambda;</span>, so the mean is{' '}
                    <span className="mono">1/(&mu; - &lambda;)</span> and the 99th percentile is{' '}
                    <span className="mono">ln(100)/(&mu; - &lambda;)</span>. Both diverge as{' '}
                    <span className="mono">&lambda; &rarr; &mu;</span>.
                  </p>
                }
                code={
                  <Code>{`const spare = mu - lambda;
if (spare <= 0) return SATURATED;      // the queue grows for ever
const mean = 1 / spare;
const p99  = Math.log(100) / spare;`}</Code>
                }
              />
            </Panel>

            <Panel title="Your deployment under load">
              <div className="grid gap-3 sm:grid-cols-2">
                <Slider
                  label="Load"
                  value={loadFraction}
                  min={0.05}
                  max={1.5}
                  step={0.01}
                  onChange={setLoadFraction}
                  format={(v) => `${fmt(v * cost.requestsPerSecond, 2)}/s · ${fmtPct(v, 0)} of capacity`}
                  info={V.deployArrival}
                />
                <Slider
                  label="Latency you promise (p99)"
                  value={p99Target}
                  min={0.2}
                  max={Math.max(60, Math.min(600, p99Floor * 4))}
                  step={0.2}
                  onChange={setP99Target}
                  format={(v) => `${fmt(v, 1)} s`}
                  info={V.deployP99}
                />
              </div>

              <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                <Stat label="Capacity" value={`${fmt(cost.requestsPerSecond, 2)}/s`} />
                <Stat
                  label="Fastest p99 possible"
                  value={Number.isFinite(p99Floor) ? `${fmt(p99Floor, 1)} s` : '—'}
                  hint="one replica, no queue at all"
                  tone={p99Target < p99Floor ? 'err' : undefined}
                />
                <Stat
                  label="Utilisation"
                  value={fmtPct(Math.min(2, q.utilisation), 0)}
                  tone={q.utilisation > 0.9 ? 'err' : q.utilisation > 0.7 ? 'warn' : 'ok'}
                />
                <Stat
                  label="Mean latency"
                  value={q.saturated ? 'unbounded' : `${fmt(q.meanSeconds, 2)} s`}
                  tone={q.saturated ? 'err' : undefined}
                />
                <Stat
                  label="p99 latency"
                  value={q.saturated ? 'unbounded' : `${fmt(q.p99Seconds, 2)} s`}
                  tone={q.saturated ? 'err' : q.p99Seconds > p99Target ? 'warn' : 'ok'}
                />
              </div>

              <div className="mt-4">
                {q.saturated ? (
                  <Callout tone="err" title="Saturated">
                    Requests are arriving at least as fast as they can be served, so the queue grows
                    without limit and latency has no finite value at all. Nothing recovers from this
                    except fewer requests or more servers. In production this is the state where a
                    dashboard shows a healthy server and every user sees a timeout.
                  </Callout>
                ) : q.p99Seconds > p99Target ? (
                  <Callout tone="warn" title="Over your promise">
                    One replica gives a p99 of {fmt(q.p99Seconds, 2)} seconds against a promise of{' '}
                    {fmt(p99Target, 1)}. You need{' '}
                    <strong>{Number.isFinite(replicas) ? fmtInt(replicas) : 'more than any number of'}</strong>{' '}
                    replicas to hold it, which costs {money(replicas * cost.usdPerHour)} an hour. Most of that is idle capacity bought
                    purely to absorb bursts, and that is not waste &mdash; it is what the promise costs.
                  </Callout>
                ) : (
                  <Callout tone="insight" title="Comfortable">
                    One replica holds your p99 promise at this load, with{' '}
                    {fmtPct(1 - Math.min(1, q.utilisation), 0)} of the capacity spare. Push the arrival
                    rate up and watch how little headroom it takes before that stops being true.
                  </Callout>
                )}
              </div>
            </Panel>

            <Panel title="The cliff" subtitle="p99 latency against utilisation, for this deployment.">
              <LineChart
                series={[
                  {
                    id: 'p99',
                    label: 'p99 seconds',
                    color: 'var(--accent)',
                    points: curve,
                  },
                ]}
                height={200}
                yLabel="p99 seconds"
                xLabel="utilisation"
                marker={Math.min(0.99, q.utilisation)}
              />
              <p className="mt-2 text-[11.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                The marker is where you are now. Everything to the left is comfortable and everything to
                the right is a service that works until precisely the moment it does not. Note that the
                x-axis never reaches one: at full utilisation the wait is infinite.
              </p>
            </Panel>
          </div>

          <div className="grid content-start gap-5">
            <Panel title="What it takes to keep the promise">
              <div className="grid gap-2">
                <Stat
                  label="Replicas needed"
                  value={Number.isFinite(replicas) ? fmtInt(replicas) : 'impossible'}
                  tone={!Number.isFinite(replicas) ? 'err' : replicas > 4 ? 'warn' : 'ok'}
                />
                <Stat label="Cost per hour" value={money(replicas * cost.usdPerHour)} />
                <Stat label="Cost per month" value={money(replicas * cost.usdPerHour * 730)} tone="accent" />
                {!Number.isFinite(replicas) && (
                  <Callout tone="err">
                    No number of replicas can meet this promise. A single request already takes longer
                    than {fmt(p99Target, 1)} seconds at the 99th percentile even with nothing else in the
                    queue, and adding servers shortens the queue rather than the work. The answer is a
                    faster model, fewer tokens per answer, or a looser promise.
                  </Callout>
                )}
                <Stat
                  label="Utilisation per replica"
                  value={Number.isFinite(replicas) ? fmtPct(Math.min(1, arrivalUsed / replicas / Math.max(1e-9, cost.requestsPerSecond)), 0) : '—'}
                />
              </div>
              <p className="mt-3 text-[11.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                Monthly figure at 730 hours, hardware only. It ignores storage, networking, load balancing,
                the standby capacity you keep for a region failure, and the engineers who look after all of
                it.
              </p>
            </Panel>

            <Panel title="The whole lifecycle, from here">
              <p className="text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                You have now been through the sequence a company actually follows: decide what success
                means and how it will be measured, get and shape the data, choose an architecture against a
                budget, train it, evaluate it honestly against baselines, constrain it, and serve it within
                a latency and cost envelope. The one stage still missing from this walkthrough is what happens
                after launch &mdash; watching a live model, collecting what users do with it, noticing when
                the world has moved underneath it, and deciding when to go round again.
              </p>
            </Panel>
          </div>
        </div>
      )}
    </div>
  );
}
