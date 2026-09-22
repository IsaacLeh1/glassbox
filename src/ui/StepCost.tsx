import { useEffect, useRef, useState } from 'react';
import { Transformer, type TransformerConfig } from '../engine/transformer';
import { pipelineStages } from '../engine/shapes';
import { fmtDuration } from '../sim/cluster';
import { Callout, Label, Spinner, Stat } from './kit';

/**
 * What this configuration will actually cost on this machine.
 *
 * Rather than asserting a formula, it times two short forward-and-backward
 * passes and calibrates against the model's exact FLOP breakdown. Attention
 * grows with the square of the sequence length and everything else grows
 * linearly, and both terms are known exactly from the architecture, so the
 * only unknowns are how fast this machine executes each kind of work.
 *
 * Probing at the full context would be accurate but could block the page for
 * seconds every time a slider moves, which is exactly the problem this panel
 * exists to warn about.
 */

const PROBE_A = 48;
const PROBE_B = 96;

/** Stages whose cost grows with the square of the sequence length. */
const QUADRATIC = new Set(['scores', 'softmax', 'attnv']);

/**
 * Split the model's work into the part that grows linearly with context and
 * the part that grows with its square, using the exact stage-by-stage FLOP
 * counts rather than trying to infer the shape from timings.
 */
function workSplit(cfg: TransformerConfig, T: number): { linear: number; quad: number } {
  const stages = pipelineStages(
    { vocab: cfg.vocab, dModel: cfg.dModel, nHeads: cfg.nHeads, nLayers: cfg.nLayers, dFF: cfg.dFF, ctx: cfg.blockSize },
    T,
  );
  let linear = 0;
  let quad = 0;
  for (const s of stages) {
    const total = s.flops * (s.perLayer ? cfg.nLayers : 1);
    if (QUADRATIC.has(s.id)) quad += total;
    else linear += total;
  }
  return { linear, quad };
}

interface Fit {
  /** Milliseconds for one sequence, forward plus backward. */
  perSequenceMs: number;
  perStepMs: number;
  totalSeconds: number;
  linearMs: number;
  quadraticMs: number;
}

function timeOne(cfg: TransformerConfig, T: number): number {
  const m = new Transformer({ ...cfg, blockSize: T }, 1);
  const toks = Array.from({ length: T }, (_, i) => i % Math.max(1, cfg.vocab));
  const tgts = toks.map((t) => (t + 1) % Math.max(1, cfg.vocab));
  // One untimed pass so allocation and JIT warm-up are not counted.
  const warm = m.forward(toks);
  m.backward(warm, tgts);
  const t0 = performance.now();
  const fw = m.forward(toks);
  m.backward(fw, tgts);
  return performance.now() - t0;
}

function measure(cfg: TransformerConfig, batchSeqs: number, steps: number): Fit {
  const c1 = timeOne(cfg, PROBE_A);
  const c2 = timeOne(cfg, PROBE_B);
  const w1 = workSplit(cfg, PROBE_A);
  const w2 = workSplit(cfg, PROBE_B);

  // Two timings, two known work terms: solve for how many milliseconds this
  // machine spends per linear FLOP and per attention FLOP. Fitting a blind
  // quadratic to the timings instead collapses to zero, because at small
  // context lengths attention is too small a share to be visible above noise.
  const det = w1.linear * w2.quad - w2.linear * w1.quad;
  let kLin = 0;
  let kQuad = 0;
  if (Math.abs(det) > 1e-9) {
    kLin = (c1 * w2.quad - c2 * w1.quad) / det;
    kQuad = (w1.linear * c2 - w2.linear * c1) / det;
  }
  if (!Number.isFinite(kLin) || !Number.isFinite(kQuad) || kLin <= 0 || kQuad < 0) {
    // Timing noise produced a nonsensical fit. Fall back to assuming every
    // FLOP costs the same, which still captures the quadratic growth exactly.
    const rate = c2 / Math.max(1, w2.linear + w2.quad);
    kLin = rate;
    kQuad = rate;
  }

  const T = cfg.blockSize;
  const w = workSplit(cfg, T);
  const linearMs = kLin * w.linear;
  const quadraticMs = kQuad * w.quad;
  const perSequenceMs = Math.max(0.01, linearMs + quadraticMs);
  const perStepMs = perSequenceMs * batchSeqs;
  return { perSequenceMs, perStepMs, totalSeconds: (perStepMs * steps) / 1000, linearMs, quadraticMs };
}

export default function StepCost({
  cfg,
  batchSeqs,
  steps,
  frameBudgetMs,
}: {
  cfg: TransformerConfig;
  batchSeqs: number;
  steps: number;
  frameBudgetMs: number;
}) {
  const [fit, setFit] = useState<Fit | null>(null);
  const [busy, setBusy] = useState(false);
  const key = `${cfg.vocab}|${cfg.dModel}|${cfg.nHeads}|${cfg.nLayers}|${cfg.dFF}|${cfg.blockSize}|${batchSeqs}|${steps}`;
  const lastKey = useRef('');

  useEffect(() => {
    if (lastKey.current === key) return;
    setBusy(true);
    // Debounce: sliders fire continuously while dragging.
    const t = setTimeout(() => {
      try {
        setFit(measure(cfg, batchSeqs, steps));
        lastKey.current = key;
      } catch {
        setFit(null);
      } finally {
        setBusy(false);
      }
    }, 450);
    return () => clearTimeout(t);
  }, [key, cfg, batchSeqs, steps]);

  if (!fit) {
    return (
      <div className="flex items-center gap-2 text-[11.5px]" style={{ color: 'var(--text-3)' }}>
        <Spinner /> timing this configuration on your machine…
      </div>
    );
  }

  // A single sequence is the smallest unit the training loop can be paused
  // between, so it sets the worst-case pause the interface can suffer.
  const janky = fit.perSequenceMs > frameBudgetMs * 3;
  const quadShare = fit.quadraticMs / Math.max(1e-6, fit.perSequenceMs);

  return (
    <div>
      <div className="mb-2 flex items-center gap-2">
        <Label>measured cost of this configuration</Label>
        {busy && <Spinner size={11} />}
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="per sequence" value={`${fit.perSequenceMs.toFixed(1)} ms`} />
        <Stat
          label="per weight update"
          value={`${fit.perStepMs.toFixed(0)} ms`}
          sub={`${batchSeqs} sequences`}
        />
        <Stat label="whole run" value={fmtDuration(fit.totalSeconds)} tone="accent" sub={`${steps} steps`} />
        <Stat
          label="attention share"
          value={`${(quadShare * 100).toFixed(0)}%`}
          tone={quadShare > 0.6 ? 'warn' : undefined}
          hint="How much of the cost comes from the part that grows with the square of context length."
        />
      </div>

      {janky && (
        <div className="mt-3">
          <Callout tone="warn" title="This will make the page stutter">
            One sequence takes {fit.perSequenceMs.toFixed(0)} ms, and training can only pause between
            sequences, so the interface will freeze for about that long at a time. Shorten the context window,
            narrow the model, or accept a choppy page while it runs. The run itself is unaffected.
          </Callout>
        </div>
      )}

      {quadShare > 0.6 && (
        <div className="mt-3">
          <Callout tone="info" title="Attention now dominates the cost">
            {(quadShare * 100).toFixed(0)}% of the time is going into attention, which grows with the square of
            the context window. Doubling the context from here roughly quadruples that portion. This is the
            same wall that makes long-context models expensive to serve.
          </Callout>
        </div>
      )}
    </div>
  );
}
