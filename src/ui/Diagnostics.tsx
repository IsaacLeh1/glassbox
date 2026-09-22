import { useMemo } from 'react';
import {
  STAGES,
  analyseSample,
  diagnose,
  recentSlope,
  summariseRun,
  type RunFacts,
  type Severity,
  type Stage,
} from '../engine/diagnostics';
import { Badge, Callout, Depth, Empty, Label, Panel, Stat, fmt, fmtInt } from './kit';
import { BarMeter } from './viz';

const ORDER: Stage[] = ['noise', 'characters', 'words', 'grammar', 'structure'];

const SEV: Record<Severity, { color: string; label: string }> = {
  good: { color: 'var(--ok)', label: 'good' },
  info: { color: 'var(--accent)', label: 'note' },
  warn: { color: 'var(--warn)', label: 'watch' },
  bad: { color: 'var(--err)', label: 'problem' },
};

/**
 * Explains a training run as it happens: which stage of learning the model is
 * in, why the sampled text looks the way it does, and what would specifically
 * improve it. Everything shown is derived from the run's own numbers.
 */
export default function Diagnostics({
  facts,
  sample,
  corpus,
}: {
  facts: RunFacts;
  sample: string | null;
  corpus: string;
}) {
  const findings = useMemo(() => diagnose(facts, sample, corpus), [facts, sample, corpus]);
  const summary = useMemo(() => summariseRun(facts, sample, corpus), [facts, sample, corpus]);
  const analysis = useMemo(() => analyseSample(sample ?? '', corpus), [sample, corpus]);
  // The trainer pushes into the same history array, so its identity never
  // changes. Key the memo on its contents instead.
  const slope = useMemo(
    () => recentSlope(facts.history),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [facts.history, facts.history.length, facts.history[facts.history.length - 1]?.loss],
  );

  if (facts.history.length === 0) {
    return (
      <Panel title="What is happening" subtitle="Analysis appears once training starts">
        <Empty>
          Start the run. This panel will explain what the model is learning, why the samples look the way they
          do, and what to change.
        </Empty>
      </Panel>
    );
  }

  const last = facts.history[facts.history.length - 1];
  const stageIdx = ORDER.indexOf(summary.stage);

  return (
    <div className="space-y-4">
      <Panel
        title="What is happening right now"
        subtitle="Read from this run, not from a script"
        right={<Badge tone="accent">{STAGES[summary.stage].label}</Badge>}
      >
        <p className="text-[13px] leading-relaxed" style={{ color: 'var(--text)' }}>
          {summary.headline}
        </p>

        {/* ------------------------------------------------ stage ladder */}
        <div className="mt-4 flex gap-1">
          {ORDER.map((s, i) => (
            <div key={s} className="flex-1" title={STAGES[s].what}>
              <div
                className="h-1.5 rounded-full transition-colors"
                style={{ background: i <= stageIdx ? 'var(--accent)' : 'var(--panel-3)' }}
              />
              <div
                className="mt-1 truncate text-[9.5px]"
                style={{ color: i === stageIdx ? 'var(--accent)' : 'var(--text-3)' }}
              >
                {STAGES[s].label}
              </div>
            </div>
          ))}
        </div>

        <div className="mt-4 space-y-2">
          <p className="text-[12.5px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
            {STAGES[summary.stage].what}
          </p>
          <p className="text-[12.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
            <span style={{ color: 'var(--text-2)' }}>What comes next: </span>
            {STAGES[summary.stage].next}
          </p>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="loss" value={fmt(last.loss, 4)} tone="accent" />
          <Stat
            label="vs random guessing"
            value={`${(summary.progress * 100).toFixed(0)}%`}
            sub={`random = ${Math.log(Math.max(2, facts.vocab)).toFixed(2)}`}
          />
          <Stat
            label="trend"
            value={Math.abs(slope) < 1e-5 ? 'flat' : slope < 0 ? 'falling' : 'rising'}
            tone={Math.abs(slope) < 1e-5 ? 'warn' : slope < 0 ? 'ok' : 'err'}
            sub={`${slope.toExponential(1)} / step`}
          />
          <Stat label="tokens seen" value={fmtInt(last.tokensSeen)} sub={`of ${fmtInt(facts.paramCount * 20)} recommended`} />
        </div>
      </Panel>

      {/* ---------------------------------------------------- the sample */}

      <Panel
        title="That text at the bottom"
        subtitle="Where it comes from and why it reads the way it does"
      >
        <Depth
          plain={
            <>
              Every fifty steps the run pauses and asks the model to write something. It is given a short
              starting word and then predicts one token, appends it, predicts the next from the result, and
              repeats. Nothing is cherry-picked and nothing is cleaned up: that text is the honest state of the
              model at that moment, which is why the early ones look like nonsense.
            </>
          }
          math={
            <>
              Each sample is an autoregressive rollout: <span className="mono">x₍t+1₎ ~ p(· | x₍≤t₎)</span> with
              temperature 0.85 and top-k 20. The randomness is deliberate — always taking the single most
              likely token produces a loop almost immediately, because the model&apos;s favourite continuation
              of its own favourite continuation tends to cycle.
            </>
          }
          code={
            <span className="mono text-[11px]">
              for (i&lt;n) &#123; probs = model.predictNext(ids); ids.push(sampleToken(probs, cfg, rand)); &#125;
            </span>
          }
        />

        {sample && analysis.words > 0 ? (
          <>
            <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Stat
                label="real words"
                value={`${(analysis.realWordRatio * 100).toFixed(0)}%`}
                tone={analysis.realWordRatio > 0.7 ? 'ok' : analysis.realWordRatio > 0.3 ? 'warn' : 'err'}
                hint="Chunks that appear in the training text."
              />
              <Stat
                label="repetition"
                value={`${(analysis.repetitionRatio * 100).toFixed(0)}%`}
                tone={analysis.looping ? 'err' : 'ok'}
                hint="How much of the sample is one phrase cycling."
              />
              <Stat
                label="word length"
                value={fmt(analysis.avgWordLength, 1)}
                hint="Average characters per word in the sample."
              />
              <Stat
                label="spacing"
                value={`${(analysis.spaceRatio * 100).toFixed(0)}%`}
                sub={`corpus ${(analysis.targetSpaceRatio * 100).toFixed(0)}%`}
              />
            </div>

            <div className="mt-3">
              <Label>how close the sample is to the training text, per measure</Label>
              <div className="mt-2 space-y-2">
                <BarMeter
                  value={analysis.realWordRatio}
                  max={1}
                  color="var(--ok)"
                  height={6}
                  label={<span>words the model actually knows</span>}
                />
                <BarMeter
                  value={1 - Math.min(1, Math.abs(analysis.spaceRatio - analysis.targetSpaceRatio) / 0.15)}
                  max={1}
                  color="var(--accent)"
                  height={6}
                  label={<span>word length matching the corpus</span>}
                />
                <BarMeter
                  value={1 - Math.min(1, analysis.repetitionRatio)}
                  max={1}
                  color="var(--pos)"
                  height={6}
                  label={<span>variety, rather than looping</span>}
                />
              </div>
            </div>
          </>
        ) : (
          <p className="mt-3 text-[12px]" style={{ color: 'var(--text-3)' }}>
            No sample yet. One appears after the first fifty steps.
          </p>
        )}
      </Panel>

      {/* ------------------------------------------------------ findings */}

      <Panel
        title="Analysis"
        subtitle={`${findings.length} observation${findings.length === 1 ? '' : 's'} about this specific run`}
      >
        {findings.length === 0 ? (
          <Empty>Not enough data yet.</Empty>
        ) : (
          <div className="space-y-2">
            {findings.map((f) => (
              <div
                key={f.id}
                className="rounded-lg p-3"
                style={{ background: `color-mix(in srgb, ${SEV[f.severity].color} 6%, transparent)` }}
              >
                <div className="mb-1 flex items-center gap-2">
                  <span
                    className="mono text-[9.5px] font-semibold uppercase tracking-[0.08em]"
                    style={{ color: SEV[f.severity].color }}
                  >
                    {SEV[f.severity].label}
                  </span>
                  <span className="text-[12.5px] font-semibold" style={{ color: 'var(--text)' }}>
                    {f.title}
                  </span>
                </div>
                <p className="text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                  {f.detail}
                </p>
                {f.fix && (
                  <p className="mt-1.5 text-[12px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                    <span style={{ color: SEV[f.severity].color }}>What to change: </span>
                    {f.fix}
                  </p>
                )}
              </div>
            ))}
          </div>
        )}
      </Panel>

      <Callout tone="insight" title="The honest ceiling">
        A model with {fmtInt(facts.paramCount)} parameters and a {facts.blockSize}-token context window cannot
        hold a paragraph together, no matter how long it trains. It has room to learn spelling, common words
        and short phrases, and that is roughly it. Judging its output by whether it reads like a person is the
        wrong test; the right one is whether the loss keeps falling and whether the held-out loss falls with
        it.
      </Callout>
    </div>
  );
}
