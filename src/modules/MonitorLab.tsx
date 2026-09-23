import { useCallback, useMemo, useRef, useState } from 'react';
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
  ProgressBar,
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
import { useFrameLoop } from '../ui/loop';
import { CORPORA } from '../engine/corpus';
import {
  SCENARIO_INFO,
  SIGNAL_INFO,
  buildWindow,
  corpusText,
  evaluateAlert,
  psiBand,
  referenceHistogram,
  retrainCase,
  schedule,
  thresholdSweep,
  type Scenario,
  type SignalId,
  type Window,
} from '../engine/monitor';

/**
 * Module 12: after it ships.
 *
 * Every earlier module had an answer key. This one does not, and that is the
 * whole subject. Production gives you unlabelled traffic and silence, so the
 * job becomes noticing that something has gone wrong using signals that need
 * no ground truth -- and then deciding whether what you noticed is worth
 * money.
 */

type Tab = 'signals' | 'drift' | 'alerts' | 'retrain';

const signalOf = (w: Window, s: SignalId): number =>
  s === 'surprise' ? w.surprise : s === 'psi' ? w.psi : s === 'oov' ? w.oov : w.repetition;

export default function MonitorLab() {
  const trainer = useModel((s) => s.trainer);
  const origin = useModel((s) => s.origin);
  const publish = useModel((s) => s.publish);

  const [tab, setTab] = useState<Tab>('signals');
  const [busy, setBusy] = useState(false);

  /* the experiment */
  const [strangeId, setStrangeId] = useState('weather');
  const [scenario, setScenario] = useState<Scenario>('gradual');
  const [nWindows, setNWindows] = useState(14);
  const [perWindow, setPerWindow] = useState(8);
  const [reveal, setReveal] = useState(false);

  /* the run, streamed in one window at a time like a real dashboard */
  const [windows, setWindows] = useState<Window[]>([]);
  const [running, setRunning] = useState(false);
  const pending = useRef<{ sched: number[]; i: number } | null>(null);

  /* alerting */
  const [signal, setSignal] = useState<SignalId>('surprise');
  const [threshold, setThreshold] = useState(0);

  /* the decision */
  const [requestsPerDay, setRequestsPerDay] = useState(50_000);
  const [valuePerRequest, setValuePerRequest] = useState(0.02);
  const [retrainCost, setRetrainCost] = useState(2_000);

  const loadStarter = useCallback(() => {
    setBusy(true);
    setTimeout(() => {
      const t = buildLMTrainer('stories', 'small');
      for (let i = 0; i < 200; i++) t.singleStep(i % 5 === 0);
      publish(t, {
        label: 'Starter - tiny stories',
        source: 'built in, trained here for 200 steps',
        chars: 0,
        steps: t.step,
        finalLoss: t.latest()?.loss ?? NaN,
        trainedAt: Date.now(),
      });
      setBusy(false);
    }, 30);
  }, [publish]);

  const familiarText = useMemo(() => corpusText('stories'), []);
  const strangeText = useMemo(() => corpusText(strangeId), [strangeId]);
  const reference = useMemo(
    () => (trainer ? referenceHistogram(trainer.tok, familiarText) : null),
    [trainer, familiarText],
  );

  const start = () => {
    if (!trainer) return;
    setWindows([]);
    pending.current = { sched: schedule(scenario, nWindows), i: 0 };
    setRunning(true);
  };

  /* One window per frame. Each one really runs every request through the
     model, so streaming keeps the page alive and looks like the thing it is
     describing. */
  useFrameLoop(running && !!trainer, () => {
    const p = pending.current;
    if (!trainer || !p) return;
    if (p.i >= p.sched.length) {
      setRunning(false);
      return;
    }
    const w = buildWindow(
      trainer.model,
      trainer.tok,
      familiarText,
      strangeText,
      p.sched[p.i],
      p.i,
      perWindow,
      11,
      reference,
    );
    p.i++;
    setWindows((ws) => [...ws, w]);
  });

  const done = windows.length > 0 && !running;
  const alert = useMemo(
    () => (windows.length > 0 ? evaluateAlert(windows, signal, threshold) : null),
    [windows, signal, threshold],
  );
  const sweep = useMemo(
    () => (windows.length > 0 ? thresholdSweep(windows, signal, 30) : []),
    [windows, signal],
  );

  if (!trainer) {
    return (
      <div className="mx-auto max-w-[900px] px-6 py-10">
        <Empty>
          <div className="mb-3 text-[13px] font-medium" style={{ color: 'var(--text-2)' }}>
            You need a deployed model to watch.
          </div>
          <p className="mb-4">
            Every signal here is measured by really running traffic through a real model, so there has to
            be one. Build one in <Link to="/studio" style={{ color: 'var(--accent)' }}>Build your own</Link>,
            or start from one trained on the spot.
          </p>
          <Btn onClick={loadStarter} variant="primary" disabled={busy}>
            {busy ? 'Training a starter model...' : 'Train a starter model here'}
          </Btn>
        </Empty>
      </div>
    );
  }

  const first = windows[0];
  const last = windows[windows.length - 1];
  const decision =
    first && last
      ? retrainCase(last.trueQuality, first.trueQuality, requestsPerDay, valuePerRequest, retrainCost)
      : null;

  const chart = (key: SignalId, colour: string) => ({
    id: key,
    label: SIGNAL_INFO[key].label,
    color: colour,
    points: windows.filter((w) => Number.isFinite(signalOf(w, key))).map((w) => ({ x: w.index, y: signalOf(w, key) })),
  });

  return (
    <div className="mx-auto max-w-[1400px] px-6 py-6">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <Segmented
          value={tab}
          onChange={setTab}
          options={[
            { id: 'signals' as Tab, label: '1 · What you can see' },
            { id: 'drift' as Tab, label: '2 · Watch it drift' },
            { id: 'alerts' as Tab, label: '3 · Set an alert' },
            { id: 'retrain' as Tab, label: '4 · Retrain or not' },
          ]}
        />
        <div className="flex items-center gap-2">
          {running && <Badge tone="accent">watching...</Badge>}
          {windows.length > 0 && <Badge>{windows.length} windows</Badge>}
          {last && <Badge tone={psiBand(last.psi).tone}>{psiBand(last.psi).label}</Badge>}
        </div>
      </div>

      {/* ------------------------------------------ 1. what you can see -- */}
      {tab === 'signals' && (
        <div className="grid gap-5 lg:grid-cols-[1fr_360px]">
          <div className="grid content-start gap-5">
            <Panel title="Nobody tells you when it stops working">
              <Depth
                plain={
                  <>
                    <p className="mb-2">
                      Everything up to now had an answer key. Training had targets to match. Evaluation had
                      a held-out set with the right continuations written down. Both told you, precisely,
                      how well the model was doing.
                    </p>
                    <p className="mb-2">
                      Production has neither. Real users send text nobody has labelled, the model answers,
                      and almost nobody says whether the answer was any good. The handful who do are the
                      furious and the delighted, which is not a sample you can measure anything from.
                    </p>
                    <p>
                      So monitoring is the art of noticing trouble without ground truth. Everything below
                      can be computed from traffic alone. None of it measures quality. All of it moves
                      before quality collapses, and that is the entire reason it is worth watching.
                    </p>
                  </>
                }
                math={
                  <>
                    <p className="mb-2">
                      Without labels you cannot estimate <span className="mono">E[quality]</span>. You can
                      estimate <span className="mono">E[-log p(x)]</span> over incoming text, because the
                      text is its own target, and distances between the input distribution now and at
                      deployment.
                    </p>
                    <p>
                      These are proxies: correlated with quality under the assumption that the model
                      degrades when inputs move away from its training distribution. That assumption is
                      usually right and is not guaranteed.
                    </p>
                  </>
                }
                code={
                  <Code>{`// no targets needed: the text is its own target
const { trace } = model.forward(ids);
const surprise  = model.loss(trace, ids.slice(1));

// and how far the traffic has moved since launch
const drift = psi(referenceHistogram, currentHistogram);`}</Code>
                }
              />
            </Panel>

            <Panel title="The four you can actually compute">
              <div className="grid gap-2">
                {(Object.keys(SIGNAL_INFO) as SignalId[]).map((id) => {
                  const on = signal === id;
                  return (
                    <button
                      key={id}
                      onClick={() => setSignal(id)}
                      className="rounded-lg p-3 text-left transition-colors"
                      style={{
                        background: on ? 'var(--panel-2)' : 'transparent',
                        border: `1px solid ${on ? 'var(--accent)' : 'var(--border)'}`,
                        cursor: 'pointer',
                      }}
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-[12.5px] font-medium" style={{ color: on ? 'var(--accent)' : 'var(--text)' }}>
                          {SIGNAL_INFO[id].label}
                        </span>
                        <Badge tone="ok">no labels needed</Badge>
                      </div>
                      <p className="mt-1 text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                        {SIGNAL_INFO[id].blurb}
                      </p>
                    </button>
                  );
                })}
              </div>
            </Panel>

            <Panel title="And the one you cannot">
              <div className="rounded-lg p-3" style={{ background: 'color-mix(in srgb, var(--err) 7%, transparent)' }}>
                <div className="mb-1 text-[10.5px] font-semibold uppercase tracking-[0.08em]" style={{ color: 'var(--err)' }}>
                  Not available in production
                </div>
                <div className="text-[12.5px] font-medium">Whether the answers were any good</div>
                <p className="mt-1 text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                  This module computes it anyway, from the continuations that really followed each request,
                  and shows it beside the proxies so you can judge how well they track it. A real
                  deployment has no such column. Watching the two move together here is what earns the
                  proxies your trust; then you go back to only having the proxies.
                </p>
              </div>
              <div className="mt-3">
                <Callout tone="warn" title="Why feedback does not fill the gap">
                  Thumbs-up buttons look like the answer and are not. The people who click them are the
                  ones with strong feelings, so the sample is drawn from the tails of the distribution
                  rather than from the middle. A rating that goes from 4.6 to 4.4 might be a real
                  regression, or one annoyed community, or a change to where the button sits on the page.
                  Feedback is a useful alarm and a poor measurement.
                </Callout>
              </div>
            </Panel>
          </div>

          <div className="grid content-start gap-5">
            <Panel title="What is being watched">
              <div className="grid gap-2">
                <Stat label="Model" value={origin?.label ?? 'yours'} />
                <Stat label="Trained on" value="Tiny stories" />
                <Stat label="Parameters" value={fmtInt(trainer.model.paramCount)} />
                <Stat label="Vocabulary" value={fmtInt(trainer.tok.size)} />
              </div>
            </Panel>
            <Panel title="Where this sits">
              <p className="text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                This is the last stage of the lifecycle and the one that loops. What you learn here sends
                you back to <Link to="/evals" style={{ color: 'var(--accent)' }}>the evaluation</Link> to
                check whether the bar still holds, to{' '}
                <Link to="/studio" style={{ color: 'var(--accent)' }}>training</Link> with better data, or
                to <Link to="/comms" style={{ color: 'var(--accent)' }}>teaching</Link> the model something
                it now needs to know. A deployed model is not finished, it is in maintenance.
              </p>
            </Panel>
          </div>
        </div>
      )}

      {/* ---------------------------------------------- 2. watch it drift -- */}
      {tab === 'drift' && (
        <div className="grid gap-5 lg:grid-cols-[1fr_360px]">
          <div className="grid content-start gap-5">
            <Panel title="Run a period of traffic" subtitle="Real text, really processed. Only the mixture is chosen.">
              <div className="grid gap-3 sm:grid-cols-2">
                <Field label="Unfamiliar traffic comes from" info={V.monitorStrange}>
                  <Select
                    value={strangeId}
                    onChange={setStrangeId}
                    options={CORPORA.filter((c) => c.id !== 'stories').map((c) => ({ id: c.id, label: c.label }))}
                  />
                </Field>
                <Field label="What happens" info={V.monitorScenario}>
                  <Select
                    value={scenario}
                    onChange={(v) => setScenario(v as Scenario)}
                    options={(Object.keys(SCENARIO_INFO) as Scenario[]).map((id) => ({
                      id,
                      label: SCENARIO_INFO[id].label,
                    }))}
                  />
                </Field>
              </div>
              <p className="mt-2 text-[11.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                {SCENARIO_INFO[scenario].blurb}
              </p>
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <Slider
                  label="Windows to watch"
                  value={nWindows}
                  min={4}
                  max={30}
                  step={1}
                  onChange={setNWindows}
                  format={(v) => String(v)}
                  info={V.monitorWindows}
                />
                <Slider
                  label="Requests per window"
                  value={perWindow}
                  min={3}
                  max={20}
                  step={1}
                  onChange={setPerWindow}
                  format={(v) => String(v)}
                  info={V.monitorPerWindow}
                />
              </div>
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <Btn variant="primary" onClick={start} disabled={running}>
                  {running ? 'Watching...' : 'Run the period'}
                </Btn>
                {windows.length > 0 && (
                  <Btn onClick={() => setReveal((r) => !r)}>
                    {reveal ? 'Hide the answer key' : 'Reveal the answer key'}
                  </Btn>
                )}
              </div>
              {running && (
                <div className="mt-3">
                  <ProgressBar value={windows.length / Math.max(1, nWindows)} />
                  <div className="mt-1 text-[11px]" style={{ color: 'var(--text-3)' }}>
                    window {windows.length} of {nWindows}
                  </div>
                </div>
              )}
            </Panel>

            {windows.length > 0 && (
              <Panel title="The dashboard" subtitle="Everything here can be computed from traffic alone.">
                <LineChart
                  series={[chart('surprise', 'var(--accent)'), chart('psi', 'var(--warn)')]}
                  height={190}
                  xLabel="window"
                />
                <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <Stat label="Surprise now" value={fmt(last?.surprise ?? NaN, 3)} sub={first ? `was ${fmt(first.surprise, 3)}` : undefined} />
                  <Stat
                    label="Input drift"
                    value={fmt(last?.psi ?? NaN, 3)}
                    tone={last ? psiBand(last.psi).tone : undefined}
                    sub={last ? psiBand(last.psi).label : undefined}
                  />
                  <Stat label="Unreadable characters" value={fmtPct(last?.oov ?? 0, 1)} />
                  <Stat label="Repetition" value={fmtPct(last?.repetition ?? 0, 1)} />
                </div>
              </Panel>
            )}

            {windows.length > 0 && reveal && (
              <Panel title="The answer key" subtitle="Which production would not have.">
                <LineChart
                  series={[
                    {
                      id: 'truth',
                      label: 'true quality',
                      color: 'var(--pos)',
                      points: windows
                        .filter((w) => Number.isFinite(w.trueQuality))
                        .map((w) => ({ x: w.index, y: w.trueQuality })),
                    },
                    {
                      id: 'drift',
                      label: 'unfamiliar share of traffic',
                      color: 'var(--text-3)',
                      dashed: true,
                      points: windows.map((w) => ({ x: w.index, y: w.drift })),
                    },
                  ]}
                  height={190}
                  xLabel="window"
                />
                {first && last && (
                  <div className="mt-4">
                    <Callout tone={last.trueQuality < first.trueQuality * 0.95 ? 'err' : 'insight'}>
                      Quality went from {fmtPct(first.trueQuality, 1)} to {fmtPct(last.trueQuality, 1)}{' '}
                      while surprise went from {fmt(first.surprise, 2)} to {fmt(last.surprise, 2)}. The two
                      moved in opposite directions, which is what makes the proxy usable. Now hide this
                      panel again: the discipline is deciding what to do with only the chart above it.
                    </Callout>
                  </div>
                )}
              </Panel>
            )}

            {windows.length > 0 && (
              <Panel title="Window by window" pad={false}>
                <div className="max-h-[320px] overflow-y-auto">
                  <div
                    className="grid grid-cols-6 gap-2 px-4 py-2 text-[10px] font-semibold uppercase tracking-[0.07em]"
                    style={{ color: 'var(--text-3)', borderBottom: '1px solid var(--border)' }}
                  >
                    <span>#</span>
                    <span>Surprise</span>
                    <span>Drift</span>
                    <span>Unreadable</span>
                    <span>Repetition</span>
                    <span>{reveal ? 'Quality' : 'hidden'}</span>
                  </div>
                  {windows.map((w) => (
                    <div
                      key={w.index}
                      className="mono grid grid-cols-6 gap-2 px-4 py-1.5 text-[11px]"
                      style={{ borderBottom: '1px solid var(--border)', color: 'var(--text-2)' }}
                    >
                      <span style={{ color: 'var(--text-3)' }}>{w.index}</span>
                      <span>{fmt(w.surprise, 3)}</span>
                      <span style={{ color: `var(--${psiBand(w.psi).tone})` }}>{fmt(w.psi, 3)}</span>
                      <span>{fmtPct(w.oov, 1)}</span>
                      <span>{fmtPct(w.repetition, 1)}</span>
                      <span style={{ color: reveal ? 'var(--pos)' : 'var(--text-3)' }}>
                        {reveal ? fmtPct(w.trueQuality, 1) : '—'}
                      </span>
                    </div>
                  ))}
                </div>
              </Panel>
            )}
          </div>

          <div className="grid content-start gap-5">
            <Panel title="What a window is">
              <p className="text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                One row of the dashboard: an hour, or a day, of traffic summarised into a handful of
                numbers. Each of the {perWindow} requests in it is real text, really tokenised, really run
                through your model. Nothing on the chart is interpolated or assumed.
              </p>
              <div className="mt-3">
                <Stat label="Requests processed" value={fmtInt(windows.reduce((a, w) => a + w.requests.length, 0))} />
              </div>
            </Panel>

            {done && (
              <Panel title="Next">
                <p className="mb-3 text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                  You have a period of history. Now put an alert on it and find out how hard it is to
                  choose a threshold that is neither useless nor exhausting.
                </p>
                <Btn variant="primary" size="sm" onClick={() => setTab('alerts')}>
                  Set an alert
                </Btn>
              </Panel>
            )}
          </div>
        </div>
      )}

      {/* ------------------------------------------------ 3. set an alert -- */}
      {tab === 'alerts' && (
        <div className="grid gap-5 lg:grid-cols-[1fr_360px]">
          <div className="grid content-start gap-5">
            {windows.length === 0 ? (
              <Empty>
                <div className="mb-3">No history to alert on yet.</div>
                <p className="mb-4">Run a period of traffic first, then come back and try to catch it.</p>
                <Btn variant="primary" onClick={() => setTab('drift')}>
                  Go and run one
                </Btn>
              </Empty>
            ) : (
              <>
                <Panel title="An alert is a single number, and it is always wrong somewhere">
                  <Depth
                    plain={
                      <p>
                        Set the bar too low and it fires on ordinary variation, somebody gets woken at
                        three in the morning for nothing, and within a month everyone has learned to close
                        the alert without reading it. Set it too high and it never fires, including on the
                        day it should have. There is usually no setting that avoids both, and choosing
                        between them is a judgement about which mistake costs your team more.
                      </p>
                    }
                    math={
                      <p>
                        Each threshold is a point on a tradeoff curve between false positives and false
                        negatives. Sweeping it traces the curve; the choice of operating point is external
                        to the mathematics and depends on the relative cost of the two errors.
                      </p>
                    }
                    code={<Code>{`const alarm = signal(window) >= threshold;   // the entire mechanism`}</Code>}
                  />
                  <div className="mt-3 grid gap-3 sm:grid-cols-2">
                    <Field label="Watch" info={V.monitorSignal}>
                      <Select
                        value={signal}
                        onChange={(v) => setSignal(v as SignalId)}
                        options={(Object.keys(SIGNAL_INFO) as SignalId[]).map((id) => ({
                          id,
                          label: SIGNAL_INFO[id].label,
                        }))}
                      />
                    </Field>
                    <Slider
                      label="Fire above"
                      value={threshold}
                      min={0}
                      max={Math.max(
                        0.01,
                        Math.max(...windows.map((w) => signalOf(w, signal)).filter(Number.isFinite)) * 1.1,
                      )}
                      step={Math.max(
                        0.001,
                        Math.max(...windows.map((w) => signalOf(w, signal)).filter(Number.isFinite)) / 100,
                      )}
                      onChange={setThreshold}
                      format={(v) => fmt(v, 3)}
                      info={V.monitorThreshold}
                    />
                  </div>
                </Panel>

                <Panel title="How it did">
                  <LineChart
                    series={[
                      chart(signal, 'var(--accent)'),
                      {
                        id: 'threshold',
                        label: 'threshold',
                        color: 'var(--err)',
                        dashed: true,
                        points: windows.map((w) => ({ x: w.index, y: threshold })),
                      },
                    ]}
                    height={190}
                    xLabel="window"
                  />
                  {alert && (
                    <>
                      <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                        <Stat label="Caught" value={`${alert.caught} of ${alert.drifted}`} tone={alert.caught > 0 ? 'ok' : undefined} />
                        <Stat label="Missed" value={alert.missed} tone={alert.missed > 0 ? 'err' : 'ok'} />
                        <Stat label="False alarms" value={alert.falseAlarms} tone={alert.falseAlarms > 0 ? 'warn' : 'ok'} />
                        <Stat
                          label="Windows before anyone noticed"
                          value={alert.delay >= 0 ? alert.delay : 'never'}
                          tone={alert.delay < 0 && alert.drifted > 0 ? 'err' : undefined}
                        />
                      </div>
                      <div className="mt-4">
                        {alert.drifted === 0 && alert.falseAlarms > 0 ? (
                          <Callout tone="err">
                            Nothing happened in this period and the alert fired {alert.falseAlarms} times.
                            Every one of those is somebody investigating a non-problem. This is how alerts
                            get ignored.
                          </Callout>
                        ) : alert.drifted === 0 ? (
                          <Callout tone="insight">
                            Nothing happened and nothing fired, which is the correct behaviour and worth
                            establishing before you judge any other scenario. Now switch to a scenario
                            where something does happen, without touching the threshold.
                          </Callout>
                        ) : alert.missed === alert.drifted ? (
                          <Callout tone="err">
                            The traffic genuinely drifted for {alert.drifted} windows and the alert never
                            fired once. It may as well not exist. Lower the bar until it does, and watch
                            what that costs you in false alarms.
                          </Callout>
                        ) : alert.falseAlarms === 0 && alert.missed === 0 ? (
                          <Callout tone="insight">
                            Caught everything, cried wolf never, noticed after {alert.delay} window
                            {alert.delay === 1 ? '' : 's'}. Enjoy it: this threshold was chosen with the
                            whole history visible, which is a luxury nobody has in advance. Change the
                            scenario without changing the threshold and see how well it travels.
                          </Callout>
                        ) : (
                          <Callout tone="warn">
                            {alert.caught} caught, {alert.missed} missed, {alert.falseAlarms} false alarm
                            {alert.falseAlarms === 1 ? '' : 's'}. This is the normal state of affairs.
                            Moving the slider trades one column against the other and cannot empty both.
                          </Callout>
                        )}
                      </div>
                    </>
                  )}
                </Panel>

                {sweep.length > 0 && (
                  <Panel title="Every threshold at once" subtitle="The tradeoff, drawn.">
                    <LineChart
                      series={[
                        {
                          id: 'fa',
                          label: 'false alarms',
                          color: 'var(--warn)',
                          points: sweep.map((s) => ({ x: s.threshold, y: s.falseAlarms })),
                        },
                        {
                          id: 'miss',
                          label: 'missed',
                          color: 'var(--err)',
                          points: sweep.map((s) => ({ x: s.threshold, y: s.missed })),
                        },
                      ]}
                      height={180}
                      xLabel="threshold"
                      marker={threshold}
                    />
                    <p className="mt-2 text-[11.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                      The marker is where your slider sits. The two lines cannot both reach zero unless the
                      signal separates the two states perfectly, which real signals do not. Where you sit
                      between them is a decision about your team, not about the mathematics.
                    </p>
                  </Panel>
                )}
              </>
            )}
          </div>

          <div className="grid content-start gap-5">
            <Panel title="What counts as drifted">
              <p className="text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                A window is counted as genuinely drifted when more than a fifth of its traffic came from
                the unfamiliar source. That definition is only available here because the experiment was
                constructed; in production there is no such column, which is exactly why the threshold has
                to be chosen on a proxy in the first place.
              </p>
            </Panel>
            <Panel title="Try this">
              <p className="text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                Tune a threshold on the gradual scenario until it looks good. Then go back, switch to the
                spike scenario, and run it again without touching the slider. A threshold fitted to one
                pattern of trouble rarely survives a different one, and production does not tell you which
                pattern is coming.
              </p>
            </Panel>
          </div>
        </div>
      )}

      {/* --------------------------------------------- 4. retrain or not -- */}
      {tab === 'retrain' && (
        <div className="grid gap-5 lg:grid-cols-[1fr_360px]">
          <div className="grid content-start gap-5">
            {!decision ? (
              <Empty>
                <div className="mb-3">Nothing to decide about yet.</div>
                <p className="mb-4">Run a period of traffic and the decision becomes concrete.</p>
                <Btn variant="primary" onClick={() => setTab('drift')}>
                  Go and run one
                </Btn>
              </Empty>
            ) : (
              <>
                <Panel title="When a dashboard becomes a decision">
                  <Depth
                    plain={
                      <p>
                        Drift is not a reason to retrain. Drift that costs more than the retrain is. A new
                        model costs compute, it costs the time of the people who run it, and it has not
                        been through the evaluation the current one passed, so it carries risk that does
                        not appear on any invoice. The question is never "has it drifted", it is "has it
                        drifted enough".
                      </p>
                    }
                    math={
                      <p>
                        With degradation <span className="mono">d</span>, traffic{' '}
                        <span className="mono">n</span> per day and value{' '}
                        <span className="mono">v</span> per request, the loss rate is{' '}
                        <span className="mono">d &middot; n &middot; v</span> and payback is{' '}
                        <span className="mono">C / (d n v)</span> days. Retraining is worth it when payback
                        is short relative to how long the drift will persist.
                      </p>
                    }
                    code={
                      <Code>{`const degradation = (launch - now) / launch;
const lossPerDay  = degradation * requestsPerDay * valuePerRequest;
const payback     = retrainCost / lossPerDay;`}</Code>
                    }
                  />
                </Panel>

                <Panel title="Price it">
                  <div className="grid gap-3">
                    <Slider
                      label="Requests per day"
                      value={requestsPerDay}
                      min={100}
                      max={5_000_000}
                      step={100}
                      onChange={setRequestsPerDay}
                      format={(v) => fmtInt(v)}
                      info={V.monitorRequestsPerDay}
                    />
                    <Slider
                      label="What one good answer is worth"
                      value={valuePerRequest}
                      min={0.001}
                      max={1}
                      step={0.001}
                      onChange={setValuePerRequest}
                      format={(v) => `$${fmt(v, 3)}`}
                      info={V.monitorValue}
                    />
                    <Slider
                      label="Cost of retraining and shipping"
                      value={retrainCost}
                      min={100}
                      max={100_000}
                      step={100}
                      onChange={setRetrainCost}
                      format={(v) => `$${fmtInt(v)}`}
                      info={V.monitorRetrainCost}
                    />
                  </div>

                  <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                    <Stat
                      label="Quality lost"
                      value={fmtPct(decision.degradation, 1)}
                      tone={decision.degradation > 0.1 ? 'err' : decision.degradation > 0.02 ? 'warn' : 'ok'}
                    />
                    <Stat label="Costing per day" value={`$${fmtInt(Math.round(decision.lossPerDay))}`} />
                    <Stat label="Retrain costs" value={`$${fmtInt(decision.retrainCost)}`} />
                    <Stat
                      label="Pays back in"
                      value={Number.isFinite(decision.paybackDays) ? `${fmt(decision.paybackDays, 1)} days` : 'never'}
                      tone={decision.worthIt ? 'ok' : 'warn'}
                    />
                  </div>

                  <div className="mt-4">
                    {decision.degradation === 0 ? (
                      <Callout tone="insight">
                        Nothing has degraded over this period, so there is nothing to buy. Run the gradual
                        or sudden scenario to get a case worth deciding about.
                      </Callout>
                    ) : decision.worthIt ? (
                      <Callout tone="err" title="Retrain">
                        The degradation is costing ${fmtInt(Math.round(decision.lossPerDay))} a day and the
                        fix costs ${fmtInt(decision.retrainCost)}, so it pays for itself in{' '}
                        {fmt(decision.paybackDays, 1)} days. That is a clear case &mdash; provided the
                        drift is permanent. If this was the spike scenario, you would be buying a new model
                        to chase traffic that has already gone away.
                      </Callout>
                    ) : (
                      <Callout tone="warn" title="Not yet">
                        The model really is worse, and at this volume it is only costing $
                        {fmtInt(Math.round(decision.lossPerDay))} a day, so the retrain would take{' '}
                        {fmt(decision.paybackDays, 0)} days to pay for itself. Knowingly running a degraded
                        model because the fix is not worth it is a legitimate engineering decision, and it
                        only stays legitimate while somebody is still watching the number.
                      </Callout>
                    )}
                  </div>
                </Panel>
              </>
            )}
          </div>

          <div className="grid content-start gap-5">
            {decision && (
              <Panel title="The period in summary">
                <div className="grid gap-2">
                  <Stat label="Scenario" value={SCENARIO_INFO[scenario].label} />
                  <Stat label="Quality at the start" value={fmtPct(first?.trueQuality ?? 0, 1)} />
                  <Stat label="Quality at the end" value={fmtPct(last?.trueQuality ?? 0, 1)} />
                  <Stat label="Surprise at the start" value={fmt(first?.surprise ?? 0, 3)} />
                  <Stat label="Surprise at the end" value={fmt(last?.surprise ?? 0, 3)} />
                </div>
                <p className="mt-3 text-[11.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                  The quality figures are the answer key, used here to make the decision concrete. In
                  production you would be estimating the degradation from the surprise figures and your
                  own judgement about how they relate, which is a good deal less comfortable.
                </p>
              </Panel>
            )}

            <Panel title="The loop closes">
              <p className="text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                That is the whole lifecycle: decide what success means and how you will measure it, get the
                data, choose a shape, train it, evaluate it honestly, constrain it, serve it inside a cost
                envelope, and watch it afterwards until the watching tells you to start again. Nothing
                about a deployed model is finished. It is in maintenance, and this page is how you know
                when the maintenance is due.
              </p>
            </Panel>
          </div>
        </div>
      )}
    </div>
  );
}
