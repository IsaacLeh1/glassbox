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
  Slider,
  Stat,
  Toggle,
  fmt,
  fmtInt,
  fmtPct,
} from '../ui/kit';
import { V } from '../content/varInfo';
import { useModel } from '../store/model';
import { Finetuner } from '../engine/finetune';
import { buildLMTrainer } from '../engine/lmPresets';
import {
  BASELINE_INFO,
  METRIC_INFO,
  ORIGIN_INFO,
  casesFromIds,
  casesNeeded,
  contaminationOf,
  makeBaseline,
  modelPredictor,
  byOrigin,
  runEval,
  separated,
  type BaselineId,
  type EvalCase,
  type EvalRun,
  type MetricId,
} from '../engine/evals';

/**
 * Module 10: designing the evaluation before there is anything to evaluate.
 *
 * The steps are numbered and deliberately gated. You cannot see a model score
 * until you have committed to what would count as success, because the moment
 * you see a number the target stops being a judgement and starts being a
 * negotiation. That gate is not UI politeness; it is the subject of the
 * module.
 */

type Step = 'define' | 'cases' | 'metric' | 'baselines' | 'commit' | 'run';

const STEPS: { id: Step; label: string }[] = [
  { id: 'define', label: '1 · Define success' },
  { id: 'cases', label: '2 · Write the test' },
  { id: 'metric', label: '3 · Choose a metric' },
  { id: 'baselines', label: '4 · Beat the dumb thing' },
  { id: 'commit', label: '5 · Commit to a bar' },
  { id: 'run', label: '6 · Find out' },
];

interface Prereg {
  claim: string;
  metric: MetricId;
  target: number;
  cases: number;
  at: number;
}

/** A score with its interval, drawn so the interval is the thing you notice. */
function ScoreBar({ run, target }: { run: EvalRun; target?: number }) {
  return (
    <div>
      <div className="relative h-7 w-full overflow-hidden rounded" style={{ background: 'var(--panel-3)' }}>
        <div
          className="absolute inset-y-0 rounded"
          style={{
            left: `${run.lo * 100}%`,
            width: `${Math.max(0.5, (run.hi - run.lo) * 100)}%`,
            background: 'color-mix(in srgb, var(--accent) 28%, transparent)',
          }}
        />
        <div
          className="absolute inset-y-0 w-[2px]"
          style={{ left: `${run.mean * 100}%`, background: 'var(--accent)' }}
        />
        {target !== undefined && (
          <div
            className="absolute inset-y-0 w-[2px]"
            style={{ left: `${target * 100}%`, background: 'var(--warn)' }}
            title={`target ${fmtPct(target, 0)}`}
          />
        )}
      </div>
      <div className="mt-1 flex justify-between text-[10.5px]" style={{ color: 'var(--text-3)' }}>
        <span>0%</span>
        <span className="mono" style={{ color: 'var(--text-2)' }}>
          {fmtPct(run.mean, 1)} &nbsp;[{fmtPct(run.lo, 0)} to {fmtPct(run.hi, 0)}]
        </span>
        <span>100%</span>
      </div>
    </div>
  );
}

export default function EvalLab() {
  const trainer = useModel((s) => s.trainer);
  const origin = useModel((s) => s.origin);
  const publish = useModel((s) => s.publish);
  const touch = useModel((s) => s.touch);
  const [busy, setBusy] = useState(false);

  const [step, setStep] = useState<Step>('define');

  /* 1. what success means */
  const [claim, setClaim] = useState('');

  /* 2. the cases */
  const [cases, setCases] = useState<EvalCase[]>([]);
  const [count, setCount] = useState(10);
  const [promptLen, setPromptLen] = useState(8);
  const [answerLen, setAnswerLen] = useState(3);
  const [draftPrompt, setDraftPrompt] = useState('');
  const [draftExpected, setDraftExpected] = useState('');

  /* 3. the metric */
  const [metric, setMetric] = useState<MetricId>('quality');
  const [topK, setTopK] = useState(5);
  const [greedy, setGreedy] = useState(true);
  const [temperature, setTemperature] = useState(0.8);

  /* 4. baselines */
  const [baselineRuns, setBaselineRuns] = useState<EvalRun[]>([]);

  /* 5. the commitment */
  const [target, setTarget] = useState(0.5);
  const [prereg, setPrereg] = useState<Prereg | null>(null);
  const [moved, setMoved] = useState(0);

  /* 6. the result */
  const [modelRun, setModelRun] = useState<EvalRun | null>(null);

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

  const trainText = useMemo(
    () => (trainer ? trainer.tok.decode(trainer.trainIds) : ''),
    [trainer],
  );

  const contam = useMemo(
    () => cases.map((c) => contaminationOf(`${c.prompt}${c.expected}`, trainText)),
    [cases, trainText],
  );
  const dirty = contam.filter((c) => c.verbatim).length;

  const genCfg = useMemo(() => ({ greedy, temperature, seed: 1 }), [greedy, temperature]);
  const metricCfg = useMemo(() => ({ metric, topK }), [metric, topK]);

  /** Changing the test after a result exists is the thing this module is about. */
  const noteMove = useCallback(() => {
    if (modelRun) setMoved((m) => m + 1);
  }, [modelRun]);

  const addFromSplit = (which: 'heldout' | 'train') => {
    if (!trainer) return;
    const ids = which === 'heldout' ? trainer.valIds : trainer.trainIds;
    const fresh = casesFromIds(trainer.tok, ids, which, count, promptLen, answerLen);
    setCases((c) => [...c.filter((x) => x.origin !== which), ...fresh]);
    noteMove();
  };

  const addWritten = () => {
    if (!draftPrompt.trim() || !draftExpected.trim()) return;
    setCases((c) => [
      ...c,
      { id: `written-${Date.now()}`, prompt: draftPrompt, expected: draftExpected, origin: 'written' },
    ]);
    setDraftPrompt('');
    setDraftExpected('');
    noteMove();
  };

  const runBaselines = () => {
    if (!trainer || cases.length === 0) return;
    setBusy(true);
    setTimeout(() => {
      const ids: BaselineId[] = ['uniform', 'unigram', 'bigram', 'repeat'];
      setBaselineRuns(
        ids.map((id) =>
          runEval(makeBaseline(id, trainer.trainIds, trainer.tok.size), trainer.tok, cases, metricCfg, genCfg),
        ),
      );
      setBusy(false);
    }, 20);
  };

  const commit = () => {
    setPrereg({ claim, metric, target, cases: cases.length, at: Date.now() });
    setStep('run');
  };

  const runModel = () => {
    if (!trainer || cases.length === 0) return;
    setBusy(true);
    setTimeout(() => {
      setModelRun(runEval(modelPredictor(trainer.model), trainer.tok, cases, metricCfg, genCfg));
      setBusy(false);
    }, 20);
  };

  /* Deliberate contamination: train on the answers, then score them. */
  const [leak, setLeak] = useState<{ ft: Finetuner; before: number } | null>(null);

  const contaminate = () => {
    if (!trainer || !modelRun || cases.length === 0) return;
    setBusy(true);
    setTimeout(() => {
      const before = modelRun.mean;
      const ft = new Finetuner(
        trainer.model,
        trainer.tok,
        cases.map((c) => `${c.prompt}${c.expected}`),
        { lr: 0.01, epochs: 20, clipNorm: 1 },
      );
      while (ft.microStep()) {
        /* run to completion; a handful of short cases is fast */
      }
      touch();
      setLeak({ ft, before });
      setModelRun(runEval(modelPredictor(trainer.model), trainer.tok, cases, metricCfg, genCfg));
      setBusy(false);
    }, 20);
  };

  const undoContaminate = () => {
    if (!trainer || !leak) return;
    setBusy(true);
    setTimeout(() => {
      leak.ft.revert();
      touch();
      setLeak(null);
      setModelRun(runEval(modelPredictor(trainer.model), trainer.tok, cases, metricCfg, genCfg));
      setBusy(false);
    }, 20);
  };

  const info = METRIC_INFO[metric];
  /**
   * The displayed result no longer describes the test as it now stands.
   * Leaving a stale number on screen with no way to refresh it is exactly the
   * kind of quiet mismatch this module is about.
   */
  const stale = !!modelRun && (modelRun.total !== cases.length || modelRun.metric !== metric);
  // Above the early return below: a hook after a conditional return changes
  // the hook order between renders, which React rejects outright.
  const slices = useMemo(() => (modelRun ? byOrigin(modelRun) : []), [modelRun]);
  /** True when no pair of origins is distinguishable, which is the usual case here. */
  const originsOverlap = slices.every((a) => slices.every((b) => a === b || (a.lo <= b.hi && b.lo <= a.hi)));

  if (!trainer) {
    return (
      <div className="mx-auto max-w-[900px] px-6 py-10">
        <Empty>
          <div className="mb-3 text-[13px] font-medium" style={{ color: 'var(--text-2)' }}>
            You need a model to evaluate.
          </div>
          <p className="mb-4">
            Strictly, you would design the evaluation before you had one &mdash; that is the whole argument
            of this module. But the test set here is cut from the corpus the model was trained on, so it
            needs one to exist. Build one in{' '}
            <Link to="/studio" style={{ color: 'var(--accent)' }}>Build your own</Link>, or start here.
          </p>
          <Btn onClick={loadStarter} variant="primary" disabled={busy}>
            {busy ? 'Training a starter model...' : 'Train a starter model here'}
          </Btn>
        </Empty>
      </div>
    );
  }

  const best = baselineRuns.length > 0 ? baselineRuns.reduce((a, b) => (b.mean > a.mean ? b : a)) : null;
  const beatsBaseline = modelRun && best ? modelRun.mean > best.mean : false;
  const beatsClearly = modelRun && best ? separated(modelRun, best) && modelRun.mean > best.mean : false;
  const hitTarget = modelRun && prereg ? modelRun.mean >= prereg.target : false;

  return (
    <div className="mx-auto max-w-[1400px] px-6 py-6">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <Segmented value={step} onChange={setStep} options={STEPS} />
        <div className="flex items-center gap-2">
          {cases.length > 0 && <Badge>{cases.length} cases</Badge>}
          {dirty > 0 && <Badge tone="err">{dirty} contaminated</Badge>}
          {prereg && <Badge tone="ok">bar committed</Badge>}
          {moved > 0 && <Badge tone="warn">test changed {moved}x after seeing a score</Badge>}
        </div>
      </div>

      {/* ------------------------------------------------ 1. define -- */}
      {step === 'define' && (
        <div className="grid gap-5 lg:grid-cols-[1fr_360px]">
          <div className="grid content-start gap-5">
            <Panel title="Write down what success means, before anything else">
              <Depth
                plain={
                  <>
                    <p className="mb-2">
                      This is the step almost everyone skips, and skipping it is why so many models get
                      declared a success without anyone being able to say what they are good at.
                    </p>
                    <p className="mb-2">
                      The trouble is not dishonesty. It is that a number you have already seen stops being
                      evidence and starts being something to explain. If you train first and measure
                      second, you will pick the metric that flattered the run, quietly drop the cases it
                      failed, and settle on a bar just below wherever it landed. Every one of those choices
                      feels reasonable at the time. Together they mean the evaluation tells you nothing.
                    </p>
                    <p>
                      So the order is fixed: say what the model must do, build the test, find out what a
                      trivial system already scores, commit to a bar, and only then look. The rest of this
                      module walks that order and will not let you skip to the end.
                    </p>
                  </>
                }
                math={
                  <>
                    <p className="mb-2">
                      Choosing among <span className="mono">m</span> metrics after seeing results inflates
                      the apparent effect: the expected maximum of <span className="mono">m</span> noisy
                      estimates exceeds the true value, and the gap grows roughly with{' '}
                      <span className="mono">&sigma;&radic;(2 ln m)</span>.
                    </p>
                    <p>
                      Pre-registration fixes the comparison in advance, so the reported number is an
                      unbiased estimate of one quantity rather than the best of several.
                    </p>
                  </>
                }
                code={
                  <Code>{`# the order that keeps a result meaningful
spec      = define_success()      # before any training
testset   = build_heldout()       # never trained on
metric    = choose_metric(spec)   # with its blind spot written down
floor     = run(baselines)        # what does nothing-at-all score
bar       = commit(target)        # recorded, timestamped
result    = run(model)            # only now`}</Code>
                }
              />
            </Panel>

            <Panel title="What should this model be able to do?">
              <Field label="Say it in one sentence">
                <textarea
                  value={claim}
                  onChange={(e) => setClaim(e.target.value)}
                  rows={3}
                  placeholder="It should continue a sentence from the stories corpus in a way that matches how the text actually continues."
                  className="focus-ring w-full resize-y rounded-lg px-2.5 py-2 text-[12.5px]"
                  style={{ background: 'var(--bg-2)', border: '1px solid var(--border)', color: 'var(--text)' }}
                />
              </Field>
              <p className="mt-2 text-[11.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                Write it before you know what it scores. It gets recorded with your committed bar in step
                five, so you can check afterwards whether you actually tested the thing you said.
              </p>
              {claim.trim().length > 0 && (
                <div className="mt-3">
                  <Btn variant="primary" size="sm" onClick={() => setStep('cases')}>
                    Next: build the test
                  </Btn>
                </div>
              )}
            </Panel>
          </div>

          <div className="grid content-start gap-5">
            <Panel title="The model being judged">
              <div className="grid gap-2">
                <Stat label="Model" value={origin?.label ?? 'unknown'} />
                <Stat label="Parameters" value={fmtInt(trainer.model.paramCount)} />
                <Stat label="Trained for" value={origin?.steps ? `${fmtInt(origin.steps)} steps` : 'not trained'} />
                <Stat label="Held-out tokens" value={fmtInt(trainer.valIds.length)} sub="never trained on" />
              </div>
            </Panel>
            <Panel title="Why held out matters">
              <p className="text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                When this model was built, a slice of the corpus was cut off and kept back. The training
                loop has never seen those {fmtInt(trainer.valIds.length)} tokens. That is the only text
                here that can honestly test anything, and step two lets you build cases from the training
                split as well so you can see exactly how much difference it makes.
              </p>
            </Panel>
          </div>
        </div>
      )}

      {/* ------------------------------------------------- 2. cases -- */}
      {step === 'cases' && (
        <div className="grid gap-5 lg:grid-cols-[1fr_380px]">
          <div className="grid content-start gap-5">
            <Panel title="Build the test set" subtitle="A prompt, and the continuation that really followed it.">
              <div className="grid gap-3 sm:grid-cols-3">
                <Slider label="Cases" value={count} min={2} max={40} step={1} onChange={(v) => { setCount(v); }} format={(v) => String(v)} info={V.evalCaseCount} />
                <Slider label="Prompt tokens" value={promptLen} min={2} max={Math.max(3, trainer.model.cfg.blockSize - 4)} step={1} onChange={setPromptLen} format={(v) => String(v)} info={V.evalPromptLen} />
                <Slider label="Answer tokens" value={answerLen} min={1} max={8} step={1} onChange={setAnswerLen} format={(v) => String(v)} info={V.evalAnswerLen} />
              </div>
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <Btn variant="primary" size="sm" onClick={() => addFromSplit('heldout')}>
                  Take {count} from the held-out split
                </Btn>
                <Btn size="sm" onClick={() => addFromSplit('train')}>
                  Take {count} from the training split
                </Btn>
                <Btn size="sm" onClick={() => { setCases([]); setBaselineRuns([]); }} disabled={cases.length === 0}>
                  Clear
                </Btn>
              </div>
              <div className="mt-3">
                <Callout tone="insight">
                  Take a set from each and step six will score them separately, so you can see whether it
                  makes any difference. Do not assume it will. These corpora are generated from a handful
                  of sentence templates, so the held-out text is statistically identical to the training
                  text and there is nothing specific for the model to have memorised &mdash; the two scores
                  usually come out the same. That is the real lesson: whether contamination inflates a
                  result depends entirely on the data, and the only way to know is to measure it. On
                  scraped web data, where individual documents are unique and memorable, the gap is often
                  enormous. Step six also lets you create the effect deliberately and watch it appear.
                </Callout>
              </div>
            </Panel>

            <Panel title="Or write one yourself">
              <div className="grid gap-3 sm:grid-cols-2">
                <Field label="Prompt">
                  <input
                    value={draftPrompt}
                    onChange={(e) => setDraftPrompt(e.target.value)}
                    placeholder="the fox"
                    spellCheck={false}
                    className="focus-ring mono w-full rounded-lg px-2.5 py-2 text-[12px]"
                    style={{ background: 'var(--bg-2)', border: '1px solid var(--border)', color: 'var(--text)' }}
                  />
                </Field>
                <Field label="What should follow">
                  <input
                    value={draftExpected}
                    onChange={(e) => setDraftExpected(e.target.value)}
                    placeholder=" sat"
                    spellCheck={false}
                    className="focus-ring mono w-full rounded-lg px-2.5 py-2 text-[12px]"
                    style={{ background: 'var(--bg-2)', border: '1px solid var(--border)', color: 'var(--text)' }}
                  />
                </Field>
              </div>
              <div className="mt-2">
                <Btn size="sm" onClick={addWritten} disabled={!draftPrompt.trim() || !draftExpected.trim()}>
                  Add this case
                </Btn>
              </div>
            </Panel>

            {cases.length > 0 && (
              <Panel title={`${cases.length} cases`} pad={false}>
                <div className="max-h-[420px] overflow-y-auto">
                  {cases.map((c, i) => {
                    const ct = contam[i];
                    return (
                      <div key={c.id} className="px-4 py-2.5" style={{ borderBottom: '1px solid var(--border)' }}>
                        <div className="flex items-start justify-between gap-3">
                          <div className="min-w-0 flex-1">
                            <span className="mono text-[11.5px]" style={{ color: 'var(--text-3)' }}>
                              {c.prompt}
                            </span>
                            <span className="mono text-[11.5px] font-medium" style={{ color: 'var(--accent)' }}>
                              {c.expected}
                            </span>
                          </div>
                          <div className="flex shrink-0 items-center gap-1.5">
                            <Badge tone={ORIGIN_INFO[c.origin].trustworthy ? 'neutral' : 'err'}>
                              {ORIGIN_INFO[c.origin].label}
                            </Badge>
                            {ct?.verbatim ? (
                              <Badge tone="err">seen in training</Badge>
                            ) : ct && ct.overlap > 0.5 ? (
                              <Badge tone="warn">{fmtPct(ct.overlap, 0)} overlap</Badge>
                            ) : null}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </Panel>
            )}
          </div>

          <div className="grid content-start gap-5">
            <Panel title="How big does a test set need to be?">
              <p className="mb-3 text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                Bigger than feels necessary. To pin a score down to within five points either way you need
                roughly <span className="mono">{fmtInt(casesNeeded(0.5, 0.05))}</span> cases. To within one
                point, about <span className="mono">{fmtInt(casesNeeded(0.5, 0.01))}</span>. Precision costs
                the square of itself.
              </p>
              <div className="grid grid-cols-2 gap-2">
                <Stat label="You have" value={cases.length} tone={cases.length < 20 ? 'warn' : undefined} />
                <Stat label="Interval width" value={cases.length > 0 ? `±${fmtPct(0.98 / Math.sqrt(cases.length) / 2, 0)}` : '—'} hint="roughly, at a 50% score" />
              </div>
              {cases.length > 0 && cases.length < 20 && (
                <div className="mt-3">
                  <Callout tone="warn">
                    At {cases.length} cases almost nothing will be distinguishable from anything else. That
                    is worth seeing rather than avoiding: run it, look at how wide the interval is, then
                    come back and add more.
                  </Callout>
                </div>
              )}
            </Panel>

            <Panel title="Contamination">
              <div className="grid gap-2">
                <Stat label="Cases the model has seen verbatim" value={dirty} tone={dirty > 0 ? 'err' : 'ok'} />
                <Stat label="From the held-out split" value={cases.filter((c) => c.origin === 'heldout').length} />
                <Stat label="From the training split" value={cases.filter((c) => c.origin === 'train').length} tone={cases.some((c) => c.origin === 'train') ? 'warn' : undefined} />
                <Stat label="Written by you" value={cases.filter((c) => c.origin === 'written').length} />
              </div>
              <p className="mt-3 text-[11.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                Measured, not assumed: every case is checked for whether its text occurs in the training
                corpus, both verbatim and as overlapping ten-character runs.
              </p>
            </Panel>

            {cases.length > 0 && (
              <Btn variant="primary" onClick={() => setStep('metric')}>
                Next: choose a metric
              </Btn>
            )}
          </div>
        </div>
      )}

      {/* ------------------------------------------------ 3. metric -- */}
      {step === 'metric' && (
        <div className="grid gap-5 lg:grid-cols-[1fr_360px]">
          <div className="grid content-start gap-5">
            <Panel title="Every metric is a decision about what you are willing to miss">
              <div className="grid gap-2">
                {(Object.keys(METRIC_INFO) as MetricId[]).map((id) => {
                  const m = METRIC_INFO[id];
                  const on = metric === id;
                  return (
                    <button
                      key={id}
                      onClick={() => { setMetric(id); noteMove(); }}
                      className="rounded-lg p-3 text-left transition-colors"
                      style={{
                        background: on ? 'var(--panel-2)' : 'transparent',
                        border: `1px solid ${on ? 'var(--accent)' : 'var(--border)'}`,
                        cursor: 'pointer',
                      }}
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-[12.5px] font-medium" style={{ color: on ? 'var(--accent)' : 'var(--text)' }}>
                          {m.label}
                        </span>
                        {!m.binary && <Badge>continuous</Badge>}
                      </div>
                      <p className="mt-1 text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                        {m.blurb}
                      </p>
                      <p className="mt-1.5 text-[11.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                        <span style={{ color: 'var(--warn)' }}>What it misses: </span>
                        {m.blindSpot}
                      </p>
                    </button>
                  );
                })}
              </div>
            </Panel>

            <Panel title="How the answer gets produced">
              <div className="grid gap-3">
                <Toggle
                  label="Take the single best token every time"
                  checked={greedy}
                  onChange={(v) => { setGreedy(v); noteMove(); }}
                  info={V.evalGreedy}
                />
                {!greedy && (
                  <Slider label="Temperature" value={temperature} min={0.1} max={1.6} step={0.05} onChange={setTemperature} info={V.evalTemperature} />
                )}
                {metric === 'topK' && (
                  <Slider label="How many count as right" value={topK} min={1} max={Math.min(40, trainer.tok.size)} step={1} onChange={(v) => { setTopK(v); noteMove(); }} format={(v) => String(v)} info={V.evalTopK} />
                )}
              </div>
              {!greedy && (
                <div className="mt-3">
                  <Callout tone="warn">
                    With sampling on, the same model scores differently every run. A single number from a
                    sampled eval is not a measurement of the model; it is a measurement of the model and
                    one roll of the dice together. If you need sampling, run it several times and report
                    the spread.
                  </Callout>
                </div>
              )}
            </Panel>
          </div>

          <div className="grid content-start gap-5">
            <Panel title="Chosen">
              <Stat label="Metric" value={info.label} />
              <div className="mt-2">
                <Stat label="Statistic" value={info.binary ? 'pass rate, Wilson interval' : 'mean, normal interval'} />
              </div>
              <p className="mt-3 text-[11.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                A binary metric gives every case a pass or a fail, so the right summary is a rate with a
                Wilson interval. A continuous one gives every case a number, so it needs a mean and a
                standard error instead. Using the wrong one is how a result acquires false precision.
              </p>
            </Panel>
            <Btn variant="primary" onClick={() => setStep('baselines')}>
              Next: find the floor
            </Btn>
          </div>
        </div>
      )}

      {/* --------------------------------------------- 4. baselines -- */}
      {step === 'baselines' && (
        <div className="grid gap-5 lg:grid-cols-[1fr_360px]">
          <div className="grid content-start gap-5">
            <Panel title="Find out what nothing-at-all already scores">
              <Depth
                plain={
                  <>
                    <p className="mb-2">
                      Before you look at the model, run the stupidest things that could possibly work. If
                      your model scores 40 percent and a lookup table of word pairs scores 45, then your
                      model has not learned anything worth having, and no amount of context about how hard
                      the task was changes that.
                    </p>
                    <p>
                      These are not strawmen. A bigram table is a real language model, it is what the field
                      used for decades, and a small transformer that has not trained long enough genuinely
                      loses to it. Running the baselines first means you find that out before you have
                      built an argument for why your number is good.
                    </p>
                  </>
                }
                math={
                  <p>
                    A result is only meaningful relative to a floor. Reporting{' '}
                    <span className="mono">s(model)</span> without{' '}
                    <span className="mono">s(trivial)</span> leaves the reader unable to compute the only
                    quantity that matters, which is the difference between them.
                  </p>
                }
                code={
                  <Code>{`# identical harness, no special cases
for p in [uniform, unigram, bigram, repeat, model]:
    score(p, cases, metric)   # same function, same cases`}</Code>
                }
              />
              <div className="mt-3">
                <Btn variant="primary" onClick={runBaselines} disabled={busy || cases.length === 0}>
                  {busy ? 'Running...' : `Run all four over ${cases.length} cases`}
                </Btn>
              </div>
            </Panel>

            {baselineRuns.length > 0 && (
              <Panel title="The floor" subtitle="Scored through exactly the same code path your model will use.">
                <div className="grid gap-4">
                  {baselineRuns
                    .slice()
                    .sort((a, b) => b.mean - a.mean)
                    .map((r) => (
                      <div key={r.predictorId}>
                        <div className="mb-1 flex items-center justify-between gap-2">
                          <span className="text-[12.5px] font-medium">{r.predictorLabel}</span>
                          <span className="mono text-[11.5px]" style={{ color: 'var(--text-2)' }}>
                            {fmtPct(r.mean, 1)}
                          </span>
                        </div>
                        <ScoreBar run={r} />
                        <p className="mt-1 text-[11px] leading-snug" style={{ color: 'var(--text-3)' }}>
                          {BASELINE_INFO[r.predictorId as BaselineId]?.blurb}
                        </p>
                      </div>
                    ))}
                </div>
                {best && (
                  <div className="mt-4">
                    <Callout tone="insight">
                      The bar to beat is <strong>{best.predictorLabel}</strong> at {fmtPct(best.mean, 1)}.
                      Anything your model scores below that is not a result. Anything within the shaded
                      interval of it is not a result either, because the two are indistinguishable at this
                      many cases.
                    </Callout>
                  </div>
                )}
              </Panel>
            )}
          </div>

          <div className="grid content-start gap-5">
            <Panel title="What is being run">
              <div className="grid gap-2">
                <Stat label="Cases" value={cases.length} />
                <Stat label="Metric" value={info.label} />
                <Stat label="Generation" value={greedy ? 'greedy, repeatable' : `sampled at ${fmt(temperature, 2)}`} tone={greedy ? undefined : 'warn'} />
              </div>
            </Panel>
            {baselineRuns.length > 0 && (
              <Btn variant="primary" onClick={() => setStep('commit')}>
                Next: commit to a bar
              </Btn>
            )}
          </div>
        </div>
      )}

      {/* ------------------------------------------------ 5. commit -- */}
      {step === 'commit' && (
        <div className="grid gap-5 lg:grid-cols-[1fr_360px]">
          <div className="grid content-start gap-5">
            <Panel title="Say what would count as success, before you look">
              <p className="mb-3 text-[12.5px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                This is the moment the whole module exists for. Pick a number now, while you still do not
                know the answer. Once it is recorded, changing it is still possible &mdash; it always is,
                in real life too &mdash; but this page will count how many times you do, and show the count
                next to the result.
              </p>
              <Slider
                label="The bar"
                value={target}
                min={0}
                max={1}
                step={0.01}
                onChange={(v) => { setTarget(v); if (prereg) noteMove(); }}
                format={(v) => fmtPct(v, 0)}
                info={V.evalTarget}
              />
              {best && (
                <p className="mt-2 text-[11.5px]" style={{ color: target <= best.mean ? 'var(--err)' : 'var(--text-3)' }}>
                  {target <= best.mean
                    ? `A bar of ${fmtPct(target, 0)} is at or below what ${best.predictorLabel} already scores. Clearing it would prove nothing.`
                    : `Above the best trivial baseline (${fmtPct(best.mean, 1)}), so clearing it would mean something.`}
                </p>
              )}
              <div className="mt-4">
                <Btn variant="primary" onClick={commit} disabled={cases.length === 0}>
                  {prereg ? 'Re-commit and continue' : 'Commit to this and find out'}
                </Btn>
              </div>
            </Panel>

            {prereg && (
              <Panel title="Recorded">
                <div className="grid gap-2">
                  <Stat label="Claim" value={prereg.claim || '(left blank)'} />
                  <Stat label="Metric" value={METRIC_INFO[prereg.metric].label} />
                  <Stat label="Bar" value={fmtPct(prereg.target, 0)} />
                  <Stat label="Cases at the time" value={prereg.cases} />
                  <Stat label="Committed" value={new Date(prereg.at).toLocaleTimeString()} />
                </div>
              </Panel>
            )}
          </div>

          <div className="grid content-start gap-5">
            <Panel title="Why this is not bureaucracy">
              <p className="text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                Pre-registration exists because the alternative does not work, and it does not work even
                for careful, honest people. Given a result and a free choice of how to judge it, everyone
                converges on a judgement that makes the result look good. Writing the bar down first is the
                only thing that reliably stops it, which is why clinical trials are required to do it and
                why serious model evaluations now do the same.
              </p>
            </Panel>
          </div>
        </div>
      )}

      {/* --------------------------------------------------- 6. run -- */}
      {step === 'run' && (
        <div className="grid gap-5 lg:grid-cols-[1fr_360px]">
          <div className="grid content-start gap-5">
            {!prereg ? (
              <Empty>
                <div className="mb-3">You have not committed to a bar yet.</div>
                <p className="mb-4">
                  This page is deliberately locked until you have. Seeing the score first is precisely the
                  thing the module is about, so it will not let you.
                </p>
                <Btn variant="primary" onClick={() => setStep('commit')}>
                  Go and set one
                </Btn>
              </Empty>
            ) : (
              <>
                <Panel
                  title="The result"
                  right={
                    modelRun && (
                      <div className="flex items-center gap-2">
                        {stale && <Badge tone="warn">out of date</Badge>}
                        <Badge>{fmt(modelRun.ms, 0)} ms</Badge>
                        <Btn size="sm" variant={stale ? 'primary' : 'soft'} onClick={runModel} disabled={busy}>
                          {busy ? 'Running...' : 'Run again'}
                        </Btn>
                      </div>
                    )
                  }
                >
                  {!modelRun ? (
                    <div>
                      <p className="mb-3 text-[12.5px]" style={{ color: 'var(--text-2)' }}>
                        Bar committed at {fmtPct(prereg.target, 0)} on {METRIC_INFO[prereg.metric].label},
                        over {cases.length} cases.
                      </p>
                      <Btn variant="primary" onClick={runModel} disabled={busy}>
                        {busy ? 'Running...' : 'Run the model'}
                      </Btn>
                    </div>
                  ) : (
                    <>
                      {stale && (
                        <div className="mb-3">
                          <Callout tone="warn" title="This score is from a different test">
                            It was measured over {modelRun.total} cases on{' '}
                            {METRIC_INFO[modelRun.metric].label}, and the test now has {cases.length} cases
                            on {info.label}. Run it again before reading anything into the number.
                          </Callout>
                        </div>
                      )}
                      <div className="mb-1 flex items-center justify-between gap-2">
                        <span className="text-[13px] font-medium">Your model</span>
                        <span className="mono text-[12px]">{fmtPct(modelRun.mean, 1)}</span>
                      </div>
                      <ScoreBar run={modelRun} target={prereg.target} />

                      <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                        <Stat label="Cases" value={modelRun.total} />
                        {METRIC_INFO[modelRun.metric].binary && (
                          <Stat label="Passed" value={`${modelRun.passed} of ${modelRun.total}`} />
                        )}
                        <Stat label="Bar" value={fmtPct(prereg.target, 0)} />
                        <Stat
                          label="Verdict"
                          value={hitTarget ? 'cleared' : 'missed'}
                          tone={hitTarget ? 'ok' : 'err'}
                        />
                      </div>

                      <div className="mt-4 grid gap-3">
                        {!beatsBaseline && best && (
                          <Callout tone="err" title="It loses to a lookup table">
                            {best.predictorLabel} scores {fmtPct(best.mean, 1)} and your model scores{' '}
                            {fmtPct(modelRun.mean, 1)}. Whatever the model has learned, it is worth less
                            than counting which token follows which. This is the normal state of a model
                            that has not trained nearly long enough, and it is exactly what the baselines
                            are for: without them, {fmtPct(modelRun.mean, 1)} might have looked fine.
                          </Callout>
                        )}
                        {beatsBaseline && !beatsClearly && best && (
                          <Callout tone="warn" title="Ahead, but not provably">
                            Your model is at {fmtPct(modelRun.mean, 1)} and {best.predictorLabel} is at{' '}
                            {fmtPct(best.mean, 1)}, but the two intervals overlap. With {cases.length}{' '}
                            cases you cannot tell them apart. The honest report is not "we beat the
                            baseline"; it is "we could not distinguish them, and here is how many cases we
                            would need". Roughly{' '}
                            {fmtInt(casesNeeded(modelRun.mean, Math.max(0.02, Math.abs(modelRun.mean - best.mean) / 2)))}{' '}
                            would do it.
                          </Callout>
                        )}
                        {beatsClearly && best && (
                          <Callout tone="insight" title="A real difference">
                            Your model at {fmtPct(modelRun.mean, 1)} is clear of {best.predictorLabel} at{' '}
                            {fmtPct(best.mean, 1)}, with no overlap between the intervals. That is a result
                            you can report.
                          </Callout>
                        )}
                        {dirty > 0 && (
                          <Callout tone="warn" title={`${dirty} of these cases appear verbatim in the training text`}>
                            The model has seen that text. Whether that inflated the score is a separate
                            question, and the breakdown on the right answers it for this run rather than
                            assuming. If the two origins score the same, it is because there was nothing
                            specific to memorise &mdash; not because contamination is harmless.
                          </Callout>
                        )}
                        {moved > 0 && (
                          <Callout tone="warn" title={`The test was changed ${moved} time${moved === 1 ? '' : 's'} after a score was visible`}>
                            Each change may have been perfectly reasonable on its own. Together they mean
                            this number is no longer an unbiased estimate of anything: the test was tuned,
                            at least in part, to the answer. A real evaluation would be re-run from the
                            beginning on a fresh held-out set.
                          </Callout>
                        )}
                      </div>
                    </>
                  )}
                </Panel>

                {modelRun && (
                  <Panel title="Every case" pad={false}>
                    <div className="max-h-[420px] overflow-y-auto">
                      {modelRun.results.map((r) => (
                        <div key={r.case.id} className="px-4 py-2.5" style={{ borderBottom: '1px solid var(--border)' }}>
                          <div className="flex items-start justify-between gap-3">
                            <div className="min-w-0 flex-1">
                              <div className="mono truncate text-[11.5px]" style={{ color: 'var(--text-3)' }}>
                                {r.case.prompt}
                              </div>
                              <div className="mt-0.5 flex flex-wrap items-center gap-2 text-[11.5px]">
                                <span className="mono" style={{ color: 'var(--pos)' }}>
                                  wanted {r.case.expected}
                                </span>
                                <span className="mono" style={{ color: 'var(--text-2)' }}>
                                  got {r.got}
                                </span>
                              </div>
                            </div>
                            <div className="flex shrink-0 items-center gap-1.5">
                              <span className="mono text-[10.5px]" style={{ color: 'var(--text-3)' }}>
                                {r.detail}
                              </span>
                              <Badge tone={r.pass ? 'ok' : 'neutral'}>
                                {METRIC_INFO[modelRun.metric].binary ? (r.pass ? 'pass' : 'fail') : fmt(r.score, 3)}
                              </Badge>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </Panel>
                )}
              </>
            )}
          </div>

          <div className="grid content-start gap-5">
            {prereg && (
              <Panel title="What you committed to">
                <div className="grid gap-2">
                  <Stat label="Claim" value={prereg.claim || '(left blank)'} />
                  <Stat label="Metric" value={METRIC_INFO[prereg.metric].label} />
                  <Stat label="Bar" value={fmtPct(prereg.target, 0)} />
                  <Stat label="Committed at" value={new Date(prereg.at).toLocaleTimeString()} />
                  {prereg.metric !== metric && (
                    <Stat label="Metric now" value={METRIC_INFO[metric].label} tone="warn" sub="changed since committing" />
                  )}
                  {prereg.cases !== cases.length && (
                    <Stat label="Cases now" value={cases.length} tone="warn" sub={`was ${prereg.cases}`} />
                  )}
                </div>
              </Panel>
            )}

            {modelRun && slices.length > 1 && (
              <Panel title="Split by where the cases came from" subtitle="The contamination question, answered for this run.">
                <div className="grid gap-3">
                  {slices.map((s) => (
                    <div key={s.origin}>
                      <div className="mb-0.5 flex items-center justify-between gap-2 text-[11.5px]">
                        <span style={{ color: ORIGIN_INFO[s.origin].trustworthy ? 'var(--text-2)' : 'var(--warn)' }}>
                          {ORIGIN_INFO[s.origin].label} ({s.n})
                        </span>
                        <span className="mono" style={{ color: 'var(--text-3)' }}>
                          {fmtPct(s.mean, 1)}
                        </span>
                      </div>
                      <ScoreBar run={{ ...modelRun, mean: s.mean, lo: s.lo, hi: s.hi }} />
                    </div>
                  ))}
                </div>
                <p className="mt-3 text-[11.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                  {originsOverlap
                    ? 'These intervals overlap, so on this data being trained on a case made no measurable difference. That is a fact about this corpus, which is generated from templates and has nothing unique to memorise. It is not a general licence to ignore contamination.'
                    : 'These intervals do not overlap. The cases the model was trained on scored measurably differently, which is contamination showing up in the number.'}
                </p>
              </Panel>
            )}

            {modelRun && (
              <Panel title="Create the effect on purpose" subtitle="Train the model on the answers, then re-run the same test.">
                <p className="mb-3 text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                  This is contamination in its purest form: fine-tune directly on the test cases, then
                  score them. Nothing about the model has genuinely improved and the benchmark number
                  will climb anyway. It is reversible.
                </p>
                <div className="flex flex-wrap items-center gap-2">
                  <Btn
                    size="sm"
                    variant="primary"
                    onClick={contaminate}
                    disabled={busy || !!leak || cases.length === 0}
                  >
                    {busy ? 'Training...' : 'Train on the test set'}
                  </Btn>
                  {leak && (
                    <Btn size="sm" onClick={undoContaminate} disabled={busy}>
                      Undo it
                    </Btn>
                  )}
                </div>
                {leak && modelRun && (
                  <div className="mt-3 grid gap-2">
                    <Stat label="Before" value={fmtPct(leak.before, 1)} />
                    <Stat
                      label="After training on the answers"
                      value={fmtPct(modelRun.mean, 1)}
                      tone={modelRun.mean > leak.before ? 'err' : undefined}
                    />
                    <Callout tone="err">
                      The score moved from {fmtPct(leak.before, 1)} to {fmtPct(modelRun.mean, 1)} and the
                      model cannot do anything it could not do before. Any benchmark whose test set leaked
                      into training is reporting this, and from the outside the two are indistinguishable.
                    </Callout>
                  </div>
                )}
              </Panel>
            )}

            {modelRun && baselineRuns.length > 0 && (
              <Panel title="Against the floor">
                <div className="grid gap-3">
                  {[modelRun, ...baselineRuns]
                    .slice()
                    .sort((a, b) => b.mean - a.mean)
                    .map((r) => (
                      <div key={r.predictorId}>
                        <div className="mb-0.5 flex items-center justify-between gap-2 text-[11.5px]">
                          <span style={{ color: r.predictorId === 'model' ? 'var(--accent)' : 'var(--text-2)' }}>
                            {r.predictorLabel}
                          </span>
                          <span className="mono" style={{ color: 'var(--text-3)' }}>
                            {fmtPct(r.mean, 1)}
                          </span>
                        </div>
                        <ScoreBar run={r} />
                      </div>
                    ))}
                </div>
              </Panel>
            )}

            <Panel title="What you would do next">
              <p className="text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                At a company this is the loop: the evaluation is the fixed thing and the model is what
                changes. You would go back to{' '}
                <Link to="/studio" style={{ color: 'var(--accent)' }}>module 07</Link> and train longer, or
                to <Link to="/comms" style={{ color: 'var(--accent)' }}>module 09</Link> and teach it
                something, then return here and re-run this identical test. What you would not do is edit
                the test until the number improves &mdash; and now that you have watched that impulse
                appear, you will recognise it when it does.
              </p>
            </Panel>
          </div>
        </div>
      )}
    </div>
  );
}
