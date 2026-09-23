import { useCallback, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Badge,
  Btn,
  Callout,
  Depth,
  Field,
  Panel,
  ProgressBar,
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
import { useWork } from '../store/work';
import { useFrameLoop } from '../ui/loop';
import { CORPORA, getCorpus } from '../engine/corpus';
import { DEFAULT_LM_TRAIN, LMTrainer } from '../engine/lmTrainer';
import { DEFAULT_TCONFIG } from '../engine/transformer';
import { paramCountFor } from '../engine/resources';
import { prettyBytes } from '../engine/quantise';
import { GPU_PROFILES } from '../sim/cluster';
import { costOf, decodeBound } from '../engine/serving';
import { resolveBan } from '../engine/steering';
import {
  DEFAULT_GEN,
  casesFromIds,
  makeBaseline,
  modelPredictor,
  runEval,
  type EvalRun,
} from '../engine/evals';
import {
  LICENCE_INFO,
  STAGES,
  canEnter,
  dataGate,
  gate,
  inspectData,
  nextStage,
  planScaling,
  scalingGate,
  type Check,
  type DataReport,
  type Gate,
  type Licence,
  type StageId,
} from '../engine/pipeline';

/**
 * The programme, run the way a lab runs it.
 *
 * The other tabs of this step let you build a model. This one makes you do it
 * the way it is actually done: in a fixed order, behind gates, with somebody
 * signing each one off. The model that comes out is real and becomes the
 * model the rest of the walkthrough uses.
 */

function GateView({ g, onSign, signed, label }: { g: Gate; onSign: () => void; signed: boolean; label: string }) {
  return (
    <div className="rounded-lg p-3" style={{ background: 'var(--bg-2)', border: '1px solid var(--border)' }}>
      <div className="mb-2 flex items-center justify-between gap-3">
        <span className="text-[11px] font-semibold uppercase tracking-[0.08em]" style={{ color: 'var(--text-3)' }}>
          Gate
        </span>
        {signed ? (
          <Badge tone="ok">signed off</Badge>
        ) : (
          <Btn size="sm" variant={g.passed ? 'primary' : 'soft'} onClick={onSign} disabled={!g.passed}>
            {g.passed ? label : `${g.blocking} check${g.blocking === 1 ? '' : 's'} blocking`}
          </Btn>
        )}
      </div>
      <div className="grid gap-1.5">
        {g.checks.map((c) => (
          <div key={c.label} className="flex items-start gap-2">
            <span className="mt-[2px] shrink-0" style={{ color: c.pass ? 'var(--ok)' : c.soft ? 'var(--warn)' : 'var(--err)' }}>
              {c.pass ? (
                <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
                  <path d="M3 8.5l3.2 3.2L13 5" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              ) : (
                <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
                  <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
                </svg>
              )}
            </span>
            <span className="min-w-0">
              <span className="text-[12px] font-medium">
                {c.label}
                {!c.pass && c.soft && (
                  <span className="ml-1.5 text-[10.5px] font-normal" style={{ color: 'var(--warn)' }}>
                    (can be accepted with a note)
                  </span>
                )}
              </span>
              <span className="block text-[11.5px] leading-snug" style={{ color: 'var(--text-3)' }}>
                {c.detail}
              </span>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function StageHead({ n, title, owner, blurb }: { n: number; title: string; owner: string; blurb: string }) {
  return (
    <div className="mb-3">
      <div className="flex items-center gap-2">
        <span
          className="mono flex h-5 w-5 items-center justify-center rounded text-[10.5px] font-semibold"
          style={{ background: 'color-mix(in srgb, var(--accent) 18%, transparent)', color: 'var(--accent)' }}
        >
          {n}
        </span>
        <span className="text-[14px] font-semibold tracking-tight">{title}</span>
        <Badge>{owner}</Badge>
      </div>
      <p className="mt-1.5 text-[12.5px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
        {blurb}
      </p>
    </div>
  );
}

export default function LabRun() {
  const publish = useModel((s) => s.publish);

  /* Everything typed or chosen here survives a reload. Only the trained
     model does not, for the reasons in store/work.ts. */
  const lab = useWork((w) => w.lab);
  const setLab = useWork((w) => w.setLab);
  const resetLab = useWork((w) => w.resetLab);

  const done = lab.done;
  const [open, setOpen] = useState<StageId>(() => nextStage(lab.done)?.id ?? 'launch');
  const sign = (id: StageId) => {
    setLab({ done: lab.done.includes(id) ? lab.done : [...lab.done, id] });
    const i = STAGES.findIndex((s) => s.id === id);
    if (i < STAGES.length - 1) setOpen(STAGES[i + 1].id);
  };

  /* 1 charter */
  const { purpose, users, success } = lab;
  const setPurpose = (v: string) => setLab({ purpose: v });
  const setUsers = (v: string) => setLab({ users: v });
  const setSuccess = (v: string) => setLab({ success: v });

  /* 2 data */
  const { corpusId, licence } = lab;
  const setCorpusId = (v: string) => setLab({ corpusId: v });
  const setLicence = (v: Licence) => setLab({ licence: v });
  const raw = useMemo(() => getCorpus(corpusId, 12), [corpusId]);
  // The last tenth is held back before anything else happens, so the
  // decontamination check has something honest to check against.
  const split = useMemo(() => {
    const cut = Math.floor(raw.length * 0.9);
    return { train: raw.slice(0, cut), held: raw.slice(cut) };
  }, [raw]);
  const report: DataReport = useMemo(() => inspectData(split.train, split.held), [split]);
  const gData = useMemo(() => dataGate(report, licence, 2000), [report, licence]);

  /* 3 scaling */
  const budgetExp = lab.budgetExp;
  const setBudgetExp = (v: number) => setLab({ budgetExp: v });
  const flops = Math.pow(10, budgetExp);
  const availableTokens = Math.round(report.cleanedChars / 4);
  const plan = useMemo(() => planScaling(flops, availableTokens), [flops, availableTokens]);

  /* 4 architecture */
  const { dModel, nLayers, nHeads, merges } = lab;
  const setDModel = (v: number) => setLab({ dModel: v });
  const setNLayers = (v: number) => setLab({ nLayers: v });
  const setNHeads = (v: number) => setLab({ nHeads: v });
  const setMerges = (v: number) => setLab({ merges: v });
  const arch = useMemo(
    () => ({ ...DEFAULT_TCONFIG, vocab: 120, dModel, nHeads, nLayers, dFF: dModel * 2, blockSize: 24 }),
    [dModel, nHeads, nLayers],
  );
  const params = useMemo(() => paramCountFor(arch), [arch]);
  const gScaling = useMemo(() => scalingGate(plan, params), [plan, params]);
  const gArch = useMemo(
    () =>
      gate('architecture', [
        {
          label: 'Heads divide the width evenly',
          pass: dModel % nHeads === 0,
          detail:
            dModel % nHeads === 0
              ? `${dModel} split across ${nHeads} heads gives ${dModel / nHeads} each.`
              : `${dModel} does not divide by ${nHeads}. Every attention implementation assumes it does.`,
        },
        {
          label: 'Fits in a browser tab',
          pass: params < 400_000,
          detail: `${fmtInt(params)} parameters. Past about 400,000 the run stops being something you can watch.`,
        },
      ]),
    [dModel, nHeads, params],
  );

  /* 5 pretraining */
  const [trainer, setTrainer] = useState<LMTrainer | null>(null);
  const [running, setRunning] = useState(false);
  const [, setTick] = useState(0);
  const steps = lab.steps;
  const setSteps = (v: number) => setLab({ steps: v });

  const build = useCallback(() => {
    const t = new LMTrainer(
      report.cleaned,
      { ...arch, vocab: 0 },
      { ...DEFAULT_LM_TRAIN, steps },
      merges,
    );
    setTrainer(t);
    setRunning(true);
  }, [report.cleaned, arch, steps, merges]);

  useFrameLoop(running && !!trainer, () => {
    if (!trainer) return;
    const n = trainer.runSlice(10);
    if (n === 0 || trainer.status === 'done') setRunning(false);
    setTick((x) => x + 1);
  });

  const last = trainer?.latest() ?? null;
  const gPretrain = useMemo(
    () =>
      gate('pretrain', [
        {
          label: 'Run completed',
          pass: !!trainer && trainer.status === 'done',
          detail: trainer
            ? `${fmtInt(trainer.step)} of ${fmtInt(trainer.cfg.steps)} steps.`
            : 'Not started.',
        },
        {
          label: 'Loss went down',
          pass: !!last && Number.isFinite(last.loss) && last.loss < 4,
          detail: last ? `Final training loss ${fmt(last.loss, 3)}, perplexity ${fmt(Math.exp(last.loss), 1)}.` : 'No metrics yet.',
        },
        {
          label: 'Nothing diverged',
          pass: !last || Number.isFinite(last.loss),
          detail: last && !Number.isFinite(last.loss) ? 'The loss became NaN. The run is dead.' : 'Loss stayed finite throughout.',
        },
      ]),
    [trainer, last, trainer?.step, trainer?.status],
  );

  /* 6 evaluation */
  const bar = lab.bar;
  const setBar = (v: number) => setLab({ bar: v });
  const [evalRuns, setEvalRuns] = useState<{ model: EvalRun; best: EvalRun } | null>(null);
  const runEvals = () => {
    if (!trainer) return;
    const cases = casesFromIds(trainer.tok, trainer.valIds, 'heldout', 20, 8, 3);
    const cfg = { metric: 'quality' as const, topK: 5 };
    const model = runEval(modelPredictor(trainer.model), trainer.tok, cases, cfg, DEFAULT_GEN);
    const baselines = (['uniform', 'unigram', 'bigram', 'repeat'] as const).map((id) =>
      runEval(makeBaseline(id, trainer.trainIds, trainer.tok.size), trainer.tok, cases, cfg, DEFAULT_GEN),
    );
    setEvalRuns({ model, best: baselines.reduce((a, b) => (b.mean > a.mean ? b : a)) });
  };
  const gEval = useMemo(
    () =>
      gate('evaluate', [
        {
          label: 'Suite has been run',
          pass: !!evalRuns,
          detail: evalRuns ? `${evalRuns.model.total} held-out cases.` : 'Not run.',
        },
        {
          label: 'Beats the trivial baselines',
          pass: !!evalRuns && evalRuns.model.mean > evalRuns.best.mean,
          detail: evalRuns
            ? `Model ${fmtPct(evalRuns.model.mean, 1)} against ${evalRuns.best.predictorLabel} at ${fmtPct(evalRuns.best.mean, 1)}.`
            : '',
        },
        {
          label: 'Clears the bar committed to in the charter',
          pass: !!evalRuns && evalRuns.model.mean >= bar,
          soft: true,
          detail: evalRuns ? `Scored ${fmtPct(evalRuns.model.mean, 1)} against a bar of ${fmtPct(bar, 0)}.` : '',
        },
      ]),
    [evalRuns, bar],
  );

  /* 7 safety */
  const banText = lab.banText;
  const setBanText = (v: string) => setLab({ banText: v });
  const redTeamed = lab.redTeamed;
  const setRedTeamed = (v: boolean) => setLab({ redTeamed: v });
  const banned = useMemo(
    () => (trainer && banText.trim() ? resolveBan(trainer.tok, banText.split(',')) : []),
    [trainer, banText],
  );
  const gSafety = useMemo(
    () =>
      gate('safety', [
        {
          label: 'Somebody tried to break it',
          pass: redTeamed,
          detail: redTeamed
            ? 'Generated from the model and read the output.'
            : 'Nobody has looked at what it actually produces yet.',
        },
        {
          label: 'Constraints decided',
          pass: banned.length > 0,
          soft: true,
          detail:
            banned.length > 0
              ? `${banned.reduce((n, b) => n + b.ids.length, 0)} vocabulary entries blocked.`
              : 'No constraints. A defensible choice for a model that writes short stories, and a recorded one.',
        },
      ]),
    [redTeamed, banned],
  );
  const [sample, setSample] = useState('');
  const redTeam = () => {
    if (!trainer) return;
    let ids = trainer.tok.encode('the ');
    for (let i = 0; i < 30; i++) {
      const { logits } = trainer.model.predictNext(ids);
      let best = 0;
      for (let j = 1; j < logits.length; j++) if (logits[j] > logits[best]) best = j;
      ids = [...ids, best];
    }
    setSample(trainer.tok.decode(ids));
    setRedTeamed(true);
  };

  /* 8 launch */
  const gpu = GPU_PROFILES.find((g) => g.id === 'a100')!;
  const bound = trainer ? decodeBound(trainer.model.paramCount, 1, 2048, trainer.model.cfg, gpu) : null;
  const cost = trainer && bound ? costOf(bound.tokensPerSecond, 200, gpu, trainer.model.paramCount, 1, 2048, trainer.model.cfg) : null;
  const shipped = lab.shipped;
  const setShipped = (v: boolean) => setLab({ shipped: v });
  const gLaunch = useMemo(
    () =>
      gate('launch', [
        { label: 'Every earlier gate signed', pass: done.length >= 7, detail: `${done.length} of 8.` },
        {
          label: 'Cost per request understood',
          pass: !!cost,
          detail: cost
            ? cost.usdPerMillionTokens < 0.01
              ? `Under a cent per million tokens on an ${gpu.label}.`
              : `About $${fmt(cost.usdPerMillionTokens, 2)} per million tokens on an ${gpu.label}.`
            : '',
        },
      ]),
    [done.length, cost, gpu.label],
  );

  const ship = () => {
    if (!trainer) return;
    publish(trainer, {
      label: `${CORPORA.find((c) => c.id === corpusId)?.label ?? corpusId} - shipped from the programme`,
      source: `cleaned corpus, ${licence} licence, ${fmtInt(report.cleanedChars)} chars`,
      chars: report.cleanedChars,
      steps: trainer.step,
      finalLoss: last?.loss ?? NaN,
      trainedAt: Date.now(),
    });
    setShipped(true);
    sign('launch');
  };

  const gateFor = (id: StageId): Gate =>
    id === 'charter'
      ? gate('charter', [
          {
            label: 'Purpose written down',
            pass: purpose.trim().length > 12,
            detail: purpose.trim().length > 12 ? 'Recorded.' : 'One sentence on what this model is for.',
          },
          {
            label: 'Users identified',
            pass: users.trim().length > 3,
            detail: users.trim().length > 3 ? 'Recorded.' : 'Who is going to use it.',
          },
          {
            label: 'Success defined before any data is collected',
            pass: success.trim().length > 12,
            detail:
              success.trim().length > 12
                ? 'Recorded, and the evaluation gate will be judged against it.'
                : 'What would count as it working. Writing this after you have seen a score is the failure step 10 is about.',
          },
        ] as Check[])
      : id === 'data'
        ? gData
        : id === 'scaling'
          ? gScaling
          : id === 'architecture'
            ? gArch
            : id === 'pretrain'
              ? gPretrain
              : id === 'evaluate'
                ? gEval
                : id === 'safety'
                  ? gSafety
                  : gLaunch;

  return (
    <div className="grid gap-4">
      <Panel title="Run it the way a lab runs it" subtitle="Eight stages, in order, each behind a gate that has to pass.">
        <Depth
          plain={
            <>
              <p className="mb-2">
                The other tabs let you build a model. This one makes you do it the way NVIDIA, OpenAI or
                Anthropic would: in a fixed order, with a gate at the end of every stage that somebody has
                to sign before the next one opens.
              </p>
              <p className="mb-2">
                That is the actual difference between a model programme and someone training a network. Not
                the hardware and not the cleverness &mdash; the gates. Nobody starts a pretraining run
                because the data looked fine; somebody signs that it passed a defined set of checks. Nobody
                ships because the samples read nicely; the model clears a bar that was written down before
                anyone saw a number.
              </p>
              <p>
                The gates here are real checks against your real text and your real model, and one of them
                will probably stop you. That is the point. The model that comes out the far end is a
                genuine one, and it becomes the model the rest of the walkthrough works on.
              </p>
            </>
          }
          math={
            <p>
              The programme is a directed chain of stages <span className="mono">s₁ … s₈</span> where{' '}
              <span className="mono">s_k</span> is enterable only if <span className="mono">s_(k-1)</span>{' '}
              is signed. Each gate is a conjunction of predicates over the artefacts produced so far, with
              soft predicates excluded from the conjunction but recorded.
            </p>
          }
        />
        <div className="mt-3">
          <ProgressBar value={done.length / STAGES.length} tone={done.length === STAGES.length ? 'ok' : 'accent'} />
          <div className="mt-1 flex items-center justify-between gap-3 text-[11px]" style={{ color: 'var(--text-3)' }}>
            <span>
              {done.length} of {STAGES.length} stages signed off
              {done.length > 0 && ' · kept across a reload'}
            </span>
            <span className="flex items-center gap-2">
              {shipped && <span style={{ color: 'var(--ok)' }}>shipped</span>}
              {done.length > 0 && (
                <Btn
                  size="sm"
                  onClick={() => {
                    resetLab();
                    setTrainer(null);
                    setRunning(false);
                    setEvalRuns(null);
                    setSample('');
                    setOpen('charter');
                  }}
                >
                  Start over
                </Btn>
              )}
            </span>
          </div>
        </div>
      </Panel>

      {STAGES.map((s) => {
        const enterable = canEnter(s.id, done);
        const signed = done.includes(s.id);
        const isOpen = open === s.id;
        const g = gateFor(s.id);

        return (
          <Panel key={s.id} pad={false}>
            <button
              onClick={() => enterable && setOpen(isOpen ? ('' as StageId) : s.id)}
              disabled={!enterable}
              className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left"
              style={{ cursor: enterable ? 'pointer' : 'not-allowed', opacity: enterable ? 1 : 0.45 }}
            >
              <span className="flex min-w-0 items-center gap-2.5">
                <span
                  className="mono flex h-5 w-5 shrink-0 items-center justify-center rounded text-[10.5px] font-semibold"
                  style={{
                    background: signed ? 'color-mix(in srgb, var(--ok) 20%, transparent)' : 'var(--panel-3)',
                    color: signed ? 'var(--ok)' : 'var(--text-3)',
                  }}
                >
                  {s.n}
                </span>
                <span className="truncate text-[13px] font-semibold">{s.title}</span>
              </span>
              <span className="flex shrink-0 items-center gap-2">
                {signed && <Badge tone="ok">signed</Badge>}
                {!signed && enterable && g.passed && <Badge tone="accent">ready</Badge>}
                {!signed && enterable && !g.passed && <Badge tone="warn">{g.blocking} blocking</Badge>}
                {!enterable && <Badge>locked</Badge>}
              </span>
            </button>

            {isOpen && enterable && (
              <div className="px-4 pb-4" style={{ borderTop: '1px solid var(--border)', paddingTop: 14 }}>
                <StageHead n={s.n} title={s.title} owner={s.owner} blurb={s.blurb} />

                {s.id === 'charter' && (
                  <div className="mb-3 grid gap-3">
                    <Field label="What is this model for?" info={V.labPurpose}>
                      <input value={purpose} onChange={(e) => setPurpose(e.target.value)}
                        placeholder="Continue a sentence in the style of the corpus it was trained on."
                        className="w-full" />
                    </Field>
                    <Field label="Who uses it?" info={V.labUsers}>
                      <input value={users} onChange={(e) => setUsers(e.target.value)} placeholder="Readers of this walkthrough" className="w-full" />
                    </Field>
                    <Field label="What would count as it working?" info={V.labSuccess}>
                      <input value={success} onChange={(e) => setSuccess(e.target.value)}
                        placeholder="Beats a bigram lookup table on held-out text." className="w-full" />
                    </Field>
                  </div>
                )}

                {s.id === 'data' && (
                  <div className="mb-3 grid gap-3">
                    <div className="grid gap-3 sm:grid-cols-2">
                      <Field label="Source" info={V.labCorpus}>
                        <Select value={corpusId} onChange={setCorpusId}
                          options={CORPORA.map((c) => ({ id: c.id, label: c.label }))} />
                      </Field>
                      <Field label="Licence" info={V.labLicence}>
                        <Select value={licence} onChange={(v) => setLicence(v as Licence)}
                          options={(Object.keys(LICENCE_INFO) as Licence[]).map((id) => ({ id, label: LICENCE_INFO[id].label }))} />
                      </Field>
                    </div>
                    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                      <Stat label="Documents" value={fmtInt(report.docs)} />
                      <Stat label="Exact duplicates" value={fmtInt(report.exactDupes.length)} tone={report.exactDupes.length > 0 ? 'warn' : 'ok'} />
                      <Stat label="Near duplicates" value={fmtInt(report.nearDupes)} tone={report.nearDupes > 0 ? 'warn' : 'ok'} />
                      <Stat label="Overlapped held-out" value={fmtInt(report.contaminated)} tone={report.contaminated > 0 ? 'err' : 'ok'} />
                    </div>
                    <Stat label="After cleaning" value={`${fmtInt(report.cleanedChars)} characters`}
                      sub={`${fmtPct(1 - report.cleanedChars / Math.max(1, split.train.length), 1)} removed`} />
                    {report.exactDupes.length > 0 && (
                      <Callout tone="warn" title="Your corpus repeats itself">
                        The most repeated document appears {report.exactDupes[0].count} times. A document
                        trained on a hundred times gets memorised rather than learned from, which is why
                        deduplication reliably improves a model at no other cost. It has been removed.
                      </Callout>
                    )}
                  </div>
                )}

                {s.id === 'scaling' && (
                  <div className="mb-3 grid gap-3">
                    <Slider label="Compute budget" value={budgetExp} min={8} max={16} step={0.25}
                      onChange={setBudgetExp} format={(v) => `10^${fmt(v, 2)} FLOPs`} info={V.labBudget} />
                    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                      <Stat label="Budget supports" value={`${fmtInt(Math.round(plan.params))} params`} />
                      <Stat label="Wanting" value={`${fmtInt(Math.round(plan.tokens))} tokens`} />
                      <Stat label="You have" value={`${fmtInt(plan.haveTokens)} tokens`} tone={plan.dataLimited ? 'warn' : 'ok'} />
                      <Stat label="Data supports" value={`${fmtInt(Math.round(plan.paramsFromData))} params`} />
                    </div>
                    <Callout tone={plan.dataLimited ? 'warn' : 'insight'}>
                      {plan.dataLimited ? (
                        <>
                          The budget would pay for a bigger model than your text can fill. Spending it
                          anyway buys a model that memorises. This is precisely the mistake the field made
                          for years before the scaling work landed, and the fix is more data rather than
                          more compute.
                        </>
                      ) : (
                        <>
                          Roughly twenty tokens per parameter is the rule that came out of the scaling
                          work. Training costs about six operations per parameter per token, so a budget
                          fixes both numbers at once and there is no free choice left.
                        </>
                      )}
                    </Callout>
                  </div>
                )}

                {s.id === 'architecture' && (
                  <div className="mb-3 grid gap-3">
                    <div className="grid gap-3 sm:grid-cols-2">
                      <Slider label="Width" value={dModel} min={16} max={96} step={8} onChange={setDModel} format={String} info={V.localModelSize} />
                      <Slider label="Blocks" value={nLayers} min={1} max={4} step={1} onChange={setNLayers} format={String} info={V.hiddenLayers} />
                      <Slider label="Heads" value={nHeads} min={1} max={8} step={1} onChange={setNHeads} format={String} info={V.labHeads} />
                      <Slider label="Vocabulary merges" value={merges} min={40} max={200} step={10} onChange={setMerges} format={String} info={V.labMerges} />
                    </div>
                    <Stat label="Parameters" value={fmtInt(params)} sub={`weights occupy ${prettyBytes(params * 4)} at full precision`} />
                  </div>
                )}

                {s.id === 'pretrain' && (
                  <div className="mb-3 grid gap-3">
                    <Slider label="Steps" value={steps} min={100} max={900} step={50} onChange={setSteps} format={String} info={V.labSteps} />
                    <div className="flex flex-wrap items-center gap-2">
                      <Btn variant="primary" onClick={build} disabled={running}>
                        {running ? 'Training...' : trainer ? 'Start again' : 'Start the run'}
                      </Btn>
                      {trainer && <span className="mono text-[11.5px]" style={{ color: 'var(--text-3)' }}>
                        step {fmtInt(trainer.step)} of {fmtInt(trainer.cfg.steps)}
                      </span>}
                    </div>
                    {trainer && <ProgressBar value={trainer.progress} />}
                    {trainer && trainer.history.length > 1 && (
                      <LineChart height={160} xLabel="step" yLabel="loss" logY
                        series={[
                          { id: 'train', label: 'training loss', color: 'var(--accent)', points: trainer.history.map((h) => ({ x: h.step, y: h.loss })) },
                          { id: 'val', label: 'held-out loss', color: 'var(--warn)', dashed: true, points: trainer.history.filter((h) => Number.isFinite(h.valLoss)).map((h) => ({ x: h.step, y: h.valLoss })) },
                        ]} />
                    )}
                  </div>
                )}

                {s.id === 'evaluate' && (
                  <div className="mb-3 grid gap-3">
                    <Slider label="The bar you committed to" value={bar} min={0} max={1} step={0.01}
                      onChange={setBar} format={(v) => fmtPct(v, 0)} info={V.evalTarget} />
                    <Btn variant="primary" onClick={runEvals} disabled={!trainer}>
                      Run the suite against the baselines
                    </Btn>
                    {evalRuns && (
                      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                        <Stat label="Your model" value={fmtPct(evalRuns.model.mean, 1)} tone="accent" />
                        <Stat label={`Best baseline (${evalRuns.best.predictorLabel})`} value={fmtPct(evalRuns.best.mean, 1)} />
                        <Stat label="The bar" value={fmtPct(bar, 0)} />
                      </div>
                    )}
                    <Callout tone="insight">
                      This gate is the one people negotiate with. If the model misses, the tempting move is
                      to lower the bar or change the metric, and it is exactly the move{' '}
                      <Link to="/evals" style={{ color: 'var(--accent)' }}>step 10</Link> exists to make
                      you notice. The honest options are a better model or a smaller claim.
                    </Callout>
                  </div>
                )}

                {s.id === 'safety' && (
                  <div className="mb-3 grid gap-3">
                    <Btn variant="primary" onClick={redTeam} disabled={!trainer}>
                      Generate and read what it produces
                    </Btn>
                    {sample && (
                      <div className="mono rounded-lg p-3 text-[12px] leading-relaxed"
                        style={{ background: 'var(--bg-2)', border: '1px solid var(--border)', color: 'var(--text-2)' }}>
                        {sample}
                      </div>
                    )}
                    <Field label="Words the deployed model must never produce" info={V.labBanned}>
                      <input value={banText} onChange={(e) => setBanText(e.target.value)} placeholder="optional, comma separated" className="mono w-full" />
                    </Field>
                    {banned.length > 0 && (
                      <div className="text-[11.5px]" style={{ color: 'var(--text-3)' }}>
                        {banned.map((b) => `${b.fragment} -> ${b.ids.length} entries`).join(' · ')}
                      </div>
                    )}
                    <Callout tone="warn">
                      At a real lab this review is done by people who did not build the model, and who can
                      stop the launch. That independence is the whole mechanism: nobody is a fair judge of
                      their own work, however honest they are.
                    </Callout>
                  </div>
                )}

                {s.id === 'launch' && (
                  <div className="mb-3 grid gap-3">
                    {cost && bound && (
                      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                        <Stat label="Tokens per second" value={fmt(bound.tokensPerSecond, 0)} sub={`on an ${gpu.label}`} />
                        <Stat label="Per million tokens" value={cost.usdPerMillionTokens < 0.01 ? 'under a cent' : `$${fmt(cost.usdPerMillionTokens, 2)}`} />
                        <Stat label="Conversations at once" value={Number.isFinite(cost.maxConcurrent) ? fmtInt(cost.maxConcurrent) : '—'} />
                      </div>
                    )}
                    <Btn variant="primary" onClick={ship} disabled={!trainer || shipped || !gLaunch.passed}>
                      {shipped ? 'Shipped' : 'Ship it'}
                    </Btn>
                    {shipped && (
                      <Callout tone="insight" title="It is live">
                        This model is now the one the rest of the walkthrough works on. Take it to{' '}
                        <Link to="/comms" style={{ color: 'var(--accent)' }}>step 09</Link> and look
                        inside it, or to{' '}
                        <Link to="/monitor" style={{ color: 'var(--accent)' }}>step 12</Link> and watch it
                        drift. Shipping is not the end of the programme, it is the point where the
                        maintenance starts.
                      </Callout>
                    )}
                  </div>
                )}

                {s.id !== 'launch' && (
                  <GateView g={g} signed={signed} onSign={() => sign(s.id)} label={`Sign off stage ${s.n}`} />
                )}
                {s.id === 'launch' && !shipped && <GateView g={g} signed={signed} onSign={() => sign(s.id)} label="Approve launch" />}
              </div>
            )}
          </Panel>
        );
      })}
    </div>
  );
}
