import { describe, expect, it } from 'vitest';
import { LMTrainer, DEFAULT_LM_TRAIN } from './lmTrainer';
import { DEFAULT_TCONFIG } from './transformer';
import { getCorpus } from './corpus';
import { modelPredictor, teacherForcedLoss } from './evals';
import {
  SCENARIO_INFO,
  buildWindow,
  corpusText,
  evaluateAlert,
  jsDivergence,
  psi,
  psiBand,
  referenceHistogram,
  retrainCase,
  schedule,
  thresholdSweep,
  tokenHistogram,
  type Scenario,
  type Window,
} from './monitor';

/** A model genuinely trained on stories, so drift toward weather is real. */
function trained(steps = 200) {
  const t = new LMTrainer(
    getCorpus('stories', 12),
    { ...DEFAULT_TCONFIG, dModel: 48, nHeads: 4, nLayers: 2, blockSize: 24, dFF: 96 },
    { ...DEFAULT_LM_TRAIN, steps },
    120,
  );
  for (let i = 0; i < steps; i++) t.singleStep(false);
  return t;
}

describe('distribution distances', () => {
  it('a distribution against itself has moved by nothing', () => {
    const h = tokenHistogram([1, 1, 2, 3, 3, 3], 8);
    expect(psi(h, h)).toBeCloseTo(0, 12);
    expect(jsDivergence(h, h)).toBeCloseTo(0, 12);
  });

  it('a histogram is a distribution', () => {
    const h = tokenHistogram([0, 1, 1, 2], 5);
    let sum = 0;
    for (const x of h) sum += x;
    expect(sum).toBeCloseTo(1, 12);
    expect(h[1]).toBeCloseTo(0.5, 12);
  });

  it('grows as the distributions separate', () => {
    const base = tokenHistogram([0, 0, 0, 1], 4);
    const near = tokenHistogram([0, 0, 1, 1], 4);
    const far = tokenHistogram([2, 2, 3, 3], 4);
    expect(psi(base, near)).toBeGreaterThan(0);
    expect(psi(base, far)).toBeGreaterThan(psi(base, near));
    expect(jsDivergence(base, far)).toBeGreaterThan(jsDivergence(base, near));
  });

  it('survives a category that appears in only one window', () => {
    // Unfloored, this is an infinity, and an infinite drift score on a
    // dashboard is indistinguishable from a broken dashboard.
    const a = tokenHistogram([0, 0, 0], 3);
    const b = tokenHistogram([2, 2, 2], 3);
    expect(Number.isFinite(psi(a, b))).toBe(true);
    expect(Number.isFinite(jsDivergence(a, b))).toBe(true);
  });

  it('the divergence stays inside its bound', () => {
    const a = tokenHistogram([0, 0, 0], 3);
    const b = tokenHistogram([2, 2, 2], 3);
    expect(jsDivergence(a, b)).toBeLessThanOrEqual(Math.log(2) + 1e-9);
  });

  it('is symmetric, unlike the raw divergence it is built from', () => {
    const a = tokenHistogram([0, 0, 1], 4);
    const b = tokenHistogram([1, 2, 2], 4);
    expect(jsDivergence(a, b)).toBeCloseTo(jsDivergence(b, a), 12);
  });

  it('reads a score against the conventional bands', () => {
    expect(psiBand(0.05).label).toBe('stable');
    expect(psiBand(0.2).label).toBe('moving');
    expect(psiBand(0.9).label).toBe('shifted');
  });
});

describe('traffic windows', () => {
  const t = trained();
  const familiar = corpusText('stories');
  const strange = corpusText('weather');
  const ref = referenceHistogram(t.tok, familiar);

  const clean = buildWindow(t.model, t.tok, familiar, strange, 0, 0, 12, 1, ref);
  const dirty = buildWindow(t.model, t.tok, familiar, strange, 1, 1, 12, 1, ref);

  it('produces the requests it was asked for, from the right sources', () => {
    expect(clean.requests.length).toBeGreaterThan(0);
    expect(clean.requests.every((r) => r.familiar)).toBe(true);
    expect(dirty.requests.every((r) => !r.familiar)).toBe(true);
  });

  it('the model is measurably more surprised by unfamiliar traffic', () => {
    // The central claim of the module, and it has to be real rather than
    // asserted: the proxy signal must actually move.
    expect(dirty.surprise).toBeGreaterThan(clean.surprise);
  });

  it('input drift registers on both distance measures', () => {
    expect(clean.psi).toBeLessThan(dirty.psi);
    expect(clean.jsd).toBeLessThan(dirty.jsd);
    expect(dirty.psi).toBeGreaterThan(0.25);
  });

  it('true quality really does fall on traffic the model was not built for', () => {
    expect(dirty.trueQuality).toBeLessThan(clean.trueQuality);
  });

  it('the proxy moves in the opposite direction to the quality it stands for', () => {
    // Which is the property that makes it usable: surprise up, quality down.
    expect(dirty.surprise - clean.surprise).toBeGreaterThan(0);
    expect(dirty.trueQuality - clean.trueQuality).toBeLessThan(0);
  });

  it('an unfamiliar alphabet shows up as characters the tokenizer cannot read', () => {
    const brackets = corpusText('brackets');
    const w = buildWindow(t.model, t.tok, familiar, brackets, 1, 2, 10, 3, ref);
    expect(w.oov).toBeGreaterThan(0);
  });

  it('is deterministic, so two runs of the same scenario compare', () => {
    const a = buildWindow(t.model, t.tok, familiar, strange, 0.5, 4, 10, 7, ref);
    const b = buildWindow(t.model, t.tok, familiar, strange, 0.5, 4, 10, 7, ref);
    expect(b.requests.map((r) => r.text)).toEqual(a.requests.map((r) => r.text));
    expect(b.surprise).toBeCloseTo(a.surprise, 12);
  });

  it('reports zero drift against no reference rather than pretending', () => {
    const w = buildWindow(t.model, t.tok, familiar, strange, 1, 0, 8, 1, null);
    expect(w.psi).toBe(0);
    expect(w.jsd).toBe(0);
  });
});

describe('scenarios', () => {
  it('each one has the shape its name promises', () => {
    const n = 20;
    const steady = schedule('steady', n);
    expect(steady.every((x) => x === 0)).toBe(true);

    const sudden = schedule('sudden', n);
    expect(sudden[0]).toBe(0);
    expect(sudden[n - 1]).toBeGreaterThan(0.5);
    // One step, not a ramp: only a single change point.
    const changes = sudden.filter((x, i) => i > 0 && x !== sudden[i - 1]).length;
    expect(changes).toBe(1);

    const gradual = schedule('gradual', n);
    for (let i = 1; i < n; i++) expect(gradual[i]).toBeGreaterThanOrEqual(gradual[i - 1]);
    expect(gradual[n - 1]).toBeGreaterThan(gradual[0]);

    const spike = schedule('spike', n);
    expect(spike[0]).toBe(0);
    expect(spike[n - 1]).toBe(0);
    expect(Math.max(...spike)).toBeGreaterThan(0.5);
  });

  it('every scenario is described', () => {
    for (const k of Object.keys(SCENARIO_INFO) as Scenario[]) {
      expect(SCENARIO_INFO[k].blurb.length).toBeGreaterThan(60);
    }
  });

  it('works at a single point without dividing by zero', () => {
    expect(schedule('gradual', 1)).toEqual([0]);
  });
});

describe('alerting', () => {
  /** Hand-built windows, so the arithmetic is checked rather than the model. */
  const mk = (drift: number, surprise: number, i: number): Window => ({
    index: i,
    requests: [],
    drift,
    surprise,
    psi: 0,
    jsd: 0,
    oov: 0,
    repetition: 0,
    trueQuality: 1 - drift,
  });
  // Five calm windows, then five drifted ones with a higher surprise.
  const windows = [...[0, 1, 2, 3, 4].map((i) => mk(0, 1, i)), ...[5, 6, 7, 8, 9].map((i) => mk(1, 3, i))];

  it('a well-placed threshold catches everything and cries wolf never', () => {
    const r = evaluateAlert(windows, 'surprise', 2);
    expect(r.caught).toBe(5);
    expect(r.missed).toBe(0);
    expect(r.falseAlarms).toBe(0);
    expect(r.delay).toBe(0);
  });

  it('too tight a threshold fires on perfectly normal traffic', () => {
    const r = evaluateAlert(windows, 'surprise', 0.5);
    expect(r.falseAlarms).toBe(5);
    expect(r.missed).toBe(0);
  });

  it('too loose a threshold never fires at all', () => {
    const r = evaluateAlert(windows, 'surprise', 99);
    expect(r.fired).toEqual([]);
    expect(r.missed).toBe(5);
    expect(r.caught).toBe(0);
    expect(r.delay).toBe(-1);
  });

  it('reports how long the drift ran before anyone noticed', () => {
    const late = [...windows];
    // Raise the bar so only the last two windows trip it.
    late[8] = { ...late[8], surprise: 9 };
    late[9] = { ...late[9], surprise: 9 };
    const r = evaluateAlert(late, 'surprise', 8);
    expect(r.driftBegan).toBe(5);
    expect(r.delay).toBe(3);
  });

  it('every window is accounted for exactly once', () => {
    const r = evaluateAlert(windows, 'surprise', 2);
    expect(r.caught + r.missed).toBe(r.drifted);
    expect(r.caught + r.falseAlarms).toBe(r.fired.length);
  });

  it('a sweep trades false alarms against misses monotonically', () => {
    const sweep = thresholdSweep(windows, 'surprise', 20);
    expect(sweep.length).toBe(21);
    for (let i = 1; i < sweep.length; i++) {
      // Raising the bar can only reduce alarms and only increase misses.
      expect(sweep[i].falseAlarms).toBeLessThanOrEqual(sweep[i - 1].falseAlarms);
      expect(sweep[i].missed).toBeGreaterThanOrEqual(sweep[i - 1].missed);
    }
  });

  it('sweeping an empty history returns nothing rather than throwing', () => {
    expect(thresholdSweep([], 'surprise')).toEqual([]);
  });

  it('never fires on a signal that is not a number', () => {
    const broken = [mk(0, NaN, 0), mk(1, NaN, 1)];
    const r = evaluateAlert(broken, 'surprise', 0);
    expect(r.fired).toEqual([]);
  });
});

describe('the retrain decision', () => {
  it('pays back quickly when the loss is large', () => {
    const c = retrainCase(0.5, 1, 100_000, 0.01, 500);
    expect(c.degradation).toBeCloseTo(0.5, 9);
    expect(c.lossPerDay).toBeCloseTo(500, 6);
    expect(c.paybackDays).toBeCloseTo(1, 6);
    expect(c.worthIt).toBe(true);
  });

  it('is not worth it when almost nothing has been lost', () => {
    const c = retrainCase(0.99, 1, 1000, 0.01, 5000);
    expect(c.worthIt).toBe(false);
    expect(c.paybackDays).toBeGreaterThan(30);
  });

  it('never pays back when nothing has degraded at all', () => {
    const c = retrainCase(1, 1, 1e6, 1, 100);
    expect(c.degradation).toBe(0);
    expect(c.paybackDays).toBe(Infinity);
    expect(c.worthIt).toBe(false);
  });

  it('treats an improvement as no degradation rather than a negative loss', () => {
    // A model that got better should not produce a negative bill.
    const c = retrainCase(1.2, 1, 1000, 1, 100);
    expect(c.degradation).toBe(0);
    expect(c.lossPerDay).toBe(0);
  });

  it('scales the loss with traffic, which is what makes it a business decision', () => {
    const small = retrainCase(0.8, 1, 1_000, 0.05, 2000);
    const large = retrainCase(0.8, 1, 1_000_000, 0.05, 2000);
    expect(large.lossPerDay).toBeCloseTo(small.lossPerDay * 1000, 3);
    expect(small.worthIt).toBe(false);
    expect(large.worthIt).toBe(true);
  });
});

describe('the single-pass scoring shortcut', () => {
  it('agrees with walking the sequence one token at a time', () => {
    // buildWindow scores every position from one forward pass rather than
    // one pass per token. That is roughly ten times less work, and it is
    // only legitimate if the numbers are the same.
    const t = trained(20);
    const ids = t.tok.encode('the quiet fox sat by the river').slice(0, t.model.cfg.blockSize);
    const slow = teacherForcedLoss(modelPredictor(t.model), ids.slice(0, 1), ids.slice(1));
    const { trace } = t.model.forward(ids);
    const fast = t.model.loss(trace, ids.slice(1));
    expect(fast).toBeCloseTo(slow, 9);
  });
});

describe('end to end, on a really trained model', () => {
  it('a gradual drift raises surprise while quality falls', () => {
    // The whole module in one assertion: the signal you can see moves in
    // step with the thing you cannot.
    const t = trained(200);
    const familiar = corpusText('stories');
    const strange = corpusText('weather');
    const ref = referenceHistogram(t.tok, familiar);
    const sched = schedule('gradual', 6);
    const windows = sched.map((d, i) => buildWindow(t.model, t.tok, familiar, strange, d, i, 10, 5, ref));

    const first = windows[0];
    const last = windows[windows.length - 1];
    expect(last.surprise).toBeGreaterThan(first.surprise);
    expect(last.psi).toBeGreaterThan(first.psi);
    expect(last.trueQuality).toBeLessThan(first.trueQuality);
    // And the proxy leads: it has already moved appreciably by the midpoint,
    // which is the whole reason it is worth watching.
    const mid = windows[Math.floor(windows.length / 2)];
    expect(mid.surprise).toBeGreaterThan(first.surprise);
    // A real training run plus six windows of real traffic takes longer than
    // the default budget for a unit test, and is worth the wall time.
  }, 60000);
});
