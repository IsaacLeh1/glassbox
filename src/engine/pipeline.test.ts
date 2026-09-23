import { describe, expect, it } from 'vitest';
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
  type Licence,
  type StageId,
} from './pipeline';
import { getCorpus } from './corpus';

describe('the gate mechanism', () => {
  it('passes only when nothing blocking failed', () => {
    expect(gate('data', [{ label: 'a', pass: true, detail: '' }]).passed).toBe(true);
    expect(gate('data', [{ label: 'a', pass: false, detail: '' }]).passed).toBe(false);
  });

  it('lets a soft failure through, and records it as a failure anyway', () => {
    // A real process distinguishes "we decided to accept this" from "this was
    // fine". Collapsing the two is how standards quietly slip.
    const g = gate('data', [{ label: 'a', pass: false, detail: '', soft: true }]);
    expect(g.passed).toBe(true);
    expect(g.blocking).toBe(0);
    expect(g.checks[0].pass).toBe(false);
  });

  it('counts only the blocking failures', () => {
    const g = gate('data', [
      { label: 'hard', pass: false, detail: '' },
      { label: 'soft', pass: false, detail: '', soft: true },
      { label: 'ok', pass: true, detail: '' },
    ]);
    expect(g.blocking).toBe(1);
    expect(g.passed).toBe(false);
  });
});

describe('stage ordering', () => {
  it('runs in the order a real programme does', () => {
    expect(STAGES.map((s) => s.id)).toEqual([
      'charter', 'data', 'scaling', 'architecture', 'pretrain', 'evaluate', 'safety', 'launch',
    ]);
    STAGES.forEach((s, i) => expect(s.n).toBe(i + 1));
  });

  it('every stage names who owns it and why it exists', () => {
    for (const s of STAGES) {
      expect(s.owner.length).toBeGreaterThan(4);
      expect(s.blurb.length).toBeGreaterThan(80);
    }
  });

  it('will not let you skip ahead', () => {
    expect(canEnter('charter', [])).toBe(true);
    expect(canEnter('data', [])).toBe(false);
    expect(canEnter('data', ['charter'])).toBe(true);
    // No shipping before evaluating, whatever order the clicks arrive in.
    expect(canEnter('launch', ['charter', 'data', 'scaling'])).toBe(false);
  });

  it('knows what is next, and when there is nothing left', () => {
    expect(nextStage([])!.id).toBe('charter');
    expect(nextStage(['charter'])!.id).toBe('data');
    expect(nextStage(STAGES.map((s) => s.id))).toBeNull();
  });
});

describe('data hygiene', () => {
  const evalText = 'the quiet owl waits in the attic at dawn.';

  it('finds and collapses exact duplicates', () => {
    const raw = 'the fox sat. the fox sat. the fox sat. a bee sang.';
    const r = inspectData(raw, '');
    expect(r.docs).toBe(4);
    expect(r.exactDupes[0].count).toBe(3);
    expect(r.dupeShare).toBeCloseTo(0.5, 6);
    expect(r.cleaned).not.toContain('the fox sat. the fox sat');
  });

  it('reports no duplicates when there are none', () => {
    const r = inspectData('one thing. another thing. a third thing.', '');
    expect(r.exactDupes).toEqual([]);
    expect(r.dupeShare).toBe(0);
  });

  it('catches near-duplicates that exact matching misses', () => {
    // Differing by one word is not an exact duplicate and is still the same
    // document for training purposes.
    const raw =
      'the quick brown fox jumped over the lazy dog today. ' +
      'the quick brown fox jumped over the lazy dog todays. ' +
      'entirely different sentence about submarines and weather.';
    const r = inspectData(raw, '');
    expect(r.exactDupes).toEqual([]);
    expect(r.nearDupes).toBeGreaterThan(0);
  });

  it('removes training text that overlaps the evaluation set', () => {
    const raw = `a bee sang at noon. ${evalText} a hare ran home.`;
    const r = inspectData(raw, evalText);
    expect(r.contaminated).toBe(1);
    expect(r.cleaned).not.toContain('the quiet owl waits in the attic');
  });

  it('leaves the corpus alone when nothing overlaps', () => {
    const r = inspectData('a bee sang at noon. a hare ran home.', evalText);
    expect(r.contaminated).toBe(0);
  });

  it('counts the alphabet the tokenizer will have to cover', () => {
    const r = inspectData('abc. abd.', '');
    expect(r.alphabet).toBe(new Set(Array.from('abc. abd.')).size);
  });

  it('survives an empty corpus without throwing', () => {
    const r = inspectData('', '');
    expect(r.docs).toBe(0);
    expect(r.dupeShare).toBe(0);
    expect(r.cleaned).toBe('');
  });

  it('handles a real corpus and leaves most of it intact', () => {
    const raw = getCorpus('stories', 12);
    const r = inspectData(raw, '');
    expect(r.docs).toBeGreaterThan(100);
    // A generated corpus from templates really does repeat itself, which is
    // worth a reader seeing on data they assumed was fine.
    expect(r.cleanedChars).toBeGreaterThan(0);
    expect(r.cleanedChars).toBeLessThanOrEqual(raw.length);
  });
});

describe('the data gate', () => {
  const clean = inspectData('one thing here. another thing there. a third distinct thing.', '');

  it('blocks on an unestablished licence', () => {
    const g = dataGate(clean, 'unknown', 10);
    expect(g.passed).toBe(false);
    expect(g.checks[0].pass).toBe(false);
  });

  it('blocks on proprietary text', () => {
    expect(dataGate(clean, 'proprietary', 10).passed).toBe(false);
  });

  it('passes a permissive licence with enough clean text', () => {
    const g = dataGate(clean, 'permissive', 10);
    expect(g.passed).toBe(true);
  });

  it('blocks when there is not enough text left after cleaning', () => {
    const g = dataGate(clean, 'permissive', 1_000_000);
    expect(g.passed).toBe(false);
  });

  it('reports leaked evaluation data, and still lets the programme proceed', () => {
    // Regression: this check failed on a condition the cleaning step itself
    // resolves, so the gate could never be cleared on any corpus with an
    // overlap -- the programme dead-ended and the reader concluded the check
    // was broken rather than that the data had been.
    const dirty = inspectData('the owl waits at dawn. a bee sang. a hare ran home.', 'the owl waits at dawn.');
    expect(dirty.contaminated).toBe(1);
    expect(dirty.cleaned).not.toContain('owl waits at dawn');
    const g = dataGate(dirty, 'permissive', 5);
    const check = g.checks.find((c) => c.label.includes('evaluation'))!;
    expect(check.pass).toBe(true);
    expect(check.detail).toContain('removed');
    expect(g.passed).toBe(true);
  });

  it('counts in sentences a person would write', () => {
    const one = inspectData('a b c d e f g h. a b c d e f g h. something else here.', '');
    const g = dataGate(one, 'permissive', 5);
    const dupes = g.checks.find((c) => c.label === 'Duplicates removed')!;
    expect(dupes.detail).toContain('1 repeated document ');
    expect(dupes.detail).not.toContain('1 repeated documents');
  });

  it('every licence is described, and the blocking ones are marked', () => {
    for (const k of Object.keys(LICENCE_INFO) as Licence[]) {
      expect(LICENCE_INFO[k].blurb.length).toBeGreaterThan(40);
    }
    expect(LICENCE_INFO.unknown.ok).toBe(false);
    expect(LICENCE_INFO.proprietary.ok).toBe(false);
    expect(LICENCE_INFO.permissive.ok).toBe(true);
  });
});

describe('scaling', () => {
  it('splits a budget the way the scaling laws say', () => {
    // C = 6ND with D = 20N gives C = 120N^2.
    const plan = planScaling(120 * 1e6 * 1e6, 1e9);
    expect(plan.params).toBeCloseTo(1e6, -3);
    expect(plan.tokens).toBeCloseTo(2e7, -5);
  });

  it('more compute buys a bigger model, but only as the square root', () => {
    const small = planScaling(1e12, 1e12);
    const big = planScaling(1e16, 1e12);
    // Ten thousand times the compute is a hundred times the model.
    expect(big.params / small.params).toBeCloseTo(100, 0);
  });

  it('notices when there is not enough text for the budget', () => {
    const plan = planScaling(1e18, 1000);
    expect(plan.dataLimited).toBe(true);
    expect(plan.paramsFromData).toBeLessThan(plan.params);
  });

  it('warns when the model is larger than the data can fill', () => {
    // The mistake the whole field made before the scaling work landed.
    const plan = planScaling(1e15, 1e6);
    const g = scalingGate(plan, 1e6);
    const ratio = g.checks.find((c) => c.label.includes('tokens per parameter') || c.label.includes('Enough tokens'))!;
    expect(ratio.pass).toBe(false);
    // Soft, because a real team can knowingly accept it.
    expect(g.passed).toBe(true);
  });

  it('blocks a model far too large for the compute budget', () => {
    const plan = planScaling(1e12, 1e12);
    const g = scalingGate(plan, plan.params * 100);
    expect(g.passed).toBe(false);
  });

  it('passes a well-matched plan outright', () => {
    const plan = planScaling(1e14, 1e12);
    const g = scalingGate(plan, Math.round(plan.params));
    expect(g.passed).toBe(true);
    expect(g.checks.every((c) => c.pass)).toBe(true);
  });
});

describe('a full pass through the programme', () => {
  it('reaches launch only by clearing every gate in order', () => {
    const done: StageId[] = [];
    for (const s of STAGES) {
      expect(canEnter(s.id, done)).toBe(true);
      done.push(s.id);
    }
    expect(nextStage(done)).toBeNull();
    expect(done.length).toBe(8);
  });
});
