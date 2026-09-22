import { describe, expect, it } from 'vitest';
import { LMTrainer, DEFAULT_LM_TRAIN, type LMTrainConfig } from './lmTrainer';
import { DEFAULT_TCONFIG, type TransformerConfig } from './transformer';
import { getCorpus } from './corpus';

const TEXT = getCorpus('stories', 3);
const TCFG: TransformerConfig = { ...DEFAULT_TCONFIG, vocab: 0, dModel: 24, nHeads: 2, nLayers: 1, blockSize: 12, dFF: 48 };
const CFG: LMTrainConfig = { ...DEFAULT_LM_TRAIN, batchSeqs: 4, steps: 20 };

const make = () => new LMTrainer(TEXT, TCFG, CFG, 40);

function weights(t: LMTrainer): number[] {
  return Array.from(t.model.flatParams());
}

describe('sequence-level batching', () => {
  it('accumulating sequences one at a time matches doing the whole step at once', () => {
    // The loop was changed so it can pause between sequences rather than only
    // between steps. That must not change the arithmetic at all.
    const whole = make();
    const piecewise = make();

    for (let i = 0; i < 3; i++) whole.singleStep();

    let steps = 0;
    while (steps < 3) {
      if (piecewise.microStep() === 'step') steps++;
    }

    expect(piecewise.step).toBe(whole.step);
    expect(piecewise.tokensSeen).toBe(whole.tokensSeen);
    const a = weights(whole);
    const b = weights(piecewise);
    expect(b.length).toBe(a.length);
    for (let i = 0; i < a.length; i++) expect(b[i]).toBeCloseTo(a[i], 12);
  });

  it('reports how many sequences are still needed before the next update', () => {
    const t = make();
    expect(t.batchRemaining).toBe(4);
    t.microStep();
    expect(t.batchRemaining).toBe(3);
    t.microStep();
    t.microStep();
    expect(t.batchRemaining).toBe(1);
    expect(t.microStep()).toBe('step');
    expect(t.step).toBe(1);
  });

  it('does not touch the weights until a batch is complete', () => {
    // Constant schedule, so the assertion is about batching rather than about
    // a warmup schedule happening to scale the first step down.
    const t = new LMTrainer(TEXT, TCFG, { ...CFG, schedule: 'constant' }, 40);
    const before = weights(t);
    for (let i = 0; i < CFG.batchSeqs - 1; i++) expect(t.microStep()).toBe('seq');
    expect(weights(t)).toEqual(before);
    expect(t.step).toBe(0);
    expect(t.microStep()).toBe('step');
    expect(weights(t)).not.toEqual(before);
  });

  it('runSlice reports sequences, so a slice too short for a whole step is not mistaken for a stall', () => {
    const t = make();
    // A zero budget still does one sequence, and must report it as progress,
    // otherwise the UI would conclude training had finished and stop.
    const n = t.runSlice(0);
    expect(n).toBeGreaterThan(0);
    expect(t.step).toBe(0);
    expect(t.batchRemaining).toBeLessThan(CFG.batchSeqs);
  });

  it('runSlice keeps the partial batch across calls rather than discarding it', () => {
    const t = make();
    let guard = 0;
    while (t.step === 0 && guard++ < 50) t.runSlice(0);
    expect(t.step).toBe(1);
    // Exactly one batch worth of sequences was consumed for that step.
    expect(t.tokensSeen).toBe(CFG.batchSeqs * TCFG.blockSize);
  });

  it('reset clears a half-finished batch', () => {
    const t = make();
    t.microStep();
    t.microStep();
    expect(t.batchRemaining).toBe(2);
    t.reset();
    expect(t.batchRemaining).toBe(CFG.batchSeqs);
    expect(t.step).toBe(0);
    expect(t.tokensSeen).toBe(0);
  });

  it('stops cleanly at the configured step count', () => {
    const short = new LMTrainer(TEXT, TCFG, { ...CFG, steps: 2 }, 40);
    let guard = 0;
    while (short.status !== 'done' && guard++ < 500) short.microStep();
    expect(short.step).toBe(2);
    expect(short.status).toBe('done');
    expect(short.microStep()).toBe('done');
  });

  it('actually reduces the loss over a handful of steps', () => {
    const t = make();
    t.singleStep();
    const first = t.latest()!.loss;
    for (let i = 0; i < 15; i++) t.singleStep();
    const last = t.latest()!.loss;
    expect(last).toBeLessThan(first);
  });
});

describe('longer context windows', () => {
  it('a longer window trains without error and sees proportionally more tokens', () => {
    const wide: TransformerConfig = { ...TCFG, blockSize: 64 };
    const t = new LMTrainer(TEXT, wide, { ...CFG, batchSeqs: 2, steps: 3 }, 40);
    t.singleStep();
    expect(t.tokensSeen).toBe(2 * 64);
    expect(Number.isFinite(t.latest()!.loss)).toBe(true);
  });

  it('each position in a longer window still only sees earlier positions', () => {
    const wide: TransformerConfig = { ...TCFG, blockSize: 48 };
    const t = new LMTrainer(TEXT, wide, CFG, 40);
    const ids = t.trainIds.slice(0, 48);
    const trace = t.model.forward(ids).trace;
    for (const layer of trace.layers) {
      for (const head of layer.heads) {
        const T = head.att.rows;
        for (let i = 0; i < T; i++) {
          for (let j = i + 1; j < T; j++) expect(head.att.data[i * T + j]).toBeLessThan(1e-9);
        }
      }
    }
  });
});
