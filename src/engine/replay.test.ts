import { describe, expect, it } from 'vitest';
import { Transformer, type TransformerConfig } from './transformer';
import { BPETokenizer } from './tokenizer';
import { getCorpus } from './corpus';
import { bandsFor, buildStages, stageCount } from './replay';
import { Finetuner, buildWindows, checkVocab, showChar } from './finetune';

const CFG: TransformerConfig = { vocab: 0, dModel: 16, nHeads: 2, nLayers: 2, blockSize: 16, dFF: 32 };

function build() {
  const tok = BPETokenizer.train(getCorpus('stories', 2), 90);
  const model = new Transformer({ ...CFG, vocab: tok.size }, 3);
  return { tok, model };
}

describe('replaying a forward pass', () => {
  const { tok, model } = build();
  const ids = tok.encode('the fox sat');
  const { trace } = model.forward(ids);
  const stages = buildStages(trace, model.cfg);

  it('produces exactly the number of stages the config predicts', () => {
    // The count has to be derivable without running the model, because the UI
    // sizes the scrubber before the first pass has happened.
    expect(stages.length).toBe(stageCount(model.cfg));
  });

  it('starts at the input and ends at the choice', () => {
    expect(stages[0].kind).toBe('tokens');
    expect(stages[stages.length - 1].kind).toBe('sample');
  });

  it('visits the blocks in order, and every head within a block', () => {
    const layers = stages.filter((s) => s.layer >= 0).map((s) => s.layer);
    // Non-decreasing: never jumps back to an earlier block.
    for (let i = 1; i < layers.length; i++) expect(layers[i]).toBeGreaterThanOrEqual(layers[i - 1]);
    for (let l = 0; l < CFG.nLayers; l++) {
      const heads = new Set(stages.filter((s) => s.layer === l && s.head >= 0).map((s) => s.head));
      expect(heads.size).toBe(CFG.nHeads);
    }
  });

  it('every matrix has the shape its column space claims', () => {
    const T = ids.length;
    for (const s of stages) {
      if (!s.matrix) continue;
      expect(s.matrix.rows).toBe(T);
      const want = {
        model: CFG.dModel,
        head: CFG.dModel / CFG.nHeads,
        ff: CFG.dFF,
        positions: T,
        vocab: tok.size,
        none: -1,
      }[s.colSpace];
      expect(s.matrix.cols).toBe(want);
    }
  });

  it('hands out the model\'s own matrices, not copies', () => {
    // If these were copies, editing a weight and re-running would show stale
    // numbers in the scrubber.
    const emb = stages.find((s) => s.id === 'embedded')!;
    expect(emb.matrix).toBe(trace.embedded);
    const logits = stages.find((s) => s.id === 'logits')!;
    expect(logits.matrix).toBe(trace.logits);
  });

  it('marks exactly the stages that are the residual stream', () => {
    const res = stages.filter((s) => s.isResidual);
    // One at the start, then two per block.
    expect(res.length).toBe(1 + CFG.nLayers * 2);
    expect(res[0].id).toBe('embedded');
  });

  it('reports how much each block moved the stream, as a finite fraction', () => {
    for (const s of stages.filter((x) => x.delta !== undefined)) {
      expect(Number.isFinite(s.delta!)).toBe(true);
      expect(s.delta!).toBeGreaterThanOrEqual(0);
    }
  });

  it('an untrained block still moves the stream by something', () => {
    const d = stages.find((s) => s.id === 'L0.afterAttn')!.delta!;
    expect(d).toBeGreaterThan(0);
  });

  it('refuses a sequence longer than the model has positions for', () => {
    // Regression: reading past the end of the position table returned
    // undefined, so the whole pass silently became NaN and every delta
    // reported a flat zero.
    const long = new Array(CFG.blockSize + 1).fill(0);
    expect(() => model.forward(long)).toThrow(/positions/);
    // The windowed entry points stay safe.
    expect(() => model.predictNext(long)).not.toThrow();
  });

  it('every stage explains itself at all three depths', () => {
    for (const s of stages) {
      expect(s.plain.length).toBeGreaterThan(40);
      expect(s.math.length).toBeGreaterThan(0);
      expect(s.code.length).toBeGreaterThan(0);
      expect(s.title.length).toBeGreaterThan(0);
    }
  });

  it('gives every stage a unique id', () => {
    expect(new Set(stages.map((s) => s.id)).size).toBe(stages.length);
  });

  it('groups the timeline into contiguous bands covering everything', () => {
    const bands = bandsFor(stages);
    expect(bands[0].from).toBe(0);
    expect(bands[bands.length - 1].to).toBe(stages.length - 1);
    for (let i = 1; i < bands.length; i++) expect(bands[i].from).toBe(bands[i - 1].to + 1);
    expect(bands.some((b) => b.label === 'Block 0')).toBe(true);
    expect(bands.some((b) => b.label === 'Block 1')).toBe(true);
  });

  it('handles a single token without producing a degenerate stage list', () => {
    const one = model.forward([ids[0]]);
    const s = buildStages(one.trace, model.cfg);
    expect(s.length).toBe(stageCount(model.cfg));
    expect(s.find((x) => x.id === 'L0.h0.att')!.matrix!.rows).toBe(1);
  });
});

describe('teaching a trained model something new', () => {
  it('names the characters the vocabulary cannot represent', () => {
    const { tok } = build();
    const v = checkVocab(tok, 'the fox ☃ sat');
    // A snowman was never in the corpus, so it is dropped on the way in.
    expect(v.dropped).toContain('☃');
    expect(v.kept).toBeLessThan(v.total);
  });

  it('reports a clean bill when every character is known', () => {
    const { tok } = build();
    const v = checkVocab(tok, 'the fox sat');
    expect(v.dropped).toEqual([]);
    expect(v.kept).toBe(v.total);
  });

  it('writes invisible dropped characters so they can actually be read', () => {
    // A dropped space or newline rendered as itself makes the warning look
    // empty and broken.
    expect(showChar(' ')).toContain('space');
    expect(showChar('\n')).toContain('newline');
    expect(showChar('\t')).toContain('tab');
    expect(showChar('\u0007')).toContain('control');
    expect(showChar('q')).toBe('q');
  });

  it('does not blame the user for a separator it added itself', () => {
    // Regression: the examples were joined with a newline before checking, so
    // a model whose corpus had no newline reported one as dropped even though
    // nothing the user typed contained it.
    const tok = BPETokenizer.train('the fox sat on the mat', 20);
    expect(tok.index.has('\n')).toBe(false);
    const model = new Transformer({ ...CFG, vocab: tok.size }, 1);
    const ft = new Finetuner(model, tok, ['the fox', 'the mat'], DEFAULT_EMPTY);
    expect(ft.vocab.dropped).toEqual([]);
  });

  it('cuts long examples into overlapping windows and skips useless ones', () => {
    const { tok } = build();
    const ws = buildWindows(tok, ['the fox sat on the mat and waited for the evening', '', 'a'], 8);
    expect(ws.length).toBeGreaterThan(1);
    for (const w of ws) {
      expect(w.x.length).toBe(w.y.length);
      expect(w.x.length).toBeGreaterThanOrEqual(1);
      expect(w.x.length).toBeLessThanOrEqual(8);
      // Next-token targets: y is x shifted by one.
      expect(w.from).toBe(0);
    }
  });

  it('lowers the loss on what it is taught', () => {
    const { tok, model } = build();
    const lesson = ['the fox sat on the mat.'];
    const ft = new Finetuner(model, tok, lesson, { lr: 0.01, epochs: 20, clipNorm: 1 });
    expect(ft.windows.length).toBeGreaterThan(0);
    while (ft.microStep()) {
      /* run to completion */
    }
    expect(ft.evalNew()).toBeLessThan(ft.baselineNew);
  });

  it('can be undone exactly, weight for weight', () => {
    const { tok, model } = build();
    const snapshot = Array.from(model.flatParams());
    const ft = new Finetuner(model, tok, ['the fox sat on the mat.'], { lr: 0.02, epochs: 5, clipNorm: 1 });
    while (ft.microStep()) {
      /* run */
    }
    expect(Array.from(model.flatParams())).not.toEqual(snapshot);
    ft.revert();
    expect(Array.from(model.flatParams())).toEqual(snapshot);
  });

  it('measures drift away from the starting weights', () => {
    const { tok, model } = build();
    const ft = new Finetuner(model, tok, ['the fox sat on the mat.'], { lr: 0.02, epochs: 4, clipNorm: 1 });
    expect(ft.drift()).toBe(0);
    while (ft.microStep()) {
      /* run */
    }
    expect(ft.drift()).toBeGreaterThan(0);
  });

  it('tracks the original corpus alongside the lesson, so forgetting is visible', () => {
    const { tok, model } = build();
    const original = tok.encode(getCorpus('stories', 2));
    const ft = new Finetuner(model, tok, ['zzz zzz zzz zzz zzz'], { lr: 0.05, epochs: 12, clipNorm: 1 }, original);
    expect(Number.isFinite(ft.baselineOld)).toBe(true);
    while (ft.microStep()) {
      /* run */
    }
    expect(ft.history.length).toBe(12);
    for (const h of ft.history) expect(Number.isFinite(h.oldLoss)).toBe(true);
  });

  it('does nothing at all when there is nothing teachable', () => {
    const { tok, model } = build();
    const before = Array.from(model.flatParams());
    const ft = new Finetuner(model, tok, ['', ' '], DEFAULT_EMPTY);
    expect(ft.windows.length).toBe(0);
    expect(ft.done).toBe(true);
    expect(ft.microStep()).toBe(false);
    expect(Array.from(model.flatParams())).toEqual(before);
  });
});

const DEFAULT_EMPTY = { lr: 0.01, epochs: 3, clipNorm: 1 };
