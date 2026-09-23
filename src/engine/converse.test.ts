import { describe, expect, it } from 'vitest';
import { Transformer, type TransformerConfig } from './transformer';
import { BPETokenizer } from './tokenizer';
import { getCorpus } from './corpus';
import { DEFAULT_CONVERSE, converse, footprintMB } from './converse';
import { buildStages, stageCount } from './replay';
import { matchTokens } from './steering';

const CFG: TransformerConfig = { vocab: 0, dModel: 16, nHeads: 2, nLayers: 2, blockSize: 12, dFF: 32 };

function build() {
  const tok = BPETokenizer.train(getCorpus('stories', 2), 90);
  const model = new Transformer({ ...CFG, vocab: tok.size }, 3);
  return { tok, model };
}

describe('holding a conversation and keeping the receipts', () => {
  const { tok, model } = build();

  it('keeps one full trace per generated token', () => {
    const c = converse(model, tok, 'the fox', { ...DEFAULT_CONVERSE, maxTokens: 6, seed: 2 });
    expect(c.steps.length).toBe(6);
    for (const s of c.steps) {
      expect(s.trace.tokens.length).toBe(s.context.length);
      expect(buildStages(s.trace, model.cfg).length).toBe(stageCount(model.cfg));
    }
  });

  it('the context grows by exactly one token each step', () => {
    const c = converse(model, tok, 'the fox', { ...DEFAULT_CONVERSE, maxTokens: 4, seed: 2 });
    for (let i = 1; i < c.steps.length; i++) {
      expect(c.steps[i].context.length).toBe(c.steps[i - 1].context.length + 1);
    }
  });

  it('each step is conditioned on the token the previous step chose', () => {
    const c = converse(model, tok, 'the fox', { ...DEFAULT_CONVERSE, maxTokens: 5, seed: 4 });
    for (let i = 1; i < c.steps.length; i++) {
      const ctx = c.steps[i].context;
      expect(ctx[ctx.length - 1]).toBe(c.steps[i - 1].chosen);
    }
  });

  it('never exceeds the context window, and says when it started dropping tokens', () => {
    const c = converse(model, tok, 'the fox sat on the mat', { ...DEFAULT_CONVERSE, maxTokens: 20, seed: 1 });
    for (const s of c.steps) expect(s.context.length).toBeLessThanOrEqual(CFG.blockSize);
    // Long enough to overflow, so at least one step must be flagged.
    expect(c.steps.some((s) => s.truncated)).toBe(true);
    // And once it starts, it never goes back.
    const first = c.steps.findIndex((s) => s.truncated);
    for (let i = first; i < c.steps.length; i++) expect(c.steps[i].truncated).toBe(true);
  });

  it('flags a prompt that was already too long on its own', () => {
    const long = 'the fox sat on the mat and waited for the evening to come again';
    const c = converse(model, tok, long, { ...DEFAULT_CONVERSE, maxTokens: 2, seed: 1 });
    expect(c.promptOverflowed).toBe(true);
    expect(c.steps[0].truncated).toBe(true);
  });

  it('the same seed gives the same reply, a different seed does not', () => {
    const a = converse(model, tok, 'the fox', { ...DEFAULT_CONVERSE, maxTokens: 10, seed: 7 });
    const b = converse(model, tok, 'the fox', { ...DEFAULT_CONVERSE, maxTokens: 10, seed: 7 });
    const c = converse(model, tok, 'the fox', { ...DEFAULT_CONVERSE, maxTokens: 10, seed: 8 });
    expect(b.reply).toBe(a.reply);
    expect(c.reply).not.toBe(a.reply);
  });

  it('the reply is what was added, and full is prompt plus reply', () => {
    const c = converse(model, tok, 'the fox', { ...DEFAULT_CONVERSE, maxTokens: 5, seed: 3 });
    expect(c.full.endsWith(c.reply)).toBe(true);
    expect(c.full.length).toBeGreaterThan(c.reply.length);
  });

  it('a blocked word is never generated', () => {
    const banned = new Set(matchTokens(tok, ['the']));
    const c = converse(model, tok, 'a fox', { ...DEFAULT_CONVERSE, maxTokens: 25, seed: 11, banned });
    for (const s of c.steps) expect(banned.has(s.chosen)).toBe(false);
  });

  it('temperature at the floor makes it deterministic without a seed change', () => {
    const a = converse(model, tok, 'the fox', { ...DEFAULT_CONVERSE, maxTokens: 8, temperature: 0.001, topK: 1, seed: 1 });
    const b = converse(model, tok, 'the fox', { ...DEFAULT_CONVERSE, maxTokens: 8, temperature: 0.001, topK: 1, seed: 999 });
    expect(b.reply).toBe(a.reply);
  });

  it('an empty prompt still produces something rather than throwing', () => {
    const c = converse(model, tok, '', { ...DEFAULT_CONVERSE, maxTokens: 3, seed: 1 });
    expect(c.steps.length).toBe(3);
    expect(c.reply.length).toBeGreaterThan(0);
  });

  it('stops early when it hits the stop token', () => {
    const first = converse(model, tok, 'the fox', { ...DEFAULT_CONVERSE, maxTokens: 12, seed: 5 });
    const target = first.steps[3].chosen;
    const c = converse(model, tok, 'the fox', { ...DEFAULT_CONVERSE, maxTokens: 12, seed: 5, stopAt: target });
    expect(c.steps.length).toBe(4);
  });

  it('estimates its own memory cost, growing with tokens and with model size', () => {
    const small = footprintMB(model, 12, 10);
    expect(small).toBeGreaterThan(0);
    expect(footprintMB(model, 12, 20)).toBeCloseTo(small * 2, 6);
    const big = new Transformer({ ...CFG, vocab: 100, dModel: 64, nLayers: 4 }, 1);
    expect(footprintMB(big, 12, 10)).toBeGreaterThan(small);
  });
});
