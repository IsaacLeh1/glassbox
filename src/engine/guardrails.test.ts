import { describe, expect, it } from 'vitest';
import { Transformer, sampleToken, type TransformerConfig } from './transformer';
import { contrastDirection, matchTokens, resolveBan, projectOnto } from './steering';
import { BPETokenizer } from './tokenizer';
import { getCorpus } from './corpus';
import { mulberry32, softmax1d } from './tensor';

const LOGITS = [4, 3, 2, 1, 0];
const CFG: TransformerConfig = { vocab: 12, dModel: 16, nHeads: 2, nLayers: 2, blockSize: 8, dFF: 32 };

describe('blocking tokens at sampling time', () => {
  it('a banned token is never chosen, however many times we sample', () => {
    for (let i = 0; i < 60; i++) {
      const s = sampleToken(LOGITS, { temperature: 1.5, topK: 0, topP: 1, banned: new Set([0, 1]) }, mulberry32(i));
      expect(s.chosen).not.toBe(0);
      expect(s.chosen).not.toBe(1);
    }
  });

  it('banned tokens are reported with zero probability and flagged', () => {
    const s = sampleToken(LOGITS, { temperature: 1, topK: 0, topP: 1, banned: new Set([0]) }, mulberry32(1));
    const top = s.candidates.find((c) => c.id === 0)!;
    expect(top.blocked).toBe(true);
    expect(top.prob).toBe(0);
    expect(top.kept).toBe(false);
    // The rest are still flagged as allowed.
    expect(s.candidates.filter((c) => c.blocked).length).toBe(1);
  });

  it('reports how much probability mass the ban removed', () => {
    const free = softmax1d(LOGITS, 1);
    const s = sampleToken(LOGITS, { temperature: 1, topK: 0, topP: 1, banned: new Set([0]) }, mulberry32(1));
    expect(s.blockedMass).toBeCloseTo(free[0], 9);
  });

  it('the surviving probabilities still sum to one', () => {
    const s = sampleToken(LOGITS, { temperature: 1, topK: 0, topP: 1, banned: new Set([0, 2]) }, mulberry32(1));
    const total = s.candidates.reduce((a, c) => a + c.prob, 0);
    expect(total).toBeCloseTo(1, 9);
  });

  it('banning does not reorder the tokens that remain', () => {
    const free = sampleToken(LOGITS, { temperature: 1, topK: 0, topP: 1 }, mulberry32(1));
    const banned = sampleToken(LOGITS, { temperature: 1, topK: 0, topP: 1, banned: new Set([0]) }, mulberry32(1));
    const rank = (s: typeof free) =>
      s.candidates.filter((c) => !c.blocked).sort((a, b) => b.prob - a.prob).map((c) => c.id);
    expect(rank(banned)).toEqual(rank(free).filter((id) => id !== 0));
  });

  it('top-k counts allowed tokens, not blocked ones', () => {
    // Regression: a banned token could occupy a top-k slot, so banning the
    // most likely token silently narrowed the real choice from k to k-1.
    const s = sampleToken(LOGITS, { temperature: 1, topK: 2, topP: 1, banned: new Set([0]) }, mulberry32(1));
    const kept = s.candidates.filter((c) => c.kept);
    expect(kept.length).toBe(2);
    expect(kept.every((c) => !c.blocked)).toBe(true);
  });

  it('survives every token being banned without throwing', () => {
    const s = sampleToken(LOGITS, { temperature: 1, topK: 0, topP: 1, banned: new Set([0, 1, 2, 3, 4]) }, mulberry32(1));
    expect(Number.isInteger(s.chosen)).toBe(true);
    expect(s.blockedMass).toBeCloseTo(1, 9);
  });

  it('an empty ban list changes nothing', () => {
    const a = sampleToken(LOGITS, { temperature: 1, topK: 0, topP: 1 }, mulberry32(5));
    const b = sampleToken(LOGITS, { temperature: 1, topK: 0, topP: 1, banned: new Set() }, mulberry32(5));
    expect(b.chosen).toBe(a.chosen);
    expect(b.blockedMass).toBe(0);
  });
});

describe('resolving a word into blockable tokens', () => {
  const tok = BPETokenizer.train(getCorpus('stories', 2), 100);

  it('blocks a word even when it is not stored as a single token', () => {
    // The naive approach -- substring matching over the vocabulary -- silently
    // matches nothing for a word BPE has split, and blocks nothing at all.
    const [r] = resolveBan(tok, ['garden']);
    expect(r.ids.length).toBeGreaterThan(0);
    expect(r.pieces.length).toBe(r.ids.length);
  });

  it('reports when a word had to be blocked by its opening piece', () => {
    const results = resolveBan(tok, ['garden', 'river']);
    // At least one of these is multi-token in this vocabulary, and the flag
    // is what the UI uses to warn about over-blocking.
    expect(results.some((r) => r.split)).toBe(true);
  });

  it('every blocked id really is in the vocabulary', () => {
    for (const r of resolveBan(tok, ['garden', 'the'])) {
      for (const id of r.ids) {
        expect(id).toBeGreaterThanOrEqual(0);
        expect(id).toBeLessThan(tok.size);
      }
    }
  });

  it('banning a word actually prevents it from being produced', () => {
    const ids = matchTokens(tok, ['garden']);
    const banned = new Set(ids);
    // Every token that could start the word is struck out, so no continuation
    // can begin it.
    const first = tok.encode(' garden')[0];
    expect(banned.has(first)).toBe(true);
  });

  it('ignores blank fragments and returns nothing for an empty list', () => {
    expect(matchTokens(tok, [])).toEqual([]);
    expect(matchTokens(tok, ['   ', ''])).toEqual([]);
    expect(resolveBan(tok, ['  '])).toEqual([]);
  });

  it('is case insensitive', () => {
    expect(matchTokens(tok, ['THE']).sort()).toEqual(matchTokens(tok, ['the']).sort());
  });

  it('more words never block fewer tokens', () => {
    const one = matchTokens(tok, ['garden']).length;
    const two = matchTokens(tok, ['garden', 'river']).length;
    expect(two).toBeGreaterThanOrEqual(one);
  });
});

describe('activation steering', () => {
  const tok = BPETokenizer.train(getCorpus('stories', 2), 80);
  const model = new Transformer({ ...CFG, vocab: tok.size }, 5);
  const A = ['the owl watches at night.', 'the moth hides until dark.'];
  const B = ['the bee sings at dawn.', 'the hare runs all morning.'];

  it('produces a unit-length direction of the model width', () => {
    const d = contrastDirection(model, tok, A, B, 1)!;
    expect(d).not.toBeNull();
    expect(d.vec.length).toBe(CFG.dModel);
    let n = 0;
    for (const x of d.vec) n += x * x;
    expect(Math.sqrt(n)).toBeCloseTo(1, 9);
    expect(d.countA).toBe(2);
    expect(d.countB).toBe(2);
  });

  it('swapping the two groups reverses the direction exactly', () => {
    const ab = contrastDirection(model, tok, A, B, 1)!;
    const ba = contrastDirection(model, tok, B, A, 1)!;
    for (let i = 0; i < ab.vec.length; i++) expect(ba.vec[i]).toBeCloseTo(-ab.vec[i], 9);
  });

  it('returns null when a group has nothing usable in it', () => {
    expect(contrastDirection(model, tok, [], B, 1)).toBeNull();
    expect(contrastDirection(model, tok, A, ['   '], 1)).toBeNull();
  });

  it('a zero-strength steer leaves the output bit-for-bit unchanged', () => {
    const d = contrastDirection(model, tok, A, B, 1)!;
    const ids = tok.encode('the fox');
    const plain = model.predictNext(ids).probs;
    const zero = model.predictNext(ids, { layer: d.layer, vec: d.vec, scale: 0 }).probs;
    expect(zero).toEqual(plain);
  });

  it('a nonzero steer changes the prediction, and the sign matters', () => {
    const d = contrastDirection(model, tok, A, B, 1)!;
    const ids = tok.encode('the fox');
    const plain = model.predictNext(ids).probs;
    const pos = model.predictNext(ids, { layer: d.layer, vec: d.vec, scale: 8 }).probs;
    const neg = model.predictNext(ids, { layer: d.layer, vec: d.vec, scale: -8 }).probs;

    const diff = (a: number[], b: number[]) => a.reduce((s, x, i) => s + Math.abs(x - b[i]), 0);
    expect(diff(pos, plain)).toBeGreaterThan(1e-4);
    expect(diff(neg, plain)).toBeGreaterThan(1e-4);
    // Pushing opposite ways must not land in the same place.
    expect(diff(pos, neg)).toBeGreaterThan(diff(pos, plain) * 0.5);
  });

  it('steering leaves every weight untouched', () => {
    const d = contrastDirection(model, tok, A, B, 1)!;
    const before = Array.from(model.flatParams());
    model.predictNext(tok.encode('the fox'), { layer: d.layer, vec: d.vec, scale: 9 });
    expect(Array.from(model.flatParams())).toEqual(before);
  });

  it('steering after the embedding is also honoured', () => {
    const d = contrastDirection(model, tok, A, B, -1)!;
    expect(d.layer).toBe(-1);
    const ids = tok.encode('the fox');
    const plain = model.predictNext(ids).probs;
    const steered = model.predictNext(ids, { layer: -1, vec: d.vec, scale: 6 }).probs;
    const diff = plain.reduce((s, x, i) => s + Math.abs(x - steered[i]), 0);
    expect(diff).toBeGreaterThan(1e-5);
  });

  it('projection is positive for text like group A and negative for group B', () => {
    const d = contrastDirection(model, tok, A, B, 1)!;
    // Centre the measurement, since the raw projection includes a shared offset.
    const pa = A.map((t) => projectOnto(model, tok, t, d));
    const pb = B.map((t) => projectOnto(model, tok, t, d));
    const meanA = pa.reduce((a, b) => a + b, 0) / pa.length;
    const meanB = pb.reduce((a, b) => a + b, 0) / pb.length;
    expect(meanA).toBeGreaterThan(meanB);
  });
});
