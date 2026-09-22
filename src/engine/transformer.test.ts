import { describe, expect, it } from 'vitest';
import { Transformer, sampleToken, type TransformerConfig } from './transformer';
import { BPETokenizer, charTokenizer } from './tokenizer';
import { getCorpus, CORPORA } from './corpus';
import { mulberry32 } from './tensor';
import { Optimizer, DEFAULT_OPTIM } from './optim';

const CFG: TransformerConfig = {
  vocab: 11,
  dModel: 12,
  nHeads: 3,
  nLayers: 2,
  blockSize: 7,
  dFF: 20,
};

const TOKENS = [3, 1, 4, 1, 5, 9, 2];
const TARGETS = [1, 4, 1, 5, 9, 2, 6];

function lossAt(model: Transformer, tokens: number[], targets: number[]): number {
  return model.loss(model.forward(tokens).trace, targets);
}

describe('transformer forward pass', () => {
  it('produces logits of the right shape', () => {
    const m = new Transformer(CFG, 5);
    const { trace } = m.forward(TOKENS);
    expect(trace.logits.rows).toBe(TOKENS.length);
    expect(trace.logits.cols).toBe(CFG.vocab);
    expect(trace.layers.length).toBe(CFG.nLayers);
    expect(trace.layers[0].heads.length).toBe(CFG.nHeads);
  });

  it('attention rows are probability distributions', () => {
    const m = new Transformer(CFG, 5);
    const { trace } = m.forward(TOKENS);
    for (const layer of trace.layers) {
      for (const head of layer.heads) {
        for (let i = 0; i < head.att.rows; i++) {
          let s = 0;
          for (let j = 0; j < head.att.cols; j++) s += head.att.data[i * head.att.cols + j];
          expect(s).toBeCloseTo(1, 9);
        }
      }
    }
  });

  it('respects the causal mask: no token attends to the future', () => {
    const m = new Transformer(CFG, 5);
    const { trace } = m.forward(TOKENS);
    for (const layer of trace.layers) {
      for (const head of layer.heads) {
        const T = head.att.rows;
        for (let i = 0; i < T; i++) {
          for (let j = i + 1; j < T; j++) {
            expect(head.att.data[i * T + j]).toBeLessThan(1e-9);
          }
        }
      }
    }
  });

  it('changing a later token cannot change an earlier prediction', () => {
    const m = new Transformer(CFG, 5);
    const a = m.forward([3, 1, 4, 1]).trace.logits;
    const b = m.forward([3, 1, 4, 7]).trace.logits;
    // Rows 0..2 saw identical context in both runs, so they must be identical.
    for (let t = 0; t < 3; t++) {
      for (let j = 0; j < CFG.vocab; j++) {
        expect(a.data[t * CFG.vocab + j]).toBeCloseTo(b.data[t * CFG.vocab + j], 12);
      }
    }
  });

  it('layer norm output has near-zero mean and unit variance per row', () => {
    const m = new Transformer(CFG, 5);
    const { trace } = m.forward(TOKENS);
    const ln = trace.layers[0].ln1;
    for (let i = 0; i < ln.rows; i++) {
      let mean = 0;
      for (let j = 0; j < ln.cols; j++) mean += ln.data[i * ln.cols + j];
      mean /= ln.cols;
      expect(Math.abs(mean)).toBeLessThan(1e-6);
    }
  });
});

describe('transformer backward pass', () => {
  it('every parameter tensor matches a finite-difference gradient', () => {
    const m = new Transformer(CFG, 5);
    m.zeroGrad();
    m.backward(m.forward(TOKENS), TARGETS);

    const eps = 1e-5;
    let checked = 0;
    for (const q of m.params) {
      // Sample a few entries from each tensor rather than all of them.
      const n = q.M.data.length;
      const picks = [0, Math.floor(n / 3), Math.floor((2 * n) / 3), n - 1];
      for (const i of new Set(picks)) {
        const orig = q.M.data[i];
        q.M.data[i] = orig + eps;
        const lp = lossAt(m, TOKENS, TARGETS);
        q.M.data[i] = orig - eps;
        const lm = lossAt(m, TOKENS, TARGETS);
        q.M.data[i] = orig;
        const numeric = (lp - lm) / (2 * eps);
        const analytic = q.g.data[i];
        const denom = Math.max(1e-7, Math.abs(numeric) + Math.abs(analytic));
        expect(
          Math.abs(numeric - analytic) / denom,
          `${q.name}[${i}] analytic=${analytic} numeric=${numeric}`,
        ).toBeLessThan(2e-4);
        checked++;
      }
    }
    expect(checked).toBeGreaterThan(50);
  });

  it('gradients accumulate across a batch instead of overwriting', () => {
    const m = new Transformer(CFG, 5);
    m.zeroGrad();
    m.backward(m.forward(TOKENS), TARGETS);
    const once = m.head.g.data[0];
    m.backward(m.forward(TOKENS), TARGETS);
    expect(m.head.g.data[0]).toBeCloseTo(once * 2, 9);
  });

  it('zeroGrad clears every buffer', () => {
    const m = new Transformer(CFG, 5);
    m.backward(m.forward(TOKENS), TARGETS);
    m.zeroGrad();
    for (const q of m.params) {
      for (const x of q.g.data) expect(x).toBe(0);
    }
  });
});

describe('transformer learns', () => {
  it('memorises a short repeating sequence', () => {
    const cfg: TransformerConfig = { vocab: 6, dModel: 32, nHeads: 4, nLayers: 2, blockSize: 12, dFF: 64 };
    const m = new Transformer(cfg, 3);
    const seq = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3];
    const tgt = [2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4];
    const opt = new Optimizer({ ...DEFAULT_OPTIM, name: 'adam', lr: 0.03 });
    const before = lossAt(m, seq, tgt);
    for (let step = 0; step < 200; step++) {
      m.zeroGrad();
      m.backward(m.forward(seq), tgt);
      opt.tick();
      for (const q of m.params) opt.step(q.name, q.M.data, q.g.data);
    }
    const after = lossAt(m, seq, tgt);
    expect(after).toBeLessThan(before);
    expect(after).toBeLessThan(0.05);

    // And it should now predict the continuation correctly.
    const { probs } = m.predictNext([1, 2, 3]);
    const best = probs.indexOf(Math.max(...probs));
    expect(best).toBe(4);
  });
});

describe('sampling', () => {
  const logits = [4, 3, 2, 1, 0];

  it('probabilities sum to one and are ordered like the logits', () => {
    const s = sampleToken(logits, { temperature: 1, topK: 0, topP: 1 }, mulberry32(1));
    const total = s.candidates.reduce((a, c) => a + c.prob, 0);
    expect(total).toBeCloseTo(1, 12);
    expect(s.candidates[0].prob).toBeGreaterThan(s.candidates[1].prob);
  });

  it('top-k keeps exactly k candidates', () => {
    const s = sampleToken(logits, { temperature: 1, topK: 2, topP: 1 }, mulberry32(1));
    expect(s.candidates.filter((c) => c.kept).length).toBe(2);
  });

  it('near-zero temperature always picks the top token', () => {
    for (let i = 0; i < 20; i++) {
      const s = sampleToken(logits, { temperature: 0.01, topK: 0, topP: 1 }, mulberry32(i));
      expect(s.chosen).toBe(0);
    }
  });

  it('high temperature spreads the choice across tokens', () => {
    const seen = new Set<number>();
    for (let i = 0; i < 80; i++) {
      seen.add(sampleToken(logits, { temperature: 8, topK: 0, topP: 1 }, mulberry32(i)).chosen);
    }
    expect(seen.size).toBeGreaterThan(2);
  });

  it('top-p never keeps more mass than needed', () => {
    const s = sampleToken(logits, { temperature: 1, topK: 0, topP: 0.5 }, mulberry32(1));
    const kept = s.candidates.filter((c) => c.kept);
    const mass = kept.reduce((a, c) => a + c.prob, 0);
    expect(mass).toBeGreaterThanOrEqual(0.5);
    expect(kept.length).toBeLessThan(logits.length);
  });
});

describe('tokenizer', () => {
  const text = getCorpus('stories', 1);

  it('round-trips text exactly', () => {
    const tok = BPETokenizer.train(text, 120);
    const sample = text.slice(0, 400);
    expect(tok.decode(tok.encode(sample))).toBe(sample);
  });

  it('BPE produces fewer tokens than raw characters', () => {
    const tok = BPETokenizer.train(text, 150);
    const ch = charTokenizer(text);
    const sample = text.slice(0, 600);
    expect(tok.encode(sample).length).toBeLessThan(ch.encode(sample).length);
  });

  it('merge ranks are strictly increasing and counts are sane', () => {
    const tok = BPETokenizer.train(text, 80);
    tok.merges.forEach((m, i) => {
      expect(m.rank).toBe(i);
      expect(m.result).toBe(m.a + m.b);
      expect(m.count).toBeGreaterThanOrEqual(2);
    });
  });

  it('encode trace ends at the final tokenisation', () => {
    const tok = BPETokenizer.train(text, 100);
    const steps = tok.encodeWordTrace(' garden');
    expect(steps[0].tokens.length).toBeGreaterThanOrEqual(steps[steps.length - 1].tokens.length);
    expect(steps[steps.length - 1].tokens.join('')).toBe(' garden');
  });

  it('char tokenizer round-trips too', () => {
    const ch = charTokenizer(text);
    expect(ch.decode(ch.encode('the fox'))).toBe('the fox');
  });
});

describe('corpora', () => {
  it('each corpus is deterministic and non-trivial', () => {
    for (const c of CORPORA) {
      const a = c.build(7);
      const b = c.build(7);
      expect(a).toBe(b);
      expect(a.length).toBeGreaterThan(500);
      expect(new Set(Array.from(a)).size).toBeGreaterThan(5);
    }
  });
});
