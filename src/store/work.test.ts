import { describe, expect, it } from 'vitest';
import { makeSafeStorage, type SimpleStorage } from './work';

/** A store that refuses anything above a byte budget, like a real quota. */
function tinyStorage(limit: number): SimpleStorage & { data: Map<string, string> } {
  const data = new Map<string, string>();
  return {
    data,
    getItem: (k) => data.get(k) ?? null,
    setItem: (k, v) => {
      if (v.length > limit) throw new DOMException('quota', 'QuotaExceededError');
      data.set(k, v);
    },
    removeItem: (k) => void data.delete(k),
  };
}

const payload = (corpusChars: number) =>
  JSON.stringify({
    state: {
      corpus: { text: 'x'.repeat(corpusChars), label: 'big', source: 'hf' },
      lab: { purpose: 'keep me', done: ['charter'] },
      design: { dModel: 48 },
    },
  });

describe('work storage', () => {
  it('writes normally when there is room', () => {
    const back = tinyStorage(10_000);
    makeSafeStorage(() => back).setItem('k', payload(10));
    expect(JSON.parse(back.data.get('k')!).state.corpus.text.length).toBe(10);
  });

  it('drops the corpus and keeps the rest when the quota is hit', () => {
    // The whole reason the corpus lives in its own store: a failure to save
    // it must not cost the reader their charter or their place.
    const back = tinyStorage(400);
    makeSafeStorage(() => back).setItem('k', payload(5000));
    const saved = JSON.parse(back.data.get('k')!);
    expect(saved.state.corpus).toBeNull();
    expect(saved.state.corpusDropped).toBe(true);
    expect(saved.state.lab.purpose).toBe('keep me');
    expect(saved.state.lab.done).toEqual(['charter']);
  });

  it('gives up quietly when even the small state will not fit', () => {
    const back = tinyStorage(5);
    expect(() => makeSafeStorage(() => back).setItem('k', payload(5000))).not.toThrow();
    expect(back.data.size).toBe(0);
  });

  it('survives storage being unavailable entirely', () => {
    const dead = (): SimpleStorage => {
      throw new Error('blocked');
    };
    const s = makeSafeStorage(dead);
    expect(s.getItem('k')).toBeNull();
    expect(() => s.setItem('k', payload(10))).not.toThrow();
    expect(() => s.removeItem('k')).not.toThrow();
  });

  it('reads back what it wrote', () => {
    const back = tinyStorage(10_000);
    const s = makeSafeStorage(() => back);
    s.setItem('k', payload(20));
    expect(s.getItem('k')).toBe(back.data.get('k'));
    s.removeItem('k');
    expect(s.getItem('k')).toBeNull();
  });
});
