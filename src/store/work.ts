import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type { Licence, StageId } from '../engine/pipeline';

/**
 * The work a reader has actually done, kept across a page reload.
 *
 * Deliberately separate from the settings store. What lives here is large
 * and can fail to save: a corpus pulled from Hugging Face is comfortably a
 * hundred kilobytes, and a reader who loads several could run a browser out
 * of storage. Keeping it in its own key means a failure to write the corpus
 * cannot take the theme, the depth switch and the step progress down with it.
 *
 * What is not here: the trained model itself. A trainer holds live typed
 * arrays and a tokenizer, and serialising one would be slow, lossy and would
 * exceed the quota on anything but the smallest run. Losing the model on
 * reload is the honest behaviour and the UI says so; losing the half hour of
 * setup around it was not.
 */

export interface StoredCorpus {
  text: string;
  label: string;
  source: string;
}

export interface Design {
  dModel: number;
  nLayers: number;
  nHeads: number;
  dFF: number;
  blockSize: number;
  merges: number;
  batchSeqs: number;
  steps: number;
}

export const DEFAULT_DESIGN: Design = {
  dModel: 48,
  nLayers: 2,
  nHeads: 4,
  dFF: 96,
  blockSize: 24,
  merges: 120,
  batchSeqs: 8,
  steps: 400,
};

/** Everything typed or chosen in the gated lab programme. */
export interface LabState {
  done: StageId[];
  purpose: string;
  users: string;
  success: string;
  corpusId: string;
  licence: Licence;
  budgetExp: number;
  dModel: number;
  nLayers: number;
  nHeads: number;
  merges: number;
  steps: number;
  bar: number;
  banText: string;
  redTeamed: boolean;
  shipped: boolean;
}

export const DEFAULT_LAB: LabState = {
  done: [],
  purpose: '',
  users: '',
  success: '',
  corpusId: 'stories',
  licence: 'unknown',
  budgetExp: 11,
  dModel: 48,
  nLayers: 2,
  nHeads: 4,
  merges: 120,
  steps: 300,
  bar: 0.3,
  banText: '',
  redTeamed: false,
  shipped: false,
};

/**
 * Text beyond this is not written to storage.
 *
 * A browser gives a page a few megabytes in total. One large corpus would fit
 * and several would not, and a write that fails part way is worse than one
 * that never happened, so the limit is enforced here rather than discovered
 * at the quota.
 */
export const MAX_PERSISTED_CHARS = 400_000;

interface WorkState {
  corpus: StoredCorpus | null;
  design: Design;
  lab: LabState;
  /** Set when a corpus was too large to keep, so the UI can say so. */
  corpusDropped: boolean;

  setCorpus: (c: StoredCorpus | null) => void;
  setDesign: (patch: Partial<Design>) => void;
  setLab: (patch: Partial<LabState>) => void;
  resetLab: () => void;
  clearWork: () => void;
}

/**
 * Storage that degrades instead of throwing.
 *
 * If the quota is hit, the corpus text is dropped and the write retried, so a
 * reader keeps their charter, their architecture and their place in the
 * programme even when the text itself was too big to keep.
 */
export interface SimpleStorage {
  getItem(name: string): string | null;
  setItem(name: string, value: string): void;
  removeItem(name: string): void;
}

export function makeSafeStorage(backing: () => SimpleStorage): SimpleStorage {
  return {
    getItem: (name) => {
      try {
        return backing().getItem(name);
      } catch {
        return null;
      }
    },
    setItem: (name, value) => {
      try {
        backing().setItem(name, value);
      } catch {
        // Out of room. Drop the corpus text, which is almost all of the
        // weight, and keep the charter, the architecture and the place in
        // the programme, which are what a reader would actually mourn.
        try {
          const parsed = JSON.parse(value);
          if (parsed?.state) {
            parsed.state.corpus = null;
            parsed.state.corpusDropped = true;
          }
          backing().setItem(name, JSON.stringify(parsed));
        } catch {
          // Storage is unavailable or full even without the text. A reader
          // with no persistence is no worse off than before this existed.
        }
      }
    },
    removeItem: (name) => {
      try {
        backing().removeItem(name);
      } catch {
        /* nothing to do */
      }
    },
  };
}

const safeStorage = createJSONStorage<WorkState>(() => makeSafeStorage(() => localStorage));

export const useWork = create<WorkState>()(
  persist(
    (set) => ({
      corpus: null,
      design: DEFAULT_DESIGN,
      lab: DEFAULT_LAB,
      corpusDropped: false,

      setCorpus: (corpus) =>
        set({
          corpus:
            corpus && corpus.text.length > MAX_PERSISTED_CHARS
              ? { ...corpus, text: corpus.text.slice(0, MAX_PERSISTED_CHARS) }
              : corpus,
          corpusDropped: !!corpus && corpus.text.length > MAX_PERSISTED_CHARS,
        }),
      setDesign: (patch) => set((s) => ({ design: { ...s.design, ...patch } })),
      setLab: (patch) => set((s) => ({ lab: { ...s.lab, ...patch } })),
      resetLab: () => set({ lab: DEFAULT_LAB }),
      clearWork: () => set({ corpus: null, design: DEFAULT_DESIGN, lab: DEFAULT_LAB, corpusDropped: false }),
    }),
    {
      name: 'glassbox.work.v1',
      storage: safeStorage,
      partialize: (s) => ({ corpus: s.corpus, design: s.design, lab: s.lab, corpusDropped: s.corpusDropped }) as WorkState,
    },
  ),
);
