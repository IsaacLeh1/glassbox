import { create } from 'zustand';
import type { LMTrainer } from '../engine/lmTrainer';

/**
 * The one model the user owns.
 *
 * Until now every module built its own model and kept it in a local ref, so
 * nothing the user trained in step 07 could be examined in step 08 or
 * anywhere else. This store is the shared handle: step 07 publishes what it
 * trained, and any other module can pick it up.
 *
 * It is deliberately not persisted. A trainer holds live typed arrays and a
 * tokenizer; serialising it to localStorage would be slow, lossy and would
 * quietly exceed the quota on anything but the smallest model. Losing the
 * model on reload is the honest behaviour, and the UI says so.
 */

export interface ModelOrigin {
  /** Short human label, e.g. "tiny_shakespeare - small". */
  label: string;
  /** Where the training text came from. */
  source: string;
  /** Characters of training text. */
  chars: number;
  steps: number;
  finalLoss: number;
  trainedAt: number;
}

interface ModelState {
  trainer: LMTrainer | null;
  origin: ModelOrigin | null;
  /**
   * Bumped whenever the weights change underneath a component: after a
   * fine-tune, or after a hand edit. React cannot see a mutation inside a
   * Float64Array, so this is what tells a view to recompute.
   */
  revision: number;

  publish: (trainer: LMTrainer, origin: ModelOrigin) => void;
  touch: () => void;
  clear: () => void;
}

export const useModel = create<ModelState>()((set) => ({
  trainer: null,
  origin: null,
  revision: 0,

  publish: (trainer, origin) => set((s) => ({ trainer, origin, revision: s.revision + 1 })),
  touch: () => set((s) => ({ revision: s.revision + 1 })),
  clear: () => set((s) => ({ trainer: null, origin: null, revision: s.revision + 1 })),
}));

/** Convenience for panels that only care whether there is anything to show. */
export function hasOwnModel(): boolean {
  return useModel.getState().trainer !== null;
}
