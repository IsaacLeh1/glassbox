import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { DEFAULT_LM, type LMConfig } from '../dspy/lmClient';

/**
 * Depth is the core idea of the product: the same panel explains itself in
 * plain language, in mathematics, or in the code that actually runs.
 */
export type Depth = 'plain' | 'math' | 'code';

export const DEPTHS: { id: Depth; label: string; hint: string }[] = [
  { id: 'plain', label: 'Plain', hint: 'No notation. Everything said in words.' },
  { id: 'math', label: 'Math', hint: 'The equations behind each step.' },
  { id: 'code', label: 'Code', hint: 'The exact source that computes it.' },
];

export type ThemeMode = 'dark' | 'light';

interface AppState {
  depth: Depth;
  theme: ThemeMode;
  setDepth: (d: Depth) => void;
  setTheme: (t: ThemeMode) => void;

  visited: Record<string, boolean>;
  completed: Record<string, boolean>;
  markVisited: (id: string) => void;
  markComplete: (id: string) => void;
  resetProgress: () => void;

  /** Stored locally in this browser only, and sent only to the base URL below. */
  lm: LMConfig;
  setLM: (patch: Partial<LMConfig>) => void;

  tourSeen: boolean;
  setTourSeen: (v: boolean) => void;
}

export const useApp = create<AppState>()(
  persist(
    (set) => ({
      depth: 'plain',
      theme: 'dark',
      setDepth: (depth) => set({ depth }),
      setTheme: (theme) => {
        document.documentElement.setAttribute('data-theme', theme);
        set({ theme });
      },

      visited: {},
      completed: {},
      markVisited: (id) => set((s) => ({ visited: { ...s.visited, [id]: true } })),
      markComplete: (id) => set((s) => ({ completed: { ...s.completed, [id]: true } })),
      resetProgress: () => set({ visited: {}, completed: {} }),

      lm: DEFAULT_LM,
      setLM: (patch) => set((s) => ({ lm: { ...s.lm, ...patch } })),

      tourSeen: false,
      setTourSeen: (tourSeen) => set({ tourSeen }),
    }),
    {
      name: 'glassbox.v1',
      partialize: (s) => ({
        depth: s.depth,
        theme: s.theme,
        visited: s.visited,
        completed: s.completed,
        // The API key is intentionally included so a session survives a reload.
        // It never leaves this browser except in requests to lm.baseUrl.
        lm: s.lm,
        tourSeen: s.tourSeen,
      }),
    },
  ),
);

export function applyStoredTheme() {
  const t = useApp.getState().theme;
  document.documentElement.setAttribute('data-theme', t);
}
