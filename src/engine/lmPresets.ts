import { getCorpus } from './corpus';
import { DEFAULT_LM_TRAIN, LMTrainer } from './lmTrainer';
import { DEFAULT_TCONFIG } from './transformer';

/**
 * The three in-browser model sizes, and the one place that builds them.
 *
 * This lives outside any component file so that editing a lab does not force a
 * full page reload during development.
 */
export const MODEL_PRESETS = [
  { id: 'tiny', label: 'Tiny', dModel: 32, nHeads: 2, nLayers: 1, blockSize: 16, dFF: 64, merges: 80 },
  { id: 'small', label: 'Small', dModel: 48, nHeads: 4, nLayers: 2, blockSize: 24, dFF: 96, merges: 120 },
  { id: 'medium', label: 'Medium', dModel: 64, nHeads: 4, nLayers: 3, blockSize: 32, dFF: 128, merges: 160 },
];

export function buildLMTrainer(corpusId: string, presetId: string): LMTrainer {
  const preset = MODEL_PRESETS.find((p) => p.id === presetId) ?? MODEL_PRESETS[1];
  return new LMTrainer(
    getCorpus(corpusId, 12),
    {
      ...DEFAULT_TCONFIG,
      dModel: preset.dModel,
      nHeads: preset.nHeads,
      nLayers: preset.nLayers,
      blockSize: preset.blockSize,
      dFF: preset.dFF,
    },
    DEFAULT_LM_TRAIN,
    preset.merges,
  );
}
