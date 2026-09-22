/**
 * Shape bookkeeping for a transformer forward pass.
 *
 * Everything a language model does to your text is a sequence of matrix
 * operations with very specific shapes. This module produces that sequence for
 * any configuration, so the same code can describe the 30k-parameter model
 * running in this tab and a 70B-parameter model that is not.
 */

export interface ArchConfig {
  vocab: number;
  dModel: number;
  nHeads: number;
  nLayers: number;
  dFF: number;
  ctx: number;
}

export type StageKind = 'lookup' | 'matmul' | 'elementwise' | 'softmax' | 'norm' | 'residual';

export interface Stage {
  id: string;
  group: string;
  label: string;
  detail: string;
  /** Left operand shape, [rows, cols]. */
  a: [number, number];
  aLabel: string;
  /** Right operand shape, if this is a matrix multiply. */
  b?: [number, number];
  bLabel?: string;
  out: [number, number];
  outLabel: string;
  kind: StageKind;
  /** Multiply-accumulate operations counted as 2 FLOPs each. */
  flops: number;
  /** Learned parameters used by this stage. */
  params: number;
  /** True for stages repeated once per layer. */
  perLayer: boolean;
}

const mm = (m: number, k: number, n: number) => 2 * m * k * n;

/**
 * The full chain for T input tokens. Per-layer stages are listed once and
 * marked, so a UI can either show them once or multiply through by nLayers.
 */
export function pipelineStages(cfg: ArchConfig, T: number): Stage[] {
  const { vocab, dModel, nHeads, dFF } = cfg;
  const dHead = Math.floor(dModel / nHeads);
  const s: Stage[] = [];

  s.push({
    id: 'onehot',
    group: 'Input',
    label: 'One-hot encode',
    detail:
      'Each token id becomes a row that is all zeros except a single 1 in the column for that token. This matrix is almost entirely empty, which is exactly why nobody actually builds it.',
    a: [T, 1],
    aLabel: 'token ids',
    out: [T, vocab],
    outLabel: 'one-hot',
    kind: 'lookup',
    flops: 0,
    params: 0,
    perLayer: false,
  });

  s.push({
    id: 'embed',
    group: 'Input',
    label: 'Embedding lookup',
    detail:
      'Multiplying the one-hot matrix by the embedding table selects one row per token. Because every other term is multiplied by zero, real implementations skip the multiply and index directly. The result is mathematically identical.',
    a: [T, vocab],
    aLabel: 'one-hot',
    b: [vocab, dModel],
    bLabel: 'E (token embeddings)',
    out: [T, dModel],
    outLabel: 'x',
    kind: 'matmul',
    flops: 0,
    params: vocab * dModel,
    perLayer: false,
  });

  s.push({
    id: 'pos',
    group: 'Input',
    label: 'Add positions',
    detail:
      'Attention has no sense of order on its own, so a learned vector for each slot is added. Swap two words without this and the model cannot tell.',
    a: [T, dModel],
    aLabel: 'x',
    out: [T, dModel],
    outLabel: 'x',
    kind: 'elementwise',
    flops: T * dModel,
    params: cfg.ctx * dModel,
    perLayer: false,
  });

  s.push({
    id: 'ln1',
    group: 'Block',
    label: 'Layer norm',
    detail: 'Rescales each row to zero mean and unit variance, then applies a learned gain and bias. Keeps the numbers in a range the rest of the block can work with.',
    a: [T, dModel],
    aLabel: 'x',
    out: [T, dModel],
    outLabel: 'h',
    kind: 'norm',
    flops: 5 * T * dModel,
    params: 2 * dModel,
    perLayer: true,
  });

  for (const n of ['Q', 'K', 'V'] as const) {
    s.push({
      id: `proj${n}`,
      group: 'Attention',
      label: `Project to ${n}`,
      detail:
        n === 'Q'
          ? 'Each token forms a query: what am I looking for?'
          : n === 'K'
            ? 'Each token forms a key: what do I have to offer?'
            : 'Each token forms a value: what would I contribute if attended to?',
      a: [T, dModel],
      aLabel: 'h',
      b: [dModel, dModel],
      bLabel: `W${n.toLowerCase()}`,
      out: [T, dModel],
      outLabel: n,
      kind: 'matmul',
      flops: mm(T, dModel, dModel),
      params: dModel * dModel,
      perLayer: true,
    });
  }

  s.push({
    id: 'scores',
    group: 'Attention',
    label: 'Scores: Q times K transposed',
    detail: `Every token is compared against every earlier token, per head. This is the one stage whose cost grows with the square of sequence length, and the reason long context is expensive.`,
    a: [T, dHead],
    aLabel: 'Q (per head)',
    b: [dHead, T],
    bLabel: 'Kᵀ (per head)',
    out: [T, T],
    outLabel: 'scores',
    kind: 'matmul',
    flops: mm(T, dHead, T) * nHeads,
    params: 0,
    perLayer: true,
  });

  s.push({
    id: 'softmax',
    group: 'Attention',
    label: 'Mask and softmax',
    detail:
      'Future positions are set to minus infinity so they become exactly zero, then each row is turned into percentages that add to one.',
    a: [T, T],
    aLabel: 'scores',
    out: [T, T],
    outLabel: 'attention',
    kind: 'softmax',
    flops: 4 * T * T * nHeads,
    params: 0,
    perLayer: true,
  });

  s.push({
    id: 'attnv',
    group: 'Attention',
    label: 'Weighted sum of values',
    detail: 'Each token gets a blend of the value vectors it attended to, in exactly those proportions.',
    a: [T, T],
    aLabel: 'attention',
    b: [T, dHead],
    bLabel: 'V (per head)',
    out: [T, dHead],
    outLabel: 'head output',
    kind: 'matmul',
    flops: mm(T, T, dHead) * nHeads,
    params: 0,
    perLayer: true,
  });

  s.push({
    id: 'projo',
    group: 'Attention',
    label: 'Recombine the heads',
    detail: 'The heads are concatenated back to full width and mixed by one more learned matrix.',
    a: [T, dModel],
    aLabel: 'concat heads',
    b: [dModel, dModel],
    bLabel: 'Wo',
    out: [T, dModel],
    outLabel: 'attn out',
    kind: 'matmul',
    flops: mm(T, dModel, dModel),
    params: dModel * dModel,
    perLayer: true,
  });

  s.push({
    id: 'res1',
    group: 'Block',
    label: 'Residual add',
    detail:
      'The block output is added back onto its input rather than replacing it. Each layer edits the running representation instead of rewriting it, which is what lets very deep stacks train at all.',
    a: [T, dModel],
    aLabel: 'attn out',
    out: [T, dModel],
    outLabel: 'x',
    kind: 'residual',
    flops: T * dModel,
    params: 0,
    perLayer: true,
  });

  s.push({
    id: 'ff1',
    group: 'Feed-forward',
    label: 'Expand',
    detail: `Each token is pushed out to ${dFF} dimensions independently of the others. No mixing between tokens happens here at all.`,
    a: [T, dModel],
    aLabel: 'h',
    b: [dModel, dFF],
    bLabel: 'W1',
    out: [T, dFF],
    outLabel: 'wide',
    kind: 'matmul',
    flops: mm(T, dModel, dFF),
    params: dModel * dFF,
    perLayer: true,
  });

  s.push({
    id: 'gelu',
    group: 'Feed-forward',
    label: 'GELU',
    detail: 'The only nonlinearity in the block, applied to every element on its own.',
    a: [T, dFF],
    aLabel: 'wide',
    out: [T, dFF],
    outLabel: 'wide',
    kind: 'elementwise',
    flops: 8 * T * dFF,
    params: 0,
    perLayer: true,
  });

  s.push({
    id: 'ff2',
    group: 'Feed-forward',
    label: 'Contract',
    detail: 'Back down to the model width so it can be added to the residual stream.',
    a: [T, dFF],
    aLabel: 'wide',
    b: [dFF, dModel],
    bLabel: 'W2',
    out: [T, dModel],
    outLabel: 'ff out',
    kind: 'matmul',
    flops: mm(T, dFF, dModel),
    params: dFF * dModel,
    perLayer: true,
  });

  s.push({
    id: 'head',
    group: 'Output',
    label: 'Unembed to vocabulary',
    detail:
      'The final matrix turns each position back into one score per token in the vocabulary. Only the last row matters when generating.',
    a: [T, dModel],
    aLabel: 'x',
    b: [dModel, vocab],
    bLabel: 'Wu (unembedding)',
    out: [T, vocab],
    outLabel: 'logits',
    kind: 'matmul',
    flops: mm(T, dModel, vocab),
    params: dModel * vocab,
    perLayer: false,
  });

  s.push({
    id: 'outsoftmax',
    group: 'Output',
    label: 'Softmax',
    detail: 'Scores become probabilities over the whole vocabulary. One of them gets chosen, appended, and the entire process runs again.',
    a: [T, vocab],
    aLabel: 'logits',
    out: [T, vocab],
    outLabel: 'probabilities',
    kind: 'softmax',
    flops: 4 * T * vocab,
    params: 0,
    perLayer: false,
  });

  return s;
}

export interface ArchSummary {
  params: number;
  embedParams: number;
  attnParams: number;
  ffParams: number;
  headParams: number;
  flopsPerToken: number;
  flopsForT: number;
  kvCacheBytesPerToken: number;
}

export function summarise(cfg: ArchConfig, T: number, tiedEmbeddings = false): ArchSummary {
  const stages = pipelineStages(cfg, T);
  const perLayer = stages.filter((s) => s.perLayer);
  const once = stages.filter((s) => !s.perLayer);

  const embedParams = cfg.vocab * cfg.dModel + cfg.ctx * cfg.dModel;
  const attnParams = cfg.nLayers * (4 * cfg.dModel * cfg.dModel + 2 * cfg.dModel + 2 * cfg.dModel);
  const ffParams = cfg.nLayers * (2 * cfg.dModel * cfg.dFF + cfg.dFF + cfg.dModel + 2 * cfg.dModel);
  const headParams = tiedEmbeddings ? 0 : cfg.dModel * cfg.vocab;

  const flopsForT =
    once.reduce((n, s) => n + s.flops, 0) + cfg.nLayers * perLayer.reduce((n, s) => n + s.flops, 0);

  return {
    params: embedParams + attnParams + ffParams + headParams,
    embedParams,
    attnParams,
    ffParams,
    headParams,
    flopsPerToken: T > 0 ? flopsForT / T : 0,
    flopsForT,
    // Keys and values for every layer, kept in bf16 so generation does not
    // recompute the whole prefix on every new token.
    kvCacheBytesPerToken: 2 * 2 * cfg.nLayers * cfg.dModel,
  };
}

/**
 * Published architectures, for comparing the model in this tab against the
 * ones people actually use.
 */
export interface KnownModel {
  id: string;
  label: string;
  note: string;
  cfg: ArchConfig;
  tied: boolean;
  reportedParams: string;
}

export const KNOWN_MODELS: KnownModel[] = [
  {
    id: 'gpt2-small',
    label: 'GPT-2 Small',
    note: 'The model that made this architecture famous, and still a common teaching baseline.',
    cfg: { vocab: 50257, dModel: 768, nHeads: 12, nLayers: 12, dFF: 3072, ctx: 1024 },
    tied: true,
    reportedParams: '124M',
  },
  {
    id: 'gpt2-medium',
    label: 'GPT-2 Medium',
    note: 'Same design, wider and deeper. Nothing conceptual changes.',
    cfg: { vocab: 50257, dModel: 1024, nHeads: 16, nLayers: 24, dFF: 4096, ctx: 1024 },
    tied: true,
    reportedParams: '355M',
  },
  {
    id: 'gpt2-xl',
    label: 'GPT-2 XL',
    note: 'The largest of the original family.',
    cfg: { vocab: 50257, dModel: 1600, nHeads: 25, nLayers: 48, dFF: 6400, ctx: 1024 },
    tied: true,
    reportedParams: '1.5B',
  },
  {
    id: 'llama3-8b',
    label: 'Llama 3 8B',
    note: 'A modern open-weight model. Bigger vocabulary, far longer context, gated feed-forward.',
    cfg: { vocab: 128256, dModel: 4096, nHeads: 32, nLayers: 32, dFF: 14336, ctx: 8192 },
    tied: false,
    reportedParams: '8B',
  },
  {
    id: 'llama3-70b',
    label: 'Llama 3 70B',
    note: 'Wide enough that a single weight matrix no longer fits comfortably on one accelerator.',
    cfg: { vocab: 128256, dModel: 8192, nHeads: 64, nLayers: 80, dFF: 28672, ctx: 8192 },
    tied: false,
    reportedParams: '70B',
  },
];
