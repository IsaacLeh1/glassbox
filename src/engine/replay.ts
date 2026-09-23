import type { Matrix } from './tensor';
import type { Trace, TransformerConfig } from './transformer';

/**
 * A forward pass, flattened into an ordered list of stages you can scrub.
 *
 * The model already keeps every intermediate in its Trace. What was missing
 * was an ordering: a single sequence of "and then this happened" steps that a
 * timeline can index into, so the whole computation can be walked forwards and
 * backwards instead of only seen as a finished picture.
 *
 * Nothing here computes anything. Every matrix handed out is the same object
 * the model produced during the real forward pass.
 */

export type StageKind =
  | 'tokens'
  | 'embed'
  | 'pos'
  | 'residual'
  | 'ln'
  | 'scores'
  | 'attention'
  | 'headout'
  | 'project'
  | 'ffhidden'
  | 'ffout'
  | 'logits'
  | 'sample';

/** What the columns of a stage's matrix mean, which decides how it is labelled. */
export type ColSpace = 'model' | 'head' | 'ff' | 'positions' | 'vocab' | 'none';

export interface Stage {
  id: string;
  kind: StageKind;
  /** Block index, or -1 for anything outside the stack of blocks. */
  layer: number;
  /** Head index, or -1 when the stage is not per-head. */
  head: number;
  title: string;
  /** Breadcrumb shown above the matrix, e.g. "Block 1 - Head 3". */
  crumb: string;
  matrix: Matrix | null;
  colSpace: ColSpace;
  /** True when this stage's output is the residual stream itself. */
  isResidual: boolean;
  /**
   * How far this stage moved the residual stream, as a fraction of the
   * stream's own size. Only set on stages that write back into it.
   */
  delta?: number;
  plain: string;
  math: string;
  code: string;
}

function frob(m: Matrix): number {
  let s = 0;
  for (let i = 0; i < m.data.length; i++) s += m.data[i] * m.data[i];
  return Math.sqrt(s);
}

/** Relative size of the change from `a` to `b`, guarded against a zero start. */
function relDelta(a: Matrix, b: Matrix): number {
  let d = 0;
  const n = Math.min(a.data.length, b.data.length);
  for (let i = 0; i < n; i++) {
    const x = b.data[i] - a.data[i];
    d += x * x;
  }
  const base = frob(a);
  return base > 1e-12 ? Math.sqrt(d) / base : 0;
}

const ord = (n: number) => ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth'][n] ?? `#${n + 1}`;

/**
 * Build the ordered stage list for one forward pass.
 *
 * The order is exactly the order the model computes in, so scrubbing left to
 * right is watching the computation happen.
 */
export function buildStages(trace: Trace, cfg: TransformerConfig): Stage[] {
  const out: Stage[] = [];
  const T = trace.tokens.length;
  const dh = Math.floor(cfg.dModel / cfg.nHeads);

  out.push({
    id: 'tokens',
    kind: 'tokens',
    layer: -1,
    head: -1,
    title: 'Text becomes numbers',
    crumb: 'Input',
    matrix: null,
    colSpace: 'none',
    isResidual: false,
    plain:
      `Your text was cut into ${T} pieces, and each piece was looked up in a list to get its number. ` +
      'That is the whole of it. From here on the model never sees letters again, only these numbers and ' +
      'the numbers they turn into.',
    math: `x \\in \\{0, 1, \\ldots, V-1\\}^{${T}}, \\quad V = ${cfg.vocab}`,
    code: 'const ids = tok.encode(text);',
  });

  out.push({
    id: 'tok_emb',
    kind: 'embed',
    layer: -1,
    head: -1,
    title: 'Each token looks up its meaning vector',
    crumb: 'Embedding',
    matrix: trace.tokenEmb,
    colSpace: 'model',
    isResidual: false,
    plain:
      `Every token number is used as a row number into a big table of weights. Row 40 of that table is ` +
      `the ${cfg.dModel} numbers that mean "token 40" to this model. Nobody chose what those numbers are; ` +
      'they were learned during training. Two tokens that get used in similar ways end up with similar rows.',
    math: `E_{\\text{tok}}[i,:] = W_{\\text{emb}}[x_i,:] \\in \\mathbb{R}^{${cfg.dModel}}`,
    code: 'for (let t = 0; t < T; t++)\n  tokenEmb[t] = tokEmb.row(tokens[t]);',
  });

  out.push({
    id: 'pos_emb',
    kind: 'pos',
    layer: -1,
    head: -1,
    title: 'Each slot adds where it is in the line',
    crumb: 'Position',
    matrix: trace.posEmb,
    colSpace: 'model',
    isResidual: false,
    plain:
      'The lookup above says what a token is, but not where it sits. A second table is read by position ' +
      'instead of by token: first slot, second slot, and so on. Without this the model would see your ' +
      'sentence as a bag of words with no order, and "the dog bit the man" would look identical to ' +
      '"the man bit the dog".',
    math: `E_{\\text{pos}}[i,:] = W_{\\text{pos}}[i,:], \\quad i = 0 \\ldots ${T - 1}`,
    code: 'posEmb[t] = pos.row(t);',
  });

  out.push({
    id: 'embedded',
    kind: 'residual',
    layer: -1,
    head: -1,
    title: 'The two are added together',
    crumb: 'Residual stream starts',
    matrix: trace.embedded,
    colSpace: 'model',
    isResidual: true,
    plain:
      'The two tables are simply added, cell by cell. The result is the starting state of what is called ' +
      'the residual stream: one row of numbers per token, which every block from here on will read from ' +
      'and add back into. Think of it as a shared scratchpad that the whole model writes on.',
    math: 'H^{(0)} = E_{\\text{tok}} + E_{\\text{pos}}',
    code: 'embedded[t][j] = tokenEmb[t][j] + posEmb[t][j];',
  });

  let prevResidual = trace.embedded;

  for (let l = 0; l < trace.layers.length; l++) {
    const L = trace.layers[l];
    const at = `Block ${l}`;

    out.push({
      id: `L${l}.ln1`,
      kind: 'ln',
      layer: l,
      head: -1,
      title: 'Put every row on the same scale',
      crumb: `${at} - LayerNorm 1`,
      matrix: L.ln1,
      colSpace: 'model',
      isResidual: false,
      plain:
        'Before the block reads the scratchpad it rescales each row so the numbers in it average zero and ' +
        'have a consistent spread. This is housekeeping, not thinking. It stops one row from growing so ' +
        'large that it drowns out the others, which is what makes it possible to stack blocks at all. ' +
        'Two learned knobs per column let the model undo the rescaling if it turns out to need to.',
      math: '\\hat{h} = \\frac{h - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\odot \\gamma + \\beta',
      code: 'const { y } = layerNormFwd(x, B.ln1g.M, B.ln1b.M);',
    });

    for (let h = 0; h < L.heads.length; h++) {
      const H = L.heads[h];
      const hc = `${at} - Head ${h}`;

      out.push({
        id: `L${l}.h${h}.scores`,
        kind: 'scores',
        layer: l,
        head: h,
        title: 'Every token scores every earlier token',
        crumb: hc,
        matrix: H.scores,
        colSpace: 'positions',
        isResidual: false,
        plain:
          'This head turns each row into a question vector and a label vector, then compares every ' +
          'question against every label by multiplying them together. A big number in row 5, column 2 ' +
          'means "when working on token 5, token 2 looks relevant". The upper-right of the grid is blank ' +
          'on purpose: a token is never allowed to look at tokens that come after it, because at ' +
          'prediction time those have not been written yet.',
        math: `S = \\frac{QK^{\\top}}{\\sqrt{d_h}}, \\quad d_h = ${dh}, \\quad S_{ij} = -\\infty \\text{ for } j > i`,
        code: 'scores[i][j] = dot(q[i], k[j]) / Math.sqrt(dh);\nif (j > i) scores[i][j] = -Infinity;',
      });

      out.push({
        id: `L${l}.h${h}.att`,
        kind: 'attention',
        layer: l,
        head: h,
        title: 'Scores become percentages of attention',
        crumb: hc,
        matrix: H.att,
        colSpace: 'positions',
        isResidual: false,
        plain:
          'The raw scores are squashed so that each row adds up to exactly 1. Now row 5 is a straight ' +
          'answer to "of all the attention token 5 has to spend, what share goes where?". This is the ' +
          'row to read if you want to know what the model is actually looking at. A row that is one ' +
          'bright cell is a head that has locked onto a single earlier token; a row that is evenly grey ' +
          'is a head that has not decided anything.',
        math: 'A = \\operatorname{softmax}_{\\text{row}}(S), \\quad \\textstyle\\sum_j A_{ij} = 1',
        code: 'const att = softmaxRows(scores);',
      });

      out.push({
        id: `L${l}.h${h}.out`,
        kind: 'headout',
        layer: l,
        head: h,
        title: 'Collect the information it attended to',
        crumb: hc,
        matrix: H.out,
        colSpace: 'head',
        isResidual: false,
        plain:
          'Each token now takes a weighted blend of what the tokens it attended to are carrying, using ' +
          `the percentages from the previous stage as the weights. This is the ${ord(h)} head's answer: ` +
          `${dh} numbers per token, summarising what it went and fetched.`,
        math: `O_h = A V \\in \\mathbb{R}^{${T} \\times ${dh}}`,
        code: 'const out = matmul(att, v);',
      });
    }

    out.push({
      id: `L${l}.attnOut`,
      kind: 'project',
      layer: l,
      head: -1,
      title: 'All heads are stitched back together',
      crumb: `${at} - Output projection`,
      matrix: L.attnOut,
      colSpace: 'model',
      isResidual: false,
      plain:
        `The ${L.heads.length} heads worked side by side and each produced its own narrow answer. Their ` +
        'answers are laid end to end and pushed through one more weight matrix, which is what lets the ' +
        'block decide how much of each head to believe and how to write the combination back onto the ' +
        'shared scratchpad.',
      math: 'A_{\\text{out}} = \\operatorname{concat}(O_0 \\ldots O_{H-1}) W_O + b_O',
      code: 'const attnOut = add(matmul(concat(heads), B.wo.M), B.bo.M);',
    });

    out.push({
      id: `L${l}.afterAttn`,
      kind: 'residual',
      layer: l,
      head: -1,
      title: 'Attention is added onto the scratchpad',
      crumb: `${at} - After attention`,
      matrix: L.afterAttn,
      colSpace: 'model',
      isResidual: true,
      delta: relDelta(prevResidual, L.afterAttn),
      plain:
        'Notice the word added. The block does not replace what was on the scratchpad, it adds a ' +
        'correction to it. Everything written earlier survives. This is why a model can be very deep ' +
        'without falling apart: each block only has to contribute an adjustment, and a block that has ' +
        'nothing useful to say can contribute almost nothing and do no harm. The number in the corner ' +
        'tells you how big this particular contribution was.',
      math: 'H \\leftarrow H + A_{\\text{out}}',
      code: 'afterAttn[i] = x[i] + attnOut[i];',
    });

    out.push({
      id: `L${l}.ln2`,
      kind: 'ln',
      layer: l,
      head: -1,
      title: 'Rescale again before the second half',
      crumb: `${at} - LayerNorm 2`,
      matrix: L.ln2,
      colSpace: 'model',
      isResidual: false,
      plain:
        'The same housekeeping as before, because the scratchpad has just been written to and the rows ' +
        'may have grown. The second half of the block gets a clean, evenly scaled copy to work from.',
      math: '\\hat{h} = \\frac{h - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\odot \\gamma_2 + \\beta_2',
      code: 'const { y } = layerNormFwd(afterAttn, B.ln2g.M, B.ln2b.M);',
    });

    out.push({
      id: `L${l}.ffHidden`,
      kind: 'ffhidden',
      layer: l,
      head: -1,
      title: 'Widen out and think about each token alone',
      crumb: `${at} - Feed-forward`,
      matrix: L.ffHidden,
      colSpace: 'ff',
      isResidual: false,
      plain:
        `Attention was about tokens looking at each other. This part is the opposite: every token is ` +
        `processed on its own, with no reference to any other. Each row is stretched from ${cfg.dModel} ` +
        `numbers out to ${cfg.dFF}, and anything that comes out negative is mostly flattened toward zero. ` +
        'That flattening is the only place in the whole block where something non-linear happens, and ' +
        'without it the entire stack would collapse into a single matrix multiply. In a full-sized model ' +
        'this part is four times the width of the residual stream, rather than the ' +
        `${(cfg.dFF / cfg.dModel).toFixed(1)} times it is here, which is why most of a real model's ` +
        'weights sit in it and why most of its factual recall is thought to live here rather than in ' +
        'attention. At this size the split is far more even, so do not expect your own model to show it.',
      math: 'F = \\operatorname{GELU}(\\hat{H} W_1 + b_1)',
      code: 'const ffHidden = gelu(add(matmul(ln2, B.w1.M), B.b1.M));',
    });

    out.push({
      id: `L${l}.ffOut`,
      kind: 'ffout',
      layer: l,
      head: -1,
      title: 'Squeeze back down to the stream width',
      crumb: `${at} - Feed-forward out`,
      matrix: L.ffOut,
      colSpace: 'model',
      isResidual: false,
      plain:
        `The wide version is projected back down to ${cfg.dModel} numbers so it can be written onto the ` +
        'scratchpad, which is a fixed width. The widening and narrowing together are what give this part ' +
        'its room to compute: it has space to work in, then has to commit to a compact answer.',
      math: 'F_{\\text{out}} = F W_2 + b_2',
      code: 'const ffOut = add(matmul(ffHidden, B.w2.M), B.b2.M);',
    });

    out.push({
      id: `L${l}.out`,
      kind: 'residual',
      layer: l,
      head: -1,
      title: `Block ${l} finishes and hands the scratchpad on`,
      crumb: `${at} - After feed-forward`,
      matrix: L.output,
      colSpace: 'model',
      isResidual: true,
      delta: relDelta(L.afterAttn, L.output),
      plain:
        'Added on, not written over, exactly as before. The scratchpad now leaves this block and goes ' +
        'into the next one, which will do the same two things again with completely different weights.',
      math: 'H \\leftarrow H + F_{\\text{out}}',
      code: 'output[i] = afterAttn[i] + ffOut[i];',
    });

    prevResidual = L.output;
  }

  out.push({
    id: 'final_ln',
    kind: 'ln',
    layer: -1,
    head: -1,
    title: 'One last rescale',
    crumb: 'Final LayerNorm',
    matrix: trace.finalLN,
    colSpace: 'model',
    isResidual: false,
    plain:
      'Every block has finished writing. One final rescale puts the scratchpad into a consistent state ' +
      'before the last step reads it.',
    math: '\\hat{H} = \\operatorname{LayerNorm}(H^{(L)})',
    code: 'const { y } = layerNormFwd(x, this.lnfg.M, this.lnfb.M);',
  });

  out.push({
    id: 'logits',
    kind: 'logits',
    layer: -1,
    head: -1,
    title: 'Score every word in the vocabulary',
    crumb: 'Output head',
    matrix: trace.logits,
    colSpace: 'vocab',
    isResidual: false,
    plain:
      `The final scratchpad row for each token is compared against all ${cfg.vocab} entries in the ` +
      'vocabulary, producing one raw score per entry. These are not probabilities yet and they are not ' +
      'capped at anything. Only the very last row matters for generating: it is the model\'s opinion ' +
      'about what comes next. The rows above it are what it would have predicted at each earlier point, ' +
      'which is exactly what training grades it on.',
    math: `Z = \\hat{H} W_U + b_U \\in \\mathbb{R}^{${T} \\times ${cfg.vocab}}`,
    code: 'const logits = add(matmul(finalLN, this.head.M), this.headb.M);',
  });

  out.push({
    id: 'sample',
    kind: 'sample',
    layer: -1,
    head: -1,
    title: 'Pick one',
    crumb: 'Sampling',
    matrix: null,
    colSpace: 'none',
    isResidual: false,
    plain:
      'The scores on the last row are turned into percentages and one token is drawn from them. This ' +
      'step is outside the model entirely - the weights have already done all their work and have no ' +
      'say in what happens here. Temperature, top-k and any blocked words all act at this point, which ' +
      'is why they can change the output without changing the model at all.',
    math: 'p = \\operatorname{softmax}(z / \\tau), \\quad y \\sim p',
    code: 'const step = sampleToken(logits, cfg, rand);',
  });

  return out;
}

/** How many stages a config will produce, without running anything. */
export function stageCount(cfg: TransformerConfig): number {
  return 7 + cfg.nLayers * (7 + 3 * cfg.nHeads);
}

/** Group stages into scrubber sections, for a timeline with labelled bands. */
export interface StageBand {
  label: string;
  from: number;
  to: number;
  layer: number;
}

export function bandsFor(stages: Stage[]): StageBand[] {
  const bands: StageBand[] = [];
  let cur: StageBand | null = null;
  stages.forEach((s, i) => {
    const label = s.layer < 0 ? (i < 4 ? 'In' : 'Out') : `Block ${s.layer}`;
    if (!cur || cur.label !== label) {
      cur = { label, from: i, to: i, layer: s.layer };
      bands.push(cur);
    } else {
      cur.to = i;
    }
  });
  return bands;
}

/* ------------------------------------------------------------ the lens -- */

export interface LensEntry {
  id: number;
  prob: number;
}

/**
 * What the model is currently leaning toward, read off a half-finished
 * residual stream.
 *
 * The trick, known as the logit lens, is to take the stream partway through
 * the stack and push it straight through the final normalisation and the
 * output head, skipping every block that has not run yet. The blocks in
 * between were never trained to be read this way, so the answer is not what
 * the model would have said. It is what the model would say if it stopped
 * thinking right now, which is what makes watching it change from block to
 * block interesting.
 *
 * Every number used here is the model's own. Nothing is fitted or guessed.
 */
export function logitLens(
  model: { cfg: TransformerConfig; lnfg: { M: Matrix }; lnfb: { M: Matrix }; head: { M: Matrix }; headb: { M: Matrix } },
  row: Float64Array,
  top = 6,
): LensEntry[] {
  const d = model.cfg.dModel;
  const V = model.cfg.vocab;

  let mean = 0;
  for (let j = 0; j < d; j++) mean += row[j];
  mean /= d;
  let varr = 0;
  for (let j = 0; j < d; j++) varr += (row[j] - mean) ** 2;
  varr /= d;
  const inv = 1 / Math.sqrt(varr + 1e-5);

  const h = new Float64Array(d);
  for (let j = 0; j < d; j++) h[j] = (row[j] - mean) * inv * model.lnfg.M.data[j] + model.lnfb.M.data[j];

  const logits = new Float64Array(V);
  for (let v = 0; v < V; v++) {
    let s = model.headb.M.data[v];
    for (let j = 0; j < d; j++) s += h[j] * model.head.M.data[j * V + v];
    logits[v] = s;
  }

  let max = -Infinity;
  for (let v = 0; v < V; v++) if (logits[v] > max) max = logits[v];
  let sum = 0;
  const exps = new Float64Array(V);
  for (let v = 0; v < V; v++) {
    exps[v] = Math.exp(logits[v] - max);
    sum += exps[v];
  }

  const all: LensEntry[] = [];
  for (let v = 0; v < V; v++) all.push({ id: v, prob: exps[v] / sum });
  all.sort((a, b) => b.prob - a.prob);
  return all.slice(0, top);
}

/** The last row of a stage's matrix: the position that predicts what comes next. */
export function lastRow(m: Matrix): Float64Array {
  const out = new Float64Array(m.cols);
  const off = (m.rows - 1) * m.cols;
  for (let j = 0; j < m.cols; j++) out[j] = m.data[off + j];
  return out;
}
