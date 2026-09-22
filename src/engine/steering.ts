import type { BPETokenizer } from './tokenizer';
import type { Transformer } from './transformer';

/**
 * Activation steering: finding a direction inside the model that corresponds
 * to a behaviour, and then pushing along it.
 *
 * The method here is difference-in-means, which is the same technique used in
 * published work on refusal directions. Run the model over two sets of
 * examples that differ only in the property of interest, average the internal
 * state for each set, and subtract. What is left points from one behaviour
 * toward the other.
 *
 * No weights are touched. The model is unchanged; only its running internal
 * state is nudged while it computes.
 */

export interface Direction {
  vec: Float64Array;
  layer: number;
  /** How far apart the two groups were before normalising: a confidence signal. */
  separation: number;
  countA: number;
  countB: number;
}

/** Mean residual-stream vector across all positions of one sequence. */
function meanResidual(model: Transformer, tokens: number[], layer: number): Float64Array | null {
  const ctx = tokens.slice(0, model.cfg.blockSize);
  if (ctx.length === 0) return null;
  const { trace } = model.forward(ctx);
  const m = layer < 0 ? trace.embedded : trace.layers[Math.min(layer, trace.layers.length - 1)].output;
  const out = new Float64Array(m.cols);
  for (let i = 0; i < m.rows; i++) {
    for (let j = 0; j < m.cols; j++) out[j] += m.data[i * m.cols + j];
  }
  for (let j = 0; j < out.length; j++) out[j] /= m.rows;
  return out;
}

/**
 * The direction that points from group B toward group A, at a chosen layer.
 * Adding it should make the model behave more like group A.
 */
export function contrastDirection(
  model: Transformer,
  tok: BPETokenizer,
  setA: string[],
  setB: string[],
  layer: number,
): Direction | null {
  const d = model.cfg.dModel;
  const accum = (texts: string[]) => {
    const sum = new Float64Array(d);
    let n = 0;
    for (const t of texts) {
      const ids = tok.encode(t.trim());
      if (ids.length === 0) continue;
      const m = meanResidual(model, ids, layer);
      if (!m) continue;
      for (let j = 0; j < d; j++) sum[j] += m[j];
      n++;
    }
    if (n === 0) return null;
    for (let j = 0; j < d; j++) sum[j] /= n;
    return { mean: sum, n };
  };

  const a = accum(setA);
  const b = accum(setB);
  if (!a || !b) return null;

  const vec = new Float64Array(d);
  let norm = 0;
  for (let j = 0; j < d; j++) {
    vec[j] = a.mean[j] - b.mean[j];
    norm += vec[j] * vec[j];
  }
  norm = Math.sqrt(norm);
  if (norm < 1e-9) return null;
  // Unit length, so the strength slider means the same thing regardless of
  // how far apart the two groups happened to be.
  for (let j = 0; j < d; j++) vec[j] /= norm;

  return { vec, layer, separation: norm, countA: a.n, countB: b.n };
}

/** How strongly one sequence already points along a direction. */
export function projectOnto(
  model: Transformer,
  tok: BPETokenizer,
  text: string,
  dir: Direction,
): number {
  const ids = tok.encode(text.trim());
  if (ids.length === 0) return 0;
  const m = meanResidual(model, ids, dir.layer);
  if (!m) return 0;
  let dot = 0;
  for (let j = 0; j < m.length; j++) dot += m[j] * dir.vec[j];
  return dot;
}

export interface BanResolution {
  fragment: string;
  /** Vocabulary entries that will be struck out for this fragment. */
  ids: number[];
  /** Human-readable form of those entries. */
  pieces: string[];
  /** True when the word is not a single token and had to be blocked by its first piece. */
  split: boolean;
}

/**
 * Work out which vocabulary entries to strike out for a list of words.
 *
 * This is less obvious than it sounds, and the awkwardness is the lesson. A
 * word is often not a single token: "garden" may be stored as two or three
 * pieces. Blocking by substring alone therefore matches nothing and silently
 * does no work at all.
 *
 * So two things are blocked. Any entry containing the word outright, and the
 * first piece of the word's tokenization, which prevents it from ever being
 * started. The second is effective but blunt: it also blocks every other word
 * beginning with the same piece.
 */
export function resolveBan(tok: BPETokenizer, fragments: string[]): BanResolution[] {
  const out: BanResolution[] = [];
  for (const raw of fragments) {
    const frag = raw.trim().toLowerCase();
    if (!frag) continue;
    const ids = new Set<number>();

    // Entries that contain the whole word.
    for (let id = 0; id < tok.vocab.length; id++) {
      const s = (tok.vocab[id] ?? '').toLowerCase();
      if (s.trim() && s.includes(frag)) ids.add(id);
    }

    // The opening piece, for both the bare word and the spaced form, since
    // those tokenize differently.
    let split = false;
    for (const form of [frag, ` ${frag}`]) {
      const seq = tok.encode(form);
      if (seq.length === 0) continue;
      if (seq.length > 1) split = true;
      ids.add(seq[0]);
    }

    const list = [...ids].sort((a, b) => a - b);
    out.push({
      fragment: frag,
      ids: list,
      pieces: list.map((id) => tok.display(id)),
      split,
    });
  }
  return out;
}

/** Flat set of every token id a ban list resolves to. */
export function matchTokens(tok: BPETokenizer, fragments: string[]): number[] {
  const seen = new Set<number>();
  for (const r of resolveBan(tok, fragments)) for (const id of r.ids) seen.add(id);
  return [...seen];
}
