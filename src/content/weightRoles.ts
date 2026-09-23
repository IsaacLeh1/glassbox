/**
 * What each named weight matrix in the transformer is actually for, said
 * without notation.
 *
 * This lives in content rather than in a component because it is copy, and
 * because more than one panel needs it. Exporting it from a component file
 * would also defeat Fast Refresh.
 */
export function roleOf(name: string): string {
  if (name === 'tok_emb')
    return 'The dictionary. One row per vocabulary entry, holding what that piece of text means to this model. Everything downstream is built from these rows.';
  if (name === 'pos_emb')
    return 'The sense of place. One row per slot in the context window, so the model can tell first from fifth. Without it word order would be invisible.';
  if (name.includes('.Wq'))
    return 'Turns each token into a question: what am I looking for? It is compared against every key to decide where to look.';
  if (name.includes('.Wk'))
    return 'Turns each token into a label advertising what it has to offer, so other tokens can find it.';
  if (name.includes('.Wv'))
    return 'Turns each token into the content that actually gets carried away when another token attends to it.';
  if (name.includes('.Wo'))
    return 'Decides how much of each head to believe, and writes the combined result back onto the residual stream.';
  if (name.includes('mlp.W1'))
    return 'Widens each token out on its own, giving the block room to compute. Most of a real model\'s weights live in matrices like this one.';
  if (name.includes('mlp.W2'))
    return 'Narrows the wide version back down so it fits on the residual stream again.';
  if (name.includes('ln1.gain') || name.includes('ln2.gain') || name === 'ln_f.gain')
    return 'A per-column stretch applied after rescaling, so the model can undo the normalisation wherever it turns out not to want it.';
  if (name.includes('.bias') || name.includes('.b1') || name.includes('.b2') || name.includes('.bo'))
    return 'A constant added on regardless of the input. The starting opinion, held before anything is read.';
  if (name === 'head.W')
    return 'The final scorer. Compares the finished state of each token against every vocabulary entry to produce one raw score each.';
  if (name === 'head.b')
    return 'A standing preference for or against each vocabulary entry, applied to every position equally.';
  return 'A learned weight matrix.';
}
