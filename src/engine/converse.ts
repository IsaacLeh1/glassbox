import type { BPETokenizer } from './tokenizer';
import { sampleToken, type SampleStep, type Steer, type Trace, type Transformer } from './transformer';
import { mulberry32 } from './tensor';

/**
 * Talking to a model and keeping the receipts.
 *
 * Generation normally throws away everything except the token it picked. Here
 * every forward pass is kept, so the whole exchange can be walked through
 * afterwards: what the model saw at each step, what it was thinking inside,
 * what it nearly said, and what it actually said.
 *
 * Keeping traces is not free. One trace holds roughly
 * `context x width x (15 + 4 x heads)` numbers per block, so the caller is
 * expected to cap how many tokens it asks for.
 */

export interface GenStep {
  /** Position in the generated run, from 0. */
  index: number;
  /** The tokens actually fed in, after windowing to the context limit. */
  context: number[];
  /** True when the prompt no longer fits and the oldest tokens were dropped. */
  truncated: boolean;
  trace: Trace;
  probs: number[];
  sample: SampleStep;
  chosen: number;
  /** Milliseconds this single token took. */
  ms: number;
}

export interface ConverseOptions {
  maxTokens: number;
  temperature: number;
  topK: number;
  topP: number;
  seed: number;
  banned?: Set<number>;
  steer?: Steer;
  /** Stop early once this token is produced. */
  stopAt?: number;
}

export const DEFAULT_CONVERSE: ConverseOptions = {
  maxTokens: 24,
  temperature: 0.85,
  topK: 20,
  topP: 0.95,
  seed: 1,
};

export interface Conversation {
  prompt: string;
  promptIds: number[];
  steps: GenStep[];
  /** Just the generated part, decoded. */
  reply: string;
  /** Prompt and reply together, which is what the model actually saw. */
  full: string;
  /** True if the prompt alone was already longer than the context window. */
  promptOverflowed: boolean;
  totalMs: number;
}

/**
 * Run the model forward one token at a time, keeping every intermediate.
 *
 * The windowing here is the thing worth watching. A model has a fixed number
 * of positions; once the conversation is longer than that, the oldest tokens
 * fall off the front and are gone. Nothing warns the model that this happened
 * -- it simply has no way to refer to what it can no longer see.
 */
export function converse(
  model: Transformer,
  tok: BPETokenizer,
  prompt: string,
  opts: ConverseOptions,
): Conversation {
  const t0 = performance.now();
  const promptIds = tok.encode(prompt);
  const block = model.cfg.blockSize;

  // An empty prompt still needs something to condition on, or there is no
  // first position to predict from.
  let ids = promptIds.length > 0 ? [...promptIds] : [0];
  const promptOverflowed = ids.length > block;

  const rand = mulberry32(opts.seed);
  const steps: GenStep[] = [];

  for (let i = 0; i < opts.maxTokens; i++) {
    const s0 = performance.now();
    const context = ids.slice(-block);
    const truncated = ids.length > block;
    const { trace } = model.forward(context, opts.steer);

    const T = context.length;
    const probs: number[] = [];
    for (let j = 0; j < model.cfg.vocab; j++) {
      probs.push(trace.logits.data[(T - 1) * model.cfg.vocab + j]);
    }

    const sample = sampleToken(
      probs,
      {
        temperature: opts.temperature,
        topK: opts.topK,
        topP: opts.topP,
        banned: opts.banned,
      },
      rand,
    );

    steps.push({
      index: i,
      context,
      truncated,
      trace,
      probs,
      sample,
      chosen: sample.chosen,
      ms: performance.now() - s0,
    });

    ids = [...ids, sample.chosen];
    if (opts.stopAt !== undefined && sample.chosen === opts.stopAt) break;
  }

  const generated = ids.slice(promptIds.length > 0 ? promptIds.length : 1);

  return {
    prompt,
    promptIds,
    steps,
    reply: tok.decode(generated),
    full: tok.decode(ids),
    promptOverflowed,
    totalMs: performance.now() - t0,
  };
}

/** Rough count of numbers held by one conversation's traces, for a memory warning. */
export function traceFootprint(model: Transformer, contextLen: number, tokens: number): number {
  const { dModel, nLayers, nHeads, dFF, vocab } = model.cfg;
  const T = Math.min(contextLen, model.cfg.blockSize);
  const perLayer =
    // ln1, attnOut, afterAttn, ln2, ffOut, output, input
    7 * T * dModel +
    T * dFF +
    // per head: scores, att (T x T), and q, k, v, out (T x dHead)
    nHeads * (2 * T * T + 4 * T * Math.floor(dModel / nHeads));
  const outside = 4 * T * dModel + T * vocab;
  return tokens * (nLayers * perLayer + outside);
}

/** The same figure in megabytes, at 8 bytes per number. */
export function footprintMB(model: Transformer, contextLen: number, tokens: number): number {
  return (traceFootprint(model, contextLen, tokens) * 8) / (1024 * 1024);
}
