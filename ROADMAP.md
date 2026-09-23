# Roadmap

Captured direction. Recorded here so the intent survives between working
sessions.

---

## The goal, restated

Glassbox explains how an AI system works. The intended destination is larger
than that in two specific ways:

1. **Walk the user through the exact process a professional enterprise
   business follows to produce a model** — the whole lifecycle, in order, at a
   scale small enough to actually run. Not a tour of concepts, a rehearsal of
   the real job.

2. **Make the user fluent in the language of every AI they interact with.**
   The measure of success is not that they can recite what attention is. It is
   that when they meet any model, anywhere, they recognise what is happening
   and can talk about it precisely.

---

## The Communication module — built

Module 09. The loop described below now exists, operating on the model the
user trained themselves in module 07.

- **See the matrices of their own custom-trained model.** Tab 1 lists every
  parameter tensor in the model with its shape, share of the total, a heatmap
  of the actual values, and what that tensor is for in plain language.
- **Give it a prompt and watch it processed.** Tab 2 generates, keeping a full
  trace per token, and reports how confident the model was at each one and
  what it nearly said instead.
- **Scroll back and forth through that process.** Tab 3 is the scrubber:
  `7 + layers x (7 + 3 x heads)` ordered stages per token, a banded timeline,
  arrow keys, play and pause. Nothing is recomputed — these are the numbers
  from the pass that already happened.
- **See the output**, connected to the internals, with a logit lens on every
  residual stage showing what the model would have said had it stopped there.
- **Intervene, in the same place.** Tab 4: edit any single weight, scale a
  whole matrix, switch an attention head off, block words, steer along a
  difference-in-means direction, or teach it new examples. Every change is
  reversible, and a run log re-runs the identical prompt under the identical
  seed so any difference is attributable to what was changed.

The prerequisite refactor is done: `store/model.ts` holds the one model, and
module 07 publishes to it on build and again when training finishes.

### What is still missing from it

- Only module 09 reads the shared model. Modules 04 and 08 still build their
  own, and should be pointed at the user's model with a fallback.
- Teaching uses a plain fine-tune of the whole model. A frozen-base adapter
  (the LoRA idea already described in the ⓘ copy) would show the standard
  industrial answer to the forgetting problem the module demonstrates.
- The scrubber replays one token at a time. Scrubbing across tokens *and*
  stages on one timeline would show the context window sliding.

---

## The enterprise lifecycle, and what is covered

The real order of work at a company that ships models:

| Stage | Covered |
|---|---|
| 1. Define the problem and what success means | partly — module 10 opens by making you write it down |
| 2. **Design the evaluation before training anything** | yes — module 10 |
| 3. Source, licence, clean and deduplicate data | partly — module 07 loads real data, but does not clean or inspect it |
| 4. Tokenizer decisions | yes — module 04 |
| 5. Choose architecture and size against a compute budget | yes — modules 05 and 07 |
| 6. Pretraining run, with monitoring and incident handling | yes — modules 05 and 07 |
| 7. Evaluate against the held-out set | yes — module 10, with baselines and confidence intervals |
| 8. Post-training: supervised fine-tuning, then preference optimisation | partly — module 09 fine-tunes; no preference optimisation |
| 9. Safety work: red-teaming, guardrails, classifiers | partly — modules 08 and 09 cover mechanisms, not the process |
| 10. Deploy: serving, quantisation, latency and cost per request | no |
| 11. Monitor in production, collect feedback, decide when to retrain | no |

The largest remaining gaps are now **10 and 11**: nothing covers serving a
model, quantising it, or the cost and latency of a single request, and nothing
covers watching it in production and deciding when to retrain. Stage 8 is half
done, since module 09 fine-tunes but no module covers preference optimisation.

---

## A finding from module 10, worth remembering

The built-in corpora are generated from a small template grammar, so the
held-out split is drawn from exactly the same distribution as the training
split and contains nothing unique. A normally-trained model therefore scores
the same on both, and the contamination gap everyone expects does not appear.

This was found by asserting that gap in a test, watching it pass on one seed,
then measuring it properly across three training lengths and two sample sizes,
where it was consistently absent or slightly reversed. Module 10 now measures
the split rather than claiming it, and demonstrates contamination by
fine-tuning directly on the test cases instead, which moves the score from
about 40 percent to about 90 percent and is reversible.

If a corpus with genuine long-tail content is ever added, the natural gap
should appear on its own and that copy can be revisited.

---

## Constraints that apply to anything added here

- Everything computes for real. The only simulated thing in the application is
  the per-device cluster telemetry in module 05, and it is labelled as such.
  Do not add a second exception quietly.
- Every adjustable control gets an ⓘ entry in `src/content/varInfo.ts` with
  what it is and what happens in both directions.
- Plain depth is written for someone who knows nothing, with no undefined
  jargon at that level.
- Callout and info boxes have no coloured left accent bar.
- `engine/` imports nothing from `ui/`.
- Anything exported from a component file breaks Fast Refresh. Shared helpers
  belong in `engine/` or `content/`.
