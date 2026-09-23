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

Step 09. The loop described below now exists, operating on the model the
user trained themselves in step 07.

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
step 07 publishes to it on build and again when training finishes.

### What is still missing from it

- Only step 09 reads the shared model. Modules 04 and 08 still build their
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
| 1. Define the problem and what success means | partly — step 10 opens by making you write it down |
| 2. **Design the evaluation before training anything** | yes — step 10 |
| 3. Source, licence, clean and deduplicate data | partly — step 07 loads real data, but does not clean or inspect it |
| 4. Tokenizer decisions | yes — step 04 |
| 5. Choose architecture and size against a compute budget | yes — steps 05 and 07 |
| 6. Pretraining run, with monitoring and incident handling | yes — steps 05 and 07 |
| 7. Evaluate against the held-out set | yes — step 10, with baselines and confidence intervals |
| 8. Post-training: supervised fine-tuning, then preference optimisation | partly — step 09 fine-tunes; no preference optimisation |
| 9. Safety work: red-teaming, guardrails, classifiers | partly — steps 08 and 09 cover mechanisms, not the process |
| 10. Deploy: serving, quantisation, latency and cost per request | yes — step 11 |
| 11. Monitor in production, collect feedback, decide when to retrain | yes — step 12 |

**Every stage of the lifecycle is now covered at least once**, and the loop
closes: step 12 sends the reader back to step 10 to recheck the bar, to
step 07 to retrain, or to step 09 to teach.

Two stages are covered only in part, and both are worth finishing:

- **Stage 8** &mdash; step 09 does supervised fine-tuning, but nothing covers
  preference optimisation, which is how a model is actually shaped after
  pretraining. A DPO-style pairwise preference exercise on the user's own
  model would complete it, and the machinery is mostly there already.
- **Stage 3** &mdash; step 07 loads real data from Hugging Face and never
  inspects it. Deduplication, licence checking and a look at what is actually
  in the corpus is a real day of work at a real company and currently absent.

---

## Notes from step 12

Monitoring needed a proxy signal that genuinely moves, not one asserted to.
It does: a model trained on the stories corpus is measurably more surprised by
weather-log traffic, and its true quality on that traffic really falls. A test
pins both directions, because the module teaches something false if the proxy
does not track the truth.

Window scoring uses one forward pass for the whole sequence rather than one
per token, which is about ten times less work and matters because the
dashboard streams. A test asserts the fast path equals token-by-token teacher
forcing.

The true-quality column exists only because the experiment was constructed.
The module shows it deliberately, so the proxies can be judged, and then tells
the reader to hide it again.

---

## Notes from step 11

Quantisation is applied for real, so the quality cost is genuine, but the
engine computes in Float64 throughout and has no integer kernels. The speed
benefit is therefore calculated from byte counts rather than observed, and the
panel states this plainly. Adding real int8 kernels would make it measurable
and is the obvious extension.

The cost page offers published model shapes to cost against, because a
53-thousand-parameter model costs nothing to serve and every figure rounded to
zero. The arithmetic is identical at any size; only the shapes change.

---

## A finding from step 10, worth remembering

The built-in corpora are generated from a small template grammar, so the
held-out split is drawn from exactly the same distribution as the training
split and contains nothing unique. A normally-trained model therefore scores
the same on both, and the contamination gap everyone expects does not appear.

This was found by asserting that gap in a test, watching it pass on one seed,
then measuring it properly across three training lengths and two sample sizes,
where it was consistently absent or slightly reversed. Step 10 now measures
the split rather than claiming it, and demonstrates contamination by
fine-tuning directly on the test cases instead, which moves the score from
about 40 percent to about 90 percent and is reversible.

If a corpus with genuine long-tail content is ever added, the natural gap
should appear on its own and that copy can be revisited.

---

## Constraints that apply to anything added here

- Everything computes for real. The only simulated thing in the application is
  the per-device cluster telemetry in step 05, and it is labelled as such.
  Do not add a second exception quietly.
- Every adjustable control gets an ⓘ entry in `src/content/varInfo.ts` with
  what it is and what happens in both directions.
- Plain depth is written for someone who knows nothing, with no undefined
  jargon at that level.
- Callout and info boxes have no coloured left accent bar.
- `engine/` imports nothing from `ui/`.
- Anything exported from a component file breaks Fast Refresh. Shared helpers
  belong in `engine/` or `content/`.
