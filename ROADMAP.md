# Roadmap

Captured direction, **not implemented**. Recorded here so the intent survives
between working sessions.

---

## The goal, restated

Glassbox today explains how an AI system works. The intended destination is
larger than that in two specific ways:

1. **Walk the user through the exact process a professional enterprise
   business follows to produce a model** — the whole lifecycle, in order, at a
   scale small enough to actually run. Not a tour of concepts, a rehearsal of
   the real job.

2. **Make the user fluent in the language of every AI they interact with.**
   The measure of success is not that they can recite what attention is. It is
   that when they meet any model, anywhere, they recognise what is happening
   and can talk about it precisely.

---

## The missing centrepiece: a Communication module

An all-inclusive loop in one place, operating on the model the user trained
themselves in module 07. Today the pieces are spread across modules and each
uses its own model; this would unify them around a single model the user owns.

What it needs to do, in order:

- **See the matrices of their own custom-trained model.** Not a generic
  example — the actual weights that came out of their run.
- **Give it a prompt and watch it processed live**, inside the model, as it
  happens.
- **Scroll back and forth through that process.** A timeline or scrubber over
  every stage of the forward pass, so nothing has to be caught in real time.
  This is the part that does not exist anywhere yet: current panels show a
  static forward pass, not a replayable one.
- **See the output**, connected visibly to the internals that produced it.
- **Then intervene, in the same place:**
  - modify weights directly and re-run,
  - apply or remove guardrails,
  - teach it something new.

The important property is that it is one continuous loop — observe, intervene,
re-observe — rather than separate exercises. That loop is the thing that builds
fluency.

### Notes toward building it

- Module 04 already produces a full `Trace` for every forward pass, containing
  every intermediate. A scrubber is mostly a UI problem over data that already
  exists, plus retaining traces per generated token rather than discarding them.
- Module 08 already has real token blocking and real activation steering, and
  both take a model as an argument. They should point at the user's model.
- "Teach it something new" means a short fine-tune on user-supplied examples
  against the existing weights, not a fresh run. The trainer supports resuming;
  it needs a UI and a way to add examples.
- Sharing one trained model across modules means lifting it out of per-module
  refs into a store. That refactor should come first, or it will have to be
  undone later.

---

## The enterprise lifecycle, and what is already covered

The real order of work at a company that ships models, with current coverage:

| Stage | Covered today |
|---|---|
| 1. Define the problem and what success means | no |
| 2. **Design the evaluation before training anything** | no |
| 3. Source, licence, clean and deduplicate data | partly — module 07 loads real data, but does not clean or inspect it |
| 4. Tokenizer decisions | yes — module 04 |
| 5. Choose architecture and size against a compute budget | yes — modules 05 and 07 |
| 6. Pretraining run, with monitoring and incident handling | yes — modules 05 and 07 |
| 7. Evaluate against the held-out set | partly — loss and perplexity only, no task evals |
| 8. Post-training: supervised fine-tuning, then preference optimisation | no |
| 9. Safety work: red-teaming, guardrails, classifiers | partly — module 08 covers mechanisms, not the process |
| 10. Deploy: serving, quantisation, latency and cost per request | no |
| 11. Monitor in production, collect feedback, decide when to retrain | no |

The largest gaps are **2, 8, 10 and 11**. Of those, designing the evaluation
first is the one most worth adding: it is the step beginners skip and
professionals never do, and it reframes everything downstream.

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
