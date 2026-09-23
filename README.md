# Glassbox

**See inside the machine.** An interactive course in how AI systems actually work, in which
everything on screen is genuinely computing.

Most explanations of AI show you a diagram of a neural network and ask you to imagine the rest.
Glassbox runs the arithmetic. Every weight you drag, every gradient you step through and every
attention head you inspect is a live computation happening in the browser tab. There are no
pre-recorded animations and no scripted numbers.

```bash
pnpm install
pnpm dev
```

---

## The course

| # | Module | What you do |
|---|--------|-------------|
| 01 | **Weights** | Drag the three numbers inside a single neuron and watch the decision boundary rotate and slide. Then open a layered network, click any individual connection, and find the neuron responsible for one specific bend. |
| 02 | **Matrices** | Multiply two matrices by hand with the dot product spelled out term by term. Then watch your own sentence become a one-hot matrix, an embedding, and a chain of real matmuls with live shapes and FLOP counts. Finish by putting those shapes next to GPT-2 and Llama 3. |
| 03 | **Learning** | Run real gradient descent with live loss curves and a boundary that redraws every frame. Step backpropagation one derivative at a time through a rendered computation graph. Then break training on purpose: divergence, dead ReLUs, overfitting, and a model that simply cannot express the answer. |
| 04 | **Language** | A real transformer, trained in the tab. Follow text through byte-pair tokenization, embeddings, every attention head, the residual stream, the logits and the sampling step that picks the next token. |
| 05 | **Scale** | Plan a frontier training run on real hardware. Watch a simulated cluster work the job, lose a GPU, and recover from a checkpoint. |
| 06 | **Optimizing prompts** | Connect your own model, measure it on a task, then let a DSPy optimizer rewrite the prompt and pick its own examples. Compare before and after example by example, and export the equivalent real Python. |
| 07 | **Build your own** | Pull a dataset live from Hugging Face, design a transformer to your own specification, pick a compute device, set a RAM and CPU budget your machine can live with, train it, and measure yourself against a real GPU cluster. |
| 08 | **Guardrails** | Five mechanisms sit between a request and an answer and only one is inside the model. Block tokens live and watch the model route around them, find a behaviour direction inside the network and push along it, and see why there is no list of rules in there. |
| 09 | **Communication** | Open the model you trained in module 07. Read its weights, give it a prompt, then scrub back and forth through the exact forward pass behind every token it produced, with a logit lens showing what it would have said if it had stopped early. Then edit a weight, switch an attention head off, apply a guardrail or teach it something new, and re-run the identical prompt to see what changed. |
| 10 | **Evaluation** | The step almost everyone skips, in the order professionals actually do it. Write down what success means, cut a held-out test set, find out what a bigram lookup table already scores, commit to a bar you cannot see past, and only then run the model. The page counts how many times you change the test after seeing a number. |
| 11 | **Deployment** | Quantise your own weights and measure what the lost precision actually cost, scored on the module 10 harness. Watch a real key-value cache turn quadratic generation into linear, verified to produce identical logits. Then cost the whole thing against published hardware and find the load at which the queue goes vertical. |

Every explanation is written three times. A switch in the sidebar toggles the whole application
between **Plain** (no notation at all), **Math** (the equation behind the step you are looking at)
and **Code** (the source that produced the number on screen).

Every control you can change carries a small ⓘ. Hovering it says what the variable is and what
happens if you move it in either direction.

---

## What is actually real

This is the part that matters, so it is stated precisely.

**Real, and verified by tests:**

- Scalar reverse-mode automatic differentiation (`engine/autograd.ts`), the same technique PyTorch
  uses, small enough that the whole computation graph renders on screen.
- A multi-layer perceptron with hand-written forward and backward passes. Every analytic gradient is
  checked against a central finite-difference estimate, for all three task types.
- A complete decoder-only transformer: multi-head causal attention, layer norm, residual stream,
  GELU feed-forward, learned positional embeddings. **Every parameter tensor** has its analytic
  gradient checked against finite differences, and a test proves the model learns a sequence to a
  loss below 0.05.
- A byte-pair encoding tokenizer trained on whatever corpus you pick, with its merge list exposed.
- SGD, momentum, RMSProp and Adam, plus constant, step, cosine and warmup-cosine schedules and
  global gradient-norm clipping.
- A **WebGPU compute backend**: a WGSL tiled matmul shader, checked for numerical agreement against
  the CPU before it is trusted, and benchmarked end to end including transfer cost.
- **Activation steering** by difference-in-means, the same technique used to locate refusal directions
  in published interpretability work, running live on the in-browser model.
- A **replayable forward pass**: every intermediate of every generated token is kept, ordered
  into the stages the model actually computed, and scrubbed through without recomputing it.
- The **logit lens**: a half-finished residual stream pushed through the final norm and the output
  head, using the model's own weights, to show what it was leaning toward partway through.
- **Head ablation** by zeroing the rows of the output projection that read from one head, which is
  how interpretability work establishes what an individual head contributes.
- **Fine-tuning** an already-trained model on your own examples, with loss on the original corpus
  measured alongside so catastrophic forgetting is visible rather than asserted.
- An **evaluation harness** in which the model and every trivial baseline (uniform, unigram,
  bigram, repeat) are scored through one identical code path, with Wilson score intervals so a
  perfect run on eight cases cannot masquerade as certainty.
- **Contamination measured rather than assumed**: every case is checked against the training text
  verbatim and by overlapping n-grams, and the result is broken down by case origin.
- **Post-training quantisation** that really snaps every weight onto a coarse grid, symmetric and
  linear, per tensor or per row, so the quality cost is genuine and gets re-scored on held-out
  cases. The speed benefit is calculated from byte counts, and the panel says so.
- A **key-value cache**, with a test asserting the cached decode path produces logits identical to
  recomputing the whole context, and a measured speed-up on the reader's own machine.
- **Token-level blocking**, including the awkward part: a word is usually not one token, so the panel
  shows exactly which pieces get struck out and warns when that over-blocks.
- The DSPy reimplementation: signatures, the `[[ ## field ## ]]` chat adapter format,
  BootstrapFewShot, COPRO-style instruction search and a MIPRO-style joint search, all scored
  against a real metric on real held-out data.
- Hugging Face dataset loading, straight from the public dataset viewer API.
- Every capacity figure in module 05: step time, memory footprint, all-reduce volume, wall clock,
  cost and power, computed from published accelerator specifications and the standard
  six-FLOPs-per-parameter-per-token estimate.

**Simulated, and labelled as such in the app:**

- Exactly one thing: the per-device telemetry in module 05 — GPU temperatures, utilisation jitter,
  log ordering and the timing of failures. There is no GPU fleet in a browser tab. The cost and
  timing arithmetic around it is real, and the loss values streaming through its log come from a
  transformer genuinely training on the page while you watch.

**Deliberate trade:** Glassbox stores every number as a 64-bit float for numerical clarity, which is
four times the memory a real BF16 training run uses. The Studio module shows both figures side by
side rather than hiding it.

---

## CPU and GPU

Module 07 detects what this machine has — both WebGPU power preferences and the WebGL renderer
string — and lets you benchmark the CPU against the GPU on the same matrix multiply.

The benchmark is the point. It measures rather than asserts, and on most machines it shows the GPU
losing badly at small sizes and winning enormously at large ones. A representative run on an RTX
5070 Ti Laptop:

| size | CPU | GPU | winner |
|------|-----|-----|--------|
| 32 | 0.14 ms | 3.55 ms | CPU, 25× |
| 128 | 3.3 ms | 4.6 ms | CPU, 1.4× |
| 256 | 29 ms | 3.6 ms | GPU, 8× |
| 1024 | 1357 ms | 24 ms | **GPU, 56×** |

Every GPU dispatch costs a fixed few milliseconds of driver and transfer time, so below roughly
256×256 that overhead exceeds the entire calculation. **Training therefore runs on the CPU**, because
a teaching-sized model multiplies matrices with dimensions in the tens, where the CPU is genuinely
faster. That is not a browser limitation — it is precisely why real training runs use enormous
batches, to hand the hardware matrices big enough to be worth its time. The app says all of this
plainly next to the selector.

---

## Frontier scale

Module 05 extends to runs of three trillion parameters and beyond, and nothing there is a lookup
table. The architecture is inferred from the parameter count using the standard transformer
relation, which as a check reproduces GPT-3 exactly: feed it 175B and it returns 96 layers at width
12288. The failure rate is calibrated against the one large published figure available, Meta's
report of 419 unexpected interruptions over 54 days on roughly 16k H100s for Llama 3 405B.

Sanity check on the projection: at the Llama 3 405B preset it estimates 76.6 days and 15.5 MW on
16,384 H100s, against Meta's reported figure of about 54 days. The same order of magnitude, from
first principles.

The module also answers what tweaking a weight means at that size -- one in three trillion, which
you can feel by nudging one weight in a real 25-parameter network alongside it -- and what is
actually done instead: full fine-tuning, LoRA at a rank you choose, activation steering, or just
changing the prompt.

## Explaining the run

After training starts, module 04 and module 07 analyse the run as it happens:

- Which of five learning stages it has reached — random, character statistics, words, local grammar,
  sentence structure — inferred from loss relative to uniform guessing plus statistics of the
  sampled text.
- What the sample text at the bottom actually is, where it comes from, and why it reads the way it
  does at this point in training.
- Measured properties of that sample: what fraction are real words from the corpus, how much is
  repetition, whether it has fallen into a loop, how its spacing compares to the training text.
- A list of findings specific to this run — stalled, diverging, overfitting, undertrained, spiking
  gradients, looping output — each with what to change.

---

## Layout

```
src/
  engine/       the mathematics, with no UI imports at all
    autograd.ts     scalar reverse-mode autodiff
    tensor.ts       Float64 matrix ops, seeded RNG, softmax
    activations.ts  six activations with analytic derivatives
    mlp.ts          MLP forward/backward + numerical gradient check
    optim.ts        SGD / momentum / RMSProp / Adam, LR schedules
    transformer.ts  full decoder-only transformer, forward and backward
    tokenizer.ts    byte-pair encoding with a replayable merge trace
    corpus.ts       original generated corpora
    trainer.ts      time-sliced MLP training loop
    lmTrainer.ts    time-sliced language-model training loop
    shapes.ts       shape and FLOP bookkeeping for any architecture
    resources.ts    machine detection and memory accounting
    compute.ts      WebGPU device detection, WGSL matmul, benchmarks
    diagnostics.ts  analysis of a live training run
    replay.ts       a forward pass flattened into scrubbable stages, plus the logit lens
    converse.ts     generation that keeps every trace, for replay
    finetune.ts     teaching a trained model, with a forgetting measurement
    evals.ts        baselines, metrics, Wilson intervals, contamination checks
    quantise.ts     real symmetric quantisation, per tensor or per row
    kvcache.ts      incremental decode with a key-value cache, and timing
    serving.ts      memory-bound decode, cost per token, M/M/1 queueing
    steering.ts     difference-in-means directions and token ban resolution
  sim/
    cluster.ts    capacity planning (real) + device telemetry (simulated)
  dspy/
    dspy.ts       signatures, adapter, metrics, evaluation, optimizers
    lmClient.ts   OpenAI-compatible connector
    tasks.ts      four original before/after tasks
    export.ts     emits real Python DSPy
  data/
    huggingface.ts   hub search and dataset viewer loading
  store/        depth and theme, plus the one model shared between modules
  ui/           design system, charts, canvases, network diagram, panels
  modules/      one file per course module
  content/      the ⓘ explanation registry
```

`engine/` imports nothing from `ui/`. It is plain TypeScript and can be lifted out and run in Node.

---

## Connecting your own model (module 06)

Module 06 talks to any OpenAI-compatible `/chat/completions` endpoint: the UVU AI Gateway, OpenAI,
Azure, OpenRouter, Together, Groq, or a model running locally under Ollama or LM Studio.

Your key is held in this browser's local storage and attached only to requests made to the base URL
you entered. It is never sent anywhere else, and it never appears in exported code — the generated
Python reads it from an environment variable instead. A key in browser storage is still a key on
disk, so use a scoped one and rotate it when you are done.

Browsers can only reach endpoints that permit cross-origin requests. Most gateways do. For Ollama,
set `OLLAMA_ORIGINS`.

---

## Tests

```bash
pnpm test
```

286 tests. The ones worth knowing about:

- Analytic gradients checked against central finite differences for the MLP (all three task types,
  with and without L2) and for **every tensor** in the transformer.
- The causal mask verified by showing that changing a later token cannot alter an earlier prediction.
- A network with no hidden layer verified to *fail* on XOR, which is the historical point.
- `paramCountFor()` in the shapes module asserted equal to the real transformer's parameter count,
  so the diagrams can never drift from the implementation.
- Every prompt optimizer verified never to return a program that scored worse than the baseline.
- Formatting tested not to print a misleading `0.0 s` or `$0.00`.
- The stage list for a forward pass asserted to be exactly what the config predicts, to visit blocks
  in order, to hand out the model's own matrices rather than copies, and to have the shape each
  stage claims.
- A conversation verified to grow its context by one token per step, to condition each step on what
  the last one chose, to be reproducible under a fixed seed, and to never exceed the context window.
- Fine-tuning verified to lower the loss on what it is taught and to be undoable weight for weight.
- The Wilson interval asserted not to report zero width at eight passes out of eight, which is the
  exact way a small eval set gets mistaken for a conclusive one.
- A bigram lookup table asserted to beat an untrained transformer, because the module claims it does.
- Training directly on the test cases asserted to inflate the score, and to be exactly reversible.
- The key-value cache asserted to produce logits identical to the uncached path, token after token
  through a whole generated continuation. An optimisation that changes the answer is not one.
- Quantisation asserted to restrict a four-bit tensor to at most 31 distinct values, to lose more at
  every narrower width, and for per-row scaling never to be worse than one scale per tensor.
- The decode bound checked against reality: an 8B model at fp16 on an H100 lands between 100 and 400
  tokens a second, which is where published figures put it.
- Queueing asserted to report saturation rather than a negative wait, and an unachievable latency
  promise as impossible rather than as two billion replicas.
- Regression tests for ten real bugs found during development: a repeat detector that called any
  coincidental word alignment a loop; a diagnostic that scored samples against an empty corpus; a
  warmup schedule that gave step zero a learning rate of exactly zero; a banned token that could
  still occupy a top-k slot; an energy figure inflated a thousandfold by a stray unit conversion;
  a training loop interruptible only between whole steps, which froze the page at long context;
  a forward pass that read past the end of the position table and turned silently into NaN instead
  of failing; a sampler that could not say which of top-k and top-p had actually cut a token; a
  vocabulary check that blamed the user for a newline it had inserted itself; a frame loop that
  detected a stalled page by `document.hidden` alone, so any run froze silently whenever the window
  was merely behind another one; and `predictNext` returning a field called `probs` that had always
  held logits, which every existing caller happened to handle correctly and the first new one did not;
  and `replicasFor` turning an impossible latency promise into two billion machines because of a
  guard against dividing by zero.

---

## Stack

Vite, React, TypeScript, Tailwind v4, Zustand, React Router, WebGPU. No charting or math library:
the visualizations are hand-rolled SVG and canvas so that what is drawn is exactly what was computed.
Builds to static files and deploys anywhere.
