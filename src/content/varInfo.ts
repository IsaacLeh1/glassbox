import type { VarInfo } from '../ui/kit';

/**
 * Every control in Glassbox that the reader can change carries one of these.
 * They are kept in one file so the wording stays consistent and so it is easy
 * to check that no adjustable variable has been left unexplained.
 */
export const V = {
  /* ------------------------------------------------ module 01: weights -- */

  weight1: {
    what: 'How much the first input counts. That input gets multiplied by this number before everything is added up, so this is the amount of say it gets in the final answer.',
    up: 'That input gets more say, so the answer swings further whenever it changes. The line where the two colours meet turns like a compass needle.',
    down: 'Less say. Below zero the input starts arguing the opposite way, so what used to raise the answer now lowers it, and the two colours swap sides.',
    note: 'Weights set direction. Only the bias moves the boundary without turning it.',
  },
  weight2: {
    what: 'How much the second input counts. Exactly the same job as the first weight, just for the other input.',
    up: 'The second input gets more say, and the dividing line turns the other way from the first weight.',
    down: 'Less say, and below zero that input flips which answer it argues for.',
    note: 'The pair of weights together forms a vector, and the boundary is always perpendicular to it.',
  },
  bias: {
    what: 'A number added on at the very end, after the inputs have been multiplied and added together. It belongs to no input, so it is the starting opinion the neuron holds before it looks at anything.',
    up: 'The neuron says yes more readily, even when both inputs are small or zero. The dividing line slides across the picture without turning at all.',
    down: 'The neuron gets harder to convince. Push it far enough down and it answers no nearly everywhere, whatever the inputs say.',
    note: 'Bias is what lets a neuron fire when every input is exactly zero.',
  },
  activation: {
    what: 'The squashing step applied to the total once everything has been added up. It is the reason stacking layers buys you anything: without it, a hundred layers do exactly what one layer does.',
    up: 'Saturating choices such as sigmoid and tanh squash the output into a fixed range, which keeps things stable but flattens gradients at the extremes.',
    down: 'ReLU and its relatives leave positives untouched and zero out negatives, which trains faster but can kill neurons permanently.',
    note: 'Choose linear and the entire network collapses to a single layer, no matter how many you stack.',
  },
  selectedWeight: {
    what: 'The one connection you have clicked on. This is the actual number the model stores for it, and dragging this slider edits the model directly.',
    up: 'The signal arriving along this connection is amplified, strengthening whatever feature the receiving neuron detects.',
    down: 'Passing through zero severs the connection entirely; going negative makes the receiving neuron respond in the opposite direction.',
    note: 'Watch the decision boundary as you drag. One weight out of many usually controls one visible bend.',
  },
  dataset: {
    what: 'The pattern of dots the network is being asked to tell apart. Some shapes can be split with a single straight line; others cannot, and need a bigger network.',
    up: 'Harder shapes such as spirals need more hidden units and more training before the boundary can follow them.',
    down: 'Simpler shapes such as a straight split can be solved by a single neuron with no hidden layer at all.',
    note: 'If a network cannot solve something, always ask whether it has the capacity to express the answer before blaming training.',
  },
  hiddenLayers: {
    what: 'How many rows of neurons sit between the input and the answer, and how many neurons are in each row. More of them means the model can describe more complicated shapes.',
    up: 'More capacity can represent more intricate boundaries, but it also makes memorising the training data easier and training slower.',
    down: 'Less capacity trains fast and generalises willingly, until the problem is simply too curved for it and accuracy stalls.',
    note: 'With no hidden layer at all the model can only ever draw one straight line.',
  },
  hiddenActivation: {
    what: 'Which squashing step the middle layers use. Different choices give boundaries that are smoothly curved or made of sharp straight pieces.',
    up: 'Smooth functions such as tanh and GELU give gentle, curved boundaries and well-behaved gradients.',
    down: 'ReLU produces boundaries built from sharp straight segments, and trains faster on deep stacks.',
    note: 'ReLU with a high learning rate is the usual cause of dead neurons. Leaky ReLU avoids that.',
  },

  /* --------------------------------------------- module 03: training --- */

  optimizer: {
    what: 'The rule for turning "this weight is wrong by this much" into an actual change to that weight. Some rules just step; others remember what happened on previous steps.',
    up: 'Adaptive methods such as Adam adjust the step size per parameter and usually converge in far fewer steps with far less tuning.',
    down: 'Plain SGD makes the same sized step everywhere. It is slower and fussier, but it makes what is happening much easier to see.',
    note: 'Adam is the default for almost every large model trained today.',
  },
  learningRate: {
    what: 'How big a correction the model makes each time it learns from a mistake. It is the single most consequential number in training, and the easiest one to get wrong.',
    up: 'Faster progress at first, then chaos. Set it too high and every correction overshoots, which makes the next mistake bigger, which makes the next correction worse. The run destroys itself within seconds.',
    down: 'Every correction is real but microscopic. The model would get there eventually, after far more steps than anyone is going to sit and wait for.',
    note: 'The usable band is narrower than people expect. Raise it until training becomes unstable, then back off.',
  },
  batchSize: {
    what: 'How many examples the model looks at before it changes anything. It averages what it learns from all of them, then makes one correction.',
    up: 'Smoother, more reliable gradients and better hardware utilisation, but fewer updates for the same amount of data and often slightly worse generalisation.',
    down: 'Noisier gradients, which sounds bad but often helps: the noise shakes the model out of poor minima. Training becomes jittery and slow per example.',
    note: 'Batch size and learning rate interact. If you raise one substantially, revisit the other.',
  },
  epochs: {
    what: 'How many times the model goes through the whole set of training examples from start to finish.',
    up: 'More chances to fit the data. Past a point the training loss keeps falling while held-out performance quietly gets worse.',
    down: 'The run stops before the model has extracted what is there, and both training and validation accuracy are left on the table.',
    note: 'Watch the validation curve rather than the clock. The right moment to stop is when it stops improving.',
  },
  lrSchedule: {
    what: 'How the learning rate changes over the course of the run.',
    up: 'Decaying schedules take large early steps and small late ones, which lets the model settle precisely instead of bouncing around the minimum.',
    down: 'A constant rate is simplest and fine for small problems, but it tends to leave a model oscillating near the end.',
    note: 'Warmup plus cosine decay is near universal for large transformers, because a freshly initialised model reacts badly to full-size steps.',
  },
  l2: {
    what: 'A gentle pressure that keeps every weight small unless it has really earned being large. It stops the model from latching onto individual examples.',
    up: 'Weights are pulled toward zero, boundaries get smoother and overfitting drops. Push it too far and the model becomes too rigid to fit anything.',
    down: 'At zero there is no pressure at all, so the model is free to grow large weights and memorise individual training points, including their noise.',
    note: 'This is the first thing to reach for when the validation loss starts rising while the training loss keeps falling.',
  },
  clipNorm: {
    what: 'A cap on the total length of the gradient vector before it is applied. If the gradient is longer than this, it is scaled down.',
    up: 'A loose cap rarely triggers and leaves training unchanged.',
    down: 'A tight cap limits how much damage any single bad batch can do, at the cost of slowing genuine progress when gradients are legitimately large.',
    note: 'Essentially every large training run uses this. It converts a catastrophic divergence into a merely slow step.',
  },
  labelNoise: {
    what: 'How much random scatter is added to the data, which makes some points genuinely unlearnable.',
    up: 'The classes overlap, so perfect accuracy becomes impossible. A model that still reaches 100 percent on the training set is memorising noise.',
    down: 'Clean, well-separated data. Easy to fit, and much less useful for seeing how overfitting behaves.',
    note: 'Raise this and add a large network to produce overfitting on demand.',
  },
  probeInput: {
    what: 'One input value fed into the little network so you can follow a single example through it.',
    up: 'Larger inputs produce larger weighted sums, pushing saturating activations toward their flat regions where gradients nearly vanish.',
    down: 'Negative inputs flip the sign of every product they take part in, which flips the direction each weight pushes.',
    note: 'An input of exactly zero makes the gradient for its weight exactly zero too, since that weight had no effect.',
  },
  targetY: {
    what: 'The correct answer for this example. The loss measures the distance between the prediction and this.',
    up: 'A target further from the current prediction produces a larger error and therefore larger gradients everywhere.',
    down: 'Moving the target onto the current prediction drives the loss and every gradient to zero: nothing to learn.',
    note: 'The gradient at the output is simply prediction minus target. Everything else is that number propagated backwards.',
  },

  /* ---------------------------------------------- module 02: matrices -- */

  matRowsA: {
    what: 'The number of rows in the left matrix. In a model this is usually the number of tokens or examples being processed at once.',
    up: 'More rows means more output rows and proportionally more work. Nothing about the operation changes.',
    down: 'Fewer rows is less work. A single row is the case of running one example on its own.',
    note: 'Rows of the left matrix pass straight through to rows of the result.',
  },
  matShared: {
    what: 'The shared inner dimension. It must match on both sides, and it disappears from the answer.',
    up: 'Each output cell sums over more terms, so the cost rises linearly and the values typically grow.',
    down: 'Shorter dot products, less work, smaller values.',
    note: 'This is the dimension that causes almost every shape error in real code.',
  },
  matColsB: {
    what: 'The number of columns in the right matrix. In a model this is the width of the layer being projected into.',
    up: 'A wider output, more parameters and more work, but also more room to represent distinct features.',
    down: 'A narrower output forces information through a bottleneck, which can be deliberate or can lose what you needed.',
    note: 'Columns of the right matrix pass straight through to columns of the result.',
  },
  seqLength: {
    what: 'How many tokens are being processed at once in the comparison.',
    up: 'Most costs grow in direct proportion, but attention grows with the square, so long sequences are disproportionately expensive.',
    down: 'Shorter sequences are cheap, but the model has less context to condition on.',
    note: 'This is exactly why long-context models are priced the way they are.',
  },
  localModelSize: {
    what: 'Which of the three in-browser model sizes to use for the demonstration.',
    up: 'Wider and deeper means better text and more interesting attention patterns, at the cost of slower training in this tab.',
    down: 'Smaller trains almost instantly and is easier to read on screen, but the output stays crude.',
    note: 'Every size here is real and trains for real. The differences are only in the dimensions.',
  },

  /* ---------------------------------------------- module 04: language -- */

  corpus: {
    what: 'The body of text the tokenizer and the model are both built from.',
    up: 'More structured, repetitive corpora let a tiny model show visible progress within seconds.',
    down: 'Freer, more varied text is a much harder target and a small model will only manage rough statistics.',
    note: 'The tokenizer is retrained whenever this changes, so the vocabulary changes with it.',
  },
  temperature: {
    what: 'How adventurous the model is allowed to be when picking each word. It changes nothing about what the model knows, only how boldly it chooses between the options it is already considering.',
    up: 'It starts taking chances on words it thinks are unlikely. Output gets more varied and surprising, and past about 1.3 it stops making sense.',
    down: 'It plays safe and takes its top choice nearly every time. Output becomes predictable, then repetitive, then gets stuck repeating one phrase forever.',
    note: 'At very low temperature this is effectively greedy decoding.',
  },
  topK: {
    what: 'Keeps only the k most likely tokens and discards the rest before sampling.',
    up: 'A larger pool allows more variety, including the long tail of odd choices.',
    down: 'A tighter pool blocks unlikely tokens entirely. At k equals 1 the model always takes its top choice.',
    note: 'A fixed k is blunt: the same number of options survives whether the model is certain or not.',
  },
  topP: {
    what: 'Keeps the smallest set of tokens whose probabilities add up to p, then samples from those.',
    up: 'Closer to 1 keeps nearly everything, so this filter stops doing much.',
    down: 'Lower values keep only the strongest candidates and cut the tail hard.',
    note: 'Unlike top-k this adapts automatically: narrow when the model is confident, wide when it is unsure.',
  },
  attentionLayer: {
    what: 'Which layer of the stack to inspect.',
    up: 'Later layers work with representations that earlier layers have already processed, and their attention tends to be more abstract.',
    down: 'Early layers attend to surface patterns: the previous token, matching punctuation, simple position.',
    note: 'Layers do not divide neatly into jobs, but early and late do reliably look different.',
  },
  attentionHead: {
    what: 'Which of the parallel attention heads within that layer to inspect.',
    up: 'Different heads within one layer specialise. Stepping through them shows genuinely different patterns.',
    down: 'Head zero is no more important than any other; the ordering carries no meaning.',
    note: 'Heads split the model width between them, so more heads means each is narrower.',
  },

  /* ------------------------------------------------- module 05: scale -- */

  paramsB: {
    what: 'How many learned parameters the hypothetical model has. The main driver of both capability and cost.',
    up: 'Compute, memory and money all rise in direct proportion, and the tokens needed to train it properly rise too.',
    down: 'A smaller model is cheaper in every dimension and can be trained on hardware you actually have.',
    note: 'Roughly twenty training tokens per parameter is the usual guidance for a well-trained model.',
  },
  trainTokens: {
    what: 'How much text the run will process in total.',
    up: 'A better trained model, with wall clock and cost rising in direct proportion.',
    down: 'A shorter, cheaper run that leaves the model undertrained: it has the capacity to be better but never saw enough data.',
    note: 'For a fixed budget there is a real trade between making the model bigger and training it for longer.',
  },
  globalBatch: {
    what: 'How many tokens are processed across the whole cluster before the weights are updated once.',
    up: 'Better hardware utilisation and a lower share of time spent communicating, but more memory per device and fewer total updates.',
    down: 'More frequent updates and less memory pressure, but the fixed cost of synchronising between devices starts to dominate.',
    note: 'Large runs use enormous batches, often millions of tokens, mostly to keep the interconnect from becoming the bottleneck.',
  },
  accelerator: {
    what: 'Which GPU the plan assumes, with its published throughput, memory, bandwidth and rental price.',
    up: 'Faster chips shorten the run but cost more per hour. They are usually cheaper per unit of work even so.',
    down: 'Cheaper chips need more of them and more wall clock, and their smaller memory may not fit the model at all.',
    note: 'Memory capacity, not speed, is what usually decides whether a given model can be trained on a given chip.',
  },
  nodes: {
    what: 'How many machines are in the cluster. Each holds several accelerators.',
    up: 'More total throughput, but communication between machines is far slower than within one, so returns diminish.',
    down: 'Fewer machines means less coordination overhead and a simpler job, but a longer run.',
    note: 'Watch the compute against communication bars. Once communication dominates, adding nodes stops helping.',
  },
  gpusPerNode: {
    what: 'How many accelerators sit inside each machine, connected by fast local links.',
    up: 'Filling a node is efficient, because devices inside one machine talk over a much faster link than devices in different machines.',
    down: 'Partly filled nodes waste the fast local interconnect you are paying for.',
    note: 'Eight per node is the standard configuration for training hardware.',
  },
  precision: {
    what: 'How many bits each number uses. Lower precision means less memory and faster arithmetic.',
    up: 'Higher precision is numerically safer and easier to debug, but doubles memory and halves throughput.',
    down: 'Lower precision roughly doubles speed and halves memory per step down, but needs loss scaling and careful monitoring to stay stable.',
    note: 'BF16 is the standard for training. FP8 is increasingly common but demands more care.',
  },
  parallelism: {
    what: 'How the model and the work are divided across devices.',
    up: 'More aggressive sharding reduces memory per device dramatically, at the cost of substantially more communication.',
    down: 'Plain data parallelism is the simplest and fastest per step, but every device must hold the entire model plus its optimizer state.',
    note: 'The choice is usually forced: you shard when the model stops fitting, not because you want to.',
  },
  mfu: {
    what: 'Model FLOPs utilization: the fraction of the hardware peak the run actually achieves.',
    up: 'Optimistic assumptions make the plan look cheap and fast. Very few real runs exceed about fifty percent.',
    down: 'Pessimistic assumptions inflate the estimate. Poorly tuned jobs really do land in the teens.',
    note: 'Between 35 and 50 percent is the honest range for a well-engineered large training run.',
  },
  simSpeed: {
    what: 'How much faster the simulation runs than the real job would.',
    up: 'Days of a real run compress into seconds, so you can watch checkpoints and failures arrive.',
    down: 'Closer to real time, which makes the individual phases of a single step visible.',
    note: 'This only changes the playback rate. It does not change any of the projected figures.',
  },

  /* -------------------------------------------------- module 06: dspy -- */

  lmBaseUrl: {
    what: 'The root address of an OpenAI-compatible API. The path /chat/completions is appended to it.',
    up: 'Pointing at a hosted gateway gives you strong models with per-token billing.',
    down: 'Pointing at localhost uses a model running on your own machine: free, private, and usually weaker.',
    note: 'It should normally end in /v1. A browser can only reach endpoints that permit cross-origin requests.',
  },
  lmModel: {
    what: 'Which model name to request from that endpoint.',
    up: 'Stronger models score higher before optimization, which leaves less headroom for the optimizer to demonstrate.',
    down: 'Weaker models start lower and usually gain far more from optimization, which makes the effect easier to see.',
    note: 'The most striking before-and-after results come from small models, not large ones.',
  },
  lmTemperature: {
    what: 'Sampling randomness for every call made during evaluation and optimization.',
    up: 'Above zero the same prompt gives different answers each time, so score differences become partly noise.',
    down: 'At zero the model is deterministic and any change in score is attributable to the prompt.',
    note: 'Keep this at zero while optimizing. Raise it afterwards if you want variety in production.',
  },
  lmMaxTokens: {
    what: 'The ceiling on how long each reply may be.',
    up: 'Room for chain-of-thought reasoning to finish. Cut it short and the answer field never arrives.',
    down: 'Cheaper and faster, but truncated replies are scored as failures even when the reasoning was correct.',
    note: 'Chain of thought needs noticeably more headroom than a bare answer.',
  },
  chainOfThought: {
    what: 'Adds a reasoning field that the model must fill in before it gives its answer.',
    up: 'Turning it on usually helps anything requiring more than one step, and gives the optimizer richer traces to bootstrap from.',
    down: 'Turning it off is cheaper and faster, and for pure classification or formatting it often makes no difference at all.',
    note: 'With this on, bootstrapped demonstrations contain reasoning the model itself produced, which suits it better than reasoning written by a human.',
  },
  metric: {
    what: 'How an answer is scored against the label. This is the objective the optimizer maximises.',
    up: 'A lenient metric such as token overlap gives partial credit, so progress is visible in small increments.',
    down: 'A strict metric such as exact match gives nothing for a nearly right answer, which makes formatting failures very obvious.',
    note: 'The optimizer will improve exactly what you measure. Choosing the metric is the most consequential decision here.',
  },
  optimizerChoice: {
    what: 'Which search procedure to run over the prompt.',
    up: 'Joint search over instructions and demonstrations explores the most and can find the best result, but costs the most calls.',
    down: 'Simply pasting in labelled examples costs almost nothing and is often most of the available gain.',
    note: 'Start with the cheap one. Only reach for the expensive search if the cheap one leaves something on the table.',
  },
  maxDemos: {
    what: 'The maximum number of worked examples to place in the prompt.',
    up: 'More examples pin down the expected format and edge cases more firmly, but every one is paid for in prompt tokens on every single call thereafter.',
    down: 'Fewer examples keep calls cheap and fast, but leave more room for the model to answer in the wrong shape.',
    note: 'Gains usually flatten out somewhere between four and eight examples.',
  },
  candidatesPerRound: {
    what: 'How many alternative prompts to generate and score in each round of the search.',
    up: 'A wider search is more likely to find something good, and costs proportionally more model calls.',
    down: 'A narrow search is cheap but may never propose anything better than where it started.',
    note: 'Each candidate is scored on the full validation set, so this multiplies the cost directly.',
  },
  rounds: {
    what: 'How many times to repeat the propose-and-score cycle, each round informed by the scores of the last.',
    up: 'Later rounds can build on what worked, which is where coordinate-ascent methods earn their keep.',
    down: 'A single round is a one-shot guess with no feedback from the results.',
    note: 'Cost is roughly candidates multiplied by rounds multiplied by the size of the validation set.',
  },
  instructions: {
    what: 'The natural-language objective placed at the top of the prompt. It is a parameter, not a fixed part of the framework.',
    up: 'More specific instructions about format and method usually help a great deal, especially for weaker models.',
    down: 'A vague instruction leaves the model to guess the format, which is the most common reason a task scores zero.',
    note: 'The COPRO and MIPRO optimizers will rewrite this for you and keep whatever measures better.',
  },

  /* ------------------------------------------------- module 07: studio -- */

  hfDataset: {
    what: 'A dataset on the Hugging Face hub, loaded live over the public dataset viewer API.',
    up: 'Larger and more varied datasets teach more, but take longer to download and much longer to train on.',
    down: 'Small, repetitive datasets let a tiny model reach recognisable output within seconds.',
    note: 'Only public, non-gated datasets can be read from a browser.',
  },
  maxChars: {
    what: 'How much text to download and train on, measured in characters.',
    up: 'More data means less overfitting and better output, at the cost of a longer download and slower tokenizer training.',
    down: 'A small sample loads instantly but a model will memorise it rather than learn from it.',
    note: 'A few hundred thousand characters is plenty for a model of this size.',
  },
  ramBudget: {
    what: 'The ceiling you are allowing this tab to use. Glassbox checks every configuration against it before letting training start.',
    up: 'A larger budget permits a bigger model, up to whatever the browser itself will allow.',
    down: 'A tighter budget keeps the rest of your machine responsive and stops the tab being killed, but caps model size.',
    note: 'Browsers cap a single tab regardless of installed RAM. The measured ceiling is shown beside this control.',
  },
  frameBudget: {
    what: 'How many milliseconds of each animation frame training is allowed to consume.',
    up: 'More of your CPU goes to training, so it finishes sooner and the interface becomes less responsive.',
    down: 'The interface stays perfectly smooth and the rest of your machine is left alone, but training crawls.',
    note: 'About 8 ms keeps a 60 fps interface. Push past 14 ms and the page will visibly stutter.',
  },
  dModel: {
    what: 'How many numbers the model uses to represent each word. More numbers means more room to store what a word means and how it is being used here.',
    up: 'More room to represent meaning. Parameters grow with the square of this, so cost rises sharply.',
    down: 'A narrower model is much cheaper but forces everything it knows through a smaller channel.',
    note: 'Must divide evenly by the number of heads.',
  },
  nLayers: {
    what: 'How many rounds of processing the text goes through. Each round can build on what the previous one worked out.',
    up: 'Depth allows composition: later layers build on what earlier ones worked out. Cost rises in direct proportion.',
    down: 'A shallow model is fast and easy to train but cannot chain reasoning across many steps.',
    note: 'Depth and width are substitutes to a point, but very deep and very narrow trains badly.',
  },
  nHeads: {
    what: 'How many separate things the model can pay attention to at the same time in each round. One head might track which word the verb belongs to while another tracks punctuation.',
    up: 'More heads means more distinct relationships can be tracked at once, but each head gets a narrower slice of the width.',
    down: 'Fewer, wider heads each have more capacity but less ability to specialise.',
    note: 'The model width is split between heads, so head width equals width divided by heads.',
  },
  dFF: {
    what: 'The width of the feed-forward expansion inside each block.',
    up: 'More per-token processing capacity. In large models this is where most of the parameters live.',
    down: 'A narrower expansion saves a lot of parameters and some capability.',
    note: 'Four times the model width is the near-universal convention.',
  },
  blockSize: {
    what: 'How far back the model can see. Anything further back than this simply does not exist as far as it is concerned, so it cannot learn patterns longer than this.',
    up: 'More context to condition on, but attention memory and cost grow with the square of this.',
    down: 'A short window is cheap but the model literally cannot see anything further back.',
    note: 'This is the single largest driver of activation memory during training.',
  },
  merges: {
    what: 'How many common letter pairs get glued together into single pieces before training. More merges means whole words become one piece instead of several letters.',
    up: 'A bigger vocabulary means shorter sequences and faster attention, but a much larger embedding table.',
    down: 'A small vocabulary keeps the embedding table tiny but makes every sequence longer to represent.',
    note: 'In small models the embedding table is often most of the parameters, so this matters more than it looks.',
  },
  batchSeqs: {
    what: 'How many sequences are averaged together before each weight update.',
    up: 'Less noisy gradients and steadier progress, at proportionally more work per step.',
    down: 'Faster individual steps with noisier gradients.',
    note: 'Here the sequences are processed one at a time and their gradients summed, so this scales step time linearly.',
  },
  trainSteps: {
    what: 'How many weight updates the run will perform.',
    up: 'A better trained model, taking proportionally longer.',
    down: 'A quick run that stops well short of what the model could learn.',
    note: 'The estimated wall-clock time is measured from your own machine, not guessed.',
  },

  /* --------------------------------------------- module 08: guardrails -- */

  banList: {
    what: 'Words the sampler is forbidden from producing. They are struck off the list of options before the model picks, so the model is never even consulted about it.',
    up: 'More words blocked means more certainty that those exact strings will not appear, and more collateral damage as their token pieces are shared with innocent words.',
    down: 'Fewer words blocked lets the model say what it actually wanted to. With none listed, nothing is filtered at all.',
    note: 'Watch which token pieces each word resolves to. A word that is not a single token can only be blocked bluntly.',
  },
  steerLayer: {
    what: 'Which point inside the model to read the behaviour from, and where to push it back in. The model processes text in rounds, and each round holds a different kind of information.',
    up: 'Later rounds hold more abstract, more processed meaning, so a direction found there tends to be about the idea rather than the wording.',
    down: 'Earlier rounds, and the embedding itself, hold surface features such as which exact words are present.',
    note: 'A direction found at one layer will not generally work if applied at another.',
  },
  steerStrength: {
    what: 'How hard to push the model along the direction you found, while it is running. No weight is modified; only the running internal state is nudged.',
    up: 'Pushes harder toward group A. Small amounts shift word choice; large amounts overwhelm the model and the text degrades into nonsense.',
    down: 'Negative values push toward group B instead. Zero applies nothing at all, and the output is bit-for-bit the unmodified model.',
    note: 'This is the same mechanism used to switch the refusal behaviour of a real model on and off without retraining it.',
  },
  loraRank: {
    what: 'The size of the small pair of matrices learned alongside a frozen large model. Rank is how much room the adaptation is given.',
    up: 'More capacity to change how the model behaves, and more parameters to train and store. Past a point it stops helping.',
    down: 'Fewer trainable parameters, cheaper and faster, but less able to teach the model anything genuinely new.',
    note: 'Ranks between 8 and 64 cover almost all practical use, and still amount to a tiny fraction of the full model.',
  },

  /* ---------------------------------------- module 09: communication -- */

  commsPrompt: {
    what: 'The text you hand the model. It is cut into tokens and becomes the only thing the model knows before it starts writing. There is no memory of anything you typed earlier and no instructions hidden underneath it.',
    up: 'A longer prompt gives more to condition on, but once it passes the context limit the oldest tokens fall off the front and are simply gone. Nothing tells the model this happened.',
    down: 'A short prompt leaves the model to fall back on whatever its training made most likely, so the reply drifts toward the general shape of its corpus rather than your subject.',
    note: 'Words made of characters the model never saw in training are dropped on the way in. The panel names them when that happens.',
  },
  commsTokens: {
    what: 'How many tokens to generate. Each one is a complete pass through the whole network, and each one is kept in full so you can walk back through it afterwards.',
    up: 'A longer reply, and more to inspect. Every extra token costs another full forward pass and another stored trace, so memory and time both climb in a straight line.',
    down: 'Faster and lighter. Very short replies can look better than the model deserves, because a small model usually holds together for a few tokens before it loses the thread.',
    note: 'The estimated memory figure below the control is the real cost of keeping every intermediate, not a guess.',
  },
  commsTemperature: {
    what: 'How flat or peaked the final choice is made before a token is drawn. The model always produces the same scores; this decides how literally to take them.',
    up: 'Flattens the odds, so unlikely tokens get a real chance. The text gets more varied and less coherent, and past roughly 1.5 it mostly stops making sense.',
    down: 'Sharpens toward the single most likely token. Near zero the model becomes deterministic and tends to repeat itself, because the safest continuation is often a loop.',
    note: 'This happens after the model has finished. The weights are not consulted and nothing inside the network changes.',
  },
  commsTopK: {
    what: 'Keep only this many of the highest-scoring tokens and ignore everything else. A hard cut before the draw.',
    up: 'More candidates survive, so rarer words can appear. Set to zero the cut is off entirely and the whole vocabulary stays in play.',
    down: 'Fewer candidates, so the output is safer and more repetitive. At one, the model always takes its top choice and is fully deterministic.',
    note: 'Blocked tokens are removed before the count is taken, so k always means k tokens the sampler is actually allowed to pick.',
  },
  commsSeed: {
    what: 'The starting number for the random draws. Same seed, same prompt, same weights gives the same reply every time.',
    up: 'Any change at all produces a different reply. There is no ordering to the numbers; one seed is not warmer or colder than another.',
    down: 'The same. What matters is only whether it is the same as last time.',
    note: 'This is what lets you change one weight and be certain that any difference in the output came from the weight and not from luck.',
  },
  commsStage: {
    what: 'Where you are standing in the forward pass. Every stage is one real step the model took, in the order it took them, for the token currently selected.',
    up: 'Later in the computation. The information becomes less about which words are present and more about what the model intends to say next.',
    down: 'Earlier, back toward the raw lookup of what each token means before anything has been done with it.',
    note: 'Nothing is recomputed as you scrub. These are the numbers from the pass that already happened.',
  },
  commsToken: {
    what: 'Which generated token to inspect. Each one was produced by its own complete pass through the network, with one more token of context than the last.',
    up: 'Later in the reply, with more context to work from, including everything the model has already said. Its own earlier mistakes are part of what it is now reading.',
    down: 'Earlier, closer to your prompt, where the model still has mostly your words to go on.',
    note: 'A model that goes off the rails usually does so at one identifiable token. This is how you find it.',
  },
  commsCell: {
    what: 'One single weight, picked out of the matrix. This is the smallest thing in the model it is possible to change: one number among the many thousands.',
    up: 'Raises this one connection. A single weight in a trained model almost never matters on its own, which is itself the lesson: there is no dial in here labelled with a concept.',
    down: 'Lowers it, and below zero it starts arguing the other way. Large values in either direction distort the row they sit in and can wreck the output.',
    note: 'Change it, re-run the same prompt with the same seed, and any difference you see is caused by this number alone.',
  },
  commsScale: {
    what: 'Multiply every number in the selected matrix by this factor at once. A blunt instrument, and a fast way to see what a whole part of the model was contributing.',
    up: 'Above one, that part of the model shouts. Attention becomes sharper and more extreme, or the feed-forward output starts to swamp the residual stream.',
    down: 'Below one, it whispers. At exactly zero the matrix is switched off entirely and you can see what the rest of the model does without it.',
    note: 'Scaling is reversible here, but it is not something a real training run ever does. It is a probe, not a technique.',
  },
  commsAblate: {
    what: 'Switch one attention head off completely, by zeroing the part of the output projection that reads from it. The head still computes; nothing it produces is allowed through.',
    up: 'Not a slider. Switching a head off and comparing the two replies is how researchers work out what an individual head was for.',
    down: 'Switching it back on restores the exact original weights, since the removed values were saved first.',
    note: 'In a small model most heads have not specialised much, so many can be removed with barely any effect. That is itself worth seeing.',
  },
  commsTeachText: {
    what: 'The examples you want the model to learn. Each line is treated as its own lesson and is chopped into next-token prediction problems, exactly as pretraining does.',
    up: 'More examples give a more general lesson and are less likely to be memorised word for word. They also take proportionally longer.',
    down: 'One short line will be learned quickly and almost certainly by rote. The model will reproduce it and will have understood nothing around it.',
    note: 'The vocabulary was fixed before pretraining and cannot grow now, so characters the model has never seen are dropped. The panel lists any it had to drop.',
  },
  commsTeachLr: {
    what: 'How large a step to take for each example while teaching. The same learning rate as in pretraining, but applied to a model that already knows things.',
    up: 'Learns the new material faster, and damages what it already knew faster too. Too high and the model is destroyed in a handful of steps.',
    down: 'Gentler. The lesson takes longer to stick but the rest of the model survives better. This is why real fine-tuning uses a far smaller rate than pretraining did.',
    note: 'Watch both curves. The lesson loss falling while the original loss climbs is catastrophic forgetting, happening in front of you.',
  },
  commsTeachEpochs: {
    what: 'How many times to go over your examples. One pass rarely shifts anything; many passes will drill them in.',
    up: 'The lesson sticks harder, and eventually the model will simply recite your examples back. Everything it knew before degrades along the way.',
    down: 'A lighter touch that nudges the model without rewriting it. Fewer passes leave more of the original behaviour intact.',
    note: 'Every change here can be undone exactly, because the weights were copied before the first step.',
  },

  /* ------------------------------------------ module 10: evaluation -- */

  evalCaseCount: {
    what: 'How many test cases to cut from the corpus. Each one is a prompt paired with the continuation that actually followed it, so the right answer is known without anyone having to write it.',
    up: 'More cases means a narrower confidence interval and a result you can actually act on. The cost is linear: every case is another full run of the model.',
    down: 'Faster, and far less informative. At eight cases a score of 100 percent is still consistent with a true rate near 60 percent, which is why small evals mislead so reliably.',
    note: 'Watch the interval below the score rather than the score. It is the honest summary of how much you have learned.',
  },
  evalPromptLen: {
    what: 'How many tokens of context each test case gives the model before asking it to continue.',
    up: 'More context to work from, so the task gets easier and scores rise. That is not the model improving; it is the test getting gentler.',
    down: 'Less to go on. A very short prompt tests what the model knows in general rather than whether it can use what it was given.',
    note: 'Changing this changes the difficulty of the benchmark. Change it before you commit to a target, never after.',
  },
  evalAnswerLen: {
    what: 'How many tokens of continuation the model has to produce and be judged on.',
    up: 'Harder, and it falls away quickly: getting five tokens right in a row is far less likely than getting one right, because the errors compound.',
    down: 'Easier, and at one token the exact-match and first-token metrics become the same test.',
    note: 'A long expected answer with an exact-match metric is close to unpassable for a small model. That is a fact about your metric, not about the model.',
  },
  evalTopK: {
    what: 'How many of the model\u2019s highest-scoring candidates count as getting it right, for the top-k metric only.',
    up: 'More generous. At k equal to the whole vocabulary every case passes and the metric measures nothing at all.',
    down: 'Stricter. At one this is simply "was its single best guess correct".',
    note: 'Useful during development because it moves early, and misleading in a headline because it rewards nearly knowing.',
  },
  evalGreedy: {
    what: 'Whether to always take the single most likely token when generating an answer, rather than drawing one at random.',
    up: 'On, the eval is repeatable: the same model and the same cases give the same score every time, so a change in the score means a change in the model.',
    down: 'Off, the model samples, and the score moves from run to run even with nothing changed. Any difference you see is then partly the dice.',
    note: 'Sampled evals are sometimes what you want, but then you have to run them several times and report the spread, not a single number.',
  },
  evalTarget: {
    what: 'The score you commit to counting as success, written down before you look at any result.',
    up: 'A higher bar is a stronger claim and more likely to be missed. Set it from what the task actually needs, not from what you think you can hit.',
    down: 'A lower bar is easier to clear and proves less. A target below what the trivial baselines already score proves nothing whatsoever.',
    note: 'The point is committing before the result exists. Moving it afterwards is the most common way an evaluation quietly stops meaning anything, and this page counts it when you do.',
  },
  evalTemperature: {
    what: 'How much randomness to allow when the eval generates an answer, used only when greedy is switched off.',
    up: 'More varied answers and a noisier, generally lower score. Two runs of the identical model will disagree.',
    down: 'Closer to always taking the best guess, until at the floor it is greedy in all but name.',
    note: 'If you are changing this to move the score, you are tuning the test rather than the model.',
  },
} satisfies Record<string, VarInfo>;

export type VarKey = keyof typeof V;
