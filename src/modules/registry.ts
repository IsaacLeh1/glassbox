/**
 * The course outline. Kept out of App.tsx so that editing it does not force a
 * full reload, and so any view can read the same list.
 */
export interface ModuleDef {
  path: string;
  num: string;
  title: string;
  tagline: string;
  blurb: string;
  minutes: number;
}

export const MODULES: ModuleDef[] = [
  {
    path: '/neuron',
    num: '01',
    title: 'Weights',
    tagline: 'What a weight is, and what changing one does',
    blurb:
      'Start at a single neuron. Drag its weights and watch the output move. Then build up to a layered network and find the neuron responsible for one specific bend in the boundary.',
    minutes: 15,
  },
  {
    path: '/matrix',
    num: '02',
    title: 'Matrices',
    tagline: 'The one operation everything is built from',
    blurb:
      'Multiply two matrices by hand, then watch your own sentence become a grid of numbers and travel through every matrix in a real model. Finish by comparing those shapes against GPT-2 and Llama.',
    minutes: 20,
  },
  {
    path: '/training',
    num: '03',
    title: 'Learning',
    tagline: 'How a pile of numbers becomes a working model',
    blurb:
      'Run real gradient descent. Step backpropagation one derivative at a time, watch the loss curve, and break training on purpose to see what each hyperparameter is actually protecting you from.',
    minutes: 25,
  },
  {
    path: '/llm',
    num: '04',
    title: 'Language',
    tagline: 'Text in, one token at a time, back out',
    blurb:
      'A real transformer, trained in your browser. Follow a sentence through the tokenizer, the embeddings, every attention head, and the final sampling step that picks the next word.',
    minutes: 30,
  },
  {
    path: '/cluster',
    num: '05',
    title: 'Scale',
    tagline: 'What the same maths costs at frontier size',
    blurb:
      'Plan a training run on real hardware. Watch a simulated cluster work through your job, lose a node, and recover from a checkpoint.',
    minutes: 20,
  },
  {
    path: '/dspy',
    num: '06',
    title: 'Optimizing prompts',
    tagline: 'Before and after, measured on your own model',
    blurb:
      'Connect a real model, measure how it does on a task, then let a DSPy optimizer rewrite the prompt and choose its own examples. Compare the two side by side.',
    minutes: 20,
  },
  {
    path: '/studio',
    num: '07',
    title: 'Build your own',
    tagline: 'Real data, your model, your machine',
    blurb:
      'Pull a dataset straight from Hugging Face, design a transformer to your own specification, set a RAM and CPU budget your machine can live with, then train it and measure yourself against a real GPU cluster.',
    minutes: 35,
  },
  {
    path: '/guardrails',
    num: '08',
    title: 'Guardrails',
    tagline: 'How a model is constrained, and how that actually works',
    blurb:
      'Five different mechanisms sit between a request and an answer, and only one of them is inside the model. Block tokens live, find a behaviour direction inside the network and push along it, and see why there is no list of rules anywhere in there.',
    minutes: 25,
  },
  {
    path: '/comms',
    num: '09',
    title: 'Communication',
    tagline: 'Your own model, end to end, and every way to change it',
    blurb:
      'Open the model you trained yourself. Read its weights, give it a prompt, then scrub back and forth through the exact forward pass that produced every token. Edit a weight, switch an attention head off, apply a guardrail, or teach it something new, and re-run the identical prompt to see what changed.',
    minutes: 35,
  },
  {
    path: '/evals',
    num: '10',
    title: 'Evaluation',
    tagline: 'Deciding what counts as working, before you can see the answer',
    blurb:
      'The step almost everyone skips. Write down what success means, build a held-out test set, find out what a lookup table already scores, commit to a bar, and only then look. Watch contamination inflate a score and watch a confidence interval refuse to let eight cases prove anything.',
    minutes: 30,
  },
  {
    path: '/deploy',
    num: '11',
    title: 'Deployment',
    tagline: 'What it costs to keep a model answering',
    blurb:
      'Quantise your own weights and measure what the precision actually bought you. Watch a real key-value cache turn quadratic generation into linear. Then cost the whole thing against real hardware and find the point where the queue goes vertical.',
    minutes: 30,
  },
];
