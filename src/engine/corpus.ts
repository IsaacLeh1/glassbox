import { mulberry32 } from './tensor';

/**
 * Training corpora for the in-browser language model.
 *
 * These are generated from templates rather than scraped from anywhere. That
 * is deliberate: the text is original, it is small enough to train on in a few
 * seconds, and its structure is regular enough that a learner can watch the
 * model discover grammar instead of guessing whether it memorised something.
 */

export interface CorpusDef {
  id: string;
  label: string;
  blurb: string;
  build: (seed: number) => string;
}

function pick<T>(arr: T[], rand: () => number): T {
  return arr[Math.floor(rand() * arr.length)];
}

const ANIMALS = ['cat', 'dog', 'fox', 'owl', 'bee', 'crow', 'moth', 'hare'];
const COLORS = ['red', 'grey', 'small', 'quiet', 'quick', 'old', 'warm', 'bright'];
const PLACES = ['garden', 'river', 'attic', 'meadow', 'harbour', 'hill', 'window', 'road'];
const VERBS = ['sleeps', 'waits', 'runs', 'sings', 'hides', 'watches', 'rests', 'listens'];
const TIMES = ['at dawn', 'at night', 'in winter', 'after rain', 'all morning', 'until dark'];

function tinyStories(seed: number): string {
  const rand = mulberry32(seed);
  const lines: string[] = [];
  for (let i = 0; i < 260; i++) {
    const a = pick(ANIMALS, rand);
    const c = pick(COLORS, rand);
    const pl = pick(PLACES, rand);
    const vb = pick(VERBS, rand);
    const tm = pick(TIMES, rand);
    const form = Math.floor(rand() * 4);
    if (form === 0) lines.push(`the ${c} ${a} ${vb} in the ${pl} ${tm}.`);
    else if (form === 1) lines.push(`${tm} the ${a} ${vb} near the ${pl}.`);
    else if (form === 2) lines.push(`a ${c} ${a} and a ${c} ${pick(ANIMALS, rand)} ${vb} by the ${pl}.`);
    else lines.push(`the ${a} ${vb} because the ${pl} is ${c} ${tm}.`);
  }
  return lines.join(' ');
}

function counting(seed: number): string {
  const rand = mulberry32(seed);
  const words = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'];
  const lines: string[] = [];
  for (let i = 0; i < 200; i++) {
    const start = Math.floor(rand() * 5);
    const len = 4 + Math.floor(rand() * 5);
    lines.push(words.slice(start, start + len).join(' ') + '.');
  }
  return lines.join(' ');
}

function brackets(seed: number): string {
  const rand = mulberry32(seed);
  const out: string[] = [];
  const gen = (depth: number): string => {
    if (depth <= 0 || rand() < 0.3) return pick(['a', 'b', 'c'], rand);
    const inner = gen(depth - 1) + ' ' + gen(depth - 1);
    return `( ${inner} )`;
  };
  for (let i = 0; i < 300; i++) out.push(gen(2 + Math.floor(rand() * 2)));
  return out.join(' . ');
}

function weather(seed: number): string {
  const rand = mulberry32(seed);
  const sky = ['clear', 'cloudy', 'foggy', 'stormy', 'still'];
  const day = ['monday', 'tuesday', 'friday', 'sunday', 'today', 'tomorrow'];
  const lines: string[] = [];
  for (let i = 0; i < 240; i++) {
    const d = pick(day, rand);
    const s = pick(sky, rand);
    const t = 4 + Math.floor(rand() * 20);
    lines.push(`on ${d} the sky is ${s} and the air is ${t} degrees.`);
  }
  return lines.join(' ');
}

export const CORPORA: CorpusDef[] = [
  {
    id: 'stories',
    label: 'Tiny stories',
    blurb:
      'Short generated sentences about animals and places. Regular enough that a small model learns word order, articles and punctuation within seconds.',
    build: tinyStories,
  },
  {
    id: 'weather',
    label: 'Weather log',
    blurb:
      'A repetitive report format. Good for watching the model latch onto a rigid template and then fill the slots with plausible values.',
    build: weather,
  },
  {
    id: 'counting',
    label: 'Counting',
    blurb:
      'Number words in sequence. The simplest possible test of whether the model has learned to look at the previous token at all.',
    build: counting,
  },
  {
    id: 'brackets',
    label: 'Nested brackets',
    blurb:
      'A formal language with matched parentheses. Attention has to track depth across long spans, which is visible as clear diagonal stripes.',
    build: brackets,
  },
];

export function getCorpus(id: string, seed = 12): string {
  return (CORPORA.find((c) => c.id === id) ?? CORPORA[0]).build(seed);
}
