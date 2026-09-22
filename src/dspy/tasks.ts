import type { Example, Signature } from './dspy';

/**
 * Built-in tasks for the before/after comparison.
 *
 * These are written from scratch rather than lifted from a public benchmark,
 * for two reasons: no licensing questions, and no chance the model you connect
 * has memorised the answers, which would make the optimizer look better than
 * it is. Each task is chosen because a naive prompt fails on it in a specific,
 * visible way that optimization then fixes.
 */

export interface TaskDef {
  id: string;
  label: string;
  blurb: string;
  whyItFails: string;
  signature: Signature;
  metricId: string;
  metricField: string;
  suggestCot: boolean;
  examples: Example[];
}

let n = 0;
const ex = (inputs: Record<string, string>, outputs: Record<string, string>): Example => ({
  id: `ex${++n}`,
  inputs,
  outputs,
});

const arithmetic: TaskDef = {
  id: 'arithmetic',
  label: 'Multi-step word problems',
  blurb: 'Short arithmetic stories that need two or three chained operations.',
  whyItFails:
    'Asked cold, most models answer with a sentence and often skip a step. The gain comes from chain of thought plus demonstrations that show the working, which the optimizer generates by itself.',
  signature: {
    name: 'SolveWordProblem',
    instructions: 'Answer the question.',
    inputs: [{ name: 'problem', desc: 'A short arithmetic word problem.' }],
    outputs: [{ name: 'answer', desc: 'The final numeric answer.' }],
  },
  metricId: 'numeric',
  metricField: 'answer',
  suggestCot: true,
  examples: [
    ex({ problem: 'A shelf holds 14 boxes. Each box has 6 pens. 19 pens are removed. How many pens remain?' }, { answer: '65' }),
    ex({ problem: 'Mira bikes 8 km each morning and 5 km each evening. How far does she bike in 6 days?' }, { answer: '78' }),
    ex({ problem: 'A tank holds 240 litres. It drains 15 litres per hour for 9 hours. How many litres are left?' }, { answer: '105' }),
    ex({ problem: 'Tickets cost 12 dollars. A group buys 7 tickets and pays with 100 dollars. What is the change?' }, { answer: '16' }),
    ex({ problem: 'A printer makes 45 pages per minute for 4 minutes, then jams and loses 30 pages. How many usable pages?' }, { answer: '150' }),
    ex({ problem: 'There are 3 crates of 18 apples. A third of all the apples are bruised. How many are not bruised?' }, { answer: '36' }),
    ex({ problem: 'A bus starts with 22 riders, 9 leave at the first stop and 14 board at the second. How many riders now?' }, { answer: '27' }),
    ex({ problem: 'A recipe needs 250 grams of flour per loaf. How many grams for 7 loaves, minus the 400 grams already measured?' }, { answer: '1350' }),
    ex({ problem: 'A field is 40 metres by 25 metres. A path takes up 150 square metres. What area of field remains?' }, { answer: '850' }),
    ex({ problem: 'Sam saves 35 dollars a week for 8 weeks, then spends 90 dollars. How much is left?' }, { answer: '190' }),
    ex({ problem: 'A library lends 120 books on Monday and twice that on Tuesday. 75 are returned. How many are still out?' }, { answer: '285' }),
    ex({ problem: 'Each shelf fits 24 jars. A store fills 5 shelves and has 17 jars left over. How many jars in total?' }, { answer: '137' }),
    ex({ problem: 'A runner covers 400 metres per lap. After 11 laps she stops 250 metres short of lap 12. How far has she run?' }, { answer: '4550' }),
    ex({ problem: 'A class of 28 students splits into groups of 4. Two groups merge. How many groups are there?' }, { answer: '6' }),
    ex({ problem: 'A phone costs 480 dollars with a 25 percent discount applied. What is the price paid?' }, { answer: '360' }),
    ex({ problem: 'Water flows in at 12 litres per minute and out at 5. After 20 minutes how many litres have accumulated?' }, { answer: '140' }),
    ex({ problem: 'A box weighs 3 kg empty and holds 16 items of 250 grams each. What is the total weight in grams?' }, { answer: '7000' }),
    ex({ problem: 'A train travels 60 km in the first hour and 80 km in each of the next 3 hours. Total distance?' }, { answer: '300' }),
    ex({ problem: 'A garden has 9 rows of 12 plants. 15 plants die. How many are alive?' }, { answer: '93' }),
    ex({ problem: 'A worker earns 18 dollars an hour for 37 hours and pays 120 dollars in fees. What is the take-home pay?' }, { answer: '546' }),
    ex({ problem: 'Three friends split a 174 dollar bill evenly, then each leaves a 6 dollar tip. What does each pay?' }, { answer: '64' }),
    ex({ problem: 'A machine produces 7 parts a minute. How many parts in 2 hours, if it is idle for 15 minutes?' }, { answer: '735' }),
  ],
};

const triage: TaskDef = {
  id: 'triage',
  label: 'Support ticket triage',
  blurb: 'Route a support message to exactly one of five fixed queues.',
  whyItFails:
    'The failure here is format, not understanding. A cold model writes "This looks like a billing issue" instead of the single required token, and every one of those is scored wrong. Demonstrations fix the shape of the answer almost immediately.',
  signature: {
    name: 'TriageTicket',
    instructions: 'Classify the support ticket.',
    inputs: [{ name: 'ticket', desc: 'The text of a customer support message.' }],
    outputs: [{ name: 'queue', desc: 'One of: billing, access, bug, hardware, feedback.' }],
  },
  metricId: 'exact',
  metricField: 'queue',
  suggestCot: false,
  examples: [
    ex({ ticket: 'I was charged twice for the same month and need one refunded.' }, { queue: 'billing' }),
    ex({ ticket: 'The reset link never arrives and I cannot get into my account.' }, { queue: 'access' }),
    ex({ ticket: 'Clicking export crashes the page every single time on Firefox.' }, { queue: 'bug' }),
    ex({ ticket: 'The scanner arrived with a cracked lid and will not power on.' }, { queue: 'hardware' }),
    ex({ ticket: 'Honestly the new dashboard is much clearer than the old one, nice work.' }, { queue: 'feedback' }),
    ex({ ticket: 'My invoice shows a plan I never upgraded to.' }, { queue: 'billing' }),
    ex({ ticket: 'Two-factor codes are rejected even though the clock is correct.' }, { queue: 'access' }),
    ex({ ticket: 'Totals in the report are off by exactly one row every time.' }, { queue: 'bug' }),
    ex({ ticket: 'The docking station stopped charging after about a week.' }, { queue: 'hardware' }),
    ex({ ticket: 'It would be lovely if dark mode remembered my choice.' }, { queue: 'feedback' }),
    ex({ ticket: 'Please cancel my subscription before the next billing date.' }, { queue: 'billing' }),
    ex({ ticket: 'I was removed from the workspace and cannot see any projects.' }, { queue: 'access' }),
    ex({ ticket: 'Uploading a file over 10MB silently fails with no error shown.' }, { queue: 'bug' }),
    ex({ ticket: 'The replacement keyboard has three keys that do not register.' }, { queue: 'hardware' }),
    ex({ ticket: 'Your onboarding email was genuinely useful, thank you.' }, { queue: 'feedback' }),
    ex({ ticket: 'Why does my receipt show tax for a region I do not live in?' }, { queue: 'billing' }),
    ex({ ticket: 'Single sign-on redirects in a loop and never lands anywhere.' }, { queue: 'access' }),
    ex({ ticket: 'Timestamps display in the wrong timezone after the update.' }, { queue: 'bug' }),
    ex({ ticket: 'The monitor flickers whenever the cable is moved slightly.' }, { queue: 'hardware' }),
    ex({ ticket: 'I would pay more for a version with better keyboard shortcuts.' }, { queue: 'feedback' }),
  ],
};

const extraction: TaskDef = {
  id: 'extraction',
  label: 'Structured extraction',
  blurb: 'Pull the person, the place and the date out of a sentence, in a fixed order.',
  whyItFails:
    'Models get the content right and the formatting wrong. Watch the F1 score climb as the optimizer teaches the exact separator and ordering rather than the facts.',
  signature: {
    name: 'ExtractFields',
    instructions: 'Extract the requested details.',
    inputs: [{ name: 'sentence', desc: 'A sentence describing an event.' }],
    outputs: [{ name: 'fields', desc: 'person | place | date, separated by pipes, in that order.' }],
  },
  metricId: 'f1',
  metricField: 'fields',
  suggestCot: false,
  examples: [
    ex({ sentence: 'Dana Whitfield opened the new clinic in Provo on 4 March 2024.' }, { fields: 'Dana Whitfield | Provo | 4 March 2024' }),
    ex({ sentence: 'On 12 June 2023, Ellis Moore spoke at the archive in Bristol.' }, { fields: 'Ellis Moore | Bristol | 12 June 2023' }),
    ex({ sentence: 'The ceremony in Lagos was led by Ify Okonkwo on 2 January 2025.' }, { fields: 'Ify Okonkwo | Lagos | 2 January 2025' }),
    ex({ sentence: 'Rosa Klein reached the summit near Innsbruck on 30 August 2022.' }, { fields: 'Rosa Klein | Innsbruck | 30 August 2022' }),
    ex({ sentence: 'In Kyoto on 9 November 2021, Haru Tanaka presented the findings.' }, { fields: 'Haru Tanaka | Kyoto | 9 November 2021' }),
    ex({ sentence: 'Tomas Reyes filed the report from Bogota on 18 May 2020.' }, { fields: 'Tomas Reyes | Bogota | 18 May 2020' }),
    ex({ sentence: 'The bridge at Aberdeen was reopened by Fiona Grant on 7 July 2019.' }, { fields: 'Fiona Grant | Aberdeen | 7 July 2019' }),
    ex({ sentence: 'On 22 February 2026 Amira Hadid will host the summit in Amman.' }, { fields: 'Amira Hadid | Amman | 22 February 2026' }),
    ex({ sentence: 'Peter Lund signed the agreement in Oslo on 14 October 2018.' }, { fields: 'Peter Lund | Oslo | 14 October 2018' }),
    ex({ sentence: 'The workshop in Nairobi on 3 April 2024 was run by Grace Mwangi.' }, { fields: 'Grace Mwangi | Nairobi | 3 April 2024' }),
    ex({ sentence: 'Leon Fischer began the survey outside Dresden on 27 September 2023.' }, { fields: 'Leon Fischer | Dresden | 27 September 2023' }),
    ex({ sentence: 'On 1 December 2022 the gallery in Lisbon welcomed Ana Ferreira.' }, { fields: 'Ana Ferreira | Lisbon | 1 December 2022' }),
    ex({ sentence: 'Nadia Petrova defended the thesis in Tallinn on 16 June 2021.' }, { fields: 'Nadia Petrova | Tallinn | 16 June 2021' }),
    ex({ sentence: 'The trial in Perth concluded for Owen Bright on 5 May 2025.' }, { fields: 'Owen Bright | Perth | 5 May 2025' }),
    ex({ sentence: 'Marcus Hale toured the plant in Dayton on 11 August 2020.' }, { fields: 'Marcus Hale | Dayton | 11 August 2020' }),
    ex({ sentence: 'On 8 March 2024, Sofia Marino unveiled the mural in Palermo.' }, { fields: 'Sofia Marino | Palermo | 8 March 2024' }),
    ex({ sentence: 'Yusuf Demir joined the crew at Izmir on 19 January 2023.' }, { fields: 'Yusuf Demir | Izmir | 19 January 2023' }),
    ex({ sentence: 'The award went to Clara Bennett in Halifax on 25 November 2019.' }, { fields: 'Clara Bennett | Halifax | 25 November 2019' }),
  ],
};

const rewrite: TaskDef = {
  id: 'rewrite',
  label: 'Constrained rewriting',
  blurb: 'Rewrite a blunt sentence politely, in at most twelve words, with no exclamation marks.',
  whyItFails:
    'Two constraints have to hold at once. A cold model satisfies the tone and ignores the length. The optimizer tends to discover that stating the word limit numerically, and showing short examples, is what makes it stick.',
  signature: {
    name: 'PoliteRewrite',
    instructions: 'Rewrite the sentence.',
    inputs: [{ name: 'blunt', desc: 'A blunt or rude sentence.' }],
    outputs: [{ name: 'polite', desc: 'A polite rewrite of at most twelve words.' }],
  },
  metricId: 'contains',
  metricField: 'polite',
  suggestCot: false,
  examples: [
    ex({ blunt: 'This report is garbage, do it again.' }, { polite: 'Could you revise this report' }),
    ex({ blunt: 'You are late again and it is unacceptable.' }, { polite: 'Please try to arrive on time' }),
    ex({ blunt: 'Stop emailing me about this.' }, { polite: 'Please pause emails on this topic' }),
    ex({ blunt: 'Your code broke everything.' }, { polite: 'This change appears to have caused issues' }),
    ex({ blunt: 'I do not care about your excuses.' }, { polite: 'Let us focus on next steps' }),
    ex({ blunt: 'Fix it now.' }, { polite: 'Could you address this soon' }),
    ex({ blunt: 'That idea is pointless.' }, { polite: 'I am not sure this approach fits' }),
    ex({ blunt: 'You never read the documentation.' }, { polite: 'The documentation may help here' }),
    ex({ blunt: 'This meeting is a waste of time.' }, { polite: 'Perhaps we could shorten this meeting' }),
    ex({ blunt: 'Do not contact me again.' }, { polite: 'Please remove me from this thread' }),
    ex({ blunt: 'Your estimate is nonsense.' }, { polite: 'Could we revisit this estimate together' }),
    ex({ blunt: 'Nobody agrees with you.' }, { polite: 'Others may see this differently' }),
    ex({ blunt: 'That was a stupid mistake.' }, { polite: 'This looks like an oversight' }),
    ex({ blunt: 'Just get it done already.' }, { polite: 'Could you prioritise finishing this' }),
    ex({ blunt: 'You are wrong about all of this.' }, { polite: 'I see this rather differently' }),
    ex({ blunt: 'Why is this taking so long?' }, { polite: 'Could you share a timeline update' }),
  ],
};

export const TASKS: TaskDef[] = [arithmetic, triage, extraction, rewrite];

export function splitExamples(t: TaskDef, trainFrac = 0.5, seed = 3) {
  let a = seed >>> 0;
  const rand = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let x = Math.imul(a ^ (a >>> 15), 1 | a);
    x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
  const arr = [...t.examples];
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  const cut = Math.max(2, Math.floor(arr.length * trainFrac));
  return { trainset: arr.slice(0, cut), valset: arr.slice(cut) };
}
