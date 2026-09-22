import { describe, expect, it, vi } from 'vitest';
import {
  METRICS,
  buildMessages,
  buildSystemMessage,
  effectiveOutputs,
  evaluate,
  optimize,
  parseResponse,
  type Example,
  type Program,
} from './dspy';
import { LMClient, DEFAULT_LM, type ChatMessage } from './lmClient';
import { TASKS, splitExamples } from './tasks';
import { toPython, toJSON, toSystemPrompt } from './export';

const SIG = {
  name: 'Classify',
  instructions: 'Classify the text.',
  inputs: [{ name: 'text', desc: 'Input text.' }],
  outputs: [{ name: 'label', desc: 'One of: yes, no.' }],
};

const prog = (over: Partial<Program> = {}): Program => ({
  signature: SIG,
  demos: [],
  cot: false,
  ...over,
});

/** Stub model: no network, fully deterministic, records what it was sent. */
function stubClient(reply: (messages: ChatMessage[]) => string) {
  const c = new LMClient({ ...DEFAULT_LM, apiKey: 'test' });
  c.useCache = false;
  c.complete = vi.fn(async (messages: ChatMessage[]) => reply(messages));
  return c;
}

describe('adapter', () => {
  it('system message lists fields and the completed marker', () => {
    const s = buildSystemMessage(prog());
    expect(s).toContain('Your input fields are:');
    expect(s).toContain('`text`');
    expect(s).toContain('`label`');
    expect(s).toContain('[[ ## completed ## ]]');
    expect(s).toContain('Classify the text.');
  });

  it('chain of thought inserts reasoning before the real outputs', () => {
    const outs = effectiveOutputs(prog({ cot: true }));
    expect(outs[0].name).toBe('reasoning');
    expect(outs[1].name).toBe('label');
    expect(buildSystemMessage(prog({ cot: true }))).toContain('`reasoning`');
  });

  it('demos become alternating user and assistant turns', () => {
    const p = prog({
      demos: [
        { inputs: { text: 'a' }, outputs: { label: 'yes' }, bootstrapped: false },
        { inputs: { text: 'b' }, outputs: { label: 'no' }, bootstrapped: true },
      ],
    });
    const msgs = buildMessages(p, { text: 'c' });
    expect(msgs[0].role).toBe('system');
    expect(msgs.map((m) => m.role)).toEqual(['system', 'user', 'assistant', 'user', 'assistant', 'user']);
    expect(msgs[2].content).toContain('[[ ## label ## ]]');
    expect(msgs[2].content).toContain('yes');
    expect(msgs[msgs.length - 1].content).toContain('c');
  });

  it('parses well-formed field blocks', () => {
    const out = parseResponse(prog({ cot: true }), '[[ ## reasoning ## ]]\nbecause\n\n[[ ## label ## ]]\nyes\n\n[[ ## completed ## ]]');
    expect(out.reasoning).toBe('because');
    expect(out.label).toBe('yes');
    expect(out.completed).toBeUndefined();
  });

  it('falls back to the whole reply when the model ignores the format', () => {
    const out = parseResponse(prog(), 'I think the answer is yes.');
    expect(out.label).toBe('I think the answer is yes.');
  });

  it('puts an unformatted chain-of-thought reply in the answer field, not reasoning', () => {
    const out = parseResponse(prog({ cot: true }), 'yes');
    expect(out.label).toBe('yes');
  });
});

describe('metrics', () => {
  const m = (id: string) => METRICS.find((x) => x.id === id)!.fn('label');
  const ex = (gold: string): Example => ({ id: 'e', inputs: {}, outputs: { label: gold } });
  const pred = (got: string) => ({ outputs: { label: got }, raw: got, messages: [] });

  it('exact match ignores case, punctuation and articles', () => {
    expect(m('exact')(ex('Yes'), pred('yes.'))).toBe(1);
    expect(m('exact')(ex('the cat'), pred('cat'))).toBe(1);
    expect(m('exact')(ex('yes'), pred('no'))).toBe(0);
  });

  it('contains gives credit for a chatty but correct answer', () => {
    expect(m('contains')(ex('billing'), pred('This belongs in billing I think'))).toBe(1);
    expect(m('contains')(ex('billing'), pred('access'))).toBe(0);
  });

  it('token F1 gives partial credit', () => {
    const s = m('f1')(ex('Ana Ferreira | Lisbon'), pred('Ana Ferreira Lisbon'));
    expect(s).toBeGreaterThan(0.5);
    expect(s).toBeLessThanOrEqual(1);
    expect(m('f1')(ex('a b c'), pred('x y z'))).toBe(0);
  });

  it('numeric match reads the last number on each side', () => {
    expect(m('numeric')(ex('65'), pred('14 times 6 is 84, minus 19 leaves 65'))).toBe(1);
    expect(m('numeric')(ex('65'), pred('the answer is 64'))).toBe(0);
  });
});

describe('evaluate', () => {
  const examples: Example[] = [
    { id: 'a', inputs: { text: 'one' }, outputs: { label: 'yes' } },
    { id: 'b', inputs: { text: 'two' }, outputs: { label: 'no' } },
    { id: 'c', inputs: { text: 'three' }, outputs: { label: 'yes' } },
    { id: 'd', inputs: { text: 'four' }, outputs: { label: 'no' } },
  ];
  const metric = METRICS[0].fn('label');

  it('scores a perfect model at 1 and a broken one at 0', async () => {
    const good = stubClient((msgs) => {
      const last = msgs[msgs.length - 1].content;
      const wants = last.includes('one') || last.includes('three');
      return `[[ ## label ## ]]\n${wants ? 'yes' : 'no'}\n\n[[ ## completed ## ]]`;
    });
    const bad = stubClient(() => '[[ ## label ## ]]\nmaybe\n\n[[ ## completed ## ]]');
    expect((await evaluate(good, prog(), examples, metric)).score).toBe(1);
    expect((await evaluate(bad, prog(), examples, metric)).score).toBe(0);
  });

  it('counts a failed call as zero rather than throwing', async () => {
    const c = new LMClient({ ...DEFAULT_LM });
    c.useCache = false;
    c.complete = vi.fn(async () => {
      throw new Error('network down');
    });
    const res = await evaluate(c, prog(), examples, metric);
    expect(res.score).toBe(0);
    expect(res.results.every((r) => r.prediction.error)).toBe(true);
  });

  it('visits every example exactly once even with concurrency', async () => {
    const seen: string[] = [];
    const c = stubClient((msgs) => {
      seen.push(msgs[msgs.length - 1].content);
      return '[[ ## label ## ]]\nyes\n\n[[ ## completed ## ]]';
    });
    const res = await evaluate(c, prog(), examples, metric, { concurrency: 3 });
    expect(res.results.length).toBe(4);
    expect(seen.length).toBe(4);
  });
});

describe('optimizers', () => {
  const examples: Example[] = Array.from({ length: 8 }, (_, i) => ({
    id: `e${i}`,
    inputs: { text: `item ${i}` },
    outputs: { label: i % 2 === 0 ? 'yes' : 'no' },
  }));
  const metric = METRICS[0].fn('label');

  /**
   * A model that only answers correctly once it has seen at least two
   * demonstrations. This is the behaviour few-shot optimization exists to fix,
   * so the score must measurably rise.
   */
  const demoSensitive = () =>
    stubClient((msgs) => {
      const demoCount = msgs.filter((m) => m.role === 'assistant').length;
      const last = msgs[msgs.length - 1].content;
      const idx = parseInt(last.match(/item (\d+)/)?.[1] ?? '-1', 10);
      if (demoCount >= 2 && idx >= 0) {
        return `[[ ## label ## ]]\n${idx % 2 === 0 ? 'yes' : 'no'}\n\n[[ ## completed ## ]]`;
      }
      return '[[ ## label ## ]]\nunsure\n\n[[ ## completed ## ]]';
    });

  it('LabeledFewShot raises the score by adding demonstrations', async () => {
    const c = demoSensitive();
    const res = await optimize(c, prog(), 'labeled', {
      trainset: examples.slice(0, 4),
      valset: examples.slice(4),
      metric,
      maxDemos: 3,
      candidates: 2,
      rounds: 1,
      minibatch: 4,
    });
    expect(res.baselineScore).toBe(0);
    expect(res.optimizedScore).toBe(1);
    expect(res.program.demos.length).toBe(3);
    expect(res.program.demos.every((d) => !d.bootstrapped)).toBe(true);
  });

  it('BootstrapFewShot keeps only traces that pass the metric', async () => {
    // Correct on even items, wrong on odd ones, regardless of demos.
    const c = stubClient((msgs) => {
      const last = msgs[msgs.length - 1].content;
      const idx = parseInt(last.match(/item (\d+)/)?.[1] ?? '-1', 10);
      const answer = idx % 2 === 0 ? 'yes' : 'wrong';
      return `[[ ## label ## ]]\n${answer}\n\n[[ ## completed ## ]]`;
    });
    const res = await optimize(c, prog(), 'bootstrap', {
      trainset: examples.slice(0, 6),
      valset: examples.slice(6),
      metric,
      maxDemos: 4,
      candidates: 2,
      rounds: 1,
      minibatch: 4,
    });
    // The bootstrapped candidate is kept for inspection even when it does not
    // beat the baseline, which is exactly what happens here: the stub model
    // ignores demos, so the score cannot move.
    const cand = res.candidatePrograms.find((c) => c.label.startsWith('BootstrapFewShot'));
    expect(cand).toBeDefined();
    expect(cand!.program.demos.length).toBeGreaterThan(0);
    for (const d of cand!.program.demos) {
      expect(d.bootstrapped).toBe(true);
      // Only the even-indexed training items can have passed the metric.
      expect(d.outputs.label).toBe('yes');
    }
    // And because it did not improve anything, the baseline is still returned.
    expect(res.program.demos.length).toBe(0);
  });

  it('BootstrapFewShot reports clearly when nothing passes', async () => {
    const c = stubClient(() => '[[ ## label ## ]]\nnever-correct\n\n[[ ## completed ## ]]');
    const res = await optimize(c, prog(), 'bootstrap', {
      trainset: examples.slice(0, 4),
      valset: examples.slice(4),
      metric,
      maxDemos: 4,
      candidates: 2,
      rounds: 1,
      minibatch: 4,
    });
    expect(res.program.demos.length).toBe(0);
    expect(res.events.some((e) => e.kind === 'error' && e.msg.includes('nothing to bootstrap'))).toBe(true);
  });

  it('COPRO adopts a proposed instruction when it scores better', async () => {
    const MAGIC = 'Answer yes for even numbered items and no for odd ones.';
    const c = stubClient((msgs) => {
      const sys = msgs.find((m) => m.role === 'system')?.content ?? '';
      const last = msgs[msgs.length - 1].content;
      if (last.startsWith('You are optimizing')) return `1. ${MAGIC}\n2. Some weaker instruction text here.`;
      const idx = parseInt(last.match(/item (\d+)/)?.[1] ?? '-1', 10);
      if (sys.includes(MAGIC)) {
        return `[[ ## label ## ]]\n${idx % 2 === 0 ? 'yes' : 'no'}\n\n[[ ## completed ## ]]`;
      }
      return '[[ ## label ## ]]\nunsure\n\n[[ ## completed ## ]]';
    });
    const res = await optimize(c, prog(), 'copro', {
      trainset: examples.slice(0, 4),
      valset: examples.slice(4),
      metric,
      maxDemos: 2,
      candidates: 2,
      rounds: 1,
      minibatch: 4,
    });
    expect(res.optimizedScore).toBeGreaterThan(res.baselineScore);
    expect(res.program.signature.instructions).toBe(MAGIC);
  });

  it('never returns a program that scored worse than the baseline', async () => {
    const c = stubClient(() => '[[ ## label ## ]]\nunsure\n\n[[ ## completed ## ]]');
    for (const id of ['labeled', 'bootstrap', 'copro'] as const) {
      const res = await optimize(c, prog(), id, {
        trainset: examples.slice(0, 4),
        valset: examples.slice(4),
        metric,
        maxDemos: 2,
        candidates: 1,
        rounds: 1,
        minibatch: 4,
      });
      expect(res.optimizedScore).toBeGreaterThanOrEqual(res.baselineScore);
    }
  });
});

describe('tasks', () => {
  it('each built-in task is well formed', () => {
    for (const t of TASKS) {
      expect(t.examples.length).toBeGreaterThanOrEqual(12);
      expect(METRICS.some((m) => m.id === t.metricId)).toBe(true);
      for (const e of t.examples) {
        for (const f of t.signature.inputs) expect(e.inputs[f.name]).toBeTruthy();
        for (const f of t.signature.outputs) expect(e.outputs[f.name]).toBeTruthy();
      }
      expect(t.signature.outputs.some((f) => f.name === t.metricField)).toBe(true);
    }
  });

  it('splits are disjoint and cover the set', () => {
    for (const t of TASKS) {
      const { trainset, valset } = splitExamples(t);
      const ids = new Set([...trainset, ...valset].map((e) => e.id));
      expect(ids.size).toBe(t.examples.length);
      expect(trainset.length).toBeGreaterThan(1);
      expect(valset.length).toBeGreaterThan(1);
    }
  });

  it('triage labels all come from the declared set', () => {
    const t = TASKS.find((x) => x.id === 'triage')!;
    const allowed = new Set(['billing', 'access', 'bug', 'hardware', 'feedback']);
    for (const e of t.examples) expect(allowed.has(e.outputs.queue)).toBe(true);
  });
});

describe('python export', () => {
  const t = TASKS[0];
  const { trainset, valset } = splitExamples(t);
  const p: Program = { signature: t.signature, demos: [], cot: true };

  it('emits a plausible DSPy program', () => {
    const code = toPython({
      task: t,
      program: p,
      lm: { ...DEFAULT_LM, model: 'gpt-4o-mini' },
      optimizerId: 'bootstrap',
      metricId: t.metricId,
      metricField: t.metricField,
      trainset,
      valset,
      maxDemos: 4,
      candidates: 3,
    });
    expect(code).toContain('import dspy');
    expect(code).toContain('class SolveWordProblem(dspy.Signature):');
    expect(code).toContain('dspy.ChainOfThought(SolveWordProblem)');
    expect(code).toContain('from dspy.teleprompt import BootstrapFewShot');
    expect(code).toContain('dspy.Evaluate(');
    expect(code).toContain('_last_number');
    // The key must never be baked into exported code.
    expect(code).toContain('os.environ.get');
    expect(code).not.toContain('sk-');
  });

  it('switches module and optimizer with the settings', () => {
    const code = toPython({
      task: t,
      program: { ...p, cot: false },
      lm: DEFAULT_LM,
      optimizerId: 'mipro',
      metricId: 'exact',
      metricField: 'answer',
      trainset,
      valset,
      maxDemos: 2,
      candidates: 4,
    });
    expect(code).toContain('dspy.Predict(');
    expect(code).toContain('MIPROv2');
  });

  it('JSON export captures demos and instructions', () => {
    const withDemos: Program = {
      ...p,
      demos: [{ inputs: { problem: 'x' }, outputs: { answer: '1' }, bootstrapped: true }],
    };
    const parsed = JSON.parse(toJSON(withDemos));
    expect(parsed.chain_of_thought).toBe(true);
    expect(parsed.demos.length).toBe(1);
    expect(parsed.signature.outputs[0].name).toBe('reasoning');
    expect(toSystemPrompt(withDemos)).toContain('[[ ## completed ## ]]');
  });
});
