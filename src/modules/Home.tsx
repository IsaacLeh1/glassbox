import { Link } from 'react-router-dom';
import { STEPS } from './registry';
import { Badge, Callout, DepthSwitch, Panel } from '../ui/kit';
import { useApp } from '../store/app';

/**
 * Where each step sits in the real sequence of work. Shown on the first card
 * of each phase, so the path reads as four movements rather than twelve
 * unrelated topics.
 */
const PHASES: Record<number, string> = {
  0: 'Foundations',
  4: 'Scale and tooling',
  6: 'Build your own',
  9: 'Ship and run it',
};

function Tile({ n, title, body }: { n: string; title: string; body: string }) {
  return (
    <div className="panel-2 p-3.5">
      <div className="mono mb-1 text-[10px] font-semibold" style={{ color: 'var(--accent)' }}>
        {n}
      </div>
      <div className="mb-1 text-[12.5px] font-semibold">{title}</div>
      <p className="text-[11.5px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
        {body}
      </p>
    </div>
  );
}

export default function Home() {
  const completed = useApp((s) => s.completed);
  const done = STEPS.filter((m) => completed[m.path]).length;
  const totalMinutes = STEPS.reduce((a, m) => a + m.minutes, 0);
  // The first step not yet finished, so the button continues the path
  // instead of sending a returning reader back to the beginning.
  const resume = STEPS.find((m) => !completed[m.path]) ?? null;

  return (
    <div className="mx-auto max-w-[1080px] px-6 py-10">
      <div className="mb-10">
        <div className="mb-3 flex items-center gap-2">
          <Badge tone="accent">A guided walkthrough</Badge>
          <Badge>{STEPS.length} steps</Badge>
          <Badge>{totalMinutes} min end to end</Badge>
          {done > 0 && <Badge tone="ok">{done} done</Badge>}
        </div>
        <h1 className="text-[30px] font-semibold leading-tight tracking-tight">
          Build a model, ship it, and watch it break.
        </h1>
        <p className="mt-3 max-w-[64ch] text-[14px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
          This is not a set of lessons about AI. It is the actual sequence a company works through to
          put a model into production, walked end to end at a size small enough to run in this tab.
          You start at a single weight and finish watching a deployed model drift away from the traffic
          it was built for, and every step hands its result to the next one.
        </p>
        <p className="mt-3 max-w-[64ch] text-[14px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
          Everything here is really computing. Every weight you drag, every gradient you step through and
          every attention head you inspect is a live calculation, not a pre-recorded animation.
        </p>

        <div className="mt-6 flex flex-wrap items-center gap-3">
          <Link
            to={resume ? resume.path : STEPS[0].path}
            className="focus-ring inline-flex items-center gap-2 rounded-lg px-4 py-2 text-[13px] font-semibold"
            style={{ background: 'var(--accent)', color: '#04121a' }}
          >
            {done > 0 && resume ? `Resume at step ${resume.num}` : 'Start the walkthrough'}
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none">
              <path d="M6 3.5 10.5 8 6 12.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </Link>
          <div className="flex items-center gap-2">
            <span className="text-[11.5px]" style={{ color: 'var(--text-3)' }}>
              Reading level
            </span>
            <DepthSwitch size="sm" />
          </div>
        </div>
      </div>

      <div className="mb-3 text-[11px] font-semibold uppercase tracking-[0.09em]" style={{ color: 'var(--text-3)' }}>
        The path, in order
      </div>
      <div className="mb-10 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {STEPS.map((m, i) => {
          const isDone = !!completed[m.path];
          return (
            <Link
              key={m.path}
              to={m.path}
              className="panel group block p-4 transition-colors"
              style={{ boxShadow: 'var(--shadow)' }}
            >
              <div className="mb-2 flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="mono text-[10.5px] font-semibold" style={{ color: 'var(--accent)' }}>
                    {m.num}
                  </span>
                  {PHASES[i] && (
                    <span
                      className="text-[9.5px] font-semibold uppercase tracking-[0.08em]"
                      style={{ color: 'var(--text-3)' }}
                    >
                      {PHASES[i]}
                    </span>
                  )}
                </span>
                <span className="flex items-center gap-1.5">
                  {isDone && (
                    <span style={{ color: 'var(--ok)' }}>
                      <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
                        <path d="M3 8.5l3.2 3.2L13 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </span>
                  )}
                  <Badge>{m.minutes} min</Badge>
                </span>
              </div>
              <div className="text-[14px] font-semibold tracking-tight">{m.title}</div>
              <div className="mb-2 text-[11.5px]" style={{ color: 'var(--text-3)' }}>
                {m.tagline}
              </div>
              <p className="text-[12px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                {m.blurb}
              </p>
            </Link>
          );
        })}
      </div>

      <div className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
        <Panel title="How this is put together" subtitle="The parts that make the numbers trustworthy">
          <div className="grid gap-3 sm:grid-cols-2">
            <Tile
              n="01"
              title="A real autodiff engine"
              body="Scalar reverse-mode automatic differentiation, the same technique PyTorch uses, written small enough that the whole computation graph fits on screen."
            />
            <Tile
              n="02"
              title="Gradient-checked backprop"
              body="Every analytic gradient in the app is verified against a finite-difference estimate in the test suite, including every tensor in the transformer."
            />
            <Tile
              n="03"
              title="A transformer that trains here"
              body="Multi-head causal attention, layer norm, residual stream, GELU feed-forward, and a byte-pair tokenizer trained on the corpus you pick."
            />
            <Tile
              n="04"
              title="Hardware arithmetic, not vibes"
              body="Cluster cost, memory and wall-clock come from published accelerator specifications and the standard six-FLOPs-per-parameter-per-token estimate."
            />
            <Tile
              n="05"
              title="Real datasets, real models"
              body="Pull text straight from the Hugging Face hub, design your own transformer, and train it inside a RAM and CPU budget you set for your own machine."
            />
            <Tile
              n="06"
              title="Your own model, connected"
              body="Point the DSPy step at any OpenAI-compatible endpoint and watch an optimizer measurably improve a prompt on your model, then export the equivalent real Python."
            />
          </div>
        </Panel>

        <div className="flex flex-col gap-4">
          <Panel title="One idea, three depths">
            <p className="mb-3 text-[12.5px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
              The switch in the sidebar rewrites every explanation in the app at once.
            </p>
            <div className="space-y-2 text-[12px]">
              <div className="panel-2 p-2.5">
                <span className="font-semibold">Plain</span>
                <span style={{ color: 'var(--text-3)' }}> — no notation at all, just what is happening and why.</span>
              </div>
              <div className="panel-2 p-2.5">
                <span className="font-semibold">Math</span>
                <span style={{ color: 'var(--text-3)' }}> — the equation behind the step you are looking at.</span>
              </div>
              <div className="panel-2 p-2.5">
                <span className="font-semibold">Code</span>
                <span style={{ color: 'var(--text-3)' }}> — the source that actually produced the number on screen.</span>
              </div>
            </div>
          </Panel>

          <Callout tone="insight" title="What is simulated">
            Exactly one thing. In step 05 the cluster hardware is a simulation, because there is no GPU
            fleet in a browser tab. Its cost and timing arithmetic is real, computed from published
            specifications, and the loss values streaming through its logs come from a model genuinely
            training on this page. Everything else in Glassbox is computing for real.
          </Callout>

          <Callout title="Every control explains itself">
            Anything you can change carries a small ⓘ. Hover it to read what the variable is, and what
            happens if you move it in either direction. You are never left guessing what a slider does.
          </Callout>
        </div>
      </div>
    </div>
  );
}
