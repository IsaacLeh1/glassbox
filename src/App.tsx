import { Link, NavLink, Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom';
import { useEffect, type ReactElement } from 'react';
import { DepthSwitch, Badge, Keep } from './ui/kit';
import { useApp } from './store/app';
import Home from './modules/Home';
import NeuronLab from './modules/NeuronLab';
import MatrixLab from './modules/MatrixLab';
import TrainingLab from './modules/TrainingLab';
import LLMLab from './modules/LLMLab';
import ClusterLab from './modules/ClusterLab';
import DSPyLab from './modules/DSPyLab';
import StudioLab from './modules/StudioLab';
import GuardrailsLab from './modules/GuardrailsLab';
import CommsLab from './modules/CommsLab';
import EvalLab from './modules/EvalLab';
import DeployLab from './modules/DeployLab';
import MonitorLab from './modules/MonitorLab';
import { STEPS } from './modules/registry';

function Logo() {
  return (
    <svg width="22" height="22" viewBox="0 0 32 32" aria-hidden>
      <rect width="32" height="32" rx="7" fill="var(--panel-2)" stroke="var(--border-2)" />
      <path d="M13.6 14.6 18.6 11.4M13.6 17.4 18.6 20.6" stroke="var(--text-3)" strokeWidth="1.3" />
      <circle cx="11" cy="16" r="3.2" fill="var(--accent)" />
      <circle cx="21" cy="10" r="2.6" fill="var(--pos)" />
      <circle cx="21" cy="22" r="2.6" fill="var(--pos)" />
    </svg>
  );
}

function Sidebar() {
  const completed = useApp((s) => s.completed);
  const visited = useApp((s) => s.visited);
  const theme = useApp((s) => s.theme);
  const setTheme = useApp((s) => s.setTheme);

  return (
    <aside
      className="flex w-[228px] shrink-0 flex-col"
      style={{ background: 'var(--bg-2)', borderRight: '1px solid var(--border)' }}
    >
      <NavLink to="/" className="flex items-center gap-2.5 px-4 py-4" style={{ color: 'var(--text)' }}>
        <Logo />
        <div className="leading-tight">
          <div className="text-[14px] font-semibold tracking-tight">Glassbox</div>
          <div className="text-[10px]" style={{ color: 'var(--text-3)' }}>
            See inside the machine
          </div>
        </div>
      </NavLink>

      <nav className="flex-1 overflow-y-auto px-2 pb-3">
        <div
          className="px-2 pb-1.5 pt-2 text-[10px] font-semibold uppercase tracking-[0.09em]"
          style={{ color: 'var(--text-3)' }}
        >
          Walkthrough
        </div>
        {STEPS.map((m) => (
          <NavLink
            key={m.path}
            to={m.path}
            className="group mb-0.5 block rounded-lg px-2.5 py-2 transition-colors"
            style={({ isActive }) => ({
              background: isActive ? 'var(--panel-2)' : 'transparent',
              border: `1px solid ${isActive ? 'var(--border)' : 'transparent'}`,
            })}
          >
            {({ isActive }) => (
              <div className="flex items-start gap-2.5">
                <span
                  className="mono mt-[1px] text-[10px] font-semibold"
                  style={{ color: isActive ? 'var(--accent)' : 'var(--text-3)' }}
                >
                  {m.num}
                </span>
                <span className="min-w-0 flex-1">
                  <span
                    className="block truncate text-[12.5px] font-medium"
                    style={{ color: isActive ? 'var(--text)' : 'var(--text-2)' }}
                  >
                    {m.title}
                  </span>
                  <span className="block truncate text-[10.5px] leading-snug" style={{ color: 'var(--text-3)' }}>
                    {m.tagline}
                  </span>
                </span>
                {completed[m.path] ? (
                  <span style={{ color: 'var(--ok)' }} title="Completed">
                    <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
                      <path
                        d="M3 8.5l3.2 3.2L13 5"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  </span>
                ) : visited[m.path] ? (
                  <span
                    className="mt-1 block h-1.5 w-1.5 rounded-full"
                    style={{ background: 'var(--border-2)' }}
                    title="Visited"
                  />
                ) : null}
              </div>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="px-3 py-3" style={{ borderTop: '1px solid var(--border)' }}>
        <div className="mb-2 flex items-center justify-between">
          <span className="text-[10px] font-semibold uppercase tracking-[0.09em]" style={{ color: 'var(--text-3)' }}>
            Explain at
          </span>
          <button
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            title="Toggle theme"
            className="focus-ring cursor-pointer rounded p-1"
            style={{ color: 'var(--text-3)' }}
          >
            {theme === 'dark' ? (
              <svg width="13" height="13" viewBox="0 0 16 16" fill="none">
                <path
                  d="M13 9.5A5.5 5.5 0 0 1 6.5 3a5.5 5.5 0 1 0 6.5 6.5z"
                  stroke="currentColor"
                  strokeWidth="1.3"
                  strokeLinejoin="round"
                />
              </svg>
            ) : (
              <svg width="13" height="13" viewBox="0 0 16 16" fill="none">
                <circle cx="8" cy="8" r="3" stroke="currentColor" strokeWidth="1.3" />
                <path
                  d="M8 1v1.6M8 13.4V15M15 8h-1.6M2.6 8H1M12.9 3.1l-1.1 1.1M4.2 11.8l-1.1 1.1M12.9 12.9l-1.1-1.1M4.2 4.2 3.1 3.1"
                  stroke="currentColor"
                  strokeWidth="1.3"
                  strokeLinecap="round"
                />
              </svg>
            )}
          </button>
        </div>
        <DepthSwitch size="sm" />
        <p className="mt-2 text-[10px] leading-snug" style={{ color: 'var(--text-3)' }}>
          Changes every explanation in the app at once.
        </p>
      </div>
    </aside>
  );
}

function Topbar() {
  const loc = useLocation();
  const mod = STEPS.find((m) => loc.pathname.startsWith(m.path));
  const markComplete = useApp((s) => s.markComplete);
  const completed = useApp((s) => s.completed);

  if (!mod) return null;
  const done = !!completed[mod.path];

  return (
    <header
      className="flex shrink-0 items-center justify-between gap-4 px-6 py-3"
      style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg-2)' }}
    >
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="mono text-[10.5px] font-semibold" style={{ color: 'var(--accent)' }}>
            STEP {mod.num} OF {STEPS.length}
          </span>
          <h1 className="truncate text-[15px] font-semibold tracking-tight">{mod.title}</h1>
          <Badge>{mod.minutes} min</Badge>
        </div>
        <p className="truncate text-[11.5px]" style={{ color: 'var(--text-3)' }}>
          {mod.tagline}
        </p>
      </div>
      <button
        onClick={() => markComplete(mod.path)}
        disabled={done}
        className="focus-ring shrink-0 cursor-pointer rounded-lg px-3 py-1.5 text-[12px] font-medium transition-colors disabled:cursor-default"
        style={{
          background: done ? 'transparent' : 'var(--panel-2)',
          border: `1px solid ${done ? 'transparent' : 'var(--border)'}`,
          color: done ? 'var(--ok)' : 'var(--text-2)',
        }}
      >
        {done ? 'Completed' : 'Mark complete'}
      </button>
    </header>
  );
}

/**
 * Back and next, at the foot of every step.
 *
 * This is what makes it a walkthrough rather than a set of pages: there is
 * always one obvious way onward, and taking it marks the step behind you
 * done. The sidebar still lets anyone jump about, because a reader who wants
 * the attention diagrams should not have to sit through gradient descent
 * first.
 */
function StepNav() {
  const loc = useLocation();
  const navigate = useNavigate();
  const markComplete = useApp((s) => s.markComplete);
  const i = STEPS.findIndex((m) => loc.pathname.startsWith(m.path));
  if (i < 0) return null;

  const prev = i > 0 ? STEPS[i - 1] : null;
  const next = i < STEPS.length - 1 ? STEPS[i + 1] : null;

  const onward = () => {
    markComplete(STEPS[i].path);
    if (next) navigate(next.path);
    else navigate('/');
  };

  return (
    <nav
      className="mx-auto flex max-w-[1400px] flex-wrap items-stretch justify-between gap-3 px-6 pb-10 pt-2"
      aria-label="Walkthrough"
    >
      {prev ? (
        <Link
          to={prev.path}
          className="panel group flex min-w-0 max-w-[340px] flex-1 items-center gap-3 p-3 transition-colors"
        >
          <span className="shrink-0" style={{ color: 'var(--text-3)' }}>
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
              <path d="M10 3.5 5.5 8 10 12.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </span>
          <span className="min-w-0">
            <span className="block text-[10px] font-semibold uppercase tracking-[0.08em]" style={{ color: 'var(--text-3)' }}>
              Step {prev.num}, back
            </span>
            <span className="block truncate text-[12.5px] font-medium">{prev.title}</span>
          </span>
        </Link>
      ) : (
        <span />
      )}

      <button
        onClick={onward}
        className="focus-ring group flex min-w-0 max-w-[400px] flex-1 cursor-pointer items-center justify-end gap-3 rounded-xl p-3 text-right transition-colors"
        style={{ background: 'var(--accent)', color: '#04121a' }}
      >
        <span className="min-w-0">
          <span className="block text-[10px] font-semibold uppercase tracking-[0.08em] opacity-70">
            {next ? `Step ${next.num}, next` : 'That is the whole walkthrough'}
          </span>
          <span className="block truncate text-[13px] font-semibold">
            {next ? next.title : 'Back to the beginning'}
          </span>
        </span>
        <span className="shrink-0">
          <svg width="15" height="15" viewBox="0 0 16 16" fill="none">
            <path d="M6 3.5 10.5 8 6 12.5" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </span>
      </button>
    </nav>
  );
}

/** One element per step, so a step can be kept mounted across navigation. */
const VIEWS: Record<string, ReactElement> = {
  '/neuron': <NeuronLab />,
  '/matrix': <MatrixLab />,
  '/training': <TrainingLab />,
  '/llm': <LLMLab />,
  '/cluster': <ClusterLab />,
  '/dspy': <DSPyLab />,
  '/studio': <StudioLab />,
  '/guardrails': <GuardrailsLab />,
  '/comms': <CommsLab />,
  '/evals': <EvalLab />,
  '/deploy': <DeployLab />,
  '/monitor': <MonitorLab />,
};

export default function App() {
  const loc = useLocation();
  const markVisited = useApp((s) => s.markVisited);
  const atStep = STEPS.some((s) => loc.pathname.startsWith(s.path));

  useEffect(() => {
    const m = STEPS.find((x) => loc.pathname.startsWith(x.path));
    if (m) markVisited(m.path);
  }, [loc.pathname, markVisited]);

  return (
    <div className="flex h-full">
      <Sidebar />
      <main className="flex min-w-0 flex-1 flex-col">
        <Topbar />
        <div className="min-h-0 flex-1 overflow-y-auto">
          {/* Steps are kept mounted once opened rather than swapped by the
              router, because unmounting throws away everything the reader
              did there: a dataset pulled from Hugging Face, a half-finished
              lab programme, a trained model. Nothing mounts until it is
              first visited, so unopened steps cost nothing. */}
          {atStep ? (
            STEPS.map((s) => (
              <Keep key={s.path} when={loc.pathname.startsWith(s.path)}>
                {VIEWS[s.path]}
              </Keep>
            ))
          ) : (
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          )}
          <StepNav />
        </div>
      </main>
    </div>
  );
}
