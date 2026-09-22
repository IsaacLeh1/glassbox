import React from 'react';
import { DEPTHS, useApp, type Depth as DepthId } from '../store/app';
import { usePageVisible } from './loop';

/* ------------------------------------------------------------- colours -- */

const NEG: [number, number, number] = [76, 126, 243];
const POS: [number, number, number] = [245, 158, 11];

/**
 * Diverging colour for a signed value. Magnitude drives opacity so a weight
 * near zero fades into the background instead of shouting.
 */
export function signedColor(v: number, max = 1, boost = 1): string {
  if (!Number.isFinite(v)) return 'transparent';
  const t = Math.min(1, Math.abs(v) / (max || 1));
  const [r, g, b] = v >= 0 ? POS : NEG;
  return `rgba(${r}, ${g}, ${b}, ${Math.min(1, t * boost).toFixed(3)})`;
}

export function signedTextColor(v: number): string {
  return v >= 0 ? 'var(--pos)' : 'var(--neg)';
}

/** Sequential ramp for activations, which are usually one-sided. */
export function heatColor(t: number): string {
  const x = Math.max(0, Math.min(1, t));
  const r = Math.round(12 + x * 243);
  const g = Math.round(20 + x * 138);
  const b = Math.round(30 + x * (x > 0.6 ? -14 : 40));
  return `rgb(${r}, ${g}, ${b})`;
}

/* ------------------------------------------------------------ numbers -- */

export function fmt(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return '--';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e6 || a < 1e-4) return v.toExponential(1);
  return v.toFixed(digits);
}

export function fmtPct(v: number, digits = 1): string {
  return Number.isFinite(v) ? `${(v * 100).toFixed(digits)}%` : '--';
}

export function fmtInt(v: number): string {
  return Number.isFinite(v) ? Math.round(v).toLocaleString() : '--';
}

/* ------------------------------------------------------------- layout -- */

export function Panel({
  title,
  subtitle,
  right,
  children,
  className = '',
  pad = true,
  id,
}: {
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
  right?: React.ReactNode;
  children?: React.ReactNode;
  className?: string;
  pad?: boolean;
  id?: string;
}) {
  return (
    <section id={id} className={`panel overflow-hidden ${className}`} style={{ boxShadow: 'var(--shadow)' }}>
      {(title || right) && (
        <header
          className="flex items-start justify-between gap-3 px-4 py-3"
          style={{ borderBottom: '1px solid var(--border)' }}
        >
          <div className="min-w-0">
            {title && <h3 className="text-[13px] font-semibold tracking-tight">{title}</h3>}
            {subtitle && (
              <p className="mt-0.5 text-[11.5px] leading-snug" style={{ color: 'var(--text-3)' }}>
                {subtitle}
              </p>
            )}
          </div>
          {right && <div className="shrink-0">{right}</div>}
        </header>
      )}
      <div className={pad ? 'p-4' : ''}>{children}</div>
    </section>
  );
}

export function Row({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return <div className={`flex items-center gap-2 ${className}`}>{children}</div>;
}

export function Label({ children, hint }: { children: React.ReactNode; hint?: string }) {
  return (
    <span
      className="text-[11px] font-medium uppercase tracking-[0.07em]"
      style={{ color: 'var(--text-3)' }}
      title={hint}
    >
      {children}
    </span>
  );
}

/* ---------------------------------------------------------- info dots -- */

/**
 * Explanation attached to any control the reader can change. Every adjustable
 * variable in Glassbox carries one, because "what happens if I turn this up"
 * is the question an interactive tool exists to answer.
 */
export interface VarInfo {
  /** What the variable actually is. */
  what: string;
  /** What happens as it increases. */
  up: string;
  /** What happens as it decreases. */
  down: string;
  /** Optional practical guidance, defaults or gotchas. */
  note?: string;
}

/**
 * A hover card positioned in the viewport rather than inside its parent, so it
 * is never clipped by a panel with hidden overflow.
 */
export function InfoDot({ info, title }: { info: VarInfo; title?: React.ReactNode }) {
  const [open, setOpen] = React.useState(false);
  const [pos, setPos] = React.useState({ x: 0, y: 0, above: false });
  const ref = React.useRef<HTMLButtonElement>(null);

  const show = () => {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const above = r.bottom + 210 > window.innerHeight;
    setPos({
      x: Math.min(Math.max(12, r.left + r.width / 2), window.innerWidth - 158),
      y: above ? r.top - 8 : r.bottom + 8,
      above,
    });
    setOpen(true);
  };

  return (
    <>
      <button
        ref={ref}
        type="button"
        aria-label="What does this do?"
        className="focus-ring inline-flex shrink-0 items-center justify-center rounded-full align-middle transition-colors"
        style={{
          width: 13,
          height: 13,
          border: `1px solid ${open ? 'var(--accent)' : 'var(--border-2)'}`,
          color: open ? 'var(--accent)' : 'var(--text-3)',
          fontSize: 9,
          lineHeight: 1,
          fontStyle: 'italic',
          fontFamily: 'Georgia, serif',
          cursor: 'help',
          background: 'transparent',
        }}
        onMouseEnter={show}
        onMouseLeave={() => setOpen(false)}
        onFocus={show}
        onBlur={() => setOpen(false)}
        onClick={(e) => {
          e.preventDefault();
          open ? setOpen(false) : show();
        }}
      >
        i
      </button>
      {open && (
        <div
          role="tooltip"
          className="gb-fade"
          style={{
            position: 'fixed',
            left: pos.x,
            top: pos.y,
            transform: pos.above ? 'translate(-50%, -100%)' : 'translate(-50%, 0)',
            width: 292,
            zIndex: 9999,
            background: 'var(--panel)',
            border: '1px solid var(--border-2)',
            borderRadius: 10,
            boxShadow: '0 8px 28px rgba(0,0,0,.45)',
            padding: '10px 12px',
            pointerEvents: 'none',
          }}
        >
          {title && (
            <div className="mb-1 text-[11px] font-semibold" style={{ color: 'var(--text)' }}>
              {title}
            </div>
          )}
          <p className="text-[11.5px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
            {info.what}
          </p>
          <div className="mt-2 space-y-1">
            <div className="flex gap-1.5 text-[11px] leading-snug">
              <span className="mono shrink-0 font-semibold" style={{ color: 'var(--pos)' }}>
                ↑
              </span>
              <span style={{ color: 'var(--text-3)' }}>{info.up}</span>
            </div>
            <div className="flex gap-1.5 text-[11px] leading-snug">
              <span className="mono shrink-0 font-semibold" style={{ color: 'var(--neg)' }}>
                ↓
              </span>
              <span style={{ color: 'var(--text-3)' }}>{info.down}</span>
            </div>
          </div>
          {info.note && (
            <p
              className="mt-2 border-t pt-1.5 text-[10.5px] leading-snug"
              style={{ color: 'var(--text-3)', borderColor: 'var(--border)' }}
            >
              {info.note}
            </p>
          )}
        </div>
      )}
    </>
  );
}

/* ------------------------------------------------------------ controls -- */

type BtnVariant = 'primary' | 'ghost' | 'soft' | 'danger';

export function Btn({
  children,
  onClick,
  variant = 'soft',
  disabled,
  size = 'md',
  title,
  active,
  className = '',
}: {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: BtnVariant;
  disabled?: boolean;
  size?: 'sm' | 'md';
  title?: string;
  active?: boolean;
  className?: string;
}) {
  const base =
    'focus-ring inline-flex items-center justify-center gap-1.5 rounded-lg font-medium transition-colors select-none whitespace-nowrap';
  const sizes = size === 'sm' ? 'px-2.5 py-1 text-[11.5px]' : 'px-3 py-1.5 text-[12.5px]';
  const style: React.CSSProperties =
    variant === 'primary'
      ? { background: 'var(--accent)', color: '#04121a', border: '1px solid transparent' }
      : variant === 'danger'
        ? { background: 'transparent', color: 'var(--err)', border: '1px solid var(--border-2)' }
        : variant === 'ghost'
          ? { background: 'transparent', color: 'var(--text-2)', border: '1px solid transparent' }
          : {
              background: active ? 'var(--panel-3)' : 'var(--panel-2)',
              color: active ? 'var(--text)' : 'var(--text-2)',
              border: `1px solid ${active ? 'var(--border-2)' : 'var(--border)'}`,
            };
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      disabled={disabled}
      className={`${base} ${sizes} ${disabled ? 'cursor-not-allowed opacity-45' : 'cursor-pointer'} ${className}`}
      style={style}
    >
      {children}
    </button>
  );
}

export function Segmented<T extends string>({
  options,
  value,
  onChange,
  size = 'md',
}: {
  options: { id: T; label: React.ReactNode; hint?: string }[];
  value: T;
  onChange: (v: T) => void;
  size?: 'sm' | 'md';
}) {
  return (
    <div
      className="inline-flex rounded-lg p-0.5"
      style={{ background: 'var(--panel-2)', border: '1px solid var(--border)' }}
      role="tablist"
    >
      {options.map((o) => {
        const on = o.id === value;
        return (
          <button
            key={o.id}
            role="tab"
            aria-selected={on}
            title={o.hint}
            onClick={() => onChange(o.id)}
            className={`focus-ring cursor-pointer rounded-[6px] font-medium transition-colors ${
              size === 'sm' ? 'px-2 py-0.5 text-[11px]' : 'px-2.5 py-1 text-[12px]'
            }`}
            style={{
              background: on ? 'var(--panel-3)' : 'transparent',
              color: on ? 'var(--text)' : 'var(--text-3)',
              boxShadow: on ? '0 1px 2px rgba(0,0,0,.35)' : 'none',
            }}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

export function Slider({
  label,
  value,
  min,
  max,
  step = 0.01,
  onChange,
  format,
  hint,
  disabled,
  tone,
  info,
}: {
  label: React.ReactNode;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  hint?: string;
  disabled?: boolean;
  tone?: 'signed';
  info?: VarInfo;
}) {
  const display = format ? format(value) : fmt(value, 3);
  return (
    <label className="block" title={hint}>
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="inline-flex items-center gap-1.5">
          <Label>{label}</Label>
          {info && <InfoDot info={info} title={label} />}
        </span>
        <span
          className="mono tnum text-[12px] font-semibold"
          style={{ color: tone === 'signed' ? signedTextColor(value) : 'var(--text)' }}
        >
          {display}
        </span>
      </div>
      <input
        type="range"
        className="w-full"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </label>
  );
}

export function Toggle({
  checked,
  onChange,
  label,
  hint,
  info,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: React.ReactNode;
  hint?: string;
  info?: VarInfo;
}) {
  return (
    <div
      className="flex w-full items-center justify-between gap-3 rounded-lg px-2 py-1.5"
      style={{ background: 'transparent' }}
    >
      <span className="inline-flex items-center gap-1.5 text-[12.5px]" style={{ color: 'var(--text-2)' }}>
        <button
          type="button"
          onClick={() => onChange(!checked)}
          title={hint}
          className="focus-ring cursor-pointer text-left"
          style={{ background: 'transparent', color: 'inherit' }}
        >
          {label}
        </button>
        {info && <InfoDot info={info} title={label} />}
      </span>
      <button
        type="button"
        onClick={() => onChange(!checked)}
        title={hint}
        className="focus-ring cursor-pointer"
        style={{ background: 'transparent', lineHeight: 0 }}
      >
      <span
        className="relative inline-block h-[18px] w-[32px] shrink-0 rounded-full transition-colors"
        style={{ background: checked ? 'var(--accent)' : 'var(--panel-3)' }}
      >
        <span
          className="absolute top-[2px] h-[14px] w-[14px] rounded-full transition-all"
          style={{ left: checked ? 16 : 2, background: checked ? '#04121a' : 'var(--text-3)' }}
        />
        </span>
      </button>
    </div>
  );
}

export function Field({
  label,
  hint,
  children,
  info,
}: {
  label: React.ReactNode;
  hint?: string;
  children: React.ReactNode;
  info?: VarInfo;
}) {
  return (
    <label className="block">
      <div className="mb-1 inline-flex items-center gap-1.5">
        <Label hint={hint}>{label}</Label>
        {info && <InfoDot info={info} title={label} />}
      </div>
      {children}
    </label>
  );
}

export function Select<T extends string>({
  value,
  onChange,
  options,
}: {
  value: T;
  onChange: (v: T) => void;
  options: { id: T; label: string }[];
}) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value as T)}>
      {options.map((o) => (
        <option key={o.id} value={o.id}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

/* --------------------------------------------------------------- data -- */

export function Stat({
  label,
  value,
  unit,
  tone,
  hint,
  sub,
}: {
  label: React.ReactNode;
  value: React.ReactNode;
  unit?: string;
  tone?: 'ok' | 'warn' | 'err' | 'accent';
  hint?: string;
  sub?: React.ReactNode;
}) {
  const color =
    tone === 'ok'
      ? 'var(--ok)'
      : tone === 'warn'
        ? 'var(--warn)'
        : tone === 'err'
          ? 'var(--err)'
          : tone === 'accent'
            ? 'var(--accent)'
            : 'var(--text)';
  return (
    <div title={hint} className="min-w-0">
      <div className="truncate text-[10.5px] font-medium uppercase tracking-[0.07em]" style={{ color: 'var(--text-3)' }}>
        {label}
      </div>
      <div className="mono tnum mt-0.5 truncate text-[17px] font-semibold leading-tight" style={{ color }}>
        {value}
        {unit && (
          <span className="ml-1 text-[11px] font-medium" style={{ color: 'var(--text-3)' }}>
            {unit}
          </span>
        )}
      </div>
      {sub && (
        <div className="mt-0.5 truncate text-[10.5px]" style={{ color: 'var(--text-3)' }}>
          {sub}
        </div>
      )}
    </div>
  );
}

export function Badge({
  children,
  tone = 'neutral',
}: {
  children: React.ReactNode;
  tone?: 'neutral' | 'ok' | 'warn' | 'err' | 'accent';
}) {
  const map = {
    neutral: ['var(--panel-3)', 'var(--text-2)'],
    ok: ['color-mix(in srgb, var(--ok) 16%, transparent)', 'var(--ok)'],
    warn: ['color-mix(in srgb, var(--warn) 16%, transparent)', 'var(--warn)'],
    err: ['color-mix(in srgb, var(--err) 16%, transparent)', 'var(--err)'],
    accent: ['color-mix(in srgb, var(--accent) 16%, transparent)', 'var(--accent)'],
  } as const;
  const [bg, fg] = map[tone];
  return (
    <span
      className="mono inline-flex items-center rounded px-1.5 py-[1px] text-[10.5px] font-medium"
      style={{ background: bg, color: fg }}
    >
      {children}
    </span>
  );
}

export function ProgressBar({ value, tone = 'accent' }: { value: number; tone?: 'accent' | 'ok' | 'warn' }) {
  const c = tone === 'ok' ? 'var(--ok)' : tone === 'warn' ? 'var(--warn)' : 'var(--accent)';
  return (
    <div className="h-1 w-full overflow-hidden rounded-full" style={{ background: 'var(--panel-3)' }}>
      <div
        className="h-full rounded-full transition-[width] duration-150"
        style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%`, background: c }}
      />
    </div>
  );
}

export function Spinner({ size = 13 }: { size?: number }) {
  return (
    <span
      className="gb-spin inline-block rounded-full"
      style={{
        width: size,
        height: size,
        border: '2px solid var(--border-2)',
        borderTopColor: 'var(--accent)',
      }}
    />
  );
}

/* --------------------------------------------------------- explanation -- */

export function Callout({
  children,
  tone = 'info',
  title,
}: {
  children: React.ReactNode;
  tone?: 'info' | 'warn' | 'insight' | 'err';
  title?: React.ReactNode;
}) {
  const map = {
    info: ['var(--accent)', 'Note'],
    warn: ['var(--warn)', 'Careful'],
    insight: ['var(--pos)', 'The point'],
    err: ['var(--err)', 'Problem'],
  } as const;
  const [c, deflt] = map[tone];
  return (
    <div
      className="rounded-lg px-3 py-2.5 text-[12.5px] leading-relaxed"
      style={{
        background: `color-mix(in srgb, ${c} 7%, transparent)`,
        color: 'var(--text-2)',
      }}
    >
      <div className="mb-0.5 text-[10.5px] font-semibold uppercase tracking-[0.08em]" style={{ color: c }}>
        {title ?? deflt}
      </div>
      {children}
    </div>
  );
}

export function Code({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <pre
      className={`mono overflow-auto rounded-lg p-3 text-[11.5px] leading-relaxed ${className}`}
      style={{ background: 'var(--bg-2)', border: '1px solid var(--border)', color: 'var(--text-2)' }}
    >
      {children}
    </pre>
  );
}

export function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd
      className="mono rounded px-1 py-[1px] text-[10.5px]"
      style={{ background: 'var(--panel-3)', border: '1px solid var(--border-2)', color: 'var(--text-2)' }}
    >
      {children}
    </kbd>
  );
}

/**
 * The adaptive-depth primitive. Give it up to three versions of the same
 * explanation and it shows whichever the reader has asked for, falling back
 * to plain language when a deeper version was not written.
 */
export function Depth({
  plain,
  math,
  code,
}: {
  plain?: React.ReactNode;
  math?: React.ReactNode;
  code?: React.ReactNode;
}) {
  const depth = useApp((s) => s.depth);
  const chosen = depth === 'code' ? (code ?? math ?? plain) : depth === 'math' ? (math ?? plain) : plain;
  if (!chosen) return null;
  return <div className="gb-fade text-[12.5px] leading-relaxed" style={{ color: 'var(--text-2)' }}>{chosen}</div>;
}

export function DepthSwitch({ size = 'md' }: { size?: 'sm' | 'md' }) {
  const depth = useApp((s) => s.depth);
  const setDepth = useApp((s) => s.setDepth);
  return (
    <Segmented<DepthId>
      size={size}
      value={depth}
      onChange={setDepth}
      options={DEPTHS.map((d) => ({ id: d.id, label: d.label, hint: d.hint }))}
    />
  );
}

/** Inline mathematical notation. Not a full typesetter, just disciplined CSS. */
export function M({ children }: { children: React.ReactNode }) {
  return (
    <span className="mono" style={{ color: 'var(--text)', fontSize: '0.95em' }}>
      {children}
    </span>
  );
}

export function Eq({ children, note }: { children: React.ReactNode; note?: React.ReactNode }) {
  return (
    <div
      className="my-2 rounded-lg px-3 py-2"
      style={{ background: 'var(--bg-2)', border: '1px solid var(--border)' }}
    >
      <div className="mono text-[12.5px]" style={{ color: 'var(--text)' }}>
        {children}
      </div>
      {note && (
        <div className="mt-1 text-[11px]" style={{ color: 'var(--text-3)' }}>
          {note}
        </div>
      )}
    </div>
  );
}

export function Empty({ children, icon }: { children: React.ReactNode; icon?: React.ReactNode }) {
  return (
    <div
      className="flex flex-col items-center justify-center gap-2 rounded-lg px-6 py-10 text-center text-[12.5px]"
      style={{ border: '1px dashed var(--border-2)', color: 'var(--text-3)' }}
    >
      {icon}
      <div className="max-w-[46ch] leading-relaxed">{children}</div>
    </div>
  );
}

/**
 * Shown while a run continues with the tab in the background. Browsers suspend
 * the animation frame in a hidden tab, so the loop falls back to a timer; the
 * run keeps going but more slowly, and saying so beats a frozen counter.
 */
export function BackgroundNotice({ running }: { running: boolean }) {
  const visible = usePageVisible();
  if (!running || visible) return null;
  return (
    <span title="Browsers throttle hidden tabs, so this is running slower than it would in the foreground.">
      <Badge tone="warn">running in background</Badge>
    </span>
  );
}
