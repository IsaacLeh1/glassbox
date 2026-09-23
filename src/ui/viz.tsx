import React, { useEffect, useMemo, useRef, useState } from 'react';
import { fmt, heatColor, signedColor } from './kit';

/* ------------------------------------------------------------ LineChart -- */

export interface Series {
  id: string;
  label: string;
  color: string;
  points: { x: number; y: number }[];
  dashed?: boolean;
}

export function LineChart({
  series,
  height = 160,
  yLabel,
  xLabel,
  logY = false,
  yMin,
  yMax,
  marker,
  className = '',
}: {
  series: Series[];
  height?: number;
  yLabel?: string;
  xLabel?: string;
  logY?: boolean;
  yMin?: number;
  yMax?: number;
  marker?: number;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(520);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(160, e.contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);

  const pad = { l: 44, r: 10, t: 8, b: 20 };
  const iw = Math.max(10, w - pad.l - pad.r);
  const ih = Math.max(10, height - pad.t - pad.b);

  const all = series.flatMap((s) => s.points);
  const hasData = all.length > 0;
  const xs = all.map((p) => p.x);
  const x0 = hasData ? Math.min(...xs) : 0;
  const x1 = hasData ? Math.max(...xs) : 1;

  const tf = (y: number) => (logY ? Math.log10(Math.max(1e-8, y)) : y);
  const ysRaw = all.map((p) => p.y).filter(Number.isFinite);
  const ys = ysRaw.map(tf);
  const autoMin = ys.length ? Math.min(...ys) : 0;
  const autoMax = ys.length ? Math.max(...ys) : 1;
  const lo = yMin !== undefined ? tf(yMin) : autoMin;
  const hi = yMax !== undefined ? tf(yMax) : autoMax;
  const span = hi - lo || 1;
  const padY = span * 0.08;

  const sx = (x: number) => pad.l + ((x - x0) / (x1 - x0 || 1)) * iw;
  const sy = (y: number) => pad.t + ih - ((tf(y) - (lo - padY)) / (span + padY * 2)) * ih;

  const ticks = useMemo(() => {
    const out: number[] = [];
    for (let i = 0; i <= 4; i++) {
      const t = lo - padY + ((span + padY * 2) * i) / 4;
      out.push(logY ? Math.pow(10, t) : t);
    }
    return out;
  }, [lo, span, padY, logY]);

  const nearest = hover !== null && hasData ? x0 + ((hover - pad.l) / iw) * (x1 - x0) : null;

  return (
    <div ref={ref} className={className}>
      <svg
        width={w}
        height={height}
        onMouseMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          setHover(e.clientX - r.left);
        }}
        onMouseLeave={() => setHover(null)}
        style={{ display: 'block', cursor: 'crosshair' }}
      >
        {ticks.map((t, i) => {
          const y = sy(t);
          return (
            <g key={i}>
              <line x1={pad.l} x2={w - pad.r} y1={y} y2={y} stroke="var(--grid)" strokeWidth={1} />
              <text
                x={pad.l - 6}
                y={y + 3}
                textAnchor="end"
                className="mono"
                fontSize={9.5}
                fill="var(--text-3)"
              >
                {fmt(t, t < 1 ? 3 : 2)}
              </text>
            </g>
          );
        })}

        {marker !== undefined && hasData && (
          <line
            x1={sx(marker)}
            x2={sx(marker)}
            y1={pad.t}
            y2={pad.t + ih}
            stroke="var(--border-2)"
            strokeWidth={1}
            strokeDasharray="3 3"
          />
        )}

        {series.map((s) => {
          if (s.points.length === 0) return null;
          const d = s.points
            .filter((p) => Number.isFinite(p.y))
            .map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`)
            .join(' ');
          return (
            <path
              key={s.id}
              d={d}
              fill="none"
              stroke={s.color}
              strokeWidth={1.6}
              strokeDasharray={s.dashed ? '4 3' : undefined}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          );
        })}

        {nearest !== null &&
          series.map((s) => {
            if (!s.points.length) return null;
            let best = s.points[0];
            for (const p of s.points) if (Math.abs(p.x - nearest) < Math.abs(best.x - nearest)) best = p;
            if (!Number.isFinite(best.y)) return null;
            return <circle key={s.id} cx={sx(best.x)} cy={sy(best.y)} r={2.6} fill={s.color} />;
          })}

        {hover !== null && (
          <line x1={hover} x2={hover} y1={pad.t} y2={pad.t + ih} stroke="var(--border-2)" strokeWidth={1} />
        )}

        {yLabel && (
          <text x={4} y={11} className="mono" fontSize={9.5} fill="var(--text-3)">
            {yLabel}
            {logY ? ' (log)' : ''}
          </text>
        )}
        {xLabel && (
          <text x={w - pad.r} y={height - 4} textAnchor="end" className="mono" fontSize={9.5} fill="var(--text-3)">
            {xLabel}
          </text>
        )}
      </svg>

      <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1">
        {series.map((s) => {
          const last = s.points[s.points.length - 1];
          return (
            <span key={s.id} className="inline-flex items-center gap-1.5 text-[10.5px]" style={{ color: 'var(--text-3)' }}>
              <span
                style={{
                  width: 9,
                  height: 2,
                  background: s.color,
                  display: 'inline-block',
                  borderRadius: 2,
                  opacity: s.dashed ? 0.6 : 1,
                }}
              />
              {s.label}
              {last && <span className="mono tnum" style={{ color: s.color }}>{fmt(last.y, 4)}</span>}
            </span>
          );
        })}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------- HeatGrid -- */

/**
 * Canvas matrix view. Used for weight matrices, attention maps and gradients.
 * Cells are drawn, not laid out as DOM, so a 64x64 map stays smooth.
 */
export function HeatGrid({
  rows,
  cols,
  get,
  max,
  mode = 'signed',
  cell = 14,
  gap = 1,
  onHover,
  onClick,
  highlight,
  className = '',
  maxWidth,
}: {
  rows: number;
  cols: number;
  get: (i: number, j: number) => number;
  max?: number;
  mode?: 'signed' | 'heat';
  cell?: number;
  gap?: number;
  onHover?: (i: number, j: number, v: number) => void;
  onClick?: (i: number, j: number, v: number) => void;
  highlight?: { i?: number; j?: number };
  className?: string;
  maxWidth?: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const size = useMemo(() => {
    let c = cell;
    if (maxWidth) c = Math.max(3, Math.min(cell, Math.floor((maxWidth - gap * (cols - 1)) / cols)));
    return c;
  }, [cell, cols, gap, maxWidth]);

  const W = cols * size + (cols - 1) * gap;
  const H = rows * size + (rows - 1) * gap;

  const scale = useMemo(() => {
    if (max !== undefined) return max;
    let m = 1e-9;
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) m = Math.max(m, Math.abs(get(i, j)));
    return m;
  }, [rows, cols, get, max]);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = W * dpr;
    cv.height = H * dpr;
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const v = get(i, j);
        ctx.fillStyle = mode === 'signed' ? signedColor(v, scale, 1) : heatColor(Math.abs(v) / scale);
        ctx.fillRect(j * (size + gap), i * (size + gap), size, size);
      }
    }
    if (highlight) {
      ctx.strokeStyle = 'var(--accent)';
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1.5;
      if (highlight.i !== undefined && highlight.j !== undefined) {
        ctx.strokeRect(
          highlight.j * (size + gap) - 0.5,
          highlight.i * (size + gap) - 0.5,
          size + 1,
          size + 1,
        );
      }
    }
  }, [rows, cols, get, scale, size, gap, W, H, mode, highlight]);

  const pick = (e: React.MouseEvent) => {
    const r = (e.target as HTMLCanvasElement).getBoundingClientRect();
    const j = Math.floor((e.clientX - r.left) / (size + gap));
    const i = Math.floor((e.clientY - r.top) / (size + gap));
    if (i < 0 || j < 0 || i >= rows || j >= cols) return null;
    return { i, j, v: get(i, j) };
  };

  return (
    <canvas
      ref={ref}
      className={className}
      style={{ width: W, height: H, imageRendering: 'pixelated', cursor: onClick ? 'pointer' : 'default' }}
      onMouseMove={(e) => {
        const p = pick(e);
        if (p) onHover?.(p.i, p.j, p.v);
      }}
      onMouseLeave={() => onHover?.(-1, -1, 0)}
      onClick={(e) => {
        const p = pick(e);
        if (p) onClick?.(p.i, p.j, p.v);
      }}
    />
  );
}

/* ----------------------------------------------------------- FieldCanvas -- */

/**
 * Paints a scalar field over the input plane: what the network predicts, or
 * what one hidden neuron responds to, at every point.
 */
export function FieldCanvas({
  field,
  res,
  size = 260,
  points,
  mode = 'prob',
  onClickPoint,
  probe,
  className = '',
}: {
  field: Float32Array | null;
  res: number;
  size?: number;
  points?: { x: number; y: number; label: number }[];
  mode?: 'prob' | 'signed';
  onClickPoint?: (x: number, y: number) => void;
  probe?: { x: number; y: number } | null;
  className?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = size * dpr;
    cv.height = size * dpr;
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, size * dpr, size * dpr);

    if (field) {
      const img = ctx.createImageData(res, res);
      for (let k = 0; k < res * res; k++) {
        const v = field[k];
        let r: number, g: number, b: number;
        if (mode === 'prob') {
          // 0 -> blue class, 1 -> orange class, 0.5 -> neutral.
          const t = Math.max(0, Math.min(1, v));
          const d = (t - 0.5) * 2;
          const a = Math.abs(d) * 0.62;
          const base = 22;
          r = Math.round(base + (d > 0 ? 245 - base : 76 - base) * a);
          g = Math.round(base + (d > 0 ? 158 - base : 126 - base) * a);
          b = Math.round(base + (d > 0 ? 11 - base : 243 - base) * a);
        } else {
          const a = Math.min(1, Math.abs(v)) * 0.72;
          const base = 22;
          r = Math.round(base + (v > 0 ? 245 - base : 76 - base) * a);
          g = Math.round(base + (v > 0 ? 158 - base : 126 - base) * a);
          b = Math.round(base + (v > 0 ? 11 - base : 243 - base) * a);
        }
        img.data[k * 4] = r;
        img.data[k * 4 + 1] = g;
        img.data[k * 4 + 2] = b;
        img.data[k * 4 + 3] = 255;
      }
      const off = document.createElement('canvas');
      off.width = res;
      off.height = res;
      off.getContext('2d')!.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, 0, 0, size * dpr, size * dpr);
    } else {
      ctx.fillStyle = '#0d0f14';
      ctx.fillRect(0, 0, size * dpr, size * dpr);
    }

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const toPx = (x: number) => ((x + 1.25) / 2.5) * size;
    const toPy = (y: number) => ((1.25 - y) / 2.5) * size;

    if (points) {
      for (const p of points) {
        ctx.beginPath();
        ctx.arc(toPx(p.x), toPy(p.y), 3.1, 0, Math.PI * 2);
        ctx.fillStyle = p.label === 1 ? '#f59e0b' : '#4c7ef3';
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = 'rgba(0,0,0,.55)';
        ctx.stroke();
      }
    }
    if (probe) {
      ctx.beginPath();
      ctx.arc(toPx(probe.x), toPy(probe.y), 6, 0, Math.PI * 2);
      ctx.strokeStyle = '#e9ecf1';
      ctx.lineWidth = 1.8;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(toPx(probe.x), toPy(probe.y), 2, 0, Math.PI * 2);
      ctx.fillStyle = '#e9ecf1';
      ctx.fill();
    }
  }, [field, res, size, points, mode, probe]);

  return (
    <canvas
      ref={ref}
      className={className}
      style={{
        width: size,
        height: size,
        borderRadius: 10,
        border: '1px solid var(--border)',
        cursor: onClickPoint ? 'crosshair' : 'default',
      }}
      onClick={(e) => {
        if (!onClickPoint) return;
        const r = (e.target as HTMLCanvasElement).getBoundingClientRect();
        const x = ((e.clientX - r.left) / size) * 2.5 - 1.25;
        const y = 1.25 - ((e.clientY - r.top) / size) * 2.5;
        onClickPoint(x, y);
      }}
    />
  );
}

/* ------------------------------------------------------------- Sparkline -- */

export function Sparkline({
  values,
  width = 90,
  height = 22,
  color = 'var(--accent)',
}: {
  values: number[];
  width?: number;
  height?: number;
  color?: string;
}) {
  if (values.length < 2) return <svg width={width} height={height} />;
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const d = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * width;
      const y = height - 1 - ((v - lo) / (hi - lo || 1)) * (height - 2);
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      <path d={d} fill="none" stroke={color} strokeWidth={1.3} strokeLinejoin="round" />
    </svg>
  );
}

/* -------------------------------------------------------------- BarMeter -- */

export function BarMeter({
  value,
  max = 1,
  color = 'var(--accent)',
  height = 6,
  label,
}: {
  value: number;
  max?: number;
  color?: string;
  height?: number;
  label?: React.ReactNode;
}) {
  const pct = Math.max(0, Math.min(1, value / (max || 1)));
  return (
    <div className="w-full">
      {label && (
        <div className="mb-0.5 flex justify-between text-[10px]" style={{ color: 'var(--text-3)' }}>
          {label}
        </div>
      )}
      <div className="w-full overflow-hidden rounded-full" style={{ height, background: 'var(--panel-3)' }}>
        <div style={{ width: `${pct * 100}%`, height: '100%', background: color, transition: 'width .12s linear' }} />
      </div>
    </div>
  );
}
