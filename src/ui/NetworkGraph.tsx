import React, { useMemo } from 'react';
import type { MLP } from '../engine/mlp';
import { fmt, signedColor } from './kit';

export interface WeightRef {
  layer: number;
  from: number;
  to: number;
}

export interface BiasRef {
  layer: number;
  unit: number;
}

export interface UnitRef {
  layer: number;
  unit: number;
}

function sameW(a: WeightRef | null, b: WeightRef) {
  return !!a && a.layer === b.layer && a.from === b.from && a.to === b.to;
}

/**
 * The network, drawn to scale from the live model. Edge colour is the sign of
 * the weight and edge thickness is its magnitude, so the shape of what the
 * model has learned is visible at a glance.
 */
export function NetworkGraph({
  net,
  activations,
  gradients,
  selected,
  onSelectWeight,
  selectedUnit,
  onSelectUnit,
  width = 520,
  height = 300,
  showValues = false,
  flowing = false,
}: {
  net: MLP;
  /** Per-layer activation vectors for a single probe input, if one is set. */
  activations?: number[][] | null;
  /** Per-layer weight gradients, to show what backprop wants to change. */
  gradients?: { dW: { rows: number; cols: number; data: Float64Array }[] } | null;
  selected?: WeightRef | null;
  onSelectWeight?: (w: WeightRef) => void;
  selectedUnit?: UnitRef | null;
  onSelectUnit?: (u: UnitRef) => void;
  width?: number;
  height?: number;
  showValues?: boolean;
  flowing?: boolean;
}) {
  const pad = { x: 46, y: 26 };
  const cols = net.sizes.length;
  const colX = (l: number) => pad.x + (l * (width - pad.x * 2)) / Math.max(1, cols - 1);

  const maxUnits = Math.max(...net.sizes);
  const rowY = (l: number, i: number) => {
    const n = net.sizes[l];
    const avail = height - pad.y * 2;
    const step = Math.min(38, avail / Math.max(1, maxUnits - 1 || 1));
    const total = (n - 1) * step;
    return height / 2 - total / 2 + i * step;
  };

  const maxW = useMemo(() => {
    let m = 1e-6;
    for (const W of net.W) for (let i = 0; i < W.data.length; i++) m = Math.max(m, Math.abs(W.data[i]));
    return m;
  }, [net]);

  const maxG = useMemo(() => {
    if (!gradients) return 1;
    let m = 1e-9;
    for (const G of gradients.dW) for (let i = 0; i < G.data.length; i++) m = Math.max(m, Math.abs(G.data[i]));
    return m;
  }, [gradients]);

  const edges: React.ReactNode[] = [];
  for (let l = 0; l < net.layerCount; l++) {
    const W = net.W[l];
    for (let i = 0; i < W.rows; i++) {
      for (let j = 0; j < W.cols; j++) {
        const w = W.data[i * W.cols + j];
        const ref = { layer: l, from: i, to: j };
        const on = sameW(selected ?? null, ref);
        const t = Math.abs(w) / maxW;
        const g = gradients ? gradients.dW[l].data[i * W.cols + j] : 0;
        edges.push(
          <g key={`e${l}-${i}-${j}`}>
            <line
              x1={colX(l)}
              y1={rowY(l, i)}
              x2={colX(l + 1)}
              y2={rowY(l + 1, j)}
              stroke={on ? 'var(--accent)' : signedColor(w, maxW, 1.15)}
              strokeWidth={on ? 2.6 : 0.5 + t * 3}
              strokeLinecap="round"
              className={flowing ? 'gb-flow' : undefined}
              opacity={on ? 1 : 0.28 + t * 0.72}
            />
            {gradients && Math.abs(g) / maxG > 0.12 && (
              <line
                x1={colX(l)}
                y1={rowY(l, i)}
                x2={colX(l + 1)}
                y2={rowY(l + 1, j)}
                stroke="var(--ok)"
                strokeWidth={0.6 + (Math.abs(g) / maxG) * 2}
                strokeDasharray="2 4"
                opacity={0.55}
              />
            )}
            {onSelectWeight && (
              <line
                x1={colX(l)}
                y1={rowY(l, i)}
                x2={colX(l + 1)}
                y2={rowY(l + 1, j)}
                stroke="transparent"
                strokeWidth={9}
                style={{ cursor: 'pointer' }}
                onClick={() => onSelectWeight(ref)}
              >
                <title>{`W[layer ${l}] input ${i} to unit ${j} = ${fmt(w, 4)}`}</title>
              </line>
            )}
            {showValues && on && (
              <text
                x={(colX(l) + colX(l + 1)) / 2}
                y={(rowY(l, i) + rowY(l + 1, j)) / 2 - 4}
                textAnchor="middle"
                className="mono"
                fontSize={9.5}
                fill="var(--accent)"
              >
                {fmt(w, 2)}
              </text>
            )}
          </g>,
        );
      }
    }
  }

  const nodes: React.ReactNode[] = [];
  for (let l = 0; l < cols; l++) {
    for (let i = 0; i < net.sizes[l]; i++) {
      const act = activations?.[l]?.[i];
      const isSel = selectedUnit?.layer === l && selectedUnit?.unit === i;
      const hasAct = act !== undefined && Number.isFinite(act);
      const mag = hasAct ? Math.min(1, Math.abs(act)) : 0;
      const r = 9;
      nodes.push(
        <g
          key={`n${l}-${i}`}
          style={{ cursor: onSelectUnit ? 'pointer' : 'default' }}
          onClick={() => onSelectUnit?.({ layer: l, unit: i })}
        >
          <circle
            cx={colX(l)}
            cy={rowY(l, i)}
            r={r}
            fill={hasAct ? signedColor(act!, 1, 1) : 'var(--panel-2)'}
            stroke={isSel ? 'var(--accent)' : 'var(--border-2)'}
            strokeWidth={isSel ? 2 : 1}
          />
          {hasAct && (
            <text
              x={colX(l)}
              y={rowY(l, i) + 3}
              textAnchor="middle"
              className="mono"
              fontSize={8}
              fill={mag > 0.55 ? '#08090c' : 'var(--text-2)'}
              style={{ pointerEvents: 'none' }}
            >
              {act!.toFixed(1)}
            </text>
          )}
          <title>
            {l === 0
              ? `input ${i}`
              : l === cols - 1
                ? `output ${i}`
                : `hidden layer ${l}, unit ${i}`}
            {hasAct ? ` = ${fmt(act!, 4)}` : ''}
          </title>
        </g>,
      );
    }
  }

  const labels = net.sizes.map((n, l) => (
    <text
      key={`l${l}`}
      x={colX(l)}
      y={height - 6}
      textAnchor="middle"
      className="mono"
      fontSize={9}
      fill="var(--text-3)"
    >
      {l === 0 ? `input (${n})` : l === cols - 1 ? `output (${n})` : `h${l} (${n})`}
    </text>
  ));

  return (
    <svg width={width} height={height} style={{ display: 'block', maxWidth: '100%' }}>
      {edges}
      {nodes}
      {labels}
    </svg>
  );
}
