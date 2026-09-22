import { type Matrix, gaussian, mat, mulberry32 } from './tensor';

export type DatasetName =
  | 'xor'
  | 'circles'
  | 'moons'
  | 'spirals'
  | 'blobs'
  | 'linear'
  | 'stripes';

export interface Dataset {
  name: DatasetName;
  X: Matrix;
  Y: Matrix;
  points: { x: number; y: number; label: number }[];
  task: 'binary';
  domain: [number, number];
}

export const DATASET_INFO: Record<DatasetName, { label: string; blurb: string; difficulty: number }> = {
  linear: {
    label: 'Linear',
    blurb: 'A single straight line separates the classes. One neuron can solve this; no hidden layer needed.',
    difficulty: 1,
  },
  blobs: {
    label: 'Two blobs',
    blurb: 'Two Gaussian clusters. Nearly linear, but the overlap in the middle sets a floor on achievable accuracy.',
    difficulty: 1,
  },
  xor: {
    label: 'XOR',
    blurb: 'The famous one. No straight line works, which is exactly why a hidden layer had to be invented.',
    difficulty: 2,
  },
  circles: {
    label: 'Circles',
    blurb: 'One class ringed by the other. The network has to learn a curved boundary by bending several straight ones.',
    difficulty: 2,
  },
  moons: {
    label: 'Moons',
    blurb: 'Two interlocking crescents. Solvable with a handful of hidden units, and a good test of whether training got stuck.',
    difficulty: 3,
  },
  stripes: {
    label: 'Stripes',
    blurb: 'Alternating bands. Needs enough hidden units to carve out each band separately.',
    difficulty: 3,
  },
  spirals: {
    label: 'Spirals',
    blurb: 'Two interleaved spiral arms. Genuinely hard for a small network, and the clearest demonstration that capacity matters.',
    difficulty: 5,
  },
};

export const DATASET_LIST = Object.keys(DATASET_INFO) as DatasetName[];

export function makeDataset(
  name: DatasetName,
  n = 200,
  noise = 0.1,
  seed = 42,
): Dataset {
  const rand = mulberry32(seed);
  const pts: { x: number; y: number; label: number }[] = [];
  const jitter = () => gaussian(rand) * noise;

  for (let i = 0; i < n; i++) {
    switch (name) {
      case 'linear': {
        const x = rand() * 2 - 1;
        const y = rand() * 2 - 1;
        pts.push({ x: x + jitter() * 0.2, y: y + jitter() * 0.2, label: y > x * 0.6 - 0.1 ? 1 : 0 });
        break;
      }
      case 'blobs': {
        const cls = i % 2;
        const cx = cls === 0 ? -0.45 : 0.45;
        const cy = cls === 0 ? -0.35 : 0.35;
        pts.push({ x: cx + gaussian(rand) * (0.22 + noise), y: cy + gaussian(rand) * (0.22 + noise), label: cls });
        break;
      }
      case 'xor': {
        const x = rand() * 2 - 1;
        const y = rand() * 2 - 1;
        pts.push({ x: x + jitter() * 0.3, y: y + jitter() * 0.3, label: x * y > 0 ? 1 : 0 });
        break;
      }
      case 'circles': {
        const cls = i % 2;
        const r = cls === 0 ? 0.28 + rand() * 0.14 : 0.72 + rand() * 0.16;
        const th = rand() * Math.PI * 2;
        pts.push({ x: r * Math.cos(th) + jitter() * 0.5, y: r * Math.sin(th) + jitter() * 0.5, label: cls });
        break;
      }
      case 'moons': {
        const cls = i % 2;
        const th = rand() * Math.PI;
        const x = cls === 0 ? Math.cos(th) * 0.7 - 0.25 : 1 - Math.cos(th) * 0.7 - 0.75;
        const y = cls === 0 ? Math.sin(th) * 0.55 - 0.2 : 0.2 - Math.sin(th) * 0.55;
        pts.push({ x: x + jitter() * 0.6, y: y + jitter() * 0.6, label: cls });
        break;
      }
      case 'stripes': {
        const x = rand() * 2 - 1;
        const y = rand() * 2 - 1;
        const band = Math.floor((x + 1) * 2.5);
        pts.push({ x: x + jitter() * 0.15, y, label: band % 2 === 0 ? 1 : 0 });
        break;
      }
      case 'spirals': {
        const cls = i % 2;
        const t = (i / n) * 3.6 * Math.PI + (cls === 0 ? 0 : Math.PI);
        const r = 0.06 + (i / n) * 0.82;
        pts.push({
          x: r * Math.cos(t) + jitter() * 0.7,
          y: r * Math.sin(t) + jitter() * 0.7,
          label: cls,
        });
        break;
      }
    }
  }

  const X = mat(pts.length, 2);
  const Y = mat(pts.length, 1);
  pts.forEach((p, i) => {
    X.data[i * 2] = p.x;
    X.data[i * 2 + 1] = p.y;
    Y.data[i] = p.label;
  });

  return { name, X, Y, points: pts, task: 'binary', domain: [-1.25, 1.25] };
}

export interface Split {
  train: Dataset;
  test: Dataset;
}

/** Hold back a slice of the data the model never trains on. */
export function splitDataset(ds: Dataset, testFrac = 0.3, seed = 7): Split {
  const rand = mulberry32(seed);
  const idx = ds.points.map((_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const cut = Math.floor(idx.length * (1 - testFrac));
  const build = (ids: number[]): Dataset => {
    const pts = ids.map((i) => ds.points[i]);
    const X = mat(pts.length, 2);
    const Y = mat(pts.length, 1);
    pts.forEach((p, i) => {
      X.data[i * 2] = p.x;
      X.data[i * 2 + 1] = p.y;
      Y.data[i] = p.label;
    });
    return { ...ds, X, Y, points: pts };
  };
  return { train: build(idx.slice(0, cut)), test: build(idx.slice(cut)) };
}

/** Shuffled index batches, the way a real data loader feeds a training run. */
export function* batches(n: number, batchSize: number, rand: () => number): Generator<number[]> {
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  for (let s = 0; s < n; s += batchSize) yield idx.slice(s, s + batchSize);
}

export function gatherRows(m: Matrix, rows: number[]): Matrix {
  const out = mat(rows.length, m.cols);
  rows.forEach((r, i) => {
    for (let j = 0; j < m.cols; j++) out.data[i * m.cols + j] = m.data[r * m.cols + j];
  });
  return out;
}
