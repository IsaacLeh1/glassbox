import { describe, expect, it } from 'vitest';
import { Value, v } from './autograd';
import { MLP } from './mlp';
import { Optimizer, DEFAULT_OPTIM, lrScale } from './optim';
import { makeDataset, batches, gatherRows } from './datasets';
import { fromRows, matmul, matmulTA, matmulTB, softmax1d, transpose, mulberry32 } from './tensor';
import { ACTIVATIONS, ACT_LIST } from './activations';

describe('tensor', () => {
  it('matmul matches a hand-computed product', () => {
    const a = fromRows([
      [1, 2],
      [3, 4],
    ]);
    const b = fromRows([
      [5, 6],
      [7, 8],
    ]);
    expect(Array.from(matmul(a, b).data)).toEqual([19, 22, 43, 50]);
  });

  it('matmulTA equals transpose-then-multiply', () => {
    const a = fromRows([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const b = fromRows([
      [7, 8],
      [9, 10],
    ]);
    expect(Array.from(matmulTA(a, b).data)).toEqual(Array.from(matmul(transpose(a), b).data));
  });

  it('matmulTB equals multiply-by-transpose', () => {
    const a = fromRows([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const b = fromRows([
      [1, 0, 2],
      [0, 1, 3],
    ]);
    expect(Array.from(matmulTB(a, b).data)).toEqual(Array.from(matmul(a, transpose(b)).data));
  });

  it('softmax sums to one and respects temperature', () => {
    const p = softmax1d([2, 1, 0]);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    const cold = softmax1d([2, 1, 0], 0.2);
    const hot = softmax1d([2, 1, 0], 5);
    // Low temperature concentrates mass on the top choice; high temperature flattens.
    expect(cold[0]).toBeGreaterThan(p[0]);
    expect(hot[0]).toBeLessThan(p[0]);
  });

  it('seeded RNG is reproducible', () => {
    const a = Array.from({ length: 5 }, mulberry32(99));
    const b = Array.from({ length: 5 }, mulberry32(99));
    expect(a).toEqual(b);
  });
});

describe('activations', () => {
  it('every derivative matches a numerical estimate', () => {
    const h = 1e-6;
    for (const act of ACT_LIST) {
      for (const x of [-2.3, -0.5, 0.4, 1.7]) {
        const numeric = (act.f(x + h) - act.f(x - h)) / (2 * h);
        expect(Math.abs(numeric - act.df(x))).toBeLessThan(1e-4);
      }
    }
  });
});

describe('autograd', () => {
  it('reproduces the chain rule on a small expression', () => {
    // f = (a * b + c) ^ 2, at a=2 b=-3 c=10  ->  f = 16
    const a = v(2, 'a');
    const b = v(-3, 'b');
    const c = v(10, 'c');
    const f = a.mul(b).add(c).pow(2);
    expect(f.data).toBeCloseTo(16, 12);
    f.backward();
    // df/da = 2*(ab+c)*b = 2*4*-3 = -24
    expect(a.grad).toBeCloseTo(-24, 9);
    expect(b.grad).toBeCloseTo(16, 9);
    expect(c.grad).toBeCloseTo(8, 9);
  });

  it('accumulates gradient when a node is reused', () => {
    const x = v(3, 'x');
    const y = x.add(x); // y = 2x, so dy/dx must be 2, not 1
    y.backward();
    expect(x.grad).toBeCloseTo(2, 12);
  });

  it('agrees with finite differences through tanh', () => {
    const run = (xv: number) => {
      const x = v(xv);
      const w = v(0.7);
      const out = x.mul(w).tanh();
      out.backward();
      return { out: out.data, grad: x.grad };
    };
    const h = 1e-6;
    const base = run(0.4);
    const numeric = (run(0.4 + h).out - run(0.4 - h).out) / (2 * h);
    expect(Math.abs(numeric - base.grad)).toBeLessThan(1e-5);
  });

  it('backwardTrace visits every node exactly once', () => {
    const a = v(1.5);
    const out = a.mul(2).add(a.tanh()).sigmoid();
    const steps = out.backwardTrace();
    const ids = new Set(steps.map((s) => s.node.id));
    expect(ids.size).toBe(steps.length);
    expect(steps.length).toBe(out.topo().length);
  });

  it('exp and log invert each other with correct gradient', () => {
    const x = new Value(1.3);
    const y = x.exp().log();
    y.backward();
    expect(y.data).toBeCloseTo(1.3, 9);
    expect(x.grad).toBeCloseTo(1, 6);
  });
});

describe('MLP backpropagation', () => {
  const cases: { task: 'binary' | 'regression' | 'multiclass'; out: number }[] = [
    { task: 'binary', out: 1 },
    { task: 'regression', out: 1 },
    { task: 'multiclass', out: 3 },
  ];

  for (const c of cases) {
    it(`analytic gradients match numerical ones for ${c.task}`, () => {
      const net = new MLP([2, 5, 4, c.out], ['tanh', 'relu'], c.task, 11);
      const X = fromRows([
        [0.4, -0.2],
        [-0.9, 0.7],
        [0.1, 0.35],
        [0.8, 0.8],
      ]);
      const Y =
        c.task === 'multiclass'
          ? fromRows([
              [1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [0, 1, 0],
            ])
          : fromRows([[1], [0], [1], [0]]);
      for (const r of net.gradCheck(X, Y, 16)) {
        expect(r.relErr).toBeLessThan(1e-5);
      }
    });
  }

  it('gradient check still passes with L2 regularisation on', () => {
    const net = new MLP([2, 6, 1], ['tanh'], 'binary', 5);
    net.l2 = 0.01;
    const X = fromRows([
      [0.3, 0.9],
      [-0.6, 0.2],
    ]);
    const Y = fromRows([[1], [0]]);
    for (const r of net.gradCheck(X, Y, 10)) expect(r.relErr).toBeLessThan(1e-5);
  });

  it('parameter count matches the flattened buffer length', () => {
    const net = new MLP([2, 8, 8, 1], ['tanh', 'tanh'], 'binary');
    expect(net.flatParams().length).toBe(net.paramCount);
    expect(net.paramCount).toBe(2 * 8 + 8 + 8 * 8 + 8 + 8 * 1 + 1);
  });

  it('round-trips parameters through flatten and load', () => {
    const net = new MLP([2, 4, 1], ['relu'], 'binary', 3);
    const before = Array.from(net.flatParams());
    const other = new MLP([2, 4, 1], ['relu'], 'binary', 99);
    other.loadFlat(Float64Array.from(before));
    expect(Array.from(other.flatParams())).toEqual(before);
  });
});

describe('training actually learns', () => {
  it('drives XOR loss down and reaches high accuracy', () => {
    const ds = makeDataset('xor', 200, 0.05, 1);
    const net = new MLP([2, 8, 8, 1], ['tanh', 'tanh'], 'binary', 7);
    const opt = new Optimizer({ ...DEFAULT_OPTIM, name: 'adam', lr: 0.05 });
    const rand = mulberry32(2);
    const startLoss = net.loss(net.forward(ds.X), ds.Y);

    for (let epoch = 0; epoch < 120; epoch++) {
      for (const idx of batches(ds.points.length, 32, rand)) {
        const xb = gatherRows(ds.X, idx);
        const yb = gatherRows(ds.Y, idx);
        const pass = net.forward(xb);
        const g = net.backward(pass, yb);
        opt.tick();
        for (let l = 0; l < net.layerCount; l++) {
          opt.step(`W${l}`, net.W[l].data, g.dW[l].data);
          opt.step(`b${l}`, net.b[l], g.db[l]);
        }
      }
    }
    const endLoss = net.loss(net.forward(ds.X), ds.Y);
    expect(endLoss).toBeLessThan(startLoss);
    expect(net.accuracy(ds.X, ds.Y)).toBeGreaterThan(0.95);
  });

  it('a network with no hidden layer cannot solve XOR', () => {
    const ds = makeDataset('xor', 200, 0.02, 1);
    const net = new MLP([2, 1], [], 'binary', 7);
    const opt = new Optimizer({ ...DEFAULT_OPTIM, lr: 0.1 });
    const rand = mulberry32(3);
    for (let epoch = 0; epoch < 200; epoch++) {
      for (const idx of batches(ds.points.length, 32, rand)) {
        const pass = net.forward(gatherRows(ds.X, idx));
        const g = net.backward(pass, gatherRows(ds.Y, idx));
        opt.tick();
        opt.step('W0', net.W[0].data, g.dW[0].data);
        opt.step('b0', net.b[0], g.db[0]);
      }
    }
    // This is the whole historical point of hidden layers: a single linear
    // boundary tops out around chance on XOR.
    expect(net.accuracy(ds.X, ds.Y)).toBeLessThan(0.75);
  });
});

describe('optimizers', () => {
  const quadratic = (p: Float64Array) => Float64Array.from(p, (x) => 2 * x);

  for (const name of ['sgd', 'momentum', 'rmsprop', 'adam'] as const) {
    it(`${name} descends a simple quadratic`, () => {
      const opt = new Optimizer({ ...DEFAULT_OPTIM, name, lr: 0.1 });
      const p = Float64Array.from([3, -4]);
      for (let i = 0; i < 300; i++) {
        opt.tick();
        opt.step('p', p, quadratic(p));
      }
      expect(Math.abs(p[0])).toBeLessThan(0.1);
      expect(Math.abs(p[1])).toBeLessThan(0.1);
    });
  }

  it('gradient clipping caps the global norm', () => {
    const opt = new Optimizer({ ...DEFAULT_OPTIM, clipNorm: 1 });
    const g = [Float64Array.from([3, 4])];
    const res = opt.clip(g);
    expect(res.norm).toBeCloseTo(5, 9);
    expect(res.scaled).toBe(true);
    expect(opt.globalNorm(g)).toBeCloseTo(1, 9);
  });
});

describe('learning rate schedules', () => {
  const SCHEDULES = ['constant', 'step', 'cosine', 'warmup_cosine'] as const;

  it('no schedule wastes the first step on a learning rate of exactly zero', () => {
    // Regression: warmup was computed as a fraction of total steps, so step 0
    // scaled the rate to 0 and a whole batch of work changed nothing.
    for (const s of SCHEDULES) {
      expect(lrScale(s, 0, 400), s).toBeGreaterThan(0);
    }
  });

  it('every schedule stays within a sane multiplier', () => {
    for (const s of SCHEDULES) {
      for (let step = 0; step < 400; step++) {
        const v = lrScale(s, step, 400);
        expect(v, `${s} at ${step}`).toBeGreaterThanOrEqual(0);
        expect(v, `${s} at ${step}`).toBeLessThanOrEqual(1.0001);
      }
    }
  });

  it('warmup ramps up and then decays away', () => {
    const early = lrScale('warmup_cosine', 0, 1000);
    const peak = lrScale('warmup_cosine', 60, 1000);
    const late = lrScale('warmup_cosine', 990, 1000);
    expect(early).toBeLessThan(peak);
    expect(late).toBeLessThan(peak);
    expect(peak).toBeCloseTo(1, 1);
  });

  it('cosine decays monotonically', () => {
    let prev = Infinity;
    for (let step = 0; step <= 200; step += 10) {
      const v = lrScale('cosine', step, 200);
      expect(v).toBeLessThanOrEqual(prev + 1e-12);
      prev = v;
    }
  });
});

describe('datasets', () => {
  it('every dataset is balanced enough to be learnable', () => {
    for (const name of ['xor', 'circles', 'moons', 'spirals', 'blobs', 'linear', 'stripes'] as const) {
      const ds = makeDataset(name, 300, 0.08, 4);
      const ones = ds.points.filter((p) => p.label === 1).length;
      const frac = ones / ds.points.length;
      expect(frac).toBeGreaterThan(0.25);
      expect(frac).toBeLessThan(0.75);
      expect(ds.X.rows).toBe(300);
    }
  });

  it('batches cover every index exactly once', () => {
    const seen = new Set<number>();
    let count = 0;
    for (const b of batches(50, 16, mulberry32(1))) {
      for (const i of b) {
        seen.add(i);
        count++;
      }
    }
    expect(count).toBe(50);
    expect(seen.size).toBe(50);
  });
});

describe('sanity of activation registry', () => {
  it('sigmoid and tanh stay inside their advertised ranges', () => {
    for (let x = -8; x <= 8; x += 0.5) {
      expect(ACTIVATIONS.sigmoid.f(x)).toBeGreaterThan(0);
      expect(ACTIVATIONS.sigmoid.f(x)).toBeLessThan(1);
      expect(Math.abs(ACTIVATIONS.tanh.f(x))).toBeLessThan(1);
    }
  });
});
