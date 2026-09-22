/**
 * Scalar reverse-mode automatic differentiation.
 *
 * This is the honest, slow version of what every deep-learning framework does.
 * We keep it scalar so the UI can render the ENTIRE computation graph, node by
 * node, and step the chain rule one derivative at a time. The matrix engine in
 * tensor.ts does the same math fast; this one does it visibly.
 */

let _uid = 0;

export type Op =
  | ''
  | '+'
  | '*'
  | '^'
  | 'tanh'
  | 'relu'
  | 'sigmoid'
  | 'exp'
  | 'log'
  | 'neg';

export class Value {
  readonly id: number;
  data: number;
  grad: number;
  label: string;
  readonly op: Op;
  readonly prev: Value[];
  /** Local rule: pushes this node's grad into its parents. */
  backwardStep: () => void;

  constructor(data: number, prev: Value[] = [], op: Op = '', label = '') {
    this.id = _uid++;
    this.data = data;
    this.grad = 0;
    this.prev = prev;
    this.op = op;
    this.label = label;
    this.backwardStep = () => {};
  }

  static of(x: number | Value, label = ''): Value {
    return x instanceof Value ? x : new Value(x, [], '', label);
  }

  add(other: number | Value): Value {
    const o = Value.of(other);
    const out = new Value(this.data + o.data, [this, o], '+');
    // d(a+b)/da = 1, so the gradient flows through unchanged to both parents.
    out.backwardStep = () => {
      this.grad += out.grad;
      o.grad += out.grad;
    };
    return out;
  }

  mul(other: number | Value): Value {
    const o = Value.of(other);
    const out = new Value(this.data * o.data, [this, o], '*');
    // d(a*b)/da = b -- each parent is scaled by the OTHER parent's value.
    out.backwardStep = () => {
      this.grad += o.data * out.grad;
      o.grad += this.data * out.grad;
    };
    return out;
  }

  pow(k: number): Value {
    const out = new Value(Math.pow(this.data, k), [this], '^');
    out.backwardStep = () => {
      this.grad += k * Math.pow(this.data, k - 1) * out.grad;
    };
    out.label = `^${k}`;
    return out;
  }

  neg(): Value {
    return this.mul(-1);
  }
  sub(other: number | Value): Value {
    return this.add(Value.of(other).neg());
  }
  div(other: number | Value): Value {
    return this.mul(Value.of(other).pow(-1));
  }

  tanh(): Value {
    const t = Math.tanh(this.data);
    const out = new Value(t, [this], 'tanh');
    // d/dx tanh(x) = 1 - tanh(x)^2
    out.backwardStep = () => {
      this.grad += (1 - t * t) * out.grad;
    };
    return out;
  }

  relu(): Value {
    const r = this.data > 0 ? this.data : 0;
    const out = new Value(r, [this], 'relu');
    // Gradient is 1 where the input was positive, 0 where it was clipped.
    out.backwardStep = () => {
      this.grad += (this.data > 0 ? 1 : 0) * out.grad;
    };
    return out;
  }

  sigmoid(): Value {
    const s = 1 / (1 + Math.exp(-this.data));
    const out = new Value(s, [this], 'sigmoid');
    // d/dx sigmoid(x) = s(1 - s)
    out.backwardStep = () => {
      this.grad += s * (1 - s) * out.grad;
    };
    return out;
  }

  exp(): Value {
    const e = Math.exp(this.data);
    const out = new Value(e, [this], 'exp');
    out.backwardStep = () => {
      this.grad += e * out.grad;
    };
    return out;
  }

  log(): Value {
    const out = new Value(Math.log(this.data), [this], 'log');
    out.backwardStep = () => {
      this.grad += (1 / this.data) * out.grad;
    };
    return out;
  }

  /** Reverse topological order: a node always appears after everything it feeds. */
  topo(): Value[] {
    const order: Value[] = [];
    const seen = new Set<number>();
    const visit = (v: Value) => {
      if (seen.has(v.id)) return;
      seen.add(v.id);
      for (const p of v.prev) visit(p);
      order.push(v);
    };
    visit(this);
    return order;
  }

  /** Full backward pass. Seeds dL/dL = 1, then applies every local rule. */
  backward(): void {
    const order = this.topo();
    for (const v of order) v.grad = 0;
    this.grad = 1;
    for (let i = order.length - 1; i >= 0; i--) order[i].backwardStep();
  }

  /**
   * Backward as a list of discrete steps, so the UI can play it one frame at a
   * time and show exactly which edge is being credited with what.
   */
  backwardTrace(): { node: Value; beforeGrads: Map<number, number> }[] {
    const order = this.topo();
    for (const v of order) v.grad = 0;
    this.grad = 1;
    const steps: { node: Value; beforeGrads: Map<number, number> }[] = [];
    for (let i = order.length - 1; i >= 0; i--) {
      const node = order[i];
      const beforeGrads = new Map(order.map((v) => [v.id, v.grad]));
      node.backwardStep();
      steps.push({ node, beforeGrads });
    }
    return steps;
  }
}

export const v = (x: number, label = '') => new Value(x, [], '', label);
