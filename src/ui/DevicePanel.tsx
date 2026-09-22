import { useCallback, useEffect, useState } from 'react';
import {
  BENCH_SIZES,
  GpuCompute,
  crossoverSize,
  detectGraphics,
  prettyRenderer,
  verifyBackends,
  benchmarkMatmul,
  type BackendId,
  type BenchPoint,
  type MachineGraphics,
} from '../engine/compute';
import { Badge, Btn, Callout, Depth, Label, Panel, Segmented, Slider, Spinner, Stat, fmt } from './kit';
import { BarMeter, LineChart, type Series } from './viz';
import { V } from '../content/varInfo';

/**
 * Device detection, selection and an honest benchmark.
 *
 * The benchmark is the point of this panel. Rather than asserting that a GPU
 * is faster, it measures both backends on this machine at several matrix sizes
 * and shows where the crossover actually falls.
 */
export default function DevicePanel({
  backend,
  onBackend,
  modelMatrixSize,
}: {
  backend: BackendId;
  onBackend: (b: BackendId) => void;
  /** The largest matmul dimension the current model actually performs. */
  modelMatrixSize: number;
}) {
  const [gfx, setGfx] = useState<MachineGraphics | null>(null);
  const [points, setPoints] = useState<BenchPoint[]>([]);
  const [running, setRunning] = useState(false);
  const [verify, setVerify] = useState<{ maxAbsErr: number; ok: boolean } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [demoN, setDemoN] = useState(512);
  const [demo, setDemo] = useState<BenchPoint | null>(null);

  useEffect(() => {
    let alive = true;
    detectGraphics().then((g) => {
      if (alive) setGfx(g);
    });
    return () => {
      alive = false;
    };
  }, []);

  const openGpu = useCallback(async () => {
    const pref: GPUPowerPreference = backend === 'gpu-low' ? 'low-power' : 'high-performance';
    return GpuCompute.create(pref);
  }, [backend]);

  const runBench = async () => {
    setRunning(true);
    setErr(null);
    setPoints([]);
    setVerify(null);
    let gpu: GpuCompute | null = null;
    try {
      if (backend !== 'cpu') {
        gpu = await openGpu();
        if (!gpu) throw new Error('Could not acquire that GPU adapter.');
        setVerify(await verifyBackends(gpu));
      }
      const collected: BenchPoint[] = [];
      await benchmarkMatmul(BENCH_SIZES, gpu, (p) => {
        collected.push(p);
        setPoints([...collected]);
      });
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      gpu?.destroy();
      setRunning(false);
    }
  };

  const runDemo = async () => {
    setRunning(true);
    setErr(null);
    setDemo(null);
    let gpu: GpuCompute | null = null;
    try {
      if (backend !== 'cpu') {
        gpu = await openGpu();
        if (!gpu) throw new Error('Could not acquire that GPU adapter.');
      }
      const [p] = await benchmarkMatmul([demoN], gpu);
      setDemo(p);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      gpu?.destroy();
      setRunning(false);
    }
  };

  const lo = gfx?.adapters.find((a) => a.id === 'gpu-low');
  const hi = gfx?.adapters.find((a) => a.id === 'gpu-high');
  const cross = crossoverSize(points);
  const card = prettyRenderer(gfx?.webglRenderer ?? null);

  const series: Series[] = [
    { id: 'cpu', label: 'CPU', color: 'var(--accent)', points: points.map((p) => ({ x: p.n, y: p.cpuGflops })) },
    {
      id: 'gpu',
      label: 'GPU',
      color: 'var(--ok)',
      points: points.filter((p) => p.gpuGflops !== null).map((p) => ({ x: p.n, y: p.gpuGflops as number })),
    },
  ];

  const describe = (a: typeof lo) => {
    if (!a) return 'detecting…';
    if (!a.available) return a.note || 'unavailable';
    const bits = [a.vendor, a.architecture, a.device].filter(Boolean).join(' · ');
    return bits || 'reported anonymously by the browser';
  };

  return (
    <Panel
      title="Compute device"
      subtitle="What actually runs the arithmetic, detected on this machine"
      right={gfx ? <Badge tone={gfx.webgpu ? 'ok' : 'warn'}>{gfx.webgpu ? 'WebGPU available' : 'no WebGPU'}</Badge> : <Spinner />}
    >
      <div className="mb-3">
        <Label>run on</Label>
        <div className="mt-1.5">
          <Segmented<BackendId>
            value={backend}
            onChange={onBackend}
            options={[
              { id: 'cpu', label: 'CPU', hint: 'Single-threaded JavaScript, 64-bit floats.' },
              { id: 'gpu-low', label: 'Integrated GPU', hint: lo?.available ? describe(lo) : 'unavailable' },
              { id: 'gpu-high', label: 'Dedicated GPU', hint: hi?.available ? describe(hi) : 'unavailable' },
            ]}
          />
        </div>
      </div>

      <div className="space-y-1.5 text-[11.5px]">
        <div className="flex justify-between gap-3">
          <span style={{ color: 'var(--text-3)' }}>graphics card</span>
          <span className="mono truncate text-right" style={{ color: 'var(--text-2)', maxWidth: 260 }}>
            {card ?? 'not reported'}
          </span>
        </div>
        <div className="flex justify-between gap-3">
          <span style={{ color: 'var(--text-3)' }}>low-power adapter</span>
          <span className="mono truncate text-right" style={{ color: lo?.available ? 'var(--text-2)' : 'var(--text-3)', maxWidth: 260 }}>
            {describe(lo)}
          </span>
        </div>
        <div className="flex justify-between gap-3">
          <span style={{ color: 'var(--text-3)' }}>high-performance adapter</span>
          <span className="mono truncate text-right" style={{ color: hi?.available ? 'var(--text-2)' : 'var(--text-3)', maxWidth: 260 }}>
            {describe(hi)}
          </span>
        </div>
        <div className="flex justify-between gap-3">
          <span style={{ color: 'var(--text-3)' }}>CPU threads reported</span>
          <span className="mono" style={{ color: 'var(--text-2)' }}>
            {navigator.hardwareConcurrency ?? 'unknown'}
          </span>
        </div>
      </div>

      {gfx?.singleGpu && (
        <div className="mt-3">
          <Callout tone="info" title="Only one GPU is visible to this browser">
            Both the low-power and high-performance requests returned the same device, so either this machine
            has a single GPU or the browser has already been bound to one of them. Switching between the two
            options will not change anything until the browser exposes both.
          </Callout>
        </div>
      )}

      {/* ------------------------------------------------------- benchmark */}

      <div className="mt-5">
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <Label>measured on this machine</Label>
          <div className="flex gap-1.5">
            <Btn size="sm" variant="primary" onClick={runBench} disabled={running}>
              {running ? 'Measuring…' : 'Benchmark CPU vs GPU'}
            </Btn>
          </div>
        </div>

        {points.length === 0 && !running && (
          <p className="text-[12px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
            Runs the same square matrix multiply on both backends at sizes from 32 up to 1024 and times them,
            including the cost of getting data to the GPU and back. Takes a few seconds.
          </p>
        )}

        {points.length > 0 && (
          <>
            <LineChart series={series} height={150} yLabel="GFLOP/s" xLabel="matrix size (n × n)" logY />

            <div className="mt-3 overflow-x-auto">
              <table className="w-full text-[11px]" style={{ borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ color: 'var(--text-3)' }}>
                    {['size', 'CPU', 'GPU', 'winner', 'speed-up'].map((h) => (
                      <th key={h} className="px-2 py-1 text-left font-medium" style={{ borderBottom: '1px solid var(--border)' }}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {points.map((p) => (
                    <tr key={p.n} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td className="mono px-2 py-1.5">{p.n}</td>
                      <td className="mono px-2 py-1.5" style={{ color: 'var(--accent)' }}>
                        {p.cpuMs.toFixed(2)} ms
                        <span style={{ color: 'var(--text-3)' }}> · {p.cpuGflops.toFixed(1)} GF/s</span>
                      </td>
                      <td className="mono px-2 py-1.5" style={{ color: 'var(--ok)' }}>
                        {p.gpuMs === null ? '—' : `${p.gpuMs.toFixed(2)} ms`}
                        {p.gpuGflops !== null && <span style={{ color: 'var(--text-3)' }}> · {p.gpuGflops.toFixed(1)} GF/s</span>}
                      </td>
                      <td className="mono px-2 py-1.5" style={{ color: p.gpuWins ? 'var(--ok)' : 'var(--accent)' }}>
                        {p.gpuMs === null ? 'CPU only' : p.gpuWins ? 'GPU' : 'CPU'}
                      </td>
                      <td className="mono px-2 py-1.5" style={{ color: 'var(--text-3)' }}>
                        {p.gpuMs === null ? '—' : `${(p.cpuMs / p.gpuMs).toFixed(2)}×`}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {verify && (
              <p className="mt-2 text-[10.5px]" style={{ color: verify.ok ? 'var(--ok)' : 'var(--err)' }}>
                {verify.ok
                  ? `Backends agree: largest disagreement ${verify.maxAbsErr.toExponential(1)} on a 64×64 product, which is ordinary 32-bit rounding.`
                  : `Backends disagree by ${verify.maxAbsErr.toExponential(1)}, which is too much to be rounding. Treat the GPU numbers with suspicion.`}
              </p>
            )}

            <div className="mt-3">
              {cross === null ? (
                <Callout tone="warn" title="The GPU never won at these sizes">
                  On this machine the CPU was faster at every size tested. Each GPU dispatch costs a fixed
                  amount of driver and transfer time, and below a certain matrix size that overhead is larger
                  than the entire calculation. Try the larger single-matmul test below.
                </Callout>
              ) : (
                <Callout tone="insight" title={`The GPU takes over at ${cross} × ${cross}`}>
                  Below that size the fixed cost of a dispatch — sending the data across, launching the
                  kernel, reading the result back — outweighs the arithmetic saved. Above it, the thousands of
                  parallel units win and keep winning by a widening margin. Your model&apos;s largest matrix is{' '}
                  {modelMatrixSize} on its longest side, which is {modelMatrixSize < cross ? 'below' : 'above'}{' '}
                  that crossover.
                </Callout>
              )}
            </div>
          </>
        )}
      </div>

      {/* ------------------------------------------------- one big matmul */}

      <div className="mt-5">
        <Label>run one large multiply on the selected device</Label>
        <div className="mt-2 grid gap-3 md:grid-cols-[1fr_auto] md:items-end">
          <Slider
            label="matrix size"
            value={Math.log2(demoN)}
            min={5}
            max={11}
            step={1}
            format={(x) => {
              const n = Math.pow(2, Math.round(x));
              return `${n} × ${n}`;
            }}
            onChange={(x) => setDemoN(Math.pow(2, Math.round(x)))}
            info={V.matShared}
          />
          <Btn onClick={runDemo} disabled={running}>
            {running ? 'Running…' : 'Multiply'}
          </Btn>
        </div>
        {demo && (
          <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Stat label="size" value={`${demo.n}×${demo.n}`} />
            <Stat label="CPU" value={`${demo.cpuMs.toFixed(1)} ms`} sub={`${demo.cpuGflops.toFixed(1)} GFLOP/s`} tone="accent" />
            <Stat
              label={backend === 'cpu' ? 'GPU (not selected)' : 'GPU'}
              value={demo.gpuMs === null ? '—' : `${demo.gpuMs.toFixed(1)} ms`}
              sub={demo.gpuGflops !== null ? `${demo.gpuGflops.toFixed(1)} GFLOP/s` : 'select a GPU above'}
              tone="ok"
            />
            <Stat
              label="result"
              value={demo.gpuMs === null ? 'CPU only' : demo.gpuWins ? `GPU ${fmt(demo.cpuMs / demo.gpuMs, 1)}× faster` : `CPU ${fmt(demo.gpuMs / demo.cpuMs, 1)}× faster`}
              tone={demo.gpuWins ? 'ok' : 'warn'}
            />
          </div>
        )}
        {demo && (
          <div className="mt-2">
            <BarMeter
              value={demo.cpuGflops}
              max={Math.max(demo.cpuGflops, demo.gpuGflops ?? 0)}
              color="var(--accent)"
              height={6}
              label={<span>CPU {demo.cpuGflops.toFixed(1)} GFLOP/s</span>}
            />
            {demo.gpuGflops !== null && (
              <div className="mt-1.5">
                <BarMeter
                  value={demo.gpuGflops}
                  max={Math.max(demo.cpuGflops, demo.gpuGflops)}
                  color="var(--ok)"
                  height={6}
                  label={<span>GPU {demo.gpuGflops.toFixed(1)} GFLOP/s</span>}
                />
              </div>
            )}
          </div>
        )}
      </div>

      {err && (
        <div className="mt-3">
          <Callout tone="err" title="Device error">
            {err}
          </Callout>
        </div>
      )}

      {/* --------------------------------------------------------- honesty */}

      <div className="mt-5">
        <Callout tone="warn" title="What this selector does and does not change">
          The GPU path here is real: a WGSL compute shader, checked against the CPU for numerical agreement
          before it is trusted, and timed end to end. It powers the benchmark and the large multiply above.
          <br />
          <br />
          Training still runs on the CPU, and the benchmark shows why. A model of this size performs matrix
          multiplies with dimensions in the tens, and at that scale the CPU is the faster device on this
          machine by a wide margin — the arithmetic finishes before a GPU dispatch has even been submitted.
          Moving training to the GPU would make it slower, not faster. That is not a limitation of the browser;
          it is exactly why real training runs use enormous batches, to give the hardware matrices big enough
          to be worth its time.
        </Callout>
      </div>

      <div className="mt-3">
        <Depth
          plain={
            <>
              A CPU has a handful of very fast, very general cores. A GPU has thousands of simple ones that
              must all do the same thing at once. Matrix multiplication suits the second arrangement perfectly,
              because every cell of the answer is independent — but only if there are enough cells to go
              around. Hand a GPU a tiny matrix and most of its hardware sits idle while the driver overhead
              dominates.
            </>
          }
          math={
            <>
              An n × n multiply costs 2n³ FLOPs and moves 3n² numbers. The ratio of work to traffic is O(n), so
              small matrices are bound by memory and dispatch latency while large ones are bound by arithmetic.
              A dispatch costs roughly a fixed 0.1 to 0.5 ms here; the crossover is where 2n³ divided by the
              GPU rate first exceeds that fixed cost.
            </>
          }
          code={
            <span className="mono text-[11px]">
              pass.dispatchWorkgroups(ceil(N/16), ceil(M/16)) — one 16×16 workgroup per output tile
            </span>
          }
        />
      </div>
    </Panel>
  );
}
