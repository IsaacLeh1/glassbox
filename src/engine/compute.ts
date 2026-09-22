/**
 * Compute backends: the CPU path Glassbox normally uses, and a real WebGPU
 * path that runs matrix multiplication in a compute shader on an actual GPU.
 *
 * The honest headline, which the UI states plainly: a GPU is not automatically
 * faster. Every dispatch costs roughly a tenth of a millisecond in driver and
 * transfer overhead, so for the small matrices inside a teaching-sized model
 * the CPU wins comfortably. The benchmark in this module measures exactly where
 * that crossover sits on your machine rather than asserting it.
 */

export type BackendId = 'cpu' | 'gpu-low' | 'gpu-high';

export interface AdapterInfo {
  id: BackendId;
  label: string;
  available: boolean;
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  /** What the browser calls this preference: integrated versus discrete. */
  powerPreference: 'low-power' | 'high-performance' | null;
  maxBufferBytes: number;
  maxWorkgroupSize: number;
  note: string;
}

export interface MachineGraphics {
  webgpu: boolean;
  adapters: AdapterInfo[];
  /** Renderer string from WebGL, which usually names the exact card. */
  webglRenderer: string | null;
  webglVendor: string | null;
  /** True when both power preferences resolved to the same physical device. */
  singleGpu: boolean;
}

function blankAdapter(id: BackendId, label: string, note: string): AdapterInfo {
  return {
    id,
    label,
    available: false,
    vendor: '',
    architecture: '',
    device: '',
    description: '',
    powerPreference: null,
    maxBufferBytes: 0,
    maxWorkgroupSize: 0,
    note,
  };
}

export function webglInfo(): { renderer: string | null; vendor: string | null } {
  try {
    const canvas = document.createElement('canvas');
    const gl = (canvas.getContext('webgl2') ?? canvas.getContext('webgl')) as WebGLRenderingContext | null;
    if (!gl) return { renderer: null, vendor: null };
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    return {
      renderer: dbg ? String(gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL)) : String(gl.getParameter(gl.RENDERER)),
      vendor: dbg ? String(gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL)) : null,
    };
  } catch {
    return { renderer: null, vendor: null };
  }
}

/** Tidies "ANGLE (NVIDIA, NVIDIA GeForce RTX 5070 Ti Laptop GPU (0x...) Direct3D11...)". */
export function prettyRenderer(raw: string | null): string | null {
  if (!raw) return null;
  const angle = raw.match(/ANGLE \(([^,]+), ([^(]+?)(?:\s*\(0x[^)]*\))?\s*(?:Direct3D|OpenGL|Vulkan)[^)]*\)/i);
  if (angle) return angle[2].trim();
  return raw.replace(/\s*\(0x[0-9a-f]+\)/i, '').trim();
}

/**
 * Ask the browser for both power preferences. On a laptop with switchable
 * graphics these usually resolve to the integrated and the discrete GPU
 * respectively, but the browser is free to return the same device for both,
 * and we report that honestly rather than inventing a second card.
 */
export async function detectGraphics(): Promise<MachineGraphics> {
  const wgl = webglInfo();
  if (!('gpu' in navigator) || !navigator.gpu) {
    return {
      webgpu: false,
      adapters: [
        blankAdapter('gpu-low', 'Integrated GPU', 'WebGPU is not available in this browser.'),
        blankAdapter('gpu-high', 'Dedicated GPU', 'WebGPU is not available in this browser.'),
      ],
      webglRenderer: wgl.renderer,
      webglVendor: wgl.vendor,
      singleGpu: true,
    };
  }

  const prefs: { id: BackendId; pref: GPUPowerPreference; label: string }[] = [
    { id: 'gpu-low', pref: 'low-power', label: 'Integrated GPU' },
    { id: 'gpu-high', pref: 'high-performance', label: 'Dedicated GPU' },
  ];

  const adapters: AdapterInfo[] = [];
  for (const { id, pref, label } of prefs) {
    try {
      const a = await navigator.gpu.requestAdapter({ powerPreference: pref });
      if (!a) {
        adapters.push(blankAdapter(id, label, 'The browser returned no adapter for this preference.'));
        continue;
      }
      const info = a.info ?? {
        vendor: '',
        architecture: '',
        device: '',
        description: '',
      };
      adapters.push({
        id,
        label,
        available: true,
        vendor: info.vendor ?? '',
        architecture: info.architecture ?? '',
        device: info.device ?? '',
        description: info.description ?? '',
        powerPreference: pref,
        maxBufferBytes: a.limits?.maxStorageBufferBindingSize ?? 0,
        maxWorkgroupSize: a.limits?.maxComputeWorkgroupSizeX ?? 0,
        note: '',
      });
    } catch (e) {
      adapters.push(blankAdapter(id, label, e instanceof Error ? e.message : String(e)));
    }
  }

  const [lo, hi] = adapters;
  const singleGpu =
    lo.available &&
    hi.available &&
    lo.vendor === hi.vendor &&
    lo.architecture === hi.architecture &&
    lo.device === hi.device;

  if (singleGpu) {
    const msg =
      'Both power preferences resolved to the same device, so this browser is only exposing one GPU.';
    lo.note = msg;
    hi.note = msg;
  }

  return { webgpu: true, adapters, webglRenderer: wgl.renderer, webglVendor: wgl.vendor, singleGpu };
}

/* ------------------------------------------------------ WebGPU matmul -- */

const MATMUL_WGSL = /* wgsl */ `
struct Dims { M: u32, K: u32, N: u32, pad: u32 };

@group(0) @binding(0) var<storage, read>       A : array<f32>;
@group(0) @binding(1) var<storage, read>       B : array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;
@group(0) @binding(3) var<uniform>             d : Dims;

const TILE : u32 = 16u;

var<workgroup> As : array<f32, 256>;
var<workgroup> Bs : array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_id)  lid : vec3<u32>) {
  let row = gid.y;
  let col = gid.x;
  var acc : f32 = 0.0;

  let tiles : u32 = (d.K + TILE - 1u) / TILE;
  for (var t : u32 = 0u; t < tiles; t = t + 1u) {
    let aCol = t * TILE + lid.x;
    let bRow = t * TILE + lid.y;

    var av : f32 = 0.0;
    if (row < d.M && aCol < d.K) { av = A[row * d.K + aCol]; }
    var bv : f32 = 0.0;
    if (bRow < d.K && col < d.N) { bv = B[bRow * d.N + col]; }

    As[lid.y * TILE + lid.x] = av;
    Bs[lid.y * TILE + lid.x] = bv;
    workgroupBarrier();

    for (var k : u32 = 0u; k < TILE; k = k + 1u) {
      acc = acc + As[lid.y * TILE + k] * Bs[k * TILE + lid.x];
    }
    workgroupBarrier();
  }

  if (row < d.M && col < d.N) {
    C[row * d.N + col] = acc;
  }
}
`;

/** A live WebGPU device with a compiled tiled matmul pipeline. */
export class GpuCompute {
  readonly device: GPUDevice;
  readonly info: AdapterInfo;
  private pipeline: GPUComputePipeline;
  private layout: GPUBindGroupLayout;
  private disposed = false;

  private constructor(device: GPUDevice, info: AdapterInfo) {
    this.device = device;
    this.info = info;
    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    this.pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      compute: { module: device.createShaderModule({ code: MATMUL_WGSL }), entryPoint: 'main' },
    });
  }

  static async create(pref: GPUPowerPreference): Promise<GpuCompute | null> {
    if (!('gpu' in navigator) || !navigator.gpu) return null;
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: pref });
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    const raw = adapter.info ?? { vendor: '', architecture: '', device: '', description: '' };
    const info: AdapterInfo = {
      id: pref === 'low-power' ? 'gpu-low' : 'gpu-high',
      label: pref === 'low-power' ? 'Integrated GPU' : 'Dedicated GPU',
      available: true,
      vendor: raw.vendor ?? '',
      architecture: raw.architecture ?? '',
      device: raw.device ?? '',
      description: raw.description ?? '',
      powerPreference: pref,
      maxBufferBytes: adapter.limits?.maxStorageBufferBindingSize ?? 0,
      maxWorkgroupSize: adapter.limits?.maxComputeWorkgroupSizeX ?? 0,
      note: '',
    };
    return new GpuCompute(device, info);
  }

  /** C = A @ B, computed on the GPU. Shapes are (M x K) and (K x N). */
  async matmul(a: Float32Array, b: Float32Array, M: number, K: number, N: number): Promise<Float32Array> {
    if (this.disposed) throw new Error('GPU device already released');
    const dev = this.device;

    const bufA = dev.createBuffer({ size: Math.max(4, a.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const bufB = dev.createBuffer({ size: Math.max(4, b.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const outBytes = Math.max(4, M * N * 4);
    const bufC = dev.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const bufD = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const readback = dev.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    dev.queue.writeBuffer(bufA, 0, a);
    dev.queue.writeBuffer(bufB, 0, b);
    dev.queue.writeBuffer(bufD, 0, new Uint32Array([M, K, N, 0]));

    const bind = dev.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: bufA } },
        { binding: 1, resource: { buffer: bufB } },
        { binding: 2, resource: { buffer: bufC } },
        { binding: 3, resource: { buffer: bufD } },
      ],
    });

    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(N / 16), Math.ceil(M / 16));
    pass.end();
    enc.copyBufferToBuffer(bufC, 0, readback, 0, outBytes);
    dev.queue.submit([enc.finish()]);

    await readback.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();

    for (const buf of [bufA, bufB, bufC, bufD, readback]) buf.destroy();
    return out.subarray(0, M * N);
  }

  destroy() {
    if (this.disposed) return;
    this.disposed = true;
    this.device.destroy();
  }
}

/* --------------------------------------------------------- benchmarks -- */

export interface BenchPoint {
  n: number;
  cpuMs: number;
  cpuGflops: number;
  gpuMs: number | null;
  gpuGflops: number | null;
  /** True when the GPU actually beat the CPU at this size. */
  gpuWins: boolean;
}

function cpuMatmulF32(a: Float32Array, b: Float32Array, M: number, K: number, N: number): Float32Array {
  const out = new Float32Array(M * N);
  for (let i = 0; i < M; i++) {
    for (let t = 0; t < K; t++) {
      const av = a[i * K + t];
      if (av === 0) continue;
      const bo = t * N;
      const oo = i * N;
      for (let j = 0; j < N; j++) out[oo + j] += av * b[bo + j];
    }
  }
  return out;
}

function randF32(n: number, seed = 1): Float32Array {
  const out = new Float32Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = (s / 4294967296) * 2 - 1;
  }
  return out;
}

/**
 * Times a square matmul on both backends at several sizes. Deliberately
 * includes upload and readback in the GPU figure, because that cost is real
 * and is precisely why small matrices lose.
 */
export async function benchmarkMatmul(
  sizes: number[],
  gpu: GpuCompute | null,
  onPoint?: (p: BenchPoint) => void,
): Promise<BenchPoint[]> {
  const points: BenchPoint[] = [];
  for (const n of sizes) {
    const a = randF32(n * n, 7);
    const b = randF32(n * n, 13);
    const flops = 2 * n * n * n;

    // Warm up, then time enough repeats to get out of timer noise.
    cpuMatmulF32(a, b, n, n, n);
    const reps = n <= 64 ? 20 : n <= 192 ? 5 : 1;
    let t0 = performance.now();
    for (let r = 0; r < reps; r++) cpuMatmulF32(a, b, n, n, n);
    const cpuMs = (performance.now() - t0) / reps;

    let gpuMs: number | null = null;
    if (gpu) {
      try {
        await gpu.matmul(a, b, n, n, n); // warm up pipeline and allocations
        t0 = performance.now();
        for (let r = 0; r < reps; r++) await gpu.matmul(a, b, n, n, n);
        gpuMs = (performance.now() - t0) / reps;
      } catch {
        gpuMs = null;
      }
    }

    const p: BenchPoint = {
      n,
      cpuMs,
      cpuGflops: flops / (cpuMs / 1000) / 1e9,
      gpuMs,
      gpuGflops: gpuMs !== null ? flops / (gpuMs / 1000) / 1e9 : null,
      gpuWins: gpuMs !== null && gpuMs < cpuMs,
    };
    points.push(p);
    onPoint?.(p);
    // Let the browser breathe between sizes.
    await new Promise((r) => setTimeout(r, 0));
  }
  return points;
}

/** Numerical agreement check between the two backends, run before trusting either. */
export async function verifyBackends(gpu: GpuCompute, n = 64): Promise<{ maxAbsErr: number; ok: boolean }> {
  const a = randF32(n * n, 3);
  const b = randF32(n * n, 5);
  const cpu = cpuMatmulF32(a, b, n, n, n);
  const got = await gpu.matmul(a, b, n, n, n);
  let maxAbsErr = 0;
  for (let i = 0; i < cpu.length; i++) maxAbsErr = Math.max(maxAbsErr, Math.abs(cpu[i] - got[i]));
  // Both sides are fp32 and accumulate in a different order, so exact equality
  // is not expected. Anything past 1e-2 at this scale means a real bug.
  return { maxAbsErr, ok: maxAbsErr < 1e-2 };
}

export const BENCH_SIZES = [32, 64, 128, 256, 512, 1024];

/** Where the GPU first becomes the faster option, or null if it never does. */
export function crossoverSize(points: BenchPoint[]): number | null {
  for (const p of points) if (p.gpuWins) return p.n;
  return null;
}
