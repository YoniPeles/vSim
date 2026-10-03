// Compare vSim's predictions with real observations and (optionally) fit calibration constants.
//
// Usage: node scripts/calibrate.ts observations.json [--fit]
//
// observations.json is an array of runs you measured yourself:
// [{
//   "name": "llama8b-h100",
//   "model": "meta-llama/Llama-3.1-8B-Instruct",       // a bundled preset repo id
//   "cluster": "1xh100",                               // a cluster preset id
//   "flags": { "tp": 1, "gpuMemoryUtilization": 0.9 }, // vLLM flags as in src/core/types.ts
//   "log": "runs/llama8b.log",                          // optional: vLLM startup log to parse
//   "observed": {                                       // optional: anything you measured
//     "kvGiB": 53.7, "kvTokens": 439984, "weightsGiB": 14.99,
//     "itl": [{ "batch": 64, "ctx": 2048, "ms": 14.1 }],
//     "ttft": [{ "isl": 8192, "ms": 610 }]
//   }
// }]
//
// --fit runs a coordinate search over η/α constants per GPU architecture, minimizing the squared
// log error, and prints overrides to paste into src/core/hardware/calib.ts.
import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { presetModel } from '../src/core/model/presets.ts';
import { clusterPreset } from '../src/core/hardware/clusters.ts';
import { calibrationFor } from '../src/core/hardware/calib.ts';
import { singleInstance } from '../src/core/vllm/flags.ts';
import { parseVllmLog } from '../src/core/vllm/logParse.ts';
import { evaluate } from '../src/core/engine/evaluate.ts';
import { CostModel } from '../src/core/engine/cost.ts';
import { decodeStep, ttftUnloaded } from '../src/core/engine/analytic.ts';
import type { Calibration, GpuArch, VllmFlags } from '../src/core/types.ts';
import { GiB } from '../src/core/units.ts';

interface Obs {
  name: string;
  model: string;
  cluster: string;
  flags: Partial<VllmFlags>;
  log?: string;
  observed?: {
    kvGiB?: number;
    kvTokens?: number;
    weightsGiB?: number;
    itl?: { batch: number; ctx: number; ms: number }[];
    ttft?: { isl: number; ms: number }[];
  };
}

interface Point {
  run: string;
  arch: string;
  metric: string;
  real: number;
  predict: (c: Partial<Calibration>) => number;
}

const [file, ...rest] = process.argv.slice(2);
if (!file) {
  console.error('usage: node scripts/calibrate.ts observations.json [--fit]');
  process.exit(1);
}
const fit = rest.includes('--fit');
const runs = JSON.parse(await readFile(file, 'utf8')) as Obs[];
const points: Point[] = [];

for (const r of runs) {
  const model = presetModel(r.model);
  const cluster = clusterPreset(r.cluster);
  const dep = singleInstance(r.flags);
  const arch = cluster.gpu.arch;
  const obs = { ...r.observed };
  if (r.log) {
    const f = parseVllmLog(await readFile(resolve(dirname(file), r.log), 'utf8'));
    obs.kvGiB ??= f.kvGiB;
    obs.kvTokens ??= f.kvTokens;
    obs.weightsGiB ??= f.weightsGiB;
  }
  const ev = (c: Partial<Calibration>) => evaluate(model, cluster, dep, c);
  const cost = (c: Partial<Calibration>) => {
    const e = ev(c);
    const i = e.instances[0]!;
    return new CostModel(model, i.inst, i.placement, cluster, e.calib);
  };
  if (obs.weightsGiB) points.push({ run: r.name, arch, metric: 'weights GiB', real: obs.weightsGiB, predict: (c) => ev(c).instances[0]!.memory.perRank[0]!.weights / GiB });
  if (obs.kvGiB) points.push({ run: r.name, arch, metric: 'KV GiB', real: obs.kvGiB, predict: (c) => ev(c).instances[0]!.memory.perRank[0]!.kv / GiB });
  if (obs.kvTokens) points.push({ run: r.name, arch, metric: 'KV tokens', real: obs.kvTokens, predict: (c) => ev(c).instances[0]!.memory.kvTokens });
  for (const x of obs.itl ?? []) points.push({ run: r.name, arch, metric: `ITL b${x.batch} ctx${x.ctx} ms`, real: x.ms, predict: (c) => decodeStep(cost(c), x.batch, x.ctx).time * 1e3 });
  for (const x of obs.ttft ?? []) points.push({ run: r.name, arch, metric: `TTFT ${x.isl} ms`, real: x.ms, predict: (c) => ttftUnloaded(cost(c), x.isl).time * 1e3 });
}

const err = (p: Point, c: Partial<Calibration>) => Math.log(p.predict(c) / p.real);
function report(overrides: Record<string, Partial<Calibration>>) {
  for (const p of points) {
    const v = p.predict(overrides[p.arch] ?? {});
    console.log(`${p.run.padEnd(24)} ${p.metric.padEnd(26)} real ${p.real.toFixed(2).padStart(10)}  vSim ${v.toFixed(2).padStart(10)}  ${(((v - p.real) / p.real) * 100).toFixed(1).padStart(6)}%`);
  }
}

if (!points.length) {
  console.error('No observations with values found.');
  process.exit(1);
}
report({});
if (fit) {
  const knobs: (keyof Calibration)[] = ['etaCompute', 'etaMem', 'etaAttnCompute', 'minKernel', 'alphaScaleUp', 'actRho', 'nonTorchBase', 'cudagraphPerSizePerLayer'];
  const out: Record<string, Partial<Calibration>> = {};
  for (const arch of new Set(points.map((p) => p.arch))) {
    const pts = points.filter((p) => p.arch === arch);
    const base = calibrationFor(arch as GpuArch);
    const c: Partial<Calibration> = {};
    const loss = () => pts.reduce((s, p) => s + err(p, c) ** 2, 0);
    let best = loss();
    for (let round = 0; round < 6; round++) {
      for (const k of knobs) {
        for (const f of [0.7, 0.85, 0.95, 1.05, 1.18, 1.4]) {
          const prev = c[k];
          const b = base[k] as number;
          // Stay within ½×–2× of the first-principles default: few observations overfit easily.
          (c as Record<string, number>)[k] = Math.min(2 * b, Math.max(0.5 * b, ((prev as number | undefined) ?? b) * f));
          const l = loss();
          if (l < best - 1e-9) best = l;
          else if (prev === undefined) delete c[k];
          else (c as Record<string, number>)[k] = prev as number;
        }
      }
    }
    out[arch] = c;
    console.log(`\n${arch}: RMS log error ${Math.sqrt(best / pts.length).toFixed(3)} with overrides`, JSON.stringify(c));
  }
  console.log('\nAfter fitting:');
  report(out);
}
