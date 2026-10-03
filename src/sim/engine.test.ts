import { describe, expect, it } from 'vitest';
import { findPreset } from '../core/model/presets.ts';
import { CLUSTER_PRESETS } from '../core/hardware/clusters.ts';
import { DEFAULT_FLAGS } from '../core/vllm/flags.ts';
import type { VllmFlags } from '../core/types.ts';
import { DEFAULT_WORKLOAD } from '../core/engine/analytic.ts';
import { CostModel } from '../core/engine/cost.ts';
import { ttftUnloaded, decodeStep } from '../core/engine/analytic.ts';
import { Simulator } from './engine.ts';
import type { SimInputs, SimWorkload } from './types.ts';

const W: SimWorkload = { ...DEFAULT_WORKLOAD, spread: 0, arrival: 'closed', rate: 1 };

function inputs(repo: string, cluster: string, flags: Partial<VllmFlags>, w: Partial<SimWorkload> = {}, pd?: Partial<VllmFlags>): SimInputs {
  const f = { ...DEFAULT_FLAGS, ...flags };
  return {
    snapshot: findPreset(repo)!,
    cluster: CLUSTER_PRESETS.find((c) => c.id === cluster)!.build,
    deployment: pd
      ? {
          instances: [
            { id: 'prefill', role: 'prefill', flags: f, gpuStart: 0 },
            { id: 'decode', role: 'decode', flags: { ...DEFAULT_FLAGS, ...pd }, gpuStart: f.tp * f.pp * f.dp },
          ],
        }
      : { instances: [{ id: 'serve', role: 'mixed', flags: f, gpuStart: 0 }] },
    workload: { ...W, ...w },
    routingSkew: 0.3,
    seed: 7,
  };
}

describe('scheduler simulation', () => {
  it('a single request matches the analytic TTFT and ITL', () => {
    const sim = new Simulator(inputs('meta-llama/Llama-3.1-8B-Instruct', '1xh100', {}, { concurrency: 1, isl: 4096, osl: 64 }));
    sim.runUntil(5, 1e9);
    const f = sim.frame(1);
    const ev = sim.evaluation.instances[0]!;
    const cost = new CostModel(sim.model, ev.inst, ev.placement, sim.evaluation.cluster, sim.evaluation.calib);
    const ttft = ttftUnloaded(cost, 4096).time;
    expect(f.stats.ttft.p50).toBeCloseTo(ttft, 3);
    const itl = decodeStep(cost, 1, 4096 + 32).time;
    expect(f.stats.itl.p50 / itl).toBeGreaterThan(0.97);
    expect(f.stats.itl.p50 / itl).toBeLessThan(1.03);
  });

  it('conserves blocks and completes requests under load with preemption', () => {
    // Tiny KV budget forces preemptions.
    const sim = new Simulator(
      inputs('meta-llama/Llama-3.1-8B-Instruct', '1xh100', { kvCacheMemoryBytes: 2 * 1024 ** 3, maxModelLen: 8192 }, { concurrency: 64, isl: 2048, osl: 512, spread: 0.3 }),
    );
    sim.runUntil(60, 1e9);
    const f = sim.frame(1);
    expect(f.stats.completed).toBeGreaterThan(50);
    expect(f.stats.preemptions).toBeGreaterThan(0);
    expect(sim.auditBlocks().leaked).toBe(0);
  });

  it('prefix caching hits on shared system prompts', () => {
    const sim = new Simulator(inputs('Qwen/Qwen3-8B', '1xh100', {}, { concurrency: 16, isl: 4096, osl: 64, prefixHit: 0.75 }));
    sim.runUntil(30, 1e9);
    expect(sim.frame(1).stats.prefixHitRate).toBeGreaterThan(0.5);
  });

  it('DP lockstep MoE + EP runs and stays consistent', () => {
    const sim = new Simulator(
      inputs('deepseek-ai/DeepSeek-V3', '4x8xh200', { tp: 8, dp: 4, ep: true, all2allBackend: 'deepep_low_latency' }, { concurrency: 128, isl: 1024, osl: 256 }),
    );
    sim.runUntil(20, 1e9);
    const f = sim.frame(1);
    expect(f.stats.completed).toBeGreaterThan(0);
    expect(f.links[0]!.inter).toBeGreaterThan(0);
    expect(sim.auditBlocks().leaked).toBe(0);
  });

  it('PP keeps multiple batches in flight and spec decode raises tokens per step', () => {
    const base = new Simulator(inputs('meta-llama/Llama-3.3-70B-Instruct', '8xh100', { tp: 4, pp: 2 }, { concurrency: 64, isl: 1024, osl: 256 }));
    base.runUntil(20, 1e9);
    const spec = new Simulator(
      inputs('meta-llama/Llama-3.3-70B-Instruct', '8xh100', { tp: 8, speculative: { method: 'eagle3', k: 3, acceptance: 0.8, decay: 0.85, draftParams: 0 } }, { concurrency: 8, isl: 1024, osl: 256 }),
    );
    spec.runUntil(20, 1e9);
    const plain = new Simulator(inputs('meta-llama/Llama-3.3-70B-Instruct', '8xh100', { tp: 8 }, { concurrency: 8, isl: 1024, osl: 256 }));
    plain.runUntil(20, 1e9);
    expect(base.frame(1).stats.completed).toBeGreaterThan(0);
    expect(spec.frame(1).stats.itl.p50).toBeLessThan(plain.frame(1).stats.itl.p50);
  });

  it('P/D moves requests from prefill to decode', () => {
    const sim = new Simulator(
      inputs('deepseek-ai/DeepSeek-V3', '2x8xb300', { tp: 8 }, { concurrency: 64, isl: 4096, osl: 128 }, { tp: 1, dp: 8, ep: true }),
    );
    sim.runUntil(20, 1e9);
    const f = sim.frame(1);
    expect(f.stats.completed).toBeGreaterThan(10);
    expect(f.links[0]!.kvx).toBeGreaterThan(0);
    expect(sim.auditBlocks().leaked).toBe(0);
  });

  it('simulates 2000 concurrent users much faster than real time', () => {
    const sim = new Simulator(inputs('meta-llama/Llama-3.1-8B-Instruct', '8xh100', { dp: 8 }, { concurrency: 2000, isl: 2048, osl: 512, prefixHit: 0.2, spread: 0.3 }));
    const t0 = performance.now();
    sim.runUntil(60, 1e9);
    const ms = performance.now() - t0;
    expect(sim.frame(1).stats.completed).toBeGreaterThan(1000);
    // 60 s of simulated serving must take a small fraction of 60 s of CPU.
    expect(ms).toBeLessThan(6000);
  });

  it('is deterministic for a seed', () => {
    const a = new Simulator(inputs('Qwen/Qwen3-8B', '1xh100', {}, { concurrency: 32, spread: 0.4 }));
    const b = new Simulator(inputs('Qwen/Qwen3-8B', '1xh100', {}, { concurrency: 32, spread: 0.4 }));
    a.runUntil(10, 1e9);
    b.runUntil(10, 1e9);
    expect(a.frame(1).stats).toEqual(b.frame(1).stats);
  });
});
