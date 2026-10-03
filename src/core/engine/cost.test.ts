import { describe, expect, it } from 'vitest';
import fc from 'fast-check';
import { presetModel } from '../model/presets.ts';
import { clusterPreset, buildCluster } from '../hardware/clusters.ts';
import { calibrationFor } from '../hardware/calib.ts';
import { defaultSpec, singleInstance } from '../vllm/flags.ts';
import { evaluate } from './evaluate.ts';
import { CostModel } from './cost.ts';
import { CommModel } from './comm.ts';
import { decodeStep, steadyState, ttftUnloaded, DEFAULT_WORKLOAD } from './analytic.ts';
import type { VllmFlags } from '../types.ts';
import { MiB } from '../units.ts';

function costFor(repo: string, cluster: string, flags: Partial<VllmFlags>) {
  const m = presetModel(repo);
  const c = clusterPreset(cluster);
  const e = evaluate(m, c, singleInstance(flags));
  const ie = e.instances[0]!;
  return { cost: new CostModel(m, ie.inst, ie.placement, c, e.calib), ev: ie };
}

describe('roofline cost model', () => {
  const llama = costFor('meta-llama/Llama-3.3-70B-Instruct', '8xh100', { tp: 8 });
  const dsv3 = costFor('deepseek-ai/DeepSeek-V3', '8xh200', { tp: 8 });

  it('decode step time never decreases with batch size or context', () => {
    for (const { cost } of [llama, dsv3]) {
      fc.assert(
        fc.property(fc.integer({ min: 1, max: 512 }), fc.integer({ min: 1, max: 256 }), fc.integer({ min: 128, max: 32768 }), (b, db, ctx) => {
          const a = decodeStep(cost, b, ctx).time;
          expect(decodeStep(cost, b + db, ctx).time).toBeGreaterThanOrEqual(a * 0.999);
          expect(decodeStep(cost, b, ctx * 2).time).toBeGreaterThanOrEqual(a * 0.999);
        }),
        { numRuns: 60 },
      );
    }
  });

  it('TTFT grows super-linearly with prompt length (attention is quadratic)', () => {
    const t8 = ttftUnloaded(llama.cost, 8192).time;
    const t32 = ttftUnloaded(llama.cost, 32768).time;
    expect(t32 / t8).toBeGreaterThan(4);
  });

  it('small-batch decode is memory-bound, big prefill is compute-bound', () => {
    expect(decodeStep(llama.cost, 1, 1024).bound).not.toBe('compute');
    expect(ttftUnloaded(llama.cost, 8192).steps[0]!.bound).toBe('compute');
  });

  it('MTP speculative decoding raises per-user speed at low load', () => {
    const spec = costFor('deepseek-ai/DeepSeek-V3', '8xh200', { tp: 8, speculative: defaultSpec('mtp') });
    const w = { ...DEFAULT_WORKLOAD, concurrency: 4 };
    const base = steadyState(dsv3.cost, dsv3.ev, w, 4).tokPerUser;
    const fast = steadyState(spec.cost, spec.ev, w, 4).tokPerUser;
    expect(fast).toBeGreaterThan(base * 1.2);
  });

  it('per-layer trace sums to the GPU time of the step', () => {
    const s = steadyState(dsv3.cost, dsv3.ev, DEFAULT_WORKLOAD, 32);
    const d = dsv3.cost.step(s.shape, true);
    const traced = d.trace!.reduce((a, p) => a + p.dur, 0);
    // The trace omits only kernel-launch overhead and drafting.
    expect(traced).toBeLessThanOrEqual(d.cost.gpu + 1e-9);
    expect(traced).toBeGreaterThan(d.cost.gpu - d.cost.parts.launch - d.cost.parts.draft - 1e-6);
  });
});

describe('collectives', () => {
  const h200 = buildCluster({ gpu: 'h200', nodes: 4, gpusPerNode: 8, scaleOut: 'ib-ndr' });
  const comm = new CommModel(h200, calibrationFor('sm90'));

  it('is free for a single rank', () => {
    expect(comm.allReduce(MiB, { n: 1, perDomain: 1, domains: 1 }).t).toBe(0);
  });

  it('all-reduce across IB is slower than inside NVLink', () => {
    const intra = comm.allReduce(16 * MiB, { n: 8, perDomain: 8, domains: 1 });
    const inter = comm.allReduce(16 * MiB, { n: 16, perDomain: 8, domains: 2 });
    expect(inter.t).toBeGreaterThan(intra.t);
    expect(inter.inter).toBeGreaterThan(0);
    expect(intra.inter).toBe(0);
  });

  it('DeepEP node dedup moves fewer bytes across nodes', () => {
    const g = { n: 32, perDomain: 8, domains: 4 };
    const plain = comm.allToAll(64 * MiB, g);
    const dedup = comm.allToAll(64 * MiB, g, 8);
    expect(dedup.inter).toBeLessThan(plain.inter);
  });

  it('small all-reduces pick the custom one-shot kernel on Hopper', () => {
    expect(comm.allReduce(64 * 1024, { n: 8, perDomain: 8, domains: 1 }).algo).toMatch(/custom AR/);
  });
});
