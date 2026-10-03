import { describe, expect, it } from 'vitest';
import fc from 'fast-check';
import { presetModel } from '../model/presets.ts';
import { clusterPreset } from '../hardware/clusters.ts';
import { singleInstance } from '../vllm/flags.ts';
import { evaluate, allIssues } from './evaluate.ts';
import { ppPartition } from './placement.ts';
import { GiB } from '../units.ts';

describe('placement', () => {
  it('ppPartition matches vLLM get_pp_indices', () => {
    // 61 layers over 4 stages: [15, 15, 16, 15] (remainder goes to stages before the last).
    expect(ppPartition(61, 4).map(([a, b]) => b - a)).toEqual([15, 15, 16, 15]);
    expect(ppPartition(80, 3).map(([a, b]) => b - a)).toEqual([27, 27, 26]);
  });

  it('EP places whole experts: DeepSeek-V3 TP8×DP4 → 8 experts per GPU', () => {
    const e = evaluate(presetModel('deepseek-ai/DeepSeek-V3'), clusterPreset('4x8xh200'), singleInstance({ tp: 8, dp: 4, ep: true }));
    const { placement, inst } = e.instances[0]!;
    expect(inst.epSize).toBe(32);
    expect(placement.shards.every((s) => s.experts?.count === 8)).toBe(true);
    expect(placement.epGroups[0]!.domains).toBe(4);
  });

  it('TP shards attention; KV heads replicate past nKV', () => {
    const e = evaluate(presetModel('Qwen/Qwen3-235B-A22B'), clusterPreset('8xh100'), singleInstance({ tp: 8 }));
    expect(e.instances[0]!.placement.shards[0]!.kvReplicated).toBe(true);
    expect(allIssues(e).some((i) => i.msg.includes('replicated'))).toBe(true);
  });
});

describe('memory model', () => {
  it('Llama-3.1-8B on 1×H100 lands in the range vLLM reports', () => {
    const e = evaluate(presetModel('meta-llama/Llama-3.1-8B-Instruct'), clusterPreset('1xh100'), singleInstance({ gpuMemoryUtilization: 0.9 }));
    const m = e.instances[0]!.memory;
    expect(m.perRank[0]!.weights / GiB).toBeCloseTo(14.96, 1);
    // vLLM logs ≈ 53–56 GiB / ≈ 440–460K tokens for this setup.
    expect(m.perRank[0]!.kv / GiB).toBeGreaterThan(52);
    expect(m.perRank[0]!.kv / GiB).toBeLessThan(57);
    expect(m.kvTokens).toBeGreaterThan(420_000);
    expect(m.kvTokens).toBeLessThan(470_000);
  });

  it('flags a model that does not fit', () => {
    const e = evaluate(presetModel('meta-llama/Llama-3.3-70B-Instruct'), clusterPreset('1xh100'), singleInstance());
    expect(allIssues(e).some((i) => i.level === 'error')).toBe(true);
  });

  it('memory parts never exceed the request; KV grows with utilization; weights shrink with TP', () => {
    const model = presetModel('Qwen/Qwen3-32B');
    const cluster = clusterPreset('8xh100');
    fc.assert(
      fc.property(fc.constantFrom(1, 2, 4, 8), fc.double({ min: 0.5, max: 0.95, noNaN: true }), (tp, util) => {
        const a = evaluate(model, cluster, singleInstance({ tp, gpuMemoryUtilization: util })).instances[0]!.memory.perRank[0]!;
        const b = evaluate(model, cluster, singleInstance({ tp, gpuMemoryUtilization: util + 0.04 })).instances[0]!.memory.perRank[0]!;
        fc.pre(a.kv > 0); // only meaningful when the model fits
        const sum = a.weights + a.nonTorch + a.persistent + a.actPeak + a.cudagraph + a.kv;
        expect(sum).toBeLessThanOrEqual(a.requested + 1);
        expect(b.kv).toBeGreaterThanOrEqual(a.kv);
        if (tp < 8) {
          const c = evaluate(model, cluster, singleInstance({ tp: tp * 2, gpuMemoryUtilization: util })).instances[0]!.memory.perRank[0]!;
          expect(c.weights).toBeLessThan(a.weights);
        }
      }),
      { numRuns: 40 },
    );
  });
});
