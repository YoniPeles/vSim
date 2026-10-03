// Prints vLLM-style startup lines for a model × cluster × flags combination.
// Usage: node scripts/kvcheck.ts --model deepseek-ai/DeepSeek-V3 --cluster 8xh200 -tp 8 [-pp 1] [-dp 1] [--ep]
//        [--kv-cache-dtype fp8] [--max-model-len 32768] [--gpu-memory-utilization 0.9]
import { parseArgs } from 'node:util';
import { presetModel } from '../src/core/model/presets.ts';
import { clusterPreset } from '../src/core/hardware/clusters.ts';
import { singleInstance } from '../src/core/vllm/flags.ts';
import { evaluate, allIssues } from '../src/core/engine/evaluate.ts';
import { GiB } from '../src/core/units.ts';
import type { KvCacheDtype } from '../src/core/types.ts';

// Accept vLLM's short forms -tp/-pp/-dp.
const args = process.argv.slice(2).map((a) => (/^-(tp|pp|dp)$/.test(a) ? `-${a}` : a));
const { values } = parseArgs({
  args,
  options: {
    model: { type: 'string', default: 'meta-llama/Llama-3.1-8B-Instruct' },
    cluster: { type: 'string', default: '1xh100' },
    tp: { type: 'string', default: '1' },
    pp: { type: 'string', default: '1' },
    dp: { type: 'string', default: '1' },
    ep: { type: 'boolean', default: false },
    'kv-cache-dtype': { type: 'string', default: 'auto' },
    'max-model-len': { type: 'string' },
    'gpu-memory-utilization': { type: 'string', default: '0.92' },
  },
});

const model = presetModel(values.model!);
const cluster = clusterPreset(values.cluster!);
const dep = singleInstance({
  tp: Number(values.tp),
  pp: Number(values.pp),
  dp: Number(values.dp),
  ep: values.ep!,
  kvCacheDtype: values['kv-cache-dtype'] as KvCacheDtype,
  maxModelLen: values['max-model-len'] ? Number(values['max-model-len']) : null,
  gpuMemoryUtilization: Number(values['gpu-memory-utilization']),
});
const e = evaluate(model, cluster, dep);
const { inst, memory } = e.instances[0]!;
const r0 = memory.perRank[0]!;
const g = (b: number) => (b / GiB).toFixed(2);
console.log(`model=${model.repo} (${model.quant.label})  cluster=${cluster.name}  TP=${inst.tp} PP=${inst.pp} DP=${inst.dp}${inst.ep ? ` EP=${inst.epSize}` : ''}`);
console.log(`INFO Model loading took ${g(r0.weights)} GiB memory`);
console.log(`INFO Memory profiling: non-torch ${g(r0.nonTorch)} GiB, activation peak ${g(r0.actPeak)} GiB, CUDA graphs ${g(r0.cudagraph)} GiB`);
console.log(`INFO Available KV cache memory: ${g(r0.kv)} GiB`);
console.log(`INFO GPU KV cache size: ${memory.kvTokens.toLocaleString('en-US')} tokens`);
console.log(
  `INFO Maximum concurrency for ${inst.maxModelLen.toLocaleString('en-US')} tokens per request: ${memory.maxConcurrency.toFixed(2)}x`,
);
for (const i of allIssues(e)) console.log(`${i.level.toUpperCase()} ${i.msg}`);
