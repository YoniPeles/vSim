import type { ClusterSpec, InstanceSpec, ModelSpec, VllmFlags } from '../types.ts';
import { DEFAULT_FLAGS } from './flags.ts';

/** `vllm serve …` for one instance, listing only flags that differ from vLLM defaults. */
export function serveCommand(model: ModelSpec, inst: InstanceSpec, cluster: ClusterSpec, pd: boolean): string[] {
  const f: VllmFlags = inst.flags;
  const d = DEFAULT_FLAGS;
  const args: string[] = [`vllm serve ${model.repo}`];
  if (f.tp !== 1) args.push(`--tensor-parallel-size ${f.tp}`);
  if (f.pp !== 1) args.push(`--pipeline-parallel-size ${f.pp}`);
  if (f.dp !== 1) {
    args.push(`--data-parallel-size ${f.dp}`);
    const perNode = Math.floor(cluster.gpusPerNode / (f.tp * f.pp));
    if (perNode >= 1 && f.dp > perNode) args.push(`--data-parallel-size-local ${perNode}`);
  }
  if (f.ep) args.push('--enable-expert-parallel');
  if (f.ep && f.all2allBackend !== d.all2allBackend) args.push(`--all2all-backend ${f.all2allBackend}`);
  if (f.numRedundantExperts > 0) {
    args.push('--enable-eplb', `--eplb-config '{"num_redundant_experts":${f.numRedundantExperts}}'`);
  }
  if (f.dcp > 1) args.push(`--decode-context-parallel-size ${f.dcp}`);
  if (f.quantization === 'fp8') args.push('--quantization fp8');
  if (f.kvCacheDtype !== d.kvCacheDtype) args.push(`--kv-cache-dtype ${f.kvCacheDtype}`);
  if (f.gpuMemoryUtilization !== d.gpuMemoryUtilization) args.push(`--gpu-memory-utilization ${f.gpuMemoryUtilization}`);
  if (f.kvCacheMemoryBytes !== null) args.push(`--kv-cache-memory-bytes ${f.kvCacheMemoryBytes}`);
  if (f.maxModelLen !== null) args.push(`--max-model-len ${f.maxModelLen}`);
  if (f.maxNumSeqs !== null) args.push(`--max-num-seqs ${f.maxNumSeqs}`);
  if (f.maxNumBatchedTokens !== null) args.push(`--max-num-batched-tokens ${f.maxNumBatchedTokens}`);
  if (f.blockSize !== null) args.push(`--block-size ${f.blockSize}`);
  if (!f.enablePrefixCaching) args.push('--no-enable-prefix-caching');
  if (!f.enableChunkedPrefill) args.push('--no-enable-chunked-prefill');
  if (f.performanceMode !== d.performanceMode) args.push(`--performance-mode ${f.performanceMode}`);
  if (f.optimizationLevel !== d.optimizationLevel) args.push(`-O${f.optimizationLevel}`);
  if (f.speculative) {
    const s = f.speculative;
    const cfg: Record<string, unknown> = { method: s.method, num_speculative_tokens: s.k };
    if (s.method === 'ngram') cfg['prompt_lookup_max'] = 4;
    if (s.method === 'eagle' || s.method === 'eagle3' || s.method === 'draft_model') cfg['model'] = '<draft-model-repo>';
    args.push(`--speculative-config '${JSON.stringify(cfg)}'`);
  }
  if (pd) {
    args.push(`--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'`);
  }
  return args;
}
