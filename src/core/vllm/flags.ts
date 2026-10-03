import type { All2AllBackend, Deployment, KvCacheDtype, SpecConfig, SpecMethod, VllmFlags } from '../types.ts';

/** vLLM defaults (v0.30, Model Runner V2). null = "let vLLM derive it". */
export const DEFAULT_FLAGS: VllmFlags = {
  tp: 1,
  pp: 1,
  dp: 1,
  ep: false,
  dcp: 1,
  quantization: 'auto',
  kvCacheDtype: 'auto',
  gpuMemoryUtilization: 0.92,
  kvCacheMemoryBytes: null,
  maxModelLen: null,
  maxNumSeqs: null,
  maxNumBatchedTokens: null,
  blockSize: null,
  enablePrefixCaching: true,
  enableChunkedPrefill: true,
  performanceMode: 'balanced',
  optimizationLevel: 2,
  all2allBackend: 'allgather_reducescatter',
  numRedundantExperts: 0,
  speculative: null,
};

export const KV_DTYPES: { id: KvCacheDtype; label: string; mlaOnly?: boolean }[] = [
  { id: 'auto', label: 'auto (model dtype)' },
  { id: 'fp8', label: 'fp8 (e4m3)' },
  { id: 'fp8_e5m2', label: 'fp8_e5m2' },
  { id: 'nvfp4', label: 'nvfp4' },
  { id: 'fp8_ds_mla', label: 'fp8_ds_mla (656 B/token)', mlaOnly: true },
  { id: 'nvfp4_ds_mla', label: 'nvfp4_ds_mla (352 B/token)', mlaOnly: true },
];

export const ALL2ALL_BACKENDS: { id: All2AllBackend; label: string }[] = [
  { id: 'allgather_reducescatter', label: 'allgather_reducescatter (default)' },
  { id: 'deepep_high_throughput', label: 'DeepEP high-throughput' },
  { id: 'deepep_low_latency', label: 'DeepEP low-latency' },
  { id: 'pplx', label: 'pplx-kernels' },
  { id: 'flashinfer_nvlink_one_sided', label: 'FlashInfer NVLink one-sided' },
  { id: 'naive', label: 'naive (broadcast)' },
];

export const SPEC_METHODS: { id: SpecMethod; label: string; acceptance: number; decay: number }[] = [
  { id: 'mtp', label: 'MTP (model’s own MTP heads)', acceptance: 0.85, decay: 0.85 },
  { id: 'eagle3', label: 'EAGLE-3', acceptance: 0.8, decay: 0.85 },
  { id: 'eagle', label: 'EAGLE', acceptance: 0.72, decay: 0.82 },
  { id: 'draft_model', label: 'Draft model', acceptance: 0.7, decay: 0.85 },
  { id: 'ngram', label: 'n-gram prompt lookup', acceptance: 0.35, decay: 0.8 },
];

export function defaultSpec(method: SpecMethod): SpecConfig {
  const m = SPEC_METHODS.find((s) => s.id === method) ?? SPEC_METHODS[0]!;
  return {
    method,
    k: method === 'mtp' ? 1 : method === 'ngram' ? 4 : 3,
    acceptance: m.acceptance,
    decay: m.decay,
    draftParams: method === 'draft_model' ? 1e9 : 0,
  };
}

export function singleInstance(flags: Partial<VllmFlags> = {}): Deployment {
  return { instances: [{ id: 'serve', role: 'mixed', flags: { ...DEFAULT_FLAGS, ...flags }, gpuStart: 0 }] };
}

export function disaggregated(prefill: Partial<VllmFlags>, decode: Partial<VllmFlags>): Deployment {
  const p = { ...DEFAULT_FLAGS, ...prefill };
  const d = { ...DEFAULT_FLAGS, ...decode };
  return {
    instances: [
      { id: 'prefill', role: 'prefill', flags: p, gpuStart: 0 },
      { id: 'decode', role: 'decode', flags: d, gpuStart: p.tp * p.pp * p.dp },
    ],
  };
}

/** Expected accepted draft tokens per step: Σ_{i=1..k} Π_{j<i} a·decay^j. */
export function expectedAccepted(spec: SpecConfig): number {
  let p = 1;
  let s = 0;
  for (let i = 0; i < spec.k; i++) {
    p *= Math.min(1, spec.acceptance * spec.decay ** i);
    s += p;
  }
  return s;
}
