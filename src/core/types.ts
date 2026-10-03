// Core domain types shared by the analytical engine, the scheduler simulation and the UI.
// Units: bytes, seconds, FLOP/s, tokens.

// ───────────────────────────── dtypes ─────────────────────────────

export type WeightDtype =
  | 'f32'
  | 'bf16'
  | 'fp8' // per-tensor/channel fp8
  | 'fp8_block' // DeepSeek-style 128×128 block scales
  | 'int8'
  | 'mxfp4' // e2m1 + e8m0 scale per 32
  | 'nvfp4' // e2m1 + fp8 scale per 16
  | 'int4_g128'; // AWQ / GPTQ / compressed-tensors W4A16 g128

export type ComputeTier = 'bf16' | 'fp8' | 'fp4' | 'int8';

// ───────────────────────────── model ─────────────────────────────

export interface GqaAttn {
  kind: 'gqa';
  /** full = global causal; sliding = window; chunked = Llama-4 local chunks. */
  scope: 'full' | 'sliding' | 'chunked';
  window: number; // 0 for full
  nQ: number;
  nKV: number;
  dQK: number;
  dV: number;
}

export interface MlaAttn {
  kind: 'mla';
  nQ: number;
  qLora: number; // 0 = no q compression
  kvLora: number;
  rope: number;
  nope: number;
  dV: number;
  /** DeepSeek Sparse Attention (V3.2): lightning indexer + top-k token selection. */
  dsa?: { nIdx: number; dIdx: number; topk: number };
}

export interface LinearAttn {
  kind: 'linear';
  variant: 'gdn' | 'kda' | 'mamba';
  nK: number;
  nV: number;
  dK: number;
  dV: number;
  conv: number; // short conv kernel size
}

export type AttnSpec = GqaAttn | MlaAttn | LinearAttn;

export interface DenseFfn {
  kind: 'dense';
  inter: number;
}

export interface MoeFfn {
  kind: 'moe';
  E: number;
  topK: number;
  eInter: number;
  nShared: number;
  sInter: number;
}

export type FfnSpec = DenseFfn | MoeFfn;

/** Parameter counts (not bytes) of one decoder layer, split by how they are sharded/quantized. */
export interface LayerParams {
  attn: number;
  norms: number;
  dense: number;
  shared: number;
  expertEach: number;
  router: number;
}

export interface LayerSpec {
  index: number;
  attn: AttnSpec;
  ffn: FfnSpec;
  p: LayerParams;
}

/** Identical layers collapsed so the cost model is O(groups), not O(layers). */
export interface LayerGroup {
  key: string;
  label: string;
  indices: number[];
  layer: LayerSpec;
}

export interface QuantScheme {
  method: string; // 'none' | 'fp8' | 'mxfp4' | 'modelopt-nvfp4' | 'awq' | 'gptq' | 'compressed-tensors' …
  label: string;
  attn: WeightDtype;
  dense: WeightDtype;
  shared: WeightDtype;
  experts: WeightDtype;
  router: WeightDtype;
  embed: WeightDtype;
  lmHead: WeightDtype;
  /** KV-cache dtype the checkpoint recommends (modelopt kv_cache_quant_algo), if any. */
  kvHint?: 'fp8';
}

/** Context-length transforms used by attention layers: KV read per token is ctxEff(ctx). */
export interface CtxWindow {
  kind: 'sliding' | 'chunked' | 'topk';
  size: number;
}

export interface ModelSpec {
  repo: string;
  revision: string;
  label: string;
  modelType: string;
  architectures: string[];
  hidden: number;
  vocab: number;
  maxPos: number;
  tied: boolean;
  baseDtype: WeightDtype;
  layers: LayerSpec[];
  groups: LayerGroup[];
  /** Multi-token-prediction modules; loaded by vLLM only for MTP speculative decoding. */
  mtp: { layers: LayerSpec[]; extraParams: number };
  quant: QuantScheme;
  /** Distinct windowed context transforms (index i ↔ window id i+1; id 0 = full context). */
  windows: CtxWindow[];
  params: {
    total: number; // main model (no MTP, no vision)
    active: number; // per token
    embed: number;
    lmHead: number;
    mtp: number;
    vision: number;
  };
  warnings: string[];
  source: 'preset' | 'hub';
  hfSafetensors?: { total: number; parameters: Record<string, number> };
}

// ───────────────────────────── hardware ─────────────────────────────

export type GpuArch = 'sm80' | 'sm89' | 'sm90' | 'sm100' | 'sm103' | 'sm120' | 'gfx942' | 'gfx950';

export interface GpuSpec {
  id: string;
  name: string;
  vendor: 'nvidia' | 'amd';
  arch: GpuArch;
  /** Memory as torch sees it (cudaMemGetInfo total), not the marketing figure. */
  memBytes: number;
  memLabel: string;
  hbmBW: number;
  /** Dense peak FLOP/s per tier. */
  peak: { bf16: number; fp8?: number; fp4?: number; int8?: number };
  /** Scale-up bandwidth per GPU per direction (NVLink / Infinity Fabric aggregate); 0 if none. */
  scaleUpBWDir: number;
  /** For switchless meshes: bandwidth of one peer link per direction. */
  meshLinkBWDir?: number;
  pcieBWDir: number;
  tdpW: number;
  approximate?: boolean;
  notes?: string;
}

export type ScaleUpKind = 'nvswitch' | 'mesh' | 'pcie';

export interface ClusterSpec {
  id: string;
  name: string;
  gpu: GpuSpec;
  /** Physical nodes (HGX boxes, or NVL72 compute trays). */
  nodes: number;
  gpusPerNode: number;
  /** GPUs per scale-up (NVLink) domain: 8 for HGX, 72 for NVL72, 1 for PCIe cards. */
  domainSize: number;
  scaleUp: ScaleUpKind;
  nvls: boolean;
  scaleOut: {
    kind: string; // 'IB NDR 400G' …
    bwPerGpuDir: number; // bytes/s per GPU per direction
    latency: number;
    oversub: number;
  };
}

export interface Calibration {
  arch: string;
  /** GEMM compute efficiency at large M. */
  etaCompute: number;
  /** Achievable fraction of HBM bandwidth. */
  etaMem: number;
  etaAttnCompute: number;
  etaScaleUp: number;
  etaScaleOut: number;
  /** Per-collective fixed latency (s). */
  alphaScaleUp: number;
  alphaScaleOut: number;
  alphaA2A: number;
  alphaP2P: number;
  alphaKvx: number;
  /** Minimum GPU time of one small kernel inside a CUDA graph (s); floors tiny-batch roofline. */
  minKernel: number;
  /** Kernel launch overheads (s). */
  launchPiecewisePerLayer: number;
  launchEagerPerLayer: number;
  /** Host-side scheduler/runner overhead per step: base + per running request + per new request. */
  hostBase: number;
  hostPerReq: number;
  hostPerNew: number;
  /** Non-torch memory (CUDA context, NCCL, cuBLAS workspaces). */
  nonTorchBase: number;
  nonTorchPerComm: number;
  /** Allocator fragmentation factor on the profiled activation peak. */
  actRho: number;
  /** CUDA graph pool bytes per captured size per layer. */
  cudagraphPerSizePerLayer: number;
}

// ───────────────────────────── vLLM flags ─────────────────────────────

export type KvCacheDtype = 'auto' | 'bfloat16' | 'fp8' | 'fp8_e5m2' | 'fp8_ds_mla' | 'nvfp4_ds_mla' | 'nvfp4';

export type All2AllBackend =
  | 'allgather_reducescatter'
  | 'deepep_high_throughput'
  | 'deepep_low_latency'
  | 'pplx'
  | 'naive'
  | 'flashinfer_nvlink_one_sided';

export type QuantOverride = 'auto' | 'fp8' | 'none';

export type SpecMethod = 'mtp' | 'eagle' | 'eagle3' | 'draft_model' | 'ngram';

export interface SpecConfig {
  method: SpecMethod;
  k: number;
  /** Acceptance probability of the first drafted token; position i accepts with a·decay^i. */
  acceptance: number;
  decay: number;
  /** Parameters of the drafter (EAGLE head / draft model); MTP uses the model's own MTP layers. */
  draftParams: number;
}

export interface VllmFlags {
  tp: number;
  pp: number;
  dp: number;
  ep: boolean;
  dcp: number;
  quantization: QuantOverride;
  kvCacheDtype: KvCacheDtype;
  gpuMemoryUtilization: number;
  kvCacheMemoryBytes: number | null;
  maxModelLen: number | null;
  maxNumSeqs: number | null;
  maxNumBatchedTokens: number | null;
  blockSize: number | null;
  enablePrefixCaching: boolean;
  enableChunkedPrefill: boolean;
  performanceMode: 'balanced' | 'interactivity' | 'throughput';
  /** -O level; 0 = eager (no CUDA graphs). */
  optimizationLevel: 0 | 1 | 2 | 3;
  all2allBackend: All2AllBackend;
  numRedundantExperts: number;
  speculative: SpecConfig | null;
}

export type InstanceRole = 'mixed' | 'prefill' | 'decode';

export interface InstanceSpec {
  id: string;
  role: InstanceRole;
  flags: VllmFlags;
  /** First global GPU index this instance occupies (instances take contiguous GPU ranges). */
  gpuStart: number;
}

export interface Deployment {
  instances: InstanceSpec[];
}

export interface Issue {
  level: 'error' | 'warn' | 'info';
  flag?: string;
  msg: string;
}

export interface ResolvedInstance {
  id: string;
  role: InstanceRole;
  flags: VllmFlags;
  gpuStart: number;
  world: number;
  tp: number;
  pp: number;
  dp: number;
  ep: boolean;
  epSize: number;
  moeTp: number;
  dcp: number;
  maxModelLen: number;
  maxNumSeqs: number;
  maxNumBatchedTokens: number;
  blockSize: number;
  attnBackend: string;
  quant: QuantScheme;
  kv: { label: string; bytesPerElem: number; mlaBytesPerToken: number | null };
  cudagraph: { mode: 'FULL_AND_PIECEWISE' | 'PIECEWISE' | 'NONE'; maxSize: number; sizes: number[] };
  why: Record<string, string>;
  issues: Issue[];
}

// ───────────────────────────── placement & memory ─────────────────────────────

export interface WeightBreakdown {
  attn: number;
  dense: number;
  shared: number;
  experts: number;
  router: number;
  norms: number;
  embed: number;
  lmHead: number;
  draft: number;
  total: number;
}

export interface RankShard {
  rank: number; // rank within the instance
  gpu: number; // global GPU index
  dpRank: number;
  ppRank: number;
  tpRank: number;
  layerLo: number;
  layerHi: number; // exclusive
  embed: boolean;
  lmHead: boolean;
  qHeads: number;
  kvHeads: number;
  kvReplicated: boolean;
  /** Local routed experts [lo, hi) for EP; null when experts are TP-sharded (each rank holds 1/shard of all). */
  experts: { lo: number; hi: number; count: number } | null;
  expertShard: number; // 1 with EP; TP×DP without EP
  weights: WeightBreakdown;
}

export type CommKind = 'tp' | 'ep' | 'pp' | 'kvx';

export interface CommGroup {
  kind: CommKind;
  gpus: number[];
  /** Number of distinct scale-up domains spanned. */
  domains: number;
  /** Members per domain (max). */
  perDomain: number;
}

export interface Placement {
  shards: RankShard[];
  tpGroups: CommGroup[];
  epGroups: CommGroup[];
  ppLinks: { from: number; to: number }[];
}

export interface MemoryBreakdown {
  gpu: number;
  total: number;
  requested: number;
  weights: number;
  nonTorch: number;
  persistent: number;
  actPeak: number;
  cudagraph: number;
  kv: number;
  /** KV bytes per token on this rank (all its layers). */
  kvBytesPerToken: number;
  pageBytes: number;
  blocks: number;
  why: Record<string, string>;
}

export interface InstanceMemory {
  perRank: MemoryBreakdown[];
  /** vLLM takes the min across workers. */
  blocks: number;
  blockSize: number;
  kvTokens: number; // per DP replica
  maxConcurrency: number; // at max_model_len, per DP replica
  fits: boolean;
  /** Linear-attention / mamba state slots per request (bytes/rank), part of each page. */
  stateBytesPerSeq: number;
  why: Record<string, string>;
}

// ───────────────────────────── cost ─────────────────────────────

/** Aggregate description of one forward step on one DP rank. */
export interface BatchShape {
  decodeSeqs: number;
  /** Query tokens per decode sequence (1 + speculative k). */
  q: number;
  /** Σ effective context of decode seqs: [full, window_1, …] (matches ModelSpec.windows). */
  decodeCtx: number[];
  prefillTokens: number;
  prefillSeqs: number;
  /** Σ attention (query,key) pairs of prefill chunks: [full, window_1, …]. */
  prefillPairs: number[];
  /** Positions that need logits. */
  sampled: number;
  /** Tokens across all DP ranks entering MoE layers (lockstep). */
  globalTokens: number;
}

export interface StepParts {
  gemm: number;
  attn: number;
  moe: number;
  lmHead: number;
  launch: number;
  tpComm: number;
  epComm: number;
  ppComm: number;
  draft: number;
}

export interface StepCost {
  /** Wall time between consecutive steps of this rank. */
  time: number;
  gpu: number;
  host: number;
  /** Per-PP-stage GPU time (incl. comm). */
  stages: number[];
  parts: StepParts;
  flops: number;
  hbmBytes: number;
  /** Bytes sent per rank per step, split by scale-up (intra) vs scale-out (inter). */
  comm: Record<CommKind, { intra: number; inter: number }>;
  bound: 'memory' | 'compute' | 'comm' | 'host';
}
