// KV-cache budget per rank, in vLLM's order (vllm/v1/worker/gpu_worker.py::determine_available_memory):
//   requested = total × gpu_memory_utilization
//   KV = requested − (weights + non-torch + persistent buffers) − transient activation peak − CUDA graphs

import { attnWindow } from '../model/normalize.ts';
import type {
  Calibration,
  ClusterSpec,
  InstanceMemory,
  LayerSpec,
  MemoryBreakdown,
  ModelSpec,
  Placement,
  RankShard,
  ResolvedInstance,
} from '../types.ts';
import { GiB, MiB, fmtBytes, fmtTokens } from '../units.ts';
import { kvHeadsPerRank } from './placement.ts';

/** KV bytes per token for one layer on one rank (0 for linear-attention layers). */
export function kvBytesPerTokenLayer(l: LayerSpec, inst: ResolvedInstance): number {
  const a = l.attn;
  if (a.kind === 'linear') return 0;
  if (a.kind === 'mla') {
    // One latent per token, replicated on every TP rank; DSA adds the indexer key cache (fp8 + scale).
    const idx = a.dsa ? a.dsa.dIdx + 4 : 0;
    return (inst.kv.mlaBytesPerToken! + idx) / inst.dcp;
  }
  return (kvHeadsPerRank(a.nKV, inst.tp) * (a.dQK + a.dV) * inst.kv.bytesPerElem) / inst.dcp;
}

/** Recurrent state bytes per sequence for a linear-attention layer on one rank. */
export function stateBytesPerSeqLayer(l: LayerSpec, inst: ResolvedInstance): number {
  const a = l.attn;
  if (a.kind !== 'linear') return 0;
  const ssm = a.nV * a.dK * a.dV * 2;
  const conv = (a.conv - 1) * (2 * a.nK * a.dK + a.nV * a.dV) * 2;
  return (ssm + conv) / inst.tp;
}

export interface KvProfile {
  /** Per-token bytes of layers whose KV grows with the full context. */
  fullPerToken: number;
  /** Windowed layers: bytes/token and window size. */
  windowed: { perToken: number; size: number; kind: 'sliding' | 'chunked' | 'topk' }[];
  /** Fixed recurrent state per sequence. */
  statePerSeq: number;
}

export function kvProfile(model: ModelSpec, inst: ResolvedInstance, lo: number, hi: number): KvProfile {
  const prof: KvProfile = { fullPerToken: 0, windowed: [], statePerSeq: 0 };
  for (let i = lo; i < hi; i++) {
    const l = model.layers[i]!;
    const b = kvBytesPerTokenLayer(l, inst);
    prof.statePerSeq += stateBytesPerSeqLayer(l, inst);
    if (b === 0) continue;
    const w = attnWindow(l.attn);
    // DSA still caches every token (it only reads top-k), so it counts as full here.
    if (!w || w.kind === 'topk') {
      prof.fullPerToken += b;
      continue;
    }
    const ex = prof.windowed.find((x) => x.size === w.size && x.kind === w.kind);
    if (ex) ex.perToken += b;
    else prof.windowed.push({ perToken: b, size: w.size, kind: w.kind });
  }
  return prof;
}

/** KV bytes one request occupies on a rank at context length L (hybrid KV-cache manager semantics). */
export function bytesPerRequest(p: KvProfile, L: number, blockSize: number): number {
  const blocks = (n: number) => Math.ceil(n / blockSize) * blockSize;
  let b = p.fullPerToken * blocks(L) + p.statePerSeq;
  for (const w of p.windowed) b += w.perToken * blocks(Math.min(L, w.size + blockSize));
  return b;
}

function activationPeak(model: ModelSpec, inst: ResolvedInstance, shard: RankShard, calib: Calibration): number {
  const T = inst.maxNumBatchedTokens;
  const h = model.hidden;
  let peak = 0;
  for (let i = shard.layerLo; i < shard.layerHi; i++) {
    const l = model.layers[i]!;
    const a = l.attn;
    const qkv =
      a.kind === 'gqa'
        ? ((a.nQ + 2 * a.nKV) * a.dQK) / inst.tp
        : a.kind === 'mla'
          ? (a.nQ * (a.nope + a.rope)) / inst.tp + a.kvLora + a.rope + a.qLora
          : (2 * a.nK * a.dK + 2 * a.nV * a.dV) / inst.tp;
    let layerPeak = T * 2 * (3 * h + qkv);
    if (l.ffn.kind === 'dense') {
      layerPeak = Math.max(layerPeak, T * 2 * (2 * h + (3 * l.ffn.inter) / inst.tp));
    } else {
      const f = l.ffn;
      // MoE input is gathered over DP ranks (lockstep), fused-MoE works in chunks.
      const Tg = T * (inst.dp > 1 ? inst.dp : 1);
      const routed = Math.min(Tg, 32768) * f.topK * (inst.ep ? 1 / inst.epSize : 1);
      const inter = inst.ep ? f.eInter : f.eInter / inst.moeTp;
      const moe = Tg * h * 2 * 2 + routed * (3 * inter + h) * 2 + Tg * f.E * 4;
      layerPeak = Math.max(layerPeak, moe);
    }
    peak = Math.max(peak, layerPeak);
  }
  // Sampler on the last stage: gathered fp32 logits + probabilities for max_num_seqs rows.
  if (shard.lmHead) peak = Math.max(peak, inst.maxNumSeqs * model.vocab * 4 * 2);
  return peak * calib.actRho;
}

function persistentBuffers(model: ModelSpec, inst: ResolvedInstance): number {
  let b = 0;
  if (inst.tp > 1) b += 2 * 64 * MiB; // custom all-reduce IPC buffers
  if (inst.ep && inst.dp > 1) {
    const be = inst.flags.all2allBackend;
    if (be === 'deepep_high_throughput') b += 1.5 * GiB;
    else if (be === 'deepep_low_latency') b += Math.min(4 * GiB, 2 * 128 * inst.epSize * model.hidden * 2 * 2);
    else if (be === 'pplx' || be === 'flashinfer_nvlink_one_sided') b += 1 * GiB;
  }
  // Model Runner V2 persistent request state + sampler buffers.
  b += inst.maxNumSeqs * (inst.maxModelLen * 4 / 64 + 4096) + 64 * MiB;
  return b;
}

export function rankMemory(
  model: ModelSpec,
  inst: ResolvedInstance,
  shard: RankShard,
  cluster: ClusterSpec,
  calib: Calibration,
): MemoryBreakdown {
  const total = cluster.gpu.memBytes;
  const requested = total * inst.flags.gpuMemoryUtilization;
  const comms = [inst.tp > 1, inst.pp > 1, inst.dp > 1, inst.world > 1].filter(Boolean).length;
  const nonTorch = calib.nonTorchBase + calib.nonTorchPerComm * comms;
  const persistent = persistentBuffers(model, inst);
  const actPeak = activationPeak(model, inst, shard, calib);
  const layers = shard.layerHi - shard.layerLo;
  const cgFactor = inst.cudagraph.mode === 'FULL_AND_PIECEWISE' ? 1.3 : 1;
  const cudagraph = inst.cudagraph.sizes.length * layers * calib.cudagraphPerSizePerLayer * cgFactor;
  const weights = shard.weights.total;
  let kv = requested - weights - nonTorch - persistent - actPeak - cudagraph;
  const why: Record<string, string> = {
    requested: `${fmtBytes(total)} × gpu_memory_utilization ${inst.flags.gpuMemoryUtilization}`,
    weights: `Layers ${shard.layerLo}–${shard.layerHi - 1}${shard.embed ? ' + embeddings' : ''}${shard.lmHead ? ' + lm_head' : ''}`,
    nonTorch: `CUDA context, cuBLAS workspaces, ${comms} communicator(s)`,
    actPeak: `Profile run: dummy forward of ${inst.maxNumBatchedTokens} tokens (attention skipped) + sampler`,
    cudagraph: `${inst.cudagraph.sizes.length} capture sizes × ${layers} layers (${inst.cudagraph.mode})`,
  };
  if (inst.flags.kvCacheMemoryBytes !== null) {
    kv = inst.flags.kvCacheMemoryBytes;
    why['kv'] = '--kv-cache-memory-bytes overrides the profiler';
  } else {
    why['kv'] = 'requested − weights − non-torch − persistent − activation peak − CUDA graphs';
  }
  const prof = kvProfile(model, inst, shard.layerLo, shard.layerHi);
  const perTok = prof.fullPerToken || prof.windowed.reduce((s, w) => s + w.perToken, 0);
  const pageBytes = perTok * inst.blockSize;
  const blocks = pageBytes > 0 ? Math.max(0, Math.floor(kv / pageBytes)) : 0;
  return {
    gpu: shard.gpu,
    total,
    requested,
    weights,
    nonTorch,
    persistent,
    actPeak,
    cudagraph,
    kv: Math.max(0, kv),
    kvBytesPerToken: perTok,
    pageBytes,
    blocks,
    why,
  };
}

export function instanceMemory(
  model: ModelSpec,
  inst: ResolvedInstance,
  placement: Placement,
  cluster: ClusterSpec,
  calib: Calibration,
): InstanceMemory {
  const perRank = placement.shards.map((s) => rankMemory(model, inst, s, cluster, calib));
  // Requests occupy every PP stage; the binding stage is the one with the least room per request.
  let worst = 0;
  let worstReq = Infinity;
  let worstKv = 0;
  for (let i = 0; i < perRank.length; i++) {
    const s = placement.shards[i]!;
    const prof = kvProfile(model, inst, s.layerLo, s.layerHi);
    const per = bytesPerRequest(prof, inst.maxModelLen, inst.blockSize);
    const conc = per > 0 ? perRank[i]!.kv / per : Infinity;
    if (conc < worstReq) {
      worstReq = conc;
      worst = i;
      worstKv = perRank[i]!.kv;
    }
  }
  const binding = placement.shards[worst]!;
  const prof = kvProfile(model, inst, binding.layerLo, binding.layerHi);
  const blocks = Math.min(...perRank.map((m) => m.blocks));
  const fullPerTok = prof.fullPerToken || prof.windowed.reduce((s, w) => s + w.perToken, 0);
  const kvTokens = fullPerTok > 0 ? Math.floor(worstKv / fullPerTok / inst.blockSize) * inst.blockSize : Infinity;
  const maxConcurrency = Number.isFinite(worstReq) ? worstReq : 0;
  const why: Record<string, string> = {
    kvTokens: `${fmtBytes(worstKv)} free ÷ ${fmtBytes(fullPerTok)} per token on GPU ${binding.gpu}`,
    maxConcurrency: `${fmtBytes(worstKv)} ÷ ${fmtBytes(bytesPerRequest(prof, inst.maxModelLen, inst.blockSize))} per ${fmtTokens(inst.maxModelLen)}-token request`,
  };
  return {
    perRank,
    blocks,
    blockSize: inst.blockSize,
    kvTokens,
    maxConcurrency,
    fits: maxConcurrency >= 1,
    stateBytesPerSeq: prof.statePerSeq,
    why,
  };
}

/** Largest max_model_len (multiple of the block size) for which one request fits in KV. */
export function suggestMaxModelLen(model: ModelSpec, inst: ResolvedInstance, placement: Placement, mem: InstanceMemory): number {
  let best = Infinity;
  placement.shards.forEach((s, i) => {
    const prof = kvProfile(model, inst, s.layerLo, s.layerHi);
    const kv = mem.perRank[i]!.kv;
    let lo = 0;
    let hi = inst.maxModelLen;
    while (hi - lo > inst.blockSize) {
      const mid = Math.floor((lo + hi) / 2);
      if (bytesPerRequest(prof, mid, inst.blockSize) <= kv) lo = mid;
      else hi = mid;
    }
    best = Math.min(best, Math.floor(lo / inst.blockSize) * inst.blockSize);
  });
  return Number.isFinite(best) ? best : inst.maxModelLen;
}

/** Logical (unreplicated) KV bytes per token across all layers: what a P/D transfer must move. */
export function logicalKvBytesPerToken(model: ModelSpec, inst: ResolvedInstance): number {
  const one = { ...inst, tp: 1, dcp: 1 };
  return model.layers.reduce((s, l) => s + kvBytesPerTokenLayer(l, one), 0);
}
