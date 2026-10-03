// Resolve user flags into the concrete values vLLM would run with, mirroring
// vllm/engine/arg_utils.py (batch defaults), vllm/config/{parallel,cache,compilation}.py and
// platform block-size overrides. Every derived value records a human-readable "why".

import { onlineFp8 } from '../model/quant.ts';
import { attnWindow } from '../model/normalize.ts';
import type { ClusterSpec, InstanceSpec, Issue, ModelSpec, QuantScheme, ResolvedInstance } from '../types.ts';
import { GiB } from '../units.ts';
import { totalGpus } from '../hardware/clusters.ts';

const isBlackwellDC = (arch: string): boolean => arch === 'sm100' || arch === 'sm103';

export function hasMla(m: ModelSpec): boolean {
  return m.layers.some((l) => l.attn.kind === 'mla');
}

export function hasMoe(m: ModelSpec): boolean {
  return m.layers.some((l) => l.ffn.kind === 'moe');
}

export function firstMoe(m: ModelSpec) {
  for (const l of m.layers) if (l.ffn.kind === 'moe') return l.ffn;
  return null;
}

function gqaHeads(m: ModelSpec): { nQ: number; nKV: number } | null {
  for (const l of m.layers) if (l.attn.kind === 'gqa') return { nQ: l.attn.nQ, nKV: l.attn.nKV };
  return null;
}

/** CUDA graph capture sizes: [1,2,4] + range(8,256,8) + range(256,max+1,16), capped at max. */
export function captureSizes(max: number, interactivity: boolean): number[] {
  const s = new Set<number>([1, 2, 4]);
  for (let x = 8; x < 256; x += 8) s.add(x);
  for (let x = 256; x <= max; x += 16) s.add(x);
  if (interactivity) for (let x = 1; x <= 32; x++) s.add(x);
  return [...s].filter((x) => x <= max).sort((a, b) => a - b);
}

export function resolveInstance(inst: InstanceSpec, model: ModelSpec, cluster: ClusterSpec): ResolvedInstance {
  const f = inst.flags;
  const gpu = cluster.gpu;
  const why: Record<string, string> = {};
  const issues: Issue[] = [];
  const mla = hasMla(model);
  const moe = firstMoe(model);

  const tp = Math.max(1, Math.floor(f.tp));
  const pp = Math.max(1, Math.floor(f.pp));
  const dp = Math.max(1, Math.floor(f.dp));
  const world = tp * pp * dp;
  why['world'] = `DP×PP×TP = ${dp}×${pp}×${tp} = ${world} GPUs`;

  const ep = f.ep && !!moe;
  const epSize = ep ? tp * dp : 1;
  const moeTp = ep ? 1 : tp * dp;
  if (moe) {
    why['moe'] = ep
      ? `--enable-expert-parallel: EP = TP×DP = ${epSize}; each rank holds whole experts (MoE TP = 1)`
      : dp > 1
        ? `No EP: every expert is TP-sharded across TP×DP = ${moeTp} ranks`
        : `No EP: every expert is TP-sharded across the ${tp} TP ranks`;
  }
  if (f.ep && !moe) issues.push({ level: 'info', flag: 'ep', msg: 'Expert parallel has no effect on a dense model.' });

  // Quantization.
  let quant: QuantScheme = model.quant;
  if (f.quantization === 'fp8') {
    if (model.quant.method === 'none') {
      quant = onlineFp8(model.baseDtype);
      why['quant'] = '--quantization fp8: weights quantized to FP8 at load time (per-tensor, dynamic activations)';
    } else {
      issues.push({ level: 'warn', flag: 'quantization', msg: `Checkpoint is already ${model.quant.label}; --quantization fp8 ignored.` });
    }
  }
  if (!why['quant']) why['quant'] = `From checkpoint: ${quant.label}`;
  const usesFp8 = [quant.attn, quant.dense, quant.experts].some((d) => d === 'fp8' || d === 'fp8_block');
  const usesFp4 = [quant.attn, quant.dense, quant.experts].some((d) => d === 'nvfp4' || d === 'mxfp4');
  if (usesFp8 && !gpu.peak.fp8) issues.push({ level: 'info', msg: `${gpu.name} has no FP8 tensor cores: FP8 weights run as W8A16 (Marlin).` });
  if (usesFp4 && !gpu.peak.fp4) issues.push({ level: 'info', msg: `${gpu.name} has no FP4 tensor cores: FP4 weights run as W4A16 (Marlin).` });

  // KV cache dtype.
  let kvDtype = f.kvCacheDtype;
  if (kvDtype === 'auto' && quant.kvHint === 'fp8') {
    kvDtype = 'fp8';
    why['kvDtype'] = 'auto → fp8 (checkpoint ships an FP8 KV-cache scheme)';
  } else if (kvDtype === 'auto' && model.modelType === 'deepseek_v32') {
    kvDtype = 'fp8_ds_mla';
    why['kvDtype'] = 'auto → fp8_ds_mla (DeepSeek-V3.2 default)';
  }
  if ((kvDtype === 'fp8_ds_mla' || kvDtype === 'nvfp4_ds_mla') && !mla) {
    issues.push({ level: 'error', flag: 'kvCacheDtype', msg: `${kvDtype} is only valid for MLA models.` });
    kvDtype = 'fp8';
  }
  const kvBytes = kvDtype === 'auto' || kvDtype === 'bfloat16' ? 2 : kvDtype === 'nvfp4' ? 0.5625 : 1;
  const mlaBytesPerToken = !mla
    ? null
    : kvDtype === 'fp8_ds_mla'
      ? 656
      : kvDtype === 'nvfp4_ds_mla'
        ? 352
        : (() => {
            const a = model.layers.find((l) => l.attn.kind === 'mla')!.attn as { kvLora: number; rope: number };
            return (a.kvLora + a.rope) * kvBytes;
          })();
  if (!why['kvDtype']) why['kvDtype'] = kvDtype === 'auto' ? 'auto → model dtype (BF16, 2 bytes)' : `--kv-cache-dtype ${kvDtype}`;

  // Max model len.
  const maxModelLen = Math.max(16, Math.floor(f.maxModelLen ?? model.maxPos));
  why['maxModelLen'] =
    f.maxModelLen === null ? `Derived from config max_position_embeddings = ${model.maxPos}` : `--max-model-len ${maxModelLen}`;
  if (maxModelLen > model.maxPos) {
    issues.push({ level: 'warn', flag: 'maxModelLen', msg: `max_model_len ${maxModelLen} exceeds the model's ${model.maxPos} positions.` });
  }

  // Batch defaults (API server usage context).
  let mnbt: number;
  let mns: number;
  const memGiB = gpu.memBytes / GiB;
  let tier: string;
  if (memGiB >= 160) {
    mnbt = 16384;
    mns = 1024;
    tier = '≥160 GiB GPU';
  } else if (memGiB >= 70 && gpu.arch !== 'sm80') {
    mnbt = 8192;
    mns = 1024;
    tier = '≥70 GiB non-A100 GPU';
  } else {
    mnbt = 2048;
    mns = 256;
    tier = 'default GPU tier';
  }
  if (f.performanceMode === 'throughput') {
    mnbt *= 2;
    mns *= 2;
    tier += ', ×2 for --performance-mode throughput';
  }
  if (ep && dp > 1 && f.all2allBackend === 'deepep_low_latency') {
    mnbt = 256;
    tier = 'DeepEP low-latency with EP and DP>1 caps the token budget';
  }
  if (f.maxNumSeqs !== null) {
    mns = f.maxNumSeqs;
    why['maxNumSeqs'] = `--max-num-seqs ${mns}`;
  } else why['maxNumSeqs'] = `Default ${mns} (${tier})`;
  if (f.maxNumBatchedTokens !== null) {
    mnbt = f.maxNumBatchedTokens;
    why['maxNumBatchedTokens'] = `--max-num-batched-tokens ${mnbt}`;
  } else why['maxNumBatchedTokens'] = `Default ${mnbt} (${tier})`;
  if (!f.enableChunkedPrefill && mnbt < maxModelLen) {
    mnbt = maxModelLen;
    why['maxNumBatchedTokens'] += `; raised to max_model_len because chunked prefill is off`;
  }
  mnbt = Math.min(mnbt, mns * maxModelLen);
  mns = Math.min(mns, mnbt);

  // Attention backend and block size.
  let attnBackend: string;
  let blockSize: number;
  if (mla) {
    attnBackend = model.modelType === 'deepseek_v32' ? 'FLASHMLA_SPARSE' : isBlackwellDC(gpu.arch) ? 'FLASHINFER_MLA' : 'FLASHMLA';
    blockSize = 64;
  } else if (gpu.vendor === 'amd') {
    attnBackend = 'ROCM_AITER_FA';
    blockSize = 16;
  } else {
    attnBackend = isBlackwellDC(gpu.arch) ? 'FLASHINFER' : 'FLASH_ATTN';
    blockSize = 16;
  }
  why['blockSize'] = `${attnBackend} prefers block_size ${blockSize}`;
  if (f.blockSize !== null) {
    blockSize = f.blockSize;
    why['blockSize'] = `--block-size ${blockSize}`;
  }

  // CUDA graphs.
  const q = 1 + (f.speculative?.k ?? 0);
  const mode = f.optimizationLevel === 0 ? 'NONE' : f.optimizationLevel === 1 ? 'PIECEWISE' : 'FULL_AND_PIECEWISE';
  const cgMax = mode === 'NONE' ? 0 : Math.min(mns * q * 2, isBlackwellDC(gpu.arch) ? 1024 : 512, mnbt);
  const sizes = mode === 'NONE' ? [] : captureSizes(cgMax, f.performanceMode === 'interactivity');
  why['cudagraph'] =
    mode === 'NONE'
      ? '-O0: eager mode, no CUDA graphs'
      : `${mode}, ${sizes.length} capture sizes up to ${cgMax} tokens`;

  // Parallelism validation.
  const heads = gqaHeads(model);
  const firstAttn = model.layers.find((l) => l.attn.kind !== 'linear')?.attn;
  const nQ = firstAttn && firstAttn.kind !== 'linear' ? firstAttn.nQ : 1;
  if (nQ % tp !== 0) issues.push({ level: 'error', flag: 'tp', msg: `num_attention_heads (${nQ}) is not divisible by TP=${tp}.` });
  if (heads && tp > heads.nKV) {
    if (tp % heads.nKV !== 0) issues.push({ level: 'error', flag: 'tp', msg: `TP=${tp} must be a multiple of num_key_value_heads (${heads.nKV}).` });
    else
      issues.push({
        level: 'warn',
        flag: 'tp',
        msg: `TP=${tp} > ${heads.nKV} KV heads: each KV head is replicated on ${tp / heads.nKV} GPUs, multiplying KV memory.`,
      });
  } else if (heads && heads.nKV % tp !== 0) {
    issues.push({ level: 'error', flag: 'tp', msg: `num_key_value_heads (${heads.nKV}) is not divisible by TP=${tp}.` });
  }
  if (mla && tp > 1) {
    issues.push({
      level: 'info',
      flag: 'tp',
      msg: `MLA keeps one latent KV per token, replicated on all ${tp} TP ranks. DP attention (TP=1, DP>1, EP) avoids the duplication.`,
    });
  }
  if (pp > model.layers.length) issues.push({ level: 'error', flag: 'pp', msg: `PP=${pp} exceeds ${model.layers.length} layers.` });
  if (ep && moe) {
    const slots = moe.E + f.numRedundantExperts;
    if (slots % epSize !== 0) {
      issues.push({ level: 'warn', flag: 'ep', msg: `${slots} expert slots don't divide evenly over EP=${epSize}; some ranks hold one more expert.` });
    }
  }
  if (moe && !ep && moe.eInter % moeTp !== 0) {
    issues.push({ level: 'warn', flag: 'tp', msg: `Expert intermediate size ${moe.eInter} is not divisible by MoE TP=${moeTp}.` });
  }
  const dcp = Math.max(1, f.dcp);
  if (dcp > 1) {
    const maxDcp = heads ? Math.max(1, tp / heads.nKV) : tp;
    if (tp % dcp !== 0 || dcp > maxDcp) issues.push({ level: 'error', flag: 'dcp', msg: `DCP must divide TP and be ≤ TP/num_kv_heads (${maxDcp}).` });
    if (model.layers.some((l) => attnWindow(l.attn)?.kind === 'sliding')) {
      issues.push({ level: 'error', flag: 'dcp', msg: 'Decode context parallel does not support sliding-window attention.' });
    }
  }
  if (tp > cluster.domainSize && totalGpus(cluster) > 1) {
    issues.push({
      level: 'warn',
      flag: 'tp',
      msg: `TP=${tp} spans ${Math.ceil(tp / cluster.domainSize)} NVLink domains: every layer's all-reduce crosses ${cluster.scaleOut.kind.split(' /')[0]}.`,
    });
  }
  if (f.gpuMemoryUtilization <= 0 || f.gpuMemoryUtilization > 1) {
    issues.push({ level: 'error', flag: 'gpuMemoryUtilization', msg: 'gpu_memory_utilization must be in (0, 1].' });
  }
  if (f.speculative) {
    if (f.speculative.method === 'mtp' && model.mtp.layers.length === 0) {
      issues.push({ level: 'error', flag: 'speculative', msg: 'This checkpoint has no MTP modules; choose EAGLE, a draft model or n-gram.' });
    }
    if (pp > 1) issues.push({ level: 'warn', flag: 'speculative', msg: 'Speculative decoding with PP>1 is not supported by Model Runner V2.' });
  }

  return {
    id: inst.id,
    role: inst.role,
    flags: f,
    gpuStart: inst.gpuStart,
    world,
    tp,
    pp,
    dp,
    ep,
    epSize,
    moeTp,
    dcp,
    maxModelLen,
    maxNumSeqs: mns,
    maxNumBatchedTokens: mnbt,
    blockSize,
    attnBackend,
    quant,
    kv: { label: kvDtype, bytesPerElem: kvBytes, mlaBytesPerToken },
    cudagraph: { mode, maxSize: cgMax, sizes },
    why,
    issues,
  };
}
