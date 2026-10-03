// config.json → canonical ModelSpec.
// Handles nested text_config, field-name aliases, per-layer pattern arrays, MLA/DSA, sliding/chunked
// windows, linear-attention hybrids and MoE variants. Unknown fields degrade to a llama-like reading
// with a warning rather than failing.

import type {
  AttnSpec,
  CtxWindow,
  FfnSpec,
  LayerGroup,
  LayerParams,
  LayerSpec,
  ModelSpec,
  WeightDtype,
} from '../types.ts';
import { parseQuant } from './quant.ts';

type Json = Record<string, unknown>;

const isObj = (x: unknown): x is Json => typeof x === 'object' && x !== null && !Array.isArray(x);

function num(c: Json, ...keys: string[]): number | undefined {
  for (const k of keys) {
    const v = c[k];
    if (typeof v === 'number' && Number.isFinite(v)) return v;
  }
  return undefined;
}

function req(c: Json, ...keys: string[]): number {
  const v = num(c, ...keys);
  if (v === undefined) throw new Error(`config.json is missing ${keys.join(' / ')}`);
  return v;
}

function baseDtype(c: Json): WeightDtype {
  const d = String(c['torch_dtype'] ?? c['dtype'] ?? 'bfloat16');
  // vLLM's dtype=auto loads fp32 checkpoints as bf16; fp16 has the same footprint as bf16.
  return d.includes('float') || d.includes('bf') ? 'bf16' : 'bf16';
}

export interface NormalizeInput {
  repo: string;
  revision?: string;
  label?: string;
  config: Json;
  hfQuantConfig?: Json;
  safetensors?: { total: number; parameters: Record<string, number> };
  source: 'preset' | 'hub';
}

// ───────────────────────────── attention ─────────────────────────────

type LayerKind = 'full' | 'sliding' | 'chunked' | 'linear';

function layerKinds(c: Json, L: number, modelType: string): LayerKind[] {
  const out: LayerKind[] = new Array<LayerKind>(L).fill('full');
  const lt = c['layer_types'];
  if (Array.isArray(lt) && lt.length >= L) {
    for (let i = 0; i < L; i++) {
      const t = String(lt[i]);
      out[i] = t.includes('linear') ? 'linear' : t.includes('sliding') ? 'sliding' : t.includes('chunk') ? 'chunked' : 'full';
    }
    return out;
  }
  const fai = num(c, 'full_attention_interval');
  if (fai && (modelType.startsWith('qwen3_next') || num(c, 'linear_num_value_heads'))) {
    for (let i = 0; i < L; i++) out[i] = (i + 1) % fai === 0 ? 'full' : 'linear';
    return out;
  }
  const swp = num(c, 'sliding_window_pattern');
  const sw = num(c, 'sliding_window');
  if (swp && sw) {
    for (let i = 0; i < L; i++) out[i] = (i + 1) % swp === 0 ? 'full' : 'sliding';
    return out;
  }
  const chunk = num(c, 'attention_chunk_size');
  if (chunk) {
    const nope = c['no_rope_layers'];
    for (let i = 0; i < L; i++) {
      // Llama 4: no_rope_layers[i] == 0 → NoPE global layer; else chunked local attention.
      const isNope = Array.isArray(nope) ? nope[i] === 0 : (i + 1) % 4 === 0;
      out[i] = isNope ? 'full' : 'chunked';
    }
    return out;
  }
  const useSw = c['use_sliding_window'];
  if (sw && useSw !== false && modelType !== 'qwen3' && modelType !== 'qwen3_moe' && modelType !== 'qwen2') {
    // Mistral-style: every layer sliding when sliding_window is set.
    out.fill('sliding');
  }
  return out;
}

function gqaParams(c: Json, modelType: string, h: number, nQ: number, nKV: number, d: number, gatedQ: boolean): number {
  const q = h * nQ * d * (gatedQ ? 2 : 1);
  const kv = 2 * h * nKV * d;
  const o = nQ * d * h;
  let p = q + kv + o;
  const bias = c['attention_bias'] === true || modelType === 'qwen2' || modelType === 'qwen2_moe';
  if (bias) p += nQ * d + 2 * nKV * d;
  if (c['attention_bias'] === true && modelType !== 'qwen2') p += h; // o_proj bias
  if (modelType === 'gpt_oss') p += nQ; // attention sinks
  if (/^(qwen3|gemma3|olmo|glm4)/.test(modelType)) p += 2 * d; // q_norm / k_norm
  return p;
}

function mlaParams(h: number, a: Extract<AttnSpec, { kind: 'mla' }>): number {
  const qk = a.nope + a.rope;
  const q = a.qLora > 0 ? h * a.qLora + a.qLora + a.qLora * a.nQ * qk : h * a.nQ * qk;
  const kvA = h * (a.kvLora + a.rope) + a.kvLora;
  const kvB = a.kvLora * a.nQ * (a.nope + a.dV);
  const o = a.nQ * a.dV * h;
  let p = q + kvA + kvB + o;
  if (a.dsa) p += (a.qLora || h) * a.dsa.nIdx * a.dsa.dIdx + h * a.dsa.dIdx + 2 * a.dsa.dIdx + h * a.dsa.nIdx;
  return p;
}

function linearParams(h: number, a: Extract<AttnSpec, { kind: 'linear' }>): number {
  const keyDim = a.nK * a.dK;
  const valDim = a.nV * a.dV;
  const inProj = h * (2 * keyDim + 2 * valDim) + h * 2 * a.nV;
  const conv = (2 * keyDim + valDim) * a.conv;
  return inProj + conv + 2 * a.nV + a.dV + valDim * h;
}

// ───────────────────────────── FFN ─────────────────────────────

interface MoeInfo {
  E: number;
  topK: number;
  eInter: number;
  nShared: number;
  sInter: number;
  isMoe: (i: number) => boolean;
  denseInter: number;
  expertBias: boolean;
  routerBias: boolean;
  sharedGate: boolean;
}

function moeInfo(c: Json, modelType: string, L: number): MoeInfo | null {
  const E = num(c, 'n_routed_experts', 'num_local_experts', 'num_experts');
  if (!E || E <= 1) return null;
  const topK = req(c, 'num_experts_per_tok', 'num_experts_per_token', 'experts_per_token', 'moe_topk');
  const inter = num(c, 'intermediate_size') ?? 0;
  const eInter = num(c, 'moe_intermediate_size', 'routed_expert_hidden_size') ?? inter;
  let denseInter = num(c, 'intermediate_size_mlp') ?? inter;
  let nShared = num(c, 'n_shared_experts', 'num_shared_experts') ?? 0;
  let sInter = eInter;
  const sharedSize = num(c, 'shared_expert_intermediate_size');
  if (sharedSize) {
    nShared = Math.max(nShared, 1);
    sInter = sharedSize;
  }
  let isMoe: (i: number) => boolean = () => true;

  const fkd = num(c, 'first_k_dense_replace');
  const mlf = c['moe_layer_freq'];
  const mlt = c['mlp_layer_types'];
  const mlpOnly = Array.isArray(c['mlp_only_layers']) ? (c['mlp_only_layers'] as number[]) : [];
  const step = num(c, 'decoder_sparse_step');
  const interleave = num(c, 'interleave_moe_layer_step');
  const moeLayers = Array.isArray(c['moe_layers']) ? (c['moe_layers'] as number[]) : null;

  if (Array.isArray(mlt) && mlt.length >= L) {
    isMoe = (i) => /sparse|moe/.test(String(mlt[i]));
  } else if (Array.isArray(mlf) && mlf.length >= L) {
    isMoe = (i) => Number(mlf[i]) === 1;
  } else if (fkd !== undefined || typeof mlf === 'number') {
    const k = fkd ?? 0;
    const f = typeof mlf === 'number' ? mlf : 1;
    isMoe = (i) => i >= k && i % f === 0;
  } else if (moeLayers) {
    const set = new Set(moeLayers);
    isMoe = (i) => set.has(i);
  } else if (interleave) {
    isMoe = (i) => (i + 1) % interleave === 0;
  } else if (step || mlpOnly.length) {
    const s = step ?? 1;
    isMoe = (i) => !mlpOnly.includes(i) && (i + 1) % s === 0;
  }

  if (modelType === 'llama4' || modelType === 'llama4_text') {
    // Llama 4 MoE layers: routed experts + one shared expert, both of size intermediate_size.
    nShared = 1;
    sInter = eInter;
    denseInter = num(c, 'intermediate_size_mlp') ?? inter;
  }

  return {
    E,
    topK,
    eInter,
    nShared,
    sInter,
    isMoe,
    denseInter,
    expertBias: modelType === 'gpt_oss',
    routerBias: modelType === 'gpt_oss' || c['topk_method'] === 'noaux_tc' || /deepseek_v3|kimi_k2|deepseek_v32/.test(modelType),
    sharedGate: /qwen2_moe|qwen3_next/.test(modelType),
  };
}

// ───────────────────────────── main ─────────────────────────────

function textConfig(config: Json): { tc: Json; nested: boolean } {
  const t = config['text_config'];
  if (isObj(t) && (t['hidden_size'] !== undefined || t['num_hidden_layers'] !== undefined)) {
    // Fields absent in text_config may live at the top level (e.g. tie_word_embeddings).
    return { tc: { ...config, ...t }, nested: true };
  }
  return { tc: config, nested: false };
}

function groupLabel(l: LayerSpec): string {
  const a = l.attn;
  const attn =
    a.kind === 'mla'
      ? a.dsa
        ? 'MLA+DSA'
        : 'MLA'
      : a.kind === 'linear'
        ? `linear (${a.variant.toUpperCase()})`
        : a.scope === 'full'
          ? a.nKV === a.nQ
            ? 'MHA'
            : 'GQA'
          : `${a.scope} ${a.window}`;
  const ffn = l.ffn.kind === 'moe' ? `MoE ${l.ffn.E}e top-${l.ffn.topK}` : 'dense MLP';
  return `${attn} · ${ffn}`;
}

export function attnWindow(a: AttnSpec): CtxWindow | null {
  if (a.kind === 'gqa' && a.scope !== 'full') return { kind: a.scope === 'sliding' ? 'sliding' : 'chunked', size: a.window };
  if (a.kind === 'mla' && a.dsa) return { kind: 'topk', size: a.dsa.topk };
  return null;
}

/** Window id for an attention spec: 0 = full context, i+1 = model.windows[i]; -1 = no KV (linear). */
export function windowId(a: AttnSpec, windows: CtxWindow[]): number {
  if (a.kind === 'linear') return -1;
  const w = attnWindow(a);
  if (!w) return 0;
  return windows.findIndex((x) => x.kind === w.kind && x.size === w.size) + 1;
}

export function normalizeConfig(input: NormalizeInput): ModelSpec {
  const { config } = input;
  const { tc, nested } = textConfig(config);
  const warnings: string[] = [];
  const modelType = String(tc['model_type'] ?? config['model_type'] ?? 'unknown');
  const architectures = (Array.isArray(config['architectures']) ? config['architectures'] : []).map(String);

  const h = req(tc, 'hidden_size', 'd_model');
  const L = req(tc, 'num_hidden_layers', 'n_layers', 'num_layers');
  const nQ = req(tc, 'num_attention_heads', 'n_heads');
  const nKV = num(tc, 'num_key_value_heads', 'num_kv_heads') ?? nQ;
  const d = num(tc, 'head_dim') ?? Math.floor(h / nQ);
  const vocab = req(tc, 'vocab_size');
  const maxPos = num(tc, 'max_position_embeddings', 'max_seq_len', 'seq_length') ?? 4096;
  const tiedRaw = tc['tie_word_embeddings'];
  const tied = typeof tiedRaw === 'boolean' ? tiedRaw : modelType.startsWith('gemma');
  const base = baseDtype(tc);
  const quant = parseQuant(config, input.hfQuantConfig, base);

  if (nested && (config['vision_config'] || config['vision_tower'])) {
    warnings.push('Multimodal checkpoint: vision encoder weights are counted, but its compute is not modeled.');
  }

  const kinds = layerKinds(tc, L, modelType);
  const sw = num(tc, 'sliding_window') ?? 0;
  const chunk = num(tc, 'attention_chunk_size') ?? 0;
  const kvLora = num(tc, 'kv_lora_rank');
  const dsaTopk = num(tc, 'index_topk');
  const moe = moeInfo(tc, modelType, L);
  const denseInter = moe?.denseInter ?? req(tc, 'intermediate_size', 'ffn_hidden_size');
  const normsPerLayer = (modelType.startsWith('gemma') ? 4 : 2) * h;
  const gatedQ = tc['attn_output_gate'] === true || modelType === 'qwen3_next';

  if (num(tc, 'compress_ratios') !== undefined || Array.isArray(tc['compress_ratios'])) {
    warnings.push('Compressed-KV attention (DeepSeek-V4 style) is approximated as MLA without compression.');
  }
  if (isObj(tc['linear_attn_config'])) {
    warnings.push('KDA linear attention is approximated with the GDN state model.');
  }

  const layers: LayerSpec[] = [];
  for (let i = 0; i < L; i++) {
    let attn: AttnSpec;
    const k = kinds[i];
    if (k === 'linear') {
      attn = {
        kind: 'linear',
        variant: 'gdn',
        nK: num(tc, 'linear_num_key_heads') ?? nKV,
        nV: num(tc, 'linear_num_value_heads') ?? nQ,
        dK: num(tc, 'linear_key_head_dim') ?? d,
        dV: num(tc, 'linear_value_head_dim') ?? d,
        conv: num(tc, 'linear_conv_kernel_dim') ?? 4,
      };
    } else if (kvLora) {
      attn = {
        kind: 'mla',
        nQ,
        qLora: num(tc, 'q_lora_rank') ?? 0,
        kvLora,
        rope: req(tc, 'qk_rope_head_dim'),
        nope: req(tc, 'qk_nope_head_dim'),
        dV: req(tc, 'v_head_dim'),
        ...(dsaTopk
          ? { dsa: { nIdx: num(tc, 'index_n_heads') ?? 64, dIdx: num(tc, 'index_head_dim') ?? 128, topk: dsaTopk } }
          : {}),
      };
    } else {
      attn = {
        kind: 'gqa',
        scope: k === 'sliding' ? 'sliding' : k === 'chunked' ? 'chunked' : 'full',
        window: k === 'sliding' ? sw : k === 'chunked' ? chunk : 0,
        nQ,
        nKV,
        dQK: d,
        dV: d,
      };
    }

    const isMoe = moe ? moe.isMoe(i) : false;
    let ffn: FfnSpec;
    const p: LayerParams = { attn: 0, norms: normsPerLayer, dense: 0, shared: 0, expertEach: 0, router: 0 };
    if (attn.kind === 'mla') p.attn = mlaParams(h, attn);
    else if (attn.kind === 'linear') p.attn = linearParams(h, attn);
    else p.attn = gqaParams(tc, modelType, h, nQ, nKV, d, gatedQ);

    if (moe && isMoe) {
      ffn = { kind: 'moe', E: moe.E, topK: moe.topK, eInter: moe.eInter, nShared: moe.nShared, sInter: moe.sInter };
      p.expertEach = 3 * h * moe.eInter + (moe.expertBias ? 2 * moe.eInter + h : 0);
      p.shared = moe.nShared * 3 * h * moe.sInter + (moe.sharedGate ? h : 0);
      p.router = h * moe.E + (moe.routerBias ? moe.E : 0);
    } else {
      ffn = { kind: 'dense', inter: denseInter };
      p.dense = 3 * h * denseInter;
    }
    layers.push({ index: i, attn, ffn, p });
  }

  // MTP modules (DeepSeek num_nextn_predict_layers): one MoE decoder layer + eh_proj + norms each.
  // Qwen3-Next ships one MTP module without declaring it in config.json (vLLM's qwen3_next_mtp).
  const nMtp = num(tc, 'num_nextn_predict_layers', 'mtp_num_hidden_layers') ?? (modelType === 'qwen3_next' ? 1 : 0);
  const mtpLayers: LayerSpec[] = [];
  const lastMoe = [...layers].reverse().find((l) => l.ffn.kind === 'moe') ?? layers[L - 1];
  if (lastMoe) for (let j = 0; j < nMtp; j++) mtpLayers.push({ ...lastMoe, index: L + j });
  const mtpExtra = nMtp * (2 * h * h + 3 * h);

  // Groups of identical layers.
  const groupMap = new Map<string, LayerGroup>();
  for (const l of layers) {
    const key = JSON.stringify([l.attn, l.ffn, l.p]);
    let g = groupMap.get(key);
    if (!g) {
      g = { key, label: groupLabel(l), indices: [], layer: l };
      groupMap.set(key, g);
    }
    g.indices.push(l.index);
  }

  const windows: CtxWindow[] = [];
  for (const l of layers) {
    const w = attnWindow(l.attn);
    if (w && !windows.some((x) => x.kind === w.kind && x.size === w.size)) windows.push(w);
  }

  const embed = vocab * h;
  const lmHead = tied ? 0 : vocab * h;
  let total = embed + lmHead + h;
  let active = embed + lmHead + h;
  for (const l of layers) {
    const common = l.p.attn + l.p.norms + l.p.dense + l.p.shared + l.p.router;
    if (l.ffn.kind === 'moe') {
      total += common + l.ffn.E * l.p.expertEach;
      active += common + l.ffn.topK * l.p.expertEach;
    } else {
      total += common;
      active += common;
    }
  }
  const mtpParams = mtpLayers.reduce(
    (s, l) => s + l.p.attn + l.p.norms + l.p.dense + l.p.shared + l.p.router + (l.ffn.kind === 'moe' ? l.ffn.E : 0) * l.p.expertEach,
    0,
  ) + mtpExtra;

  let vision = 0;
  if (nested && input.safetensors) {
    // Vision tower size is not derivable from text fields; use the checkpoint total when available.
    const diff = input.safetensors.total - total;
    if (diff > 0 && diff < 0.2 * total && quant.method === 'none') vision = diff;
  }

  if (input.safetensors && quant.method === 'none' && !nested) {
    const rel = Math.abs(input.safetensors.total - total - mtpParams) / input.safetensors.total;
    if (rel > 0.02) {
      warnings.push(
        `Derived parameter count differs from the checkpoint's by ${(rel * 100).toFixed(1)}%; some layers may be modeled approximately.`,
      );
    }
  }

  return {
    repo: input.repo,
    revision: input.revision ?? 'main',
    label: input.label ?? input.repo.split('/').pop() ?? input.repo,
    modelType,
    architectures,
    hidden: h,
    vocab,
    maxPos,
    tied,
    baseDtype: base,
    layers,
    groups: [...groupMap.values()],
    mtp: { layers: mtpLayers, extraParams: mtpExtra },
    quant,
    windows,
    params: { total, active, embed, lmHead, mtp: mtpParams, vision },
    warnings,
    source: input.source,
    ...(input.safetensors ? { hfSafetensors: input.safetensors } : {}),
  };
}
