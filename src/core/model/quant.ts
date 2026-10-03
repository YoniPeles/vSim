import type { QuantScheme, WeightDtype } from '../types.ts';

/** Bytes per weight including scale/zero-point overhead: bits/8 + (scale + zero bytes)/group. */
export const BYTES_PER_PARAM: Record<WeightDtype, number> = {
  f32: 4,
  bf16: 2,
  fp8: 1,
  fp8_block: 1 + 4 / (128 * 128),
  int8: 1,
  mxfp4: 0.5 + 1 / 32,
  nvfp4: 0.5 + 1 / 16,
  int4_g128: 0.5 + (2 + 0.5) / 128,
};

export const DTYPE_LABEL: Record<WeightDtype, string> = {
  f32: 'FP32',
  bf16: 'BF16',
  fp8: 'FP8',
  fp8_block: 'FP8 (block)',
  int8: 'INT8',
  mxfp4: 'MXFP4',
  nvfp4: 'NVFP4',
  int4_g128: 'INT4 g128',
};

type Json = Record<string, unknown>;

const isObj = (x: unknown): x is Json => typeof x === 'object' && x !== null && !Array.isArray(x);

export function unquantized(base: WeightDtype = 'bf16'): QuantScheme {
  return {
    method: 'none',
    label: DTYPE_LABEL[base],
    attn: base,
    dense: base,
    shared: base,
    experts: base,
    router: base,
    embed: base,
    lmHead: base,
  };
}

function uniformLinear(method: string, label: string, dt: WeightDtype, base: WeightDtype): QuantScheme {
  return { ...unquantized(base), method, label, attn: dt, dense: dt, shared: dt, experts: dt };
}

/** Whether an exclusion list (globs / regexes / module names) keeps attention in the base dtype. */
function excludesAttention(list: unknown): boolean {
  if (!Array.isArray(list)) return false;
  return list.some((m) => typeof m === 'string' && /self_attn|\.attn|attention/.test(m));
}

/** Online quantization requested with `--quantization fp8` on an unquantized checkpoint. */
export function onlineFp8(base: WeightDtype): QuantScheme {
  return uniformLinear('fp8-online', 'FP8 (online, per-tensor)', 'fp8', base);
}

/**
 * Derive the per-component weight dtypes from config.json's quantization_config, or from a
 * ModelOpt hf_quant_config.json.
 */
export function parseQuant(config: Json, hfQuantConfig: Json | undefined, base: WeightDtype): QuantScheme {
  const qc = isObj(config['quantization_config']) ? config['quantization_config'] : undefined;

  const mo = hfQuantConfig && isObj(hfQuantConfig['quantization']) ? hfQuantConfig['quantization'] : undefined;
  if (mo || (qc && qc['quant_method'] === 'modelopt')) {
    const src = (mo ?? qc) as Json;
    const algo = String(src['quant_algo'] ?? '').toUpperCase();
    const dt: WeightDtype = algo.includes('FP4') ? 'nvfp4' : 'fp8';
    const s = uniformLinear('modelopt', `ModelOpt ${algo || 'NVFP4'}`, dt, base);
    if (excludesAttention(src['exclude_modules'])) s.attn = base;
    if (String(src['kv_cache_quant_algo'] ?? '').toUpperCase() === 'FP8') s.kvHint = 'fp8';
    return s;
  }

  if (!qc) return unquantized(base);
  const method = String(qc['quant_method'] ?? '');

  if (method === 'fp8') {
    const blk = Array.isArray(qc['weight_block_size']);
    const s = uniformLinear('fp8', blk ? 'FP8 block-wise (128×128)' : 'FP8', blk ? 'fp8_block' : 'fp8', base);
    if (excludesAttention(qc['modules_to_not_convert'])) s.attn = base;
    return s;
  }

  if (method === 'mxfp4') {
    // gpt-oss: only routed experts are MXFP4; attention, router, embeddings, lm_head stay BF16.
    const s = { ...unquantized(base), method, label: 'MXFP4 (experts)', experts: 'mxfp4' as const };
    return s;
  }

  if (method === 'awq' || method === 'gptq') {
    const bits = Number(qc['bits'] ?? 4);
    const label = `${method.toUpperCase()} INT${bits} g${qc['group_size'] ?? 128}`;
    return uniformLinear(method, label, bits === 8 ? 'int8' : 'int4_g128', base);
  }

  if (method === 'compressed-tensors') {
    const groups = isObj(qc['config_groups']) ? Object.values(qc['config_groups']) : [];
    const g = groups.find(isObj);
    const w = g && isObj(g['weights']) ? g['weights'] : undefined;
    const fmt = String(qc['format'] ?? '');
    let dt: WeightDtype = 'fp8';
    let label = 'compressed-tensors';
    if (fmt.includes('mxfp4')) {
      dt = 'mxfp4';
      label = 'compressed-tensors MXFP4';
    } else if (fmt.includes('nvfp4') || (w?.['num_bits'] === 4 && w?.['type'] === 'float')) {
      dt = 'nvfp4';
      label = 'compressed-tensors NVFP4';
    } else if (w?.['num_bits'] === 4) {
      dt = 'int4_g128';
      label = `compressed-tensors W4A16 g${w?.['group_size'] ?? 128}`;
    } else if (w?.['num_bits'] === 8 && w?.['type'] === 'int') {
      dt = 'int8';
      label = 'compressed-tensors W8A8 INT8';
    } else {
      dt = w?.['strategy'] === 'block' ? 'fp8_block' : 'fp8';
      label = `compressed-tensors FP8 (${String(w?.['strategy'] ?? 'tensor')})`;
    }
    const s = uniformLinear(method, label, dt, base);
    if (excludesAttention(qc['ignore'])) s.attn = base;
    return s;
  }

  return { ...unquantized(base), method, label: `${method} (unrecognized; treated as BF16)` };
}
