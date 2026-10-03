import { describe, expect, it } from 'vitest';
import { PRESETS, presetModel, snapshotToModel } from './presets.ts';
import { BYTES_PER_PARAM } from './quant.ts';

const B = 1e9;

describe('parameter counts', () => {
  it.each([
    ['meta-llama/Llama-3.1-8B-Instruct', 8.03 * B, 8.03 * B],
    ['meta-llama/Llama-3.3-70B-Instruct', 70.55 * B, 70.55 * B],
    ['mistralai/Mixtral-8x7B-Instruct-v0.1', 46.7 * B, 12.88 * B],
    ['deepseek-ai/DeepSeek-V3', 671.0 * B, 37.5 * B],
    ['Qwen/Qwen3-235B-A22B', 235.1 * B, 22.2 * B],
    ['openai/gpt-oss-120b', 116.8 * B, 5.7 * B],
    ['moonshotai/Kimi-K2-Instruct', 1026.4 * B, 32.9 * B],
  ])('%s', (repo, total, active) => {
    const m = presetModel(repo);
    expect(m.params.total / total).toBeCloseTo(1, 2);
    expect(m.params.active / active).toBeCloseTo(1, 1);
  });

  it('matches checkpoint totals for every unquantized text-only preset (incl. MTP)', () => {
    for (const p of PRESETS) {
      const m = snapshotToModel(p);
      if (!p.safetensors || m.quant.method !== 'none' || p.config['text_config']) continue;
      const derived = m.params.total + m.params.mtp;
      expect(Math.abs(derived - p.safetensors.total) / p.safetensors.total, p.repo).toBeLessThan(0.005);
    }
  });

  it('DeepSeek-V3 with MTP and the MTP embed/head copies equals the checkpoint', () => {
    const m = presetModel('deepseek-ai/DeepSeek-V3');
    const withMtp = m.params.total + m.params.mtp + m.params.embed + m.params.lmHead;
    expect(withMtp / m.hfSafetensors!.total).toBeCloseTo(1, 3);
  });
});

describe('architecture parsing', () => {
  it('DeepSeek-V3: 3 dense + 58 MoE MLA layers', () => {
    const m = presetModel('deepseek-ai/DeepSeek-V3');
    expect(m.layers.filter((l) => l.ffn.kind === 'dense')).toHaveLength(3);
    expect(m.layers.every((l) => l.attn.kind === 'mla')).toBe(true);
    expect(m.mtp.layers).toHaveLength(1);
    expect(m.quant.experts).toBe('fp8_block');
  });

  it('gpt-oss: alternating sliding(128)/full, MXFP4 experts only', () => {
    const m = presetModel('openai/gpt-oss-120b');
    expect(m.windows).toEqual([{ kind: 'sliding', size: 128 }]);
    expect(m.quant.experts).toBe('mxfp4');
    expect(m.quant.attn).toBe('bf16');
  });

  it('Gemma 3: 5 sliding : 1 global', () => {
    const m = presetModel('google/gemma-3-27b-it');
    const full = m.layers.filter((l) => l.attn.kind === 'gqa' && l.attn.scope === 'full').length;
    expect(full).toBe(10);
    expect(m.tied).toBe(true);
  });

  it('Llama 4 Maverick interleaves dense and MoE layers', () => {
    const m = presetModel('meta-llama/Llama-4-Maverick-17B-128E-Instruct');
    expect(m.layers.filter((l) => l.ffn.kind === 'moe')).toHaveLength(24);
  });

  it('Qwen3-Next: 3 linear (GDN) : 1 full attention', () => {
    const m = presetModel('Qwen/Qwen3-Next-80B-A3B-Instruct');
    expect(m.layers.filter((l) => l.attn.kind === 'linear')).toHaveLength(36);
  });

  it('NVFP4 ModelOpt checkpoint keeps attention in BF16 and hints FP8 KV', () => {
    const m = presetModel('nvidia/DeepSeek-R1-0528-FP4');
    expect(m.quant.experts).toBe('nvfp4');
    expect(m.quant.attn).toBe('bf16');
    expect(m.quant.kvHint).toBe('fp8');
  });

  it('weight bytes of quantized checkpoints track on-disk size', () => {
    const m = presetModel('hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4');
    expect(m.quant.dense).toBe('int4_g128');
    // ~405B × 0.52 B + BF16 embeddings ≈ 213 GB (checkpoint is ~203 GiB).
    const bytes = (m.params.total - m.params.embed - m.params.lmHead) * BYTES_PER_PARAM.int4_g128 + (m.params.embed + m.params.lmHead) * 2;
    expect(bytes / 1e9).toBeGreaterThan(200);
    expect(bytes / 1e9).toBeLessThan(225);
  });
});
