import type { ModelSpec } from '../types.ts';
import { normalizeConfig } from './normalize.ts';
import { PRESET_SNAPSHOTS } from './presets/index.ts';

export interface PresetSnapshot {
  repo: string;
  source: string;
  label: string;
  fetchedAt: string;
  gated: boolean;
  safetensors?: { total: number; parameters: Record<string, number> };
  config: Record<string, unknown>;
  hfQuantConfig?: Record<string, unknown>;
}

export const PRESETS: PresetSnapshot[] = PRESET_SNAPSHOTS;

export function findPreset(repo: string): PresetSnapshot | undefined {
  const r = repo.toLowerCase();
  return PRESETS.find((p) => p.repo.toLowerCase() === r || p.source.toLowerCase() === r);
}

export function presetModel(repo: string): ModelSpec {
  const p = findPreset(repo);
  if (!p) throw new Error(`No bundled preset for ${repo}`);
  return snapshotToModel(p);
}

export function snapshotToModel(p: PresetSnapshot): ModelSpec {
  return normalizeConfig({
    repo: p.repo,
    label: p.label,
    config: p.config,
    ...(p.hfQuantConfig ? { hfQuantConfig: p.hfQuantConfig } : {}),
    ...(p.safetensors ? { safetensors: p.safetensors } : {}),
    source: 'preset',
  });
}
