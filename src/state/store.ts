import { create } from 'zustand';
import type { ClusterBuild } from '../core/hardware/clusters.ts';
import { CLUSTER_PRESETS } from '../core/hardware/clusters.ts';
import { fetchSnapshot, HubError } from '../core/model/hf.ts';
import { findPreset, type PresetSnapshot } from '../core/model/presets.ts';
import { DEFAULT_FLAGS } from '../core/vllm/flags.ts';
import type { VllmFlags } from '../core/types.ts';
import { DEFAULT_WORKLOAD, type Workload } from '../core/engine/analytic.ts';
import { get as idbGet, set as idbSet } from 'idb-keyval';

export type Mode = 'single' | 'pd';
export type PdSide = 'prefill' | 'decode';

export interface Inputs {
  modelRepo: string;
  cluster: ClusterBuild;
  clusterPreset: string | null;
  mode: Mode;
  flags: VllmFlags;
  pd: { prefill: VllmFlags; decode: VllmFlags };
  workload: Workload;
  routingSkew: number;
}

interface AppState extends Inputs {
  snapshots: Record<string, PresetSnapshot>;
  modelStatus: 'ready' | 'loading' | 'error';
  modelError: string | null;
  hfToken: string;
  selectedGpu: number | null;
  hoveredGpu: number | null;
  pdSide: PdSide;
  setModel: (repo: string) => Promise<void>;
  setClusterPreset: (id: string) => void;
  setCluster: (patch: Partial<ClusterBuild>) => void;
  setFlags: (patch: Partial<VllmFlags>) => void;
  setMode: (mode: Mode) => void;
  setPdSide: (side: PdSide) => void;
  setWorkload: (patch: Partial<Workload>) => void;
  setRoutingSkew: (x: number) => void;
  setToken: (t: string) => void;
  selectGpu: (g: number | null) => void;
  hoverGpu: (g: number | null) => void;
  applyInputs: (i: Partial<Inputs>) => void;
}

const TOKEN_KEY = 'vsim.hfToken';
const readToken = (): string => {
  try {
    return sessionStorage.getItem(TOKEN_KEY) ?? '';
  } catch {
    return '';
  }
};

export const INITIAL_INPUTS: Inputs = {
  modelRepo: 'deepseek-ai/DeepSeek-V3',
  cluster: CLUSTER_PRESETS.find((c) => c.id === '8xh200')!.build,
  clusterPreset: '8xh200',
  mode: 'single',
  flags: { ...DEFAULT_FLAGS, tp: 8 },
  pd: {
    prefill: { ...DEFAULT_FLAGS, tp: 8 },
    decode: { ...DEFAULT_FLAGS, tp: 8 },
  },
  workload: DEFAULT_WORKLOAD,
  routingSkew: 0.3,
};

export const useApp = create<AppState>()((set, get) => ({
  ...INITIAL_INPUTS,
  snapshots: {},
  modelStatus: 'ready',
  modelError: null,
  hfToken: readToken(),
  selectedGpu: null,
  hoveredGpu: null,
  pdSide: 'decode',

  setModel: async (repo) => {
    const preset = findPreset(repo);
    if (preset) {
      set({ modelRepo: preset.repo, modelStatus: 'ready', modelError: null, selectedGpu: null });
      return;
    }
    const cached = get().snapshots[repo];
    if (cached) {
      set({ modelRepo: repo, modelStatus: 'ready', modelError: null });
      return;
    }
    set({ modelStatus: 'loading', modelError: null });
    try {
      const key = `snapshot:${repo}`;
      let snap = (await idbGet<PresetSnapshot>(key).catch(() => undefined)) ?? undefined;
      if (!snap) {
        snap = await fetchSnapshot(repo, get().hfToken ? { token: get().hfToken } : {});
        await idbSet(key, snap).catch(() => undefined);
      }
      set((s) => ({
        snapshots: { ...s.snapshots, [snap!.repo]: snap! },
        modelRepo: snap!.repo,
        modelStatus: 'ready',
        selectedGpu: null,
      }));
    } catch (e) {
      set({ modelStatus: 'error', modelError: e instanceof HubError ? e.message : `Couldn't load ${repo}: ${(e as Error).message}` });
    }
  },

  setClusterPreset: (id) => {
    const p = CLUSTER_PRESETS.find((c) => c.id === id);
    if (p) set({ cluster: p.build, clusterPreset: id, selectedGpu: null });
  },
  setCluster: (patch) => set((s) => ({ cluster: { ...s.cluster, ...patch }, clusterPreset: null, selectedGpu: null })),
  setFlags: (patch) =>
    set((s) => {
      if (s.mode === 'single') return { flags: { ...s.flags, ...patch } };
      const side = s.pdSide;
      return { pd: { ...s.pd, [side]: { ...s.pd[side], ...patch } } };
    }),
  setMode: (mode) => set({ mode, selectedGpu: null }),
  setPdSide: (pdSide) => set({ pdSide }),
  setWorkload: (patch) => set((s) => ({ workload: { ...s.workload, ...patch } })),
  setRoutingSkew: (routingSkew) => set({ routingSkew }),
  setToken: (t) => {
    try {
      if (t) sessionStorage.setItem(TOKEN_KEY, t);
      else sessionStorage.removeItem(TOKEN_KEY);
    } catch {
      /* storage unavailable */
    }
    set({ hfToken: t });
  },
  selectGpu: (g) => set({ selectedGpu: g }),
  hoverGpu: (g) => set({ hoveredGpu: g }),
  applyInputs: (i) => set(i),
}));

/** The flags currently being edited (single instance, or the selected P/D side). */
export function useEditedFlags(): VllmFlags {
  return useApp((s) => (s.mode === 'single' ? s.flags : s.pd[s.pdSide]));
}
