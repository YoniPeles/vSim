import { useEffect, useRef } from 'react';
import { z } from 'zod';
import { DEFAULT_FLAGS } from '../core/vllm/flags.ts';
import { INITIAL_INPUTS, useApp, type Inputs } from './store.ts';

// The URL hash carries every input (model repo, cluster, flags, workload) so a configuration can be
// shared as a link. The HF token is never included.

const flagsSchema = z
  .object({
    tp: z.number().int().min(1),
    pp: z.number().int().min(1),
    dp: z.number().int().min(1),
    ep: z.boolean(),
    dcp: z.number().int().min(1),
    quantization: z.enum(['auto', 'fp8', 'none']),
    kvCacheDtype: z.enum(['auto', 'bfloat16', 'fp8', 'fp8_e5m2', 'fp8_ds_mla', 'nvfp4_ds_mla', 'nvfp4']),
    gpuMemoryUtilization: z.number().min(0.05).max(1),
    kvCacheMemoryBytes: z.number().nullable(),
    maxModelLen: z.number().nullable(),
    maxNumSeqs: z.number().nullable(),
    maxNumBatchedTokens: z.number().nullable(),
    blockSize: z.number().nullable(),
    enablePrefixCaching: z.boolean(),
    enableChunkedPrefill: z.boolean(),
    performanceMode: z.enum(['balanced', 'interactivity', 'throughput']),
    optimizationLevel: z.union([z.literal(0), z.literal(1), z.literal(2), z.literal(3)]),
    all2allBackend: z.enum(['allgather_reducescatter', 'deepep_high_throughput', 'deepep_low_latency', 'pplx', 'naive', 'flashinfer_nvlink_one_sided']),
    numRedundantExperts: z.number().int().min(0),
    speculative: z
      .object({
        method: z.enum(['mtp', 'eagle', 'eagle3', 'draft_model', 'ngram']),
        k: z.number().int().min(1).max(16),
        acceptance: z.number().min(0).max(1),
        decay: z.number().min(0).max(1),
        draftParams: z.number().min(0),
      })
      .nullable(),
  })
  .partial();

const inputsSchema = z
  .object({
    modelRepo: z.string(),
    cluster: z.object({
      gpu: z.string(),
      nodes: z.number().int().min(1).max(256),
      gpusPerNode: z.number().int().min(1).max(8),
      domainSize: z.number().optional(),
      scaleOut: z.string(),
      oversub: z.number().optional(),
      name: z.string().optional(),
    }),
    clusterPreset: z.string().nullable(),
    mode: z.enum(['single', 'pd']),
    flags: flagsSchema,
    pd: z.object({ prefill: flagsSchema, decode: flagsSchema }),
    workload: z.object({
      isl: z.number(),
      osl: z.number(),
      prefixHit: z.number(),
      concurrency: z.number(),
      slaTokPerSec: z.number(),
      slaTtft: z.number(),
    }),
    routingSkew: z.number(),
  })
  .partial();

function encode(i: Inputs): string {
  return btoa(unescape(encodeURIComponent(JSON.stringify(i)))).replace(/=+$/, '');
}

function decode(hash: string): Partial<Inputs> | null {
  try {
    const raw = JSON.parse(decodeURIComponent(escape(atob(hash))));
    const parsed = inputsSchema.safeParse(raw);
    if (!parsed.success) return null;
    const p = parsed.data;
    const out: Partial<Inputs> = {};
    if (p.modelRepo) out.modelRepo = p.modelRepo;
    if (p.cluster) out.cluster = p.cluster as Inputs['cluster'];
    if (p.clusterPreset !== undefined) out.clusterPreset = p.clusterPreset;
    if (p.mode) out.mode = p.mode;
    if (p.flags) out.flags = { ...DEFAULT_FLAGS, ...p.flags } as Inputs['flags'];
    if (p.pd) out.pd = { prefill: { ...DEFAULT_FLAGS, ...p.pd.prefill }, decode: { ...DEFAULT_FLAGS, ...p.pd.decode } } as Inputs['pd'];
    if (p.workload) out.workload = { ...INITIAL_INPUTS.workload, ...p.workload };
    if (p.routingSkew !== undefined) out.routingSkew = p.routingSkew;
    return out;
  } catch {
    return null;
  }
}

function snapshot(): Inputs {
  const s = useApp.getState();
  return {
    modelRepo: s.modelRepo,
    cluster: s.cluster,
    clusterPreset: s.clusterPreset,
    mode: s.mode,
    flags: s.flags,
    pd: s.pd,
    workload: s.workload,
    routingSkew: s.routingSkew,
  };
}

/** Load inputs from the hash once, then keep the hash in sync (debounced). */
export function useUrlState(): void {
  const loaded = useRef(false);
  useEffect(() => {
    if (!loaded.current) {
      loaded.current = true;
      const h = location.hash.slice(1);
      const i = h ? decode(h) : null;
      if (i) {
        const { modelRepo, ...rest } = i;
        useApp.getState().applyInputs(rest);
        if (modelRepo) void useApp.getState().setModel(modelRepo);
      }
    }
    let t: ReturnType<typeof setTimeout> | undefined;
    const unsub = useApp.subscribe(() => {
      clearTimeout(t);
      t = setTimeout(() => history.replaceState(null, '', `#${encode(snapshot())}`), 250);
    });
    return () => {
      unsub();
      clearTimeout(t);
    };
  }, []);
}
