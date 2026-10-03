import { useMemo } from 'react';
import { buildCluster } from '../core/hardware/clusters.ts';
import { findPreset, snapshotToModel } from '../core/model/presets.ts';
import { evaluate, type Evaluation, type InstanceEval } from '../core/engine/evaluate.ts';
import { CostModel } from '../core/engine/cost.ts';
import {
  concurrencySweep,
  maxUsersAtSla,
  steadyState,
  ttftUnloaded,
  decodeStep,
  mdQueueWait,
  prefillCapacity,
  type PrefillCapacity,
  type SteadyState,
  type SweepPoint,
} from '../core/engine/analytic.ts';
import type { ClusterSpec, Deployment, ModelSpec, StepCost } from '../core/types.ts';
import { logicalKvBytesPerToken } from '../core/engine/memory.ts';
import { useApp } from './store.ts';

export interface InstanceView {
  ev: InstanceEval;
  cost: CostModel | null;
  steady: SteadyState | null;
  ttft: number;
  decodeB1: StepCost | null;
  sweep: SweepPoint[];
  maxUsers: number;
  /** Average live context of a running request (tokens). */
  ctxAvg: number;
}

export interface Derived {
  model: ModelSpec;
  cluster: ClusterSpec;
  deployment: Deployment;
  evaluation: Evaluation;
  views: InstanceView[];
  /** P/D: KV transfer per request (bytes), its time, and NIC utilization on prefill GPUs. */
  kvTransfer: { bytes: number; time: number; util: number } | null;
  /** P/D: prefill capacity vs. the prompt rate the decode side generates, and end-to-end TTFT. */
  pd: { capacity: PrefillCapacity; demand: number; utilization: number; ttft: number } | null;
}

export function useModel(): ModelSpec {
  const repo = useApp((s) => s.modelRepo);
  const snaps = useApp((s) => s.snapshots);
  return useMemo(() => {
    const snap = findPreset(repo) ?? snaps[repo] ?? findPreset('deepseek-ai/DeepSeek-V3')!;
    return snapshotToModel(snap);
  }, [repo, snaps]);
}

export function useDerived(): Derived {
  const model = useModel();
  const build = useApp((s) => s.cluster);
  const mode = useApp((s) => s.mode);
  const flags = useApp((s) => s.flags);
  const pdFlags = useApp((s) => s.pd);
  const workload = useApp((s) => s.workload);
  const skew = useApp((s) => s.routingSkew);

  const cluster = useMemo(() => buildCluster(build), [build]);
  const deployment = useMemo<Deployment>(
    () =>
      mode === 'single'
        ? { instances: [{ id: 'serve', role: 'mixed', flags, gpuStart: 0 }] }
        : {
            instances: [
              { id: 'prefill', role: 'prefill', flags: pdFlags.prefill, gpuStart: 0 },
              { id: 'decode', role: 'decode', flags: pdFlags.decode, gpuStart: pdFlags.prefill.tp * pdFlags.prefill.pp * pdFlags.prefill.dp },
            ],
          },
    [mode, flags, pdFlags],
  );
  const evaluation = useMemo(() => evaluate(model, cluster, deployment), [model, cluster, deployment]);

  const views = useMemo(
    () =>
      evaluation.instances.map((ev): InstanceView => {
        const broken = ev.inst.issues.some((i) => i.level === 'error') || ev.memory.perRank.some((m) => m.kv <= 0);
        const ctxAvg = ev.inst.role === 'prefill' ? workload.isl / 2 : workload.isl + workload.osl / 2;
        if (broken) return { ev, cost: null, steady: null, ttft: NaN, decodeB1: null, sweep: [], maxUsers: 0, ctxAvg };
        const cost = new CostModel(model, ev.inst, ev.placement, cluster, evaluation.calib, { routingSkew: skew });
        const usersPerReplica = Math.max(1, Math.round(workload.concurrency / ev.inst.dp));
        const isDecodeOnly = ev.inst.role === 'decode';
        // Decode instances receive prompts with their KV already computed; prefill instances are
        // summarized by prefillCapacity() instead of a decode steady state.
        const w = isDecodeOnly ? { ...workload, prefixHit: 1 } : workload;
        return {
          ev,
          cost,
          steady: steadyState(cost, ev, w, usersPerReplica),
          ttft: ttftUnloaded(cost, workload.isl, workload.prefixHit).time,
          decodeB1: decodeStep(cost, 1, workload.isl),
          sweep: concurrencySweep(cost, ev, w),
          maxUsers: maxUsersAtSla(cost, ev, w),
          ctxAvg,
        };
      }),
    [evaluation, model, cluster, workload, skew],
  );

  const kvTransfer = useMemo(() => {
    if (evaluation.instances.length < 2) return null;
    const p = evaluation.instances[0]!;
    const bytes = logicalKvBytesPerToken(model, p.inst) * workload.isl;
    const view = views[0];
    const comm = view?.cost?.comm;
    const sameDomain = cluster.domainSize >= p.inst.world + evaluation.instances[1]!.inst.world;
    const t = comm ? comm.p2p(bytes / p.inst.world, sameDomain).t + evaluation.calib.alphaKvx : NaN;
    const dec = views[1];
    const reqPerSec = (dec?.steady?.throughput ?? 0) / Math.max(1, workload.osl);
    const bw = sameDomain ? cluster.gpu.scaleUpBWDir : cluster.scaleOut.bwPerGpuDir;
    const util = bw > 0 ? (reqPerSec * bytes) / p.inst.world / bw : 0;
    return { bytes, time: t, util };
  }, [evaluation, views, workload.isl, workload.osl, cluster, model]);

  const pd = useMemo(() => {
    if (views.length < 2) return null;
    const pre = views[0]!;
    const dec = views[1]!;
    if (!pre.cost || !dec.steady) return null;
    const capacity = prefillCapacity(pre.cost, workload.isl, workload.prefixHit);
    // Closed loop: every finished decode brings a new prompt.
    const demand = dec.steady.throughput / Math.max(1, workload.osl);
    const utilization = demand / capacity.reqPerSec;
    const service = pre.ttft;
    const ttft = service + mdQueueWait(service, utilization) + (kvTransfer?.time ?? 0) + dec.steady.itl;
    return { capacity, demand, utilization, ttft };
  }, [views, workload, kvTransfer]);

  return { model, cluster, deployment, evaluation, views, kvTransfer, pd };
}
