// Closed-form serving estimates built on the compiled cost model: unloaded TTFT, ITL vs batch,
// steady-state continuous batching with chunked prefill, and max users under an SLA.

import type { BatchShape, CtxWindow, ModelSpec, StepCost } from '../types.ts';
import { expectedAccepted } from '../vllm/flags.ts';
import type { CostModel } from './cost.ts';
import type { InstanceEval } from './evaluate.ts';
import { bytesPerRequest, kvProfile } from './memory.ts';

export interface Workload {
  isl: number;
  osl: number;
  /** Fraction of each prompt served from the prefix cache. */
  prefixHit: number;
  /** Concurrent users (closed loop) across the whole deployment. */
  concurrency: number;
  /** SLA targets used for "max users". */
  slaTokPerSec: number;
  slaTtft: number;
}

export const DEFAULT_WORKLOAD: Workload = {
  isl: 2048,
  osl: 512,
  prefixHit: 0,
  concurrency: 64,
  slaTokPerSec: 30,
  slaTtft: 2,
};

export function ctxEff(L: number, w: CtxWindow): number {
  if (w.kind === 'chunked') return L < w.size ? L : w.size / 2;
  return Math.min(L, w.size);
}

/** Σ (query,key) pairs for a chunk of c tokens appended after p context tokens. */
export function chunkPairs(c: number, p: number, w?: CtxWindow): number {
  if (!w) return c * p + (c * c) / 2;
  const reach = p + c / 2;
  return c * Math.min(reach, w.kind === 'chunked' ? w.size / 2 : w.size);
}

export function emptyShape(model: ModelSpec): BatchShape {
  const n = model.windows.length + 1;
  return {
    decodeSeqs: 0,
    q: 1,
    decodeCtx: new Array<number>(n).fill(0),
    prefillTokens: 0,
    prefillSeqs: 0,
    prefillPairs: new Array<number>(n).fill(0),
    sampled: 0,
    globalTokens: 0,
  };
}

export function addDecode(s: BatchShape, model: ModelSpec, seqs: number, ctx: number): void {
  s.decodeSeqs += seqs;
  s.decodeCtx[0]! += seqs * ctx;
  model.windows.forEach((w, i) => (s.decodeCtx[i + 1]! += seqs * ctxEff(ctx, w)));
  s.sampled += seqs * s.q;
}

export function addPrefill(s: BatchShape, model: ModelSpec, c: number, p: number, seqs = 1, sampled = 1): void {
  s.prefillTokens += c * seqs;
  s.prefillSeqs += seqs;
  s.prefillPairs[0]! += seqs * chunkPairs(c, p);
  model.windows.forEach((w, i) => (s.prefillPairs[i + 1]! += seqs * chunkPairs(c, p, w)));
  s.sampled += sampled;
}

function finalize(s: BatchShape, dp: number): BatchShape {
  s.globalTokens = (s.decodeSeqs * s.q + s.prefillTokens) * dp;
  return s;
}

/** Unloaded time to first token for one prompt (sum over chunked-prefill steps, all PP stages). */
export function ttftUnloaded(cost: CostModel, isl: number, prefixHit = 0): { time: number; chunks: number; steps: StepCost[] } {
  const { inst, model } = cost;
  const cached = Math.floor(isl * prefixHit);
  let p = cached;
  let time = 0;
  const steps: StepCost[] = [];
  const budget = inst.maxNumBatchedTokens;
  while (p < isl) {
    const c = Math.min(budget, isl - p);
    const s = emptyShape(model);
    addPrefill(s, model, c, p, 1, p + c >= isl ? 1 : 0);
    const st = cost.step(finalize(s, inst.dp)).cost;
    steps.push(st);
    time += Math.max(st.gpu, st.host);
    p += c;
  }
  return { time, chunks: steps.length, steps };
}

export function decodeStep(cost: CostModel, batch: number, ctx: number): StepCost {
  const { inst, model } = cost;
  const s = emptyShape(model);
  s.q = 1 + (inst.flags.speculative?.k ?? 0);
  addDecode(s, model, batch, ctx);
  return cost.step(finalize(s, inst.dp)).cost;
}

export interface SteadyState {
  /** Concurrent sequences per DP replica. */
  batch: number;
  /** Time between tokens for a user (one decode step through all PP stages). */
  itl: number;
  tokPerUser: number;
  ttft: number;
  /** Output tokens/s across the deployment instance. */
  throughput: number;
  /** Input (prefill) tokens/s computed. */
  prefillThroughput: number;
  step: StepCost;
  prefillTokensPerStep: number;
  kvLimited: boolean;
  prefillLimited: boolean;
  /** Max sequences one DP replica can hold at isl+osl. */
  kvCapacity: number;
  /** Utilization of scale-up / scale-out links per GPU (0..1+). */
  scaleUpUtil: number;
  scaleOutUtil: number;
  acceptedPerStep: number;
  /** The representative step's batch (for step traces). */
  shape: BatchShape;
}

export function kvCapacityAt(ev: InstanceEval, model: ModelSpec, len: number): number {
  const { inst, placement, memory } = ev;
  let cap = Infinity;
  placement.shards.forEach((sh, i) => {
    const per = bytesPerRequest(kvProfile(model, inst, sh.layerLo, sh.layerHi), len, inst.blockSize);
    if (per > 0) cap = Math.min(cap, memory.perRank[i]!.kv / per);
  });
  return Number.isFinite(cap) ? Math.floor(cap) : inst.maxNumSeqs;
}

/**
 * Steady state of a closed loop of `users` concurrent users per DP replica: each step decodes
 * every running sequence and spends leftover token budget on new prompts (chunked prefill).
 */
export function steadyState(cost: CostModel, ev: InstanceEval, w: Workload, usersPerReplica: number): SteadyState {
  const { inst, model } = cost;
  const spec = inst.flags.speculative;
  const acc = spec ? expectedAccepted(spec) : 0;
  const q = 1 + (spec?.k ?? 0);
  const kvCapacity = Math.max(0, kvCapacityAt(ev, model, w.isl + w.osl));
  const B = Math.max(1, Math.min(usersPerReplica, inst.maxNumSeqs, Math.max(1, kvCapacity)));
  const kvLimited = usersPerReplica > kvCapacity;
  // With PP, sequences are split over PP+1 in-flight micro-batches; a micro-batch visits every stage.
  const inflight = inst.pp > 1 ? inst.pp : 1;
  const mb = Math.max(1, B / inflight);
  const Lavg = w.isl + w.osl / 2;
  const newPerStep = (mb * (1 + acc)) / w.osl;
  const prefillLen = w.isl * (1 - w.prefixHit);
  const budget = Math.max(0, inst.maxNumBatchedTokens - mb * q);
  let P = newPerStep * prefillLen;
  const prefillLimited = P > budget;
  if (prefillLimited) P = budget;
  const s = emptyShape(model);
  s.q = q;
  addDecode(s, model, mb, Lavg);
  if (P > 0) {
    const chunk = Math.min(prefillLen, inst.maxNumBatchedTokens);
    const seqs = P / Math.max(1, chunk);
    const p0 = w.isl * w.prefixHit;
    addPrefill(s, model, chunk, p0 + (prefillLen - chunk) / 2, seqs, newPerStep);
  }
  const shape = finalize(s, inst.dp);
  const step = cost.step(shape).cost;
  const interval = inst.pp > 1 ? Math.max(...step.stages, step.host) : step.time;
  const itl = inst.pp > 1 ? Math.max(step.gpu, interval) : step.time;
  const tokPerUser = (1 + acc) / itl;
  const unloaded = ttftUnloaded(cost, w.isl, w.prefixHit).time;
  const stepsToPrefill = budget > 0 ? Math.ceil(prefillLen / Math.max(1, budget)) : Infinity;
  const ttft = Math.max(unloaded, stepsToPrefill * itl + itl / 2);
  const throughput = ((B * (1 + acc)) / itl) * inst.dp;
  const prefillThroughput = (P * inflight * inst.dp) / interval;
  const su = cost.comm.suBW / cost.calib.etaScaleUp;
  const so = cost.comm.soBW / cost.calib.etaScaleOut;
  const intra = step.comm.tp.intra + step.comm.ep.intra + step.comm.pp.intra;
  const inter = step.comm.tp.inter + step.comm.ep.inter + step.comm.pp.inter;
  return {
    batch: B,
    itl,
    tokPerUser,
    ttft,
    throughput,
    prefillThroughput,
    step,
    prefillTokensPerStep: P,
    kvLimited,
    prefillLimited,
    kvCapacity,
    scaleUpUtil: su > 0 ? (intra * inflight) / interval / su : 0,
    scaleOutUtil: so > 0 ? (inter * inflight) / interval / so : 0,
    acceptedPerStep: acc,
    shape,
  };
}

export interface SweepPoint {
  users: number;
  tokPerUser: number;
  throughput: number;
  ttft: number;
  itl: number;
}

/** Throughput–interactivity frontier over concurrency (per instance). */
export function concurrencySweep(cost: CostModel, ev: InstanceEval, w: Workload): SweepPoint[] {
  const { inst } = cost;
  const cap = Math.max(1, Math.min(inst.maxNumSeqs, kvCapacityAt(ev, cost.model, w.isl + w.osl)));
  const pts: SweepPoint[] = [];
  for (let b = 1; b <= cap; b = b < 8 ? b + 1 : Math.ceil(b * 1.25)) {
    const ss = steadyState(cost, ev, w, b);
    pts.push({ users: b * inst.dp, tokPerUser: ss.tokPerUser, throughput: ss.throughput, ttft: ss.ttft, itl: ss.itl });
  }
  if (pts[pts.length - 1]?.users !== cap * inst.dp) {
    const ss = steadyState(cost, ev, w, cap);
    pts.push({ users: cap * inst.dp, tokPerUser: ss.tokPerUser, throughput: ss.throughput, ttft: ss.ttft, itl: ss.itl });
  }
  return pts;
}

/** Largest concurrency (deployment-wide) meeting both SLA targets. */
export function maxUsersAtSla(cost: CostModel, ev: InstanceEval, w: Workload): number {
  const { inst } = cost;
  const cap = Math.max(1, Math.min(inst.maxNumSeqs, kvCapacityAt(ev, cost.model, w.isl + w.osl)));
  const ok = (b: number) => {
    const ss = steadyState(cost, ev, w, b);
    return ss.tokPerUser >= w.slaTokPerSec && ss.ttft <= w.slaTtft;
  };
  if (!ok(1)) return 0;
  let lo = 1;
  let hi = cap;
  if (ok(hi)) return hi * inst.dp;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (ok(mid)) lo = mid;
    else hi = mid;
  }
  return lo * inst.dp;
}

export interface PrefillCapacity {
  /** Prompt tokens/s computed by one DP replica with a full token budget every step. */
  tokPerSec: number;
  /** Prompts/s for the whole instance. */
  reqPerSec: number;
  step: StepCost;
}

/** Throughput of a prefill-only instance (P/D): every step is filled with prompt chunks. */
export function prefillCapacity(cost: CostModel, isl: number, prefixHit: number): PrefillCapacity {
  const { inst, model } = cost;
  const S = Math.max(1, isl * (1 - prefixHit));
  const M = inst.maxNumBatchedTokens;
  const chunk = Math.min(S, M);
  const s = emptyShape(model);
  addPrefill(s, model, chunk, isl * prefixHit + (S - chunk) / 2, M / chunk, M / S);
  const step = cost.step(finalize(s, inst.dp)).cost;
  const interval = inst.pp > 1 ? Math.max(...step.stages, step.host) : step.time;
  const tokPerSec = M / interval;
  return { tokPerSec, reqPerSec: (tokPerSec * inst.dp) / S, step };
}

/** Mean wait in an M/D/1 queue with service time `s` at utilization `rho`. */
export function mdQueueWait(s: number, rho: number): number {
  if (rho >= 1) return Infinity;
  return (rho * s) / (2 * (1 - rho));
}
