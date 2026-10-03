// Compiled roofline cost model. A config is compiled once into per-PP-stage layer-group coefficients;
// evaluating a step is then O(groups). Each kernel family is timed as max(FLOPs / (peak·η_c),
// bytes / (HBM·η_m)); communication is α–β via CommModel and is serial with compute within a layer.

import { BYTES_PER_PARAM } from '../model/quant.ts';
import { windowId } from '../model/normalize.ts';
import type {
  BatchShape,
  Calibration,
  ClusterSpec,
  CommKind,
  GpuSpec,
  LayerSpec,
  ModelSpec,
  Placement,
  ResolvedInstance,
  StepCost,
  StepParts,
  WeightDtype,
} from '../types.ts';
import { domainOf } from '../hardware/clusters.ts';
import { CommModel, type CommResult, type GroupShape } from './comm.ts';
import { kvBytesPerTokenLayer, stateBytesPerSeqLayer } from './memory.ts';
import { attnParamsPerRank, draftParamsPerRank, ppPartition } from './placement.ts';

export function tierPeak(dt: WeightDtype, gpu: GpuSpec): number {
  switch (dt) {
    case 'fp8':
    case 'fp8_block':
      return gpu.peak.fp8 ?? gpu.peak.bf16;
    case 'nvfp4':
      return gpu.peak.fp4 ?? gpu.peak.bf16;
    case 'mxfp4':
      // Blackwell runs MXFP4 experts as W4A8 (MXFP8 activations); elsewhere W4A16.
      return gpu.peak.fp4 ? (gpu.peak.fp8 ?? gpu.peak.bf16) : gpu.peak.bf16;
    case 'int8':
      return gpu.peak.int8 ?? gpu.peak.bf16;
    default:
      return gpu.peak.bf16;
  }
}

interface MoeCoeff {
  E: number;
  topK: number;
  localExperts: number;
  /** Params of one expert as held on this rank (TP-sharded without EP). */
  expertParams: number;
  expertBytes: number;
  peak: number;
}

type Roof = (flops: number, bytes: number, peak: number, kernels?: number) => number;

interface GroupCoeff {
  n: number;
  /** Layer indices of this group on its stage. */
  layers: number[];
  /** Small kernels in the attention projection block (GEMMs, norms, rope, cache write). */
  projKernels: number;
  layer: LayerSpec;
  win: number;
  attnProj: { params: number; bytes: number; peak: number };
  dense: { params: number; bytes: number; peak: number };
  shared: { params: number; bytes: number; peak: number };
  routerParams: number;
  moe: MoeCoeff | null;
  kvPerToken: number;
  flopsPerPairDecode: number;
  flopsPerPairPrefill: number;
  /** MLA prefill up-projects the cached latent for every context token. */
  prefillCtxFlopsPerToken: number;
  indexer: { flopsPerPair: number; bytesPerToken: number } | null;
  linear: { stateBytes: number; flopsPerToken: number } | null;
}

interface StageCoeff {
  groups: GroupCoeff[];
  layers: number;
  embed: boolean;
  lmHead: boolean;
}

export interface CostOptions {
  /** Expert routing imbalance: 0 = ideal balls-in-bins, 1 = heavy skew. */
  routingSkew: number;
}

export const DEFAULT_COST_OPTIONS: CostOptions = { routingSkew: 0.3 };

export interface TracePhase {
  stage: number;
  /** Layer index, or −1 for the LM head / pipeline hand-off. */
  layer: number;
  kind: 'attn' | 'tp' | 'dispatch' | 'ffn' | 'combine' | 'pp' | 'lmhead';
  dur: number;
}

export interface StepDetail {
  cost: StepCost;
  /** Per-collective algorithm picked (for the "why" popovers). */
  algos: Partial<Record<CommKind, string>>;
  /** Per-layer phase timeline of one step (only when requested). */
  trace?: TracePhase[];
}

export class CostModel {
  readonly stages: StageCoeff[];
  readonly comm: CommModel;
  readonly tpShape: GroupShape;
  readonly epShape: GroupShape;
  readonly ppSameDomain: boolean;
  readonly hbm: number;
  readonly attnPeak: number;
  readonly lmHeadParams: number;
  readonly lmHeadBytes: number;
  readonly lmHeadPeak: number;
  readonly draftParams: number;
  readonly draftLayer: GroupCoeff | null;
  readonly model: ModelSpec;
  readonly inst: ResolvedInstance;
  readonly cluster: ClusterSpec;
  readonly calib: Calibration;
  readonly opts: CostOptions;

  constructor(
    model: ModelSpec,
    inst: ResolvedInstance,
    placement: Placement,
    cluster: ClusterSpec,
    calib: Calibration,
    opts: CostOptions = DEFAULT_COST_OPTIONS,
  ) {
    this.model = model;
    this.inst = inst;
    this.cluster = cluster;
    this.calib = calib;
    this.opts = opts;
    const gpu = cluster.gpu;
    this.comm = new CommModel(cluster, calib);
    this.hbm = gpu.hbmBW * calib.etaMem;
    this.attnPeak = gpu.peak.bf16 * calib.etaAttnCompute;
    const tpG = placement.tpGroups[0];
    this.tpShape = tpG ? { n: tpG.gpus.length, perDomain: tpG.perDomain, domains: tpG.domains } : { n: 1, perDomain: 1, domains: 1 };
    const epG = placement.epGroups[0];
    this.epShape = epG ? { n: epG.gpus.length, perDomain: epG.perDomain, domains: epG.domains } : { n: 1, perDomain: 1, domains: 1 };
    const pl = placement.ppLinks[0];
    this.ppSameDomain = pl ? domainOf(cluster, pl.from) === domainOf(cluster, pl.to) : true;

    const parts = ppPartition(model.layers.length, inst.pp);
    this.stages = parts.map(([lo, hi], s) => {
      const groups: GroupCoeff[] = [];
      const byKey = new Map<string, GroupCoeff>();
      for (let i = lo; i < hi; i++) {
        const l = model.layers[i]!;
        const key = model.groups.find((g) => g.indices.includes(i))!.key;
        const ex = byKey.get(key);
        if (ex) {
          ex.n++;
          ex.layers.push(i);
          continue;
        }
        const c = this.compileLayer(l);
        c.layers = [i];
        byKey.set(key, c);
        groups.push(c);
      }
      return { groups, layers: hi - lo, embed: s === 0, lmHead: s === inst.pp - 1 };
    });

    const q = inst.quant;
    const lmParams = model.params.lmHead || model.params.embed;
    this.lmHeadParams = lmParams / inst.tp;
    this.lmHeadBytes = this.lmHeadParams * BYTES_PER_PARAM[q.lmHead];
    this.lmHeadPeak = tierPeak(q.lmHead, gpu) * calib.etaCompute;
    this.draftParams = draftParamsPerRank(model, inst);
    const spec = inst.flags.speculative;
    this.draftLayer =
      spec && (spec.method === 'mtp' || spec.method === 'eagle' || spec.method === 'eagle3')
        ? this.compileLayer(spec.method === 'mtp' && model.mtp.layers[0] ? model.mtp.layers[0] : model.layers[0]!)
        : null;
  }

  private compileLayer(l: LayerSpec): GroupCoeff {
    const { model, inst, cluster, calib } = this;
    const gpu = cluster.gpu;
    const q = inst.quant;
    const B = BYTES_PER_PARAM;
    const h = model.hidden;
    const tp = inst.tp;
    const a = l.attn;
    const projParams = attnParamsPerRank(l, h, tp);
    const c: GroupCoeff = {
      n: 1,
      layers: [],
      projKernels: a.kind === 'mla' ? (a.dsa ? 12 : 9) : a.kind === 'linear' ? 7 : 5,
      layer: l,
      win: windowId(a, model.windows),
      attnProj: { params: projParams, bytes: projParams * B[q.attn], peak: tierPeak(q.attn, gpu) * calib.etaCompute },
      dense: { params: l.p.dense / tp, bytes: (l.p.dense / tp) * B[q.dense], peak: tierPeak(q.dense, gpu) * calib.etaCompute },
      shared: { params: l.p.shared / tp, bytes: (l.p.shared / tp) * B[q.shared], peak: tierPeak(q.shared, gpu) * calib.etaCompute },
      routerParams: l.p.router,
      moe: null,
      kvPerToken: kvBytesPerTokenLayer(l, inst),
      flopsPerPairDecode: 0,
      flopsPerPairPrefill: 0,
      prefillCtxFlopsPerToken: 0,
      indexer: null,
      linear: null,
    };
    if (l.ffn.kind === 'moe') {
      const f = l.ffn;
      const shard = inst.ep ? 1 : inst.moeTp;
      const slots = f.E + inst.flags.numRedundantExperts;
      c.moe = {
        E: f.E,
        topK: f.topK,
        localExperts: inst.ep ? slots / inst.epSize : f.E,
        expertParams: l.p.expertEach / shard,
        expertBytes: (l.p.expertEach / shard) * B[q.experts],
        peak: tierPeak(q.experts, gpu) * calib.etaCompute,
      };
    }
    if (a.kind === 'gqa') {
      const nQr = a.nQ / tp;
      c.flopsPerPairDecode = 2 * (a.dQK + a.dV) * nQr;
      c.flopsPerPairPrefill = c.flopsPerPairDecode;
    } else if (a.kind === 'mla') {
      const nQr = a.nQ / tp;
      // Decode uses the absorbed form over the latent (576 for QK, 512 for PV); prefill is not absorbed.
      c.flopsPerPairDecode = 2 * (a.kvLora + a.rope + a.kvLora) * nQr;
      c.flopsPerPairPrefill = 2 * (a.nope + a.rope + a.dV) * nQr;
      c.prefillCtxFlopsPerToken = 2 * a.kvLora * nQr * (a.nope + a.dV);
      if (a.dsa) c.indexer = { flopsPerPair: 2 * a.dsa.nIdx * a.dsa.dIdx, bytesPerToken: a.dsa.dIdx + 4 };
    } else {
      c.linear = {
        stateBytes: stateBytesPerSeqLayer(l, inst),
        flopsPerToken: (4 * a.nV * a.dK * a.dV) / tp,
      };
    }
    return c;
  }

  /** Smallest CUDA-graph capture size ≥ T (padding), or T if it runs outside graphs. */
  padTokens(T: number): number {
    const sizes = this.inst.cudagraph.sizes;
    if (!sizes.length || T > this.inst.cudagraph.maxSize) return T;
    let lo = 0;
    let hi = sizes.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (sizes[mid]! >= T) hi = mid;
      else lo = mid + 1;
    }
    return sizes[lo]!;
  }

  step(shape: BatchShape, withTrace = false): StepDetail {
    const { inst, model, calib } = this;
    const h = model.hidden;
    const T = Math.max(1, shape.decodeSeqs * shape.q + shape.prefillTokens);
    const Tg = Math.max(T, shape.globalTokens);
    const uniformDecode = shape.prefillTokens === 0;
    const Tpad = this.padTokens(T);
    const parts: StepParts = { gemm: 0, attn: 0, moe: 0, lmHead: 0, launch: 0, tpComm: 0, epComm: 0, ppComm: 0, draft: 0 };
    const comm: StepCost['comm'] = {
      tp: { intra: 0, inter: 0 },
      ep: { intra: 0, inter: 0 },
      pp: { intra: 0, inter: 0 },
      kvx: { intra: 0, inter: 0 },
    };
    const algos: StepDetail['algos'] = {};
    let flops = 0;
    let hbmBytes = 0;
    let memT = 0;
    let compT = 0;
    // Layers represented by the current roofline call (a group's per-layer cost counts n times).
    let mult = 1;
    // Roofline with a floor of `kernels` × the minimum time of a tiny kernel.
    const roof = (f: number, b: number, peak: number, kernels = 1): number => {
      const tc = f / peak;
      const tm = b / this.hbm;
      flops += f * mult;
      hbmBytes += b * mult;
      const t = Math.max(tc, tm, kernels * calib.minKernel);
      if (tc > tm) compT += t * mult;
      else memT += t * mult;
      return t;
    };
    const addComm = (kind: CommKind, r: CommResult, times: number) => {
      comm[kind].intra += r.intra * times;
      comm[kind].inter += r.inter * times;
      if (r.algo !== '-') algos[kind] = r.algo;
      return r.t * times;
    };

    const tpAR = inst.tp > 1 ? this.comm.allReduce(T * h * 2, this.tpShape) : null;
    const moeComm = this.moeComm(T, Tg);

    const stages: number[] = [];
    const trace: TracePhase[] | undefined = withTrace ? [] : undefined;
    for (let s = 0; s < this.stages.length; s++) {
      const st = this.stages[s]!;
      let t = 0;
      const perLayer = new Map<number, { attn: number; ffn: number; ep: number }>();
      for (const g of st.groups) {
        mult = g.n;
        const projT = roof(2 * g.attnProj.params * Tpad, g.attnProj.bytes + T * h * 4, g.attnProj.peak, g.projKernels);
        let ffnGemm = 0;
        if (g.dense.params) ffnGemm += roof(2 * g.dense.params * Tpad, g.dense.bytes + T * h * 4, g.dense.peak, 3);
        if (g.shared.params) ffnGemm += roof(2 * g.shared.params * Tpad, g.shared.bytes, g.shared.peak, 3);
        if (g.routerParams) ffnGemm += roof(2 * g.routerParams * T, g.routerParams * 2, g.attnProj.peak, 2);
        const gemm = projT + ffnGemm;
        const attn = this.attnTime(g, shape, T, roof);
        let moe = 0;
        if (g.moe) moe = this.moeTime(g.moe, Tg, roof);
        if (trace) {
          const ep = g.moe && moeComm ? moeComm.t : 0;
          for (const li of g.layers) perLayer.set(li, { attn: projT + attn, ffn: ffnGemm + moe, ep });
        }
        parts.gemm += g.n * gemm;
        parts.attn += g.n * attn;
        parts.moe += g.n * moe;
        let ct = 0;
        if (tpAR) ct += addComm('tp', tpAR, 2 * g.n);
        parts.tpComm += tpAR ? tpAR.t * 2 * g.n : 0;
        if (g.moe && moeComm) {
          const e = addComm('ep', moeComm, g.n);
          parts.epComm += e;
          ct += e;
        }
        t += g.n * (gemm + attn + moe) + ct;
      }
      mult = 1;
      if (trace) {
        const ar = tpAR ? tpAR.t : 0;
        for (const [li, p] of [...perLayer.entries()].sort((a, b) => a[0] - b[0])) {
          trace.push({ stage: s, layer: li, kind: 'attn', dur: p.attn });
          if (ar) trace.push({ stage: s, layer: li, kind: 'tp', dur: ar });
          if (p.ep) trace.push({ stage: s, layer: li, kind: 'dispatch', dur: p.ep / 2 });
          trace.push({ stage: s, layer: li, kind: 'ffn', dur: p.ffn });
          if (p.ep) trace.push({ stage: s, layer: li, kind: 'combine', dur: p.ep / 2 });
          if (ar) trace.push({ stage: s, layer: li, kind: 'tp', dur: ar });
        }
      }
      if (st.lmHead) {
        const N = Math.max(1, shape.sampled);
        const lm = roof(2 * N * this.lmHeadParams, this.lmHeadBytes + N * (model.vocab / inst.tp) * 4, this.lmHeadPeak, 3);
        const sampler = (N * model.vocab * 4 * 3) / this.hbm;
        parts.lmHead += lm + sampler;
        t += lm + sampler;
        trace?.push({ stage: s, layer: -1, kind: 'lmhead', dur: lm + sampler });
        const draft = this.draftTime(shape, roof);
        parts.draft += draft;
        t += draft;
      }
      const launch =
        inst.cudagraph.mode === 'NONE' || T > inst.cudagraph.maxSize
          ? st.layers * calib.launchEagerPerLayer
          : uniformDecode && inst.cudagraph.mode === 'FULL_AND_PIECEWISE'
            ? 20e-6
            : st.layers * calib.launchPiecewisePerLayer;
      parts.launch += launch;
      t += launch;
      if (s < this.stages.length - 1) {
        const p = this.comm.p2p(2 * T * h * 2, this.ppSameDomain);
        t += addComm('pp', p, 1);
        parts.ppComm += p.t;
        trace?.push({ stage: s, layer: -1, kind: 'pp', dur: p.t });
      }
      stages.push(t);
    }
    const gpu = stages.reduce((a, b) => a + b, 0);
    const host = calib.hostBase + calib.hostPerReq * (shape.decodeSeqs + shape.prefillSeqs) + calib.hostPerNew * shape.prefillSeqs;
    const commT = parts.tpComm + parts.epComm + parts.ppComm;
    const time = Math.max(gpu, host);
    const bound: StepCost['bound'] =
      host > gpu ? 'host' : commT > Math.max(memT, compT) ? 'comm' : compT > memT ? 'compute' : 'memory';
    return { cost: { time, gpu, host, stages, parts, flops, hbmBytes, comm, bound }, algos, ...(trace ? { trace } : {}) };
  }

  private attnTime(g: GroupCoeff, s: BatchShape, T: number, roof: Roof): number {
    if (g.linear) {
      const seqs = s.decodeSeqs + s.prefillSeqs;
      return roof(T * g.linear.flopsPerToken, seqs * 2 * g.linear.stateBytes + T * this.model.hidden * 4, this.attnPeak, 3);
    }
    const w = Math.max(0, g.win);
    const ctxDec = s.decodeCtx[w] ?? s.decodeCtx[0] ?? 0;
    const pairs = s.prefillPairs[w] ?? s.prefillPairs[0] ?? 0;
    let f = s.q * ctxDec * g.flopsPerPairDecode + pairs * g.flopsPerPairPrefill;
    // KV of the decode contexts is streamed once per step (shared by the q speculative query tokens).
    let b = ctxDec * g.kvPerToken + T * g.kvPerToken;
    if (s.prefillTokens) b += (pairs / s.prefillTokens) * s.prefillSeqs * g.kvPerToken; // prefix KV re-read per chunk
    if (g.prefillCtxFlopsPerToken && s.prefillTokens) {
      // Latent up-projection over the prefix the chunk attends to (≈ pairs / chunk + chunk).
      f += (pairs / Math.max(1, s.prefillTokens) + s.prefillTokens) * g.prefillCtxFlopsPerToken;
    }
    if (g.indexer) {
      const full = s.decodeCtx[0] ?? 0;
      f += (s.q * full + (s.prefillPairs[0] ?? 0)) * g.indexer.flopsPerPair;
      b += full * g.indexer.bytesPerToken;
    }
    return roof(f, b, this.attnPeak, g.indexer ? 4 : 2);
  }

  private moeTime(m: MoeCoeff, Tg: number, roof: Roof): number {
    const { inst, model } = this;
    const hitFrac = 1 - (1 - m.topK / m.E) ** Tg;
    if (inst.ep && inst.epSize > 1) {
      const mu = (m.topK * Tg) / inst.epSize;
      const tmax = Math.min(Tg * Math.min(m.topK, m.localExperts), mu + (1 + this.opts.routingSkew) * Math.sqrt(2 * mu * Math.log(inst.epSize)));
      const U = m.localExperts * hitFrac;
      return roof(2 * tmax * m.expertParams, U * m.expertBytes + tmax * model.hidden * 4, m.peak, 6);
    }
    const U = m.E * hitFrac;
    return roof(2 * m.topK * Tg * m.expertParams, U * m.expertBytes + Tg * model.hidden * 4, m.peak, 6);
  }

  /** MoE token exchange per MoE layer (DP lockstep / expert parallel). */
  private moeComm(T: number, Tg: number): CommResult | null {
    const { inst, model } = this;
    if (!this.stages.some((s) => s.groups.some((g) => g.moe))) return null;
    if (inst.dp <= 1 && !(inst.ep && inst.epSize > 1)) return null;
    const h = model.hidden;
    const moe = model.layers.find((l) => l.ffn.kind === 'moe')!.ffn as { topK: number };
    const be = inst.flags.all2allBackend;
    if (be === 'allgather_reducescatter' || be === 'naive' || !inst.ep) {
      // Gather every DP rank's tokens, run local experts, reduce-scatter back.
      const ag = this.comm.allGather(Tg * h * 2, this.epShape);
      const rs = this.comm.allGather(Tg * h * 2, this.epShape);
      const k = be === 'naive' ? 2 : 1;
      return { t: k * (ag.t + rs.t), intra: k * (ag.intra + rs.intra), inter: k * (ag.inter + rs.inter), algo: be === 'naive' ? 'naive broadcast' : 'all-gather + reduce-scatter' };
    }
    // Token-level all-to-all: dispatch (fp8 + scales) and combine (bf16). With TP>1 the MoE input is
    // sequence-parallel across TP ranks.
    const Tr = T / Math.max(1, inst.tp);
    const remote = (inst.epSize - 1) / inst.epSize;
    const dispatch = Tr * moe.topK * h * (1 + 4 / 128) * remote;
    const combine = Tr * moe.topK * h * 2 * remote;
    const dedup = be === 'deepep_high_throughput' ? moe.topK : 0;
    const d = this.comm.allToAll(dispatch, this.epShape, dedup);
    const c = this.comm.allToAll(combine, this.epShape, dedup);
    return { t: d.t + c.t, intra: d.intra + c.intra, inter: d.inter + c.inter, algo: `${be} dispatch+combine` };
  }

  private draftTime(s: BatchShape, roof: Roof): number {
    const spec = this.inst.flags.speculative;
    if (!spec || s.decodeSeqs === 0) return 0;
    const N = s.decodeSeqs;
    const h = this.model.hidden;
    if (spec.method === 'ngram') return 0.1e-3;
    const lm = () => roof(2 * N * this.lmHeadParams, this.lmHeadBytes, this.lmHeadPeak, 2);
    if (this.draftLayer) {
      const g = this.draftLayer;
      let t = 0;
      for (let i = 0; i < spec.k; i++) {
        t += roof(2 * (g.attnProj.params + g.dense.params + g.shared.params) * N, g.attnProj.bytes + g.dense.bytes + g.shared.bytes, g.attnProj.peak, g.projKernels + 6);
        if (g.moe) t += this.moeTime(g.moe, N, roof);
        const ctx = s.decodeCtx[0] ?? 0;
        t += roof(ctx * g.flopsPerPairDecode, ctx * g.kvPerToken, this.attnPeak, 2);
        t += lm();
      }
      return t + roof(2 * N * 3 * h * h / this.inst.tp, (3 * h * h * 2) / this.inst.tp, this.lmHeadPeak);
    }
    // Independent draft model: k memory-bound forwards.
    let t = 0;
    for (let i = 0; i < spec.k; i++) t += roof(2 * N * this.draftParams, this.draftParams * 2, this.lmHeadPeak) + lm();
    return t;
  }
}
