// Rank → GPU mapping and per-rank weight shards, following vLLM's sharding rules:
// rank = dp·(PP·TP) + pp·TP + tp (TP innermost); PP partitions from vllm/distributed/utils.get_pp_indices;
// MLA q_a/kv_a projections are ReplicatedLinear; experts are EP-placed or TP×DP-sharded.

import { BYTES_PER_PARAM } from '../model/quant.ts';
import type {
  ClusterSpec,
  CommGroup,
  GqaAttn,
  LayerSpec,
  ModelSpec,
  Placement,
  RankShard,
  ResolvedInstance,
  WeightBreakdown,
} from '../types.ts';
import { domainOf } from '../hardware/clusters.ts';

/** vLLM's PP layer partition: even split, remainder added to the stages before the last one. */
export function ppPartition(numLayers: number, pp: number): [number, number][] {
  const per = Math.floor(numLayers / pp);
  const parts = new Array<number>(pp).fill(per);
  const rem = numLayers % pp;
  for (let i = 2; i < rem + 2; i++) parts[pp - i]! += 1;
  const out: [number, number][] = [];
  let lo = 0;
  for (const n of parts) {
    out.push([lo, lo + n]);
    lo += n;
  }
  return out;
}

export function kvHeadsPerRank(nKV: number, tp: number): number {
  return Math.max(1, Math.floor(nKV / tp));
}

/** Per-rank attention params of one layer. */
export function attnParamsPerRank(l: LayerSpec, h: number, tp: number): number {
  const a = l.attn;
  if (a.kind === 'mla') {
    const replicated = (a.qLora > 0 ? h * a.qLora + a.qLora : 0) + h * (a.kvLora + a.rope) + a.kvLora;
    const indexer = a.dsa ? (a.qLora || h) * a.dsa.nIdx * a.dsa.dIdx + h * a.dsa.dIdx + 2 * a.dsa.dIdx + h * a.dsa.nIdx : 0;
    const sharded = l.p.attn - replicated - indexer;
    return replicated + indexer + sharded / tp;
  }
  if (a.kind === 'linear') return l.p.attn / tp;
  const kv = h * a.nKV * (a.dQK + a.dV);
  const qo = l.p.attn - kv;
  return qo / tp + (kv * kvHeadsPerRank(a.nKV, tp)) / a.nKV;
}

function emptyWeights(): WeightBreakdown {
  return { attn: 0, dense: 0, shared: 0, experts: 0, router: 0, norms: 0, embed: 0, lmHead: 0, draft: 0, total: 0 };
}

function expertRange(slots: number, ep: number, epRank: number): { lo: number; hi: number; count: number } {
  const base = Math.floor(slots / ep);
  const extra = slots % ep;
  const count = base + (epRank < extra ? 1 : 0);
  const lo = epRank * base + Math.min(epRank, extra);
  return { lo, hi: lo + count, count };
}

function group(kind: CommGroup['kind'], gpus: number[], cluster: ClusterSpec): CommGroup {
  const counts = new Map<number, number>();
  for (const g of gpus) {
    const d = domainOf(cluster, g);
    counts.set(d, (counts.get(d) ?? 0) + 1);
  }
  return { kind, gpus, domains: counts.size, perDomain: Math.max(...counts.values()) };
}

/** Draft-model parameters resident on each rank for speculative decoding. */
export function draftParamsPerRank(model: ModelSpec, inst: ResolvedInstance): number {
  const s = inst.flags.speculative;
  if (!s) return 0;
  if (s.method === 'mtp') {
    const l = model.mtp.layers[0];
    if (!l) return 0;
    const n = Math.min(model.mtp.layers.length, 1);
    const moe = l.ffn.kind === 'moe' ? l.ffn : null;
    const experts = moe ? (inst.ep ? Math.ceil(moe.E / inst.epSize) : moe.E / inst.moeTp) * l.p.expertEach : 0;
    return n * (attnParamsPerRank(l, model.hidden, inst.tp) + (l.p.dense + l.p.shared) / inst.tp + experts + l.p.router + l.p.norms) +
      model.mtp.extraParams / Math.max(1, model.mtp.layers.length);
  }
  if (s.method === 'ngram') return 0;
  if (s.method === 'eagle' || s.method === 'eagle3') {
    // One decoder layer + fusion projection; shares the target's embeddings/lm_head.
    const l = model.layers[0]!;
    const dense = l.ffn.kind === 'dense' ? l.p.dense : 3 * model.hidden * (l.ffn.eInter * Math.min(l.ffn.topK, 4));
    const own = s.draftParams || attnParamsPerRank(l, model.hidden, 1) + dense + 3 * model.hidden * model.hidden;
    return own / inst.tp;
  }
  return s.draftParams / inst.tp;
}

export function placeInstance(model: ModelSpec, inst: ResolvedInstance, cluster: ClusterSpec): Placement {
  const { tp, pp, dp } = inst;
  const h = model.hidden;
  const q = inst.quant;
  const B = BYTES_PER_PARAM;
  const stages = ppPartition(model.layers.length, pp);
  const shards: RankShard[] = [];
  const firstGqa = model.layers.find((l) => l.attn.kind === 'gqa')?.attn as GqaAttn | undefined;
  const firstAttnNQ = (() => {
    const a = model.layers.find((l) => l.attn.kind !== 'linear')?.attn;
    return a && a.kind !== 'linear' ? a.nQ : 1;
  })();
  const firstMoe = model.layers.find((l) => l.ffn.kind === 'moe')?.ffn;
  const slots = firstMoe && firstMoe.kind === 'moe' ? firstMoe.E + inst.flags.numRedundantExperts : 0;
  const draftParams = draftParamsPerRank(model, inst);

  for (let d = 0; d < dp; d++) {
    for (let s = 0; s < pp; s++) {
      for (let t = 0; t < tp; t++) {
        const rank = d * pp * tp + s * tp + t;
        const [lo, hi] = stages[s]!;
        const w = emptyWeights();
        const epRank = d * tp + t;
        const experts = inst.ep && slots ? expertRange(slots, inst.epSize, epRank) : null;
        for (let i = lo; i < hi; i++) {
          const l = model.layers[i]!;
          w.attn += attnParamsPerRank(l, h, tp) * B[q.attn];
          w.norms += l.p.norms * B.bf16;
          w.dense += (l.p.dense / tp) * B[q.dense];
          if (l.ffn.kind === 'moe') {
            w.shared += (l.p.shared / tp) * B[q.shared];
            w.router += l.p.router * B[q.router];
            const local = experts ? experts.count : l.ffn.E / inst.moeTp;
            w.experts += local * l.p.expertEach * B[q.experts];
          }
        }
        const isFirst = s === 0;
        const isLast = s === pp - 1;
        if (isFirst) w.embed = (model.params.embed / tp) * B[q.embed];
        if (isLast) {
          const lmParams = model.tied ? (pp > 1 ? model.params.embed : 0) : model.params.lmHead;
          w.lmHead = (lmParams / tp) * B[q.lmHead];
          w.norms += h * B.bf16;
          w.draft = draftParams * B[q.attn === 'bf16' ? 'bf16' : q.dense];
        }
        w.total = w.attn + w.dense + w.shared + w.experts + w.router + w.norms + w.embed + w.lmHead + w.draft;
        shards.push({
          rank,
          gpu: inst.gpuStart + rank,
          dpRank: d,
          ppRank: s,
          tpRank: t,
          layerLo: lo,
          layerHi: hi,
          embed: isFirst,
          lmHead: isLast,
          qHeads: firstAttnNQ / tp,
          kvHeads: firstGqa ? kvHeadsPerRank(firstGqa.nKV, tp) : 1,
          kvReplicated: firstGqa ? tp > firstGqa.nKV : tp > 1,
          experts,
          expertShard: inst.ep ? 1 : inst.moeTp,
          weights: w,
        });
      }
    }
  }

  const tpGroups: CommGroup[] = [];
  const epGroups: CommGroup[] = [];
  const ppLinks: { from: number; to: number }[] = [];
  const gpuOf = (d: number, s: number, t: number) => inst.gpuStart + d * pp * tp + s * tp + t;
  for (let d = 0; d < dp; d++) {
    for (let s = 0; s < pp; s++) {
      if (tp > 1) tpGroups.push(group('tp', Array.from({ length: tp }, (_, t) => gpuOf(d, s, t)), cluster));
      if (s < pp - 1) for (let t = 0; t < tp; t++) ppLinks.push({ from: gpuOf(d, s, t), to: gpuOf(d, s + 1, t) });
    }
  }
  if (firstMoe && dp > 1) {
    // MoE layers synchronize all DP ranks of a PP stage (EP all-to-all, or all-gather/reduce-scatter without EP).
    for (let s = 0; s < pp; s++) {
      const gpus: number[] = [];
      for (let d = 0; d < dp; d++) for (let t = 0; t < tp; t++) gpus.push(gpuOf(d, s, t));
      epGroups.push(group('ep', gpus, cluster));
    }
  }
  return { shards, tpGroups, epGroups, ppLinks };
}
