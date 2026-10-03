// Evaluation → per-GPU visuals (HBM segments, layer slices, expert stripes) and traffic links.
// Pure functions; the React/three components only turn these into instanced buffers.

import type { InstanceView } from '../state/derived.ts';
import type { CommKind, ModelSpec, RankShard } from '../core/types.ts';
import { bytesPerRequest, kvProfile } from '../core/engine/memory.ts';
import { TOWER_H, type Layout, type Vec3 } from './layout.ts';

export interface MemSegments {
  weights: number;
  activations: number;
  overhead: number;
  kvUsed: number;
  kvFree: number;
  unreserved: number;
}

export interface GpuVisual {
  gpu: number;
  active: boolean;
  instance: number;
  role: 'mixed' | 'prefill' | 'decode';
  shard: RankShard | null;
  /** Fractions of total memory. */
  mem: MemSegments;
  memBytes: MemSegments;
}

export interface PlateVisual {
  gpu: number;
  inst: number;
  stage: number;
  layer: number;
  y: number;
  h: number;
  /** Slice in tile-local x (−0.5..0.5). */
  x0: number;
  x1: number;
  attn: 'full' | 'local' | 'linear';
  moe: boolean;
  /** Whole experts resident (EP) → one stripe each; −1 = every expert TP-sharded (hatched). */
  stripes: number;
  /** Global id of the first resident expert (EP). */
  expertLo: number;
}

/** Footprint of the layer stack inside a GPU tile (tile-local x/z). */
export const STACK = { x0: -0.1, x1: 0.44, z0: -0.4, z1: 0.4 };
/** Footprint of the HBM tower inside a GPU tile. */
export const TOWER = { x0: -0.44, x1: -0.18, z0: -0.4, z1: 0.4 };

export interface LinkVisual {
  kind: CommKind;
  /** Which instance's traffic this link carries, and whether it crosses the network. */
  inst: number;
  inter: boolean;
  from: Vec3;
  ctrl: Vec3;
  to: Vec3;
  /** 0..1+ utilization of the link from the analytical steady state (drives particle count). */
  util: number;
}

export function gpuVisuals(views: InstanceView[], total: number, totalMem: number): GpuVisual[] {
  const out: GpuVisual[] = [];
  const empty: MemSegments = { weights: 0, activations: 0, overhead: 0, kvUsed: 0, kvFree: 0, unreserved: 1 };
  for (let g = 0; g < total; g++) {
    out.push({ gpu: g, active: false, instance: -1, role: 'mixed', shard: null, mem: { ...empty }, memBytes: { ...empty, unreserved: totalMem } });
  }
  views.forEach((v, idx) => {
    const { ev, steady } = v;
    const model = v.cost?.model;
    ev.placement.shards.forEach((sh, i) => {
      if (sh.gpu >= total) return;
      const m = ev.memory.perRank[i]!;
      const kvUsedBytes = steady && model ? kvInUse(v, i) : 0;
      const bytes: MemSegments = {
        weights: m.weights,
        activations: m.actPeak,
        overhead: m.nonTorch + m.persistent + m.cudagraph,
        kvUsed: kvUsedBytes,
        kvFree: Math.max(0, m.kv - kvUsedBytes),
        unreserved: Math.max(0, m.total - m.requested),
      };
      const sum = Object.values(bytes).reduce((a, b) => a + b, 0);
      const scale = Math.max(m.total, sum);
      out[sh.gpu] = {
        gpu: sh.gpu,
        active: true,
        instance: idx,
        role: ev.inst.role,
        shard: sh,
        memBytes: bytes,
        mem: Object.fromEntries(Object.entries(bytes).map(([k, b]) => [k, b / scale])) as unknown as MemSegments,
      };
    });
  });
  return out;
}

/** KV bytes held on rank `i` by the steady-state running batch (average live context). */
export function kvInUse(v: InstanceView, i: number): number {
  const model = v.cost?.model;
  const sh = v.ev.placement.shards[i];
  if (!model || !v.steady || !sh) return 0;
  const prof = kvProfile(model, v.ev.inst, sh.layerLo, sh.layerHi);
  const per = bytesPerRequest(prof, Math.round(v.ctxAvg), v.ev.inst.blockSize);
  return Math.min(v.ev.memory.perRank[i]!.kv, per * v.steady.batch);
}

export function plateVisuals(model: ModelSpec, gpus: GpuVisual[], views: InstanceView[]): PlateVisual[] {
  const L = model.layers.length;
  const step = (TOWER_H - 0.1) / L;
  const out: PlateVisual[] = [];
  for (const g of gpus) {
    const sh = g.shard;
    if (!sh) continue;
    const inst = views[g.instance]!.ev.inst;
    const w = (STACK.x1 - STACK.x0) / inst.tp;
    const x0 = STACK.x0 + sh.tpRank * w;
    for (let i = sh.layerLo; i < sh.layerHi; i++) {
      const l = model.layers[i]!;
      const a = l.attn;
      const attn = a.kind === 'linear' ? 'linear' : a.kind === 'gqa' && a.scope !== 'full' ? 'local' : 'full';
      const moe = l.ffn.kind === 'moe';
      const stripes = !moe ? 0 : sh.experts ? sh.experts.count : -1;
      out.push({ gpu: g.gpu, inst: g.instance, stage: sh.ppRank, layer: i, y: 0.14 + i * step, h: step * 0.62, x0, x1: x0 + w * (inst.tp > 1 ? 0.9 : 1), attn, moe, stripes, expertLo: sh.experts?.lo ?? 0 });
    }
  }
  return out;
}

/** Traffic links with utilization from the steady-state step. */
export function linkVisuals(layout: Layout, views: InstanceView[], kvxUtil: number): LinkVisual[] {
  const out: LinkVisual[] = [];
  const slot = (g: number) => layout.gpus[g];
  const spineY = layout.spine ? layout.spine.center.y : TOWER_H + 1.4;
  const spineZ = layout.spine ? layout.spine.center.z : 0;
  views.forEach((v, inst) => {
    const s = v.steady;
    if (!s || !v.cost) return;
    const inflight = v.ev.inst.pp > 1 ? v.ev.inst.pp : 1;
    const interval = v.ev.inst.pp > 1 ? Math.max(...s.step.stages, s.step.host) : s.step.time;
    const suBW = v.cost.comm.suBW / v.cost.calib.etaScaleUp;
    const soBW = v.cost.comm.soBW / v.cost.calib.etaScaleOut;
    const util = (bytes: number, bw: number) => (bw > 0 ? (bytes * inflight) / interval / bw : 0);
    const c = s.step.comm;
    for (const sh of v.ev.placement.shards) {
      const p = slot(sh.gpu);
      if (!p) continue;
      const tpU = util(c.tp.intra, suBW);
      const epU = util(c.ep.intra, suBW);
      const interU = util(c.tp.inter + c.ep.inter, soBW);
      if (layout.meshEdges.length === 0 && layout.switches.length) {
        const top = lift(p.pos, TOWER_H + 0.05);
        const mid = { x: (top.x + p.switchPort.x) / 2, y: (top.y + p.switchPort.y) / 2, z: (top.z + p.switchPort.z) / 2 };
        const base = { inst, inter: false, ctrl: mid };
        if (tpU > 0) out.push({ ...base, kind: 'tp', from: top, to: p.switchPort, util: tpU });
        if (tpU > 0) out.push({ ...base, kind: 'tp', from: p.switchPort, to: top, util: tpU });
        if (epU > 0) out.push({ ...base, kind: 'ep', from: p.switchPort, to: top, util: epU });
        if (epU > 0) out.push({ ...base, kind: 'ep', from: top, to: p.switchPort, util: epU });
      }
      if (interU > 0) {
        const top = { x: p.nic.x, y: spineY, z: spineZ };
        const ctrl = { x: p.nic.x, y: spineY * 0.7, z: (p.nic.z + spineZ) / 2 };
        out.push({ kind: c.ep.inter > c.tp.inter ? 'ep' : 'tp', inst, inter: true, from: p.nic, ctrl, to: top, util: interU });
        out.push({ kind: c.ep.inter > c.tp.inter ? 'ep' : 'tp', inst, inter: true, from: top, ctrl, to: p.nic, util: interU });
      }
    }
    if (layout.meshEdges.length) {
      const tpU = util(c.tp.intra, suBW) / 7;
      const epU = util(c.ep.intra, suBW) / 7;
      const active = new Set(v.ev.placement.shards.map((x) => x.gpu));
      for (const [a, b] of layout.meshEdges) {
        if (!active.has(a) || !active.has(b)) continue;
        const pa = lift(slot(a)!.pos, 0.18);
        const pb = lift(slot(b)!.pos, 0.18);
        const mid = { x: (pa.x + pb.x) / 2, y: 0.5, z: (pa.z + pb.z) / 2 };
        if (tpU > 0) out.push({ kind: 'tp', inst, inter: false, from: pa, ctrl: mid, to: pb, util: tpU });
        if (epU > 0) out.push({ kind: 'ep', inst, inter: false, from: pb, ctrl: mid, to: pa, util: epU });
      }
    }
    // Pipeline hand-offs arc over the boards from stage s to s+1.
    const ppU = util(c.pp.intra + c.pp.inter, c.pp.inter > 0 ? soBW : suBW);
    for (const l of v.ev.placement.ppLinks) {
      const a = slot(l.from);
      const b = slot(l.to);
      if (!a || !b) continue;
      const from = lift(a.pos, TOWER_H * 0.9);
      const to = lift(b.pos, TOWER_H * 0.2);
      const ctrl = { x: (from.x + to.x) / 2, y: TOWER_H + 1.2, z: (from.z + to.z) / 2 };
      out.push({ kind: 'pp', inst, inter: c.pp.inter > 0, from, ctrl, to, util: Math.max(0.05, ppU) });
    }
  });
  // Disaggregated prefill → decode KV transfer.
  if (views.length > 1 && kvxUtil > 0) {
    const pre = views[0]!.ev.placement.shards;
    const dec = views[1]!.ev.placement.shards;
    pre.forEach((sh, i) => {
      const a = slot(sh.gpu);
      const b = slot(dec[i % dec.length]!.gpu);
      if (!a || !b) return;
      const from = lift(a.pos, 0.3);
      const to = lift(b.pos, 0.3);
      const ctrl = { x: (from.x + to.x) / 2, y: TOWER_H + 2.2, z: (from.z + to.z) / 2 - 1 };
      out.push({ kind: 'kvx', inst: 0, inter: true, from, ctrl, to, util: kvxUtil });
    });
  }
  return out;
}

const lift = (p: Vec3, dy: number): Vec3 => ({ x: p.x, y: p.y + dy, z: p.z });

/** Particle density (0..1) for a link at a given utilization. */
export function intensity(util: number): number {
  return util > 0 ? Math.min(1, 0.12 + Math.sqrt(util) * 1.1) : 0;
}

/** Live utilization of a link from a simulation frame's per-instance link rates. */
export function liveUtil(l: LinkVisual, rates: { tpIntra: number; epIntra: number; ppIntra: number; inter: number; kvx: number } | undefined): number {
  if (!rates) return 0;
  if (l.kind === 'kvx') return rates.kvx;
  if (l.inter) return rates.inter;
  return l.kind === 'tp' ? rates.tpIntra : l.kind === 'ep' ? rates.epIntra : rates.ppIntra;
}

/** Simulation replica index (global) for each GPU, or −1. */
export function gpuReplicas(views: InstanceView[], total: number): Int32Array {
  const out = new Int32Array(total).fill(-1);
  let offset = 0;
  for (const v of views) {
    for (const sh of v.ev.placement.shards) if (sh.gpu < total) out[sh.gpu] = offset + sh.dpRank;
    offset += v.ev.inst.dp;
  }
  return out;
}
