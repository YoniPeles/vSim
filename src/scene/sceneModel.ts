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
  /** Bottom of the wafer (world y) and its thickness. */
  y: number;
  h: number;
  /** This GPU's tensor-parallel slice of the layer, as a fraction of the layer width (0..1). */
  tpLo: number;
  tpHi: number;
  attn: 'full' | 'local' | 'linear';
  moe: boolean;
  /** Expert FFN held as whole experts (EP): the FFN half spans the full wafer, one cell per expert. */
  ffnWhole: boolean;
  /** Whole experts resident (EP) → one cell each; −1 = every expert TP-sharded (hatched); 0 = dense. */
  cells: number;
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
  /** GPU at the near end, for links that attach to a scale-up switch. */
  gpu: number;
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

/** Vertical pitch between adjacent layers in the stack, and the y of a layer's bottom. */
export const layerStep = (layers: number) => (TOWER_H - 0.1) / Math.max(1, layers);
export const layerY = (layer: number, layers: number) => 0.14 + layer * layerStep(layers);

export function plateVisuals(model: ModelSpec, gpus: GpuVisual[], views: InstanceView[]): PlateVisual[] {
  const L = model.layers.length;
  const step = layerStep(L);
  const out: PlateVisual[] = [];
  for (const g of gpus) {
    const sh = g.shard;
    if (!sh) continue;
    const inst = views[g.instance]!.ev.inst;
    const tpLo = sh.tpRank / inst.tp;
    const tpHi = (sh.tpRank + 1) / inst.tp;
    for (let i = sh.layerLo; i < sh.layerHi; i++) {
      const l = model.layers[i]!;
      const a = l.attn;
      const attn = a.kind === 'linear' ? 'linear' : a.kind === 'gqa' && a.scope !== 'full' ? 'local' : 'full';
      const moe = l.ffn.kind === 'moe';
      const cells = !moe ? 0 : sh.experts ? sh.experts.count : -1;
      out.push({
        gpu: g.gpu,
        inst: g.instance,
        stage: sh.ppRank,
        layer: i,
        y: layerY(i, L),
        h: Math.max(0.005, step * 0.3),
        tpLo,
        tpHi,
        attn,
        moe,
        ffnWhole: cells > 0,
        cells,
        expertLo: sh.experts?.lo ?? 0,
      });
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
  const stackX = (STACK.x0 + STACK.x1) / 2;
  const towerX = (TOWER.x0 + TOWER.x1) / 2;
  views.forEach((v, inst) => {
    const s = v.steady;
    if (!s || !v.cost) return;
    const inflight = v.ev.inst.pp > 1 ? v.ev.inst.pp : 1;
    const interval = v.ev.inst.pp > 1 ? Math.max(...s.step.stages, s.step.host) : s.step.time;
    const suBW = v.cost.comm.suBW / v.cost.calib.etaScaleUp;
    const soBW = v.cost.comm.soBW / v.cost.calib.etaScaleOut;
    const util = (bytes: number, bw: number) => (bw > 0 ? (bytes * inflight) / interval / bw : 0);
    const c = s.step.comm;
    const L = v.cost.model.layers.length;
    for (const sh of v.ev.placement.shards) {
      const p = slot(sh.gpu);
      if (!p) continue;
      const tpU = util(c.tp.intra, suBW);
      const epU = util(c.ep.intra, suBW);
      const interU = util(c.tp.inter + c.ep.inter, soBW);
      if (layout.meshEdges.length === 0 && p.switchIdx >= 0) {
        // Four lanes per spoke (TP and MoE, each way), side by side across the spoke's direction.
        const top = { x: p.pos.x + stackX, y: TOWER_H + 0.05, z: p.pos.z };
        const port = p.switchPort;
        let px = -(port.z - top.z);
        let pz = port.x - top.x;
        const n = Math.hypot(px, pz);
        if (n < 1e-3) [px, pz] = [1, 0];
        else [px, pz] = [px / n, pz / n];
        const lane = (off: number, a: Vec3, b: Vec3) => {
          const from = { x: a.x + px * off, y: a.y, z: a.z + pz * off };
          const to = { x: b.x + px * off, y: b.y, z: b.z + pz * off };
          return { from, to, ctrl: { x: (from.x + to.x) / 2, y: (from.y + to.y) / 2, z: (from.z + to.z) / 2 } };
        };
        const base = { inst, inter: false, gpu: sh.gpu };
        if (tpU > 0) out.push({ ...base, kind: 'tp', ...lane(-0.05, top, port), util: tpU });
        if (tpU > 0) out.push({ ...base, kind: 'tp', ...lane(-0.022, port, top), util: tpU });
        if (epU > 0) out.push({ ...base, kind: 'ep', ...lane(0.022, port, top), util: epU });
        if (epU > 0) out.push({ ...base, kind: 'ep', ...lane(0.05, top, port), util: epU });
      }
      if (interU > 0) {
        const top = { x: p.nic.x, y: spineY, z: spineZ };
        const ctrl = { x: p.nic.x, y: spineY * 0.7, z: (p.nic.z + spineZ) / 2 };
        const kind = c.ep.inter > c.tp.inter ? 'ep' : 'tp';
        out.push({ kind, inst, inter: true, gpu: sh.gpu, from: p.nic, ctrl, to: top, util: interU });
        out.push({ kind, inst, inter: true, gpu: sh.gpu, from: top, ctrl, to: p.nic, util: interU });
      }
    }
    if (layout.meshEdges.length) {
      const tpU = util(c.tp.intra, suBW) / 7;
      const epU = util(c.ep.intra, suBW) / 7;
      const active = new Set(v.ev.placement.shards.map((x) => x.gpu));
      for (const [a, b] of layout.meshEdges) {
        if (!active.has(a) || !active.has(b)) continue;
        const [pa, pb] = meshEnds(slot(a)!.pos, slot(b)!.pos);
        const mid = { x: (pa.x + pb.x) / 2, y: 0.5, z: (pa.z + pb.z) / 2 };
        if (tpU > 0) out.push({ kind: 'tp', inst, inter: false, gpu: a, from: pa, ctrl: mid, to: pb, util: tpU });
        if (epU > 0) out.push({ kind: 'ep', inst, inter: false, gpu: b, from: pb, ctrl: mid, to: pa, util: epU });
      }
    }
    // Pipeline hand-offs: activations leave the top of stage s's layers and enter the bottom of stage s+1's.
    const ppU = util(c.pp.intra + c.pp.inter, c.pp.inter > 0 ? soBW : suBW);
    const shardOf = new Map(v.ev.placement.shards.map((x) => [x.gpu, x]));
    for (const l of v.ev.placement.ppLinks) {
      const a = slot(l.from);
      const b = slot(l.to);
      const sa = shardOf.get(l.from);
      const sb = shardOf.get(l.to);
      if (!a || !b || !sa || !sb) continue;
      const from = { x: a.pos.x + stackX, y: layerY(sa.layerHi, L), z: a.pos.z };
      const to = { x: b.pos.x + stackX, y: layerY(sb.layerLo, L), z: b.pos.z };
      const ctrl = { x: (from.x + to.x) / 2, y: Math.max(from.y, to.y) + 0.9, z: (from.z + to.z) / 2 };
      out.push({ kind: 'pp', inst, inter: c.pp.inter > 0, gpu: l.from, from, ctrl, to, util: Math.max(0.05, ppU) });
    }
  });
  // Disaggregated prefill → decode KV transfer: from one HBM column to another.
  if (views.length > 1 && kvxUtil > 0) {
    const pre = views[0]!.ev.placement.shards;
    const dec = views[1]!.ev.placement.shards;
    pre.forEach((sh, i) => {
      const a = slot(sh.gpu);
      const b = slot(dec[i % dec.length]!.gpu);
      if (!a || !b) return;
      const from = { x: a.pos.x + towerX, y: 0.12 + TOWER_H + 0.04, z: a.pos.z };
      const to = { x: b.pos.x + towerX, y: 0.12 + TOWER_H + 0.04, z: b.pos.z };
      const ctrl = { x: (from.x + to.x) / 2, y: TOWER_H + 2.2, z: (from.z + to.z) / 2 - 1 };
      out.push({ kind: 'kvx', inst: 0, inter: true, gpu: sh.gpu, from, ctrl, to, util: kvxUtil });
    });
  }
  return out;
}

/** Ends of a GPU-to-GPU mesh link: on each package's edge, facing the peer. */
export function meshEnds(a: Vec3, b: Vec3): [Vec3, Vec3] {
  const dx = b.x - a.x;
  const dz = b.z - a.z;
  const n = Math.hypot(dx, dz) || 1;
  const r = 0.44;
  return [
    { x: a.x + (dx / n) * r, y: 0.16, z: a.z + (dz / n) * r },
    { x: b.x - (dx / n) * r, y: 0.16, z: b.z - (dz / n) * r },
  ];
}

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
