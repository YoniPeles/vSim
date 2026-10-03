// α–β collective model. For each collective we evaluate the algorithms vLLM/NCCL could pick
// (custom one-shot/two-shot all-reduce, NVLS, ring, hierarchical across NVLink domains) and take
// the cheapest, like NCCL's tuner. Returns time plus bytes moved per rank on scale-up vs scale-out.

import type { Calibration, ClusterSpec, GpuArch } from '../types.ts';
import { KiB, MiB } from '../units.ts';

export interface CommResult {
  t: number;
  intra: number;
  inter: number;
  algo: string;
}

export const ZERO_COMM: CommResult = { t: 0, intra: 0, inter: 0, algo: '-' };

export interface GroupShape {
  n: number;
  /** Members per scale-up domain. */
  perDomain: number;
  /** Number of domains spanned. */
  domains: number;
}

// vllm/distributed/device_communicators/all_reduce_utils.py custom all-reduce caps (bytes) by TP size.
const CUSTOM_AR_CAP: Partial<Record<GpuArch, Record<number, number>>> = {
  sm90: { 2: 64 * MiB, 4: 32 * MiB, 6: 512 * KiB, 8: 256 * KiB },
  sm100: { 2: 2 * MiB, 4: 2 * MiB, 6: 1 * MiB, 8: 1 * MiB },
  sm103: { 2: 4 * MiB, 4: 4 * MiB, 6: 8 * MiB, 8: 4 * MiB },
};

export class CommModel {
  readonly suBW: number; // scale-up bytes/s per GPU per direction (effective)
  readonly soBW: number; // scale-out bytes/s per GPU per direction (effective)
  readonly linkBW: number; // mesh: one peer link
  readonly cluster: ClusterSpec;
  readonly calib: Calibration;
  constructor(cluster: ClusterSpec, calib: Calibration) {
    this.cluster = cluster;
    this.calib = calib;
    const g = cluster.gpu;
    const su = cluster.scaleUp === 'pcie' ? g.pcieBWDir * 0.8 : g.scaleUpBWDir;
    this.suBW = su * calib.etaScaleUp;
    this.linkBW = (g.meshLinkBWDir ?? su) * calib.etaScaleUp;
    this.soBW = (cluster.scaleOut.bwPerGpuDir * calib.etaScaleOut) / cluster.scaleOut.oversub;
  }

  private customArCap(n: number): number {
    if (this.cluster.scaleUp !== 'nvswitch' && n > 2) return 0;
    const caps = CUSTOM_AR_CAP[this.cluster.gpu.arch];
    return caps?.[n] ?? (n <= 8 ? 8 * MiB : 0);
  }

  /** All-reduce of m bytes inside one scale-up domain over p ranks. */
  private arDomain(m: number, p: number): CommResult {
    if (p <= 1) return ZERO_COMM;
    const c = this.calib;
    const ringBytes = (2 * (p - 1) * m) / p;
    const cands: CommResult[] = [];
    if (this.cluster.scaleUp === 'mesh') {
      // Full mesh: every peer pair has its own link.
      cands.push({ t: c.alphaScaleUp + m / this.linkBW, intra: (p - 1) * m, inter: 0, algo: 'mesh one-shot' });
      cands.push({ t: 2 * c.alphaScaleUp + (2 * m) / (p * this.linkBW), intra: ringBytes, inter: 0, algo: 'mesh two-shot' });
    } else {
      const cap = this.customArCap(p);
      if (m <= cap) {
        cands.push({ t: c.alphaScaleUp + ((p - 1) * m) / this.suBW, intra: (p - 1) * m, inter: 0, algo: 'custom AR one-shot' });
        cands.push({ t: 2 * c.alphaScaleUp + ringBytes / this.suBW, intra: ringBytes, inter: 0, algo: 'custom AR two-shot' });
      }
      if (this.cluster.nvls && p > 2) {
        cands.push({ t: 1.5 * c.alphaScaleUp + (1.1 * m) / this.suBW, intra: m, inter: 0, algo: 'NCCL NVLS' });
      }
      cands.push({
        t: 2 * (p - 1) * (c.alphaScaleUp * 0.5) + ringBytes / this.suBW,
        intra: ringBytes,
        inter: 0,
        algo: 'NCCL ring',
      });
    }
    return cands.reduce((a, b) => (b.t < a.t ? b : a));
  }

  allReduce(m: number, g: GroupShape): CommResult {
    if (g.n <= 1 || m <= 0) return ZERO_COMM;
    if (g.domains <= 1) return this.arDomain(m, g.n);
    // Hierarchical: reduce-scatter in domain, ring all-reduce of m/p across domains (rail-parallel NICs), all-gather in domain.
    const p = g.perDomain;
    const D = g.domains;
    const c = this.calib;
    const intraRS = p > 1 ? ((p - 1) / p) * m : 0;
    const shard = m / p;
    const interBytes = (2 * (D - 1) * shard) / D;
    const tIntra = p > 1 ? 2 * c.alphaScaleUp + (2 * intraRS) / this.suBW : 0;
    const tInter = 2 * (D - 1) * c.alphaScaleOut * 0.5 + interBytes / this.soBW;
    return { t: tIntra + tInter, intra: 2 * intraRS, inter: interBytes, algo: `hierarchical (${D} domains)` };
  }

  /** All-gather where every rank ends up with `total` bytes (each contributes total/n). */
  allGather(total: number, g: GroupShape): CommResult {
    if (g.n <= 1 || total <= 0) return ZERO_COMM;
    const c = this.calib;
    if (g.domains <= 1) {
      const b = ((g.n - 1) / g.n) * total;
      const bw = this.cluster.scaleUp === 'mesh' ? this.linkBW * (g.n - 1) : this.suBW;
      return { t: c.alphaScaleUp + b / bw, intra: b, inter: 0, algo: 'all-gather' };
    }
    const D = g.domains;
    const p = g.perDomain;
    const inter = ((D - 1) / D) * (total / p);
    const intra = ((p - 1) / p) * total;
    const t = c.alphaScaleOut + inter / this.soBW + (p > 1 ? c.alphaScaleUp + intra / this.suBW : 0);
    return { t, intra, inter, algo: `hierarchical all-gather (${D} domains)` };
  }

  /**
   * All-to-all: each rank sends `bytes` spread uniformly over the other n−1 ranks. `dedupK` models
   * DeepEP high-throughput node-level deduplication: a token routed to k experts crosses to each
   * remote domain at most once.
   */
  allToAll(bytes: number, g: GroupShape, dedupK = 0): CommResult {
    if (g.n <= 1 || bytes <= 0) return ZERO_COMM;
    const c = this.calib;
    const intraFrac = (g.perDomain - 1) / (g.n - 1);
    let intra = bytes * intraFrac;
    let inter = bytes * (1 - intraFrac);
    if (dedupK > 0 && g.domains > 1) {
      const D = g.domains;
      const perTokenRemote = (D - 1) * (1 - (1 - 1 / D) ** dedupK);
      const naiveRemote = dedupK * (1 - intraFrac);
      if (naiveRemote > 0) {
        const f = perTokenRemote / naiveRemote;
        intra += inter * (1 - f);
        inter *= f;
      }
    }
    const bwIntra = this.cluster.scaleUp === 'mesh' ? this.linkBW * Math.max(1, g.perDomain - 1) : this.suBW;
    const t = c.alphaA2A + (inter > 0 ? c.alphaScaleOut : 0) + Math.max(intra / bwIntra, inter / this.soBW);
    return { t, intra, inter, algo: dedupK ? 'DeepEP (node-dedup)' : 'all-to-all' };
  }

  p2p(bytes: number, sameDomain: boolean): CommResult {
    if (bytes <= 0) return ZERO_COMM;
    const c = this.calib;
    return sameDomain
      ? { t: c.alphaP2P + bytes / this.suBW, intra: bytes, inter: 0, algo: 'p2p NVLink' }
      : { t: c.alphaP2P + c.alphaScaleOut + bytes / this.soBW, intra: 0, inter: bytes, algo: 'p2p network' };
  }
}
