import type { ClusterSpec, GpuSpec, ScaleUpKind } from '../types.ts';
import { gbitps } from '../units.ts';
import { gpuById } from './gpus.ts';

export interface ScaleOutOption {
  id: string;
  label: string;
  bwPerGpuDir: number;
  latency: number;
}

export const SCALE_OUT: ScaleOutOption[] = [
  { id: 'ib-hdr', label: 'InfiniBand HDR 200G / GPU', bwPerGpuDir: gbitps(200), latency: 1.5e-6 },
  { id: 'ib-ndr', label: 'InfiniBand NDR 400G / GPU (ConnectX-7)', bwPerGpuDir: gbitps(400), latency: 1.2e-6 },
  { id: 'roce-400', label: 'RoCE 400G / GPU', bwPerGpuDir: gbitps(400), latency: 2.4e-6 },
  { id: 'ib-xdr', label: 'InfiniBand XDR 800G / GPU (ConnectX-8)', bwPerGpuDir: gbitps(800), latency: 1.2e-6 },
];

export function scaleOutById(id: string): ScaleOutOption {
  return SCALE_OUT.find((s) => s.id === id) ?? SCALE_OUT[1]!;
}

export interface ClusterBuild {
  gpu: string;
  nodes: number;
  gpusPerNode: number;
  /** 0 = derive from the platform (8 for HGX, 72 for NVL72, node for PCIe/mesh). */
  domainSize?: number;
  scaleOut: string;
  oversub?: number;
  name?: string;
}

function defaultScaleUp(g: GpuSpec): ScaleUpKind {
  if (g.scaleUpBWDir === 0) return 'pcie';
  if (g.meshLinkBWDir) return 'mesh';
  return 'nvswitch';
}

export function buildCluster(b: ClusterBuild): ClusterSpec {
  const gpu = gpuById(b.gpu);
  const scaleUp = defaultScaleUp(gpu);
  const nvl72 = gpu.id === 'gb200' || gpu.id === 'gb300';
  const total = b.nodes * b.gpusPerNode;
  let domainSize = b.domainSize || (nvl72 ? Math.min(72, total) : b.gpusPerNode);
  domainSize = Math.max(1, Math.min(domainSize, total));
  const so = scaleOutById(b.scaleOut);
  const id = `${b.nodes}x${b.gpusPerNode}x${gpu.id}-${so.id}`;
  return {
    id,
    name:
      b.name ??
      (b.nodes === 1 ? `${b.gpusPerNode}× ${gpu.name}` : `${b.nodes} × ${b.gpusPerNode}× ${gpu.name} (${so.label.split(' /')[0]})`),
    gpu,
    nodes: b.nodes,
    gpusPerNode: b.gpusPerNode,
    domainSize,
    scaleUp,
    nvls: scaleUp === 'nvswitch' && gpu.arch !== 'sm80',
    scaleOut: { kind: so.label, bwPerGpuDir: so.bwPerGpuDir, latency: so.latency, oversub: b.oversub ?? 1 },
  };
}

export interface ClusterPreset {
  id: string;
  label: string;
  build: ClusterBuild;
}

export const CLUSTER_PRESETS: ClusterPreset[] = [
  { id: '1xh100', label: '1× H100', build: { gpu: 'h100', nodes: 1, gpusPerNode: 1, scaleOut: 'ib-ndr' } },
  { id: '8xh100', label: '8× H100 (HGX)', build: { gpu: 'h100', nodes: 1, gpusPerNode: 8, scaleOut: 'ib-ndr' } },
  { id: '8xh200', label: '8× H200 (HGX)', build: { gpu: 'h200', nodes: 1, gpusPerNode: 8, scaleOut: 'ib-ndr' } },
  {
    id: '4x8xh200',
    label: '4 × 8× H200 over IB NDR',
    build: { gpu: 'h200', nodes: 4, gpusPerNode: 8, scaleOut: 'ib-ndr' },
  },
  { id: '8xb200', label: '8× B200 (HGX)', build: { gpu: 'b200', nodes: 1, gpusPerNode: 8, scaleOut: 'ib-ndr' } },
  { id: '8xb300', label: '8× B300 (HGX)', build: { gpu: 'b300', nodes: 1, gpusPerNode: 8, scaleOut: 'ib-xdr' } },
  {
    id: '2x8xb300',
    label: '2 × 8× B300 over IB XDR',
    build: { gpu: 'b300', nodes: 2, gpusPerNode: 8, scaleOut: 'ib-xdr' },
  },
  {
    id: 'gb200-nvl72',
    label: 'GB200 NVL72 (18 trays × 4)',
    build: { gpu: 'gb200', nodes: 18, gpusPerNode: 4, scaleOut: 'ib-ndr', name: 'GB200 NVL72' },
  },
  {
    id: 'gb300-nvl72',
    label: 'GB300 NVL72 (18 trays × 4)',
    build: { gpu: 'gb300', nodes: 18, gpusPerNode: 4, scaleOut: 'ib-xdr', name: 'GB300 NVL72' },
  },
  { id: '8xa100', label: '8× A100 80GB (HGX)', build: { gpu: 'a100-80g', nodes: 1, gpusPerNode: 8, scaleOut: 'ib-hdr' } },
  { id: '8xmi300x', label: '8× MI300X', build: { gpu: 'mi300x', nodes: 1, gpusPerNode: 8, scaleOut: 'roce-400' } },
  { id: '8xmi355x', label: '8× MI355X', build: { gpu: 'mi355x', nodes: 1, gpusPerNode: 8, scaleOut: 'roce-400' } },
  { id: '4xl40s', label: '4× L40S (PCIe)', build: { gpu: 'l40s', nodes: 1, gpusPerNode: 4, scaleOut: 'roce-400' } },
  {
    id: '1xrtxpro6000',
    label: '1× RTX PRO 6000',
    build: { gpu: 'rtx-pro-6000', nodes: 1, gpusPerNode: 1, scaleOut: 'roce-400' },
  },
];

export function clusterPreset(id: string): ClusterSpec {
  const p = CLUSTER_PRESETS.find((c) => c.id === id);
  if (!p) throw new Error(`Unknown cluster preset ${id}`);
  return buildCluster(p.build);
}

export const totalGpus = (c: ClusterSpec): number => c.nodes * c.gpusPerNode;
export const domainOf = (c: ClusterSpec, gpu: number): number => Math.floor(gpu / c.domainSize);
export const nodeOf = (c: ClusterSpec, gpu: number): number => Math.floor(gpu / c.gpusPerNode);
