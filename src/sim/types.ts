import type { ClusterBuild } from '../core/hardware/clusters.ts';
import type { PresetSnapshot } from '../core/model/presets.ts';
import type { Workload } from '../core/engine/analytic.ts';
import type { Deployment } from '../core/types.ts';

export interface SimWorkload extends Workload {
  /** Coefficient of variation of prompt/output lengths (log-normal). */
  spread: number;
  arrival: 'closed' | 'poisson';
  /** Requests/s for Poisson arrivals. */
  rate: number;
}

export interface SimInputs {
  snapshot: PresetSnapshot;
  cluster: ClusterBuild;
  deployment: Deployment;
  workload: SimWorkload;
  routingSkew: number;
  seed: number;
}

export interface Percentiles {
  p50: number;
  p90: number;
  p99: number;
}

export interface ReplicaFrame {
  instance: number;
  dp: number;
  running: number;
  waiting: number;
  kvUsed: number;
  kvTotal: number;
  /** Tokens in the most recent step. */
  decodeTokens: number;
  prefillTokens: number;
  stepTime: number;
}

export interface LinkRates {
  /** Utilization (0..1+) per GPU, averaged over the frame interval. */
  tpIntra: number;
  epIntra: number;
  ppIntra: number;
  inter: number;
  kvx: number;
}

export interface SimFrame {
  t: number;
  wallRatio: number;
  replicas: ReplicaFrame[];
  links: LinkRates[];
  stats: {
    outTokPerSec: number;
    reqPerSec: number;
    ttft: Percentiles;
    itl: Percentiles;
    e2e: Percentiles;
    completed: number;
    preemptions: number;
    prefixHitRate: number;
    running: number;
    waiting: number;
    transferring: number;
  };
  /** KV block owners of the focused replica: 0 free, 1 free but cached, 2 shared, else 3 + request colour. */
  kvMap: { replica: number; blocks: Uint16Array } | null;
  /** Expert load of a watched MoE layer per resident expert (0..1), for the focused replica's GPU. */
  expertLoad: Float32Array | null;
}

export type ToWorker =
  | { type: 'init'; inputs: SimInputs }
  | { type: 'play' }
  | { type: 'pause' }
  | { type: 'speed'; speed: number }
  | { type: 'focus'; replica: number | null }
  | { type: 'workload'; workload: SimWorkload }
  | { type: 'fastForward'; seconds: number };

export type FromWorker =
  | { type: 'ready'; replicas: number; blocksPerReplica: number[] }
  | { type: 'frame'; frame: SimFrame }
  | { type: 'error'; message: string };
