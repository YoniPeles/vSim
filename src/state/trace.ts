import { create } from 'zustand';
import type { TracePhase } from '../core/engine/cost.ts';

// Slow-motion playback of one forward step. Phase durations span µs (collectives) to ms (GEMMs),
// so wall time per phase is proportional to √duration: short phases stay visible, long ones longer.

export interface TraceRun {
  inst: number;
  phases: TracePhase[];
  /** Cumulative wall-clock offsets (s) of each phase start, plus the end. */
  wallAt: number[];
  realTotal: number;
  started: number;
}

interface TraceState {
  run: TraceRun | null;
  start: (inst: number, phases: TracePhase[]) => void;
  stop: () => void;
}

const WALL = 9;

export const useTrace = create<TraceState>()((set) => ({
  run: null,
  start: (inst, phases) => {
    const w = phases.map((p) => Math.sqrt(Math.max(1e-7, p.dur)));
    const sum = w.reduce((a, b) => a + b, 0);
    const wallAt = [0];
    for (const x of w) wallAt.push(wallAt[wallAt.length - 1]! + (x / sum) * WALL);
    set({ run: { inst, phases, wallAt, realTotal: phases.reduce((a, p) => a + p.dur, 0), started: performance.now() / 1000 } });
  },
  stop: () => set({ run: null }),
}));

/** Current phase index of a run (−1 when finished). */
export function phaseAt(run: TraceRun, nowSec: number): number {
  const t = nowSec - run.started;
  if (t >= run.wallAt[run.wallAt.length - 1]!) return -1;
  let lo = 0;
  let hi = run.phases.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (run.wallAt[mid]! <= t) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/** Real (simulated GPU) time elapsed at a phase index. */
export function realAt(run: TraceRun, idx: number): number {
  let s = 0;
  for (let i = 0; i < idx; i++) s += run.phases[i]!.dur;
  return s;
}
