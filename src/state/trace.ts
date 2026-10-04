import { create } from 'zustand';
import type { TracePhase } from '../core/engine/cost.ts';

// Slow-motion playback of one forward step. Phase durations span µs (collectives) to ms (GEMMs),
// so wall time per phase is proportional to √duration: short phases stay visible, long ones longer.
// The whole step is paced per layer (LAYER_WALL seconds at 1×) so the wavefront is easy to follow;
// the HUD offers pause, speed and scrubbing.

export interface TraceRun {
  inst: number;
  phases: TracePhase[];
  /** Cumulative wall-clock offsets (s, at 1×) of each phase start, plus the end. */
  wallAt: number[];
  realTotal: number;
  /** Playback position in wall seconds. Mutated by the HUD's ticker, not React state. */
  pos: number;
}

interface TraceState {
  run: TraceRun | null;
  speed: number;
  paused: boolean;
  start: (inst: number, phases: TracePhase[]) => void;
  stop: () => void;
  setSpeed: (s: number) => void;
  setPaused: (p: boolean) => void;
  seek: (pos: number) => void;
}

const LAYER_WALL = 0.55;

export const useTrace = create<TraceState>()((set, get) => ({
  run: null,
  speed: 1,
  paused: false,
  start: (inst, phases) => {
    const w = phases.map((p) => Math.sqrt(Math.max(1e-7, p.dur)));
    const sum = w.reduce((a, b) => a + b, 0);
    const layers = new Set(phases.filter((p) => p.layer >= 0).map((p) => p.layer)).size;
    const total = Math.max(6, layers * LAYER_WALL + 1);
    const wallAt = [0];
    for (const x of w) wallAt.push(wallAt[wallAt.length - 1]! + (x / sum) * total);
    set({ run: { inst, phases, wallAt, realTotal: phases.reduce((a, p) => a + p.dur, 0), pos: 0 }, paused: false });
  },
  stop: () => set({ run: null, paused: false }),
  setSpeed: (speed) => set({ speed }),
  setPaused: (paused) => {
    const run = get().run;
    // Playing again from the end restarts the trace.
    if (!paused && run && run.pos >= traceEnd(run)) run.pos = 0;
    set({ paused });
  },
  seek: (pos) => {
    const run = get().run;
    if (run) run.pos = Math.max(0, Math.min(traceEnd(run), pos));
  },
}));

export const traceEnd = (run: TraceRun): number => run.wallAt[run.wallAt.length - 1]!;

/** Index of the phase at the current playback position (the last phase once finished). */
export function phaseAt(run: TraceRun): number {
  const t = Math.min(run.pos, traceEnd(run) - 1e-9);
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

/** Phase index whose real-time span contains `real` seconds into the step. */
export function phaseAtReal(run: TraceRun, real: number): number {
  let s = 0;
  for (let i = 0; i < run.phases.length; i++) {
    s += run.phases[i]!.dur;
    if (s >= real) return i;
  }
  return run.phases.length - 1;
}
