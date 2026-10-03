import { create } from 'zustand';
import type { FromWorker, SimFrame, SimInputs, SimWorkload, ToWorker } from '../sim/types.ts';

export interface Series {
  t: number[];
  tok: number[];
  running: number[];
  waiting: number[];
  kv: number[];
}

const MAX_POINTS = 900;

interface SimState {
  status: 'idle' | 'running' | 'paused' | 'error';
  error: string | null;
  speed: number;
  frame: SimFrame | null;
  series: Series;
  /** Bumped on every frame so subscribers can cheaply detect change. */
  version: number;
  start: (inputs: SimInputs) => void;
  pause: () => void;
  resume: () => void;
  stop: () => void;
  setSpeed: (s: number) => void;
  focus: (replica: number | null) => void;
  updateWorkload: (w: SimWorkload) => void;
  fastForward: (seconds: number) => void;
}

let worker: Worker | null = null;
let focused: number | null = null;

const emptySeries = (): Series => ({ t: [], tok: [], running: [], waiting: [], kv: [] });

function send(m: ToWorker) {
  worker?.postMessage(m);
}

export const useSim = create<SimState>()((set, get) => ({
  status: 'idle',
  error: null,
  speed: 1,
  frame: null,
  series: emptySeries(),
  version: 0,

  start: (inputs) => {
    worker?.terminate();
    worker = new Worker(new URL('../sim/worker.ts', import.meta.url), { type: 'module' });
    set({ status: 'running', error: null, frame: null, series: emptySeries() });
    worker.onmessage = (ev: MessageEvent<FromWorker>) => {
      const m = ev.data;
      if (m.type === 'error') {
        set({ status: 'error', error: m.message });
        worker?.terminate();
        worker = null;
      } else if (m.type === 'frame') {
        const f = m.frame;
        const s = get().series;
        const last = s.t[s.t.length - 1] ?? -1;
        if (f.t > last + 0.05) {
          let used = 0;
          let total = 0;
          for (const r of f.replicas) {
            used += r.kvUsed;
            total += r.kvTotal;
          }
          s.t.push(f.t);
          s.tok.push(f.stats.outTokPerSec);
          s.running.push(f.stats.running);
          s.waiting.push(f.stats.waiting);
          s.kv.push(total ? (100 * used) / total : 0);
          if (s.t.length > MAX_POINTS) for (const k of Object.keys(s) as (keyof Series)[]) s[k].splice(0, s[k].length - MAX_POINTS);
        }
        set((st) => ({ frame: f, version: st.version + 1 }));
      }
    };
    send({ type: 'init', inputs });
    send({ type: 'speed', speed: get().speed });
    send({ type: 'focus', replica: focused });
    send({ type: 'play' });
  },
  pause: () => {
    send({ type: 'pause' });
    set({ status: 'paused' });
  },
  resume: () => {
    send({ type: 'play' });
    set({ status: 'running' });
  },
  stop: () => {
    worker?.terminate();
    worker = null;
    set({ status: 'idle', frame: null, series: emptySeries(), error: null });
  },
  setSpeed: (speed) => {
    send({ type: 'speed', speed });
    set({ speed });
  },
  focus: (replica) => {
    focused = replica;
    send({ type: 'focus', replica });
  },
  updateWorkload: (w) => send({ type: 'workload', workload: w }),
  fastForward: (seconds) => send({ type: 'fastForward', seconds }),
}));
