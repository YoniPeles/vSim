/// <reference lib="webworker" />
// Runs the Simulator off the main thread. Sim time advances at `speed` × wall time, within a CPU
// budget per tick; frames go out ~30 times a second with the KV map as a transferable buffer.

import { Simulator } from './engine.ts';
import type { FromWorker, ToWorker } from './types.ts';

declare const self: DedicatedWorkerGlobalScope;

let sim: Simulator | null = null;
let playing = false;
let speed = 1;
let lastTick = 0;
let lastFrame = 0;
let timer: ReturnType<typeof setTimeout> | undefined;
let ratio = 1;

function post(m: FromWorker, transfer: Transferable[] = []) {
  self.postMessage(m, transfer);
}

function sendFrame() {
  if (!sim) return;
  const f = sim.frame(ratio);
  post({ type: 'frame', frame: f }, f.kvMap ? [f.kvMap.blocks.buffer] : []);
}

function tick() {
  timer = undefined;
  if (!sim || !playing) return;
  const now = performance.now();
  const dtWall = Math.min(0.1, (now - lastTick) / 1000);
  lastTick = now;
  const before = sim.now;
  try {
    sim.runUntil(before + dtWall * speed, 12);
  } catch (e) {
    post({ type: 'error', message: (e as Error).message });
    playing = false;
    return;
  }
  ratio = dtWall > 0 ? (sim.now - before) / dtWall : speed;
  if (now - lastFrame > 33) {
    lastFrame = now;
    sendFrame();
  }
  timer = setTimeout(tick, 8);
}

self.onmessage = (ev: MessageEvent<ToWorker>) => {
  const m = ev.data;
  try {
    switch (m.type) {
      case 'init': {
        sim = new Simulator(m.inputs);
        post({ type: 'ready', replicas: sim.replicas.length, blocksPerReplica: sim.replicas.map((r) => r.pool.size) });
        sendFrame();
        break;
      }
      case 'play':
        playing = true;
        lastTick = performance.now();
        if (!timer) tick();
        break;
      case 'pause':
        playing = false;
        sendFrame();
        break;
      case 'speed':
        speed = m.speed;
        break;
      case 'focus':
        if (sim) sim.focus = m.replica;
        if (!playing) sendFrame();
        break;
      case 'workload':
        sim?.setWorkload(m.workload);
        break;
      case 'fastForward':
        if (sim) {
          sim.runUntil(sim.now + m.seconds, 4000);
          sendFrame();
        }
        break;
    }
  } catch (e) {
    post({ type: 'error', message: (e as Error).message });
  }
};
