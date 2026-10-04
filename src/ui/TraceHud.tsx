import { useEffect, useState } from 'react';
import { fmtCount, fmtTime } from '../core/units.ts';
import { phaseAt, phaseAtReal, realAt, traceEnd, useTrace } from '../state/trace.ts';
import { Segmented } from './controls.tsx';
import { useDerivedContext } from './DerivedContext.tsx';
import { C } from './theme.ts';

const PHASE_LABEL: Record<string, string> = {
  attn: 'Attention',
  tp: 'TP all-reduce',
  dispatch: 'MoE dispatch',
  ffn: 'Feed-forward / experts',
  combine: 'MoE combine',
  pp: 'Pipeline hand-off',
  lmhead: 'LM head + sampling',
};

const PHASE_COLOR: Record<string, string> = {
  attn: '#91a6b9',
  ffn: '#a39573',
  lmhead: '#91a6b9',
  tp: C.tp,
  dispatch: C.ep,
  combine: C.ep,
  pp: C.pp,
};

const SPEEDS = [0.5, 1, 2, 4];

// Overlay for the step trace: start button, current phase, playback controls, and a strip of the
// whole step where each segment's width is its real GPU time (click or drag it to scrub).
export function TraceHud() {
  const { views } = useDerivedContext();
  const run = useTrace((s) => s.run);
  const speed = useTrace((s) => s.speed);
  const paused = useTrace((s) => s.paused);
  const [, setTick] = useState(0);
  useEffect(() => {
    if (!run) return;
    let raf = 0;
    let last = performance.now();
    let lastPaint = 0;
    const loop = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const st = useTrace.getState();
      if (!st.paused) {
        run.pos += dt * st.speed;
        if (run.pos >= traceEnd(run)) {
          run.pos = traceEnd(run);
          st.setPaused(true);
        }
      }
      // The scene reads run.pos every frame; the HUD only needs ~15 repaints a second.
      if (now - lastPaint > 66) {
        lastPaint = now;
        setTick((x) => x + 1);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [run]);

  const idx = views.findIndex((v) => v.ev.inst.role !== 'prefill' && v.cost && v.steady);
  const v = views[idx];
  const start = () => {
    if (!v?.cost || !v.steady) return;
    const tr = v.cost.step(v.steady.shape, true).trace;
    if (tr?.length) useTrace.getState().start(idx, tr);
  };

  if (!run) {
    return (
      <button
        type="button"
        disabled={!v}
        className="ctl absolute top-3 right-3 bg-[#0d1926cc] px-3 py-1 backdrop-blur-sm hover:bg-[#1b3550] disabled:opacity-40"
        onClick={start}
        title="Replay one decode step layer by layer, slowed down"
      >
        Trace one step
      </button>
    );
  }
  const i = phaseAt(run);
  const ph = run.phases[i]!;
  const t = realAt(run, i);
  const nLayers = new Set(run.phases.filter((p) => p.layer >= 0).map((p) => p.layer)).size;
  const done = run.pos >= traceEnd(run);
  const slowdown = traceEnd(run) / speed / Math.max(1e-9, run.realTotal);
  const scrub = (e: React.PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const f = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    const j = phaseAtReal(run, f * run.realTotal);
    useTrace.getState().seek(run.wallAt[j]!);
  };
  return (
    <div className="absolute top-3 right-3 left-3 rounded bg-[#0d1926d9] px-3 py-2 backdrop-blur-sm">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[13px]">
        <button
          type="button"
          className="ctl w-16 px-2 py-0.5 text-[12px]"
          onClick={() => useTrace.getState().setPaused(!paused)}
        >
          {done ? 'Replay' : paused ? 'Play' : 'Pause'}
        </button>
        <Segmented label="Trace speed" value={speed} onChange={(x) => useTrace.getState().setSpeed(x)} options={SPEEDS.map((x) => ({ value: x, label: x === 0.5 ? '½×' : `${x}×` }))} />
        <span className="font-semibold">{PHASE_LABEL[ph.kind]}</span>
        <span className="num text-muted">
          {ph.layer >= 0 ? `layer ${ph.layer + 1} of ${nLayers}` : ''}
          {run.phases.some((p) => p.stage > 0) ? `, PP stage ${ph.stage}` : ''}
        </span>
        <span className="num text-muted">{fmtTime(ph.dur)}</span>
        <span className="num ml-auto text-muted">
          {fmtTime(t)} of {fmtTime(run.realTotal)} GPU time, {fmtCount(slowdown, 1)}× slower than real
        </span>
        <button type="button" className="ctl px-2 py-0.5 text-[12px]" onClick={() => useTrace.getState().stop()}>
          Close
        </button>
      </div>
      <div
        className="mt-1.5 flex h-2.5 w-full cursor-pointer overflow-hidden rounded-[2px]"
        role="slider"
        aria-label="Position in the step"
        aria-valuemin={0}
        aria-valuemax={Math.round(run.realTotal * 1e6)}
        aria-valuenow={Math.round(t * 1e6)}
        tabIndex={0}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          scrub(e);
        }}
        onPointerMove={(e) => {
          if (e.buttons & 1) scrub(e);
        }}
        onKeyDown={(e) => {
          const st = useTrace.getState();
          if (e.key === 'ArrowRight') st.seek(run.wallAt[Math.min(run.phases.length - 1, i + 1)]!);
          if (e.key === 'ArrowLeft') st.seek(run.wallAt[Math.max(0, i - 1)]!);
          if (e.key === ' ') {
            e.preventDefault();
            st.setPaused(!st.paused);
          }
        }}
      >
        {run.phases.map((p, j) => (
          <span
            key={j}
            style={{
              width: `${(p.dur / run.realTotal) * 100}%`,
              background: PHASE_COLOR[p.kind],
              opacity: j === i ? 1 : j < i ? 0.75 : 0.25,
            }}
          />
        ))}
      </div>
    </div>
  );
}
