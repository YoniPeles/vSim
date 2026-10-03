import { useEffect, useState } from 'react';
import { fmtTime } from '../core/units.ts';
import { phaseAt, realAt, useTrace } from '../state/trace.ts';
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

// Overlay for the step trace: start button, current phase, and a strip of the whole step where
// each segment's width is its real GPU time.
export function TraceHud() {
  const { views } = useDerivedContext();
  const run = useTrace((s) => s.run);
  const [now, setNow] = useState(0);
  useEffect(() => {
    if (!run) return;
    let raf = 0;
    const loop = () => {
      const t = performance.now() / 1000;
      setNow(t);
      if (phaseAt(run, t) < 0) {
        useTrace.getState().stop();
        return;
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
  const i = phaseAt(run, Math.max(now, run.started));
  const ph = run.phases[Math.max(0, i)]!;
  const t = realAt(run, Math.max(0, i));
  const nLayers = new Set(run.phases.filter((p) => p.layer >= 0).map((p) => p.layer)).size;
  return (
    <div className="absolute top-3 right-3 left-3 rounded bg-[#0d1926d9] px-3 py-2 backdrop-blur-sm">
      <div className="flex items-baseline gap-3 text-[13px]">
        <span className="font-semibold">{PHASE_LABEL[ph.kind]}</span>
        <span className="num text-muted">
          {ph.layer >= 0 ? `layer ${ph.layer + 1} of ${nLayers}` : ''}
          {run.phases.some((p) => p.stage > 0) ? `, PP stage ${ph.stage}` : ''}
        </span>
        <span className="num text-muted">{fmtTime(ph.dur)}</span>
        <span className="num ml-auto text-muted">
          {fmtTime(t)} of {fmtTime(run.realTotal)} GPU time, slowed down
        </span>
        <button type="button" className="ctl px-2 py-0.5 text-[12px]" onClick={() => useTrace.getState().stop()}>
          Close
        </button>
      </div>
      <div className="mt-1.5 flex h-2 w-full overflow-hidden rounded-[2px]" aria-hidden>
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
