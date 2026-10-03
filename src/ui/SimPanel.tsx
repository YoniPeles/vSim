import { useEffect, useMemo, useRef } from 'react';
import { findPreset } from '../core/model/presets.ts';
import { fmtCount, fmtPct, fmtTime } from '../core/units.ts';
import { useApp } from '../state/store.ts';
import { useSim } from '../state/sim.ts';
import type { SimInputs, SimWorkload } from '../sim/types.ts';
import type { Workload } from '../core/engine/analytic.ts';
import { useDerivedContext } from './DerivedContext.tsx';
import { Segmented } from './controls.tsx';
import { TimeChart } from './TimeChart.tsx';
import { C } from './theme.ts';

const SPEEDS = [0.25, 1, 4, 16, 64];

function useSimInputs(): SimInputs {
  const repo = useApp((s) => s.modelRepo);
  const snaps = useApp((s) => s.snapshots);
  const cluster = useApp((s) => s.cluster);
  const workload = useApp((s) => s.workload);
  const skew = useApp((s) => s.routingSkew);
  const { deployment } = useDerivedContext();
  return useMemo(
    () => ({
      snapshot: findPreset(repo) ?? snaps[repo]!,
      cluster,
      deployment,
      workload: toSimWorkload(workload),
      routingSkew: skew,
      seed: 1,
    }),
    // Workload changes are pushed to the running sim instead of restarting it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [repo, snaps, cluster, deployment, skew],
  );
}

export function toSimWorkload(w: Workload): SimWorkload {
  return { ...w, spread: 0.3, arrival: 'closed', rate: 1 };
}

function Metric({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="min-w-0">
      <div className="truncate text-[12px] text-muted">{label}</div>
      <div className="num text-[17px] leading-tight font-semibold">{value}</div>
      {sub && <div className="num truncate text-[11.5px] text-faint">{sub}</div>}
    </div>
  );
}

export function SimPanel() {
  const status = useSim((s) => s.status);
  const error = useSim((s) => s.error);
  const speed = useSim((s) => s.speed);
  const frame = useSim((s) => s.frame);
  const series = useSim((s) => s.series);
  const version = useSim((s) => s.version);
  const start = useSim((s) => s.start);
  const pause = useSim((s) => s.pause);
  const resume = useSim((s) => s.resume);
  const stop = useSim((s) => s.stop);
  const setSpeed = useSim((s) => s.setSpeed);
  const updateWorkload = useSim((s) => s.updateWorkload);
  const fastForward = useSim((s) => s.fastForward);
  const inputs = useSimInputs();
  const workload = useApp((s) => s.workload);
  const { evaluation } = useDerivedContext();
  const blocked = evaluation.instances.some((i) => i.inst.issues.some((x) => x.level === 'error')) || evaluation.issues.length > 0;

  // Restart when the deployment changes; push workload edits into the live run.
  const first = useRef(true);
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    const st = useSim.getState().status;
    if (st === 'running' || st === 'paused') {
      if (blocked) stop();
      else start(inputs);
    }
  }, [inputs, blocked, start, stop]);
  useEffect(() => {
    if (useSim.getState().status !== 'idle') updateWorkload(toSimWorkload(workload));
  }, [workload, updateWorkload]);

  const live = status === 'running' || status === 'paused';
  const s = frame?.stats;
  const kvPct = frame ? frame.replicas.reduce((a, r) => a + r.kvUsed, 0) / Math.max(1, frame.replicas.reduce((a, r) => a + r.kvTotal, 0)) : 0;
  const tokSeries = useMemo(() => [{ label: 'Output tok/s', color: C.tp, values: series.tok }], [series]);
  const reqSeries = useMemo(
    () => [
      { label: 'Running', color: C.kvUsed, values: series.running },
      { label: 'Waiting', color: C.weights, values: series.waiting },
    ],
    [series],
  );
  const kvSeries = useMemo(() => [{ label: 'KV used', color: C.kvUsed, values: series.kv }], [series]);

  return (
    <div className="border-t border-rule bg-panel px-4 py-2.5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        {live ? (
          <>
            <button type="button" className="ctl px-3 py-1 hover:bg-[#1b3550]" onClick={status === 'running' ? pause : resume}>
              {status === 'running' ? 'Pause' : 'Resume'}
            </button>
            <button type="button" className="ctl px-3 py-1 hover:bg-[#1b3550]" onClick={() => start(inputs)}>
              Restart
            </button>
            <button type="button" className="ctl px-3 py-1 hover:bg-[#1b3550]" onClick={stop}>
              Stop
            </button>
          </>
        ) : (
          <button
            type="button"
            disabled={blocked}
            className="ctl bg-[#1f6b66] px-3 py-1 font-semibold text-white hover:bg-[#23807a] disabled:cursor-not-allowed disabled:opacity-40"
            onClick={() => start(inputs)}
          >
            Simulate {fmtCount(workload.concurrency, 1)} users
          </button>
        )}
        <div className="flex items-center gap-1.5">
          <span className="text-[12.5px] text-muted">Speed</span>
          <Segmented label="Simulation speed" value={speed} onChange={setSpeed} options={SPEEDS.map((v) => ({ value: v, label: v === 0.25 ? '¼×' : `${v}×` }))} />
        </div>
        {live && (
          <button type="button" className="ctl px-2 py-1 text-[12.5px] hover:bg-[#1b3550]" onClick={() => fastForward(30)}>
            Skip 30 s
          </button>
        )}
        <span className="num ml-auto text-[12.5px] text-muted">
          {frame ? `t = ${frame.t.toFixed(1)} s${status === 'running' && frame.wallRatio < speed * 0.8 ? `, running at ${frame.wallRatio.toFixed(1)}× (CPU-limited)` : ''}` : status === 'error' ? '' : 'Requests flow through vLLM’s scheduler in real time'}
        </span>
      </div>
      {status === 'error' && <p className="mt-2 text-[12.5px] text-error">{error}</p>}
      {live && s && (
        <div className="mt-2 grid grid-cols-[minmax(0,1.1fr)_minmax(0,2fr)] gap-4 max-[1100px]:grid-cols-1">
          <div className="grid grid-cols-3 gap-x-3 gap-y-2">
            <Metric label="Output" value={`${fmtCount(s.outTokPerSec, 1)} tok/s`} sub={`${s.reqPerSec.toFixed(1)} req/s done`} />
            <Metric label="TTFT p50 / p99" value={fmtTime(s.ttft.p50)} sub={`p99 ${fmtTime(s.ttft.p99)}`} />
            <Metric label="ITL p50 / p99" value={fmtTime(s.itl.p50)} sub={`p99 ${fmtTime(s.itl.p99)}`} />
            <Metric label="Running / waiting" value={`${s.running} / ${s.waiting}`} sub={s.transferring ? `${s.transferring} KV transfers` : undefined} />
            <Metric label="KV cache used" value={fmtPct(kvPct)} sub={`${s.preemptions} preemptions`} />
            <Metric label="Prefix cache hits" value={fmtPct(s.prefixHitRate)} sub={`${s.completed} requests done`} />
          </div>
          <div className="grid grid-cols-3 gap-3 max-[760px]:grid-cols-1">
            <TimeChart title="Output tokens/s" t={series.t} series={tokSeries} version={version} />
            <TimeChart title="Requests" t={series.t} series={reqSeries} version={version} />
            <TimeChart title="KV cache used" t={series.t} series={kvSeries} version={version} unit="%" yMax={100} />
          </div>
        </div>
      )}
    </div>
  );
}
