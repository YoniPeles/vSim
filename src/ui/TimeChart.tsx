import { useEffect, useRef } from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import { fmtCount } from '../core/units.ts';

export interface ChartSeries {
  label: string;
  color: string;
  values: number[];
}

// Streaming time series (uPlot): 2px lines, hairline grid, one y-axis, crosshair with values in the
// legend row. Data arrays are read by reference and redrawn when `version` changes.
export function TimeChart({
  title,
  t,
  series,
  version,
  height = 86,
  unit = '',
  yMax,
}: {
  title: string;
  t: number[];
  series: ChartSeries[];
  version: number;
  height?: number;
  unit?: string;
  yMax?: number;
}) {
  const host = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const opts: uPlot.Options = {
      width: el.clientWidth,
      height,
      padding: [6, 6, 0, 0],
      legend: { show: false },
      cursor: { points: { size: 7 }, drag: { x: false, y: false } },
      scales: { x: { time: false }, y: { range: (_u, _min, max) => [0, yMax ?? Math.max(1, max * 1.1)] } },
      axes: [
        { stroke: '#8ca1b4', grid: { stroke: '#1c3246', width: 1 }, ticks: { show: false }, size: 22, font: '10px IBM Plex Sans Condensed', values: (_u, v) => v.map((x) => `${Math.round(x)}s`) },
        { stroke: '#8ca1b4', grid: { stroke: '#1c3246', width: 1 }, ticks: { show: false }, size: 40, font: '10px IBM Plex Sans Condensed', values: (_u, v) => v.map((x) => fmtCount(x, x < 10 ? 1 : 0) + unit) },
      ],
      series: [{}, ...series.map((s) => ({ label: s.label, stroke: s.color, width: 2, points: { show: false }, value: (_u: uPlot, v: number | null) => (v == null ? '–' : fmtCount(v, 1) + unit) }))],
    };
    const u = new uPlot(opts, [t, ...series.map((s) => s.values)], el);
    plot.current = u;
    const ro = new ResizeObserver(() => u.setSize({ width: el.clientWidth, height }));
    ro.observe(el);
    return () => {
      ro.disconnect();
      u.destroy();
      plot.current = null;
    };
    // Recreate only when the series set changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [series.map((s) => s.label).join('|'), height, unit, yMax]);

  useEffect(() => {
    plot.current?.setData([t, ...series.map((s) => s.values)]);
  }, [version, t, series]);

  return (
    <div className="min-w-0">
      <div className="flex flex-wrap items-center gap-x-2.5 text-[12px] text-muted">
        <span>{title}</span>
        {series.length > 1 &&
          series.map((s) => (
            <span key={s.label} className="flex items-center gap-1">
              <span className="inline-block h-[2px] w-2.5 rounded-full" style={{ background: s.color }} />
              {s.label}
              <span className="num text-text">{fmtCount(s.values[s.values.length - 1] ?? 0, 0)}</span>
            </span>
          ))}
      </div>
      <div ref={host} className="vsim-chart w-full" />
    </div>
  );
}
