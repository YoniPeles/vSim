import { useMemo, useState } from 'react';
import type { SweepPoint } from '../core/engine/analytic.ts';
import { fmtCount, fmtTime } from '../core/units.ts';

// Throughput vs. interactivity: each point is a concurrency level. Moving right = faster per user,
// moving up = more total tokens. One series, so no legend; the current load is the ringed marker.
export function FrontierChart({ points, current, color }: { points: SweepPoint[]; current: SweepPoint | null; color: string }) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 320;
  const H = 150;
  const pad = { l: 44, r: 10, t: 8, b: 30 };
  const { xs, ys, xTicks, yTicks } = useMemo(() => {
    const xMax = niceMax(Math.max(...points.map((p) => p.tokPerUser), current?.tokPerUser ?? 0));
    const yMax = niceMax(Math.max(...points.map((p) => p.throughput), current?.throughput ?? 0));
    const toX = (v: number) => pad.l + (v / xMax) * (W - pad.l - pad.r);
    const toY = (v: number) => H - pad.b - (v / yMax) * (H - pad.t - pad.b);
    return { xs: toX, ys: toY, xTicks: ticks(xMax), yTicks: ticks(yMax) };
  }, [points, current, pad.l, pad.r, pad.t, pad.b]);
  if (points.length < 2) return null;
  const path = points.map((p, i) => `${i ? 'L' : 'M'}${xs(p.tokPerUser).toFixed(1)},${ys(p.throughput).toFixed(1)}`).join('');
  const hp = hover !== null ? points[hover] : null;

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label="Total output throughput versus per-user speed as concurrency grows"
        onMouseLeave={() => setHover(null)}
        onMouseMove={(e) => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
          const x = ((e.clientX - r.left) / r.width) * W;
          let best = 0;
          let bd = Infinity;
          points.forEach((p, i) => {
            const d = Math.abs(xs(p.tokPerUser) - x);
            if (d < bd) {
              bd = d;
              best = i;
            }
          });
          setHover(best);
        }}
      >
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line x1={pad.l} x2={W - pad.r} y1={ys(t)} y2={ys(t)} stroke="#1f364c" strokeWidth={1} />
            <text x={pad.l - 6} y={ys(t) + 3.5} textAnchor="end" fontSize={10} fill="#8ca1b4">
              {fmtCount(t, t % 1000 === 0 || t < 1000 ? 0 : 1)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={`x${t}`} x={xs(t)} y={H - pad.b + 13} textAnchor="middle" fontSize={10} fill="#8ca1b4">
            {t}
          </text>
        ))}
        <text x={(pad.l + W - pad.r) / 2} y={H - 3} textAnchor="middle" fontSize={10.5} fill="#8ca1b4">
          tokens/s per user
        </text>
        <path d={path} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        {current && (
          <circle cx={xs(current.tokPerUser)} cy={ys(current.throughput)} r={5} fill={color} stroke="#0d1926" strokeWidth={2} />
        )}
        {hp && (
          <g pointerEvents="none">
            <line x1={xs(hp.tokPerUser)} x2={xs(hp.tokPerUser)} y1={pad.t} y2={H - pad.b} stroke="#5d7489" strokeWidth={1} />
            <circle cx={xs(hp.tokPerUser)} cy={ys(hp.throughput)} r={4} fill={color} stroke="#0d1926" strokeWidth={2} />
          </g>
        )}
      </svg>
      <figcaption className="num min-h-[2.6em] text-[12px] text-muted">
        {hp ? (
          <>
            {fmtCount(hp.users, 1)} users: {Math.round(hp.tokPerUser)} tok/s each, {fmtCount(hp.throughput, 1)} tok/s total, TTFT{' '}
            {fmtTime(hp.ttft)}
          </>
        ) : (
          <>Total output tokens/s as concurrency grows. The ringed dot is your workload.</>
        )}
      </figcaption>
    </figure>
  );
}

function niceMax(v: number): number {
  if (!(v > 0)) return 1;
  const raw = (v * 1.04) / 4;
  const p = 10 ** Math.floor(Math.log10(raw));
  const step = ([1, 2, 2.5, 5, 10].find((m) => m * p >= raw) ?? 10) * p;
  return Math.ceil((v * 1.04) / step) * step;
}

/** Ticks on a 1/2/2.5/5 × 10^k grid, about four intervals. */
function ticks(max: number): number[] {
  const raw = max / 4;
  const p = 10 ** Math.floor(Math.log10(raw));
  const step = ([1, 2, 2.5, 5, 10].find((m) => m * p >= raw) ?? 10) * p;
  const out: number[] = [];
  for (let v = 0; v <= max * 1.0001; v += step) out.push(Number(v.toPrecision(6)));
  return out;
}
