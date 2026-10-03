import { useEffect, useMemo, useRef } from 'react';
import { useSim } from '../state/sim.ts';
import { fmtPct } from '../core/units.ts';

// Paged KV cache of one replica, one cell per block: who owns it, what is cached for reuse, and
// what is shared between requests (prefix caching). Shades stay inside the KV hue family; each
// request gets its own lightness so neighbours are distinguishable.
const W = 336;
const H = 150;

function palette(): Uint8ClampedArray {
  const n = 603;
  const p = new Uint8ClampedArray(n * 4);
  const set = (i: number, r: number, g: number, b: number) => p.set([r, g, b, 255], i * 4);
  set(0, 20, 39, 57); // free
  set(1, 32, 77, 72); // free but cached
  set(2, 232, 255, 250); // shared (ref > 1)
  for (let i = 3; i < n; i++) {
    const f = ((i * 0.618034) % 1) * 0.55 + 0.35; // lightness spread
    set(i, Math.round(25 * f + 10), Math.round(158 * f + 40), Math.round(112 * f + 30));
  }
  return p;
}

export function KvGrid({ replica, label }: { replica: number; label: string }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const status = useSim((s) => s.status);
  const frame = useSim((s) => s.frame);
  const pal = useMemo(() => palette(), []);

  useEffect(() => {
    useSim.getState().focus(replica);
  }, [replica]);

  const map = frame?.kvMap && frame.kvMap.replica === replica ? frame.kvMap.blocks : null;
  const n = map?.length ?? 0;
  const cell = n ? Math.max(1, Math.floor(Math.sqrt((W * H) / n))) : 1;
  const cols = Math.max(1, Math.floor(W / cell));
  const rows = Math.max(1, Math.ceil(n / cols));

  useEffect(() => {
    const c = canvas.current;
    if (!c || !map) return;
    c.width = cols;
    c.height = rows;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const img = ctx.createImageData(cols, rows);
    const d = img.data;
    for (let i = 0; i < map.length; i++) {
      const k = Math.min(602, map[i]!) * 4;
      d[i * 4] = pal[k]!;
      d[i * 4 + 1] = pal[k + 1]!;
      d[i * 4 + 2] = pal[k + 2]!;
      d[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [map, cols, rows, pal]);

  if (status === 'idle' || status === 'error') return null;
  const used = map ? map.reduce((a, v) => a + (v >= 2 ? 1 : 0), 0) : 0;
  const cached = map ? map.reduce((a, v) => a + (v === 1 ? 1 : 0), 0) : 0;
  const shared = map ? map.reduce((a, v) => a + (v === 2 ? 1 : 0), 0) : 0;

  return (
    <section className="border-b border-rule px-4 py-3">
      <h2 className="mb-1.5 text-[15px] font-semibold tracking-tight">KV blocks, {label}</h2>
      {map ? (
        <>
          <canvas
            ref={canvas}
            className="block rounded-[2px]"
            style={{ width: cols * cell, height: rows * cell, imageRendering: 'pixelated' }}
            role="img"
            aria-label={`KV cache block map: ${fmtPct(used / n)} in use`}
          />
          <ul className="num mt-1.5 grid grid-cols-2 gap-x-3 gap-y-0.5 text-[12px]">
            <li className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-2.5 rounded-[2px] bg-[#28a07c]" />
              <span className="text-muted">In use</span>
              <span className="ml-auto">{fmtPct(used / n)}</span>
            </li>
            <li className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-2.5 rounded-[2px] bg-[#e8fffa]" />
              <span className="text-muted">Shared prefix</span>
              <span className="ml-auto">{fmtPct(shared / n, 1)}</span>
            </li>
            <li className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-2.5 rounded-[2px] bg-[#204d48]" />
              <span className="text-muted">Cached, reusable</span>
              <span className="ml-auto">{fmtPct(cached / n)}</span>
            </li>
            <li className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-2.5 rounded-[2px] bg-[#142739] ring-1 ring-[#2a4560]" />
              <span className="text-muted">Free</span>
              <span className="ml-auto">{fmtPct((n - used - cached) / n)}</span>
            </li>
          </ul>
          <p className="mt-1.5 text-[12px] leading-snug text-muted">
            {n.toLocaleString('en-US')} blocks, one cell each; a request’s blocks share a shade. Freed blocks keep their content until reused, which is
            how prefix caching and preemption-by-recompute find them again.
          </p>
        </>
      ) : (
        <p className="text-[12.5px] text-muted">Waiting for the first frame…</p>
      )}
    </section>
  );
}
