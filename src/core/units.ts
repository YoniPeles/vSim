// Internal units: bytes, seconds, FLOPs, tokens. Convert only at display time.

export const KiB = 1024;
export const MiB = 1024 ** 2;
export const GiB = 1024 ** 3;
export const GB = 1e9;
export const TB = 1e12;

/** Gb/s link rate → bytes/s. */
export const gbitps = (g: number): number => (g * 1e9) / 8;

export function fmtBytes(b: number, digits = 1): string {
  const a = Math.abs(b);
  if (a >= GiB) return `${(b / GiB).toFixed(digits)} GiB`;
  if (a >= MiB) return `${(b / MiB).toFixed(digits)} MiB`;
  if (a >= KiB) return `${(b / KiB).toFixed(digits)} KiB`;
  return `${Math.round(b)} B`;
}

export function fmtCount(n: number, digits = 2): string {
  const a = Math.abs(n);
  if (a >= 1e12) return `${(n / 1e12).toFixed(digits)}T`;
  if (a >= 1e9) return `${(n / 1e9).toFixed(digits)}B`;
  if (a >= 1e6) return `${(n / 1e6).toFixed(digits)}M`;
  if (a >= 1e3) return `${(n / 1e3).toFixed(digits)}K`;
  return `${Math.round(n)}`;
}

export function fmtTime(s: number): string {
  if (!Number.isFinite(s)) return '∞';
  const a = Math.abs(s);
  if (a >= 1) return `${s.toFixed(2)} s`;
  if (a >= 1e-3) return `${(s * 1e3).toFixed(a >= 0.1 ? 0 : 1)} ms`;
  return `${(s * 1e6).toFixed(0)} µs`;
}

/** Bytes/s → "450 GB/s" (decimal, as vendors quote links). */
export function fmtBW(bps: number): string {
  if (bps >= TB) return `${(bps / TB).toFixed(2)} TB/s`;
  return `${(bps / GB).toFixed(bps >= 100 * GB ? 0 : 1)} GB/s`;
}

export function fmtPct(x: number, digits = 0): string {
  return `${(x * 100).toFixed(digits)}%`;
}

export function fmtTokens(n: number): string {
  return fmtCount(n, n >= 1e6 ? 2 : 1);
}

export const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
export const ceilDiv = (a: number, b: number): number => Math.ceil(a / b);
