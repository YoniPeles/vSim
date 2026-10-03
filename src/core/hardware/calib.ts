import type { Calibration, GpuArch } from '../types.ts';
import { GiB } from '../units.ts';

// Efficiency (η) and latency (α) constants. These are first-principles defaults; scripts/calibrate.ts
// fits them against published vLLM measurements, and the fitted values override these per arch.
const DEFAULT: Calibration = {
  arch: 'default',
  etaCompute: 0.62,
  etaMem: 0.82,
  etaAttnCompute: 0.45,
  etaScaleUp: 0.75,
  etaScaleOut: 0.8,
  alphaScaleUp: 7e-6,
  alphaScaleOut: 18e-6,
  alphaA2A: 22e-6,
  alphaP2P: 12e-6,
  alphaKvx: 1.5e-3,
  minKernel: 3e-6,
  launchPiecewisePerLayer: 12e-6,
  launchEagerPerLayer: 110e-6,
  hostBase: 0.6e-3,
  hostPerReq: 3e-6,
  hostPerNew: 25e-6,
  nonTorchBase: 0.6 * GiB,
  nonTorchPerComm: 0.12 * GiB,
  actRho: 1.2,
  cudagraphPerSizePerLayer: 180e3,
};

const OVERRIDES: Partial<Record<GpuArch, Partial<Calibration>>> = {
  sm80: { etaCompute: 0.68, alphaScaleUp: 9e-6 },
  sm90: { etaCompute: 0.65 },
  sm100: { etaCompute: 0.6, etaMem: 0.8 },
  sm103: { etaCompute: 0.6, etaMem: 0.8 },
  sm89: { etaCompute: 0.6, alphaScaleUp: 20e-6, etaScaleUp: 0.7 },
  sm120: { etaCompute: 0.55, alphaScaleUp: 20e-6, etaScaleUp: 0.7 },
  gfx942: { etaCompute: 0.45, etaMem: 0.7, alphaScaleUp: 12e-6, hostBase: 0.9e-3, minKernel: 5e-6 },
  gfx950: { etaCompute: 0.45, etaMem: 0.72, alphaScaleUp: 12e-6, hostBase: 0.9e-3, minKernel: 5e-6 },
};

export function calibrationFor(arch: GpuArch): Calibration {
  return { ...DEFAULT, ...OVERRIDES[arch], arch };
}
