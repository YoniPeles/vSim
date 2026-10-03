// Color encodes quantities only: memory categories and traffic types. Chrome stays neutral.
export const C = {
  bg: '#0d1926',
  floor: '#10202f',
  platform: '#132638',
  board: '#18304a',
  tile: '#1d2f40',
  tileEdge: '#3b5873',
  line: '#2a4560',
  text: '#d8e3ec',
  muted: '#8ca1b4',

  // memory (validated categorical order for the dark surface: adjacent CVD ΔE ≥ 8)
  weights: '#d95926',
  activations: '#9085e9',
  overhead: '#c98500',
  kvUsed: '#199e70',
  kvFree: '#199e70',
  unreserved: '#1f3448',

  // traffic (validated order: TP, MoE, PP, KV transfer)
  tp: '#3987e5',
  ep: '#d55181',
  pp: '#c98500',
  kvx: '#199e70',

  // model anatomy (luminance steps, not hues)
  attnFull: '#91a6b9',
  attnLocal: '#6f879c',
  attnLinear: '#5f7d92',
  ffnDense: '#687b8d',
  moeA: '#a39573',
  moeB: '#6a6150',

  error: '#ff6b6b',
  warn: '#ffc54d',
} as const;

export const MEMORY_LEGEND = [
  { key: 'weights', label: 'Weights', color: C.weights },
  { key: 'activations', label: 'Activation peak', color: C.activations },
  { key: 'overhead', label: 'Runtime + CUDA graphs', color: C.overhead },
  { key: 'kvUsed', label: 'KV cache in use', color: C.kvUsed },
  { key: 'kvFree', label: 'KV cache free', color: C.kvFree },
] as const;

export const TRAFFIC_LEGEND = [
  { key: 'tp', label: 'TP all-reduce', color: C.tp },
  { key: 'ep', label: 'MoE dispatch / combine', color: C.ep },
  { key: 'pp', label: 'Pipeline hand-off', color: C.pp },
  { key: 'kvx', label: 'KV transfer (P/D)', color: C.kvx },
] as const;
