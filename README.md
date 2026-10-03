# vSim

Visualization engine for vLLM. Pick a Hugging Face model, a GPU cluster and `vllm serve` flags, and see
how vLLM would lay the model out and serve it:

- **3D cluster.** Every GPU shows an HBM tower (weights, activation peak, runtime + CUDA graphs, KV in
  use, KV free) and the slice of the model it holds. Layer plates stack by pipeline stage, their width
  is the tensor-parallel share, and stripes are resident experts. NVSwitch, Infinity Fabric and
  InfiniBand links carry particles for TP all-reduce, MoE dispatch/combine, pipeline hand-offs and P/D
  KV transfers.
- **Readout.** Parameters, per-GPU memory and KV capacity (vLLM's own accounting order), time to first
  token, decode speed, the throughput/interactivity frontier, users served at a target speed,
  per-step anatomy, link utilization, and vLLM-style errors, e.g. "max_model_len doesn't fit".
- **Live simulation.** A Web Worker runs vLLM's scheduler: token budget, chunked prefill, prefix
  caching, recompute preemption, DP lockstep for MoE, PP in-flight batches, speculative decoding and
  prefill/decode disaggregation. The KV block map, towers, particles and charts update in real time.
- **Trace one step.** Replays a single forward pass layer by layer, slowed down, from the cost
  model's per-layer phases.
- **Check against a real log.** Paste a `vllm serve` startup log to compare its memory and KV figures
  with the prediction.

The `vllm serve …` line at the top is the configuration; copy it to run the same thing for real.

## Running

```sh
npm install
npm run dev          # http://localhost:5173
npm test             # unit + property tests (engine, memory, cost, simulator)
npm run e2e          # Playwright smoke tests (needs `npx playwright install chromium`)
npm run build        # typecheck + production build into dist/
```

Node ≥ 24 runs the TypeScript scripts directly:

```sh
node scripts/kvcheck.ts --model deepseek-ai/DeepSeek-V3 --cluster 8xh200 -tp 8   # vLLM-style startup lines
node scripts/fetch-presets.ts                                                     # refresh bundled configs
node scripts/calibrate.ts observations.json --fit                                 # fit η/α to your measurements
```

## How the numbers are made

| Layer | Where | What it does |
|---|---|---|
| Model | `src/core/model` | `config.json` → per-layer IR (GQA / sliding / chunked / MLA / DSA / linear attention; dense / MoE FFN), quantization bytes per weight, MTP modules. Twenty bundled presets; any other repo loads live from the Hub (gated repos need a token, kept in sessionStorage only). |
| Hardware | `src/core/hardware` | Dense TFLOPS, HBM bandwidth, per-direction NVLink/IF, scale-up domains (HGX 8, NVL72 72, AMD mesh, PCIe), NICs; calibration constants per architecture. |
| vLLM | `src/core/vllm` | Flag resolution mirroring `arg_utils.py` (batch defaults by GPU memory, backend block sizes, CUDA-graph capture sizes) and validation. |
| Engine | `src/core/engine` | Placement (TP innermost, PP partition as `get_pp_indices`, EP whole experts vs TP×DP sharding), memory (`requested − weights − non-torch − activation peak − CUDA graphs`), roofline cost per layer group, α–β collectives (custom AR, NVLS, ring, hierarchical, DeepEP dedup), analytic steady state. |
| Simulation | `src/sim` | Discrete-event scheduler simulation on a paged block pool with prefix caching, timed by the cost model. |

`src/core` is plain TypeScript with no DOM (enforced by `tsconfig.core.json`), so the same code runs on
the main thread, in the worker and in Node.

## Accuracy

The model is first-principles: roofline kernels, α–β collectives and vLLM's memory accounting.
Parameter counts match checkpoint totals exactly for the bundled presets. Efficiency and latency
constants in `src/core/hardware/calib.ts` are defaults, not fitted to measurements yet. Treat absolute
latencies as estimates and relative comparisons (TP vs EP, FP8 vs BF16, PP vs DP) as the reliable part.
To improve them, collect startup logs and ITL/TTFT measurements from real runs and use
`scripts/calibrate.ts`.

Known simplifications: decode-context-parallel and dual-batch overlap only affect capacity, not timing;
AMD and Rubin specs are approximate; vision encoders are counted in weights but not in compute;
DeepSeek-V4-style compressed KV and KDA linear attention are approximated.
