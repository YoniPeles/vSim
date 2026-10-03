// Extract the memory/KV figures vLLM prints at startup, so predictions can be checked against a
// real run. Tolerant of log prefixes, thousands separators and minor wording changes across versions.

export interface VllmLogFacts {
  model?: string;
  tp?: number;
  pp?: number;
  dp?: number;
  maxModelLen?: number;
  kvCacheDtype?: string;
  quantization?: string;
  weightsGiB?: number;
  kvGiB?: number;
  kvTokens?: number;
  maxConcurrency?: { len: number; x: number };
  cudagraphGiB?: number;
  torchPeakGiB?: number;
  nonTorchGiB?: number;
}

const num = (s: string | undefined): number | undefined => (s === undefined ? undefined : Number(s.replace(/,/g, '')));

function last(re: RegExp, text: string): RegExpExecArray | null {
  let m: RegExpExecArray | null = null;
  const g = new RegExp(re.source, re.flags.includes('g') ? re.flags : `${re.flags}g`);
  for (let x = g.exec(text); x; x = g.exec(text)) m = x;
  return m;
}

export function parseVllmLog(text: string): VllmLogFacts {
  const f: VllmLogFacts = {};
  const model = last(/model=['"]([^'"]+)['"]/, text);
  if (model) f.model = model[1]!;
  const tp = last(/tensor_parallel_size=(\d+)/, text);
  if (tp) f.tp = num(tp[1])!;
  const pp = last(/pipeline_parallel_size=(\d+)/, text);
  if (pp) f.pp = num(pp[1])!;
  const dp = last(/data_parallel_size=(\d+)/, text);
  if (dp) f.dp = num(dp[1])!;
  const msl = last(/max_seq_len=(\d+)/, text) ?? last(/max_model_len=(\d+)/, text);
  if (msl) f.maxModelLen = num(msl[1])!;
  const kvd = last(/kv_cache_dtype=([\w.]+)/, text);
  if (kvd) f.kvCacheDtype = kvd[1]!;
  const q = last(/quantization=([\w.-]+)/, text);
  if (q && q[1] !== 'None') f.quantization = q[1]!;

  const w = last(/Model loading took ([\d.]+) ?GiB/i, text);
  if (w) f.weightsGiB = num(w[1])!;
  const kv = last(/Available KV cache memory: ([\d.]+) ?GiB/i, text);
  if (kv) f.kvGiB = num(kv[1])!;
  const kt = last(/GPU KV cache size: ([\d,]+) tokens/i, text);
  if (kt) f.kvTokens = num(kt[1])!;
  const mc = last(/Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x/i, text);
  if (mc) f.maxConcurrency = { len: num(mc[1])!, x: num(mc[2])! };
  const cg = last(/Graph capturing finished in [\d.]+ secs?, took ([\d.]+) ?GiB/i, text);
  if (cg) f.cudagraphGiB = num(cg[1])!;
  const tpk = last(/torch peak memory increase: ([\d.]+) ?GiB/i, text);
  if (tpk) f.torchPeakGiB = num(tpk[1])!;
  const nt = last(/non-torch (?:forward increase )?memory(?: increase)?: ([\d.]+) ?GiB/i, text);
  if (nt) f.nonTorchGiB = num(nt[1])!;
  return f;
}
