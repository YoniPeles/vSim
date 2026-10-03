import { describe, expect, it } from 'vitest';
import { parseVllmLog } from './logParse.ts';

const LOG = `INFO 09-12 10:01:02 [core.py:74] Initializing a V1 LLM engine (v0.30.0) with config: model='meta-llama/Llama-3.1-8B-Instruct', speculative_config=None, tokenizer='meta-llama/Llama-3.1-8B-Instruct', dtype=torch.bfloat16, max_seq_len=131072, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, quantization=None, kv_cache_dtype=auto
(EngineCore pid=1234) INFO 09-12 10:01:20 [gpu_model_runner.py:2410] Model loading took 14.9889 GiB memory and 6.120 seconds
(EngineCore pid=1234) INFO 09-12 10:01:31 [gpu_worker.py:298] Available KV cache memory: 53.71 GiB
(EngineCore pid=1234) INFO 09-12 10:01:31 [kv_cache_utils.py:1087] GPU KV cache size: 439,984 tokens
(EngineCore pid=1234) INFO 09-12 10:01:31 [kv_cache_utils.py:1091] Maximum concurrency for 131,072 tokens per request: 3.36x
(EngineCore pid=1234) INFO 09-12 10:01:45 [gpu_model_runner.py:3020] Graph capturing finished in 14 secs, took 0.52 GiB`;

describe('parseVllmLog', () => {
  it('extracts config and memory facts', () => {
    const f = parseVllmLog(LOG);
    expect(f.model).toBe('meta-llama/Llama-3.1-8B-Instruct');
    expect(f.tp).toBe(1);
    expect(f.maxModelLen).toBe(131072);
    expect(f.weightsGiB).toBeCloseTo(14.9889);
    expect(f.kvGiB).toBeCloseTo(53.71);
    expect(f.kvTokens).toBe(439984);
    expect(f.maxConcurrency).toEqual({ len: 131072, x: 3.36 });
    expect(f.cudagraphGiB).toBeCloseTo(0.52);
    expect(f.quantization).toBeUndefined();
  });

  it('returns nothing for unrelated text', () => {
    expect(parseVllmLog('hello world')).toEqual({});
  });
});
