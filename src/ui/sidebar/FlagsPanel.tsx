import { totalGpus } from '../../core/hardware/clusters.ts';
import { ALL2ALL_BACKENDS, defaultSpec, KV_DTYPES, SPEC_METHODS } from '../../core/vllm/flags.ts';
import type { All2AllBackend, KvCacheDtype, SpecMethod } from '../../core/types.ts';
import { fmtTokens } from '../../core/units.ts';
import { useApp, useEditedFlags } from '../../state/store.ts';
import { hasMla, hasMoe } from '../../core/vllm/resolve.ts';
import { useDerivedContext } from '../DerivedContext.tsx';
import { Field, NumberInput, Section, Segmented, Select, Slider, Toggle } from '../controls.tsx';

const pow2 = (max: number) => [1, 2, 4, 8, 16, 32, 64].filter((x) => x <= max);

export function FlagsPanel() {
  const f = useEditedFlags();
  const setFlags = useApp((s) => s.setFlags);
  const mode = useApp((s) => s.mode);
  const setMode = useApp((s) => s.setMode);
  const pdSide = useApp((s) => s.pdSide);
  const setPdSide = useApp((s) => s.setPdSide);
  const pd = useApp((s) => s.pd);
  const { model, cluster, evaluation } = useDerivedContext();
  const total = totalGpus(cluster);
  const moe = hasMoe(model);
  const mla = hasMla(model);
  const resolved = evaluation.instances[mode === 'single' ? 0 : pdSide === 'prefill' ? 0 : 1]?.inst;
  const otherWorld = mode === 'pd' ? (pdSide === 'prefill' ? pd.decode : pd.prefill) : null;
  const room = total - (otherWorld ? otherWorld.tp * otherWorld.pp * otherWorld.dp : 0);
  const world = f.tp * f.pp * f.dp;

  return (
    <>
      <Section title="Serving">
        <Segmented
          label="Serving topology"
          value={mode}
          onChange={setMode}
          options={[
            { value: 'single', label: 'One vLLM instance' },
            { value: 'pd', label: 'Prefill / decode split' },
          ]}
        />
        {mode === 'pd' && (
          <>
            <p className="text-[12.5px] text-muted">
              Prefill and decode run as separate instances; each prompt's KV cache moves from the prefill GPUs to the decode GPUs over NIXL.
            </p>
            <Segmented
              label="Edit instance"
              value={pdSide}
              onChange={setPdSide}
              options={[
                { value: 'prefill', label: `Prefill (${pd.prefill.tp * pd.prefill.pp * pd.prefill.dp} GPUs)` },
                { value: 'decode', label: `Decode (${pd.decode.tp * pd.decode.pp * pd.decode.dp} GPUs)` },
              ]}
            />
          </>
        )}
      </Section>

      <Section
        title="Parallelism"
        aside={
          <span className={`num text-[12.5px] ${world > room ? 'text-error' : 'text-muted'}`}>
            {world} of {room} GPUs
          </span>
        }
      >
        <Field label="Tensor (TP)">
          <Segmented label="Tensor parallel size" value={f.tp} onChange={(tp) => setFlags({ tp })} options={pow2(Math.min(room, 64)).map((v) => ({ value: v }))} />
        </Field>
        <Field label="Pipeline (PP)">
          <Segmented label="Pipeline parallel size" value={f.pp} onChange={(pp) => setFlags({ pp })} options={pow2(Math.min(room, 16)).map((v) => ({ value: v }))} />
        </Field>
        <Field label="Data (DP)">
          <NumberInput value={f.dp} min={1} max={512} width="w-16" onChange={(dp) => setFlags({ dp: dp ?? 1 })} />
          <button
            type="button"
            className="ctl px-2 text-[12.5px] hover:bg-[#1b3550]"
            onClick={() => setFlags({ dp: Math.max(1, Math.floor(room / (f.tp * f.pp))) })}
          >
            Fill cluster
          </button>
        </Field>
        <Field label="Expert parallel" hint={moe ? 'EP = TP × DP: whole experts per GPU' : 'Dense model: no experts'}>
          <Toggle checked={f.ep && moe} onChange={(ep) => setFlags({ ep })} label="Enable expert parallel" />
          {moe && (
            <span className="num text-[12.5px] text-muted">
              {f.ep ? `EP ${f.tp * f.dp}` : `experts sharded ${f.tp * f.dp}-way`}
            </span>
          )}
        </Field>
        {moe && f.ep && (
          <>
            <Field label="All-to-all">
              <Select<All2AllBackend>
                value={f.all2allBackend}
                options={ALL2ALL_BACKENDS.map((b) => ({ value: b.id, label: b.label }))}
                onChange={(all2allBackend) => setFlags({ all2allBackend })}
              />
            </Field>
            <Field label="Redundant experts" hint="EPLB replicas of hot experts (static here)">
              <NumberInput value={f.numRedundantExperts} min={0} max={256} width="w-16" onChange={(v) => setFlags({ numRedundantExperts: v ?? 0 })} />
            </Field>
          </>
        )}
      </Section>

      <Section title="Memory and precision">
        <Field label="GPU memory util">
          <Slider label="gpu-memory-utilization" value={f.gpuMemoryUtilization} min={0.5} max={0.98} step={0.01} onChange={(gpuMemoryUtilization) => setFlags({ gpuMemoryUtilization })} />
          <span className="num w-9 text-right text-[13px]">{f.gpuMemoryUtilization.toFixed(2)}</span>
        </Field>
        <Field label="Quantization">
          <Select
            value={f.quantization}
            options={[
              { value: 'auto', label: model.quant.method === 'none' ? 'From checkpoint (BF16)' : `From checkpoint (${model.quant.label})` },
              ...(model.quant.method === 'none' ? [{ value: 'fp8' as const, label: 'fp8 (quantize at load)' }] : []),
            ]}
            onChange={(quantization) => setFlags({ quantization })}
          />
        </Field>
        <Field label="KV cache dtype">
          <Select<KvCacheDtype>
            value={f.kvCacheDtype}
            options={KV_DTYPES.filter((k) => !k.mlaOnly || mla).map((k) => ({ value: k.id, label: k.label }))}
            onChange={(kvCacheDtype) => setFlags({ kvCacheDtype })}
          />
        </Field>
        <Field label="Max model len">
          <NumberInput value={f.maxModelLen} min={256} max={10_000_000} placeholder={fmtTokens(model.maxPos)} onChange={(maxModelLen) => setFlags({ maxModelLen })} />
        </Field>
      </Section>

      <Section title="Scheduler">
        <Field label="Max sequences">
          <NumberInput value={f.maxNumSeqs} min={1} max={16384} placeholder={String(resolved?.maxNumSeqs ?? '')} onChange={(maxNumSeqs) => setFlags({ maxNumSeqs })} />
        </Field>
        <Field label="Token budget" hint="--max-num-batched-tokens: tokens per scheduler step">
          <NumberInput
            value={f.maxNumBatchedTokens}
            min={64}
            max={1_000_000}
            placeholder={String(resolved?.maxNumBatchedTokens ?? '')}
            onChange={(maxNumBatchedTokens) => setFlags({ maxNumBatchedTokens })}
          />
        </Field>
        <Field label="Chunked prefill">
          <Toggle checked={f.enableChunkedPrefill} onChange={(enableChunkedPrefill) => setFlags({ enableChunkedPrefill })} label="Chunked prefill" />
        </Field>
        <Field label="Prefix caching">
          <Toggle checked={f.enablePrefixCaching} onChange={(enablePrefixCaching) => setFlags({ enablePrefixCaching })} label="Prefix caching" />
        </Field>
        <Field label="CUDA graphs">
          <Segmented
            label="Optimization level"
            value={f.optimizationLevel}
            onChange={(optimizationLevel) => setFlags({ optimizationLevel })}
            options={[
              { value: 0, label: 'Off (-O0)' },
              { value: 1, label: 'Piecewise' },
              { value: 2, label: 'Full + piecewise' },
            ]}
          />
        </Field>
        <Field label="Performance mode">
          <Segmented
            label="Performance mode"
            value={f.performanceMode}
            onChange={(performanceMode) => setFlags({ performanceMode })}
            options={[
              { value: 'interactivity', label: 'Interactive' },
              { value: 'balanced', label: 'Balanced' },
              { value: 'throughput', label: 'Throughput' },
            ]}
          />
        </Field>
      </Section>

      <Section title="Speculative decoding">
        <Field label="Method">
          <Select<SpecMethod | 'off'>
            value={f.speculative?.method ?? 'off'}
            options={[
              { value: 'off', label: 'Off' },
              ...SPEC_METHODS.filter((m) => m.id !== 'mtp' || model.mtp.layers.length > 0).map((m) => ({ value: m.id, label: m.label })),
            ]}
            onChange={(m) => setFlags({ speculative: m === 'off' ? null : defaultSpec(m) })}
          />
        </Field>
        {f.speculative && (
          <>
            <Field label="Draft tokens (k)">
              <Segmented
                label="Number of speculative tokens"
                value={f.speculative.k}
                onChange={(k) => setFlags({ speculative: { ...f.speculative!, k } })}
                options={[1, 2, 3, 4, 5, 7].map((v) => ({ value: v }))}
              />
            </Field>
            <Field label="Acceptance" hint="Probability the first drafted token is accepted; later positions decay">
              <Slider
                label="Acceptance rate"
                value={f.speculative.acceptance}
                min={0.1}
                max={0.98}
                step={0.01}
                onChange={(acceptance) => setFlags({ speculative: { ...f.speculative!, acceptance } })}
              />
              <span className="num w-9 text-right text-[13px]">{Math.round(f.speculative.acceptance * 100)}%</span>
            </Field>
          </>
        )}
      </Section>
    </>
  );
}
