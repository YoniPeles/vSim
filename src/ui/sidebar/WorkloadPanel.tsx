import { useApp } from '../../state/store.ts';
import { Field, NumberInput, Section, Slider } from '../controls.tsx';

export function WorkloadPanel() {
  const w = useApp((s) => s.workload);
  const set = useApp((s) => s.setWorkload);
  const skew = useApp((s) => s.routingSkew);
  const setSkew = useApp((s) => s.setRoutingSkew);
  return (
    <Section title="Workload">
      <Field label="Prompt tokens">
        <NumberInput value={w.isl} min={1} max={2_000_000} onChange={(isl) => set({ isl: isl ?? 1024 })} />
      </Field>
      <Field label="Output tokens">
        <NumberInput value={w.osl} min={1} max={500_000} onChange={(osl) => set({ osl: osl ?? 256 })} />
      </Field>
      <Field label="Concurrent users">
        <NumberInput value={w.concurrency} min={1} max={1_000_000} onChange={(concurrency) => set({ concurrency: concurrency ?? 1 })} />
      </Field>
      <Field label="Prefix cache hits" hint="Share of each prompt already in the prefix cache (system prompts, multi-turn)">
        <Slider label="Prefix cache hit rate" value={w.prefixHit} min={0} max={0.95} step={0.05} onChange={(prefixHit) => set({ prefixHit })} />
        <span className="num w-9 text-right text-[13px]">{Math.round(w.prefixHit * 100)}%</span>
      </Field>
      <Field label="Target speed" hint="Per-user decode speed used for the “users at target” figure">
        <NumberInput value={w.slaTokPerSec} min={1} max={2000} width="w-16" suffix="tok/s" onChange={(slaTokPerSec) => set({ slaTokPerSec: slaTokPerSec ?? 20 })} />
      </Field>
      <Field label="Target TTFT">
        <NumberInput value={w.slaTtft} min={0.05} max={120} step={0.1} width="w-16" suffix="s" onChange={(slaTtft) => set({ slaTtft: slaTtft ?? 2 })} />
      </Field>
      <Field label="Expert skew" hint="How unevenly the router spreads tokens over EP ranks (0 = perfectly random)">
        <Slider label="Expert routing skew" value={skew} min={0} max={2} step={0.1} onChange={setSkew} />
        <span className="num w-9 text-right text-[13px]">{skew.toFixed(1)}</span>
      </Field>
    </Section>
  );
}
