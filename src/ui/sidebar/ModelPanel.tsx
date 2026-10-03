import { useState } from 'react';
import { PRESETS } from '../../core/model/presets.ts';
import { DTYPE_LABEL } from '../../core/model/quant.ts';
import { fmtCount } from '../../core/units.ts';
import { useApp } from '../../state/store.ts';
import { useDerivedContext } from '../DerivedContext.tsx';
import { Field, Section, Select } from '../controls.tsx';

function presetGroup(cfg: Record<string, unknown>): string {
  const c = (cfg['text_config'] as Record<string, unknown>) ?? cfg;
  const moe = c['n_routed_experts'] ?? c['num_local_experts'] ?? c['num_experts'];
  if (c['kv_lora_rank']) return 'MLA + mixture of experts';
  if (moe && Number(moe) > 1) return 'Mixture of experts';
  return 'Dense';
}

export function ModelPanel() {
  const repo = useApp((s) => s.modelRepo);
  const setModel = useApp((s) => s.setModel);
  const status = useApp((s) => s.modelStatus);
  const error = useApp((s) => s.modelError);
  const token = useApp((s) => s.hfToken);
  const setToken = useApp((s) => s.setToken);
  const { model } = useDerivedContext();
  const [query, setQuery] = useState('');
  const [showToken, setShowToken] = useState(false);

  const options = PRESETS.map((p) => ({ value: p.repo, label: p.label, group: presetGroup(p.config) }));
  const isPreset = PRESETS.some((p) => p.repo === repo);
  if (!isPreset) options.unshift({ value: repo, label: repo, group: 'Loaded from the Hub' });

  const dtypes = [...new Set([model.quant.attn, model.quant.dense, model.quant.experts])].map((d) => DTYPE_LABEL[d]);
  const moe = model.layers.find((l) => l.ffn.kind === 'moe')?.ffn;

  return (
    <Section title="Model">
      <Select value={repo} options={options} onChange={(v) => void setModel(v)} />
      <form
        className="flex gap-1.5"
        onSubmit={(e) => {
          e.preventDefault();
          if (query.trim()) void setModel(query.trim());
        }}
      >
        <input
          className="ctl min-w-0 flex-1"
          placeholder="Any Hugging Face repo, e.g. Qwen/Qwen3-14B"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Hugging Face repo id"
        />
        <button type="submit" className="ctl shrink-0 px-2 hover:bg-[#1b3550]" disabled={status === 'loading'}>
          {status === 'loading' ? 'Loading…' : 'Load'}
        </button>
      </form>
      {error && (
        <p className="text-[12.5px] text-error" role="alert">
          {error}{' '}
          {error.includes('gated') && (
            <button type="button" className="underline" onClick={() => setShowToken(true)}>
              Add a token
            </button>
          )}
        </p>
      )}
      {(showToken || token) && (
        <Field label="HF token" hint="Kept in this tab only (sessionStorage); never put in the URL.">
          <input
            className="ctl w-full"
            type="password"
            placeholder="hf_…"
            value={token}
            onChange={(e) => setToken(e.target.value.trim())}
            autoComplete="off"
          />
        </Field>
      )}
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[13px]">
        <dt className="text-muted">Parameters</dt>
        <dd className="num">
          {fmtCount(model.params.total)} total
          {model.params.active < model.params.total * 0.95 && `, ${fmtCount(model.params.active)} active`}
        </dd>
        <dt className="text-muted">Layers</dt>
        <dd>
          {model.groups.map((g) => (
            <div key={g.key}>
              <span className="num">{g.indices.length}</span> × {g.label.replace(' · ', ', ')}
            </div>
          ))}
        </dd>
        {moe && moe.kind === 'moe' && (
          <>
            <dt className="text-muted">Experts</dt>
            <dd className="num">
              {moe.E} routed{moe.nShared ? ` + ${moe.nShared} shared` : ''}, {moe.topK} per token
            </dd>
          </>
        )}
        <dt className="text-muted">Weights</dt>
        <dd>{model.quant.method === 'none' ? DTYPE_LABEL[model.baseDtype] : `${model.quant.label} (${dtypes.join(' / ')})`}</dd>
        {model.mtp.layers.length > 0 && (
          <>
            <dt className="text-muted">MTP</dt>
            <dd className="num">{model.mtp.layers.length} module, {fmtCount(model.params.mtp)} params</dd>
          </>
        )}
      </dl>
      {model.warnings.map((w) => (
        <p key={w} className="text-[12.5px] text-warn">
          {w}
        </p>
      ))}
    </Section>
  );
}
