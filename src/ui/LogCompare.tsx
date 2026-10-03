import { useMemo, useState } from 'react';
import { parseVllmLog } from '../core/vllm/logParse.ts';
import { GiB } from '../core/units.ts';
import { useDerivedContext } from './DerivedContext.tsx';

// Paste a real vLLM startup log; compare its memory figures with this configuration's prediction.
export function LogCompare() {
  const [text, setText] = useState('');
  const [open, setOpen] = useState(false);
  const { views, model } = useDerivedContext();
  const facts = useMemo(() => parseVllmLog(text), [text]);
  const v = views[views.length - 1];
  if (!v) return null;
  const m = v.ev.memory.perRank[0]!;
  const rows: { label: string; real?: number; ours: number; fmt: (x: number) => string }[] = [
    { label: 'Weights per GPU', real: facts.weightsGiB, ours: m.weights / GiB, fmt: (x) => `${x.toFixed(2)} GiB` },
    { label: 'Available KV memory', real: facts.kvGiB, ours: m.kv / GiB, fmt: (x) => `${x.toFixed(2)} GiB` },
    { label: 'GPU KV cache size', real: facts.kvTokens, ours: v.ev.memory.kvTokens, fmt: (x) => `${Math.round(x).toLocaleString('en-US')} tok` },
    {
      label: `Max concurrency at ${(facts.maxConcurrency?.len ?? v.ev.inst.maxModelLen).toLocaleString('en-US')}`,
      real: facts.maxConcurrency?.x,
      ours: v.ev.memory.maxConcurrency,
      fmt: (x) => `${x.toFixed(2)}×`,
    },
    { label: 'CUDA graphs', real: facts.cudagraphGiB, ours: m.cudagraph / GiB, fmt: (x) => `${x.toFixed(2)} GiB` },
  ];
  const found = rows.filter((r) => r.real !== undefined);
  const mismatch: string[] = [];
  if (facts.model && facts.model.toLowerCase() !== model.repo.toLowerCase()) mismatch.push(`model ${facts.model}`);
  if (facts.tp && facts.tp !== v.ev.inst.tp) mismatch.push(`TP ${facts.tp}`);
  if (facts.pp && facts.pp !== v.ev.inst.pp) mismatch.push(`PP ${facts.pp}`);
  if (facts.maxModelLen && facts.maxModelLen !== v.ev.inst.maxModelLen) mismatch.push(`max_model_len ${facts.maxModelLen}`);

  return (
    <section className="border-b border-rule px-4 py-3">
      <button type="button" className="text-[15px] font-semibold tracking-tight hover:underline" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        {open ? '▾' : '▸'} Check against a real vLLM log
      </button>
      {open && (
        <div className="mt-2 flex flex-col gap-2">
          <textarea
            className="ctl num h-24 w-full resize-y font-mono text-[11.5px]"
            placeholder="Paste the startup log of `vllm serve` (lines like “Available KV cache memory: … GiB”)"
            value={text}
            onChange={(e) => setText(e.target.value)}
            aria-label="vLLM startup log"
          />
          {text && !found.length && <p className="text-[12.5px] text-muted">No memory lines found. Paste the log from engine start until “Graph capturing finished”.</p>}
          {mismatch.length > 0 && (
            <p className="text-[12.5px] text-warn">The log was produced with a different setup ({mismatch.join(', ')}); match the controls to compare like for like.</p>
          )}
          {found.length > 0 && (
            <table className="num w-full text-[12.5px]">
              <thead>
                <tr className="text-left text-muted">
                  <th className="font-normal">Metric</th>
                  <th className="text-right font-normal">vLLM</th>
                  <th className="text-right font-normal">vSim</th>
                  <th className="text-right font-normal">Δ</th>
                </tr>
              </thead>
              <tbody>
                {found.map((r) => {
                  const d = (r.ours - r.real!) / Math.max(1e-9, Math.abs(r.real!));
                  return (
                    <tr key={r.label}>
                      <td className="text-muted">{r.label}</td>
                      <td className="text-right">{r.fmt(r.real!)}</td>
                      <td className="text-right">{r.fmt(r.ours)}</td>
                      <td className={`text-right ${Math.abs(d) > 0.1 ? 'text-warn' : ''}`}>{`${d >= 0 ? '+' : ''}${(d * 100).toFixed(1)}%`}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      )}
    </section>
  );
}
