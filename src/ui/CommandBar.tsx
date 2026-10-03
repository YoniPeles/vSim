import { useState } from 'react';
import { serveCommand } from '../core/vllm/command.ts';
import { useDerivedContext } from './DerivedContext.tsx';

// The live `vllm serve` invocation is the page's title: every control edits this line.
export function CommandBar() {
  const { model, cluster, deployment } = useDerivedContext();
  const [copied, setCopied] = useState(false);
  const pd = deployment.instances.length > 1;
  const cmds = deployment.instances.map((inst) => ({
    role: inst.role,
    args: serveCommand(model, inst, cluster, pd),
  }));
  const text = cmds.map((c) => (pd ? `# ${c.role}\n` : '') + c.args.join(' \\\n  ')).join('\n\n');

  return (
    <header className="flex items-start gap-4 border-b border-rule bg-panel px-4 py-2.5">
      <div className="shrink-0 pt-[1px]">
        <div className="text-[19px] leading-none font-semibold tracking-tight">vSim</div>
        <div className="mt-1 text-[11.5px] text-muted">vLLM serving, simulated</div>
      </div>
      <div className="min-w-0 flex-1 font-mono text-[12.5px] leading-[1.55] text-[#cfe0ee]">
        {cmds.map((c) => (
          <div key={c.role} className="flex min-w-0 gap-2">
            {pd && <span className="w-14 shrink-0 text-muted">{c.role}</span>}
            <code className="min-w-0 break-words">
              <span className="text-[#7fb0de]">$ </span>
              {c.args.map((a, i) => (
                <span key={i} className={i === 0 ? 'text-text' : 'text-[#a9c1d6]'}>
                  {i > 0 ? ' ' : ''}
                  {a}
                </span>
              ))}
            </code>
          </div>
        ))}
      </div>
      <button
        type="button"
        className="ctl shrink-0 px-2.5 py-1 text-[12.5px] hover:bg-[#1b3550]"
        onClick={() => {
          void navigator.clipboard.writeText(text).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 1400);
          });
        }}
      >
        {copied ? 'Copied' : 'Copy command'}
      </button>
    </header>
  );
}
