import { useId, useState, type ReactNode } from 'react';
import { Popover } from 'radix-ui';

export function Section({ title, children, aside }: { title: string; children: ReactNode; aside?: ReactNode }) {
  return (
    <section className="border-b border-rule px-4 py-3">
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <h2 className="text-[15px] font-semibold tracking-tight text-text">{title}</h2>
        {aside}
      </div>
      <div className="flex flex-col gap-2">{children}</div>
    </section>
  );
}

export function Field({ label, hint, children, htmlFor }: { label: string; hint?: string; children: ReactNode; htmlFor?: string }) {
  return (
    <div className="grid grid-cols-[minmax(0,7.5rem)_minmax(0,1fr)] items-center gap-x-2">
      <label htmlFor={htmlFor} className="truncate text-[13px] text-muted" title={hint ?? label}>
        {label}
      </label>
      <div className="flex min-w-0 items-center gap-1.5">{children}</div>
    </div>
  );
}

export function Segmented<T extends string | number>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label?: string; disabled?: boolean; title?: string }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="flex min-w-0 flex-wrap overflow-hidden rounded border border-rule">
      {options.map((o) => {
        const on = o.value === value;
        return (
          <button
            key={String(o.value)}
            type="button"
            role="radio"
            aria-checked={on}
            disabled={o.disabled}
            title={o.title}
            onClick={() => onChange(o.value)}
            className={`num min-w-7 flex-1 px-1.5 py-[3px] text-[13px] transition-colors ${
              on ? 'bg-[#244361] text-text' : 'bg-panel-2 text-muted hover:text-text'
            } ${o.disabled ? 'cursor-not-allowed opacity-35' : ''} border-r border-rule last:border-r-0`}
          >
            {o.label ?? String(o.value)}
          </button>
        );
      })}
    </div>
  );
}

export function NumberInput({
  value,
  onChange,
  min,
  max,
  step = 1,
  placeholder,
  id,
  width = 'w-24',
  suffix,
}: {
  value: number | null;
  onChange: (v: number | null) => void;
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
  id?: string;
  width?: string;
  suffix?: string;
}) {
  const [text, setText] = useState(value === null ? '' : String(value));
  // Re-sync the draft text when the value changes from outside (adjust state during render).
  const [prev, setPrev] = useState(value);
  if (value !== prev) {
    setPrev(value);
    setText(value === null ? '' : String(value));
  }
  const commit = () => {
    if (text.trim() === '') return onChange(null);
    let n = Number(text.replace(/[_,\s]/g, '').replace(/k$/i, '000'));
    if (!Number.isFinite(n)) return setText(value === null ? '' : String(value));
    if (min !== undefined) n = Math.max(min, n);
    if (max !== undefined) n = Math.min(max, n);
    onChange(n);
  };
  return (
    <span className="flex items-center gap-1">
      <input
        id={id}
        className={`ctl num ${width}`}
        inputMode="decimal"
        value={text}
        step={step}
        placeholder={placeholder}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') commit();
        }}
      />
      {suffix && <span className="text-[12px] text-faint">{suffix}</span>}
    </span>
  );
}

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className={`relative h-[18px] w-8 shrink-0 rounded-full border border-rule transition-colors ${checked ? 'bg-[#1f6b66]' : 'bg-panel-2'}`}
    >
      <span
        className={`absolute top-[2px] left-0 h-3 w-3 rounded-full bg-text transition-transform ${checked ? 'translate-x-[15px]' : 'translate-x-[2px]'}`}
      />
    </button>
  );
}

export function Select<T extends string>({
  value,
  options,
  onChange,
  id,
  className = '',
}: {
  value: T;
  options: { value: T; label: string; group?: string }[];
  onChange: (v: T) => void;
  id?: string;
  className?: string;
}) {
  const groups = [...new Set(options.map((o) => o.group ?? ''))];
  return (
    <select id={id} className={`ctl w-full ${className}`} value={value} onChange={(e) => onChange(e.target.value as T)}>
      {groups.map((g) =>
        g ? (
          <optgroup key={g} label={g}>
            {options
              .filter((o) => o.group === g)
              .map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
          </optgroup>
        ) : (
          options
            .filter((o) => !o.group)
            .map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))
        ),
      )}
    </select>
  );
}

export function Slider({
  value,
  min,
  max,
  step,
  onChange,
  label,
}: {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  label: string;
}) {
  const id = useId();
  return <input id={id} aria-label={label} className="range" type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />;
}

/** A number with a "how was this computed" popover. */
export function Why({ children, why }: { children: ReactNode; why?: string | undefined }) {
  if (!why) return <>{children}</>;
  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        <button type="button" className="cursor-help underline decoration-faint decoration-dotted underline-offset-[3px] hover:decoration-muted">
          {children}
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content sideOffset={6} className="tooltip z-50 max-w-80 whitespace-normal" collisionPadding={12}>
          {why}
          <Popover.Arrow className="fill-[#223a52]" />
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
