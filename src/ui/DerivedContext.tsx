import { createContext, useContext, type ReactNode } from 'react';
import { useDerived, type Derived } from '../state/derived.ts';

const Ctx = createContext<Derived | null>(null);

/** Computes the evaluation once per input change and shares it with every pane. */
export function DerivedProvider({ children }: { children: ReactNode }) {
  const d = useDerived();
  return <Ctx.Provider value={d}>{children}</Ctx.Provider>;
}

export function useDerivedContext(): Derived {
  const d = useContext(Ctx);
  if (!d) throw new Error('useDerivedContext outside DerivedProvider');
  return d;
}
