import { Suspense, lazy } from 'react';
import { DerivedProvider, useDerivedContext } from './DerivedContext.tsx';
import { CommandBar } from './CommandBar.tsx';
import { ModelPanel } from './sidebar/ModelPanel.tsx';
import { HardwarePanel } from './sidebar/HardwarePanel.tsx';
import { FlagsPanel } from './sidebar/FlagsPanel.tsx';
import { WorkloadPanel } from './sidebar/WorkloadPanel.tsx';
import { MetricsPane } from './MetricsPane.tsx';
import { useUrlState } from '../state/urlState.ts';
import { MEMORY_LEGEND, TRAFFIC_LEGEND } from './theme.ts';
import { SimPanel } from './SimPanel.tsx';
import { TraceHud } from './TraceHud.tsx';

const ClusterScene = lazy(() => import('../scene/ClusterScene.tsx').then((m) => ({ default: m.ClusterScene })));

function SceneLegend() {
  const { model } = useDerivedContext();
  const moe = model.layers.some((l) => l.ffn.kind === 'moe');
  return (
    <div className="pointer-events-none absolute bottom-3 left-3 flex max-w-[calc(100%-1.5rem)] flex-wrap gap-x-4 gap-y-1 rounded bg-[#0d1926cc] px-3 py-2 text-[12px] text-muted backdrop-blur-sm">
      <span className="text-text">HBM stacks: GPU memory</span>
      {MEMORY_LEGEND.map((l) => (
        <span key={l.key} className="flex items-center gap-1">
          <span
            className="inline-block h-2.5 w-2.5 rounded-[2px]"
            style={{ background: l.key === 'kvFree' ? `repeating-linear-gradient(135deg, ${l.color}aa 0 2px, ${l.color}33 2px 4px)` : l.color }}
          />
          {l.label}
        </span>
      ))}
      <span className="basis-full" />
      <span className="text-text">
        Hologram: a line per layer held, bright = this GPU's TP share
        {moe ? ', cells = its experts (brighter = busier)' : ''}
      </span>
      {TRAFFIC_LEGEND.map((l) => (
        <span key={l.key} className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-full" style={{ background: l.color }} />
          {l.label}
        </span>
      ))}
      <span className="text-faint">Double-click a GPU to fly to it, Esc to return</span>
    </div>
  );
}

function Shell() {
  useUrlState();
  return (
    <div className="grid h-full grid-rows-[auto_minmax(0,1fr)]">
      <CommandBar />
      <div className="grid min-h-0 grid-cols-[340px_minmax(0,1fr)_370px] max-[1100px]:grid-cols-[300px_minmax(0,1fr)] max-[760px]:grid-cols-1 max-[760px]:overflow-y-auto">
        <aside className="scroll-thin min-h-0 overflow-y-auto border-r border-rule bg-panel max-[760px]:order-2 max-[760px]:overflow-visible">
          <ModelPanel />
          <HardwarePanel />
          <FlagsPanel />
          <WorkloadPanel />
        </aside>
        <main className="grid min-h-0 grid-rows-[minmax(0,1fr)_auto] max-[760px]:h-[85vh]">
          <div className="relative min-h-0">
            <Suspense fallback={<div className="grid h-full place-items-center text-muted">Building the cluster…</div>}>
              <ClusterScene />
            </Suspense>
            <SceneLegend />
            <TraceHud />
          </div>
          <SimPanel />
        </main>
        <aside className="min-h-0 border-l border-rule bg-panel max-[1100px]:col-span-2 max-[1100px]:border-t max-[760px]:col-span-1">
          <MetricsPane />
        </aside>
      </div>
    </div>
  );
}

export function App() {
  return (
    <DerivedProvider>
      <Shell />
    </DerivedProvider>
  );
}
