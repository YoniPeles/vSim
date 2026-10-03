import type { ReactNode } from 'react';
import { allIssues } from '../core/engine/evaluate.ts';
import type { Issue, MemoryBreakdown, StepParts } from '../core/types.ts';
import { fmtBW, fmtBytes, fmtCount, fmtPct, fmtTime, fmtTokens } from '../core/units.ts';
import type { InstanceView } from '../state/derived.ts';
import { useApp } from '../state/store.ts';
import { useDerivedContext } from './DerivedContext.tsx';
import { gpuReplicas, kvInUse } from '../scene/sceneModel.ts';
import { KvGrid } from './KvGrid.tsx';
import { LogCompare } from './LogCompare.tsx';
import { totalGpus } from '../core/hardware/clusters.ts';
import { FrontierChart } from './FrontierChart.tsx';
import { Why } from './controls.tsx';
import { C, MEMORY_LEGEND, TRAFFIC_LEGEND } from './theme.ts';

function Block({ title, children, note }: { title: string; children: ReactNode; note?: ReactNode }) {
  return (
    <section className="border-b border-rule px-4 py-3">
      <h2 className="mb-1.5 text-[15px] font-semibold tracking-tight">{title}</h2>
      {children}
      {note && <p className="mt-1.5 text-[12px] leading-snug text-muted">{note}</p>}
    </section>
  );
}

function Stat({ label, value, why, sub }: { label: string; value: ReactNode; why?: string; sub?: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="text-[12.5px] text-muted">{label}</div>
      <div className="num text-[22px] leading-tight font-semibold">
        <Why why={why}>{value}</Why>
      </div>
      {sub && <div className="num text-[12px] text-muted">{sub}</div>}
    </div>
  );
}

function Row({ label, value, why }: { label: string; value: ReactNode; why?: string | undefined }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-[1px] text-[13px]">
      <span className="text-muted">{label}</span>
      <span className="num text-right">
        <Why why={why}>{value}</Why>
      </span>
    </div>
  );
}

function MemoryBar({ m, kvUsed }: { m: MemoryBreakdown; kvUsed: number }) {
  const segs = [
    { key: 'weights', v: m.weights, color: C.weights },
    { key: 'activations', v: m.actPeak, color: C.activations },
    { key: 'overhead', v: m.nonTorch + m.persistent + m.cudagraph, color: C.overhead },
    { key: 'kvUsed', v: Math.min(kvUsed, m.kv), color: C.kvUsed },
    { key: 'kvFree', v: Math.max(0, m.kv - kvUsed), color: C.kvFree },
  ];
  const total = m.total;
  return (
    <div>
      <div className="flex h-3.5 w-full gap-[2px] overflow-hidden rounded-[3px] bg-[#16293b]" role="img" aria-label="GPU memory breakdown">
        {segs.map((s) =>
          s.v > 0 ? (
            <div
              key={s.key}
              title={`${MEMORY_LEGEND.find((l) => l.key === s.key)?.label}: ${fmtBytes(s.v)}`}
              style={{
                width: `${(s.v / total) * 100}%`,
                background: s.key === 'kvFree' ? `repeating-linear-gradient(135deg, ${s.color}66 0 3px, ${s.color}22 3px 6px)` : s.color,
              }}
            />
          ) : null,
        )}
      </div>
      <ul className="mt-1.5 grid grid-cols-1 gap-y-0.5 text-[12.5px]">
        {MEMORY_LEGEND.map((l, i) => (
          <li key={l.key} className="flex items-center gap-1.5">
            <span
              className="inline-block h-2.5 w-2.5 shrink-0 rounded-[2px]"
              style={{ background: l.key === 'kvFree' ? `repeating-linear-gradient(135deg, ${l.color}aa 0 2px, ${l.color}33 2px 4px)` : l.color }}
            />
            <span className="text-muted">{l.label}</span>
            <span className="num ml-auto">{fmtBytes(segs[i]!.v)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

const PART_LABELS: [keyof StepParts, string][] = [
  ['gemm', 'Dense GEMMs'],
  ['attn', 'Attention'],
  ['moe', 'Routed experts'],
  ['tpComm', 'TP all-reduce'],
  ['epComm', 'MoE all-to-all'],
  ['ppComm', 'PP hand-off'],
  ['lmHead', 'LM head + sampler'],
  ['draft', 'Drafting'],
  ['launch', 'Kernel launch'],
];

function StepAnatomy({ parts, total }: { parts: StepParts; total: number }) {
  const rows = PART_LABELS.filter(([k]) => parts[k] > total * 0.005);
  const max = Math.max(...rows.map(([k]) => parts[k]));
  return (
    <div className="grid grid-cols-[7.5rem_1fr_3.6rem] items-center gap-x-2 gap-y-[3px] text-[12.5px]">
      {rows.map(([k, label]) => (
        <div key={k} className="contents">
          <span className="text-muted">{label}</span>
          <span className="h-2 rounded-r-[3px] bg-[#6d8aa6]" style={{ width: `${Math.max(2, (parts[k] / max) * 100)}%` }} />
          <span className="num text-right">{fmtTime(parts[k])}</span>
        </div>
      ))}
    </div>
  );
}

function Meter({ label, util, color, detail }: { label: string; util: number; color: string; detail: string }) {
  return (
    <div className="text-[13px]">
      <div className="flex items-baseline justify-between">
        <span className="flex items-center gap-1.5 text-muted">
          <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: color }} />
          {label}
        </span>
        <span className="num">{fmtPct(Math.min(util, 9.99), util < 0.1 ? 1 : 0)}</span>
      </div>
      <div className="mt-0.5 h-1.5 w-full rounded-full bg-[#1a3046]">
        <div className="h-full rounded-full" style={{ width: `${Math.min(100, util * 100)}%`, background: util > 0.85 ? C.error : '#7f9cb8' }} />
      </div>
      <div className="mt-0.5 text-[11.5px] text-faint">{detail}</div>
    </div>
  );
}

function IssueList({ issues }: { issues: Issue[] }) {
  if (!issues.length) return null;
  const order = { error: 0, warn: 1, info: 2 } as const;
  const sorted = [...issues].sort((a, b) => order[a.level] - order[b.level]);
  return (
    <ul className="flex flex-col gap-1.5 px-4 py-3 text-[12.5px] leading-snug">
      {sorted.map((i, k) => (
        <li key={k} className="flex gap-2">
          <span
            aria-label={i.level}
            className={`mt-[3px] inline-block h-2 w-2 shrink-0 rounded-full ${i.level === 'error' ? 'bg-error' : i.level === 'warn' ? 'bg-warn' : 'bg-[#5d7489]'}`}
          />
          <span className={i.level === 'error' ? 'text-[#ffb3b3]' : i.level === 'warn' ? 'text-text' : 'text-muted'}>{i.msg}</span>
        </li>
      ))}
    </ul>
  );
}

function PrefillReadout({ v }: { v: InstanceView }) {
  const { pd } = useDerivedContext();
  const workload = useApp((s) => s.workload);
  if (!pd) return null;
  const u = pd.utilization;
  return (
    <Block
      title="Prefill capacity"
      note={u >= 1 ? 'The decode side finishes requests faster than prefill can admit new ones: prompts queue up and TTFT grows without bound.' : undefined}
    >
      <div className="grid grid-cols-2 gap-3">
        <Stat label="Prompts per second" value={fmtCount(pd.capacity.reqPerSec, 1)} sub={`${fmtCount(pd.capacity.tokPerSec * v.ev.inst.dp, 1)} prompt tok/s`} />
        <Stat label="Busy" value={fmtPct(Math.min(u, 9.99))} sub={`${fmtCount(pd.demand, 1)} prompts/s demanded`} />
        <Stat label={`TTFT, idle (${fmtTokens(workload.isl)} prompt)`} value={fmtTime(v.ttft)} />
        <Stat label="TTFT under load" value={Number.isFinite(pd.ttft) ? fmtTime(pd.ttft) : '∞'} sub="queue + prefill + KV transfer + first decode" />
      </div>
    </Block>
  );
}

function InstanceReadout({ v, title }: { v: InstanceView; title?: string }) {
  const { ev, steady } = v;
  const { pd } = useDerivedContext();
  const { inst, memory, placement } = ev;
  const selected = useApp((s) => s.selectedGpu);
  const workload = useApp((s) => s.workload);
  const rankIdx = Math.max(0, placement.shards.findIndex((s) => s.gpu === selected));
  const m = memory.perRank[rankIdx]!;
  const shard = placement.shards[rankIdx]!;
  const kvUsed = kvInUse(v, rankIdx);
  const gpus = inst.world;
  const role = inst.role;

  return (
    <>
      {title && <div className="border-b border-rule bg-panel-2 px-4 py-1.5 text-[13px] font-semibold">{title}</div>}
      <Block
        title={`Memory on GPU ${shard.gpu}`}
        note={
          selected === null ? 'Click a GPU in the scene to inspect it.' : `DP ${shard.dpRank}, PP ${shard.ppRank}, TP ${shard.tpRank}; layers ${shard.layerLo}–${shard.layerHi - 1}.`
        }
      >
        <MemoryBar m={m} kvUsed={kvUsed} />
        <div className="mt-2 grid grid-cols-2 gap-3">
          <Stat
            label="KV capacity per replica"
            value={`${fmtTokens(memory.kvTokens)} tok`}
            why={memory.why['kvTokens']}
            sub={`${fmtBytes(m.kvBytesPerToken)} per token per GPU`}
          />
          <Stat
            label={`Fits at ${fmtTokens(inst.maxModelLen)} context`}
            value={`${memory.maxConcurrency.toFixed(memory.maxConcurrency < 10 ? 2 : 0)}×`}
            why={memory.why['maxConcurrency']}
            sub={steady ? `${fmtCount(steady.kvCapacity, 1)} requests at ${fmtTokens(workload.isl + workload.osl)}` : undefined}
          />
        </div>
        <div className="mt-2">
          <Row label="Requested (util)" value={fmtBytes(m.requested)} why={m.why['requested']} />
          <Row label="Weights" value={fmtBytes(m.weights)} why={m.why['weights']} />
          <Row label="Activation peak" value={fmtBytes(m.actPeak)} why={m.why['actPeak']} />
          <Row label="CUDA graphs" value={fmtBytes(m.cudagraph)} why={m.why['cudagraph'] ?? inst.why['cudagraph']} />
          <Row label="KV cache" value={fmtBytes(m.kv)} why={m.why['kv']} />
        </div>
      </Block>

      {role === 'prefill' && v.cost ? (
        <PrefillReadout v={v} />
      ) : v.cost && steady ? (
        <>
          <Block title="One user, idle server">
            <div className="grid grid-cols-2 gap-3">
              {role === 'decode' && pd ? (
                <Stat label="Time to first token" value={fmtTime(pd.ttft)} sub="via the prefill instance" />
              ) : (
                <Stat
                  label={`Time to first token (${fmtTokens(workload.isl)} prompt)`}
                  value={fmtTime(v.ttft)}
                  sub={`${Math.ceil(workload.isl / inst.maxNumBatchedTokens)} chunk(s) of ≤${fmtTokens(inst.maxNumBatchedTokens)}`}
                />
              )}
              {v.decodeB1 && (
                <Stat
                  label="Decode speed"
                  value={`${Math.round((1 + steady.acceptedPerStep) / v.decodeB1.time)} tok/s`}
                  sub={`${fmtTime(v.decodeB1.time)} per step, ${v.decodeB1.bound}-bound`}
                />
              )}
            </div>
          </Block>
          <Block
            title={`Under load: ${fmtCount(steady.batch * inst.dp, 1)} concurrent users`}
            note={
              steady.kvLimited
                ? `KV cache holds only ${fmtCount(steady.kvCapacity * inst.dp, 1)} requests of this size; the rest wait in the queue.`
                : steady.prefillLimited
                  ? 'Prompts arrive faster than the token budget can prefill them; TTFT grows with the queue.'
                  : undefined
            }
          >
            <div className="grid grid-cols-2 gap-3">
              <Stat label="Per user" value={`${Math.round(steady.tokPerUser)} tok/s`} sub={`ITL ${fmtTime(steady.itl)}`} />
              <Stat label="Time to first token" value={role === 'decode' && pd ? (Number.isFinite(pd.ttft) ? fmtTime(pd.ttft) : '∞') : fmtTime(steady.ttft)} />
              <Stat label="Output throughput" value={`${fmtCount(steady.throughput, 1)} tok/s`} sub={`${fmtCount(steady.throughput / gpus, 1)} per GPU`} />
              <Stat
                label={`Users at ≥${workload.slaTokPerSec} tok/s, TTFT ≤${workload.slaTtft}s`}
                value={fmtCount(v.maxUsers, 1)}
                sub={inst.dp > 1 ? `across ${inst.dp} replicas` : undefined}
              />
            </div>
            {v.sweep.length > 1 && (
              <div className="mt-3">
                <FrontierChart
                  points={v.sweep}
                  color={C.tp}
                  current={{ users: steady.batch * inst.dp, tokPerUser: steady.tokPerUser, throughput: steady.throughput, ttft: steady.ttft, itl: steady.itl }}
                />
              </div>
            )}
          </Block>
          <Block title={`One step: ${fmtTime(steady.step.time)}`} note={`Mostly ${steady.step.bound}-bound. ${Math.round(steady.prefillTokensPerStep)} prefill tokens ride along with ${Math.round(steady.batch / (inst.pp > 1 ? inst.pp : 1))} decodes.`}>
            <StepAnatomy parts={steady.step.parts} total={steady.step.gpu} />
          </Block>
          <Block title="Interconnect">
            <div className="flex flex-col gap-2">
              <Meter
                label={v.cost.cluster.scaleUp === 'pcie' ? 'PCIe' : v.cost.cluster.scaleUp === 'mesh' ? 'Infinity Fabric' : 'NVLink'}
                util={steady.scaleUpUtil}
                color={steady.step.comm.ep.intra > steady.step.comm.tp.intra ? C.ep : C.tp}
                detail={`${fmtBW(v.cost.cluster.scaleUp === 'pcie' ? v.cost.cluster.gpu.pcieBWDir : v.cost.cluster.gpu.scaleUpBWDir)} per GPU per direction`}
              />
              <Meter
                label={v.cost.cluster.scaleOut.kind.split(' /')[0] ?? 'Network'}
                util={steady.scaleOutUtil}
                color={C.ep}
                detail={`${fmtBW(v.cost.cluster.scaleOut.bwPerGpuDir)} per GPU per direction`}
              />
              <ul className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[11.5px] text-muted">
                {TRAFFIC_LEGEND.map((t) => (
                  <li key={t.key} className="flex items-center gap-1">
                    <span className="inline-block h-2 w-2 rounded-full" style={{ background: t.color }} />
                    {t.label}
                  </li>
                ))}
              </ul>
            </div>
          </Block>
        </>
      ) : (
        <Block title="Performance">
          <p className="text-[13px] text-muted">Fix the errors below to see latency and throughput.</p>
        </Block>
      )}
    </>
  );
}

export function MetricsPane() {
  const d = useDerivedContext();
  const issues = allIssues(d.evaluation);
  const selected = useApp((s) => s.selectedGpu);
  const replicas = gpuReplicas(d.views, totalGpus(d.cluster));
  const firstGpu = d.views[d.views.length - 1]?.ev.placement.shards[0]?.gpu ?? 0;
  const gpu = selected !== null && replicas[selected]! >= 0 ? selected : firstGpu;
  const replica = Math.max(0, replicas[gpu] ?? 0);
  return (
    <div className="scroll-thin h-full overflow-y-auto">
      <KvGrid replica={replica} label={`replica of GPU ${gpu}`} />
      {d.views.map((v, i) => (
        <InstanceReadout
          key={v.ev.inst.id}
          v={v}
          title={d.views.length > 1 ? (i === 0 ? `Prefill instance, ${v.ev.inst.world} GPUs` : `Decode instance, ${v.ev.inst.world} GPUs`) : undefined}
        />
      ))}
      {d.kvTransfer && (
        <Block title="KV transfer per request">
          <Row label="Bytes moved" value={fmtBytes(d.kvTransfer.bytes)} />
          <Row label="Transfer time" value={fmtTime(d.kvTransfer.time)} why="Prefill GPUs push their KV shards in parallel over NIXL; includes connector setup latency." />
        </Block>
      )}
      <IssueList issues={issues} />
      <LogCompare />
    </div>
  );
}
