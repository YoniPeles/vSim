import { CLUSTER_PRESETS, SCALE_OUT } from '../../core/hardware/clusters.ts';
import { GPUS } from '../../core/hardware/gpus.ts';
import { fmtBW, fmtBytes } from '../../core/units.ts';
import { useApp } from '../../state/store.ts';
import { useDerivedContext } from '../DerivedContext.tsx';
import { Field, NumberInput, Section, Select } from '../controls.tsx';

export function HardwarePanel() {
  const preset = useApp((s) => s.clusterPreset);
  const build = useApp((s) => s.cluster);
  const setPreset = useApp((s) => s.setClusterPreset);
  const setCluster = useApp((s) => s.setCluster);
  const { cluster } = useDerivedContext();
  const g = cluster.gpu;
  const tf = (x?: number) => (x ? `${Math.round(x / 1e12).toLocaleString('en-US')}` : '–');

  return (
    <Section title="Hardware">
      <Select
        value={preset ?? 'custom'}
        options={[...CLUSTER_PRESETS.map((c) => ({ value: c.id, label: c.label })), { value: 'custom', label: 'Custom cluster…' }]}
        onChange={(v) => v !== 'custom' && setPreset(v)}
      />
      <Field label="GPU">
        <Select value={build.gpu} options={GPUS.map((x) => ({ value: x.id, label: x.name }))} onChange={(v) => setCluster({ gpu: v })} />
      </Field>
      <Field label="Nodes × GPUs">
        <NumberInput value={build.nodes} min={1} max={64} width="w-14" onChange={(v) => setCluster({ nodes: v ?? 1 })} />
        <span className="text-faint">×</span>
        <NumberInput value={build.gpusPerNode} min={1} max={8} width="w-14" onChange={(v) => setCluster({ gpusPerNode: v ?? 1 })} />
      </Field>
      <Field label="Scale-out">
        <Select value={build.scaleOut} options={SCALE_OUT.map((s) => ({ value: s.id, label: s.label }))} onChange={(v) => setCluster({ scaleOut: v })} />
      </Field>
      <p className="num text-[12.5px] leading-snug text-muted">
        {g.memLabel} ({fmtBytes(g.memBytes)} usable), {fmtBW(g.hbmBW)}. Dense TFLOPS BF16 {tf(g.peak.bf16)} / FP8 {tf(g.peak.fp8)} / FP4{' '}
        {tf(g.peak.fp4)}.{' '}
        {cluster.scaleUp === 'nvswitch'
          ? `NVLink ${fmtBW(g.scaleUpBWDir)} per direction, ${cluster.domainSize}-GPU domain${cluster.nvls ? ' with NVLS' : ''}.`
          : cluster.scaleUp === 'mesh'
            ? `Infinity Fabric mesh, ${fmtBW(g.meshLinkBWDir ?? 0)} per peer link.`
            : 'PCIe only, no NVLink.'}
        {g.approximate ? ' Specs approximate.' : ''}
      </p>
    </Section>
  );
}
