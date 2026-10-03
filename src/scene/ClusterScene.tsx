import { useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { OrbitControls, PerformanceMonitor } from '@react-three/drei';
import { Bloom, EffectComposer } from '@react-three/postprocessing';
import { useApp } from '../state/store.ts';
import { useDerivedContext } from '../ui/DerivedContext.tsx';
import { C } from '../ui/theme.ts';
import { layoutCluster, PITCH, TOWER_H, type Vec3 } from './layout.ts';
import { gpuReplicas, gpuVisuals, intensity, linkVisuals, liveUtil, plateVisuals, type LinkVisual } from './sceneModel.ts';
import { useSim } from '../state/sim.ts';
import { phaseAt, useTrace } from '../state/trace.ts';
import { Boxes, BoardLabels, GhostStacks, GpuTiles, GpuTooltip, GroupOutlines, HbmTowers, LayerPlates, Particles, Wiring } from './parts.tsx';
import { totalGpus } from '../core/hardware/clusters.ts';

export function ClusterScene() {
  const d = useDerivedContext();
  const hovered = useApp((s) => s.hoveredGpu);
  const selected = useApp((s) => s.selectedGpu);
  const hoverGpu = useApp((s) => s.hoverGpu);
  const selectGpu = useApp((s) => s.selectGpu);
  const [fx, setFx] = useState(true);

  const layout = useMemo(() => layoutCluster(d.cluster), [d.cluster]);
  const gpus = useMemo(() => gpuVisuals(d.views, totalGpus(d.cluster), d.cluster.gpu.memBytes), [d.views, d.cluster]);
  const plates = useMemo(() => {
    const p = plateVisuals(d.model, gpus, d.views);
    // Level of detail: very large clusters only draw plates for the first 1500 GPU-layers per stage.
    return p.length > 40000 ? p.filter((_, i) => i % Math.ceil(p.length / 40000) === 0) : p;
  }, [d.model, gpus, d.views]);
  const kvxUtil = d.kvTransfer?.util ?? 0;
  const links = useMemo(() => linkVisuals(layout, d.views, kvxUtil), [layout, d.views, kvxUtil]);
  const replicaOf = useMemo(() => gpuReplicas(d.views, totalGpus(d.cluster)), [d.views, d.cluster]);

  // Live simulation drives tower KV and particle density; otherwise the analytical steady state does.
  const liveKv = useMemo(
    () => (gpu: number) => {
      const f = useSim.getState().frame;
      const r = replicaOf[gpu]!;
      if (!f || r < 0) return null;
      const rep = f.replicas[r];
      return rep ? rep.kvUsed / Math.max(1, rep.kvTotal) : null;
    },
    [replicaOf],
  );
  const level = useMemo(
    () => (l: LinkVisual) => {
      const run = useTrace.getState().run;
      if (run) {
        // Step trace: only the link family of the current phase lights up.
        const i = phaseAt(run, performance.now() / 1000);
        const k = i >= 0 ? run.phases[i]!.kind : null;
        if (l.inst !== run.inst) return 0;
        if (k === 'tp') return l.kind === 'tp' ? 1 : 0;
        if (k === 'dispatch' || k === 'combine') return l.kind === 'ep' ? 1 : 0;
        if (k === 'pp') return l.kind === 'pp' ? 1 : 0;
        return 0;
      }
      const f = useSim.getState().frame;
      return intensity(f ? liveUtil(l, f.links[l.inst]) : l.util);
    },
    [],
  );
  const expertLoad = useMemo(() => () => useSim.getState().frame?.expertLoad ?? null, []);
  const traceWasActive = useRef(false);
  const plateHeat = useMemo(() => {
    return (out: Float32Array) => {
      const run = useTrace.getState().run;
      if (!run) {
        if (!traceWasActive.current) return false;
        out.fill(0);
        traceWasActive.current = false;
        return true;
      }
      traceWasActive.current = true;
      const i = phaseAt(run, performance.now() / 1000);
      const ph = i >= 0 ? run.phases[i]! : null;
      // Find the layer the wavefront is on (LM head / hand-off phases keep the last layer lit).
      let layer = -1;
      for (let j = i; j >= 0 && layer < 0; j--) layer = run.phases[j]!.layer;
      for (let p = 0; p < plates.length && p < out.length; p++) {
        const pl = plates[p]!;
        if (pl.inst !== run.inst || !ph) {
          out[p] = 0;
          continue;
        }
        out[p] = pl.layer === layer ? (ph.kind === 'attn' || ph.kind === 'ffn' ? 0.9 : 0.45) : pl.layer < layer ? 0.08 : 0;
      }
      return true;
    };
  }, [plates]);

  const outlines = useMemo(() => {
    const rects: { min: Vec3; max: Vec3; label: string; color: string }[] = [];
    d.views.forEach((v) => {
      const { inst } = v.ev;
      const byDp = new Map<number, number[]>();
      for (const s of v.ev.placement.shards) {
        const arr = byDp.get(s.dpRank) ?? [];
        arr.push(s.gpu);
        byDp.set(s.dpRank, arr);
      }
      const roleName = inst.role === 'prefill' ? 'Prefill' : inst.role === 'decode' ? 'Decode' : 'Replica';
      const shape = `TP${inst.tp}${inst.pp > 1 ? ` PP${inst.pp}` : ''}${inst.ep ? ` EP${inst.epSize}` : ''}`;
      // One outline per DP replica, unless replicas are single GPUs or sit on a ring (mesh), where
      // bounding boxes would overlap: then one outline per instance.
      const single = (inst.world / inst.dp === 1 || layout.meshEdges.length > 0) && inst.dp > 1;
      const groups = single ? [[...byDp.values()].flat()] : [...byDp.values()];
      groups.forEach((g, i) => {
        const pts = g.map((x) => layout.gpus[x]).filter((x) => !!x);
        if (!pts.length) return;
        const min = { x: Infinity, y: 0, z: Infinity };
        const max = { x: -Infinity, y: 0, z: -Infinity };
        for (const p of pts) {
          min.x = Math.min(min.x, p.pos.x - PITCH / 2 + 0.05);
          max.x = Math.max(max.x, p.pos.x + PITCH / 2 - 0.05);
          min.z = Math.min(min.z, p.pos.z - PITCH / 2 + 0.02);
          max.z = Math.max(max.z, p.pos.z + PITCH / 2 - 0.02);
        }
        const label =
          groups.length === 1 && inst.dp > 1
            ? `${roleName}s: DP${inst.dp} × ${shape}`
            : `${roleName}${inst.dp > 1 ? ` ${i}` : ''}: ${shape}`;
        rects.push({ min, max, label, color: C.text });
      });
    });
    return rects;
  }, [d.views, layout]);

  const center = useMemo(() => {
    const b = layout.bounds;
    return [(b.min.x + b.max.x) / 2, TOWER_H * 0.35, (b.min.z + b.max.z) / 2] as [number, number, number];
  }, [layout]);

  const hoveredVisual = hovered !== null ? gpus[hovered] : selected !== null ? gpus[selected] : undefined;
  const hoveredLabel = (() => {
    const g = hoveredVisual;
    if (!g?.shard) return '';
    const role = d.views[g.instance]?.ev.inst.role;
    const sh = g.shard;
    const ranks = `DP ${sh.dpRank}, PP ${sh.ppRank}, TP ${sh.tpRank}`;
    return role === 'prefill' ? `Prefill, ${ranks}` : role === 'decode' ? `Decode, ${ranks}` : ranks;
  })();

  return (
    <Canvas
      dpr={[1, 2]}
      camera={{ position: [8, 10, 12], fov: 38, near: 0.1, far: 2000 }}
      gl={{ antialias: true }}
      onPointerMissed={() => selectGpu(null)}
    >
      <PerformanceMonitor onDecline={() => setFx(false)} />
      <color attach="background" args={[C.bg]} />
      <fog attach="fog" args={[C.bg, 30, 160]} />
      <ambientLight intensity={0.55} />
      <directionalLight position={[8, 14, 10]} intensity={1.35} />
      <directionalLight position={[-10, 6, -8]} intensity={0.35} color={'#9fc3ff'} />
      <Boxes boxes={layout.platforms} color={C.platform} />
      <Boxes boxes={layout.boards} color={C.board} />
      <Boxes boxes={layout.switches} color={'#28465f'} emissive={'#1b3a55'} />
      {layout.spine && <Boxes boxes={[layout.spine]} color={'#28465f'} emissive={'#1b3a55'} />}
      <GpuTiles layout={layout} gpus={gpus} hovered={hovered} selected={selected} onHover={hoverGpu} onSelect={selectGpu} />
      <HbmTowers layout={layout} gpus={gpus} liveKv={liveKv} />
      <LayerPlates layout={layout} plates={plates} heat={plateHeat} experts={expertLoad} />
      <GhostStacks layout={layout} plates={plates} />
      <Wiring layout={layout} links={links} />
      <Particles links={links} level={level} />
      <GroupOutlines rects={outlines} />
      <BoardLabels layout={layout} />
      <GpuTooltip layout={layout} g={hoveredVisual} label={hoveredLabel} />
      <OrbitControls makeDefault enableDamping dampingFactor={0.12} maxPolarAngle={Math.PI * 0.49} />
      <CameraRig bounds={layout.bounds} center={center} />
      {fx && (
        <EffectComposer multisampling={4}>
          <Bloom luminanceThreshold={0.85} luminanceSmoothing={0.2} intensity={0.9} mipmapBlur />
        </EffectComposer>
      )}
    </Canvas>
  );
}

/** Frames the cluster whenever its layout changes, using the narrower of the two fields of view. */
function CameraRig({ bounds, center }: { bounds: { min: Vec3; max: Vec3 }; center: [number, number, number] }) {
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera;
  const size = useThree((s) => s.size);
  const controls = useThree((s) => s.controls) as unknown as { target: THREE.Vector3; update: () => void } | null;
  useEffect(() => {
    const r = 0.5 * Math.hypot(bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y, bounds.max.z - bounds.min.z);
    const vFov = (camera.fov * Math.PI) / 180;
    const hFov = 2 * Math.atan(Math.tan(vFov / 2) * (size.width / Math.max(1, size.height)));
    const dist = Math.max(5, (r / Math.sin(Math.min(vFov, hFov) / 2)) * 0.96);
    const dir = new THREE.Vector3(0.3, 0.62, 0.74).normalize();
    camera.position.set(center[0] + dir.x * dist, center[1] + dir.y * dist, center[2] + dir.z * dist);
    camera.lookAt(...center);
    if (controls) {
      controls.target.set(...center);
      controls.update();
    }
    // Refit only when the cluster geometry changes, not on every resize.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bounds, center, controls]);
  return null;
}
