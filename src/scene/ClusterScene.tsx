import { useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { Environment, Lightformer, OrbitControls, PerformanceMonitor } from '@react-three/drei';
import { Bloom, EffectComposer, Vignette } from '@react-three/postprocessing';
import { useApp } from '../state/store.ts';
import { useDerivedContext } from '../ui/DerivedContext.tsx';
import { layoutCluster, PITCH, TOWER_H, type Vec3 } from './layout.ts';
import { gpuReplicas, gpuVisuals, intensity, linkVisuals, liveUtil, plateVisuals, type LinkVisual } from './sceneModel.ts';
import { useSim } from '../state/sim.ts';
import { phaseAt, useTrace } from '../state/trace.ts';
import { totalGpus } from '../core/hardware/clusters.ts';
import { Buses, Decals, Floor, FLOOR_Y, HitTargets, OVERLAY_LAYER, Packages, Sky, Trays } from './hardware.tsx';
import { HbmColumns } from './memory.tsx';
import { LayerStacks } from './hologram.tsx';
import { LinkLines, Wiring } from './traffic.tsx';
import { FloorLabels, GpuTooltip, GroupOutlines, SelectionMarkers, type GroupRect } from './annotations.tsx';
import { prefersReducedMotion } from './instancing.ts';

/** Scene background: a deeper ink than the panels, so the light in the scene carries. */
const VOID = '#0a1621';
const ZENITH = '#04090f';

/** Render quality: 2 = bloom + floor reflections, 1 = bloom only, 0 = neither (and no MSAA). */
type Quality = 0 | 1 | 2;

/** `?quality=0|1|2` pins the render quality (and stops the automatic step-down). */
function pinnedQuality(): Quality | null {
  const q = typeof location !== 'undefined' ? new URLSearchParams(location.search).get('quality') : null;
  return q === '0' || q === '1' || q === '2' ? (Number(q) as Quality) : null;
}

/** CPU rasterizers (SwiftShader, llvmpipe) start at the cheapest tier rather than stepping down to it. */
function softwareRenderer(): boolean {
  try {
    const gl = document.createElement('canvas').getContext('webgl2');
    if (!gl) return false;
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    const name = String(gl.getParameter(ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER));
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return /swiftshader|llvmpipe|softpipe|software/i.test(name);
  } catch {
    return false;
  }
}

export function ClusterScene() {
  const d = useDerivedContext();
  const hovered = useApp((s) => s.hoveredGpu);
  const selected = useApp((s) => s.selectedGpu);
  const hoverGpu = useApp((s) => s.hoverGpu);
  const selectGpu = useApp((s) => s.selectGpu);
  const pinned = useMemo(() => pinnedQuality(), []);
  const [quality, setQuality] = useState<Quality>(() => pinned ?? (softwareRenderer() ? 0 : 2));
  const motion = useMemo(() => !prefersReducedMotion(), []);

  const layout = useMemo(() => layoutCluster(d.cluster), [d.cluster]);
  const [focus, setFocus] = useState<{ layout: typeof layout; gpu: number } | null>(null);
  const focusGpu = focus && focus.layout === layout ? focus.gpu : null;
  useEffect(() => {
    // Escape flies back out to the whole cluster.
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setFocus(null);
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  const gpus = useMemo(() => gpuVisuals(d.views, totalGpus(d.cluster), d.cluster.gpu.memBytes), [d.views, d.cluster]);
  const plates = useMemo(() => {
    const p = plateVisuals(d.model, gpus, d.views);
    // Level of detail: very large clusters keep an even subsample of the GPU-layers.
    return p.length > 40000 ? p.filter((_, i) => i % Math.ceil(p.length / 40000) === 0) : p;
  }, [d.model, gpus, d.views]);
  const kvxUtil = d.kvTransfer?.util ?? 0;
  const links = useMemo(() => linkVisuals(layout, d.views, kvxUtil), [layout, d.views, kvxUtil]);
  const replicaOf = useMemo(() => gpuReplicas(d.views, totalGpus(d.cluster)), [d.views, d.cluster]);

  // Live simulation drives memory fill and traffic density; otherwise the analytical steady state does.
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
        const i = phaseAt(run);
        const k = run.phases[i]!.kind;
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
  const levels = useMemo(() => new Float32Array(links.length), [links]);
  const busBoxes = useMemo(() => (layout.spine ? [...layout.switches, layout.spine] : layout.switches), [layout]);
  const busLevel = useMemo(() => {
    // Which links feed each bus: scale-up spokes to their switch, network links to the spine.
    const members: number[][] = busBoxes.map(() => []);
    links.forEach((l, i) => {
      if (l.inter && l.kind !== 'kvx' && layout.spine) members[layout.switches.length]!.push(i);
      else if (!l.inter && (l.kind === 'tp' || l.kind === 'ep') && !layout.meshEdges.length) {
        const sw = layout.gpus[l.gpu]?.switchIdx ?? -1;
        if (sw >= 0) members[sw]!.push(i);
      }
    });
    return (i: number) => {
      const m = members[i];
      if (!m?.length) return 0;
      let s = 0;
      for (const j of m) s += levels[j] ?? 0;
      return s / m.length;
    };
  }, [busBoxes, links, layout, levels]);
  const expertLoad = useMemo(() => () => useSim.getState().frame?.expertLoad ?? null, []);
  // Step trace: the layer being computed glows (its attention or FFN half, by phase), with a
  // fading wake over the layers just finished. Writes (attention, FFN) per layer.
  const layerHeat = useMemo(() => {
    return (out: Float32Array) => {
      const run = useTrace.getState().run;
      const i = run ? phaseAt(run) : -1;
      if (!run || i < 0) {
        out.fill(0);
        return -1;
      }
      const ph = run.phases[i]!;
      // The layer the wavefront is on (LM head / hand-off phases keep the last layer lit).
      let layer = -1;
      for (let j = i; j >= 0 && layer < 0; j--) layer = run.phases[j]!.layer;
      const k = ph.kind;
      const [ha, hf] =
        k === 'attn' ? [1, 0.12] : k === 'ffn' ? [0.12, 1] : k === 'dispatch' || k === 'combine' ? [0.15, 0.55] : k === 'tp' ? [0.45, 0.45] : [0.35, 0.35];
      for (let l = 0; 2 * l + 1 < out.length; l++) {
        let a = 0;
        let f = 0;
        if (l === layer) [a, f] = [ha, hf];
        else if (l < layer) a = f = 0.22 * Math.exp(-(layer - l) / 2);
        out[2 * l] = a;
        out[2 * l + 1] = f;
      }
      return run.inst;
    };
  }, []);

  const outlines = useMemo(() => {
    const rects: GroupRect[] = [];
    d.views.forEach((v) => {
      const { inst } = v.ev;
      const byDp = new Map<number, number[]>();
      for (const s of v.ev.placement.shards) {
        const arr = byDp.get(s.dpRank) ?? [];
        arr.push(s.gpu);
        byDp.set(s.dpRank, arr);
      }
      const roleName = inst.role === 'prefill' ? 'Prefill' : inst.role === 'decode' ? 'Decode' : 'Replica';
      const style = inst.role === 'prefill' ? 2 : inst.role === 'decode' ? 3 : 1;
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
        rects.push({ min, max, label, style });
      });
    });
    return rects;
  }, [d.views, layout]);
  // Rack-scale domains (several trays) get a floor marking; a lone node's tray already says it all.
  const domains = useMemo(
    () =>
      (layout.boards.length > layout.platforms.length ? layout.platforms : []).map((p) => ({
        min: { x: p.center.x - p.size.x / 2, y: 0, z: p.center.z - p.size.z / 2 },
        max: { x: p.center.x + p.size.x / 2, y: 0, z: p.center.z + p.size.z / 2 },
        style: 0,
        color: '#7d9cbd',
      })),
    [layout],
  );

  const center = useMemo(() => {
    const b = layout.bounds;
    return [(b.min.x + b.max.x) / 2, TOWER_H * 0.35, (b.min.z + b.max.z) / 2] as [number, number, number];
  }, [layout]);
  const radius = useMemo(() => {
    const b = layout.bounds;
    return 0.5 * Math.hypot(b.max.x - b.min.x, b.max.y - b.min.y, b.max.z - b.min.z);
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
      flat
      dpr={quality === 2 ? [1, 2] : 1}
      camera={{ position: [8, 10, 12], fov: 38, near: 0.1, far: 2000 }}
      gl={{ antialias: true }}
      onPointerMissed={(e) => (e.type === 'dblclick' ? setFocus(null) : selectGpu(null))}
    >
      <PerformanceMonitor onDecline={() => pinned === null && setQuality((q) => (q > 0 ? ((q - 1) as Quality) : q))} flipflops={3} />
      <color attach="background" args={[VOID]} />
      <Sky horizon={VOID} zenith={ZENITH} />
      <fog attach="fog" args={[VOID, radius * 2.4, radius * 7 + 20]} />
      <ambientLight intensity={0.35} />
      <directionalLight position={[8, 14, 10]} intensity={0.85} />
      <directionalLight position={[-10, 6, -8]} intensity={0.5} color={'#9fc3ff'} />
      <Environment resolution={64} frames={1}>
        <Lightformer form="rect" intensity={1.2} color="#cfe3ff" position={[0, 6, 0]} rotation-x={Math.PI / 2} scale={[12, 3, 1]} />
        <Lightformer form="rect" intensity={0.7} color="#7fa6d6" position={[-6, 2, 4]} rotation-y={Math.PI / 3} scale={[6, 1.2, 1]} />
        <Lightformer form="rect" intensity={0.45} color="#ffd9b0" position={[6, 1.5, -4]} rotation-y={-Math.PI / 2.5} scale={[5, 1, 1]} />
      </Environment>

      <Floor bounds={layout.bounds} reflect={quality === 2} />
      <Decals rects={domains} y={FLOOR_Y + 0.004} />
      <Trays layout={layout} />
      <Packages layout={layout} gpus={gpus} />
      <HbmColumns layout={layout} gpus={gpus} liveKv={liveKv} motion={motion} />
      <LayerStacks layout={layout} plates={plates} model={d.model} heat={layerHeat} experts={expertLoad} />
      <Buses boxes={busBoxes} level={busLevel} />
      <Wiring layout={layout} />
      <LinkLines links={links} level={level} levels={levels} />
      <GroupOutlines rects={outlines} scale={Math.min(2.2, Math.max(1, radius / 4.5))} />
      <FloorLabels layout={layout} />
      <SelectionMarkers layout={layout} hovered={hovered} selected={selected} motion={motion} />
      <HitTargets
        layout={layout}
        selected={selected}
        onHover={hoverGpu}
        onSelect={selectGpu}
        onFocus={(g) => {
          selectGpu(g);
          setFocus({ layout, gpu: g });
        }}
      />
      <GpuTooltip layout={layout} g={hoveredVisual} label={hoveredLabel} />
      <OrbitControls makeDefault enableDamping dampingFactor={0.12} maxPolarAngle={Math.PI * 0.47} />
      <CameraRig bounds={layout.bounds} center={center} focus={focusGpu !== null ? layout.gpus[focusGpu]?.pos ?? null : null} motion={motion} />
      {/* Always composite through a linear half-float target: additive light (hologram, glass,
          link lines) must sum in linear space, or it blows out. The lowest tier drops bloom and MSAA. */}
      <EffectComposer key={quality > 0 ? 'fx' : 'lite'} multisampling={quality > 0 ? 4 : 0}>
        {quality > 0 ? <Bloom luminanceThreshold={0.9} luminanceSmoothing={0.25} intensity={0.3} radius={0.5} mipmapBlur /> : <></>}
        <Vignette offset={0.3} darkness={0.55} />
      </EffectComposer>
    </Canvas>
  );
}

const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/**
 * Frames the cluster whenever its layout changes (or flies to a double-clicked GPU), easing from the
 * current view. The first framing dollies in from further out; reduced motion snaps instead.
 */
function CameraRig({
  bounds,
  center,
  focus,
  motion,
}: {
  bounds: { min: Vec3; max: Vec3 };
  center: [number, number, number];
  focus: Vec3 | null;
  motion: boolean;
}) {
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera;
  const size = useThree((s) => s.size);
  const controls = useThree((s) => s.controls) as unknown as (THREE.EventDispatcher<{ start: object }> & { target: THREE.Vector3; update: () => void }) | null;
  const anim = useRef<{ from: THREE.Vector3; to: THREE.Vector3; tFrom: THREE.Vector3; tTo: THREE.Vector3; t: number; dur: number } | null>(null);
  const framed = useRef(false);
  useEffect(() => {
    if (!controls) return;
    let pos: THREE.Vector3;
    let target: THREE.Vector3;
    if (focus) {
      // Keep the current viewing direction, close in on the GPU.
      target = new THREE.Vector3(focus.x, TOWER_H * 0.5, focus.z);
      const dir = camera.position.clone().sub(controls.target).normalize();
      pos = target.clone().addScaledVector(dir, 5.2);
    } else {
      const r = 0.5 * Math.hypot(bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y, bounds.max.z - bounds.min.z);
      const vFov = (camera.fov * Math.PI) / 180;
      const hFov = 2 * Math.atan(Math.tan(vFov / 2) * (size.width / Math.max(1, size.height)));
      const dist = Math.max(4.6, (r / Math.sin(Math.min(vFov, hFov) / 2)) * 0.92);
      const dir = new THREE.Vector3(0.32, 0.54, 0.78).normalize();
      target = new THREE.Vector3(...center);
      pos = target.clone().addScaledVector(dir, dist);
    }
    if (!framed.current) {
      // First framing: start further out and higher, then dolly in.
      framed.current = true;
      const start = pos.clone().sub(target).multiplyScalar(1.45).add(target);
      start.y += 1.5;
      camera.position.copy(start);
      controls.target.copy(target);
      controls.update();
    }
    if (!motion) {
      camera.position.copy(pos);
      controls.target.copy(target);
      controls.update();
      anim.current = null;
      return;
    }
    anim.current = { from: camera.position.clone(), to: pos, tFrom: controls.target.clone(), tTo: target, t: 0, dur: focus ? 0.9 : 1.3 };
    // Refit only when the cluster geometry or focus changes, not on every resize.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bounds, center, focus, controls]);
  useEffect(() => {
    // Floor-level overlays (labels, outlines, tile seams) live on their own layer so the floor's
    // reflection pass skips them; the main camera sees both.
    camera.layers.enable(OVERLAY_LAYER);
  }, [camera]);
  useEffect(() => {
    if (!controls) return;
    // Any orbit gesture hands the camera back to the user.
    const stop = () => (anim.current = null);
    controls.addEventListener('start', stop);
    return () => controls.removeEventListener('start', stop);
  }, [controls]);
  useFrame((_, dt) => {
    const a = anim.current;
    if (!a || !controls) return;
    a.t = Math.min(1, a.t + dt / a.dur);
    const e = ease(a.t);
    camera.position.lerpVectors(a.from, a.to, e);
    controls.target.lerpVectors(a.tFrom, a.tTo, e);
    controls.update();
    if (a.t >= 1) anim.current = null;
  });
  return null;
}
