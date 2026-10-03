// HBM memory columns: a glass stack per GPU filled with luminous liquid, one band per memory category.
// While the simulation runs, the KV-in-use band rises and falls with the live block pool.

import { useLayoutEffect, useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { C } from '../ui/theme.ts';
import { TOWER_H, type Layout } from './layout.ts';
import { TOWER, type GpuVisual } from './sceneModel.ts';
import { hbmFillMaterial, hbmGlassMaterial } from './shaders.ts';
import { attr, instancedBox, setBox, useDispose } from './instancing.ts';

const BASE = 0.12;
const TW = TOWER.x1 - TOWER.x0;
const TX = (TOWER.x0 + TOWER.x1) / 2;
/** HBM stacks per package, as z-ranges inside the tower footprint. Memory is interleaved across stacks, so each fills alike. */
const GAP = 0.08;
const STACKS: [number, number][] = [
  [TOWER.z0, (TOWER.z0 + TOWER.z1) / 2 - GAP / 2],
  [(TOWER.z0 + TOWER.z1) / 2 + GAP / 2, TOWER.z1],
];
const K = STACKS.length;

/** Cumulative band tops (fractions of total memory) for weights, +activations, +overhead. */
function fixedTops(g: GpuVisual): [number, number, number] {
  const w = g.mem.weights;
  const a = w + g.mem.activations;
  return [w, a, a + g.mem.overhead];
}

export function HbmColumns({
  layout,
  gpus,
  liveKv,
  motion,
}: {
  layout: Layout;
  gpus: GpuVisual[];
  liveKv?: (gpu: number) => number | null;
  /** Animate the KV block shimmer (off for reduced motion). */
  motion: boolean;
}) {
  const fillRef = useRef<THREE.InstancedMesh>(null);
  const glassRef = useRef<THREE.InstancedMesh>(null);
  const fillMat = useMemo(() => {
    const m = hbmFillMaterial();
    const u = m.uniforms;
    (u['uColW']!.value as THREE.Color).set(C.weights);
    (u['uColA']!.value as THREE.Color).set(C.activations);
    (u['uColO']!.value as THREE.Color).set(C.overhead);
    (u['uColK']!.value as THREE.Color).set(C.kvUsed);
    u['uBase']!.value = BASE;
    u['uH']!.value = TOWER_H;
    return m;
  }, []);
  const glassMat = useMemo(() => {
    const m = hbmGlassMaterial();
    (m.uniforms['uColK']!.value as THREE.Color).set(C.kvFree);
    m.uniforms['uBase']!.value = BASE;
    m.uniforms['uH']!.value = TOWER_H;
    return m;
  }, []);
  const n = layout.gpus.length;
  const fillGeom = useMemo(() => instancedBox(n * K, { aBands: 4 }), [n]);
  const glassGeom = useMemo(() => instancedBox(n * K, { aKv: 2 }), [n]);
  useDispose(fillGeom);
  useDispose(glassGeom);
  useDispose(fillMat);
  useDispose(glassMat);
  // Current KV-in-use fraction per GPU (eased towards the live value).
  const kvNow = useRef<Float32Array>(new Float32Array(0));

  const placeFill = (m: THREE.InstancedMesh, i: number, top: number) => {
    const s = layout.gpus[i]!;
    const h = top * TOWER_H;
    // An empty column has no liquid at all (not even a sliver of a top face).
    const w = h > 1e-4 ? 1 : 0;
    STACKS.forEach(([z0, z1], k) =>
      setBox(m, i * K + k, { x: s.pos.x + TX, y: BASE + h / 2, z: s.pos.z + (z0 + z1) / 2 }, { x: (TW - 0.07) * w, y: h, z: (z1 - z0 - 0.07) * w }),
    );
  };

  useLayoutEffect(() => {
    const mf = fillRef.current;
    const mg = glassRef.current;
    if (!mf || !mg) return;
    const bands = attr(fillGeom, 'aBands');
    const kv = attr(glassGeom, 'aKv');
    kvNow.current = new Float32Array(n);
    layout.gpus.forEach((s, i) => {
      const g = gpus[i]!;
      const [w, a, o] = fixedTops(g);
      const used = o + g.mem.kvUsed;
      kvNow.current[i] = g.mem.kvUsed;
      placeFill(mf, i, g.active ? used : 0);
      STACKS.forEach(([z0, z1], k) => {
        bands.setXYZW(i * K + k, w, a, o, used);
        // Idle GPUs reserve nothing: no free-KV hatch, no utilization line.
        if (g.active) kv.setXY(i * K + k, used, used + g.mem.kvFree);
        else kv.setXY(i * K + k, -1, -1);
        setBox(mg, i * K + k, { x: s.pos.x + TX, y: BASE + (TOWER_H + 0.012) / 2, z: s.pos.z + (z0 + z1) / 2 }, { x: TW + 0.01, y: TOWER_H + 0.012, z: z1 - z0 + 0.01 });
      });
    });
    for (const m of [mf, mg]) {
      m.instanceMatrix.needsUpdate = true;
      m.computeBoundingSphere();
    }
    bands.needsUpdate = kv.needsUpdate = true;
    // placeFill closes over the layout passed in this render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout, gpus, fillGeom, glassGeom, n]);

  useFrame((_, dt) => {
    const mf = fillRef.current;
    if (!mf) return;
    let live = false;
    let changed = false;
    const bands = attr(fillGeom, 'aBands');
    const kv = attr(glassGeom, 'aKv');
    const ease = Math.min(1, dt * 6);
    for (let i = 0; i < n; i++) {
      const g = gpus[i];
      if (!g?.active) continue;
      const f = liveKv?.(i) ?? null;
      if (f !== null) live = true;
      const total = g.mem.kvUsed + g.mem.kvFree;
      const target = f === null ? g.mem.kvUsed : total * Math.min(1, Math.max(0, f));
      const cur = kvNow.current[i] ?? target;
      if (Math.abs(target - cur) < 1e-5) continue;
      const next = Math.abs(target - cur) < 1e-4 ? target : cur + (target - cur) * ease;
      kvNow.current[i] = next;
      const o = fixedTops(g)[2];
      for (let j = 0; j < K; j++) {
        bands.setW(i * K + j, o + next);
        kv.setX(i * K + j, o + next);
      }
      placeFill(mf, i, o + next);
      changed = true;
    }
    if (changed) {
      mf.instanceMatrix.needsUpdate = true;
      bands.needsUpdate = kv.needsUpdate = true;
    }
    const u = fillMat.uniforms;
    u['uTime']!.value += dt;
    u['uShimmer']!.value = live && motion ? 1 : 0;
  });

  return (
    <group>
      <instancedMesh key={`f${n}`} ref={fillRef} args={[fillGeom, fillMat, n * K]} raycast={() => null} frustumCulled={false} />
      <instancedMesh key={`g${n}`} ref={glassRef} args={[glassGeom, glassMat, n * K]} raycast={() => null} frustumCulled={false} renderOrder={2} />
    </group>
  );
}
