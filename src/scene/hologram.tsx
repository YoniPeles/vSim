// The model, projected above each die as a hologram: a shell spanning the whole model whose sides
// carry one line per layer, and a flat wafer for each layer the GPU holds.

import { useLayoutEffect, useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { C } from '../ui/theme.ts';
import type { ModelSpec } from '../core/types.ts';
import type { Layout } from './layout.ts';
import { layerStep, layerY, STACK, type PlateVisual } from './sceneModel.ts';
import { stackShellMaterial, waferMaterial } from './shaders.ts';
import { attr, setBox, useDispose } from './instancing.ts';

const SW = STACK.x1 - STACK.x0;
const SD = STACK.z1 - STACK.z0;
const SX = (STACK.x0 + STACK.x1) / 2;
const SZ = (STACK.z0 + STACK.z1) / 2;
const col = new THREE.Color();

function attnKind(model: ModelSpec, i: number): 'full' | 'local' | 'linear' {
  const a = model.layers[i]!.attn;
  return a.kind === 'linear' ? 'linear' : a.kind === 'gqa' && a.scope !== 'full' ? 'local' : 'full';
}

/** Overall brightness of the hologram light (additive wafers stack up, so keep this modest). */
const HOLO_GAIN = 0.4;

/** Per-layer light: row 0 attention, row 1 FFN. */
function layerTexture(model: ModelSpec): THREE.DataTexture {
  const L = model.layers.length;
  const data = new Float32Array(L * 2 * 4);
  const attnColor = { full: C.attnFull, local: C.attnLocal, linear: C.attnLinear } as const;
  for (let i = 0; i < L; i++) {
    col.set(attnColor[attnKind(model, i)]).multiplyScalar(HOLO_GAIN);
    data.set([col.r, col.g, col.b, 1], i * 4);
    col.set(model.layers[i]!.ffn.kind === 'moe' ? C.moeA : C.ffnDense).multiplyScalar(HOLO_GAIN);
    data.set([col.r, col.g, col.b, 1], (L + i) * 4);
  }
  const t = new THREE.DataTexture(data, L, 2, THREE.RGBAFormat, THREE.FloatType);
  t.minFilter = t.magFilter = THREE.NearestFilter;
  t.needsUpdate = true;
  return t;
}

export function LayerStacks({
  layout,
  plates,
  model,
  heat,
  experts,
}: {
  layout: Layout;
  plates: PlateVisual[];
  model: ModelSpec;
  /** Writes the step trace's (attention, FFN) glow per layer; returns the traced instance, or −1. */
  heat?: (out: Float32Array) => number;
  /** Live per-expert load (0..1) indexed by global expert id, or null. */
  experts?: () => Float32Array | null;
}) {
  const L = model.layers.length;
  const shellRef = useRef<THREE.InstancedMesh>(null);
  const waferRef = useRef<THREE.InstancedMesh>(null);
  const shellMat = useMemo(() => stackShellMaterial(), []);
  const waferMat = useMemo(() => waferMaterial(), []);
  const layerTex = useMemo(() => layerTexture(model), [model]);
  const heatTex = useMemo(() => {
    const t = new THREE.DataTexture(new Float32Array(L * 2), L, 1, THREE.RGFormat, THREE.FloatType);
    t.minFilter = t.magFilter = THREE.NearestFilter;
    t.needsUpdate = true;
    return t;
  }, [L]);
  useDispose(layerTex);
  useDispose(heatTex);
  useDispose(shellMat);
  useDispose(waferMat);
  const expertTex = useRef<THREE.DataTexture | null>(null);
  const lastLoad = useRef<Float32Array | null>(null);
  const tracing = useRef(false);

  // One shell per GPU holding layers: its held range and tensor-parallel share.
  const stacks = useMemo(() => {
    const by = new Map<number, { gpu: number; inst: number; lo: number; hi: number; p: PlateVisual }>();
    for (const p of plates) {
      const s = by.get(p.gpu);
      if (!s) by.set(p.gpu, { gpu: p.gpu, inst: p.inst, lo: p.layer, hi: p.layer + 1, p });
      else {
        s.lo = Math.min(s.lo, p.layer);
        s.hi = Math.max(s.hi, p.layer + 1);
        // Prefer an MoE layer's view of the FFN share (whole experts span the wafer).
        if (p.ffnWhole) s.p = p;
      }
    }
    return [...by.values()];
  }, [plates]);
  const ns = stacks.length;
  const nw = plates.length;
  const shellGeom = useMemo(() => {
    const g = new THREE.BoxGeometry(1, 1, 1);
    const n = Math.max(1, ns);
    g.setAttribute('aRange', new THREE.InstancedBufferAttribute(new Float32Array(n * 2), 2));
    g.setAttribute('aSlice', new THREE.InstancedBufferAttribute(new Float32Array(n * 4), 4));
    g.setAttribute('aInst', new THREE.InstancedBufferAttribute(new Float32Array(n), 1));
    return g;
  }, [ns]);
  useDispose(shellGeom);
  const waferGeom = useMemo(() => {
    const g = new THREE.PlaneGeometry(1, 1);
    g.rotateX(-Math.PI / 2);
    const n = Math.max(1, nw);
    g.setAttribute('aSlice', new THREE.InstancedBufferAttribute(new Float32Array(n * 4), 4));
    g.setAttribute('aExp', new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3));
    g.setAttribute('aLayer', new THREE.InstancedBufferAttribute(new Float32Array(n * 2), 2));
    return g;
  }, [nw]);
  useDispose(waferGeom);

  useLayoutEffect(() => {
    const step = layerStep(L);
    for (const m of [shellMat, waferMat]) {
      const u = m.uniforms;
      u['uLayerTex']!.value = layerTex;
      u['uHeatTex']!.value = heatTex;
      u['uLayers']!.value = L;
    }
    shellMat.uniforms['uStep']!.value = step;
    shellMat.uniforms['uY0']!.value = layerY(0, L);
    shellMat.uniforms['uWaferH']!.value = plates[0]?.h ?? step * 0.3;
    // Each wafer's face glow scales with the layer pitch, so a stack's total glow does not depend on depth.
    waferMat.uniforms['uFace']!.value = Math.min(0.025, step * 0.3);

    const ms = shellRef.current;
    if (ms) {
      const range = attr(shellGeom, 'aRange');
      const slice = attr(shellGeom, 'aSlice');
      const inst = attr(shellGeom, 'aInst');
      const y0 = layerY(0, L);
      const y1 = layerY(L, L);
      stacks.forEach((s, i) => {
        const g = layout.gpus[s.gpu]!;
        setBox(ms, i, { x: g.pos.x + SX, y: (y0 + y1) / 2, z: g.pos.z + SZ }, { x: SW, y: y1 - y0, z: SD });
        range.setXY(i, s.lo, s.hi);
        slice.setXYZW(i, s.p.tpLo, s.p.tpHi, s.p.ffnWhole ? 0 : s.p.tpLo, s.p.ffnWhole ? 1 : s.p.tpHi);
        inst.setX(i, s.inst);
      });
      ms.instanceMatrix.needsUpdate = true;
      range.needsUpdate = slice.needsUpdate = inst.needsUpdate = true;
      ms.computeBoundingSphere();
    }
    const mw = waferRef.current;
    if (mw) {
      const slice = attr(waferGeom, 'aSlice');
      const ex = attr(waferGeom, 'aExp');
      const ly = attr(waferGeom, 'aLayer');
      const top = new Map(stacks.map((s) => [s.gpu, s.hi - 1]));
      plates.forEach((p, i) => {
        const g = layout.gpus[p.gpu]!;
        setBox(mw, i, { x: g.pos.x + SX, y: p.y + p.h, z: g.pos.z + SZ }, { x: SW, y: 1, z: SD });
        slice.setXYZW(i, p.tpLo, p.tpHi, p.ffnWhole ? 0 : p.tpLo, p.ffnWhole ? 1 : p.tpHi);
        ex.setXYZ(i, Math.min(256, p.cells), p.expertLo, top.get(p.gpu) === p.layer ? 1 : 0);
        ly.setXY(i, p.layer, p.inst);
      });
      mw.instanceMatrix.needsUpdate = true;
      slice.needsUpdate = ex.needsUpdate = ly.needsUpdate = true;
      mw.computeBoundingSphere();
    }
  }, [plates, stacks, layout, L, layerTex, heatTex, shellGeom, waferGeom, shellMat, waferMat]);

  useFrame(() => {
    if (heat) {
      const data = heatTex.image.data as Float32Array;
      const inst = heat(data);
      if (inst >= 0 || tracing.current) heatTex.needsUpdate = true;
      tracing.current = inst >= 0;
      shellMat.uniforms['uTraceInst']!.value = inst;
      waferMat.uniforms['uTraceInst']!.value = inst;
    }
    const u = waferMat.uniforms;
    const load = experts?.() ?? null;
    if (!load) {
      u['uExpertOn']!.value = 0;
      return;
    }
    // The simulator sends a fresh array per frame; upload only when it changes.
    if (load === lastLoad.current) return;
    lastLoad.current = load;
    let t = expertTex.current;
    if (!t || t.image.width !== load.length) {
      t?.dispose();
      t = new THREE.DataTexture(new Float32Array(load.length), load.length, 1, THREE.RedFormat, THREE.FloatType);
      t.minFilter = t.magFilter = THREE.NearestFilter;
      expertTex.current = t;
    }
    (t.image.data as Float32Array).set(load);
    t.needsUpdate = true;
    u['uExperts']!.value = t;
    u['uExpertCount']!.value = load.length;
    u['uExpertOn']!.value = 1;
  });
  return (
    <group>
      {nw > 0 && <instancedMesh key={`w${nw}`} ref={waferRef} args={[waferGeom, waferMat, nw]} raycast={() => null} frustumCulled={false} renderOrder={3} />}
      {ns > 0 && <instancedMesh key={`s${ns}`} ref={shellRef} args={[shellGeom, shellMat, ns]} raycast={() => null} frustumCulled={false} renderOrder={3} />}
    </group>
  );
}
