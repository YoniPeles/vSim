// Traffic: static fibre paths, and per-link lines whose colour comes up with utilization.

import { useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { C } from '../ui/theme.ts';
import { TOWER_H, type Layout, type Vec3 } from './layout.ts';
import { meshEnds, STACK, type LinkVisual } from './sceneModel.ts';
import { linkLineMaterial } from './shaders.ts';
import { useDispose } from './instancing.ts';

export const TRAFFIC_COLOR: Record<LinkVisual['kind'], string> = { tp: C.tp, ep: C.ep, pp: C.pp, kvx: C.kvx };

const SEGS = 16;

function bezierPoints(l: { from: Vec3; ctrl: Vec3; to: Vec3 }, segs: number): number[] {
  const out: number[] = [];
  let prev: number[] | null = null;
  for (let i = 0; i <= segs; i++) {
    const t = i / segs;
    const u = 1 - t;
    const p = [
      u * u * l.from.x + 2 * u * t * l.ctrl.x + t * t * l.to.x,
      u * u * l.from.y + 2 * u * t * l.ctrl.y + t * t * l.to.y,
      u * u * l.from.z + 2 * u * t * l.ctrl.z + t * t * l.to.z,
    ];
    if (prev) out.push(...prev, ...p);
    prev = p;
  }
  return out;
}

/** Faint fibre: NVLink spokes, mesh edges and NIC rails (traffic paths are drawn by LinkLines). */
export function Wiring({ layout }: { layout: Layout }) {
  const geom = useMemo(() => {
    const pts: number[] = [];
    const cols: number[] = [];
    const push = (seg: number[], c: THREE.Color) => {
      pts.push(...seg);
      for (let i = 0; i < seg.length / 3; i++) cols.push(c.r, c.g, c.b);
    };
    const fibre = new THREE.Color('#4f79a3').multiplyScalar(0.3);
    const sx = (STACK.x0 + STACK.x1) / 2;
    if (!layout.meshEdges.length) {
      for (const g of layout.gpus) {
        if (g.switchIdx < 0) continue;
        push([g.pos.x + sx, TOWER_H + 0.05, g.pos.z, g.switchPort.x, g.switchPort.y, g.switchPort.z], fibre);
      }
    }
    for (const [a, b] of layout.meshEdges) {
      const [pa, pb] = meshEnds(layout.gpus[a]!.pos, layout.gpus[b]!.pos);
      push(bezierPoints({ from: pa, ctrl: { x: (pa.x + pb.x) / 2, y: 0.5, z: (pa.z + pb.z) / 2 }, to: pb }, 8), fibre);
    }
    if (layout.spine) {
      const y = layout.spine.center.y;
      const z = layout.spine.center.z;
      for (const l of layout.leaves) push([l.x, l.y, l.z, l.x, y, z], fibre);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
    return g;
  }, [layout]);
  useDispose(geom);
  return (
    <lineSegments geometry={geom} raycast={() => null} frustumCulled={false}>
      <lineBasicMaterial vertexColors transparent depthWrite={false} blending={THREE.AdditiveBlending} />
    </lineSegments>
  );
}

/**
 * One line per link in its traffic colour, brightening with the link's level (utilization, or the
 * step trace's current phase). `levels` receives the per-link level each frame for other readers
 * (the switch filaments).
 */
export function LinkLines({
  links,
  level,
  levels,
}: {
  links: LinkVisual[];
  level: (l: LinkVisual, i: number) => number;
  levels: Float32Array;
}) {
  const material = useMemo(() => linkLineMaterial(), []);
  const tex = useMemo(() => {
    const n = Math.max(1, links.length);
    const t = new THREE.DataTexture(new Float32Array(n), n, 1, THREE.RedFormat, THREE.FloatType);
    t.minFilter = t.magFilter = THREE.NearestFilter;
    t.needsUpdate = true;
    return t;
  }, [links]);
  const geom = useMemo(() => {
    const pos: number[] = [];
    const col: number[] = [];
    const idx: number[] = [];
    links.forEach((l, li) => {
      const c = new THREE.Color(TRAFFIC_COLOR[l.kind]);
      const seg = bezierPoints(l, SEGS);
      pos.push(...seg);
      for (let k = 0; k < seg.length / 3; k++) {
        col.push(c.r, c.g, c.b);
        idx.push(li);
      }
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('aColor', new THREE.Float32BufferAttribute(col, 3));
    g.setAttribute('aLink', new THREE.Float32BufferAttribute(idx, 1));
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e4);
    return g;
  }, [links]);
  useDispose(geom);
  useDispose(tex);
  useDispose(material);
  useFrame(() => {
    const u = material.uniforms;
    u['uLevel']!.value = tex;
    u['uLinks']!.value = Math.max(1, links.length);
    const data = tex.image.data as Float32Array;
    let changed = false;
    for (let i = 0; i < links.length; i++) {
      const k = level(links[i]!, i);
      if (i < levels.length) levels[i] = k;
      if (Math.abs(data[i]! - k) > 0.01) {
        data[i] = k;
        changed = true;
      }
    }
    if (changed) tex.needsUpdate = true;
  });
  if (!links.length) return null;
  return <lineSegments geometry={geom} material={material} raycast={() => null} frustumCulled={false} renderOrder={4} />;
}
