// Traffic: static fibre paths, and comets of light travelling them at a density set by utilization.

import { useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { C } from '../ui/theme.ts';
import { TOWER_H, type Layout, type Vec3 } from './layout.ts';
import { meshEnds, STACK, type LinkVisual } from './sceneModel.ts';
import { cometMaterial } from './shaders.ts';
import { useDispose } from './instancing.ts';

export const TRAFFIC_COLOR: Record<LinkVisual['kind'], string> = { tp: C.tp, ep: C.ep, pp: C.pp, kvx: C.kvx };

const PER_LINK = 24;
const vp = new THREE.Vector2();

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

/** Faint fibre: NVLink spokes, mesh edges, NIC rails, and the pipeline / KV-transfer arcs. */
export function Wiring({ layout, links }: { layout: Layout; links: LinkVisual[] }) {
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
    for (const l of links) {
      if (l.kind !== 'pp' && l.kind !== 'kvx') continue;
      push(bezierPoints(l, 20), new THREE.Color(TRAFFIC_COLOR[l.kind]).multiplyScalar(0.22));
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
    return g;
  }, [layout, links]);
  useDispose(geom);
  return (
    <lineSegments geometry={geom} raycast={() => null} frustumCulled={false}>
      <lineBasicMaterial vertexColors transparent depthWrite={false} blending={THREE.AdditiveBlending} />
    </lineSegments>
  );
}

/**
 * Comets of light along each link. Each link owns PER_LINK comets; a per-link intensity texture
 * decides how many are visible. `levels` receives the per-link level each frame for other readers.
 */
export function Comets({
  links,
  level,
  levels,
  speed = 1,
}: {
  links: LinkVisual[];
  level: (l: LinkVisual, i: number) => number;
  levels: Float32Array;
  speed?: number;
}) {
  const material = useMemo(() => cometMaterial(), []);
  const tex = useMemo(() => {
    const n = Math.max(1, links.length);
    const t = new THREE.DataTexture(new Float32Array(n), n, 1, THREE.RedFormat, THREE.FloatType);
    t.minFilter = t.magFilter = THREE.NearestFilter;
    t.needsUpdate = true;
    return t;
  }, [links]);
  const geom = useMemo(() => {
    const n = links.length * PER_LINK;
    const from = new Float32Array(n * 3);
    const ctrl = new Float32Array(n * 3);
    const to = new Float32Array(n * 3);
    const phase = new Float32Array(n);
    const spd = new Float32Array(n);
    const trail = new Float32Array(n);
    const color = new Float32Array(n * 3);
    const link = new Float32Array(n);
    const rank = new Float32Array(n);
    let seed = 1;
    const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    links.forEach((l, li) => {
      const c = new THREE.Color(TRAFFIC_COLOR[l.kind]);
      // Curve length (chord + control polygon average) sets speed and trail in curve parameter units.
      const chord = Math.hypot(l.to.x - l.from.x, l.to.y - l.from.y, l.to.z - l.from.z);
      const poly =
        Math.hypot(l.ctrl.x - l.from.x, l.ctrl.y - l.from.y, l.ctrl.z - l.from.z) + Math.hypot(l.to.x - l.ctrl.x, l.to.y - l.ctrl.y, l.to.z - l.ctrl.z);
      const len = (chord + poly) / 2 + 0.05;
      for (let i = 0; i < PER_LINK; i++) {
        const j = li * PER_LINK + i;
        from.set([l.from.x, l.from.y, l.from.z], j * 3);
        ctrl.set([l.ctrl.x, l.ctrl.y, l.ctrl.z], j * 3);
        to.set([l.to.x, l.to.y, l.to.z], j * 3);
        // Interleave ranks so any visible subset is spread evenly along the link.
        rank[j] = ((i * 7) % PER_LINK) / PER_LINK;
        phase[j] = i / PER_LINK + rnd() * 0.1;
        spd[j] = 0.7 / len;
        trail[j] = Math.min(0.4, 0.32 / len);
        color.set([c.r, c.g, c.b], j * 3);
        link[j] = li;
      }
    });
    const g = new THREE.InstancedBufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute([0, -1, 0, 1, -1, 0, 1, 1, 0, 0, 1, 0], 3));
    g.setIndex([0, 1, 2, 0, 2, 3]);
    g.setAttribute('aFrom', new THREE.InstancedBufferAttribute(from, 3));
    g.setAttribute('aCtrl', new THREE.InstancedBufferAttribute(ctrl, 3));
    g.setAttribute('aTo', new THREE.InstancedBufferAttribute(to, 3));
    g.setAttribute('aPhase', new THREE.InstancedBufferAttribute(phase, 1));
    g.setAttribute('aSpeed', new THREE.InstancedBufferAttribute(spd, 1));
    g.setAttribute('aTrail', new THREE.InstancedBufferAttribute(trail, 1));
    g.setAttribute('aColor', new THREE.InstancedBufferAttribute(color, 3));
    g.setAttribute('aLink', new THREE.InstancedBufferAttribute(link, 1));
    g.setAttribute('aRank', new THREE.InstancedBufferAttribute(rank, 1));
    g.instanceCount = n;
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e4);
    return g;
  }, [links]);
  useDispose(geom);
  useDispose(tex);
  useDispose(material);
  useFrame((state, dt) => {
    const u = material.uniforms;
    u['uTime']!.value += dt * speed;
    state.gl.getDrawingBufferSize(vp);
    (u['uViewport']!.value as THREE.Vector2).copy(vp);
    u['uIntensity']!.value = tex;
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
  return <mesh geometry={geom} material={material} raycast={() => null} frustumCulled={false} renderOrder={4} />;
}
