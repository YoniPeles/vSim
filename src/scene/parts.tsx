import { useLayoutEffect, useMemo, useRef } from 'react';
import { useFrame, type ThreeEvent } from '@react-three/fiber';
import { Html, Text } from '@react-three/drei';
import * as THREE from 'three';
import fontUrl from '@fontsource/ibm-plex-sans-condensed/files/ibm-plex-sans-condensed-latin-500-normal.woff?url';
import { C } from '../ui/theme.ts';
import { TOWER_H, type Box, type Layout, type Vec3 } from './layout.ts';
import { STACK, TOWER, type GpuVisual, type LinkVisual, type PlateVisual } from './sceneModel.ts';
import { particleMaterial, plateMaterial } from './shaders.ts';
import { fmtBytes } from '../core/units.ts';

const tmp = new THREE.Object3D();
const col = new THREE.Color();

function setBox(mesh: THREE.InstancedMesh, i: number, center: Vec3, size: Vec3) {
  tmp.position.set(center.x, center.y, center.z);
  tmp.scale.set(Math.max(1e-4, size.x), Math.max(1e-4, size.y), Math.max(1e-4, size.z));
  tmp.rotation.set(0, 0, 0);
  tmp.updateMatrix();
  mesh.setMatrixAt(i, tmp.matrix);
}

const unitBox = new THREE.BoxGeometry(1, 1, 1);

/** Static boxes (platforms, boards, switches) in one draw call. */
export function Boxes({ boxes, color, opacity = 1, emissive }: { boxes: Box[]; color: string; opacity?: number; emissive?: string }) {
  const ref = useRef<THREE.InstancedMesh>(null);
  useLayoutEffect(() => {
    const m = ref.current;
    if (!m) return;
    boxes.forEach((b, i) => setBox(m, i, b.center, b.size));
    m.instanceMatrix.needsUpdate = true;
    m.computeBoundingSphere();
  }, [boxes]);
  if (!boxes.length) return null;
  return (
    <instancedMesh key={boxes.length} ref={ref} args={[unitBox, undefined, boxes.length]} frustumCulled={false}>
      <meshStandardMaterial
        color={color}
        roughness={0.85}
        metalness={0.15}
        transparent={opacity < 1}
        opacity={opacity}
        emissive={emissive ?? '#000000'}
        emissiveIntensity={emissive ? 0.6 : 0}
      />
    </instancedMesh>
  );
}

export function GpuTiles({
  layout,
  gpus,
  hovered,
  selected,
  onHover,
  onSelect,
}: {
  layout: Layout;
  gpus: GpuVisual[];
  hovered: number | null;
  selected: number | null;
  onHover: (g: number | null) => void;
  onSelect: (g: number | null) => void;
}) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const n = layout.gpus.length;
  useLayoutEffect(() => {
    const m = ref.current;
    if (!m) return;
    layout.gpus.forEach((s, i) => {
      setBox(m, i, { x: s.pos.x, y: 0.06, z: s.pos.z }, { x: 0.94, y: 0.12, z: 0.94 });
      const g = gpus[i];
      const base = !g?.active ? '#16222e' : g.role === 'prefill' ? '#26344a' : g.role === 'decode' ? '#1c3640' : C.tile;
      col.set(i === selected ? '#3d6487' : i === hovered ? '#2f4c66' : base);
      m.setColorAt(i, col);
    });
    m.instanceMatrix.needsUpdate = true;
    if (m.instanceColor) m.instanceColor.needsUpdate = true;
    m.computeBoundingSphere();
  }, [layout, gpus, hovered, selected]);
  return (
    <instancedMesh
      key={n}
      ref={ref}
      args={[unitBox, undefined, n]}
      onPointerMove={(e: ThreeEvent<PointerEvent>) => {
        e.stopPropagation();
        if (e.instanceId !== undefined) onHover(e.instanceId);
      }}
      onPointerOut={() => onHover(null)}
      onClick={(e: ThreeEvent<MouseEvent>) => {
        e.stopPropagation();
        if (e.instanceId !== undefined) onSelect(e.instanceId === selected ? null : e.instanceId);
      }}
    >
      <meshStandardMaterial roughness={0.55} metalness={0.4} />
    </instancedMesh>
  );
}

const SEG_ORDER = ['weights', 'activations', 'overhead', 'kvUsed'] as const;
const SEG_COLORS: Record<(typeof SEG_ORDER)[number], string> = {
  weights: C.weights,
  activations: C.activations,
  overhead: C.overhead,
  kvUsed: C.kvUsed,
};

/** HBM towers: opaque used segments + translucent KV-free and unreserved headroom. */
export function HbmTowers({ layout, gpus, liveKv }: { layout: Layout; gpus: GpuVisual[]; liveKv?: (gpu: number) => number | null }) {
  const solid = useRef<THREE.InstancedMesh>(null);
  const free = useRef<THREE.InstancedMesh>(null);
  const n = layout.gpus.length;
  useLayoutEffect(() => {
    const ms = solid.current;
    const mf = free.current;
    if (!ms || !mf) return;
    layout.gpus.forEach((s, i) => {
      const g = gpus[i]!;
      let y = 0.12;
      const z = s.pos.z + (TOWER.z0 + TOWER.z1) / 2;
      const x = s.pos.x + (TOWER.x0 + TOWER.x1) / 2;
      const tw = TOWER.x1 - TOWER.x0;
      const td = TOWER.z1 - TOWER.z0;
      SEG_ORDER.forEach((k, j) => {
        const h = g.mem[k] * TOWER_H;
        // 2% surface gap between stacked segments keeps neighbours distinct.
        const gap = h > 0.04 ? 0.012 : 0;
        setBox(ms, i * SEG_ORDER.length + j, { x, y: y + h / 2, z }, { x: tw, y: Math.max(0, h - gap), z: td });
        col.set(SEG_COLORS[k]);
        ms.setColorAt(i * SEG_ORDER.length + j, col);
        y += h;
      });
      const hFree = g.mem.kvFree * TOWER_H;
      setBox(mf, i * 2, { x, y: y + hFree / 2, z }, { x: tw, y: hFree, z: td });
      mf.setColorAt(i * 2, col.set(C.kvFree).multiplyScalar(1.6));
      y += hFree;
      const hU = g.mem.unreserved * TOWER_H;
      setBox(mf, i * 2 + 1, { x, y: y + hU / 2, z }, { x: tw, y: hU, z: td });
      mf.setColorAt(i * 2 + 1, col.set('#3a5670'));
    });
    for (const m of [ms, mf]) {
      m.instanceMatrix.needsUpdate = true;
      if (m.instanceColor) m.instanceColor.needsUpdate = true;
      m.computeBoundingSphere();
    }
  }, [layout, gpus]);
  // While the simulation runs, the KV-in-use segment tracks the live block pool.
  useFrame(() => {
    const ms = solid.current;
    const mf = free.current;
    if (!liveKv || !ms || !mf) return;
    let any = false;
    layout.gpus.forEach((s, i) => {
      const g = gpus[i]!;
      const f = liveKv(i);
      if (f === null || !g.active) return;
      any = true;
      const kv = g.mem.kvUsed + g.mem.kvFree;
      const used = kv * f;
      const x = s.pos.x + (TOWER.x0 + TOWER.x1) / 2;
      const z = s.pos.z + (TOWER.z0 + TOWER.z1) / 2;
      const tw = TOWER.x1 - TOWER.x0;
      const td = TOWER.z1 - TOWER.z0;
      const y0 = 0.12 + (g.mem.weights + g.mem.activations + g.mem.overhead) * TOWER_H;
      const hU = used * TOWER_H;
      setBox(ms, i * SEG_ORDER.length + 3, { x, y: y0 + hU / 2, z }, { x: tw, y: Math.max(0, hU - 0.012), z: td });
      const hF = (kv - used) * TOWER_H;
      setBox(mf, i * 2, { x, y: y0 + hU + hF / 2, z }, { x: tw, y: hF, z: td });
    });
    if (any) {
      ms.instanceMatrix.needsUpdate = true;
      mf.instanceMatrix.needsUpdate = true;
    }
  });
  return (
    <group>
      <instancedMesh key={`s${n}`} ref={solid} args={[unitBox, undefined, n * SEG_ORDER.length]} raycast={() => null}>
        <meshStandardMaterial roughness={0.45} metalness={0.25} />
      </instancedMesh>
      <instancedMesh key={`f${n}`} ref={free} args={[unitBox, undefined, n * 2]} raycast={() => null}>
        <meshStandardMaterial transparent opacity={0.22} depthWrite={false} roughness={0.3} />
      </instancedMesh>
    </group>
  );
}

const ATTN_COLOR = { full: C.attnFull, local: C.attnLocal, linear: C.attnLinear } as const;

export function LayerPlates({
  layout,
  plates,
  heat,
  experts,
}: {
  layout: Layout;
  plates: PlateVisual[];
  heat?: (out: Float32Array) => boolean;
  /** Live per-expert load (0..1) indexed by global expert id, or null. */
  experts?: () => Float32Array | null;
}) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const material = useMemo(() => plateMaterial(), []);
  const n = plates.length;
  const geom = useMemo(() => {
    const g = new THREE.BoxGeometry(1, 1, 1);
    const count = Math.max(1, n);
    g.setAttribute('aAttn', new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3));
    g.setAttribute('aFfn', new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3));
    g.setAttribute('aStripes', new THREE.InstancedBufferAttribute(new Float32Array(count), 1));
    g.setAttribute('aHeat', new THREE.InstancedBufferAttribute(new Float32Array(count), 1));
    g.setAttribute('aExpertLo', new THREE.InstancedBufferAttribute(new Float32Array(count), 1));
    return g;
  }, [n]);
  const expertTex = useRef<THREE.DataTexture | null>(null);
  useLayoutEffect(() => {
    const m = ref.current;
    if (!m) return;
    const aAttn = geom.getAttribute('aAttn') as THREE.InstancedBufferAttribute;
    const aFfn = geom.getAttribute('aFfn') as THREE.InstancedBufferAttribute;
    const aStr = geom.getAttribute('aStripes') as THREE.InstancedBufferAttribute;
    const aLo = geom.getAttribute('aExpertLo') as THREE.InstancedBufferAttribute;
    plates.forEach((p, i) => {
      const s = layout.gpus[p.gpu]!;
      const w = p.x1 - p.x0;
      setBox(m, i, { x: s.pos.x + p.x0 + w / 2, y: p.y + p.h / 2, z: s.pos.z + (STACK.z0 + STACK.z1) / 2 }, { x: w, y: p.h, z: STACK.z1 - STACK.z0 });
      col.set(ATTN_COLOR[p.attn]);
      aAttn.setXYZ(i, col.r, col.g, col.b);
      col.set(p.moe ? C.moeA : C.ffnDense);
      aFfn.setXYZ(i, col.r, col.g, col.b);
      aStr.setX(i, p.stripes > 64 ? 0 : Math.max(0, p.stripes));
      aLo.setX(i, p.expertLo);
    });
    m.instanceMatrix.needsUpdate = true;
    aAttn.needsUpdate = aFfn.needsUpdate = aStr.needsUpdate = aLo.needsUpdate = true;
    m.computeBoundingSphere();
  }, [plates, layout, geom]);
  // Per-frame glow (step trace wavefront, expert load); the callback reports whether it wrote.
  useFrame(() => {
    if (heat) {
      const a = geom.getAttribute('aHeat') as THREE.InstancedBufferAttribute;
      if (heat(a.array as Float32Array)) a.needsUpdate = true;
    }
    const load = experts?.() ?? null;
    const u = material.uniforms;
    if (!load) {
      u['uExpertOn']!.value = 0;
      return;
    }
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
  if (!n) return null;
  return <instancedMesh key={n} ref={ref} args={[geom, material, n]} raycast={() => null} frustumCulled={false} />;
}

/** Translucent outline of the whole stage per GPU, so a TP slice reads as a fraction of the layer. */
export function GhostStacks({ layout, plates }: { layout: Layout; plates: PlateVisual[] }) {
  const boxes = useMemo(() => {
    const byGpu = new Map<number, { lo: number; hi: number }>();
    for (const p of plates) {
      const b = byGpu.get(p.gpu);
      if (!b) byGpu.set(p.gpu, { lo: p.y, hi: p.y + p.h });
      else {
        b.lo = Math.min(b.lo, p.y);
        b.hi = Math.max(b.hi, p.y + p.h);
      }
    }
    const out: Box[] = [];
    for (const [g, r] of byGpu) {
      const s = layout.gpus[g];
      if (!s) continue;
      out.push({
        center: { x: s.pos.x + (STACK.x0 + STACK.x1) / 2, y: (r.lo + r.hi) / 2, z: s.pos.z + (STACK.z0 + STACK.z1) / 2 },
        size: { x: STACK.x1 - STACK.x0 + 0.02, y: r.hi - r.lo + 0.02, z: STACK.z1 - STACK.z0 + 0.02 },
      });
    }
    return out;
  }, [layout, plates]);
  return <Boxes boxes={boxes} color={'#7fa3c4'} opacity={0.07} />;
}

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

/** Faint wiring: NVSwitch spokes, mesh edges, NIC rails, and the active traffic paths. */
export function Wiring({ layout, links }: { layout: Layout; links: LinkVisual[] }) {
  const geom = useMemo(() => {
    const pts: number[] = [];
    if (!layout.meshEdges.length) {
      for (const g of layout.gpus) {
        if (layout.switches.length === 0) continue;
        pts.push(g.pos.x, TOWER_H + 0.05, g.pos.z, g.switchPort.x, g.switchPort.y, g.switchPort.z);
      }
    }
    for (const [a, b] of layout.meshEdges) {
      const pa = layout.gpus[a]!.pos;
      const pb = layout.gpus[b]!.pos;
      pts.push(pa.x, 0.2, pa.z, pb.x, 0.2, pb.z);
    }
    if (layout.spine) {
      const y = layout.spine.center.y;
      const z = layout.spine.center.z;
      for (const l of layout.leaves) pts.push(l.x, l.y, l.z, l.x, y, z);
    }
    for (const l of links) if (l.kind === 'pp' || l.kind === 'kvx') pts.push(...bezierPoints(l, 16));
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    return g;
  }, [layout, links]);
  return (
    <lineSegments geometry={geom} raycast={() => null}>
      <lineBasicMaterial color={C.line} transparent opacity={0.85} />
    </lineSegments>
  );
}

const TRAFFIC_COLOR: Record<LinkVisual['kind'], string> = { tp: C.tp, ep: C.ep, pp: C.pp, kvx: C.kvx };

const PER_LINK = 28;

/**
 * Traffic particles. Each link owns PER_LINK particles; a per-link intensity texture decides how
 * many are visible, so live utilization changes cost one tiny texture upload, not a rebuild.
 */
export function Particles({ links, level, speed = 1 }: { links: LinkVisual[]; level: (l: LinkVisual, i: number) => number; speed?: number }) {
  const material = useMemo(() => particleMaterial(), []);
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
    const color = new Float32Array(n * 3);
    const link = new Float32Array(n);
    const rank = new Float32Array(n);
    let seed = 1;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    links.forEach((l, li) => {
      const c = new THREE.Color(TRAFFIC_COLOR[l.kind]);
      const len = Math.hypot(l.to.x - l.from.x, l.to.y - l.from.y, l.to.z - l.from.z) + 0.5;
      for (let i = 0; i < PER_LINK; i++) {
        const j = li * PER_LINK + i;
        from.set([l.from.x, l.from.y, l.from.z], j * 3);
        ctrl.set([l.ctrl.x, l.ctrl.y, l.ctrl.z], j * 3);
        to.set([l.to.x, l.to.y, l.to.z], j * 3);
        // Interleave ranks so any visible subset is spread evenly along the link.
        const r = ((i * 11) % PER_LINK) / PER_LINK;
        rank[j] = r;
        phase[j] = i / PER_LINK + rnd() * 0.12;
        spd[j] = 0.55 / len;
        color.set([c.r, c.g, c.b], j * 3);
        link[j] = li;
      }
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    g.setAttribute('aFrom', new THREE.BufferAttribute(from, 3));
    g.setAttribute('aCtrl', new THREE.BufferAttribute(ctrl, 3));
    g.setAttribute('aTo', new THREE.BufferAttribute(to, 3));
    g.setAttribute('aPhase', new THREE.BufferAttribute(phase, 1));
    g.setAttribute('aSpeed', new THREE.BufferAttribute(spd, 1));
    g.setAttribute('aColor', new THREE.BufferAttribute(color, 3));
    g.setAttribute('aLink', new THREE.BufferAttribute(link, 1));
    g.setAttribute('aRank', new THREE.BufferAttribute(rank, 1));
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e4);
    return g;
  }, [links]);
  useFrame((state, dt) => {
    const u = material.uniforms;
    u['uTime']!.value += dt * speed;
    u['uPixelRatio']!.value = state.gl.getPixelRatio();
    u['uIntensity']!.value = tex;
    u['uLinks']!.value = Math.max(1, links.length);
    const data = tex.image.data as Float32Array;
    let changed = false;
    for (let i = 0; i < links.length; i++) {
      const k = level(links[i]!, i);
      if (Math.abs(data[i]! - k) > 0.01) {
        data[i] = k;
        changed = true;
      }
    }
    if (changed) tex.needsUpdate = true;
  });
  if (!links.length) return null;
  return <points geometry={geom} material={material} raycast={() => null} frustumCulled={false} />;
}

export function BoardLabels({ layout }: { layout: Layout }) {
  const many = layout.boards.length > 8;
  return (
    <group>
      {(many ? layout.platforms : layout.boards).map((b, i) => (
        <Text
          key={i}
          font={fontUrl}
          fontSize={many ? 0.34 : 0.2}
          color={C.muted}
          anchorX="left"
          anchorY="top"
          position={[b.center.x - b.size.x / 2 + 0.1, 0.02, b.center.z + b.size.z / 2 + 0.12]}
          rotation={[-Math.PI / 2, 0, 0]}
        >
          {b.label}
        </Text>
      ))}
    </group>
  );
}

export function GroupOutlines({ rects }: { rects: { min: Vec3; max: Vec3; label: string; color: string }[] }) {
  const geom = useMemo(() => {
    const pts: number[] = [];
    for (const r of rects) {
      const y = 0.015;
      const c = [
        [r.min.x, r.min.z],
        [r.max.x, r.min.z],
        [r.max.x, r.max.z],
        [r.min.x, r.max.z],
      ];
      for (let i = 0; i < 4; i++) {
        const a = c[i]!;
        const b = c[(i + 1) % 4]!;
        pts.push(a[0]!, y, a[1]!, b[0]!, y, b[1]!);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    return g;
  }, [rects]);
  return (
    <group>
      <lineSegments geometry={geom} raycast={() => null}>
        <lineBasicMaterial color={'#7fa3c4'} transparent opacity={0.7} />
      </lineSegments>
      {rects.map((r, i) => (
        <Text
          key={i}
          font={fontUrl}
          fontSize={0.19}
          color={r.color}
          anchorX="left"
          anchorY="bottom"
          position={[r.min.x + 0.05, 0.02, r.min.z - 0.06]}
          rotation={[-Math.PI / 2, 0, 0]}
        >
          {r.label}
        </Text>
      ))}
    </group>
  );
}

export function GpuTooltip({ layout, g, label }: { layout: Layout; g: GpuVisual | undefined; label: string }) {
  if (!g) return null;
  const s = layout.gpus[g.gpu];
  if (!s) return null;
  const sh = g.shard;
  return (
    <Html position={[s.pos.x, TOWER_H + 0.5, s.pos.z]} center style={{ pointerEvents: 'none' }} zIndexRange={[20, 0]}>
      <div className="tooltip">
        <div className="tooltip-title">GPU {g.gpu}</div>
        {sh ? (
          <>
            <div>{label}</div>
            <div>
              Layers {sh.layerLo}–{sh.layerHi - 1}
              {sh.experts ? `, experts ${sh.experts.lo}–${sh.experts.hi - 1}` : ''}
            </div>
            <div>
              Weights {fmtBytes(g.memBytes.weights)}, KV {fmtBytes(g.memBytes.kvUsed + g.memBytes.kvFree)}
            </div>
          </>
        ) : (
          <div>Idle (not used by this deployment)</div>
        )}
      </div>
    </Html>
  );
}
