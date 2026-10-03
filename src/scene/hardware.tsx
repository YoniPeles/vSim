// Physical hardware: the floor, scale-up domain markings, node trays, GPU packages (substrate,
// silicon die, HBM base), the switch fabric, and the invisible hit volumes used for hover/click.

import { useLayoutEffect, useMemo, useRef } from 'react';
import { useFrame, type ThreeEvent } from '@react-three/fiber';
import { MeshReflectorMaterial } from '@react-three/drei';
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import { TOWER_H, type Box, type Layout, type Vec3 } from './layout.ts';
import { STACK, TOWER, type GpuVisual } from './sceneModel.ts';
import { busMaterial, decalMaterial, dieMaterial, floorGridMaterial, skyMaterial } from './shaders.ts';
import { attr, instancedBox, setBox, useDispose } from './instancing.ts';

export const FLOOR_Y = -0.06;
/** Layer for floor-level overlays, which the reflection pass leaves out. */
export const OVERLAY_LAYER = 1;
export const onOverlay = (o: THREE.Object3D) => o.layers.set(OVERLAY_LAYER);
const SUB_Y0 = 0.03;
const SUB_Y1 = 0.075;

const tmp = new THREE.Object3D();
const col = new THREE.Color();

/** Raised-floor slab with soft reflections (when the GPU can afford them) and tile seams. */
export function Floor({ bounds, reflect }: { bounds: { min: Vec3; max: Vec3 }; reflect: boolean }) {
  const cx = (bounds.min.x + bounds.max.x) / 2;
  const cz = (bounds.min.z + bounds.max.z) / 2;
  const r = 0.5 * Math.hypot(bounds.max.x - bounds.min.x, bounds.max.z - bounds.min.z);
  const size = Math.max(60, r * 10);
  const grid = useMemo(() => floorGridMaterial(), []);
  useDispose(grid);
  useLayoutEffect(() => {
    (grid.uniforms['uCenter']!.value as THREE.Vector2).set(cx, cz);
    grid.uniforms['uRadius']!.value = Math.max(4, r * 1.3);
  }, [grid, cx, cz, r]);
  return (
    <group>
      <mesh rotation-x={-Math.PI / 2} position={[cx, FLOOR_Y, cz]} raycast={() => null}>
        <planeGeometry args={[size, size]} />
        {reflect ? (
          <MeshReflectorMaterial
            key="reflect"
            resolution={512}
            blur={[240, 90]}
            mixBlur={0.9}
            mixStrength={14}
            mixContrast={1}
            mirror={0}
            depthScale={0.9}
            minDepthThreshold={0.3}
            maxDepthThreshold={1.3}
            color="#0c1824"
            roughness={0.92}
            metalness={0.35}
          />
        ) : (
          <meshStandardMaterial key="plain" color="#0c1824" roughness={0.9} metalness={0.3} />
        )}
      </mesh>
      <mesh rotation-x={-Math.PI / 2} position={[cx, FLOOR_Y + 0.002, cz]} material={grid} raycast={() => null} renderOrder={-1} onUpdate={onOverlay}>
        <planeGeometry args={[size, size]} />
      </mesh>
    </group>
  );
}

/** Dark dome around the scene, lighter towards the horizon. */
export function Sky({ horizon, zenith }: { horizon: string; zenith: string }) {
  const material = useMemo(() => skyMaterial(), []);
  useDispose(material);
  useLayoutEffect(() => {
    (material.uniforms['uHorizon']!.value as THREE.Color).set(horizon);
    (material.uniforms['uZenith']!.value as THREE.Color).set(zenith);
  }, [material, horizon, zenith]);
  return (
    <mesh material={material} renderOrder={-10} raycast={() => null} frustumCulled={false}>
      <sphereGeometry args={[900, 32, 16]} />
    </mesh>
  );
}

/** Flat outlines on the floor: scale-up domains (racks) and parallel groups. */
export function Decals({ rects, y }: { rects: { min: Vec3; max: Vec3; style: number; color: string }[]; y: number }) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const material = useMemo(() => decalMaterial(), []);
  const n = rects.length;
  const geom = useMemo(() => {
    const g = new THREE.PlaneGeometry(1, 1);
    g.rotateX(-Math.PI / 2);
    const count = Math.max(1, n);
    g.setAttribute('aStyle', new THREE.InstancedBufferAttribute(new Float32Array(count), 1));
    g.setAttribute('aColor', new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3));
    return g;
  }, [n]);
  useDispose(geom);
  useDispose(material);
  useLayoutEffect(() => {
    const m = ref.current;
    if (!m) return;
    const st = attr(geom, 'aStyle');
    const ac = attr(geom, 'aColor');
    rects.forEach((r, i) => {
      setBox(m, i, { x: (r.min.x + r.max.x) / 2, y, z: (r.min.z + r.max.z) / 2 }, { x: r.max.x - r.min.x, y: 1, z: r.max.z - r.min.z });
      st.setX(i, r.style);
      col.set(r.color);
      ac.setXYZ(i, col.r, col.g, col.b);
    });
    m.instanceMatrix.needsUpdate = true;
    st.needsUpdate = ac.needsUpdate = true;
  }, [rects, geom, y]);
  if (!n) return null;
  return <instancedMesh key={n} ref={ref} args={[geom, material, n]} raycast={() => null} frustumCulled={false} onUpdate={onOverlay} />;
}

/** Instanced rounded slabs that all share one size (trays, package substrates, HBM base dies). */
function Slabs({
  centers,
  size,
  radius,
  colors,
  roughness,
  metalness,
}: {
  centers: Vec3[];
  size: Vec3;
  radius: number;
  colors: string[];
  roughness: number;
  metalness: number;
}) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const geom = useMemo(() => new RoundedBoxGeometry(size.x, size.y, size.z, 2, Math.min(radius, size.y / 2 - 1e-3)), [size.x, size.y, size.z, radius]);
  useDispose(geom);
  const n = centers.length;
  useLayoutEffect(() => {
    const m = ref.current;
    if (!m) return;
    centers.forEach((c, i) => {
      tmp.position.set(c.x, c.y, c.z);
      tmp.scale.set(1, 1, 1);
      tmp.updateMatrix();
      m.setMatrixAt(i, tmp.matrix);
      m.setColorAt(i, col.set(colors[i] ?? colors[0] ?? '#ffffff'));
    });
    m.instanceMatrix.needsUpdate = true;
    if (m.instanceColor) m.instanceColor.needsUpdate = true;
    m.computeBoundingSphere();
  }, [centers, colors]);
  if (!n) return null;
  return (
    <instancedMesh key={n} ref={ref} args={[geom, undefined, n]} raycast={() => null}>
      <meshStandardMaterial roughness={roughness} metalness={metalness} />
    </instancedMesh>
  );
}

/** Node trays: thin machined slabs. */
export function Trays({ layout }: { layout: Layout }) {
  const b = layout.boards[0];
  const centers = useMemo(() => layout.boards.map((x) => ({ x: x.center.x, y: (-0.045 + SUB_Y0) / 2, z: x.center.z })), [layout]);
  if (!b) return null;
  return (
    <Slabs centers={centers} size={{ x: b.size.x, y: SUB_Y0 + 0.045, z: b.size.z }} radius={0.035} colors={['#142131']} roughness={0.5} metalness={0.6} />
  );
}

/** GPU packages: substrate, silicon die under the layer stack, and the HBM base die under the memory column. */
export function Packages({ layout, gpus }: { layout: Layout; gpus: GpuVisual[] }) {
  const subs = useMemo(() => layout.gpus.map((s) => ({ x: s.pos.x, y: (SUB_Y0 + SUB_Y1) / 2, z: s.pos.z })), [layout]);
  const subColors = useMemo(() => layout.gpus.map((_, i) => (gpus[i]?.active ? '#1f2f40' : '#151f2a')), [layout, gpus]);
  const bases = useMemo(
    () =>
      layout.gpus.flatMap((s) =>
        [-1, 1].map((k) => ({ x: s.pos.x + (TOWER.x0 + TOWER.x1) / 2, y: (SUB_Y1 + 0.12) / 2, z: s.pos.z + (TOWER.z0 + TOWER.z1) / 2 + k * 0.2 })),
      ),
    [layout],
  );
  return (
    <group>
      <Slabs centers={subs} size={{ x: 0.94, y: SUB_Y1 - SUB_Y0, z: 0.94 }} radius={0.02} colors={subColors} roughness={0.45} metalness={0.55} />
      <Slabs
        centers={bases}
        size={{ x: TOWER.x1 - TOWER.x0 + 0.02, y: 0.12 - SUB_Y1, z: (TOWER.z1 - TOWER.z0) / 2 - 0.02 }}
        radius={0.012}
        colors={['#3a4a5c']}
        roughness={0.3}
        metalness={0.85}
      />
      <Dies layout={layout} gpus={gpus} />
    </group>
  );
}

function Dies({ layout, gpus }: { layout: Layout; gpus: GpuVisual[] }) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const material = useMemo(() => dieMaterial(), []);
  const n = layout.gpus.length;
  const geom = useMemo(() => instancedBox(n, { aActive: 1 }), [n]);
  useDispose(geom);
  useDispose(material);
  useLayoutEffect(() => {
    const m = ref.current;
    if (!m) return;
    const a = attr(geom, 'aActive');
    layout.gpus.forEach((s, i) => {
      const x0 = STACK.x0 + 0.02;
      const x1 = STACK.x1 - 0.02;
      setBox(m, i, { x: s.pos.x + (x0 + x1) / 2, y: SUB_Y1 + 0.015, z: s.pos.z }, { x: x1 - x0, y: 0.03, z: STACK.z1 - STACK.z0 - 0.04 });
      a.setX(i, gpus[i]?.active ? 1 : 0);
    });
    m.instanceMatrix.needsUpdate = true;
    a.needsUpdate = true;
    m.computeBoundingSphere();
  }, [layout, gpus, geom]);
  return <instancedMesh key={n} ref={ref} args={[geom, material, n]} raycast={() => null} />;
}

/** Scale-up switches and the scale-out spine. `level(i)` is the live traffic through bus i (0..1). */
export function Buses({ boxes, level }: { boxes: Box[]; level: (i: number) => number }) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const material = useMemo(() => busMaterial(), []);
  const n = boxes.length;
  const geom = useMemo(() => instancedBox(n, { aLevel: 1 }), [n]);
  useDispose(geom);
  useDispose(material);
  useLayoutEffect(() => {
    const m = ref.current;
    if (!m) return;
    boxes.forEach((b, i) => setBox(m, i, b.center, b.size));
    m.instanceMatrix.needsUpdate = true;
    m.computeBoundingSphere();
  }, [boxes]);
  useFrame((_, dt) => {
    material.uniforms['uTime']!.value += dt;
    const a = attr(geom, 'aLevel');
    let changed = false;
    for (let i = 0; i < n; i++) {
      // Ease towards the target so the filament breathes rather than flickers.
      const cur = a.getX(i);
      const next = cur + (level(i) - cur) * Math.min(1, dt * 4);
      if (Math.abs(next - cur) > 1e-3) {
        a.setX(i, next);
        changed = true;
      }
    }
    if (changed) a.needsUpdate = true;
  });
  if (!n) return null;
  return <instancedMesh key={n} ref={ref} args={[geom, material, n]} raycast={() => null} />;
}

/** Invisible volumes around each GPU (package + memory column + stack) for hover, click and double-click. */
export function HitTargets({
  layout,
  selected,
  onHover,
  onSelect,
  onFocus,
}: {
  layout: Layout;
  selected: number | null;
  onHover: (g: number | null) => void;
  onSelect: (g: number | null) => void;
  onFocus: (g: number) => void;
}) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const n = layout.gpus.length;
  const geom = useMemo(() => new THREE.BoxGeometry(1, 1, 1), []);
  useDispose(geom);
  useLayoutEffect(() => {
    const m = ref.current;
    if (!m) return;
    const h = TOWER_H + 0.2;
    layout.gpus.forEach((s, i) => setBox(m, i, { x: s.pos.x, y: h / 2, z: s.pos.z }, { x: 0.96, y: h, z: 0.96 }));
    m.instanceMatrix.needsUpdate = true;
    m.computeBoundingSphere();
  }, [layout]);
  return (
    <instancedMesh
      key={n}
      ref={ref}
      args={[geom, undefined, n]}
      onPointerMove={(e: ThreeEvent<PointerEvent>) => {
        e.stopPropagation();
        if (e.instanceId !== undefined) onHover(e.instanceId);
      }}
      onPointerOut={() => onHover(null)}
      onClick={(e: ThreeEvent<MouseEvent>) => {
        e.stopPropagation();
        if (e.instanceId !== undefined) onSelect(e.instanceId === selected ? null : e.instanceId);
      }}
      onDoubleClick={(e: ThreeEvent<MouseEvent>) => {
        e.stopPropagation();
        if (e.instanceId !== undefined) onFocus(e.instanceId);
      }}
    >
      <meshBasicMaterial colorWrite={false} depthWrite={false} />
    </instancedMesh>
  );
}
