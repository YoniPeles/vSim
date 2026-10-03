// Labels, parallel-group outlines, the selection marker and the GPU tooltip.

import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Html, Text } from '@react-three/drei';
import * as THREE from 'three';
import fontUrl from '@fontsource/ibm-plex-sans-condensed/files/ibm-plex-sans-condensed-latin-500-normal.woff?url';
import { C } from '../ui/theme.ts';
import { fmtBytes } from '../core/units.ts';
import { TOWER_H, type Layout, type Vec3 } from './layout.ts';
import type { GpuVisual } from './sceneModel.ts';
import { Decals, onOverlay } from './hardware.tsx';
import { useDispose } from './instancing.ts';

export interface GroupRect {
  min: Vec3;
  max: Vec3;
  label: string;
  /** 1 = DP replica / instance, 2 = prefill instance, 3 = decode instance. */
  style: number;
}

const FLAT: [number, number, number] = [-Math.PI / 2, 0, 0];

/** Node / rack names on the floor behind each tray (or each rack, when trays are many). */
export function FloorLabels({ layout }: { layout: Layout }) {
  const many = layout.boards.length > 8;
  const items = many ? layout.platforms : layout.boards;
  return (
    <group>
      {items.map((b, i) => (
        <Text
          key={i}
          font={fontUrl}
          fontSize={many ? 0.3 : 0.15}
          color={C.muted}
          fillOpacity={0.8}
          anchorX="left"
          anchorY="bottom"
          position={[b.center.x - b.size.x / 2 + 0.05, 0.035, b.center.z - b.size.z / 2 - 0.08]}
          rotation={FLAT}
          onUpdate={onOverlay}
        >
          {b.label}
        </Text>
      ))}
    </group>
  );
}

/** Corner-bracketed outlines around DP replicas or instances, labelled with their parallel shape. */
export function GroupOutlines({ rects, scale = 1 }: { rects: GroupRect[]; scale?: number }) {
  const decals = useMemo(() => rects.map((r) => ({ min: r.min, max: r.max, style: r.style, color: '#a9c6e4' })), [rects]);
  return (
    <group>
      <Decals rects={decals} y={0.036} />
      {rects.map((r, i) => (
        <Text
          key={i}
          font={fontUrl}
          fontSize={0.14 * scale}
          color={C.text}
          anchorX="left"
          anchorY="top"
          position={[r.min.x + 0.04, 0.04, r.max.z + 0.06]}
          rotation={FLAT}
          onUpdate={onOverlay}
        >
          {r.label}
        </Text>
      ))}
    </group>
  );
}

/** Corner brackets on the package plus faint posts rising around the GPU's column and stack. */
function markerGeometry(): THREE.BufferGeometry {
  const pos: number[] = [];
  const cols: number[] = [];
  const h = 0.52;
  const arm = 0.16;
  const y = 0.085;
  const top = TOWER_H + 0.14;
  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) {
      const x = sx * h;
      const z = sz * h;
      pos.push(x, y, z, x - sx * arm, y, z, x, y, z, x, y, z - sz * arm);
      cols.push(1, 1, 1, 0.2, 0.2, 0.2, 1, 1, 1, 0.2, 0.2, 0.2);
      pos.push(x, y, z, x, top, z);
      cols.push(0.55, 0.55, 0.55, 0, 0, 0);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
  return g;
}

export function SelectionMarkers({ layout, hovered, selected, motion }: { layout: Layout; hovered: number | null; selected: number | null; motion: boolean }) {
  const geom = useMemo(() => markerGeometry(), []);
  const selMat = useMemo(() => new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }), []);
  const hovMat = useMemo(() => new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }), []);
  useDispose(geom);
  useDispose(selMat);
  useDispose(hovMat);
  const t = useRef(0);
  useFrame((_, dt) => {
    t.current += dt;
    const pulse = motion ? 0.8 + 0.2 * Math.sin(t.current * 3) : 1;
    selMat.color.setRGB(0.85, 0.93, 1).multiplyScalar(pulse);
    hovMat.color.setRGB(0.45, 0.55, 0.68);
  });
  const at = (g: number | null) => (g !== null ? layout.gpus[g] : undefined);
  const s = at(selected);
  const h = hovered !== selected ? at(hovered) : undefined;
  return (
    <group>
      {s && <lineSegments geometry={geom} material={selMat} position={[s.pos.x, 0, s.pos.z]} raycast={() => null} renderOrder={5} />}
      {h && <lineSegments geometry={geom} material={hovMat} position={[h.pos.x, 0, h.pos.z]} raycast={() => null} renderOrder={5} />}
    </group>
  );
}

export function GpuTooltip({ layout, g, label }: { layout: Layout; g: GpuVisual | undefined; label: string }) {
  if (!g) return null;
  const s = layout.gpus[g.gpu];
  if (!s) return null;
  const sh = g.shard;
  return (
    <Html position={[s.pos.x, TOWER_H + 0.55, s.pos.z]} center style={{ pointerEvents: 'none' }} zIndexRange={[20, 0]}>
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
