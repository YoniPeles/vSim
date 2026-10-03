import { useEffect } from 'react';
import * as THREE from 'three';
import type { Vec3 } from './layout.ts';

/** Frees a geometry, material or texture's GPU memory when it is replaced or unmounted. */
export function useDispose(obj: { dispose(): void } | null | undefined) {
  useEffect(() => () => obj?.dispose(), [obj]);
}

const tmp = new THREE.Object3D();

/** Places instance `i` as an axis-aligned box of `size` centred on `center`. */
export function setBox(mesh: THREE.InstancedMesh, i: number, center: Vec3, size: Vec3) {
  tmp.position.set(center.x, center.y, center.z);
  tmp.scale.set(Math.max(1e-4, size.x), Math.max(1e-4, size.y), Math.max(1e-4, size.z));
  tmp.rotation.set(0, 0, 0);
  tmp.updateMatrix();
  mesh.setMatrixAt(i, tmp.matrix);
}

/** Box geometry with extra per-instance float attributes of the given item sizes. */
export function instancedBox(count: number, attrs: Record<string, number>): THREE.BoxGeometry {
  const g = new THREE.BoxGeometry(1, 1, 1);
  const n = Math.max(1, count);
  for (const [name, size] of Object.entries(attrs)) {
    g.setAttribute(name, new THREE.InstancedBufferAttribute(new Float32Array(n * size), size));
  }
  return g;
}

export const attr = (g: THREE.BufferGeometry, name: string) => g.getAttribute(name) as THREE.InstancedBufferAttribute;

/** Linear RGB of a CSS hex colour (the scene renders with colour management on). */
export function linear(hex: string): [number, number, number] {
  const c = new THREE.Color(hex);
  return [c.r, c.g, c.b];
}

/** Whether the viewer asked the OS to minimise motion. */
export function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
}
