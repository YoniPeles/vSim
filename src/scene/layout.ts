// Physical layout of a cluster in scene units (x right, y up, z toward the viewer).
// Scale-up domains are platforms on the floor; GPUs sit on node boards; each GPU carries an HBM
// tower and a stack of layer plates rising in y.

import type { ClusterSpec } from '../core/types.ts';

export const TILE = 1;
export const GAP = 0.42;
export const PITCH = TILE + GAP;
export const TOWER_H = 1.8;
/** Height of the NVSwitch bar floating above each node (drawn above so link pulses stay visible). */
export const SWITCH_Y = TOWER_H + 0.95;

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface GpuSlot {
  gpu: number;
  node: number;
  domain: number;
  pos: Vec3; // tile center (top surface at y = 0.12)
  /** Where its scale-up link meets the switch (or peer, for meshes). */
  switchPort: Vec3;
  /** NIC uplink start (back edge of the board). */
  nic: Vec3;
  /** Index into `Layout.switches` of the scale-up switch this GPU hangs off, or −1 (mesh / single GPU). */
  switchIdx: number;
}

export interface Box {
  center: Vec3;
  size: Vec3;
}

export interface Layout {
  gpus: GpuSlot[];
  boards: (Box & { node: number; label: string })[];
  platforms: (Box & { domain: number; label: string })[];
  switches: (Box & { domain: number })[];
  /** Scale-out spine (above and behind everything). */
  spine: Box | null;
  /** Leaf point per node where rails meet before going to the spine. */
  leaves: Vec3[];
  meshEdges: [number, number][];
  bounds: { min: Vec3; max: Vec3 };
}

const v = (x: number, y: number, z: number): Vec3 => ({ x, y, z });

export function layoutCluster(c: ClusterSpec): Layout {
  const gpus: GpuSlot[] = [];
  const boards: Layout['boards'] = [];
  const platforms: Layout['platforms'] = [];
  const switches: Layout['switches'] = [];
  const meshEdges: [number, number][] = [];
  const leaves: Vec3[] = [];
  const total = c.nodes * c.gpusPerNode;
  const nodesPerDomain = Math.max(1, Math.round(c.domainSize / c.gpusPerNode));
  const domains = Math.ceil(c.nodes / nodesPerDomain);

  // Domain footprint depends on its internal arrangement.
  let domainW = 0;
  let domainD = 0;
  const nvl = nodesPerDomain > 2; // rack-scale domain (NVL72-like): trays around a switch spine
  const mesh = c.scaleUp === 'mesh';
  const perNode = c.gpusPerNode;
  const cols = perNode >= 8 ? 4 : perNode;
  const rows = Math.ceil(perNode / cols);
  if (nvl) {
    const traysPerSide = Math.ceil(nodesPerDomain / 2);
    domainW = 2 * perNode * PITCH + 2.2;
    domainD = traysPerSide * PITCH * 1.05 + 0.8;
  } else if (mesh) {
    domainW = 5.2 * PITCH;
    domainD = 5.2 * PITCH;
  } else {
    domainW = cols * PITCH + 0.9;
    domainD = rows * PITCH + (rows > 1 ? 1.1 : 0.9);
  }
  const domainsPerRow = Math.max(1, Math.ceil(Math.sqrt(domains * (domainD / domainW))));
  const rowsOfDomains = Math.ceil(domains / domainsPerRow);
  const dGapX = 1.6;
  const dGapZ = 2.4;

  let gpu = 0;
  for (let d = 0; d < domains; d++) {
    const dc = d % domainsPerRow;
    const dr = Math.floor(d / domainsPerRow);
    const cx = (dc - (Math.min(domains, domainsPerRow) - 1) / 2) * (domainW + dGapX);
    const cz = (dr - (rowsOfDomains - 1) / 2) * (domainD + dGapZ);
    platforms.push({
      domain: d,
      label: nvl ? `NVLink domain ${d} (${c.domainSize} GPUs)` : `Node ${d}`,
      center: v(cx, -0.08, cz),
      size: v(domainW, 0.08, domainD),
    });
    const nodesHere = Math.min(nodesPerDomain, c.nodes - d * nodesPerDomain);

    if (nvl) {
      // Switch spine down the middle (z), trays on both sides.
      switches.push({ domain: d, center: v(cx, SWITCH_Y, cz), size: v(0.5, 0.14, domainD - 0.6) });
      const sw = switches.length - 1;
      const traysPerSide = Math.ceil(nodesHere / 2);
      for (let n = 0; n < nodesHere; n++) {
        const node = d * nodesPerDomain + n;
        const side = n < traysPerSide ? -1 : 1;
        const row = n % traysPerSide;
        const tz = cz + (row - (traysPerSide - 1) / 2) * PITCH * 1.05;
        const tx = cx + side * (0.9 + (perNode * PITCH) / 2);
        boards.push({ node, label: `Tray ${n}`, center: v(tx, 0, tz), size: v(perNode * PITCH - 0.1, 0.06, TILE + 0.16) });
        leaves.push(v(tx + side * ((perNode * PITCH) / 2), 0.1, tz));
        for (let g = 0; g < perNode && gpu < total; g++) {
          const gx = tx + (g - (perNode - 1) / 2) * PITCH;
          gpus.push({
            gpu,
            node,
            domain: d,
            pos: v(gx, 0.12, tz),
            switchPort: v(cx + side * 0.25, SWITCH_Y, tz),
            nic: v(gx, 0.12, tz + (TILE / 2) * 0.9),
            switchIdx: sw,
          });
          gpu++;
        }
      }
    } else {
      for (let n = 0; n < nodesHere; n++) {
        const node = d * nodesPerDomain + n;
        const bz = cz + (n - (nodesHere - 1) / 2) * (rows * PITCH + 0.8);
        boards.push({
          node,
          label: `Node ${node}`,
          center: v(cx, 0, bz),
          size: v(domainW - 0.3, 0.06, rows * PITCH + (rows > 1 ? 0.7 : 0.5)),
        });
        const sw = c.scaleUp !== 'mesh' && perNode > 1 ? switches.length : -1;
        if (sw >= 0) {
          switches.push({
            domain: d,
            center: v(cx, SWITCH_Y, bz),
            size: v(cols * PITCH - 0.4, 0.12, 0.22),
          });
        }
        leaves.push(v(cx, 0.1, bz - (rows * PITCH) / 2 - 0.3));
        const first = gpu;
        for (let g = 0; g < perNode && gpu < total; g++) {
          let gx: number;
          let gz: number;
          if (mesh) {
            const ang = (g / perNode) * Math.PI * 2 - Math.PI / 2;
            const r = 1.9 * PITCH * 0.95;
            gx = cx + Math.cos(ang) * r;
            gz = bz + Math.sin(ang) * r;
          } else {
            const col = g % cols;
            const row = Math.floor(g / cols);
            gx = cx + (col - (cols - 1) / 2) * PITCH;
            gz = bz + (row - (rows - 1) / 2) * (PITCH + 0.35);
          }
          const toward = rows > 1 ? Math.sign(gz - bz) : 0;
          gpus.push({
            gpu,
            node,
            domain: d,
            pos: v(gx, 0.12, gz),
            switchPort: mesh ? v(gx, 0.12, gz) : v(gx, SWITCH_Y, bz + toward * 0.08),
            nic: v(gx, 0.12, gz - TILE / 2),
            switchIdx: sw,
          });
          gpu++;
        }
        if (mesh) for (let a = first; a < gpu; a++) for (let b = a + 1; b < gpu; b++) meshEdges.push([a, b]);
      }
    }
  }

  // Bounds before the spine.
  const min = v(Infinity, 0, Infinity);
  // Without a switch or spine overhead the scene ends at the top of the memory columns.
  const overhead = switches.length > 0 || c.nodes > 1 || domains > 1;
  const max = v(-Infinity, overhead ? SWITCH_Y + 0.3 : TOWER_H + 0.3, -Infinity);
  for (const p of platforms) {
    min.x = Math.min(min.x, p.center.x - p.size.x / 2);
    max.x = Math.max(max.x, p.center.x + p.size.x / 2);
    min.z = Math.min(min.z, p.center.z - p.size.z / 2);
    max.z = Math.max(max.z, p.center.z + p.size.z / 2);
  }
  const spine =
    c.nodes > 1 || domains > 1
      ? { center: v((min.x + max.x) / 2, SWITCH_Y + 1.3, min.z - 1.2), size: v(Math.max(2, max.x - min.x), 0.14, 0.3) }
      : null;
  if (spine) min.z = Math.min(min.z, spine.center.z - 0.4);
  return { gpus, boards, platforms, switches, spine, leaves, meshEdges, bounds: { min, max } };
}
