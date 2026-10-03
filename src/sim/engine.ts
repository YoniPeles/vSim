// Discrete-event simulation of vLLM serving: requests flow through the V1 scheduler
// (vllm/v1/core/sched/scheduler.py) on each DP replica, steps are timed by the compiled roofline
// cost model, and KV lives in a paged block pool with prefix caching. Pure TypeScript: runs in a
// Web Worker for the UI and in Node for tests.

import { buildCluster } from '../core/hardware/clusters.ts';
import { snapshotToModel } from '../core/model/presets.ts';
import { evaluate, type Evaluation, type InstanceEval } from '../core/engine/evaluate.ts';
import { CostModel } from '../core/engine/cost.ts';
import { chunkPairs, emptyShape } from '../core/engine/analytic.ts';
import { kvProfile, logicalKvBytesPerToken } from '../core/engine/memory.ts';
import type { BatchShape, CtxWindow, ModelSpec, StepCost } from '../core/types.ts';
import { BlockPool } from './blocks.ts';
import { Rng } from './rng.ts';
import type { LinkRates, Percentiles, ReplicaFrame, SimFrame, SimInputs, SimWorkload } from './types.ts';

const WAITING = 0;
const RUNNING = 1;
const TRANSFER = 2;
const DONE = 3;

const EV_ARRIVAL = 0;
const EV_LAUNCH = 1;
const EV_BATCH = 2;
const EV_XFER = 3;

/** Max blocks simulated per replica (bounds memory for huge KV pools; scaled back for display). */
const MAX_BLOCKS = 400_000;
const COLORS = 600;
const PREFIXES = 6;
const STATS_WINDOW = 20;

interface Req {
  id: number;
  user: number;
  prompt: number;
  out: number;
  prefixId: number;
  sharedTokens: number;
  computed: number;
  generated: number;
  state: number;
  inst: number;
  rep: Replica | null;
  blocks: number[];
  fixed: number[];
  busy: boolean;
  remote: boolean;
  arrival: number;
  firstTok: number;
  lastTok: number;
}

interface Replica {
  idx: number;
  inst: number;
  dp: number;
  pool: BlockPool;
  /** Blocks of the real pool represented by one simulated block. */
  scale: number;
  running: Req[];
  waiting: Req[];
  unit: Unit;
  lastDecode: number;
  lastPrefill: number;
  lastStep: number;
}

interface Unit {
  inst: number;
  replicas: Replica[];
  inflight: number;
  maxInflight: number;
  nextLaunch: number;
  launchPending: boolean;
}

interface Entry {
  req: Req;
  n: number;
  decode: boolean;
}

interface Batch {
  unit: Unit;
  entries: Entry[][];
}

interface InstRuntime {
  ev: InstanceEval;
  cost: CostModel;
  replicas: Replica[];
  units: Unit[];
  bs: number;
  fixedPerReq: number;
  /** Comm bytes per GPU accumulated since the last frame. */
  bytes: { tpIntra: number; epIntra: number; ppIntra: number; inter: number; kvx: number };
  bw: { su: number; so: number };
}

class Heap {
  private t: number[] = [];
  private k: number[] = [];
  private d: unknown[] = [];
  private s: number[] = [];
  private seq = 0;
  get size(): number {
    return this.t.length;
  }
  peekTime(): number {
    return this.t.length ? this.t[0]! : Infinity;
  }
  push(time: number, kind: number, data: unknown): void {
    const i0 = this.t.length;
    this.t.push(time);
    this.k.push(kind);
    this.d.push(data);
    this.s.push(this.seq++);
    let i = i0;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.less(i, p)) {
        this.swap(i, p);
        i = p;
      } else break;
    }
  }
  pop(): { t: number; kind: number; data: unknown } {
    const out = { t: this.t[0]!, kind: this.k[0]!, data: this.d[0] };
    const last = this.t.length - 1;
    this.swap(0, last);
    this.t.pop();
    this.k.pop();
    this.d.pop();
    this.s.pop();
    let i = 0;
    for (;;) {
      const l = 2 * i + 1;
      const r = l + 1;
      let m = i;
      if (l < this.t.length && this.less(l, m)) m = l;
      if (r < this.t.length && this.less(r, m)) m = r;
      if (m === i) break;
      this.swap(i, m);
      i = m;
    }
    return out;
  }
  private less(a: number, b: number): boolean {
    return this.t[a]! < this.t[b]! || (this.t[a] === this.t[b] && this.s[a]! < this.s[b]!);
  }
  private swap(a: number, b: number): void {
    [this.t[a], this.t[b]] = [this.t[b]!, this.t[a]!];
    [this.k[a], this.k[b]] = [this.k[b]!, this.k[a]!];
    [this.d[a], this.d[b]] = [this.d[b], this.d[a]];
    [this.s[a], this.s[b]] = [this.s[b]!, this.s[a]!];
  }
}

/** Timestamped samples kept for a sliding window. */
class Window {
  private t: number[] = [];
  private v: number[] = [];
  push(t: number, v: number): void {
    this.t.push(t);
    this.v.push(v);
    if (this.t.length > 20000) this.trim(t, STATS_WINDOW / 2);
  }
  trim(now: number, span = STATS_WINDOW): void {
    let i = 0;
    while (i < this.t.length && this.t[i]! < now - span) i++;
    if (i) {
      this.t.splice(0, i);
      this.v.splice(0, i);
    }
  }
  percentiles(now: number): Percentiles {
    this.trim(now);
    if (!this.v.length) return { p50: NaN, p90: NaN, p99: NaN };
    const s = [...this.v].sort((a, b) => a - b);
    const q = (p: number) => s[Math.min(s.length - 1, Math.floor(p * s.length))]!;
    return { p50: q(0.5), p90: q(0.9), p99: q(0.99) };
  }
  get count(): number {
    return this.v.length;
  }
}

export class Simulator {
  readonly model: ModelSpec;
  readonly evaluation: Evaluation;
  readonly insts: InstRuntime[];
  readonly replicas: Replica[] = [];
  now = 0;
  private heap = new Heap();
  private rng: Rng;
  private w: SimWorkload;
  private nextId = 0;
  private kvBytesPerToken: number;
  private activeXfers = 0;
  private xfers = new Set<Req>();
  private lastFrameT = 0;
  // stats
  private tokOut = 0;
  private tokOutAtFrame = 0;
  private completedAtFrame = 0;
  private completed = 0;
  private preemptions = 0;
  private promptTokens = 0;
  private hitTokens = 0;
  private ttft = new Window();
  private itl = new Window();
  private e2e = new Window();
  private rateEma = 0;
  private reqRateEma = 0;
  focus: number | null = null;
  // Sampled expert routing (for the expert-load heat map).
  private expertCdf: Float64Array | null = null;
  private expertCounts: Float64Array | null = null;
  private topK = 0;

  constructor(inputs: SimInputs) {
    this.rng = new Rng(inputs.seed);
    this.w = inputs.workload;
    this.model = snapshotToModel(inputs.snapshot);
    const cluster = buildCluster(inputs.cluster);
    this.evaluation = evaluate(this.model, cluster, inputs.deployment);
    this.insts = this.evaluation.instances.map((ev, i) => {
      if (ev.inst.issues.some((x) => x.level === 'error') || ev.memory.blocks <= 0) {
        throw new Error(`Instance "${ev.inst.id}" can't start: ${ev.inst.issues.find((x) => x.level === 'error')?.msg ?? 'no KV memory'}`);
      }
      const cost = new CostModel(this.model, ev.inst, ev.placement, cluster, this.evaluation.calib, { routingSkew: inputs.routingSkew });
      const realBlocks = ev.memory.blocks;
      const simBlocks = Math.min(realBlocks, MAX_BLOCKS);
      const bs = ev.inst.blockSize * (realBlocks / simBlocks);
      const sh = ev.placement.shards[0]!;
      const prof = kvProfile(this.model, ev.inst, sh.layerLo, sh.layerHi);
      const page = (prof.fullPerToken || 1) * ev.inst.blockSize;
      const typical = this.w.isl + this.w.osl;
      const windowed = prof.windowed.reduce((s, x) => s + x.perToken * Math.min(typical, x.size + ev.inst.blockSize), 0);
      const fixedPerReq = prof.fullPerToken ? Math.ceil((windowed + prof.statePerSeq) / page / (realBlocks / simBlocks)) : 0;
      const rt: InstRuntime = {
        ev,
        cost,
        replicas: [],
        units: [],
        bs,
        fixedPerReq,
        bytes: { tpIntra: 0, epIntra: 0, ppIntra: 0, inter: 0, kvx: 0 },
        bw: { su: cost.comm.suBW / cost.calib.etaScaleUp, so: cost.comm.soBW / cost.calib.etaScaleOut },
      };
      const lockstep = ev.inst.dp > 1 && this.model.layers.some((l) => l.ffn.kind === 'moe');
      const maxInflight = Math.max(1, ev.inst.pp);
      const mkUnit = (): Unit => ({ inst: i, replicas: [], inflight: 0, maxInflight, nextLaunch: 0, launchPending: false });
      const shared = lockstep ? mkUnit() : null;
      if (shared) rt.units.push(shared);
      for (let d = 0; d < ev.inst.dp; d++) {
        const unit = shared ?? mkUnit();
        if (!shared) rt.units.push(unit);
        const r: Replica = {
          idx: this.replicas.length,
          inst: i,
          dp: d,
          pool: new BlockPool(simBlocks),
          scale: realBlocks / simBlocks,
          running: [],
          waiting: [],
          unit,
          lastDecode: 0,
          lastPrefill: 0,
          lastStep: 0,
        };
        unit.replicas.push(r);
        rt.replicas.push(r);
        this.replicas.push(r);
      }
      return rt;
    });
    this.kvBytesPerToken = logicalKvBytesPerToken(this.model, this.insts[0]!.ev.inst);
    const moe = this.model.layers.find((l) => l.ffn.kind === 'moe')?.ffn;
    if (moe && moe.kind === 'moe') {
      // Expert popularity: Zipf-like with exponent from the routing-skew knob, randomly permuted.
      const E = moe.E;
      const s = 0.25 + inputs.routingSkew * 0.6;
      const order = Array.from({ length: E }, (_, i) => i);
      for (let i = E - 1; i > 0; i--) {
        const j = this.rng.int(i + 1);
        [order[i], order[j]] = [order[j]!, order[i]!];
      }
      const w = new Float64Array(E);
      order.forEach((e, rank) => (w[e] = 1 / (rank + 1) ** s));
      const cdf = new Float64Array(E);
      let acc = 0;
      for (let e = 0; e < E; e++) cdf[e] = acc += w[e]!;
      for (let e = 0; e < E; e++) cdf[e]! /= acc;
      this.expertCdf = cdf;
      this.expertCounts = new Float64Array(E);
      this.topK = moe.topK;
    }
    this.seedArrivals();
  }

  // ───────────────────────────── workload ─────────────────────────────

  private seedArrivals(): void {
    if (this.w.arrival === 'poisson') {
      this.heap.push(this.rng.exp(Math.max(1e-3, this.w.rate)), EV_ARRIVAL, -1);
      return;
    }
    // Closed loop: stagger users over the first second so the system ramps instead of stampeding.
    const n = Math.max(1, Math.round(this.w.concurrency));
    const ramp = Math.min(2, 0.002 * n);
    for (let u = 0; u < n; u++) this.heap.push((u / n) * ramp, EV_ARRIVAL, u);
  }

  setWorkload(w: SimWorkload): void {
    const oldUsers = this.w.arrival === 'closed' ? Math.round(this.w.concurrency) : 0;
    this.w = w;
    if (w.arrival === 'closed') {
      const n = Math.round(w.concurrency);
      for (let u = oldUsers; u < n; u++) this.heap.push(this.now + this.rng.next() * 0.5, EV_ARRIVAL, u);
    }
  }

  private newRequest(user: number): Req {
    const w = this.w;
    const maxLen = Math.min(...this.insts.map((rt) => rt.ev.inst.maxModelLen));
    const prompt = Math.min(maxLen - 1, Math.max(1, Math.round(this.rng.lognormal(w.isl, w.spread))));
    const out = Math.min(maxLen - prompt, Math.max(1, Math.round(this.rng.lognormal(w.osl, w.spread))));
    const shared = Math.floor(Math.min(prompt - 1, w.isl * w.prefixHit));
    return {
      id: this.nextId++,
      user,
      prompt,
      out,
      prefixId: this.rng.int(PREFIXES),
      sharedTokens: shared,
      computed: 0,
      generated: 0,
      state: WAITING,
      inst: 0,
      rep: null,
      blocks: [],
      fixed: [],
      busy: false,
      remote: false,
      arrival: this.now,
      firstTok: -1,
      lastTok: -1,
    };
  }

  private route(inst: number): Replica {
    const rs = this.insts[inst]!.replicas;
    let best = rs[0]!;
    let score = Infinity;
    for (const r of rs) {
      const s = r.waiting.length * 4 + r.running.length;
      if (s < score) {
        score = s;
        best = r;
      }
    }
    return best;
  }

  private enqueue(q: Req, inst: number): void {
    const r = this.route(inst);
    q.inst = inst;
    q.rep = r;
    q.state = WAITING;
    r.waiting.push(q);
    this.wake(r.unit);
  }

  private wake(u: Unit): void {
    if (u.launchPending || u.inflight >= u.maxInflight) return;
    u.launchPending = true;
    this.heap.push(Math.max(this.now, u.nextLaunch), EV_LAUNCH, u);
  }

  // ───────────────────────────── KV blocks ─────────────────────────────

  private blockHash(q: Req, i: number, bs: number): number {
    if ((i + 1) * bs <= q.sharedTokens) return q.prefixId * 1e7 + i;
    return 1e13 + q.id * 1e6 + i;
  }

  private grow(r: Replica, q: Req, tokens: number, rt: InstRuntime): boolean {
    if (!q.fixed.length && rt.fixedPerReq > 0) {
      if (!r.pool.allocate(rt.fixedPerReq, q.id % COLORS, q.fixed)) return false;
    }
    const need = Math.ceil(tokens / rt.bs) - q.blocks.length;
    if (need <= 0) return true;
    return r.pool.allocate(need, q.id % COLORS, q.blocks);
  }

  private releaseKv(r: Replica, q: Req): void {
    r.pool.release(q.blocks);
    r.pool.release(q.fixed);
  }

  private prefixHits(r: Replica, q: Req, rt: InstRuntime): void {
    if (!rt.ev.inst.flags.enablePrefixCaching) return;
    const limit = Math.floor((q.prompt + q.generated - 1) / rt.bs);
    let hits = 0;
    for (let i = 0; i < limit; i++) {
      const b = r.pool.lookup(this.blockHash(q, i, rt.bs));
      if (b < 0) break;
      r.pool.touch(b, q.id % COLORS);
      q.blocks.push(b);
      hits++;
    }
    q.computed = Math.floor(hits * rt.bs);
    this.hitTokens += q.computed;
  }

  private commitBlocks(r: Replica, q: Req, from: number, to: number, bs: number): void {
    const a = Math.floor(from / bs);
    const b = Math.min(q.blocks.length, Math.floor(to / bs));
    for (let i = a; i < b; i++) r.pool.commit(q.blocks[i]!, this.blockHash(q, i, bs));
  }

  // ───────────────────────────── scheduler ─────────────────────────────

  private preempt(r: Replica, q: Req): void {
    this.releaseKv(r, q);
    q.computed = 0;
    q.busy = false;
    q.state = WAITING;
    r.waiting.unshift(q);
    this.preemptions++;
  }

  /** One scheduler pass for a replica (Scheduler.schedule): running first, then waiting. */
  private schedule(r: Replica, rt: InstRuntime): Entry[] {
    const inst = rt.ev.inst;
    const k = inst.flags.speculative && inst.role !== 'prefill' ? inst.flags.speculative.k : 0;
    let budget = inst.maxNumBatchedTokens;
    const out: Entry[] = [];
    let preempted = false;

    outer: for (let i = 0; i < r.running.length && budget > 0; i++) {
      const q = r.running[i]!;
      if (q.busy) continue;
      const remaining = q.prompt + q.generated - q.computed;
      const decode = remaining <= 1 && q.generated > 0;
      let n = decode ? 1 + k : Math.min(remaining, budget);
      if (!decode && !inst.flags.enableChunkedPrefill && n < remaining) continue;
      n = Math.min(n, budget);
      while (!this.grow(r, q, q.computed + n, rt)) {
        let v = r.running.length - 1;
        while (v > i && r.running[v]!.busy) v--;
        const victim = r.running[v]!;
        r.running.splice(v, 1);
        this.preempt(r, victim);
        preempted = true;
        if (victim === q) break outer;
      }
      out.push({ req: q, n, decode });
      q.busy = true;
      budget -= n;
    }

    if (!preempted) {
      while (r.waiting.length && budget > 0 && r.running.length < inst.maxNumSeqs) {
        const q = r.waiting[0]!;
        if (q.remote) {
          // P/D decode side: the prompt's KV arrived over NIXL; admit and decode right away.
          if (!this.grow(r, q, q.prompt + 1 + k, rt)) break;
          q.computed = q.prompt;
          q.remote = false;
          r.waiting.shift();
          q.state = RUNNING;
          r.running.push(q);
          const n = Math.min(1 + k, budget);
          out.push({ req: q, n, decode: true });
          q.busy = true;
          budget -= n;
          continue;
        }
        const fresh = q.computed === 0 && !q.blocks.length;
        if (fresh) this.prefixHits(r, q, rt);
        const remaining = q.prompt + q.generated - q.computed;
        const n = Math.min(remaining, budget);
        if ((!inst.flags.enableChunkedPrefill && n < remaining) || !this.grow(r, q, q.computed + n, rt)) {
          // Like allocate_slots(): prefix hits are only claimed if the whole allocation succeeds.
          if (fresh) {
            this.hitTokens -= q.computed;
            this.releaseKv(r, q);
            q.computed = 0;
          }
          break;
        }
        if (fresh) this.promptTokens += q.prompt;
        r.waiting.shift();
        q.state = RUNNING;
        r.running.push(q);
        out.push({ req: q, n, decode: false });
        q.busy = true;
        budget -= n;
      }
    }
    return out;
  }

  private shapeOf(entries: Entry[], rt: InstRuntime): BatchShape {
    const s = emptyShape(this.model);
    const inst = rt.ev.inst;
    s.q = 1 + (inst.flags.speculative && inst.role !== 'prefill' ? inst.flags.speculative.k : 0);
    const wins: CtxWindow[] = this.model.windows;
    for (const e of entries) {
      const q = e.req;
      if (e.decode) {
        s.decodeSeqs++;
        s.decodeCtx[0]! += q.computed;
        for (let i = 0; i < wins.length; i++) {
          const w = wins[i]!;
          s.decodeCtx[i + 1]! += w.kind === 'chunked' ? (q.computed % w.size) + 1 : Math.min(q.computed, w.size);
        }
        s.sampled += s.q;
      } else {
        s.prefillTokens += e.n;
        s.prefillSeqs++;
        s.prefillPairs[0]! += chunkPairs(e.n, q.computed);
        for (let i = 0; i < wins.length; i++) s.prefillPairs[i + 1]! += chunkPairs(e.n, q.computed, wins[i]);
        if (q.computed + e.n >= q.prompt + q.generated) s.sampled++;
      }
    }
    return s;
  }

  private launch(u: Unit): void {
    u.launchPending = false;
    if (u.inflight >= u.maxInflight) return;
    const rt = this.insts[u.inst]!;
    const entries = u.replicas.map((r) => this.schedule(r, rt));
    if (entries.every((e) => e.length === 0)) return; // idle until the next arrival
    const shapes = entries.map((e) => this.shapeOf(e, rt));
    const global = shapes.reduce((s, x) => s + x.decodeSeqs * x.q + x.prefillTokens, 0);
    let worst: StepCost | null = null;
    shapes.forEach((sh, i) => {
      sh.globalTokens = u.replicas.length > 1 ? global : sh.decodeSeqs * sh.q + sh.prefillTokens;
      const c = rt.cost.step(sh).cost;
      const r = u.replicas[i]!;
      r.lastDecode = sh.decodeSeqs;
      r.lastPrefill = sh.prefillTokens;
      r.lastStep = c.time;
      if (!worst || c.time > worst.time) worst = c;
    });
    const c = worst as unknown as StepCost;
    this.sampleRouting(global);
    const interval = rt.ev.inst.pp > 1 ? Math.max(Math.max(...c.stages), c.host) : c.time;
    const latency = rt.ev.inst.pp > 1 ? Math.max(c.gpu, interval) : c.time;
    rt.bytes.tpIntra += c.comm.tp.intra;
    rt.bytes.epIntra += c.comm.ep.intra;
    rt.bytes.ppIntra += c.comm.pp.intra;
    rt.bytes.inter += c.comm.tp.inter + c.comm.ep.inter + c.comm.pp.inter;
    u.inflight++;
    u.nextLaunch = this.now + interval;
    this.heap.push(this.now + latency, EV_BATCH, { unit: u, entries } satisfies Batch);
    if (u.inflight < u.maxInflight) this.wake(u);
  }

  /** Route a sample of this step's tokens through the (skewed) router. */
  private sampleRouting(tokens: number): void {
    const cdf = this.expertCdf;
    const counts = this.expertCounts;
    if (!cdf || !counts || tokens <= 0) return;
    const n = Math.min(tokens, 96);
    const scale = tokens / n;
    const E = cdf.length;
    const picked: number[] = [];
    for (let t = 0; t < n; t++) {
      picked.length = 0;
      for (let tries = 0; picked.length < Math.min(this.topK, E) && tries < this.topK * 4; tries++) {
        const u = this.rng.next();
        let lo = 0;
        let hi = E - 1;
        while (lo < hi) {
          const mid = (lo + hi) >> 1;
          if (cdf[mid]! < u) lo = mid + 1;
          else hi = mid;
        }
        if (!picked.includes(lo)) picked.push(lo);
      }
      for (const e of picked) counts[e]! += scale;
    }
  }

  private acceptedDrafts(rt: InstRuntime): number {
    const spec = rt.ev.inst.flags.speculative;
    if (!spec) return 0;
    let n = 0;
    for (let i = 0; i < spec.k; i++) {
      if (this.rng.next() < spec.acceptance * spec.decay ** i) n++;
      else break;
    }
    return n;
  }

  private complete(b: Batch): void {
    const u = b.unit;
    const rt = this.insts[u.inst]!;
    const role = rt.ev.inst.role;
    u.inflight--;
    b.entries.forEach((list, ri) => {
      const r = u.replicas[ri]!;
      for (const e of list) {
        const q = e.req;
        q.busy = false;
        if (q.state !== RUNNING || q.rep !== r) continue; // preempted meanwhile
        const before = q.computed;
        if (e.decode) {
          const produced = Math.min(1 + (e.n > 1 ? Math.min(e.n - 1, this.acceptedDrafts(rt)) : 0), q.out - q.generated);
          q.computed += produced;
          q.generated += produced;
          this.tokOut += produced;
          if (q.lastTok >= 0) this.itl.push(this.now, (this.now - q.lastTok) / produced);
          q.lastTok = this.now;
        } else {
          q.computed += e.n;
          if (q.computed >= q.prompt + q.generated) {
            q.generated++;
            this.tokOut++;
            if (q.firstTok < 0) {
              q.firstTok = this.now;
              this.ttft.push(this.now, this.now - q.arrival);
            }
            q.lastTok = this.now;
          }
        }
        this.commitBlocks(r, q, before, q.computed, rt.bs);
        if (role === 'prefill' && q.generated >= 1) this.startTransfer(r, q);
        else if (q.generated >= q.out) this.finish(r, q);
      }
    });
    this.wake(u);
  }

  private removeRunning(r: Replica, q: Req): void {
    const i = r.running.indexOf(q);
    if (i >= 0) r.running.splice(i, 1);
  }

  private finish(r: Replica, q: Req): void {
    this.removeRunning(r, q);
    this.releaseKv(r, q);
    q.state = DONE;
    this.completed++;
    this.e2e.push(this.now, this.now - q.arrival);
    if (this.w.arrival === 'closed' && q.user < Math.round(this.w.concurrency)) {
      this.heap.push(this.now, EV_ARRIVAL, q.user);
    }
  }

  private startTransfer(r: Replica, q: Req): void {
    this.removeRunning(r, q);
    q.state = TRANSFER;
    const rt = this.insts[0]!;
    const bytes = this.kvBytesPerToken * q.prompt;
    this.activeXfers++;
    this.xfers.add(q);
    const bw = rt.bw.so * rt.ev.inst.world / rt.ev.inst.dp;
    const t = this.evaluation.calib.alphaKvx + (bytes * this.activeXfers) / Math.max(1, bw);
    rt.bytes.kvx += bytes / Math.max(1, rt.ev.inst.world / rt.ev.inst.dp);
    this.heap.push(this.now + t, EV_XFER, q);
    this.wake(r.unit);
  }

  private endTransfer(q: Req): void {
    this.activeXfers--;
    this.xfers.delete(q);
    this.releaseKv(q.rep!, q);
    if (q.generated >= q.out) {
      // Single-token outputs are complete once prefill produced the token.
      q.state = DONE;
      this.completed++;
      this.e2e.push(this.now, this.now - q.arrival);
      if (this.w.arrival === 'closed' && q.user < Math.round(this.w.concurrency)) this.heap.push(this.now, EV_ARRIVAL, q.user);
      return;
    }
    q.remote = true;
    q.computed = 0;
    this.enqueue(q, 1);
  }

  // ───────────────────────────── driver ─────────────────────────────

  /** Process events up to sim time `until`, yielding after `budgetMs` of wall time. */
  runUntil(until: number, budgetMs = 8): boolean {
    const t0 = performance.now();
    let n = 0;
    while (this.heap.peekTime() <= until) {
      const e = this.heap.pop();
      this.now = e.t;
      switch (e.kind) {
        case EV_ARRIVAL: {
          if (this.w.arrival === 'poisson') {
            this.enqueue(this.newRequest(-1), 0);
            this.heap.push(this.now + this.rng.exp(Math.max(1e-3, this.w.rate)), EV_ARRIVAL, -1);
          } else {
            const user = e.data as number;
            if (user < Math.round(this.w.concurrency)) this.enqueue(this.newRequest(user), 0);
          }
          break;
        }
        case EV_LAUNCH:
          this.launch(e.data as Unit);
          break;
        case EV_BATCH:
          this.complete(e.data as Batch);
          break;
        case EV_XFER:
          this.endTransfer(e.data as Req);
          break;
      }
      if (++n % 64 === 0 && performance.now() - t0 > budgetMs) return false;
    }
    this.now = Math.max(this.now, until);
    return true;
  }

  frame(wallRatio: number): SimFrame {
    const dt = Math.max(1e-6, this.now - this.lastFrameT);
    this.lastFrameT = this.now;
    const instRate = (this.tokOut - this.tokOutAtFrame) / dt;
    const reqRate = (this.completed - this.completedAtFrame) / dt;
    this.tokOutAtFrame = this.tokOut;
    this.completedAtFrame = this.completed;
    const a = Math.min(1, dt / 1.5);
    this.rateEma += a * (instRate - this.rateEma);
    this.reqRateEma += a * (reqRate - this.reqRateEma);

    const replicas: ReplicaFrame[] = this.replicas.map((r) => ({
      instance: r.inst,
      dp: r.dp,
      running: r.running.length,
      waiting: r.waiting.length,
      kvUsed: r.pool.used,
      kvTotal: r.pool.size,
      decodeTokens: r.lastDecode,
      prefillTokens: r.lastPrefill,
      stepTime: r.lastStep,
    }));
    const links: LinkRates[] = this.insts.map((rt) => {
      const b = rt.bytes;
      const out = {
        tpIntra: b.tpIntra / dt / rt.bw.su,
        epIntra: b.epIntra / dt / rt.bw.su,
        ppIntra: b.ppIntra / dt / rt.bw.su,
        inter: b.inter / dt / rt.bw.so,
        kvx: b.kvx / dt / rt.bw.so,
      };
      rt.bytes = { tpIntra: 0, epIntra: 0, ppIntra: 0, inter: 0, kvx: 0 };
      return out;
    });
    let kvMap: SimFrame['kvMap'] = null;
    if (this.focus !== null && this.replicas[this.focus]) {
      const r = this.replicas[this.focus]!;
      const p = r.pool;
      const m = new Uint16Array(p.size);
      for (let i = 0; i < p.size; i++) {
        const ref = p.ref[i]!;
        m[i] = ref === 0 ? (p.hash[i]! >= 0 ? 1 : 0) : ref > 1 ? 2 : 3 + (p.owner[i]! % COLORS);
      }
      kvMap = { replica: this.focus, blocks: m };
    }
    let running = 0;
    let waiting = 0;
    for (const r of this.replicas) {
      running += r.running.length;
      waiting += r.waiting.length;
    }
    return {
      t: this.now,
      wallRatio,
      replicas,
      links,
      stats: {
        outTokPerSec: this.rateEma,
        reqPerSec: this.reqRateEma,
        ttft: this.ttft.percentiles(this.now),
        itl: this.itl.percentiles(this.now),
        e2e: this.e2e.percentiles(this.now),
        completed: this.completed,
        preemptions: this.preemptions,
        prefixHitRate: this.promptTokens ? this.hitTokens / this.promptTokens : 0,
        running,
        waiting,
        transferring: this.activeXfers,
      },
      kvMap,
      expertLoad: this.expertLoadFrame(),
    };
  }

  private expertLoadFrame(): Float32Array | null {
    const c = this.expertCounts;
    if (!c) return null;
    let max = 0;
    for (let e = 0; e < c.length; e++) max = Math.max(max, c[e]!);
    const out = new Float32Array(c.length);
    if (max > 0) for (let e = 0; e < c.length; e++) out[e] = c[e]! / max;
    for (let e = 0; e < c.length; e++) c[e]! *= 0.5; // decay: recent load dominates
    return out;
  }

  /** Invariant check for tests: every allocated block is referenced by a live request. */
  auditBlocks(): { leaked: number } {
    let leaked = 0;
    for (const r of this.replicas) {
      const held = new Map<number, number>();
      const live = [...r.running, ...r.waiting, ...[...this.xfers].filter((q) => q.rep === r)];
      for (const q of live) for (const b of [...q.blocks, ...q.fixed]) held.set(b, (held.get(b) ?? 0) + 1);
      for (let b = 0; b < r.pool.size; b++) {
        const ref = r.pool.ref[b]!;
        const h = held.get(b) ?? 0;
        if (ref !== h) leaked += Math.abs(ref - h);
      }
    }
    return { leaked };
  }
}
