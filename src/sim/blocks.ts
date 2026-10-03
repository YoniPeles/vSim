// Paged KV-cache block pool with prefix caching, after vllm/v1/core/kv_cache_manager.py and
// block_pool.py: a doubly linked free queue (LRU), ref counts, and a hash → block map for full
// blocks. Freed blocks keep their hash until reallocated, so a request preempted and recomputed
// (or one sharing a system prompt) can hit them again.

export const NO_HASH = -1;

export class BlockPool {
  readonly size: number;
  readonly ref: Int32Array;
  readonly hash: Float64Array;
  /** Owner request slot of each block (last allocator), −1 if never used. */
  readonly owner: Int32Array;
  private readonly prev: Int32Array;
  private readonly next: Int32Array;
  private head = -1;
  private tail = -1;
  private freeCount = 0;
  private readonly cache = new Map<number, number>();
  /** Blocks whose cached content was evicted (for stats). */
  evictions = 0;

  constructor(size: number) {
    this.size = size;
    this.ref = new Int32Array(size);
    this.hash = new Float64Array(size).fill(NO_HASH);
    this.owner = new Int32Array(size).fill(-1);
    this.prev = new Int32Array(size);
    this.next = new Int32Array(size);
    for (let i = 0; i < size; i++) this.pushTail(i);
  }

  get free(): number {
    return this.freeCount;
  }

  get used(): number {
    return this.size - this.freeCount;
  }

  private pushTail(b: number): void {
    this.prev[b] = this.tail;
    this.next[b] = -1;
    if (this.tail >= 0) this.next[this.tail] = b;
    else this.head = b;
    this.tail = b;
    this.freeCount++;
  }

  private unlink(b: number): void {
    const p = this.prev[b]!;
    const n = this.next[b]!;
    if (p >= 0) this.next[p] = n;
    else this.head = n;
    if (n >= 0) this.prev[n] = p;
    else this.tail = p;
    this.prev[b] = this.next[b] = -1;
    this.freeCount--;
  }

  /** Cached block for a content hash (does not take a reference). */
  lookup(h: number): number {
    return this.cache.get(h) ?? -1;
  }

  /** Take a reference on a cached block (prefix hit). */
  touch(b: number, owner: number): void {
    if (this.ref[b] === 0) this.unlink(b);
    this.ref[b]!++;
    if (this.ref[b] === 1) this.owner[b] = owner;
  }

  /** Pop `n` fresh blocks from the LRU head, evicting their cached content. */
  allocate(n: number, owner: number, out: number[]): boolean {
    if (n > this.freeCount) return false;
    for (let i = 0; i < n; i++) {
      const b = this.head;
      this.unlink(b);
      const h = this.hash[b]!;
      if (h !== NO_HASH) {
        if (this.cache.get(h) === b) this.cache.delete(h);
        this.hash[b] = NO_HASH;
        this.evictions++;
      }
      this.ref[b] = 1;
      this.owner[b] = owner;
      out.push(b);
    }
    return true;
  }

  /** Register a full block's content hash so later requests can reuse it. */
  commit(b: number, h: number): void {
    if (this.hash[b] !== NO_HASH) return;
    if (this.cache.has(h)) return; // duplicate content: keep the first copy
    this.hash[b] = h;
    this.cache.set(h, b);
  }

  /** Drop a request's references; blocks are returned tail-first so they are evicted first. */
  release(blocks: number[]): void {
    for (let i = blocks.length - 1; i >= 0; i--) {
      const b = blocks[i]!;
      if (--this.ref[b]! === 0) this.pushTail(b);
    }
    blocks.length = 0;
  }

  /** Cached-but-free blocks (reusable on a prefix hit). */
  get cachedFree(): number {
    let n = 0;
    for (let b = this.head; b >= 0; b = this.next[b]!) if (this.hash[b] !== NO_HASH) n++;
    return n;
  }
}
