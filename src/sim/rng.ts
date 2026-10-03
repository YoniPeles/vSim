/** sfc32: small, fast, seedable PRNG so a run can be replayed exactly. */
export class Rng {
  private a: number;
  private b: number;
  private c: number;
  private d: number;
  constructor(seed: number) {
    this.a = 0x9e3779b9;
    this.b = 0x243f6a88;
    this.c = 0xb7e15162;
    this.d = seed >>> 0;
    for (let i = 0; i < 15; i++) this.next();
  }
  next(): number {
    this.a >>>= 0;
    this.b >>>= 0;
    this.c >>>= 0;
    this.d >>>= 0;
    let t = (this.a + this.b) | 0;
    this.a = this.b ^ (this.b >>> 9);
    this.b = (this.c + (this.c << 3)) | 0;
    this.c = (this.c << 21) | (this.c >>> 11);
    this.d = (this.d + 1) | 0;
    t = (t + this.d) | 0;
    this.c = (this.c + t) | 0;
    return (t >>> 0) / 4294967296;
  }
  /** Log-normal with the given mean and coefficient of variation. */
  lognormal(mean: number, cv: number): number {
    if (cv <= 0) return mean;
    const s2 = Math.log(1 + cv * cv);
    const mu = Math.log(mean) - s2 / 2;
    const u1 = Math.max(1e-12, this.next());
    const u2 = this.next();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return Math.exp(mu + Math.sqrt(s2) * z);
  }
  exp(rate: number): number {
    return -Math.log(Math.max(1e-12, this.next())) / rate;
  }
  int(n: number): number {
    return Math.floor(this.next() * n);
  }
}
