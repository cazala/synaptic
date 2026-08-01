const UINT32_RANGE = 0x1_0000_0000;

export function randomUint32(seed: number, counter: number): number {
  let value = (seed + Math.imul(counter, 0x9e37_79b9)) >>> 0;
  value ^= value >>> 16;
  value = Math.imul(value, 0x7feb_352d);
  value ^= value >>> 15;
  value = Math.imul(value, 0x846c_a68b);
  value ^= value >>> 16;
  return value >>> 0;
}

export function randomFloat(seed: number, counter: number): number {
  return randomUint32(seed, counter) / UINT32_RANGE;
}

export class CounterPrng {
  #counter: number;

  constructor(
    readonly seed: number,
    counter = 0,
  ) {
    this.#counter = counter >>> 0;
  }

  get counter(): number {
    return this.#counter;
  }

  nextUint32(): number {
    const value = randomUint32(this.seed, this.#counter);
    this.#counter = (this.#counter + 1) >>> 0;
    return value;
  }

  nextFloat(): number {
    return this.nextUint32() / UINT32_RANGE;
  }
}
