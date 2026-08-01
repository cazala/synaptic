import { describe, expect, it } from "vitest";
import { CounterPrng, randomUint32 } from "./prng.js";

describe("counter PRNG", () => {
  it("is deterministic and independently addressable", () => {
    const random = new CounterPrng(123);
    expect(random.nextUint32()).toBe(randomUint32(123, 0));
    expect(random.nextUint32()).toBe(randomUint32(123, 1));
    expect(random.counter).toBe(2);
  });
});
