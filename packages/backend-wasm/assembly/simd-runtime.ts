export * from "./runtime";

// Force the optimized artifact to require WebAssembly SIMD.
export function simdProbe(): i32 {
  return i32x4.extract_lane(i32x4.splat(1), 0);
}
