import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: {
      "@synaptic/backend-cpu": new URL(
        "./packages/backend-cpu/src/index.ts",
        import.meta.url,
      ).pathname,
      "@synaptic/backend-paper": new URL(
        "./packages/backend-paper/src/index.ts",
        import.meta.url,
      ).pathname,
      "@synaptic/backend-wasm": new URL(
        "./packages/backend-wasm/src/index.ts",
        import.meta.url,
      ).pathname,
      "@synaptic/backend-webgpu": new URL(
        "./packages/backend-webgpu/src/index.ts",
        import.meta.url,
      ).pathname,
      "@synaptic/compat-v1": new URL(
        "./packages/compat-v1/src/index.ts",
        import.meta.url,
      ).pathname,
      "@synaptic/core": new URL("./packages/core/src/index.ts", import.meta.url).pathname,
      "@synaptic/layers": new URL("./packages/layers/src/index.ts", import.meta.url).pathname,
    },
  },
  test: {
    coverage: {
      include: ["packages/*/src/**/*.ts"],
      reporter: ["text", "html", "lcov"],
      thresholds: {
        branches: 65,
        functions: 80,
        lines: 80,
        statements: 80,
      },
    },
    include: ["packages/**/*.test.ts"],
  },
});
