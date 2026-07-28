import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    coverage: {
      include: ["packages/*/src/**/*.ts"],
      reporter: ["text", "html", "lcov"],
    },
    include: ["packages/**/*.test.ts"],
  },
});
