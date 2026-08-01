import { readdir, rm } from "node:fs/promises";

const workspaceRoot = new URL("../", import.meta.url);
const packageRoot = new URL("../packages/", import.meta.url);
const entries = await readdir(packageRoot, { withFileTypes: true });

await Promise.all([
  rm(new URL("../.cache/", import.meta.url), { force: true, recursive: true }),
  rm(new URL("../coverage/", import.meta.url), { force: true, recursive: true }),
  ...entries
    .filter((entry) => entry.isDirectory())
    .map((entry) =>
      rm(new URL(`../packages/${entry.name}/dist/`, import.meta.url), {
        force: true,
        recursive: true,
      })),
]);

// Keep TypeScript's root solution metadata clean if the compiler creates it in
// a future configuration.
await rm(new URL("tsconfig.tsbuildinfo", workspaceRoot), { force: true });
