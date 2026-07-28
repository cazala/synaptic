import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const packageRoot = fileURLToPath(new URL("../", import.meta.url));
const workspaceRoot = fileURLToPath(new URL("../../../", import.meta.url));
const compiler = fileURLToPath(
  new URL("../../../node_modules/assemblyscript/bin/asc.js", import.meta.url),
);
const config = `${packageRoot}asconfig.json`;

for (const [entry, target] of [
  ["assembly/runtime.ts", "release"],
  ["assembly/simd-runtime.ts", "simd"],
]) {
  const bindingArguments = target === "release" ? ["--bindings", "raw"] : [];
  const result = spawnSync(
    process.execPath,
    [
      compiler,
      `${packageRoot}${entry}`,
      "--config",
      config,
      "--target",
      target,
      ...bindingArguments,
    ],
    { cwd: workspaceRoot, encoding: "utf8", stdio: "inherit" },
  );
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}
