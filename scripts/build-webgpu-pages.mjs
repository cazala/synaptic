import {
  access,
  cp,
  mkdir,
  readdir,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { dirname, extname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { indexPage } from "./webgpu-test-site.mjs";

const workspaceRoot = fileURLToPath(new URL("../", import.meta.url));
const outputRoot = resolve(workspaceRoot, ".cloudflare/pages");
const runtimePackages = [
  "backend-cpu",
  "backend-paper",
  "backend-wasm",
  "backend-webgpu",
  "compat-v1",
  "conformance",
  "core",
  "layers",
  "synaptic",
];

async function copyPath(source, destination) {
  await mkdir(dirname(destination), { recursive: true });
  await cp(resolve(workspaceRoot, source), resolve(outputRoot, destination), {
    recursive: true,
  });
}

await rm(outputRoot, { force: true, recursive: true });
await mkdir(outputRoot, { recursive: true });

for (const packageName of runtimePackages) {
  await copyPath(
    `packages/${packageName}/dist`,
    `packages/${packageName}/dist`,
  );
}

await copyPath("packages/backend-wasm/wasm", "packages/backend-wasm/wasm");
await copyPath(
  "packages/backend-webgpu/test",
  "packages/backend-webgpu/test",
);
await copyPath(
  "node_modules/@cazala/automata/dist/index.js",
  "vendor/automata/index.js",
);
await copyPath(
  "node_modules/@cazala/automata/dist/index.js.map",
  "vendor/automata/index.js.map",
);
await copyPath(
  "node_modules/mnist/dist/mnist.js",
  "vendor/mnist/mnist.js",
);

const testRoot = resolve(
  outputRoot,
  "packages/backend-webgpu/test",
);
for (const entry of await readdir(testRoot, { withFileTypes: true })) {
  if (!entry.isFile() || extname(entry.name) !== ".html") continue;
  const filename = resolve(testRoot, entry.name);
  const html = await readFile(filename, "utf8");
  await writeFile(
    filename,
    html
      .replaceAll(
        "/node_modules/@cazala/automata/dist/index.js",
        "/vendor/automata/index.js",
      )
      .replaceAll(
        "/node_modules/mnist/dist/mnist.js",
        "/vendor/mnist/mnist.js",
      ),
    "utf8",
  );
}
await writeFile(resolve(outputRoot, "index.html"), indexPage(), "utf8");

const requiredFiles = [
  "index.html",
  "packages/backend-webgpu/test/growing-neural-ca.html",
  "packages/backend-webgpu/dist/index.js",
  "packages/backend-wasm/wasm/runtime.wasm",
  "vendor/automata/index.js",
  "vendor/mnist/mnist.js",
];
await Promise.all(
  requiredFiles.map((filename) => access(resolve(outputRoot, filename))),
);

console.log(`Cloudflare Pages bundle: ${outputRoot}`);
