import { spawn } from "node:child_process";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer } from "node:http";
import { extname, isAbsolute, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const tests = [
  {
    description: "Compile the WGSL module and report browser validation messages.",
    name: "Shader compilation",
    path: "/packages/backend-webgpu/test/shader-smoke.html",
  },
  {
    description: "Run a minimal recurrent train, forward, snapshot, and checkpoint lifecycle.",
    name: "Forward and state smoke test",
    path: "/packages/backend-webgpu/test/forward-smoke.html",
  },
  {
    description: "Compare recurrent training, restoration, traces, and shared parameters with CPU.",
    name: "Training parity",
    path: "/packages/backend-webgpu/test/training-parity.html",
  },
  {
    description: "Learn XOR, MNIST, and DSR on hardware and report accuracy and elapsed time.",
    name: "Learning workloads",
    path: "/packages/backend-webgpu/test/learning-workloads.html",
  },
];

const contentTypes = new Map([
  [".css", "text/css; charset=utf-8"],
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".map", "application/json; charset=utf-8"],
  [".wasm", "application/wasm"],
]);

const { values } = parseArgs({
  options: {
    host: { default: "127.0.0.1", type: "string" },
    port: { default: "4173", type: "string" },
  },
});
const host = values.host;
const port = Number(values.port);
if (!Number.isInteger(port) || port < 1 || port > 65_535) {
  throw new RangeError(`Invalid port: ${values.port}`);
}

const workspaceRoot = fileURLToPath(new URL("../", import.meta.url));

function runBuild() {
  const executable = process.platform === "win32" ? "npm.cmd" : "npm";
  return new Promise((resolveBuild, rejectBuild) => {
    const child = spawn(executable, ["run", "build"], {
      cwd: workspaceRoot,
      stdio: "inherit",
    });
    child.once("error", rejectBuild);
    child.once("exit", (code, signal) => {
      if (code === 0) {
        resolveBuild();
        return;
      }
      rejectBuild(
        new Error(
          signal === null
            ? `WebGPU test build exited with code ${code}`
            : `WebGPU test build exited after signal ${signal}`,
        ),
      );
    });
  });
}

function indexPage() {
  const links = tests.map((test) => `
    <article>
      <h2><a href="${test.path}" target="_blank">${test.name}</a></h2>
      <p>${test.description}</p>
      <code>${test.path}</code>
    </article>
  `).join("");
  return `<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Synaptic WebGPU tests</title>
<style>
  :root { color-scheme: light dark; font-family: system-ui, sans-serif; }
  body { margin: 3rem auto; max-width: 54rem; padding: 0 1.5rem; }
  article { border: 1px solid color-mix(in srgb, currentColor 20%, transparent);
    border-radius: .75rem; margin: 1rem 0; padding: 1rem 1.25rem; }
  h1, h2 { line-height: 1.2; }
  h2 { font-size: 1.1rem; margin: 0 0 .5rem; }
  p { margin: .5rem 0; }
  code { font-size: .8rem; overflow-wrap: anywhere; }
</style>
<h1>Synaptic WebGPU hardware tests</h1>
<p>These pages disable fallback and must report <code>webgpu</code> as the backend.
Open them in a browser with WebGPU enabled. Results appear as JSON in each page.</p>
${links}`;
}

function send(response, status, body, contentType = "text/plain; charset=utf-8") {
  response.writeHead(status, {
    "Cache-Control": "no-store",
    "Content-Length": Buffer.byteLength(body),
    "Content-Type": contentType,
  });
  response.end(body);
}

function safeFile(pathname) {
  const localPath = decodeURIComponent(pathname).replace(/^\/+/, "");
  if (
    !localPath.startsWith("packages/")
    && !localPath.startsWith("node_modules/mnist/")
  ) {
    return undefined;
  }
  const candidate = resolve(workspaceRoot, localPath);
  const fromRoot = relative(workspaceRoot, candidate);
  if (fromRoot.startsWith("..") || isAbsolute(fromRoot)) {
    return undefined;
  }
  return candidate;
}

await runBuild();

const server = createServer(async (request, response) => {
  if (request.method !== "GET" && request.method !== "HEAD") {
    send(response, 405, "Method not allowed");
    return;
  }
  try {
    const url = new URL(request.url ?? "/", `http://${host}:${port}`);
    if (url.pathname === "/") {
      const body = indexPage();
      if (request.method === "HEAD") {
        response.writeHead(200, {
          "Cache-Control": "no-store",
          "Content-Length": Buffer.byteLength(body),
          "Content-Type": "text/html; charset=utf-8",
        });
        response.end();
      } else {
        send(response, 200, body, "text/html; charset=utf-8");
      }
      return;
    }
    const filename = safeFile(url.pathname);
    if (filename === undefined) {
      send(response, 404, "Not found");
      return;
    }
    const metadata = await stat(filename);
    if (!metadata.isFile()) {
      send(response, 404, "Not found");
      return;
    }
    response.writeHead(200, {
      "Cache-Control": "no-store",
      "Content-Length": metadata.size,
      "Content-Type":
        contentTypes.get(extname(filename)) ?? "application/octet-stream",
    });
    if (request.method === "HEAD") {
      response.end();
      return;
    }
    createReadStream(filename).pipe(response);
  } catch (error) {
    const code =
      typeof error === "object" && error !== null && "code" in error
        ? error.code
        : undefined;
    send(response, code === "ENOENT" ? 404 : 400, code === "ENOENT" ? "Not found" : "Bad request");
  }
});

await new Promise((resolveListen, rejectListen) => {
  server.once("error", rejectListen);
  server.listen(port, host, resolveListen);
});

const origin = `http://${host}:${port}`;
console.log(`\nSynaptic WebGPU tests: ${origin}/`);
for (const test of tests) {
  console.log(`- ${test.name}: ${origin}${test.path}`);
}
console.log("\nPress Ctrl+C to stop.");

process.once("SIGINT", () => {
  server.close(() => process.exit(0));
});
