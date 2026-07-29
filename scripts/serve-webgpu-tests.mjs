import { spawn } from "node:child_process";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer } from "node:http";
import { extname, isAbsolute, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const tests = [
  {
    description: "Train a 2→3→1 network and watch all four truth-table predictions converge.",
    kind: "Interactive demo",
    name: "Learn XOR",
    path: "/packages/backend-webgpu/test/xor.html",
  },
  {
    description: "Train on MNIST, draw a digit, and inspect the network's ten output activations.",
    kind: "Interactive demo",
    name: "Draw a digit",
    path: "/packages/backend-webgpu/test/mnist.html",
  },
  {
    description: "Learn a tied 3×3 MNIST convolution, transfer it exactly, and run it as a Neural CA.",
    kind: "Interactive demo",
    name: "MNIST to Neural CA",
    path: "/packages/backend-webgpu/test/mnist-automata.html",
  },
  {
    description: "Train a four-cell LSTM, then animate random length-10 recall challenges.",
    kind: "Interactive demo",
    name: "Sequence recall",
    path: "/packages/backend-webgpu/test/dsr.html",
  },
  {
    description: "Continuously train a coordinate-to-RGB network to reconstruct a portrait.",
    kind: "Interactive demo",
    name: "Learn to paint",
    path: "/packages/backend-webgpu/test/learn-to-paint.html",
  },
  {
    description: "Compile the WGSL module and report browser validation messages.",
    kind: "Hardware check",
    name: "Shader compilation",
    path: "/packages/backend-webgpu/test/shader-smoke.html",
  },
  {
    description: "Run a minimal recurrent train, forward, snapshot, and checkpoint lifecycle.",
    kind: "Hardware check",
    name: "Forward and state smoke test",
    path: "/packages/backend-webgpu/test/forward-smoke.html",
  },
  {
    description: "Compare recurrent training, restoration, traces, and shared parameters with CPU.",
    kind: "Hardware check",
    name: "Training parity",
    path: "/packages/backend-webgpu/test/training-parity.html",
  },
  {
    description: "Learn XOR, MNIST, and DSR on hardware and report accuracy and elapsed time.",
    kind: "Automated regression",
    name: "Learning workloads",
    path: "/packages/backend-webgpu/test/learning-workloads.html",
  },
];

const contentTypes = new Map([
  [".css", "text/css; charset=utf-8"],
  [".html", "text/html; charset=utf-8"],
  [".jpg", "image/jpeg"],
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
    <article class="panel">
      <span>${test.kind}</span>
      <h2><a href="${test.path}" target="_blank">${test.name}</a></h2>
      <p>${test.description}</p>
      <code>${test.path}</code>
    </article>
  `).join("");
  return `<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Synaptic WebGPU tests</title>
<link rel="stylesheet" href="/packages/backend-webgpu/test/demo.css">
<style>
  main { max-width: 74rem; }
  h1 {
    max-width: 11ch;
    margin: 0;
    font-size: clamp(3rem, 8vw, 6rem);
    letter-spacing: -.07em;
    line-height: .88;
  }
  .intro { max-width: 42rem; margin: 1.5rem 0 3rem; color: #9cabc0; line-height: 1.65; }
  .cards {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
  article { padding: 1.35rem; }
  article span {
    color: #ff809f;
    font: 650 .6rem ui-monospace, monospace;
    letter-spacing: .09em;
    text-transform: uppercase;
  }
  h2 { margin: .55rem 0; font-size: 1.2rem; }
  h2 a { color: #edf4ff; text-decoration-color: #526983; text-underline-offset: .22em; }
  article p { min-height: 3.2em; margin: .65rem 0 1rem; color: #8999ae; line-height: 1.55; }
  code { color: #64758b; font-size: .68rem; overflow-wrap: anywhere; }
  @media (max-width: 680px) { .cards { grid-template-columns: 1fr; } }
</style>
<main>
  <h1>WebGPU, in motion.</h1>
  <p class="intro">
    Interactive learning demos and no-fallback hardware checks for Synaptic.
    Every page below compiles directly to WebGPU and must report
    <code>webgpu</code> as its backend.
  </p>
  <section class="cards">${links}</section>
</main>`;
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
    && !localPath.startsWith("node_modules/@cazala/automata/")
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
