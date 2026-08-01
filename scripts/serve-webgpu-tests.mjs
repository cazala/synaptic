import { spawn } from "node:child_process";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer } from "node:http";
import { extname, isAbsolute, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { indexPage, tests } from "./webgpu-test-site.mjs";

const contentTypes = new Map([
  [".css", "text/css; charset=utf-8"],
  [".html", "text/html; charset=utf-8"],
  [".jpg", "image/jpeg"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".map", "application/json; charset=utf-8"],
  [".svg", "image/svg+xml; charset=utf-8"],
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
