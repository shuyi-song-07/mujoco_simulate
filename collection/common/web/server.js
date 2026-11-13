import { createReadStream, existsSync, statSync, readFileSync, readdirSync } from "node:fs";
import { createHash } from "node:crypto";
import { createServer, request as createProxyRequest } from "node:http";
import { dirname, extname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const port = Number(process.env.PORT) || 8014;
const host = "127.0.0.1";
const root = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(root, "../../..");
const dependencies = join(projectRoot, "collection/teleoperation/mediapipe");
const publicAssets = new Map();
function registerWebDirectory(directory) {
  if (!existsSync(directory)) return;
  for (const item of readdirSync(directory, {withFileTypes:true})) {
    if (!item.isFile() || ![".html", ".css", ".js"].includes(extname(item.name)) || item.name.endsWith(".test.js") || item.name === "server.js") continue;
    // Only the explicitly public web directories are exposed, never task Python code.
    const key = "/" + item.name;
    if (!publicAssets.has(key)) publicAssets.set(key, join(directory, item.name));
  }
}
registerWebDirectory(root);
const selectedTask = process.env.PAPERA_COLLECTION_TASK;
if (selectedTask && !/^[a-z][a-z0-9_]*$/.test(selectedTask)) throw Error("Invalid collection task ID");
const taskDirectories = selectedTask ? [selectedTask] : readdirSync(join(projectRoot, "collection"), {withFileTypes:true}).filter(e=>e.isDirectory() && e.name!=="common").map(e=>e.name);
for (const task of taskDirectories) registerWebDirectory(join(projectRoot, "collection", task, "web"));

const backendPort = Number(process.env.DUAL_BACKEND_PORT) || 5014;
const serverSha256 = createHash("sha256").update(readFileSync(fileURLToPath(import.meta.url))).digest("hex");
const mimeTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".task": "application/octet-stream",
  ".wasm": "application/wasm",
};

createServer((request, response) => {
  response.setHeader("X-Project-Server", "paperA-task3");
  response.setHeader("X-Project-Root", encodeURIComponent(projectRoot));
  response.setHeader("X-Project-Backend-Port", String(backendPort));
  response.setHeader("X-Project-Server-Sha256", serverSha256);
  let pathname;
  try {
    pathname = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
  } catch {
    response.writeHead(400).end("Bad request");
    return;
  }
  if (pathname.includes("\\") || pathname.includes("\0") || pathname.split("/").includes("..")) {
    response.writeHead(400).end("Invalid path");
    return;
  }
  // Local browser requests must come from this page.
  if (![`${host}:${port}`, `localhost:${port}`].includes(request.headers.host)) {
    response.writeHead(403).end("Please open the local camera page.");
    return;
  }
  if (request.headers.origin && request.headers.origin !== `http://${request.headers.host}`) {
    response.writeHead(403).end("Cross-origin request denied");
    return;
  }

  const dualRoutes = new Set(["/api/dual/measurements", "/api/dual/health", "/api/dual/control", "/api/dual/top-preview", "/api/dual/oblique-preview", "/api/dual/front-preview"]);
  const isDualRoute = dualRoutes.has(pathname);
  if (isDualRoute) {
    const proxyRequest = createProxyRequest(
      {
        hostname: "127.0.0.1",
        port: backendPort,
        path: pathname.replace("/api/dual", ""),
        method: request.method,
        headers: {
          "content-type": request.headers["content-type"] ?? "application/json",
        },
      },
      (proxyResponse) => {
        response.writeHead(proxyResponse.statusCode ?? 502, {
          "Content-Type": proxyResponse.headers["content-type"] ?? "application/json",
          "Cache-Control": "no-store",
          ...(proxyResponse.headers["x-preview-age-ms"] ? {"X-Preview-Age-Ms":proxyResponse.headers["x-preview-age-ms"]} : {}),
          ...(proxyResponse.headers["x-preview-epoch"] ? {"X-Preview-Epoch":proxyResponse.headers["x-preview-epoch"]} : {}),
        });
        proxyResponse.pipe(response);
      },
    );
    proxyRequest.on("error", () => {
      if (!response.headersSent) {
        response.writeHead(502, { "Content-Type": "application/json" });
        response.end(JSON.stringify({ ok: false, error: "Task3 recorder is not running" }));
      } else response.destroy();
    });
    proxyRequest.setTimeout(10000, () => proxyRequest.destroy());
    request.pipe(proxyRequest);
    return;
  }

  const page = pathname === "/" || pathname === "/index.html" ? "/dual_arm.html" : pathname;
  const dependency = pathname.startsWith("/node_modules/@mediapipe/tasks-vision/") || pathname.startsWith("/models/");
  if (!publicAssets.has(page) && !dependency) {
    response.writeHead(404).end("Not found");
    return;
  }
  let filePath = publicAssets.get(page) || resolve(dependencies, `.${pathname}`);

  if (existsSync(filePath) && statSync(filePath).isDirectory()) {
    filePath = join(filePath, "index.html");
  }

  if (!filePath.startsWith(join(projectRoot, "collection") + sep) || !existsSync(filePath) || !statSync(filePath).isFile()) {
    response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    response.end("404 Not Found");
    return;
  }

  response.writeHead(200, {
    "Cache-Control": "no-cache",
    "Content-Type": mimeTypes[extname(filePath)] ?? "application/octet-stream",
  });
  createReadStream(filePath).pipe(response);
}).listen(port, host, () => {
  console.log(`Gesture Interaction is running at http://${host}:${port}`);
});
