import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer, request as createProxyRequest } from "node:http";
import { dirname, extname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const port = Number(process.env.PORT) || 8000;
const host = "127.0.0.1";
const root = dirname(fileURLToPath(import.meta.url));
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
  response.setHeader("X-Project-Server", "mujoco-local");
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

  const dualRoutes = new Set(["/api/dual/health", "/api/dual/control", "/api/dual/top-preview", "/api/dual/side-preview", "/api/dual/front-preview"]);
  const isDualRoute = dualRoutes.has(pathname);
  if (isDualRoute ||
    pathname === "/api/health" ||
    pathname === "/api/control" ||
    pathname === "/api/side-preview" ||
    pathname === "/api/front-preview"
  ) {
    const proxyRequest = createProxyRequest(
      {
        hostname: "127.0.0.1",
        port: isDualRoute ? (Number(process.env.DUAL_BACKEND_PORT) || 5002) : (Number(process.env.BACKEND_PORT) || 5001),
        path: pathname.replace(isDualRoute ? "/api/dual" : "/api", ""),
        method: request.method,
        headers: {
          "content-type": request.headers["content-type"] ?? "application/json",
        },
      },
      (proxyResponse) => {
        response.writeHead(proxyResponse.statusCode ?? 502, {
          "Content-Type": proxyResponse.headers["content-type"] ?? "application/json",
          "Cache-Control": "no-store",
        });
        proxyResponse.pipe(response);
      },
    );
    proxyRequest.on("error", () => {
      if (!response.headersSent) {
        response.writeHead(502, { "Content-Type": "application/json" });
        response.end(JSON.stringify({ ok: false, error: isDualRoute ? "Task3 recorder is not running" : "MuJoCo recorder is not running" }));
      } else response.destroy();
    });
    proxyRequest.setTimeout(10000, () => proxyRequest.destroy());
    request.pipe(proxyRequest);
    return;
  }

  const publicFiles = new Set(["/", "/index.html", "/app.js", "/style.css", "/interaction-state.js",
    "/dual_arm.html", "/dual-arm-app.js", "/dual-arm-style.css", "/dual-interaction-state.js", "/dual-gesture-worker.js"]);
  if (!publicFiles.has(pathname) && !pathname.startsWith("/node_modules/@mediapipe/tasks-vision/") && !pathname.startsWith("/models/")) {
    response.writeHead(404).end("Not found");
    return;
  }
  let filePath = resolve(root, pathname === "/" ? "index.html" : `.${pathname}`);

  if (existsSync(filePath) && statSync(filePath).isDirectory()) {
    filePath = join(filePath, "index.html");
  }

  if (!filePath.startsWith(root + sep) || !existsSync(filePath) || !statSync(filePath).isFile()) {
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
