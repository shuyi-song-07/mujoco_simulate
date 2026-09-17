import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer, request as createProxyRequest } from "node:http";
import { extname, join, normalize } from "node:path";

const port = Number(process.env.PORT) || 8000;
const root = process.cwd();
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
  const pathname = decodeURIComponent(new URL(request.url, "http://localhost").pathname);

  if (
    pathname === "/api/health" ||
    pathname === "/api/control" ||
    pathname === "/api/side-preview" ||
    pathname === "/api/front-preview"
  ) {
    const proxyRequest = createProxyRequest(
      {
        hostname: "127.0.0.1",
        port: 5001,
        path: pathname.replace("/api", ""),
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
      response.writeHead(502, { "Content-Type": "application/json" });
      response.end(JSON.stringify({ ok: false, error: "MuJoCo recorder is not running" }));
    });
    request.pipe(proxyRequest);
    return;
  }

  const relativePath = normalize(pathname).replace(/^(\.\.(\/|\\|$))+/, "");
  let filePath = join(root, relativePath === "/" ? "index.html" : relativePath);

  if (existsSync(filePath) && statSync(filePath).isDirectory()) {
    filePath = join(filePath, "index.html");
  }

  if (!filePath.startsWith(root) || !existsSync(filePath) || !statSync(filePath).isFile()) {
    response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    response.end("404 Not Found");
    return;
  }

  response.writeHead(200, {
    "Cache-Control": "no-cache",
    "Content-Type": mimeTypes[extname(filePath)] ?? "application/octet-stream",
  });
  createReadStream(filePath).pipe(response);
}).listen(port, "127.0.0.1", () => {
  console.log(`Gesture Interaction is running at http://localhost:${port}`);
});
