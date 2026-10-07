// Test the local camera page against a fake recorder; no datasets are written.
const assert = require('node:assert/strict');
const http = require('node:http');
const { spawn } = require('node:child_process');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
let child;
let received;
const jpeg = Buffer.from([0xff, 0xd8, 0xff, 0xd9]);
const backend = http.createServer((req, res) => {
  let body = '';
  req.on('data', chunk => body += chunk);
  req.on('end', () => {
    received = { path: req.url, body };
    if (req.url.endsWith('-preview')) {
      res.writeHead(200, { 'Content-Type': 'image/jpeg' });
      res.end(jpeg);
    } else {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: true, task_mode: 'pick_place', test: true }));
    }
  });
});
async function listen(server) {
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  return server.address().port;
}
(async () => {
  const backendPort = await listen(backend);
  const probe = http.createServer();
  const port = await listen(probe);
  await new Promise(resolve => probe.close(resolve));
  child = spawn(process.execPath, ['teleoperation/mediapipe/server.js'], {
    cwd: root, windowsHide: true, stdio: 'inherit',
    env: { ...process.env, PORT: String(port), BACKEND_PORT: String(backendPort), DUAL_BACKEND_PORT: String(backendPort) },
  });
  const base = `http://127.0.0.1:${port}`;
  let ready = false;
  for (let i = 0; i < 40; i++) {
    if (child.exitCode !== null) throw new Error('Web service exited during startup');
    try { ready = (await fetch(base)).ok; } catch {}
    if (ready) break;
    await delay(100);
  }
  assert.ok(ready, 'Web service failed to start');
  assert.equal((await fetch(base)).headers.get('x-project-server'), 'mujoco-local');
  for (const asset of ['/app.js', '/interaction-state.js', '/dual_arm.html', '/dual-arm-app.js', '/dual-interaction-state.js', '/dual-arm-style.css', '/dual-gesture-worker.js', '/node_modules/@mediapipe/tasks-vision/vision_bundle.mjs', '/models/gesture_recognizer.task']) {
    assert.equal((await fetch(base + asset)).status, 200, asset);
  }
  assert.deepEqual(await (await fetch(base + '/api/health')).json(), { ok: true, task_mode: 'pick_place', test: true });
  for (const route of ['/api/side-preview', '/api/front-preview']) {
    const response = await fetch(base + route);
    assert.equal(response.status, 200);
    assert.equal(response.headers.get('content-type'), 'image/jpeg');
    assert.deepEqual(Buffer.from(await response.arrayBuffer()), jpeg);
  }
  await fetch(base + '/api/control', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{"command":"record_start"}',
  });
  assert.deepEqual(received, { path: '/control', body: '{"command":"record_start"}' });
  assert.deepEqual(await (await fetch(base + '/api/dual/health')).json(), { ok: true, task_mode: 'pick_place', test: true });
  for (const camera of ['top', 'front', 'side']) {
    const response = await fetch(base + `/api/dual/${camera}-preview`);
    assert.equal(response.status, 200);
    assert.deepEqual(Buffer.from(await response.arrayBuffer()), jpeg);
  }
  await fetch(base + '/api/dual/control', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{"command":"dual_gripper","action":"close"}',
  });
  assert.deepEqual(received, { path: '/control', body: '{"command":"dual_gripper","action":"close"}' });
  assert.equal((await fetch(base, { headers: { origin: 'http://unrelated.example' } })).status, 403);
  const invalidHostStatus = await new Promise((resolve, reject) => {
    http.get(base, { headers: { host: `unrelated.example:${port}` } }, response => {
      response.resume();
      resolve(response.statusCode);
    }).on('error', reject);
  });
  assert.equal(invalidHostStatus, 403);
  assert.equal((await fetch(base + '/server.js')).status, 404);
  assert.equal((await fetch(base + '/models/%5c..%5cserver.js')).status, 400);
  assert.equal((await fetch(base + '/download/unused/laptop-camera.zip')).status, 404);
  await new Promise(resolve => backend.close(resolve));
  assert.equal((await fetch(base + '/api/health')).status, 502);
  console.log('PASS: Task1/2 and Task3 pages, MediaPipe assets, health/control, all preview routes, request boundaries and unavailable recorder.');
})().catch(error => { console.error(error); process.exitCode = 1; }).finally(async () => {
  if (child && child.exitCode === null && child.signalCode === null) {
    await new Promise(resolve => { child.once('close', resolve); child.kill(); });
  }
  if (backend.listening) await new Promise(resolve => backend.close(resolve));
});
