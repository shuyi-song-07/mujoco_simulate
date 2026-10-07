import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { DEFAULT_DUAL_CONFIG, DualInteractionState, detectionsFromMediaPipe, resolveOperation } from "./dual-interaction-state.js";

// Run the actual browser entry point against controlled browser boundaries.
// The real gesture engine is injected; no copy of application logic is tested.
const applicationSource = readFileSync(new URL("./dual-arm-app.js", import.meta.url), "utf8")
  .replace(/^import\s+[\s\S]*?\s+from\s+["'][^"']+["'];\s*$/gm, "")
  .replaceAll("import.meta.url", '"http://127.0.0.1:8000/dual-arm-app.js"');

const cameraNames = ["top", "side", "front"];
const imageIds = ["main-preview", "side-preview", "front-preview"];
const healthyRuntime = () => ({
  ok: true, task_mode: "dual_arm", session: "browser-regression", control_epoch: 0,
  task_state: "PREGRASP", control_mode: "IDLE", gripper_latch: "OPEN",
  recording: false, saving: false, frames: 0,
  config: { control: { ...DEFAULT_DUAL_CONFIG, max_command_age_ms: 150 } },
});

function deferred() {
  let resolve, reject;
  const promise = new Promise((accept, fail) => { resolve = accept; reject = fail; });
  return { promise, resolve, reject };
}

async function settle() {
  // Browser fetch/decode continuations run as microtasks. setImmediate drains
  // them without advancing the fake clock or running recurring timers.
  await new Promise((resolve) => setImmediate(resolve));
}

function makeHarness() {
  let now = 1000, nextTimer = 0, nextUrl = 0, nextUuid = 0;
  const trace = [], requests = [], revoked = [], timers = new Map(), animations = new Map();
  const elements = new Map(), routes = new Map(), urlCameras = new Map(), decodePlans = new Map();
  const workers = [], bitmapClosures = [], windowListeners = new Map(), documentListeners = new Map();

  class Element {
    constructor(id = "", tagName = "DIV") {
      this.id = id; this.tagName = tagName; this.dataset = {}; this.listeners = new Map();
      this.hidden = id.includes("preview"); this.disabled = false; this.checked = false;
      this.textContent = ""; this.value = 0; this.width = 640; this.height = 480;
      this.videoWidth = 640; this.videoHeight = 480; this.readyState = 2; this.currentTime = 0;
      this._src = ""; this.srcAssignments = 0; this.srcRemovals = 0;
    }
    get src() { return this._src; }
    set src(value) {
      this._src = String(value); this.srcAssignments += 1;
      trace.push({ type: "assign", id: this.id, url: this._src });
    }
    getAttribute(name) { return name === "src" ? this._src || null : null; }
    removeAttribute(name) {
      if (name === "src") { this._src = ""; this.srcRemovals += 1; trace.push({ type: "remove", id: this.id }); }
    }
    addEventListener(type, handler) {
      if (!this.listeners.has(type)) this.listeners.set(type, []);
      this.listeners.get(type).push(handler);
    }
    removeEventListener(type, handler) {
      this.listeners.set(type, (this.listeners.get(type) ?? []).filter((value) => value !== handler));
    }
    dispatch(type, data = {}) {
      for (const handler of this.listeners.get(type) ?? []) handler({ target: this, preventDefault() {}, ...data });
    }
    getContext() {
      return { clearRect() {}, save() {}, restore() {}, translate() {}, scale() {}, fillText() {}, drawImage() {}, beginPath() {}, ellipse() {}, stroke() {}, moveTo() {}, lineTo() {} };
    }
    async play() {}
    async decode() {}
  }
  const getElement = (id) => {
    if (!elements.has(id)) elements.set(id, new Element(id, imageIds.includes(id) ? "IMG" : id === "webcam" ? "VIDEO" : "DIV"));
    return elements.get(id);
  };
  const buttons = ["record_start", "record_save", "record_failure", "record_discard", "reset", "record_stop"].map((command) => {
    const button = new Element(command, "BUTTON"); button.dataset.command = command; return button;
  });

  class BrowserImage extends Element {
    get src() { return super.src; }
    set src(value) {
      super.src = value;
      const camera = urlCameras.get(String(value));
      const plans = decodePlans.get(camera) ?? [];
      const plan = plans.shift();
      const promise = plan ? plan.promise : Promise.resolve();
      this.loading = promise.then(() => {
        this.complete = true; this.naturalWidth = 640; this.naturalHeight = 480;
        trace.push({ type: "decoded", camera, url: String(value) }); this.onload?.();
      }, (error) => { this.onerror?.(error); throw error; });
      this.loading.catch(() => {}); // onerror users do not necessarily call decode().
    }
    decode() { return this.loading; }
  }
  class BrowserURL extends URL {
    static createObjectURL(blob) {
      const url = `blob:regression-${++nextUrl}`;
      urlCameras.set(url, blob.camera);
      trace.push({ type: "created", url, camera: blob.camera });
      return url;
    }
    static revokeObjectURL(url) { revoked.push(url); trace.push({ type: "revoke", url }); }
  }
  class BrowserDate extends Date {
    constructor(...values) { super(...(values.length ? values : [1900000000000 + now])); }
    static now() { return 1900000000000 + now; }
  }
  class BrowserWorker {
    constructor(url, options) { this.url = String(url); this.options = options; this.messages = []; this.listeners = new Map(); workers.push(this); }
    addEventListener(type, handler) { this.listeners.set(type, [...(this.listeners.get(type) ?? []), handler]); }
    removeEventListener(type, handler) { this.listeners.set(type, (this.listeners.get(type) ?? []).filter((value) => value !== handler)); }
    emit(data) {
      const event = { data };
      this.onmessage?.(event);
      for (const handler of this.listeners.get("message") ?? []) handler(event);
    }
    postMessage(message, transfer) {
      this.messages.push({ ...message, transfer });
      if (message.type === "init") queueMicrotask(() => this.emit({ type: "ready" }));
    }
    terminate() { this.terminated = true; }
  }
  function response(status = 200, data = {}, camera = null) {
    return {
      ok: status >= 200 && status < 300, status,
      headers: { get: (name) => name.toLowerCase() === "content-type" ? camera ? "image/jpeg" : "application/json" : null },
      async json() { return data; },
      async blob() { return { camera, type: "image/jpeg", size: 32 }; },
    };
  }
  const fetch = async (url, options = {}) => {
    const entry = { url: String(url), options, body: options.body ? JSON.parse(options.body) : null, time: now };
    requests.push(entry);
    const plans = routes.get(entry.url) ?? [];
    if (plans.length) {
      const plan = plans.shift();
      return typeof plan === "function" ? plan(entry) : plan;
    }
    if (entry.url.endsWith("/health")) return response(200, healthyRuntime());
    const match = entry.url.match(/\/(top|side|front)-preview$/);
    if (match) return response(200, {}, match[1]);
    if (entry.url.endsWith("/control")) return response(200, { ok: true, queued: true });
    throw new Error(`Unexpected fetch: ${entry.url}`);
  };
  const fakeStream = { getTracks: () => [{ stop() { trace.push({ type: "camera-stopped" }); } }] };
  const sandbox = {
    console, DEFAULT_DUAL_CONFIG, DualInteractionState, detectionsFromMediaPipe, resolveOperation,
    Date: BrowserDate, URL: BrowserURL, Image: BrowserImage, Worker: BrowserWorker,
    fetch, AbortSignal, Promise, performance: { now: () => now },
    navigator: { mediaDevices: { async getUserMedia() { return fakeStream; } } },
    location: { href: "http://127.0.0.1:8000/dual_arm.html" },
    crypto: { randomUUID: () => `test-${++nextUuid}` },
    document: { hidden: false, getElementById: getElement, querySelectorAll: () => buttons,
      addEventListener(type, handler) { documentListeners.set(type, handler); } },
    window: { addEventListener(type, handler) { windowListeners.set(type, handler); } },
    setTimeout(handler, delay) { timers.set(++nextTimer, { handler, delay, due: now + delay }); return nextTimer; },
    clearTimeout(id) { timers.delete(id); },
    setInterval(handler, delay) { timers.set(++nextTimer, { handler, delay, due: now + delay, interval: true }); return nextTimer; },
    clearInterval(id) { timers.delete(id); },
    requestAnimationFrame(handler) { animations.set(++nextTimer, handler); return nextTimer; },
    cancelAnimationFrame(id) { animations.delete(id); },
    async createImageBitmap() { return { width: 640, height: 480, close() { bitmapClosures.push(true); } }; },
    DrawingUtils: class { drawConnectors() {} drawLandmarks() {} },
    FilesetResolver: { async forVisionTasks() { return {}; } },
    GestureRecognizer: { HAND_CONNECTIONS: [], async createFromOptions() { return { recognizeForVideo() { return { landmarks: [], handedness: [], gestures: [] }; }, close() {} }; } },
  };
  const context = vm.createContext(sandbox);
  const expose = `
    globalThis.__app = {
      setConnected, postPacket, flushPackets, queueOutput, checkHealth, assignImage,
      renderViews, pollPreviews, renderState, startCamera, stopCamera, detectFrame,
      get connected() { return connected; }, get engine() { return engine; },
      get output() { return output; }, get previews() { return { ...previews }; },
      get latestPacket() { return latestPacket; }, get runtime() { return runtime; },
      get events() { return [...events]; }, get pendingOperation() { return pendingOperation; },
      get transportFrozen() { return transportFrozen; },
      setOutput(value) { output = value; },
      setPendingOperation(value) { pendingOperation = value; },
      enqueueEvent(value) { events.push(value); },
    };`;
  vm.runInContext(applicationSource + expose, context, { filename: "dual-arm-app.js" });
  const app = context.__app;
  return {
    app, elements, trace, requests, revoked, workers, response,
    get now() { return now; }, advance(milliseconds) { now += milliseconds; },
    element: getElement,
    button(command) { return buttons.find((button) => button.dataset.command === command); },
    enqueue(url, ...plans) { routes.set(url, [...(routes.get(url) ?? []), ...plans]); },
    holdDecode(camera) { const plan = deferred(); decodePlans.set(camera, [...(decodePlans.get(camera) ?? []), plan]); return plan; },
    controls() { return requests.filter((entry) => entry.url.endsWith("/control")); },
    async boot() { await settle(); assert.equal(app.connected, true); },
    async seedPreviews() { await app.pollPreviews(); assert.ok(imageIds.every((id) => !getElement(id).hidden)); },
    async animation() { const frame = animations.entries().next().value; if (frame) { animations.delete(frame[0]); frame[1](now); } await settle(); },
    async pump(milliseconds = 0) {
      now += milliseconds;
      for (let pass = 0; pass < 20; pass++) {
        const due = [...timers].filter(([, timer]) => timer.due <= now && timer.delay <= DEFAULT_DUAL_CONFIG.control_interval_ms);
        if (!due.length) return;
        for (const [id, timer] of due) { timers.delete(id); timer.handler(); }
        await settle();
      }
      throw new Error("Control pump repeatedly scheduled immediate work without advancing time");
    },
    close() { windowListeners.get("pagehide")?.(); },
  };
}

function motion(amount, telemetry = { control_mode: "XY", training_recordable: true }) {
  return { mode: "XY", reason: null, telemetry,
    command: { command: "dual_motion", mode: "xy", left: { dx: amount, dy: 0, dz: 0 },
               right: { dx: amount, dy: 0, dz: 0 }, telemetry } };
}

function recognition(gesture, confidence = 0.98, handedness = 0.99) {
  const landmarks = (x) => Array.from({ length: 21 }, () => ({ x, y: 0.5, z: 0 }));
  return { landmarks: [landmarks(0.2), landmarks(0.8)],
    handedness: [[{ categoryName: "Left", score: handedness }], [{ categoryName: "Right", score: handedness }]],
    gestures: [[{ categoryName: gesture, score: confidence }], [{ categoryName: gesture, score: confidence }]] };
}

async function deliverRecognition(harness, result, ageMs = 0) {
  const worker = harness.workers[0];
  const sent = worker.messages.filter((message) => message.type === "frame").at(-1);
  harness.advance(ageMs);
  worker.emit({ type: "result", frameId: sent.frameId, capturedAt: sent.capturedAt,
    capturedEpochMs: sent.capturedEpochMs, control_epoch: sent.control_epoch, session: sent.session, inferenceMs: ageMs || 10, result });
  await settle(); await harness.pump(40);
  harness.element("webcam").currentTime += 0.04;
  await harness.animation();
}

for (const status of [409, 429, 503]) {
  test(`one control HTTP ${status} freezes commands without a false disconnection or disappearing previews`, async (t) => {
    const harness = makeHarness(); t.after(() => harness.close());
    await harness.boot(); await harness.seedPreviews();
    const previous = imageIds.map((id) => harness.element(id).src);
    harness.enqueue("/api/dual/control", harness.response(status, { ok: false, error: "temporary command rejection" }));
    harness.app.queueOutput(motion(0.2), harness.now, true);
    await settle(); harness.app.renderState();
    assert.equal(harness.app.connected, true, "a rejected control request is not an authoritative health failure");
    assert.equal(harness.element("connection").dataset.state, "active");
    assert.ok(imageIds.every((id, i) => !harness.element(id).hidden && harness.element(id).src === previous[i]));
    assert.equal(harness.app.output.mode, "PAUSE", "motion must still freeze safely on a transport error");
    assert.equal(harness.app.latestPacket, null, "failed motion must not remain queued for replay");
    await harness.app.checkHealth();
    assert.equal(harness.app.connected, true);
    assert.equal(harness.app.output.mode, "PAUSE", "health recovery cannot restore an old motion anchor automatically");
  });
}

test("a genuine health failure pauses immediately and hides images when their freshness window expires", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await harness.seedPreviews();
  harness.enqueue("/api/dual/health", () => Promise.reject(new Error("backend socket closed")));
  await harness.app.checkHealth();
  assert.equal(harness.app.connected, false);
  assert.equal(harness.app.output.mode, "PAUSE");
  assert.equal(harness.app.engine.gripperLatch, null, "unknown backend state must not pretend the gripper is known");
  harness.advance(1001); harness.app.renderViews();
  assert.ok(imageIds.every((id) => harness.element(id).hidden), "last good pictures may not remain visible indefinitely");
});

test("rendering repeated UI state never assigns the same preview URL again", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await harness.seedPreviews();
  const assignments = imageIds.map((id) => harness.element(id).srcAssignments);
  for (let i = 0; i < 10; i++) harness.app.renderState();
  assert.deepEqual(imageIds.map((id) => harness.element(id).srcAssignments), assignments);
});

test("new preview URLs are decoded before display and old URLs are revoked only after replacement", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await harness.seedPreviews();
  const before = { ...harness.app.previews };
  const held = cameraNames.map((camera) => harness.holdDecode(camera));
  const pending = harness.app.pollPreviews(); await settle();
  for (const camera of cameraNames) {
    assert.equal(harness.app.previews[camera], before[camera], "decode in flight must preserve the previous frame");
    assert.equal(harness.revoked.includes(before[camera]), false, "the displayed old URL is still valid");
  }
  held.forEach((plan) => plan.resolve()); await pending;
  for (const camera of cameraNames) {
    const next = harness.app.previews[camera];
    assert.notEqual(next, before[camera]);
    const decodedAt = harness.trace.findIndex((entry) => entry.type === "decoded" && entry.url === next);
    const assignedAt = harness.trace.findIndex((entry) => entry.type === "assign" && imageIds.includes(entry.id) && entry.url === next);
    const revokedAt = harness.trace.findIndex((entry) => entry.type === "revoke" && entry.url === before[camera]);
    assert.ok(decodedAt >= 0 && assignedAt > decodedAt && revokedAt > assignedAt, `${camera}: decode → assign → revoke`);
  }
});

test("decode failure preserves a still-fresh good frame without extending its expiry", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await harness.seedPreviews();
  const before = harness.app.previews.side;
  harness.advance(100);
  const held = harness.holdDecode("side");
  const pending = harness.app.pollPreviews(); await settle();
  const failedUrl = harness.trace.findLast((entry) => entry.type === "created" && entry.camera === "side").url;
  held.reject(new Error("corrupt JPEG")); await pending;
  assert.equal(harness.app.previews.side, before);
  assert.equal(harness.element("side-preview").hidden, false);
  assert.equal(harness.revoked.includes(before), false);
  assert.equal(harness.revoked.includes(failedUrl), true, "the rejected new blob must be released");
  harness.advance(901); harness.app.renderViews();
  assert.equal(harness.element("side-preview").hidden, true, "failed refresh must not renew stale-image lifetime");
});

test("heartbeat merges into the freshest queued motion without refreshing the motion's capture time", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot();
  const blocked = deferred();
  harness.enqueue("/api/dual/control", () => blocked.promise);
  harness.app.queueOutput(motion(0.1), harness.now, true); await settle();
  harness.advance(10); harness.app.queueOutput(motion(0.2), harness.now, true);
  harness.advance(10); harness.app.queueOutput(motion(0.3), harness.now, true);
  const measuredAt = harness.app.latestPacket.sentAt;
  harness.advance(10);
  harness.app.queueOutput({ mode: "XY", reason: null, command: null,
    telemetry: { control_mode: "XY", training_recordable: true, newest_marker: "heartbeat" } }, harness.now, true);
  assert.equal(harness.app.latestPacket.command, "dual_motion");
  assert.equal(harness.app.latestPacket.left.dx, 0.3, "intermediate queued motion must be coalesced");
  assert.equal(harness.app.latestPacket.sentAt, measuredAt, "a heartbeat is not a new movement measurement");
  assert.equal(harness.app.latestPacket.telemetry.newest_marker, "heartbeat");
  blocked.resolve(harness.response(200, { ok: true })); await settle();
  await harness.pump(10);
  const movements = harness.controls().filter((entry) => entry.body.command === "dual_motion");
  assert.equal(movements.length, 2, "one in-flight plus one newest motion; no queue replay");
  assert.equal(movements[1].body.left.dx, 0.3);
});

test("successful ordinary telemetry and pause cannot mistake two missing event IDs for an operation ACK", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot();
  assert.equal(harness.app.pendingOperation, null);
  harness.app.queueOutput({ mode: "IDLE", reason: null, command: null,
    telemetry: { control_mode: "IDLE", training_recordable: false } }, harness.now, true);
  await settle();
  assert.equal(harness.controls().at(-1).body.command, "telemetry");
  assert.equal(harness.app.transportFrozen, false, "HTTP 200 without eventId must not write pendingOperation.queued on null");
  assert.equal(harness.app.connected, true);
  harness.advance(40);
  harness.app.queueOutput(harness.app.engine.pause("user_pause"), harness.now, true);
  await settle(); await harness.pump();
  assert.equal(harness.controls().at(-1).body.command, "pause");
  assert.equal(harness.app.transportFrozen, false);
  assert.equal(harness.app.connected, true);
});

test("only a real event ID matching a pending operation marks that operation as queued", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot();
  harness.app.setPendingOperation({ command: "record_start", eventId: "operation-expected", startedAt: harness.now });
  harness.app.enqueueEvent({ command: "telemetry", telemetry: {}, sentAt: Date.now() });
  await harness.app.flushPackets();
  assert.equal(harness.app.pendingOperation.queued, undefined);
  harness.app.enqueueEvent({ command: "record_start", eventId: "operation-other" });
  await harness.app.flushPackets();
  assert.equal(harness.app.pendingOperation.queued, undefined);
  harness.app.enqueueEvent({ command: "record_start", eventId: "operation-expected" });
  await harness.app.flushPackets();
  assert.equal(harness.app.pendingOperation.queued, true);
  assert.equal(harness.app.transportFrozen, false);
});

test("an expired queued movement is replaced by heartbeat rather than replayed after a slow POST", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot();
  const blocked = deferred();
  harness.enqueue("/api/dual/control", () => blocked.promise);
  harness.app.queueOutput(motion(0.1), harness.now, true); await settle();
  harness.app.queueOutput(motion(0.9), harness.now, true);
  harness.advance(151);
  harness.app.queueOutput({ mode: "XY", reason: null, command: null,
    telemetry: { control_mode: "XY", training_recordable: true } }, harness.now, true);
  blocked.resolve(harness.response(200, { ok: true })); await settle();
  assert.equal(harness.controls().filter((entry) => entry.body.command === "dual_motion").length, 1);
  assert.equal(harness.controls().at(-1).body.command, "telemetry");
  assert.ok(harness.app.connected);
});

test("a busy recognition Worker keeps health and three previews responsive without queuing camera frames", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await harness.app.startCamera(); await settle();
  assert.equal(harness.workers.length, 1);
  const worker = harness.workers[0];
  assert.equal(worker.messages[0].type, "init");
  assert.equal(worker.messages[0].delegate, "CPU");
  assert.equal(worker.messages.filter((message) => message.type === "frame").length, 1);
  harness.advance(100); harness.element("webcam").currentTime = 1;
  await harness.animation();
  await harness.app.checkHealth(); await harness.seedPreviews();
  assert.equal(harness.app.connected, true);
  assert.equal(worker.messages.filter((message) => message.type === "frame").length, 1,
    "a stalled inference has only one bitmap in flight, not a replay queue");
  assert.ok(imageIds.every((id) => !harness.element(id).hidden));
});

test("a recognition result older than the command freshness limit is discarded instead of moving the robot", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await harness.app.startCamera(); await settle();
  const worker = harness.workers[0];
  const frame = worker.messages.find((message) => message.type === "frame");
  for (let at = harness.now - 2000; at <= harness.now; at += 40) {
    harness.app.engine.tracker.update([
      { side: "left", handedness_score: 0.99, gesture: "Open_Palm", confidence: 0.98, wrist: [0.2, 0.5] },
      { side: "right", handedness_score: 0.99, gesture: "Open_Palm", confidence: 0.98, wrist: [0.8, 0.5] },
    ], at);
  }
  assert.equal(harness.app.engine.tracker.calibrated, true);
  harness.advance(151);
  worker.emit({ type: "result", frameId: frame.frameId, capturedAt: frame.capturedAt,
    capturedEpochMs: frame.capturedEpochMs, control_epoch: frame.control_epoch, session: frame.session,
    inferenceMs: 151, result: recognition("Closed_Fist", 0.95) });
  await settle(); await harness.pump();
  assert.equal(harness.app.output.mode, "PAUSE");
  assert.equal(harness.app.output.reason, "stale_recognition");
  assert.equal(harness.controls().filter((entry) => entry.body.command === "dual_motion").length, 0);
  const packet = harness.controls().at(-1).body;
  assert.ok(["pause", "telemetry"].includes(packet.command));
  assert.equal(packet.telemetry.recognition_result.accepted, false);
  assert.equal(packet.telemetry.recognition_result.drop_reason, "stale_recognition");
  assert.equal(packet.telemetry.recognition_result.result_age_ms, 151);
  assert.equal(packet.telemetry.recognition_result.captured_at_ms, frame.capturedEpochMs);
  assert.equal(packet.telemetry.recognition_result.observed_hands[0].gesture, "Closed_Fist");
  assert.equal(packet.telemetry.recognition_result.observed_hands[0].score, 0.95);
  assert.equal(packet.telemetry.hands.left.gesture, "Open_Palm", "stale diagnostic labels must not overwrite trusted control hands");
  assert.equal(harness.app.engine.tracker.hands.left.last_seen_ms, frame.capturedAt);
  assert.equal(harness.app.connected, true);
});

test("moderate-confidence palms and brief classification flicker advance calibration without robot actions", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await harness.app.startCamera(); await settle();
  for (let i = 0; i < 20; i++) await deliverRecognition(harness, recognition("Open_Palm", 0.65));
  assert.ok(harness.element("calibration-progress").value > 0.2);
  assert.match(harness.element("left-gesture").textContent, /张掌 · 65%/);
  assert.match(harness.element("left-identity").textContent, /99%.*80%/);
  for (let i = 0; i < 90; i++) await deliverRecognition(harness,
    recognition(i % 7 === 0 ? "None" : "Open_Palm", i % 7 === 0 ? 0.4 : 0.65));
  assert.equal(harness.app.engine.tracker.gestureCalibrationStatus(harness.now).phase, "CONFIRMED");
  assert.equal(harness.app.engine.tracker.calibrated, true);
  assert.ok(harness.controls().every((request) => ["telemetry", "pause"].includes(request.body.command)));
});

test("recognized palms below the binding threshold show a specific score explanation instead of unexplained zero", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await harness.app.startCamera(); await settle();
  for (let i = 0; i < 16; i++) await deliverRecognition(harness, recognition("Open_Palm", 0.599));
  assert.equal(harness.element("calibration-progress").value, 0);
  assert.match(harness.element("left-gesture").textContent, /张掌 · 59.9%/);
  assert.match(harness.element("calibration-detail").textContent, /评分 59.9%.*至少 60%/);
  assert.equal(harness.app.engine.tracker.calibrated, false);
});

test("identity score and duplicate labels are explained even when palm landmarks and gesture labels are visible", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await harness.app.startCamera(); await settle();
  for (let i = 0; i < 8; i++) await deliverRecognition(harness, recognition("Open_Palm", 0.98, 0.799));
  assert.match(harness.element("left-identity").textContent, /79.9%.*80%/);
  assert.match(harness.element("calibration-detail").textContent, /左右身份评分 79.9%.*至少 80%/);
  const duplicate = recognition("Open_Palm"); duplicate.handedness[1][0].categoryName = "Left";
  for (let i = 0; i < 8; i++) await deliverRecognition(harness, duplicate);
  assert.match(harness.element("calibration-detail").textContent, /两只手被识别为同一侧/);
  assert.equal(harness.app.engine.tracker.calibrated, false);
  assert.ok(harness.controls().every((request) => ["telemetry", "pause"].includes(request.body.command)));
});

test("slower fresh frames may bind paused hands but cannot relax motion or gripper freshness after binding", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await harness.app.startCamera(); await settle();
  for (let i = 0; i < 12; i++) await deliverRecognition(harness, recognition("Open_Palm", 0.65), 180);
  assert.equal(harness.app.engine.tracker.calibrated, true);
  assert.equal(harness.app.output.mode, "PAUSE");
  for (let i = 0; i < 6; i++) await deliverRecognition(harness, recognition("Closed_Fist"), 180);
  assert.equal(harness.app.output.reason, "stale_recognition");
  assert.match(harness.element("calibration-detail").textContent, /延迟 180 ms.*150 ms/);
  assert.ok(harness.controls().every((request) => ["telemetry", "pause"].includes(request.body.command)));
});

test("calibration cannot consume a result older than its own freshness limit", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await harness.app.startCamera(); await settle();
  await deliverRecognition(harness, recognition("Open_Palm"), 501);
  assert.equal(harness.element("calibration-progress").value, 0);
  assert.equal(harness.app.output.reason, "stale_recognition");
  assert.match(harness.element("calibration-detail").textContent, /延迟 501 ms.*500 ms/);
  assert.equal(harness.app.engine.tracker.calibrated, false);
});

test("one operator can bind both hands with gestures, lower them, and explicitly start recording", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await harness.app.startCamera(); await settle();
  const worker = harness.workers[0];
  async function frame(gesture, partialHand = false) {
    const sent = worker.messages.filter((message) => message.type === "frame").at(-1);
    const landmarks = (x) => Array.from({ length: 21 }, () => ({ x, y: 0.5, z: 0 }));
    const result = gesture ? {
      landmarks: [landmarks(0.2), landmarks(0.8)],
      handedness: [[{ categoryName: "Left", score: 0.99 }], [{ categoryName: "Right", score: 0.99 }]],
      gestures: [[{ categoryName: gesture, score: 0.98 }], [{ categoryName: gesture, score: 0.98 }]],
    } : { landmarks: [], handedness: [], gestures: [] };
    if (partialHand) {
      result.landmarks = result.landmarks.slice(0, 1);
      result.handedness = [[{ categoryName: "Left", score: 0.6 }]];
      result.gestures = result.gestures.slice(0, 1);
    }
    worker.emit({ type: "result", frameId: sent.frameId, capturedAt: sent.capturedAt,
      capturedEpochMs: sent.capturedEpochMs, control_epoch: sent.control_epoch, session: sent.session, inferenceMs: 10, result });
    await settle();
    await harness.pump(40);
    harness.element("webcam").currentTime += 0.04;
    await harness.animation();
  }
  for (let i = 0; i < 55; i++) await frame("Open_Palm");
  assert.equal(harness.app.engine.tracker.calibrated, true, "two seconds of palms bind directly without another gesture or click");
  assert.equal(harness.element("calibration-progress").value, 1);
  assert.match(harness.element("calibration-state").textContent, /左右手已绑定/);
  assert.equal(harness.app.engine.gripperLatch, "OPEN");
  assert.equal(harness.app.output.mode, "PAUSE");
  assert.ok(harness.controls().every((request) => ["telemetry", "pause"].includes(request.body.command)),
    "binding cannot move either arm, change grippers or start a dataset");

  for (let i = 0; i < 8; i++) await frame("Closed_Fist", true);
  assert.equal(harness.app.engine.tracker.calibrated, true,
    "an uncertain partial hand while lowering the hands must not disable recording");
  assert.equal(harness.app.output.mode, "PAUSE");
  assert.equal(harness.button("record_start").disabled, false);
  assert.ok(harness.controls().every((request) => ["telemetry", "pause"].includes(request.body.command)));
  for (let i = 0; i < 8; i++) await frame(null);
  assert.equal(harness.app.engine.tracker.calibrated, true, "lowering both hands retains confirmed identity");
  assert.equal(harness.app.output.mode, "PAUSE");
  assert.equal(harness.button("record_start").disabled, false, "one operator may lower the hands to use the mouse");
  assert.equal(harness.controls().some((request) => request.body.command === "record_start"), false);
  harness.button("record_start").dispatch("click");
  await settle(); await harness.pump(40);
  assert.equal(harness.controls().filter((request) => request.body.command === "record_start").length, 1);
  assert.equal(harness.app.pendingOperation.command, "record_start");
  assert.equal(harness.app.pendingOperation.queued, true);
});

async function bindHands(harness) {
  await harness.app.startCamera(); await settle();
  for (let i = 0; i < 55; i++) await deliverRecognition(harness, recognition("Open_Palm"));
  assert.equal(harness.app.engine.tracker.calibrated, true);
  assert.equal(harness.element("unbind-hands").disabled, false);
}

test("explicit unbinding cancels unsent hand actions and keeps recording and closed grippers intact", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot();
  const state = { ...healthyRuntime(), recording: true, frames: 123, gripper_latch: "CLOSED" };
  harness.enqueue("/api/dual/health", harness.response(200, state));
  await harness.app.checkHealth(); await bindHands(harness);
  assert.equal(harness.app.engine.gripperLatch, "CLOSED", "calibration palms cannot release a closed gripper");
  const blocked = deferred();
  harness.enqueue("/api/dual/control", () => blocked.promise);
  harness.app.queueOutput(harness.app.engine.pause("user_pause"), harness.now, true); await settle();
  harness.app.queueOutput(motion(0.9), harness.now, true);
  harness.app.enqueueEvent({ command: "dual_gripper", action: "open", eventId: "cancel-grip", session: state.session });
  harness.element("unbind-hands").dispatch("click");
  assert.equal(harness.app.engine.tracker.calibrated, false);
  assert.equal(harness.app.engine.gripperLatch, "CLOSED");
  assert.equal(harness.app.runtime.recording, true);
  assert.equal(harness.app.runtime.frames, 123);
  assert.equal(harness.app.events.some((packet) => packet.command === "dual_gripper"), false);
  assert.equal(harness.app.latestPacket.command, "pause");
  assert.equal(harness.button("record_start").disabled, true);
  assert.equal(harness.element("unbind-hands").disabled, true);
  blocked.resolve(harness.response(200, { ok: true })); await settle(); await harness.pump(40);
  assert.ok(harness.controls().every((entry) => ["pause", "telemetry"].includes(entry.body.command)));
  for (let i = 0; i < 10; i++) await deliverRecognition(harness, recognition("Open_Palm"));
  assert.equal(harness.app.engine.tracker.calibrated, false, "unbinding clears the previous two-second evidence");
  for (let i = 0; i < 45; i++) await deliverRecognition(harness, recognition("Open_Palm"));
  assert.equal(harness.app.engine.tracker.calibrated, true);
  assert.equal(harness.app.engine.gripperLatch, "CLOSED");
});

test("duplicate and reversed identities pause indefinitely while retaining binding and trusted wrist identities", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await bindHands(harness);
  const duplicate = recognition("Closed_Fist"); duplicate.handedness[1][0].categoryName = "Left";
  for (let i = 0; i < 8; i++) await deliverRecognition(harness, duplicate);
  assert.equal(harness.app.engine.tracker.calibrated, true);
  const reversed = recognition("ILoveYou"); reversed.landmarks.reverse();
  for (let i = 0; i < 80; i++) await deliverRecognition(harness, reversed);
  assert.equal(harness.app.engine.tracker.calibrated, true);
  assert.equal(harness.app.output.mode, "PAUSE");
  assert.equal(harness.app.output.reason, "identity_conflict");
  assert.ok(harness.app.engine.tracker.hands.left.wrist_raw[0] < harness.app.engine.tracker.hands.right.wrist_raw[0]);
  assert.match(harness.element("calibration-detail").textContent, /绑定保留/);
  assert.ok(harness.controls().every((entry) => ["pause", "telemetry"].includes(entry.body.command)));
});

test("updating configuration and resetting a scene retain camera binding but clear old motion baselines", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await bindHands(harness);
  const changed = healthyRuntime(); changed.config.control.wrist_sensitivity = 0.05;
  harness.enqueue("/api/dual/health", harness.response(200, changed));
  await harness.app.checkHealth();
  assert.equal(harness.app.engine.tracker.calibrated, true);
  assert.equal(harness.app.engine.config.wrist_sensitivity, 0.05);
  assert.equal(harness.app.output.mode, "PAUSE");
  assert.equal(harness.app.engine.tracker.recoveryPending, true);
  const reset = { ...changed, control_epoch: 1 };
  harness.enqueue("/api/dual/health", harness.response(200, reset));
  await harness.app.checkHealth();
  assert.equal(harness.app.engine.tracker.calibrated, true);
  assert.equal(harness.app.output.mode, "PAUSE");
  assert.equal(harness.app.engine.tracker.recoveryPending, true);
});

test("a restarted backend with the same epoch clears old operations and rejects an old-session camera result", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await bindHands(harness);
  const worker = harness.workers[0], oldFrame = worker.messages.filter((packet) => packet.type === "frame").at(-1);
  const blocked = deferred();
  harness.enqueue("/api/dual/control", () => blocked.promise);
  harness.app.queueOutput(harness.app.engine.pause("user_pause"), harness.now, true); await settle();
  harness.app.queueOutput(motion(0.9), harness.now, true);
  harness.app.enqueueEvent({ command: "record_start", eventId: "old-operation", session: oldFrame.session });
  harness.app.setPendingOperation({ command: "record_start", eventId: "old-operation", startedAt: harness.now });
  const restarted = { ...healthyRuntime(), session: "backend-restarted" };
  harness.enqueue("/api/dual/health", harness.response(200, restarted));
  await harness.app.checkHealth();
  assert.equal(harness.app.engine.tracker.calibrated, true);
  assert.equal(harness.app.pendingOperation, null);
  assert.equal(harness.app.events.length, 0);
  assert.equal(harness.app.output.mode, "PAUSE");
  const trustedSeen = harness.app.engine.tracker.hands.left.last_seen_ms;
  worker.emit({ type: "result", frameId: oldFrame.frameId, capturedAt: oldFrame.capturedAt,
    capturedEpochMs: oldFrame.capturedEpochMs, control_epoch: oldFrame.control_epoch, session: oldFrame.session,
    inferenceMs: 10, result: recognition("Closed_Fist") });
  await settle();
  assert.equal(harness.app.output.reason, "backend_session_changed");
  assert.equal(harness.app.engine.tracker.calibrated, true);
  blocked.resolve(harness.response(200, { ok: true })); await settle(); await harness.pump(40);
  assert.equal(harness.controls().some((entry) => ["record_start", "dual_motion", "dual_gripper"].includes(entry.body.command)), false);
  assert.equal(harness.controls().at(-1).body.session, "backend-restarted");
  const diagnostic = harness.controls().at(-1).body.telemetry.recognition_result;
  assert.equal(diagnostic.source_session, oldFrame.session);
  assert.equal(diagnostic.accepted, false);
  assert.equal(diagnostic.drop_reason, "backend_session_changed");
  assert.equal(diagnostic.observed_hands[0].gesture, "Closed_Fist");
  assert.equal(harness.app.engine.tracker.hands.left.gesture, "Open_Palm");
  assert.equal(harness.app.engine.tracker.hands.left.last_seen_ms, trustedSeen, "an old-session result cannot refresh trusted identity");
});

test("closing the camera or swapping left and right explicitly clears binding without a gripper command", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await bindHands(harness);
  harness.element("swap-labels").checked = true;
  harness.element("swap-labels").dispatch("change");
  assert.equal(harness.app.engine.tracker.calibrated, false);
  assert.equal(harness.app.output.reason, "mapping_changed");
  for (let i = 0; i < 55; i++) await deliverRecognition(harness, recognition("Open_Palm"));
  assert.equal(harness.app.engine.tracker.calibrated, true);
  harness.app.stopCamera(); await settle(); await harness.pump(40);
  assert.equal(harness.app.engine.tracker.calibrated, false);
  assert.equal(harness.app.engine.gripperLatch, "OPEN");
  assert.equal(harness.app.output.reason, "camera_stopped");
  assert.ok(harness.controls().every((entry) => ["pause", "telemetry"].includes(entry.body.command)));
});

test("XY, Z and manual camera choices always show all three different views", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await harness.seedPreviews();
  const assertDistinct = () => assert.equal(new Set(imageIds.map((id) => harness.element(id).src)).size, 3);
  assertDistinct();
  harness.app.setOutput({ ...harness.app.output, mode: "Z" }); harness.app.renderViews();
  assert.equal(harness.element("main-preview").src, harness.app.previews.front);
  assert.equal(harness.element("side-preview").src, harness.app.previews.top);
  assertDistinct();
  harness.element("view-select").value = "front"; harness.element("view-select").dispatch("change");
  assert.equal(harness.element("main-preview").src, harness.app.previews.front);
  assert.match(harness.element("main-view-title").textContent, /内侧近景/);
  assertDistinct();
  harness.app.setOutput({ ...harness.app.output, mode: "XY" }); harness.app.renderViews();
  assert.equal(harness.element("main-preview").src, harness.app.previews.front, "manual choice survives a motion-mode change");
  harness.element("view-select").value = "auto"; harness.element("view-select").dispatch("change");
  assert.equal(harness.element("main-preview").src, harness.app.previews.top);
  assertDistinct();
});

test("motion settings reconfigure safely before recording and are locked during a recorded episode", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await bindHands(harness);
  harness.element("motion-mapping").value = "incremental";
  harness.element("motion-speed").value = "1";
  harness.element("motion-mapping").dispatch("change"); await settle(); await harness.pump(40);
  assert.equal(harness.app.engine.config.motion_mapping, "incremental");
  assert.equal(harness.app.engine.config.motion_speed_scale, 1);
  assert.equal(harness.app.engine.tracker.calibrated, true);
  assert.equal(harness.app.output.mode, "PAUSE");
  assert.equal(harness.controls().at(-1).body.telemetry.operator_control.motion_mapping, "incremental");
  const recording = { ...healthyRuntime(), recording: true };
  harness.enqueue("/api/dual/health", harness.response(200, recording)); await harness.app.checkHealth();
  assert.equal(harness.element("motion-mapping").disabled, true);
  assert.equal(harness.element("motion-speed").disabled, true);
  harness.element("motion-mapping").value = "rate";
  harness.element("motion-mapping").dispatch("change");
  assert.equal(harness.app.engine.config.motion_mapping, "incremental");
  assert.equal(harness.element("motion-mapping").value, "incremental");
});

test("a 90 percent fist shows the other hand's unmet threshold instead of unexplained no gripper action", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await bindHands(harness);
  for (let i = 0; i < 8; i++) await deliverRecognition(harness, recognition("ILoveYou"));
  const fists = recognition("Closed_Fist", 0.9); fists.gestures[1][0].score = 0.79;
  for (let i = 0; i < 20; i++) await deliverRecognition(harness, fists);
  assert.match(harness.element("left-gesture").textContent, /握拳 · 90%/);
  assert.match(harness.element("gripper-detail").textContent, /右手评分 79% < 80%/);
  assert.equal(harness.controls().some((request) => request.body.command === "dual_gripper"), false);
});

test("the current terminal task state explains why confident fists cannot grip and how to reset", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await bindHands(harness);
  harness.enqueue("/api/dual/health", harness.response(200, { ...healthyRuntime(), task_state: "FAIL", failure_reason: "timeout" }));
  await harness.app.checkHealth();
  for (let i = 0; i < 30; i++) await deliverRecognition(harness, recognition("Closed_Fist", 0.9));
  assert.match(harness.element("gripper-detail").textContent, /超过正式任务时限.*重置场景/);
  assert.match(harness.element("notice").textContent, /本轮失败.*重置场景/);
  assert.equal(harness.controls().some((request) => request.body.command === "dual_gripper"), false);
});

test("failed episodes explain that closed grippers can still release and reflect the accepted opening", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await bindHands(harness);
  for (let i = 0; i < 8; i++) await deliverRecognition(harness, recognition("ILoveYou"));
  const state = { ...healthyRuntime(), task_state: "FAIL", failure_reason: "grasp_failure", gripper_latch: "CLOSED" };
  harness.enqueue("/api/dual/health", harness.response(200, state)); await harness.app.checkHealth();
  assert.match(harness.element("gripper-detail").textContent, /双侧抓取未建立.*仍可张掌松爪/);
  assert.match(harness.element("notice").textContent, /仍可双手张掌松爪/);
  const before = harness.controls().length;
  for (let i = 0; i < 25; i++) await deliverRecognition(harness, recognition("Open_Palm", 0.9));
  const releases = harness.controls().slice(before).filter((request) => request.body.command === "dual_gripper");
  assert.equal(releases.length, 1); assert.equal(releases[0].body.action, "open");
  assert.ok(harness.controls().slice(before).every((request) => request.body.command !== "dual_motion"));
  harness.enqueue("/api/dual/health", harness.response(200, { ...state, gripper_latch: "OPEN" }));
  await harness.app.checkHealth();
  assert.equal(harness.element("gripper").textContent, "双侧打开");
  assert.match(harness.element("gripper-detail").textContent, /禁止继续移动或抓取.*重置场景/);
});

test("a temporary physical release rejection shows the specific waiting reason and fresh retry evidence", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await bindHands(harness);
  for (let i = 0; i < 8; i++) await deliverRecognition(harness, recognition("ILoveYou"));
  const state = { ...healthyRuntime(), task_state: "DUAL_GRASPED", gripper_latch: "CLOSED", release_ready: false };
  harness.enqueue("/api/dual/health", harness.response(200, state)); await harness.app.checkHealth();
  for (let i = 0; i < 20; i++) await deliverRecognition(harness, recognition("Open_Palm", 0.9));
  const event = harness.controls().findLast((request) => request.body.command === "dual_gripper").body;
  harness.enqueue("/api/dual/health", harness.response(200, { ...state,
    last_command_result: { command: "dual_gripper", eventId: event.eventId, accepted: false, reason: "release_rejected_stop_before_opening" } }));
  await harness.app.checkHealth();
  assert.match(harness.element("gripper-detail").textContent, /等待物体与夹爪停稳，保持双掌/);
  assert.equal(harness.element("record-feedback").textContent, "", "gripper rejection belongs beside gripper status, not in the recording panel");
  for (let i = 0; i < 5; i++) await deliverRecognition(harness, recognition("Open_Palm", 0.9));
  assert.match(harness.element("notice").textContent, /等待物体与夹爪停稳/);
  assert.ok(harness.element("left-evidence").value > 0);
  assert.equal(harness.controls().filter((request) => request.body.command === "dual_gripper").length, 1, "retries cannot skip the new evidence interval");
  for (let i = 0; i < 15; i++) await deliverRecognition(harness, recognition("Open_Palm", 0.9));
  assert.equal(harness.controls().filter((request) => request.body.command === "dual_gripper").length, 2);
});

test("a non-retryable gripper rejection is explained without suggesting automatic release", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await bindHands(harness);
  for (let i = 0; i < 8; i++) await deliverRecognition(harness, recognition("ILoveYou"));
  const state = { ...healthyRuntime(), gripper_latch: "CLOSED" };
  harness.enqueue("/api/dual/health", harness.response(200, state)); await harness.app.checkHealth();
  for (let i = 0; i < 20; i++) await deliverRecognition(harness, recognition("Open_Palm", 0.9));
  const event = harness.controls().findLast((request) => request.body.command === "dual_gripper").body;
  harness.enqueue("/api/dual/health", harness.response(200, { ...state,
    last_command_result: { command: "dual_gripper", eventId: event.eventId, accepted: false, reason: "stale_command" } }));
  await harness.app.checkHealth();
  for (let i = 0; i < 25; i++) await deliverRecognition(harness, recognition("Open_Palm", 0.9));
  assert.match(harness.element("gripper-detail").textContent, /到达时已过期.*换手势再重试/);
  assert.equal(harness.controls().filter((request) => request.body.command === "dual_gripper").length, 1);
});

test("the control indicator needs fresh matching backend permission before declaring XY or Z available", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await bindHands(harness);
  for (let i = 0; i < 10; i++) await deliverRecognition(harness, recognition("ILoveYou"));
  const sync = async (extra = {}) => {
    harness.enqueue("/api/dual/health", harness.response(200, { ...healthyRuntime(), ...extra }));
    await harness.app.checkHealth();
  };
  await sync();
  assert.equal(harness.element("control-status-title").textContent, "等待控制同步");
  await sync({ control_mode: "XY", paused: false });
  assert.equal(harness.element("control-status-title").textContent, "XY 可平移");
  assert.equal(harness.element("control-status").dataset.state, "active");
  await sync({ control_mode: "PAUSE", paused: true, pause_reason: "command_watchdog" });
  assert.match(harness.element("control-status-title").textContent, /已暂停/);
  assert.match(harness.element("notice").textContent, /仿真未及时收到有效控制/);
  for (let i = 0; i < 10; i++) await deliverRecognition(harness, recognition("Victory"));
  await sync({ control_mode: "Z", paused: false });
  assert.equal(harness.element("control-status-title").textContent, "Z 可升降");
  for (let i = 0; i < 30; i++) await deliverRecognition(harness, recognition("Victory"));
  harness.app.renderState();
  assert.equal(harness.element("control-status-title").textContent, "等待仿真状态更新");
  assert.notEqual(harness.element("control-status").dataset.state, "active");
});

test("the control indicator explains lost hands, recovery and stale recognition despite visible gesture labels", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await harness.app.startCamera(); await settle();
  harness.app.renderState();
  assert.equal(harness.element("control-status-title").textContent, "等待绑定");
  for (let i = 0; i < 55; i++) await deliverRecognition(harness, recognition("Open_Palm"));
  for (let i = 0; i < 10; i++) await deliverRecognition(harness, recognition("ILoveYou"));
  harness.enqueue("/api/dual/health", harness.response(200, { ...healthyRuntime(), control_mode: "XY", paused: false }));
  await harness.app.checkHealth();
  assert.equal(harness.element("control-status-title").textContent, "XY 可平移");
  await deliverRecognition(harness, { landmarks: [], handedness: [], gestures: [] }); harness.app.renderState();
  assert.match(harness.element("control-status-title").textContent, /已暂停.*丢手/);
  assert.match(harness.element("notice").textContent, /至少一只手离开画面/);
  assert.equal(harness.element("control-gripper-state").textContent, "夹爪保持打开");
  await deliverRecognition(harness, recognition("ILoveYou")); harness.app.renderState();
  assert.equal(harness.element("control-status-title").textContent, "恢复追踪");
  assert.match(harness.element("notice").textContent, /恢复.*次/);
  for (let i = 0; i < 12; i++) await deliverRecognition(harness, recognition("ILoveYou"));
  await deliverRecognition(harness, recognition("ILoveYou"), 180); harness.app.renderState();
  assert.match(harness.element("left-gesture").textContent, /I Love You/);
  assert.equal(harness.element("control-status-title").textContent, "已暂停 · 识别结果过期");
  assert.match(harness.element("notice").textContent, /延迟 180 ms.*150 ms/);
});

test("low scores, mismatched and unassigned gestures explain paused control instead of labeling it as motion", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await bindHands(harness);
  for (let i = 0; i < 8; i++) await deliverRecognition(harness, recognition("ILoveYou"));
  for (let i = 0; i < 4; i++) await deliverRecognition(harness, recognition("ILoveYou", 0.7));
  harness.app.renderState();
  assert.equal(harness.element("control-status-title").textContent, "已暂停 · 评分不足");
  assert.match(harness.element("notice").textContent, /70% < 75%/);
  const mixed = recognition("ILoveYou"); mixed.gestures[1][0].categoryName = "Victory";
  for (let i = 0; i < 4; i++) await deliverRecognition(harness, mixed);
  harness.app.renderState();
  assert.match(harness.element("notice").textContent, /双手需要使用同一种移动手势/);
  for (let i = 0; i < 4; i++) await deliverRecognition(harness, recognition("Pointing_Up"));
  harness.app.renderState();
  assert.match(harness.element("notice").textContent, /当前手势未分配操作/);
  for (let i = 0; i < 4; i++) await deliverRecognition(harness, recognition("Thumb_Up"));
  harness.app.renderState();
  assert.match(harness.element("notice").textContent, /当前手势未分配操作/);
});

test("grip confirmation and guidance use different live opening and closing thresholds", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close()); await harness.boot();
  const state = { ...healthyRuntime(), config: { control: { ...DEFAULT_DUAL_CONFIG,
    gripper_open_confidence_min: 0.72, gripper_close_confidence_min: 0.9, max_command_age_ms: 150 } } };
  harness.enqueue("/api/dual/health", harness.response(200, state)); await harness.app.checkHealth();
  assert.match(harness.element("gripper-detail").textContent, /闭合各至少 90%.*打开各至少 72%/);
  assert.match(harness.element("gripper-guide").textContent, /闭合各至少 90%.*打开各至少 72%/);
  await bindHands(harness);
  for (let i = 0; i < 8; i++) await deliverRecognition(harness, recognition("ILoveYou"));
  for (let i = 0; i < 4; i++) await deliverRecognition(harness, recognition("Closed_Fist", 0.88));
  harness.app.renderState();
  assert.equal(harness.element("control-status-title").textContent, "已暂停 · 抓放评分不足");
  assert.match(harness.element("notice").textContent, /88% < 90%/);
  for (let i = 0; i < 20; i++) await deliverRecognition(harness, recognition("Closed_Fist", 0.95));
  harness.app.renderState();
  assert.equal(harness.element("control-status-title").textContent, "握拳确认");
  assert.equal(harness.element("control-gripper-state").textContent, "夹爪等待仿真确认");
  harness.enqueue("/api/dual/health", harness.response(200, { ...state, gripper_latch: "CLOSED" }));
  await harness.app.checkHealth();
  for (let i = 0; i < 4; i++) await deliverRecognition(harness, recognition("Open_Palm", 0.7));
  harness.app.renderState();
  assert.equal(harness.element("control-status-title").textContent, "已暂停 · 抓放评分不足");
  assert.match(harness.element("notice").textContent, /70% < 72%/);
  for (let i = 0; i < 6; i++) await deliverRecognition(harness, recognition("Open_Palm", 0.8));
  harness.app.renderState();
  assert.equal(harness.element("control-status-title").textContent, "松爪确认");
  assert.match(harness.element("notice").textContent, /打开：左手.*700 ms/);
});

test("recording status follows backend retention decisions while the operator is paused", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close()); await harness.boot();
  assert.equal(harness.element("control-recording-note").hidden, true);
  const state = { ...healthyRuntime(), recording: true, paused: true, control_mode: "PAUSE" };
  harness.enqueue("/api/dual/health", harness.response(200, { ...state,
    capture_status: { state: "stationary_pause", training_eligible: false, last_frame_recorded: false } }));
  await harness.app.checkHealth();
  assert.equal(harness.element("control-recording-note").hidden, false);
  assert.match(harness.element("control-recording-note").textContent, /静止暂停只留研究日志/);
  harness.enqueue("/api/dual/health", harness.response(200, { ...state,
    capture_status: { state: "physical_change", training_eligible: true, last_frame_recorded: true } }));
  await harness.app.checkHealth();
  assert.match(harness.element("control-status-title").textContent, /已暂停/);
  assert.match(harness.element("control-recording-note").textContent, /实际物理变化保留到训练数据/);
  harness.enqueue("/api/dual/health", harness.response(200, healthyRuntime())); await harness.app.checkHealth();
  assert.equal(harness.element("control-recording-note").hidden, true);
});

test("legacy config without split thresholds keeps the same displayed opening and closing threshold", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close()); await harness.boot();
  const control = { ...DEFAULT_DUAL_CONFIG, gripper_confidence_min: 0.81 };
  delete control.gripper_open_confidence_min; delete control.gripper_close_confidence_min;
  harness.enqueue("/api/dual/health", harness.response(200, { ...healthyRuntime(), config: { control } }));
  await harness.app.checkHealth();
  assert.match(harness.element("gripper-guide").textContent, /闭合各至少 81%.*打开各至少 81%/);
});

test("fresh recognition diagnostics are deduplicable observations without images or control authorization", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await harness.app.startCamera(); await settle();
  const frame = harness.workers[0].messages.find((message) => message.type === "frame");
  await deliverRecognition(harness, recognition("ILoveYou", 0.91, 0.97), 15);
  const first = harness.controls().findLast((request) => request.body.telemetry?.recognition_result).body.telemetry.recognition_result;
  assert.equal(first.accepted, true); assert.equal(first.drop_reason, null);
  assert.equal(first.captured_at_ms, frame.capturedEpochMs);
  assert.equal(first.received_at_ms - first.captured_at_ms, 15);
  assert.equal(first.inference_ms, 15); assert.equal(first.result_age_ms, 15);
  assert.equal(first.source_session, frame.session); assert.equal(first.source_control_epoch, frame.control_epoch);
  assert.match(first.frame_id, new RegExp(`:${frame.frameId}$`));
  assert.deepEqual(first.observed_hands[0], { label: "left", gesture: "ILoveYou", score: 0.91, identity_score: 0.97 });
  assert.deepEqual(Object.keys(first.observed_hands[1]).sort(), ["gesture", "identity_score", "label", "score"]);
  assert.equal(/landmarks|bitmap|image|wrist|anchor/.test(JSON.stringify(first)), false);
  assert.equal(harness.app.engine.tracker.calibrated, false, "accepted means fresh tracker input, not robot permission");
  assert.ok(harness.controls().every((request) => ["telemetry", "pause"].includes(request.body.command)));
  await deliverRecognition(harness, recognition("ILoveYou", 0.91, 0.97), 15);
  const second = harness.controls().findLast((request) => request.body.telemetry?.recognition_result).body.telemetry.recognition_result;
  assert.notEqual(second.frame_id, first.frame_id, "successive Worker requests must have different diagnostic identities");
});

test("an unmatched recognition reply logs a fallback id without freeing or replacing the in-flight control frame", async (t) => {
  const harness = makeHarness(); t.after(() => harness.close());
  await harness.boot(); await harness.app.startCamera(); await settle();
  const worker = harness.workers[0], frame = worker.messages.find((message) => message.type === "frame");
  worker.emit({ type: "result", capturedAt: frame.capturedAt, capturedEpochMs: frame.capturedEpochMs,
    control_epoch: frame.control_epoch, session: frame.session, inferenceMs: 10, result: recognition("Closed_Fist", 0.99) });
  await settle(); await harness.pump(40);
  const rejected = harness.controls().findLast((request) => request.body.telemetry?.recognition_result).body.telemetry.recognition_result;
  assert.equal(rejected.accepted, false); assert.equal(rejected.drop_reason, "unexpected_frame_id");
  assert.match(rejected.frame_id, /:capture-browser-regression-0-/);
  assert.notEqual(harness.app.engine.tracker.hands.left.gesture, "Closed_Fist");
  harness.element("webcam").currentTime += 0.04; await harness.animation();
  assert.equal(worker.messages.filter((message) => message.type === "frame").length, 1,
    "an unmatched reply cannot create a second in-flight camera request");
  await deliverRecognition(harness, recognition("Open_Palm"));
  const accepted = harness.controls().findLast((request) => request.body.telemetry?.recognition_result).body.telemetry.recognition_result;
  assert.equal(accepted.accepted, true);
  assert.ok(harness.controls().every((request) => ["telemetry", "pause"].includes(request.body.command)));
});
