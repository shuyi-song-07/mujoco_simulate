import {MeasurementJournal,controlObservation} from "./measurement-journal.js";
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { DEFAULT_DUAL_CONFIG, DualInteractionState, detectionsFromMediaPipe, resolveOperation } from "./dual-interaction-state.js";
import { handMeasurement } from "./hand-measurement.js";
import { ContinuousInteractionState } from "./continuous-interaction-state.js";
import { resultPair } from "../../../tests/task3/hand_fixture.mjs";
import { gestureResult } from "../../../tests/task3/gesture_hand_fixture.mjs";

// Run the actual browser entry point against controlled browser boundaries.
// The real gesture engine is injected; no copy of application logic is tested.
const applicationSource = readFileSync(new URL("./dual-arm-app.js", import.meta.url), "utf8")
  .replace(/^import\s+[\s\S]*?\s+from\s+["'][^"']+["'];\s*$/gm, "")
  .replaceAll("import.meta.url", '"http://127.0.0.1:8000/dual-arm-app.js"');

const cameraNames = ["top", "oblique", "front"];
const imageIds = ["main-preview", "auxiliary-preview-one", "auxiliary-preview-two"];
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

function makeHarness({ missingIds = new Set() } = {}) {
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
      return { clearRect() {}, save() {}, restore() {}, translate() {}, scale() {}, fillText() {}, drawImage() {}, beginPath() {}, ellipse() {}, stroke() {}, strokeRect() {}, moveTo() {}, lineTo() {} };
    }
    async play() {}
    async decode() {}
  }
  const getElement = (id) => {
    if (missingIds.has(id)) return null;
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
    const match = entry.url.match(/\/(top|oblique|front)-preview$/);
    if (match) return response(200, {}, match[1]);
    if (entry.url.endsWith("/measurements")) return response(200,{ok:true,last_sequence:entry.body.rows.at(-1).sequence});
    if (entry.url.endsWith("/control")) return response(200, { ok: true, queued: true });
    throw new Error(`Unexpected fetch: ${entry.url}`);
  };
  const cameraSettings={width:640,height:480,frameRate:30,resizeMode:"none"};
  const fakeStream = { getTracks: () => [{ stop() { trace.push({ type: "camera-stopped" }); } }], getVideoTracks:()=>[{getSettings:()=>cameraSettings}] };
  const sandbox = {
    console, MeasurementJournal, controlObservation, structuredClone, setInterval:()=>0, DEFAULT_DUAL_CONFIG, DualInteractionState, detectionsFromMediaPipe, resolveOperation, handMeasurement, ContinuousInteractionState,
    Date: BrowserDate, URL: BrowserURL, Image: BrowserImage, Worker: BrowserWorker,
    fetch, AbortSignal, Promise, performance: { now: () => now },
    navigator: { mediaDevices: { getSupportedConstraints:()=>({resizeMode:true}),async getUserMedia(constraints) { trace.push({type:"camera-request",constraints});return fakeStream; } } },
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
      measurementJournal, setConnected, postPacket, flushPackets, queueOutput, checkHealth, assignImage,
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
    app, elements, trace, requests, revoked, workers, response, cameraSettings,
    get now() { return now; }, advance(milliseconds) { now += milliseconds; },
    element: getElement,
    button(command) { return buttons.find((button) => button.dataset.command === command); },
    keyDown(code, target) { let prevented=false; windowListeners.get("keydown")?.({code,target,preventDefault(){prevented=true;}}); return prevented; },
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

async function deliverRecognition(harness, result, ageMs = 0, stepMs = 40) {
  const worker = harness.workers[0];
  const sent = worker.messages.filter((message) => message.type === "frame").at(-1);
  harness.advance(ageMs);
  worker.emit({ type: "result", frameId: sent.frameId, capturedAt: sent.capturedAt,
    capturedEpochMs: sent.capturedEpochMs, control_epoch: sent.control_epoch, session: sent.session, inferenceMs: ageMs || 10, result });
  await settle(); await harness.pump(stepMs);
  harness.element("webcam").currentTime += stepMs / 1000;
  await harness.animation();
}

async function calibratedContinuousCamera(harness) {
  await harness.boot();
  const state = { ...healthyRuntime(), gripper_latch: 'CONTINUOUS',
    grippers: {left: {command_width_m: .08}, right: {command_width_m: .08}},
    config: {control: {...DEFAULT_DUAL_CONFIG, max_command_age_ms:150}, contact_control:{enabled:true}} };
  harness.enqueue('/api/dual/health', harness.response(200, state));
  await harness.app.checkHealth();
  await harness.app.startCamera(); await settle();
  for (let i=0; i<53; i++) await deliverRecognition(harness, resultPair());
  for (let i=0; i<40; i++) await deliverRecognition(harness, resultPair([.15,.2]));
  for (let i=0; i<56; i++) await deliverRecognition(harness, resultPair());
  assert.equal(harness.app.engine.tracker.calibrated, true);
  assert.equal(harness.app.engine.apertureCalibration.phase, 'READY');
  assert.equal(harness.app.output.command?.command, 'continuous_input');
}

test("continuous edge input hides untrusted aperture, holds commands and reanchors after recovery and takeover", async(t) => {
  const harness=makeHarness();t.after(()=>harness.close());
  await calibratedContinuousCamera(harness);
  const held=[.028,.052];
  const grippers=Object.fromEntries(["left","right"].map((side,i)=>[side,{
    command_width_m:held[i],goal_width_m:held[i],requested_width_m:held[i],actual_width_m:held[i],
  }]));
  harness.enqueue("/api/dual/health",harness.response(200,{...harness.app.runtime,grippers}));
  await harness.app.checkHealth();
  const before=harness.controls().length;
  // The low ratio remains raw geometry. Real measurement logic must reject it
  // before it can close the gripper when the inferred wrist nears the bottom.
  const edge=resultPair([.25,.7]);
  for(const point of edge.landmarks[0])point.y+=.31;
  await deliverRecognition(harness,edge);
  harness.app.renderState();
  const measured=harness.app.engine.measurement.hands.find(h=>h.bound_side==="left");
  assert.equal(measured.valid,false);assert.equal(measured.invalid_reason,"hand_near_frame_edge");
  assert.ok(Math.abs(measured.aperture_ratio_image-.25)<1e-9);
  assert.equal(harness.app.output.reason,"hand_near_frame_edge");
  assert.match(harness.element("left-gesture").textContent,/边缘保护.*开度 暂不可用/);
  assert.equal(harness.element("left-evidence").hidden,true);
  assert.equal(harness.element("right-evidence").hidden,false);
  assert.ok(harness.controls().slice(before).some(r=>r.body.command==="pause"));
  assert.ok(harness.controls().slice(before).every(r=>["pause","telemetry"].includes(r.body.command)));
  for(let i=1;i<=5;i++) {
    await deliverRecognition(harness,resultPair(undefined,{dy:.03}),0,50);
    assert.notEqual(harness.app.output.command?.command,"continuous_input");
    assert.equal(harness.app.engine.tracker.recoveryPending,i<5);
  }
  assert.equal(harness.app.output.reason,"tracking_recovered");
  for(let i=0;i<4;i++)await deliverRecognition(harness,resultPair(undefined,{dy:.03}),0,50);
  assert.equal(harness.app.output.reason,"aperture_takeover");
  assert.ok(harness.controls().slice(before).every(r=>["pause","telemetry"].includes(r.body.command)));
  held.forEach((v,i)=>assert.equal(harness.app.runtime.grippers[["left","right"][i]].command_width_m,v));
  const matching=["left","right"].map((side,i)=>{
    const e=harness.app.engine.apertureCalibration.endpoints[side];
    return e.closed+(e.open-e.closed)*held[i]/.08;
  });
  await deliverRecognition(harness,resultPair(matching,{dy:.03}),0,50);
  assert.equal(harness.app.output.reason,"continuous_baseline");
  assert.notEqual(harness.app.output.command?.command,"continuous_input");
  assert.equal(harness.element("left-evidence").hidden,false);
  await deliverRecognition(harness,resultPair(matching,{dy:.03}),0,50);
  const resumed=harness.app.output.command;assert.equal(resumed.command,"continuous_input");
  for(const side of ["left","right"])assert.ok(Math.hypot(...resumed.delta_task_m[side])<1e-9);
  held.forEach((v,i)=>assert.ok(Math.abs(resumed.widths_m[i]-v)<1e-9));
  const sent=harness.controls().slice(before).filter(r=>r.body.command==="continuous_input");
  assert.equal(sent.length,1);
  held.forEach((v,i)=>assert.ok(Math.abs(sent[0].body.widths_m[i]-v)<1e-9));
});

for (const interruption of ['short_camera_stall', 'stale_recognition']) {
  test(`continuous ${interruption} requires five new samples spanning 200 ms before rebaselining`, async(t) => {
    const harness=makeHarness(); t.after(()=>harness.close());
    await calibratedContinuousCamera(harness);
    const countBefore=harness.controls().filter(r=>r.body.command==='continuous_input').length;
    if (interruption==='short_camera_stall') {
      harness.advance(260); await harness.animation();
      assert.equal(harness.app.output.reason,'camera_stalled');
      assert.equal(harness.app.engine.tracker.recoveryPending,true,
        '250-500 ms camera stall must restart recovery before any result returns');
      // Drain the pending old Worker reply so the next frame is freshly captured.
      await deliverRecognition(harness,resultPair());
    } else {
      await deliverRecognition(harness,resultPair(),180);
      assert.equal(harness.app.output.reason,'stale_recognition');
      assert.equal(harness.app.engine.tracker.recoveryPending,true);
    }
    for(let i=1;i<=5;i++) {
      await deliverRecognition(harness,resultPair(),0,50);
      assert.notEqual(harness.app.output.command?.command,'continuous_input');
      assert.equal(harness.app.engine.tracker.recoveryPending,i<5);
      assert.equal(harness.app.engine.tracker.recoveryCandidate.samples,i,
        'fresh samples must accumulate instead of restarting the recovery window');
    }
    assert.equal(harness.controls().filter(r=>r.body.command==='continuous_input').length,countBefore);
    assert.equal(harness.app.output.reason,'tracking_recovered');
    await deliverRecognition(harness,resultPair(),0,50);
    assert.equal(harness.app.output.reason,'continuous_baseline');
    await deliverRecognition(harness,resultPair(),0,50);
    assert.equal(harness.app.output.command?.command,'continuous_input');
    assert.equal(Math.hypot(...harness.app.output.command.delta_task_m.left),0);
  });
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
  const before = harness.app.previews.oblique;
  harness.advance(100);
  const held = harness.holdDecode("oblique");
  const pending = harness.app.pollPreviews(); await settle();
  const failedUrl = harness.trace.findLast((entry) => entry.type === "created" && entry.camera === "oblique").url;
  held.reject(new Error("corrupt JPEG")); await pending;
  assert.equal(harness.app.previews.oblique, before);
  assert.equal(harness.element("auxiliary-preview-one").hidden, false);
  assert.equal(harness.revoked.includes(before), false);
  assert.equal(harness.revoked.includes(failedUrl), true, "the rejected new blob must be released");
  harness.advance(901); harness.app.renderViews();
  assert.equal(harness.element("auxiliary-preview-one").hidden, true, "failed refresh must not renew stale-image lifetime");
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

test("an in-flight POST cannot let repeated pause telemetry erase the pending priority stop",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await h.boot();
  const blocked=deferred();h.enqueue("/api/dual/control",()=>blocked.promise);
  h.app.queueOutput(motion(.1),h.now,true);await settle();
  h.app.queueOutput(motion(.2),h.now,true);
  h.app.queueOutput(h.app.engine.pause("hand_lost"),h.now,true);
  const requestedAt=h.app.events[0].requestedAt;
  for(let i=0;i<8;i++){h.advance(10);h.app.queueOutput(h.app.engine.pause("hand_lost"),h.now,true);}
  assert.equal(h.app.events.length,1);assert.equal(h.app.events[0].command,"pause");
  blocked.resolve(h.response(200,{ok:true}));await settle();await h.pump();
  assert.equal(h.controls()[1].body.command,"pause");assert.equal(h.controls()[1].body.reason,"hand_lost");
  assert.equal(h.controls()[1].body.requestedAt,requestedAt);assert.ok(h.controls()[1].body.sentAt>requestedAt);
  assert.equal(h.controls().filter(r=>r.body.command==="dual_motion").length,1,"the unsent earlier motion must be discarded");
  assert.equal(h.controls().filter(r=>r.body.command==="pause").length,1,"unchanged pause reason coalesces");
});

test("a priority pause bypasses a motion throttle window and remains ahead of its later heartbeat",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await h.boot();
  h.app.queueOutput(motion(.1),h.now,true);await settle();h.advance(10);
  h.app.queueOutput(motion(.2),h.now,true);await settle();
  h.app.queueOutput(h.app.engine.pause("camera_stalled"),h.now,true);await settle();
  assert.equal(h.controls().at(-1).body.command,"pause");
  h.app.queueOutput(h.app.engine.pause("camera_stalled"),h.now,true);await h.pump(40);
  assert.equal(h.controls().filter(r=>r.body.command==="dual_motion").length,1);
  assert.equal(h.controls().filter(r=>r.body.command==="pause").length,1);
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
  assert.equal(harness.app.latestPacket, null);
  assert.equal(harness.app.events.some(packet=>packet.command==="pause"),true);
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
  assert.ok(harness.app.events.every(packet=>packet.command==="pause"&&packet.session==="backend-restarted"),"only the new session's priority stop may remain");
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
  assert.equal(harness.element("auxiliary-preview-one").src, harness.app.previews.top);
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
  assert.equal(/bitmap|data:image|camera_video/.test(JSON.stringify(first)), false);
  assert.equal(first.measurement.schema_version, "task3-hand-v1");
  assert.equal(first.measurement.measurement_id, first.frame_id);
  assert.equal(first.measurement.hands[0].valid, false, "wrist-only fixture is explicitly incomplete");
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

test("health and preview refresh cannot overwrite continuous support or raw timeline labels", async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await h.boot();
  h.app.engine.continuous=true;
  Object.assign(h.app.runtime,{recording:true,physical_contact:{support_ready:true},target_orientation_error_rad:.12});
  for(let i=0;i<3;i++) {
    h.app.setConnected(true);
    assert.match(h.element('control-gripper-state').textContent,/桌面支撑已稳定.*朝向差 6.9/);
    assert.match(h.element('control-recording-note').textContent,/完整 30 Hz 原始物理时间轴/);
    assert.doesNotMatch(h.element('control-gripper-state').textContent,/待确认|闭合|打开/);
  }
});

test("camera requests uncropped capture when supported and reports negotiated settings without claiming sensor coverage",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await h.boot();
  Object.assign(h.cameraSettings,{width:1280,height:720,frameRate:29.97});
  await h.app.startCamera();await settle();
  assert.equal(h.trace.find(row=>row.type==="camera-request").constraints.video.resizeMode,"none");
  assert.match(h.element("camera-format").textContent,/1280 × 720.*30 fps.*浏览器未裁剪/);
  assert.match(h.element("camera-format").textContent,/内框仅用于保护/);
  h.app.stopCamera();
  assert.equal(h.element("camera-format").textContent,"实际摄像头格式：未启动");
  delete h.cameraSettings.resizeMode;
  await h.app.startCamera();await settle();
  assert.match(h.element("camera-format").textContent,/裁剪信息未知/);
  assert.doesNotMatch(h.element("camera-format").textContent,/浏览器未裁剪/);
});

test("a disconnected backend clears robot telemetry while local hand measurements remain independently visible",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await calibratedContinuousCamera(h);
  h.enqueue("/api/dual/health",h.response(200,{...h.app.runtime,physical_contact:{support_ready:true},
    diagnostics:{written:83544,dropped:0,error:null},grippers:{left:{requested_width_m:.0132,command_width_m:.0168,actual_width_m:.0168},right:{}}}));
  await h.app.checkHealth();
  assert.match(h.element("gripper-detail").textContent,/请求 13.2 mm · 执行 16.8 mm · 实测 16.8 mm/);
  const hand=h.element("left-gesture").textContent;
  h.app.setConnected(false);
  assert.match(h.element("gripper-detail").textContent,/不可用/);
  assert.doesNotMatch(h.element("gripper-detail").textContent,/16.8/);
  assert.match(h.element("diagnostic-health").textContent,/不可用/);
  assert.doesNotMatch(h.element("diagnostic-health").textContent,/正常|83544/);
  assert.match(h.element("control-gripper-state").textContent,/支撑状态不可用/);
  assert.equal(h.element("left-gesture").textContent,hand,"local measurement is not robot telemetry");
  h.app.renderState();
  assert.equal(h.element("task-state").textContent,"—");
  assert.match(h.element("grasp-state").textContent,/不可用/);
});

test("partial holding and release confirmation expose accepted backend state separately from requested width",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await calibratedContinuousCamera(h);
  h.enqueue("/api/dual/health",h.response(200,{...h.app.runtime,physical_contact:{hold_width_m:{left:.023,right:null},
    release_reason:{left:"grasp_hold",right:null}},grippers:{left:{requested_width_m:.07,command_width_m:.023,actual_width_m:.026},right:{command_width_m:.08}}}));
  await h.app.checkHealth();
  assert.match(h.element("grasp-state").textContent,/左爪 · 抓持保持，可继续移动\n右爪 · 自由开合/);
  assert.match(h.element("gripper-detail").textContent,/请求 70.0 mm · 执行 23.0 mm · 实测 26.0 mm/);
  h.enqueue("/api/dual/health",h.response(200,{...h.app.runtime,physical_contact:{hold_width_m:{left:.023,right:.025},
    release_reason:{left:"release_confirmation",right:"release_requires_stable_table_support"},
    release_confirm_seconds:.7,release_progress_s:{left:.32,right:0}}}));
  await h.app.checkHealth();
  assert.match(h.element("grasp-state").textContent,/释放确认 0.3 \/ 0.7 s/);
  assert.match(h.element("grasp-state").textContent,/右爪 · 等待桌面稳定支撑/);
  h.enqueue("/api/dual/health",h.response(200,{...h.app.runtime,physical_contact:{hold_width_m:{left:null,right:null},
    release_reason:{left:"release_completed",right:"release_completed"},release_executed:{left:true,right:true},
    release_opening:{left:true,right:false}}}));
  await h.app.checkHealth();
  assert.match(h.element("grasp-state").textContent,/左爪 · 正在开爪，请保持张开并停手，直到完成\n右爪 · 释放已接受/);
  assert.match(h.element("operation-help").innerHTML,/持续保持到开爪完成/);
});

async function calibratedGestureCamera(h) {
  await h.boot();
  const runtime={...healthyRuntime(),gripper_latch:"CONTINUOUS",grippers:{left:{command_width_m:.08},right:{command_width_m:.08}},
    physical_contact:{control_protocol:"gesture-grasp-release-v2",hold_width_m:{left:null,right:null},operation_hold_width_m:{left:null,right:null},release_opening:{left:false,right:false},interaction_modes:{left:"HOLD",right:"HOLD"}},
    action_contract:{control_protocol:"gesture-grasp-release-v2"},config:{control:{...DEFAULT_DUAL_CONFIG},contact_control:{enabled:true}}};
  h.enqueue("/api/dual/health",h.response(200,runtime));await h.app.checkHealth();await h.app.startCamera();await settle();
  for(let i=0;i<53;i++)await deliverRecognition(h,gestureResult(["PALM","PALM"]));
  for(let i=0;i<40;i++)await deliverRecognition(h,gestureResult(["PINCH","PINCH"],{ratios:[.15,.15]}));
  for(let i=0;i<61;i++)await deliverRecognition(h,gestureResult(["PINCH","PINCH"]));
  assert.equal(h.app.engine.apertureCalibration.phase,"READY");
  assert.equal(h.app.output.command?.control_protocol,"gesture-grasp-release-v2");
  h.enqueue("/api/dual/health",h.response(200,{...h.app.runtime}));await h.app.checkHealth();
}

function measuredAperture(h,side,u) {
  const endpoints=h.app.engine.apertureCalibration.endpoints[side];
  h.app.engine.measurement.hands.find(hand=>hand.bound_side===side).aperture_ratio_image=endpoints.closed+u*(endpoints.open-endpoints.closed);
}

test("v2 UI and transport identify geometric None pinch, takeover and the new operating modes",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await calibratedGestureCamera(h);
  assert.deepEqual([...h.app.output.command.interaction_modes],["PINCH","PINCH"]);
  assert.match(h.element("left-interaction").textContent,/连续捏合.*已接管/);
  assert.match(h.element("left-interaction").textContent,/请求.*连续捏合.*仿真实际 保持/);
  assert.match(h.element("operation-help").innerHTML,/握拳.*锁住当前宽度/);
  assert.match(h.element("motion-detail").textContent,/三指伸展/);
  assert.doesNotMatch(h.element("motion-detail").textContent,/掌轴|横向|固定 XY/);
  const sent=h.controls().findLast(r=>r.body.command==="continuous_input").body;
  assert.equal(sent.control_protocol,"gesture-grasp-release-v2");
  assert.equal(sent.telemetry.pinch_mapping.version,"anchored-piecewise-v2");
  assert.equal(sent.telemetry.pinch_mapping.left.active,true);
});

test("v2 UI separates an empty operation lock from true holding and explains one-shot automatic opening",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await calibratedGestureCamera(h);
  h.enqueue("/api/dual/health",h.response(200,{...h.app.runtime,physical_contact:{control_protocol:"gesture-grasp-release-v2",hold_width_m:{left:null,right:.03},operation_hold_width_m:{left:.02,right:null},release_opening:{left:false,right:false}}}));
  await h.app.checkHealth();assert.match(h.element("grasp-state").textContent,/左爪 · 操作锁宽 20.0 mm · 无真实抓持锁/);
  assert.match(h.element("grasp-state").textContent,/右爪 · 真实抓持保护/);
  h.enqueue("/api/dual/health",h.response(200,{...h.app.runtime,physical_contact:{...h.app.runtime.physical_contact,release_opening:{left:true,right:true}}}));
  await h.app.checkHealth();await deliverRecognition(h,gestureResult(["PINCH_CLOSED","PINCH_CLOSED"],{ratios:[.2,.2]}));
  assert.match(h.element("grasp-state").textContent,/自动开爪.*无需持续张掌/);
  assert.deepEqual([...h.app.output.command.interaction_modes],["OPENING","OPENING"]);
  assert.deepEqual([...h.app.output.command.release_intent],[false,false]);
});

test("a dirty practice release cycle explains why recording needs a scene reset in Chinese",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await h.boot();
  h.app.setPendingOperation({command:"record_start",eventId:"dirty-start",startedAt:h.now});
  h.enqueue("/api/dual/health",h.response(200,{...h.app.runtime,last_command_result:{command:"record_start",eventId:"dirty-start",accepted:false,reason:"recording_requires_clean_control_cycle"}}));
  await h.app.checkHealth();
  assert.match(h.element("record-feedback").textContent,/练习已有释放授权.*重置场景.*开始录制/);
  assert.doesNotMatch(h.element("record-feedback").textContent,/recording_requires_clean_control_cycle/);
});

test("gesture waiting UI names manual pause, tracking recovery, wrist settling and uncertain hand shape",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await calibratedGestureCamera(h);
  const cases=[
    ["user_pause",/手动暂停/],
    ["tracking_recovery",/追踪恢复/],
    ["gesture_wrist_not_still",/停稳确认/],
    ["gesture_geometry_hold",/手形.*确认|手形.*可信/],
  ];
  h.app.engine.lastWireModes=["HOLD","HOLD"];
  for(const [reason,title] of cases){
    h.app.setOutput({...h.app.output,reason});h.app.renderState();
    assert.match(h.element("control-status-title").textContent,title,reason);
  }
});

test("a left pinch remains visible while only the right hand needs aperture matching",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await calibratedGestureCamera(h);
  h.app.engine.lastWireModes=["PINCH","HOLD"];
  h.app.runtime.physical_contact.interaction_modes.left="PINCH";
  h.app.runtime.physical_contact.aperture_direct_enabled={left:true,right:false};
  Object.assign(h.app.engine.interaction.right,{mode:"PINCH",candidate:"PINCH",takeover:true,moving:false,desired_width_m:.04,current_width_m:.08});
  measuredAperture(h,"right",.5);
  h.app.setOutput({...h.app.output,reason:"aperture_takeover"});h.app.renderState();
  assert.match(h.element("control-status-title").textContent,/左.*连续捏合/);
  assert.equal(h.element("control-status").dataset.state,"active");
  assert.match(h.element("right-interaction").textContent,/手 50%.*爪 100%.*差 50.*张大/);
  assert.doesNotMatch(h.element("left-interaction").textContent,/待匹配|手 \d+%.*爪/);
  assert.match(h.element("notice").textContent,/右.*匹配/);
});

test("aperture matching tells an unlocked side to narrow rather than open if its hand is too wide",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await calibratedGestureCamera(h);
  h.app.engine.lastWireModes=["HOLD","HOLD"];
  Object.assign(h.app.engine.interaction.left,{mode:"PINCH",candidate:"PINCH",takeover:true,moving:false,desired_width_m:.064,current_width_m:.032});
  h.app.runtime.grippers.left.command_width_m=.032;
  measuredAperture(h,"left",.8);
  h.app.setOutput({...h.app.output,reason:"aperture_takeover"});h.app.renderState();
  assert.match(h.element("control-status-title").textContent,/匹配开度/);
  assert.match(h.element("left-interaction").textContent,/手 80%.*爪 40%.*差 40.*收小/);
});

test("the currently rejected backend command outranks a benign front-end pinch baseline",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await calibratedGestureCamera(h);
  h.app.runtime.last_command_result={command:"continuous_input",eventId:"rejected-current",accepted:false,reason:"gesture_wrist_not_still"};
  h.app.runtime.paused=true;h.app.runtime.pause_reason="gesture_wrist_not_still";
  h.app.setOutput({...h.app.output,reason:"pinch_baseline"});h.app.renderState();
  assert.match(h.element("control-status-title").textContent,/停稳确认/);
  assert.match(h.element("notice").textContent,/后端|仿真/);
  assert.doesNotMatch(h.element("notice").textContent,/开度接管完成/);
  h.app.runtime.last_command_result={command:"continuous_input",eventId:"accepted-next",accepted:true};
  h.app.runtime.paused=false;h.app.runtime.control_mode="Z";
  h.app.setOutput({...h.app.output,reason:null});h.app.renderState();
  assert.match(h.element("control-status-title").textContent,/连续捏合/);
  assert.doesNotMatch(h.element("notice").textContent,/后端.*停稳确认/);
});

test("an old backend rejection does not claim the current state after health has stopped updating",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await calibratedGestureCamera(h);
  h.app.runtime.last_command_result={command:"continuous_input",eventId:"old",accepted:false,reason:"gesture_wrist_not_still"};
  h.advance(1100);h.app.renderState();
  assert.match(h.element("control-status-title").textContent,/等待仿真状态更新/);
  assert.doesNotMatch(h.element("notice").textContent,/开度接管完成|后端.*停稳确认/);
});

test("an operation lock at full width is explained without misleading takeover instructions",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await calibratedGestureCamera(h);
  h.app.runtime.physical_contact.operation_hold_width_m={left:.08,right:.08};
  h.app.engine.lastWireModes=["HOLD","HOLD"];
  h.app.setOutput({...h.app.output,reason:"operation_lock_hold"});h.app.renderState();
  assert.match(h.element("control-status-title").textContent,/握拳.*锁|操作锁/);
  assert.match(h.element("left-interaction").textContent,/握拳.*80\.0 mm.*全开.*完整张掌.*解锁/);
  assert.doesNotMatch(h.element("left-interaction").textContent,/待匹配|已接管/);
});

test("release waiting states show the actual per-side condition and confirmation progress",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await calibratedGestureCamera(h);
  Object.assign(h.app.runtime.physical_contact,{hold_width_m:{left:.02,right:.02},interaction_modes:{left:"RELEASE",right:"HOLD"},
    release_reason:{left:"release_requires_both_sides",right:"release_requires_stable_table_support"},release_progress_s:{left:.4,right:0}});
  h.app.engine.lastWireModes=["RELEASE","HOLD"];
  h.app.setOutput({...h.app.output,reason:"operation_lock_hold"});h.app.renderState();
  assert.match(h.element("control-status-title").textContent,/释放等待|释放确认/);
  assert.match(h.element("left-interaction").textContent,/等待.*另一侧|等待双侧/);
  assert.match(h.element("right-interaction").textContent,/桌面.*支撑/);
});

test("takeover UI reads current measured aperture, never a stale desired value from an earlier pinch",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await calibratedGestureCamera(h);
  h.app.engine.lastWireModes=["HOLD","HOLD"];
  Object.assign(h.app.engine.interaction.left,{mode:"PINCH",candidate:"PINCH",takeover:true,moving:false,desired_width_m:.008});
  measuredAperture(h,"left",.65);
  h.app.setOutput({...h.app.output,reason:"aperture_takeover"});h.app.renderState();
  assert.match(h.element("left-interaction").textContent,/手 65%.*爪 100%/);
  assert.doesNotMatch(h.element("left-interaction").textContent,/手 10%/);
  h.advance(200);h.app.renderState();
  assert.doesNotMatch(h.element("left-interaction").textContent,/手 65%/);
  assert.match(h.element("left-interaction").textContent,/等待可信.*读数/);
});

test("backend rejection keeps both sides blocked even when the front-end has an active pinch",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await calibratedGestureCamera(h);
  h.app.engine.lastWireModes=["PINCH","HOLD"];
  Object.assign(h.app.runtime.physical_contact,{interaction_modes:{left:"PINCH",right:"HOLD"},aperture_direct_enabled:{left:true,right:false}});
  h.app.runtime.last_command_result={command:"continuous_input",accepted:false,reason:"gesture_wrist_not_still"};
  h.app.setOutput({...h.app.output,reason:"aperture_takeover"});h.app.renderState();
  assert.equal(h.element("control-status").dataset.state,"paused");
  assert.match(h.element("left-interaction").textContent,/仿真当前拒绝.*尚不能开合/);
  assert.match(h.element("right-interaction").textContent,/当前请求未接受/);
  assert.doesNotMatch(h.element("left-interaction").textContent,/连续捏合已使能/);
});

async function boundRateCamera(h) {
  await h.boot();
  const runtime={...healthyRuntime(),gripper_latch:"CONTINUOUS",grippers:{left:{command_width_m:.08},right:{command_width_m:.08}},
    physical_contact:{control_protocol:"gesture-rate-gripper-v3",hold_width_m:{left:null,right:null},interaction_modes:{left:"HOLD",right:"HOLD"}},
    action_contract:{control_protocol:"gesture-rate-gripper-v3"},config:{control:{...DEFAULT_DUAL_CONFIG},recording:{schema_version:"task3-papera-v2"},contact_control:{enabled:true}}};
  h.enqueue("/api/dual/health",h.response(200,runtime));await h.app.checkHealth();await h.app.startCamera();await settle();
  for(let i=0;i<54;i++)await deliverRecognition(h,gestureResult(["PALM","PALM"]));
  h.enqueue("/api/dual/health",h.response(200,{...h.app.runtime}));await h.app.checkHealth();
  assert.equal(h.app.engine.tracker.calibrated,true);
}
async function boundLevelRateCamera(h) {
  await boundRateCamera(h);
  const runtime={...h.app.runtime,physical_contact:{...h.app.runtime.physical_contact,gripper_trigger_policy:"palm_fist_level_v2"},
    action_contract:{control_protocol:"gesture-rate-gripper-v3",gesture_parameters:{mode_confirm_ms:100}}};
  h.enqueue("/api/dual/health",h.response(200,runtime));await h.app.checkHealth();
}

test("bound solo operator can start with lost hands and free partly closed jaws",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundLevelRateCamera(h);
  Object.assign(h.app.runtime,{contacts:{left_fingers:[false,false],right_fingers:[false,false]}});
  Object.assign(h.app.runtime.physical_contact,{state:"FREE",ever_dual:false});
  h.app.runtime.grippers.right.command_width_m=.04672;
  await deliverRecognition(h,{gestures:[],handedness:[],landmarks:[],worldLandmarks:[]});
  h.app.renderState();
  assert.equal(h.app.engine.tracker.calibrated,true);
  assert.equal(h.button("record_start").disabled,false);
  h.button("record_start").dispatch("click");await settle();
  assert.ok(h.controls().some(r=>r.body.command==="record_start"));
});

test("DONE health pauses control immediately without another camera frame",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundLevelRateCamera(h);
  for(let i=0;i<8;i++)await deliverRecognition(h,gestureResult(["FIST","FIST"]));
  h.enqueue("/api/dual/health",h.response(200,{...h.app.runtime,task_state:"DONE",recording:true}));
  await h.app.checkHealth();
  assert.equal(h.app.output.reason,"task_done");
  assert.equal(h.app.output.mode,"PAUSE");
  assert.equal(h.button("record_save").disabled,false);
});
test("current-shape page displays 100 ms stable intent and independent release without old edge or support instructions",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundLevelRateCamera(h);
  for(let i=0;i<8;i++)await deliverRecognition(h,gestureResult(["FIST","FIST"]));
  await deliverRecognition(h,gestureResult(["PALM","FIST"]));h.app.renderState();
  assert.match(h.element("left-interaction").textContent,/稳定 100 ms.*保持/);
  for(let i=0;i<4;i++)await deliverRecognition(h,gestureResult(["PALM","FIST"]));
  h.enqueue("/api/dual/health",h.response(200,{...h.app.runtime,physical_contact:{...h.app.runtime.physical_contact,
    interaction_modes:{left:"Z_OPEN",right:"Z_CLOSE"},hold_width_m:{left:.025,right:.025},
    release_reason:{left:"release_opening",right:"grasp_hold"},release_opening:{left:true,right:false}}}));await h.app.checkHealth();
  assert.match(h.element("left-interaction").textContent,/匀速开爪.*独立释放.*无需额外释放确认/);
  assert.match(h.element("grasp-state").textContent,/独立释放/);
  assert.match(h.element("motion-detail").textContent,/稳定 100 ms.*左右独立/);
  assert.match(h.element("control-gripper-state").textContent,/左右独立/);
  assert.match(h.element("operation-help").innerHTML,/一侧张掌即可释放对应爪/);
  assert.doesNotMatch(h.element("operation-help").innerHTML,/700|掌→拳|拳→掌|同一手形不重复|支撑.*才打开|停手.*才打开/);
  assert.doesNotMatch(h.element("gripper-guide").textContent,/放稳停手|确认后释放/);
  const packet=h.controls().findLast(r=>r.body.command==="continuous_input").body;
  assert.deepEqual(packet.interaction_modes,["Z_OPEN","Z_CLOSE"]);assert.deepEqual(packet.release_intent,[true,false]);
});
test("current-shape help uses the active confirmation value instead of a fixed 100 or 200 ms",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundLevelRateCamera(h);
  h.app.engine.gestureParameters.mode_confirm_ms=0;h.app.renderState();
  assert.match(h.element("motion-detail").textContent,/识别后直接进入手势模式/);
  assert.match(h.element("operation-help").innerHTML,/识别后直接进入手势模式/);
  assert.doesNotMatch(h.element("operation-help").innerHTML,/新手势稳定 100 ms|新手势确认 200 ms|700 ms/);
});
test("v3 page hides aperture controls and permits recording after binding without pretending aperture calibration succeeded",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundRateCamera(h);
  assert.match(h.element("calibration-state").textContent,/身份已绑定.*无需开度标定/);
  assert.equal(h.app.engine.apertureCalibration.phase,"CLOSE");
  assert.equal(h.element("recalibrate-aperture").hidden,true);assert.equal(h.element("left-evidence").hidden,true);
  assert.equal(h.button("record_start").disabled,false);
  assert.match(h.element("control-status-title").textContent,/张掌保护/);
  assert.match(h.element("notice").textContent,/双手先换.*I Love You.*握拳/);
  assert.doesNotMatch(h.element("left-gesture").textContent,/待标定|开度/);
  assert.match(h.element("left-gesture").textContent,/张掌.*评分/);
  assert.doesNotMatch(h.element("operation-help").innerHTML,/三指|接管|锁住当前宽度|连续捏合/);
});
test("v3 transport and cards expose independent Z open and close without wrist stop or finger-distance matching",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundRateCamera(h);
  for(let i=0;i<9;i++)await deliverRecognition(h,gestureResult(["FIST","FIST"]));
  for(let i=0;i<9;i++)await deliverRecognition(h,gestureResult(["PALM","FIST"]));
  h.enqueue("/api/dual/health",h.response(200,{...h.app.runtime,physical_contact:{...h.app.runtime.physical_contact,interaction_modes:{left:"Z_OPEN",right:"Z_CLOSE"}}}));await h.app.checkHealth();
  assert.match(h.element("control-status-title").textContent,/Z 升降.*张掌开.*握拳闭/);
  assert.match(h.element("left-interaction").textContent,/匀速开爪.*Z/);
  assert.match(h.element("right-interaction").textContent,/匀速闭爪/);
  const packet=h.controls().findLast(r=>r.body.command==="continuous_input").body;
  assert.equal(packet.control_protocol,"gesture-rate-gripper-v3");
  assert.deepEqual(packet.interaction_modes,["Z_OPEN","Z_CLOSE"]);assert.deepEqual(packet.widths_m,[.08,0]);
  assert.deepEqual(packet.telemetry.aperture_calibration,{phase:"NOT_REQUIRED",reason:"categorical_rate_control"});
  assert.equal(packet.telemetry.pinch_mapping,null);
  assert.equal(packet.telemetry.recognition_result.measurement.hands[0].image_landmarks_21.length,21);
});
test("v3 real grasp release waiting reports support instead of stale pinch takeover guidance",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundRateCamera(h);
  for(let i=0;i<9;i++)await deliverRecognition(h,gestureResult(["FIST","FIST"]));
  for(let i=0;i<9;i++)await deliverRecognition(h,gestureResult(["PALM","PALM"]));
  h.enqueue("/api/dual/health",h.response(200,{...h.app.runtime,physical_contact:{...h.app.runtime.physical_contact,
    interaction_modes:{left:"Z_OPEN",right:"Z_OPEN"},hold_width_m:{left:.025,right:.025},release_reason:{left:"release_requires_stable_table_support",right:"release_requires_both_sides"}}}));await h.app.checkHealth();
  assert.match(h.element("control-status-title").textContent,/开爪等待支撑/);
  assert.match(h.element("left-interaction").textContent,/Z 下降.*桌面稳定支撑/);
  assert.match(h.element("right-interaction").textContent,/另一侧.*张掌/);
  assert.doesNotMatch(h.element("notice").textContent,/匹配|接管|标定/);
});
test("v3 manual pause and recovery never ask for finger aperture matching",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundRateCamera(h);
  h.app.engine.pause("user_pause");h.app.setOutput(h.app.engine.pause("user_pause"));h.app.renderState();
  assert.match(h.element("control-status-title").textContent,/手动暂停/);
  assert.match(h.element("notice").textContent,/无需匹配开度/);
  assert.equal(h.element("pause-button").textContent,"恢复控制");
});

test("v3 insufficient classification explains the actual per-side threshold and hidden calibration cannot pause it",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundRateCamera(h);
  for(let i=0;i<9;i++)await deliverRecognition(h,gestureResult(["FIST","FIST"]));
  for(let i=0;i<2;i++)await deliverRecognition(h,gestureResult(["PALM","FIST"],{scores:[.49,.79]}));
  assert.match(h.element("left-interaction").textContent,/张掌评分 49%.*至少 50%/);
  assert.match(h.element("right-interaction").textContent,/握拳评分 79%.*至少 80%/);
  const reason=h.app.output.reason;
  h.element("recalibrate-aperture").dispatch("click");
  assert.equal(h.app.output.reason,reason);
});

test("v3 interrupted release width is a protective hold, never evidence of a real stable grasp",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundRateCamera(h);
  for(let i=0;i<9;i++)await deliverRecognition(h,gestureResult(["FIST","FIST"]));
  h.enqueue("/api/dual/health",h.response(200,{...h.app.runtime,contacts:{left:false,right:true},physical_contact:{...h.app.runtime.physical_contact,
    interaction_modes:{left:"Z_CLOSE",right:"Z_CLOSE"},hold_width_m:{left:.052,right:.025},stable_grasps:{left:false,right:true},
    rate_interrupted_release_hold:{left:true,right:false},release_reason:{left:"release_input_interrupted",right:"grasp_hold"}}}));await h.app.checkHealth();
  assert.match(h.element("left-interaction").textContent,/释放已中止.*爪宽保持.*握拳.*重夹.*张掌.*确认/);
  assert.doesNotMatch(h.element("left-interaction").textContent,/已真实抓稳|真实双指抓稳|保持接触爪宽/);
  assert.match(h.element("right-interaction").textContent,/已真实抓稳/);
  const rows=h.element("grasp-state").textContent.split("\n");
  assert.match(rows[0],/释放已中止.*爪宽保持/);assert.doesNotMatch(rows[0],/真实双指抓稳/);
  assert.match(rows[1],/真实双指抓稳/);
  assert.match(h.element("gripper-detail").textContent,/左爪.*无双指接触/);
});

test("v3 retained grasp latch without present stable contact is shown as contact recovery",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundRateCamera(h);
  for(let i=0;i<9;i++)await deliverRecognition(h,gestureResult(["FIST","FIST"]));
  h.enqueue("/api/dual/health",h.response(200,{...h.app.runtime,contacts:{left:false,right:true},physical_contact:{...h.app.runtime.physical_contact,
    interaction_modes:{left:"Z_CLOSE",right:"Z_CLOSE"},hold_width_m:{left:.025,right:.025},stable_grasps:{left:false,right:false},
    rate_interrupted_release_hold:{left:false,right:false}}}));await h.app.checkHealth();
  for(const side of ["left","right"]) {
    assert.match(h.element(`${side}-interaction`).textContent,/接触.*恢复|接触.*确认/);
    assert.doesNotMatch(h.element(`${side}-interaction`).textContent,/已真实抓稳/);
  }
  assert.doesNotMatch(h.element("grasp-state").textContent,/真实双指抓稳/);
});

test("v3 release UI distinguishes table support blockers and server-clock confirmation",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundRateCamera(h);
  for(let i=0;i<9;i++)await deliverRecognition(h,gestureResult(["FIST","FIST"]));
  for(let i=0;i<9;i++)await deliverRecognition(h,gestureResult(["PALM","PALM"]));
  const contact={...h.app.runtime.physical_contact,interaction_modes:{left:"Z_OPEN",right:"Z_OPEN"},
    hold_width_m:{left:.025,right:.025},support_ready:false,release_clock:"server_monotonic",
    release_reason:{left:"release_requires_stable_table_support",right:"release_requires_stable_table_support"},
    release_progress_s:{left:0,right:0},release_confirm_seconds:.7,table_support_force_n:22,bottom_height_m:-.006,
    support_checks:{table_load:true,clearance:true,penetration:false,object_linear_speed:true,object_angular_speed:true,ee_speed:true,stable_duration:false},
    support_blockers:["excessive_table_penetration"],
    support_requirements:{minimum_table_force_n:.8,minimum_bottom_height_m:-.005,maximum_bottom_height_m:.003,
      max_object_linear_speed_mps:.08,max_object_angular_speed_radps:.4,max_ee_speed_mps:.1,stable_seconds:.12},
    support_measurements:{object_linear_speed_mps:0,object_angular_speed_radps:0,max_ee_speed_mps:0,stable_seconds:0}};
  h.enqueue("/api/dual/health",h.response(200,{...h.app.runtime,physical_contact:contact}));await h.app.checkHealth();
  assert.match(h.element("left-interaction").textContent,/压入桌面过深.*6.0.*5.0/);
  assert.match(h.element("grasp-state").textContent,/压入桌面过深/);
  assert.doesNotMatch(h.element("control-gripper-state").textContent,/支撑已稳定/);
  for(const [code,expected] of [["insufficient_table_support",/桌面承载不足/],["object_above_table",/还高于桌面/],
    ["object_linear_motion",/物体仍在平移/],["object_angular_motion",/物体仍在转动/],
    ["end_effector_motion",/夹爪仍在运动/],["support_stabilizing",/支撑稳定计时.*物理/]]) {
    contact.support_blockers=[code];h.app.renderState();assert.match(h.element("left-interaction").textContent,expected,code);
  }
  Object.assign(contact,{support_ready:true,support_blockers:[],release_reason:{left:"release_confirmation",right:"release_confirmation"},release_progress_s:{left:.4,right:.4}});
  h.app.renderState();
  assert.match(h.element("control-status-title").textContent,/释放确认/);
  assert.match(h.element("left-interaction").textContent,/0.40 \/ 0.70 s.*实际时间/);
  assert.match(h.element("grasp-state").textContent,/0.40 \/ 0.70 s.*实际时间/);
  assert.match(h.element("control-gripper-state").textContent,/仍需.*张掌.*停稳/);
  contact.release_clock="simulation";h.app.renderState();
  assert.match(h.element("left-interaction").textContent,/仿真时间/);
  assert.doesNotMatch(h.element("left-interaction").textContent,/实际时间/);
});

test("v3 fresh empty frames show missing hands instead of retained tracker classifications",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundRateCamera(h);
  for(let i=0;i<9;i++)await deliverRecognition(h,gestureResult(["FIST","FIST"]));
  await deliverRecognition(h,{landmarks:[],handedness:[],gestures:[]});h.app.renderState();
  assert.equal(h.app.engine.tracker.hands.left.gesture,"Closed_Fist","tracker retains history for diagnostics");
  assert.equal(h.element("left-gesture").textContent,"未检测到");
  assert.equal(h.element("right-gesture").textContent,"未检测到");
  assert.match(h.element("left-interaction").textContent,/未检测到.*保持/);
  assert.match(h.element("control-status-title").textContent,/丢手暂停/);
});

test("v3 expired recognition never presents retained confidence as current trusted measurement",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundRateCamera(h);
  for(let i=0;i<9;i++)await deliverRecognition(h,gestureResult(["FIST","FIST"]));
  assert.match(h.element("left-gesture").textContent,/可信.*评分/);
  await deliverRecognition(h,gestureResult(["PALM","PALM"]),170);h.app.renderState();
  assert.match(h.element("left-gesture").textContent,/过期|等待新/);
  assert.doesNotMatch(h.element("left-gesture").textContent,/可信|评分/);
  assert.doesNotMatch(h.element("left-identity").textContent,/%/);
  assert.match(h.element("left-interaction").textContent,/等待新.*保持/);
});

test("v3 measurement expiry shows waiting before the longer camera-stall grace elapses",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundRateCamera(h);
  for(let i=0;i<9;i++)await deliverRecognition(h,gestureResult(["FIST","FIST"]));
  h.advance(160);h.app.renderState();
  assert.deepEqual([...h.app.engine.lastWireModes],["Z_CLOSE","Z_CLOSE"],"display expiry does not modify control state");
  assert.match(h.element("left-interaction").textContent,/等待新.*保持/);
  assert.match(h.element("control-status-title").textContent,/等待新.*画面/);
  assert.equal(h.element("control-status").dataset.state,"waiting");
});

test("collection UI distinguishes current layout seed from session seed and next-layout reset",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundRateCamera(h);
  Object.assign(h.app.runtime,{session_seed:123,seed:126,scene_index:3});h.app.renderState();
  assert.match(h.element("scenario-name").textContent,/第 4 组摆放.*种子 126/);
  assert.match(h.element("scene-reset-help").textContent,/下一摆放.*会话种子 123.*当前摆放种子 126/);
  h.app.setPendingOperation({command:"reset",eventId:"next-layout",startedAt:h.now});
  h.enqueue("/api/dual/health",h.response(200,{...h.app.runtime,seed:127,scene_index:4,control_epoch:1,
    last_command_result:{command:"reset",eventId:"next-layout",accepted:true}}));await h.app.checkHealth();
  assert.match(h.element("scenario-name").textContent,/第 5 组摆放.*种子 127/);
  assert.match(h.element("record-feedback").textContent,/下一摆放/);
  h.app.setConnected(false);h.app.renderState();
  assert.match(h.element("scene-reset-help").textContent,/仿真未连接/);
});

async function boundEdgeCamera(h) {
  await boundRateCamera(h);
  for(let i=0;i<9;i++)await deliverRecognition(h,gestureResult(["FIST","FIST"]));
  const edge=()=>({baseline_mode:"Z_CLOSE",baseline_source:"accepted_mode",authorized_mode:null,event_id:null,edge_count:1,state:"READY",reason:null});
  h.enqueue("/api/dual/health",h.response(200,{...h.app.runtime,physical_contact:{...h.app.runtime.physical_contact,
    gripper_trigger_policy:"palm_fist_edge_v1",interaction_modes:{left:"Z_CLOSE",right:"Z_CLOSE"},gripper_edge:{left:edge(),right:edge()}}}));
  await h.app.checkHealth();
}

test("recording lifecycle replay explains grey save buttons after an unrecorded placement",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundEdgeCamera(h);
  const fixture=JSON.parse(readFileSync(new URL("../../../tests/task3/fixtures/recording_lifecycle_20261015.json",import.meta.url),"utf8"));
  for(const sample of fixture.samples) {
    Object.assign(h.app.runtime,sample,{saving:false,last_command_result:null});
    h.app.engine.pause("hand_lost");h.app.renderState();
    for(const command of ["record_save","record_failure","record_discard"])
      assert.equal(h.button(command).disabled,!sample.recording,`${sample.label}: ${command}`);
    if(sample.label==="practice_done") {
      assert.equal(h.button("reset").disabled,false);
      assert.match(h.element("recording-state").textContent,/练习.*未录制/);
      assert.equal(h.element("control-recording-note").hidden,false);
      assert.match(h.element("control-recording-note").textContent,/本轮未录制.*无法保存.*重置.*开始录制/);
      assert.match(h.button("record_save").title,/本轮未录制/);
      assert.equal(h.element("control-status-title").textContent,"练习放置完成");
      assert.match(h.element("notice").textContent,/本轮未录制.*重置.*开始录制/);
      assert.doesNotMatch(h.element("notice").textContent,/可以保存/);
    } else if(sample.label==="recorded_done") {
      assert.equal(h.element("control-status-title").textContent,"放置完成");
      assert.match(h.element("notice").textContent,/可以保存/);
    }
  }
});

test("recorded DONE save controls recover from transport errors and operation ACK waiting",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundEdgeCamera(h);
  const done={...h.app.runtime,recording:true,saving:false,task_state:"DONE",frames:872,
    episode_end:{version:"episode-done-boundary-v1",reason:"task_done",simulation_time:20,wall_time_ms:Date.now()}};
  const refresh=async()=>{h.enqueue("/api/dual/health",h.response(200,done));await h.app.checkHealth();await settle();h.app.renderState();};
  const allEnabled=()=>{for(const command of ["record_save","record_failure","record_discard"])assert.equal(h.button(command).disabled,false,command);};
  await refresh();h.app.engine.pause("hand_lost");h.app.renderState();allEnabled();
  assert.match(h.element("recording-state").textContent,/已截止.*待保存/);
  assert.match(h.element("control-recording-note").textContent,/截止时刻之前/);
  h.enqueue("/api/dual/control",h.response(409,{ok:false,error:"stale_epoch"}));
  h.app.queueOutput(h.app.engine.pause("hand_lost"),h.now,true);await settle();h.app.renderState();
  assert.equal(h.app.transportFrozen,true);
  assert.match(h.button("record_save").title,/控制通道正在恢复/);
  assert.equal(h.element("control-status-title").textContent,"放置完成 · 控制通道恢复中");
  assert.doesNotMatch(h.element("notice").textContent,/可以保存本轮/);
  await refresh();assert.equal(h.app.transportFrozen,false);allEnabled();
  h.app.setPendingOperation({command:"record_start",eventId:"missed-start-ack",startedAt:h.now,queued:true});
  h.app.renderState();assert.equal(h.button("record_save").disabled,true);
  assert.match(h.button("record_save").title,/等待仿真确认/);
  assert.equal(h.element("control-status-title").textContent,"放置完成 · 等待操作确认");
  h.advance(15001);await refresh();
  assert.equal(h.app.pendingOperation,null);allEnabled();
  done.task_state="PREGRASP";await refresh();
  assert.equal(h.button("record_save").disabled,true);
  assert.match(h.button("record_save").title,/尚未判定成功/);
  assert.equal(h.button("record_failure").disabled,false);
  assert.equal(h.button("record_discard").disabled,false);
  done.saving=true;await refresh();
  for(const command of ["record_save","record_failure","record_discard"]) {
    assert.equal(h.button(command).disabled,true);
    assert.match(h.button(command).title,/正在保存和生成视频/);
  }
});

test("a save ACK overwritten by hand input still resolves immediately and explains the saved episode",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundEdgeCamera(h);
  Object.assign(h.app.runtime,{recording:true,saving:false,task_state:"DONE",frames:1117});
  h.app.renderState();h.button("record_save").dispatch("click",{detail:1,pointerType:"mouse",isTrusted:true});await settle();
  await deliverRecognition(h,gestureResult(["PALM","PALM"]));
  const request=h.controls().find(r=>r.body.command==="record_save").body;
  assert.equal(request.ui_trigger.click_detail,1);
  assert.equal(request.ui_trigger.pointer_type,"mouse");
  assert.equal(request.ui_trigger.recording,true);
  const eventId=h.app.pendingOperation.eventId;
  const saved={command:"record_save",eventId,accepted:true,frames:1117,
    control_epoch:0,completed_control_epoch:1,recording:false,saved_success:1};
  const health={...h.app.runtime,recording:false,saving:false,task_state:"PREGRASP",control_epoch:1,
    last_command_result:{command:"continuous_input",eventId:"new-hand-input",accepted:true},
    last_operation_result:saved,operation_results:[saved]};
  h.enqueue("/api/dual/health",h.response(200,health));await h.app.checkHealth();
  assert.equal(h.app.pendingOperation,null,"hand packets must not hide a completed save ACK until the 15-second timeout");
  assert.match(h.element("record-feedback").textContent,/已保存.*1117.*新场景.*未录制/);
  assert.match(h.element("control-recording-note").textContent,/上一轮.*已保存.*1117/);
  assert.equal(h.button("reset").disabled,false);
  assert.equal(h.app.runtime.recording,false,"do not silently record the next scene");
});

test("Space on a focused save button pauses instead of triggering a native save click",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundEdgeCamera(h);
  Object.assign(h.app.runtime,{recording:true,saving:false,task_state:"DONE",frames:1117});h.app.renderState();
  const button=h.button("record_save"),prevented=h.keyDown("Space",button);
  // Native button activation follows unless keydown cancels the browser default.
  if(!prevented&&!button.disabled)button.dispatch("click",{detail:0,isTrusted:true});
  await settle();
  assert.equal(prevented,true,"the documented pause shortcut must also win while a button has focus");
  assert.equal(h.controls().some(r=>r.body.command==="record_save"),false);
  assert.equal(h.app.runtime.recording,true);
  assert.equal(h.app.engine.manualPaused,true);
});

test("Enter can still intentionally activate a save and its source is recorded",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundEdgeCamera(h);
  Object.assign(h.app.runtime,{recording:true,saving:false,task_state:"DONE",frames:1117});h.app.renderState();
  assert.equal(h.keyDown("Enter",h.button("record_save")),false);
  h.button("record_save").dispatch("click",{detail:0,isTrusted:true});await settle();
  await deliverRecognition(h,gestureResult(["PALM","PALM"]));
  const request=h.controls().find(r=>r.body.command==="record_save").body;
  assert.equal(request.ui_trigger.recent_key,"Enter");
  assert.equal(request.ui_trigger.click_detail,0);
});

test("edge v2 UI distinguishes suspended intent, single finger pressure and failed empty closure",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundEdgeCamera(h);
  const c=h.app.runtime.physical_contact;c.gripper_trigger_policy="palm_fist_edge_v2";
  c.gripper_edge.left={...c.gripper_edge.left,state:"HOLD",authorized_mode:"Z_CLOSE"};
  c.stable_grasps={left:false,right:false};
  h.app.runtime.contacts={left:false,right:false,left_fingers:[true,false],right_fingers:[false,false],finger_normal_force_n:{left:[1.54,0],right:[0,0]}};
  h.app.renderState();
  assert.match(h.element("grasp-state").textContent,/未完成的开合已保留.*恢复原手形后继续/);
  assert.match(h.element("gripper-detail").textContent,/仅单指接触，未抓稳.*1.54 \/ 0.00 N/);
  c.gripper_edge.left={...c.gripper_edge.left,state:"COMPLETE",authorized_mode:null};h.app.renderState();
  assert.match(h.element("grasp-state").textContent,/闭合行程已结束但未确认抓稳/);
  c.stable_grasps.left=true;c.hold_width_m={left:.02,right:null};h.app.runtime.contacts.left=true;h.app.renderState();
  assert.match(h.element("gripper-detail").textContent,/真实双指抓稳/);
  assert.doesNotMatch(h.element("grasp-state").textContent,/闭合行程已结束但未确认抓稳/);
});

test("edge-trigger UI never claims Z_OPEN is opening without an executor edge authorization",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundEdgeCamera(h);
  for(let i=0;i<9;i++)await deliverRecognition(h,gestureResult(["PALM","PALM"]));
  const c=h.app.runtime.physical_contact;
  for(const side of ["left","right"]){c.interaction_modes[side]="Z_OPEN";c.gripper_edge[side].baseline_mode="Z_OPEN";}
  h.app.renderState();
  assert.match(h.element("left-interaction").textContent,/请求 Z · 张掌.*仿真实际 Z · 张掌.*同一手形不开合/);
  assert.doesNotMatch(h.element("left-interaction").textContent,/匀速开爪|正在.*开爪/);
  assert.match(h.element("grasp-state").textContent,/同一手形不开合/);
  assert.equal(h.element("control-status-title").textContent,"Z 升降 · 爪宽保持");
  assert.match(h.element("operation-help").innerHTML,/掌→拳.*拳→掌.*同一手形不重复/);
  assert.doesNotMatch(h.element("motion-detail").textContent,/张掌：匀速开爪/);
});

test("edge-trigger UI explains recovery baseline and distinguishes abort, completion and execution",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundEdgeCamera(h);
  const c=h.app.runtime.physical_contact;
  c.gripper_edge.left={...c.gripper_edge.left,baseline_mode:null,state:"WAIT_BASELINE"};h.app.renderState();
  assert.match(h.element("left-interaction").textContent,/首次可信张掌或握拳只建立基准.*不开合/);
  assert.equal(h.element("control-status-title").textContent,"开合等待新基准");
  c.gripper_edge.left.state="ABORTED";h.app.renderState();
  assert.match(h.element("left-interaction").textContent,/已中止.*重新做掌↔拳切换/);
  c.gripper_edge.left.state="COMPLETE";h.app.renderState();
  assert.match(h.element("left-interaction").textContent,/已结束.*同一手形不重复/);
  c.gripper_edge.left.state="CLOSING";c.gripper_edge.left.authorized_mode="Z_CLOSE";h.app.renderState();
  assert.match(h.element("left-interaction").textContent,/掌→拳已确认.*正在匀速闭爪/);
  assert.equal(h.element("control-status-title").textContent,"Z 升降 · 本次开合执行中");
  h.app.engine.pause("user_pause");h.app.setOutput(h.app.engine.pause("user_pause"));h.app.renderState();
  assert.match(h.element("notice").textContent,/首次可信掌／拳只建立基准.*相反手形/);
});

test("edge-trigger release UI permits airborne motion while preserving bilateral intent and confirmation",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundEdgeCamera(h);
  for(let i=0;i<9;i++)await deliverRecognition(h,gestureResult(["PALM","PALM"]));
  const c=h.app.runtime.physical_contact;
  Object.assign(c,{hold_width_m:{left:.025,right:.025},interaction_modes:{left:"Z_OPEN",right:"Z_OPEN"},
    release_reason:{left:"release_confirmation",right:"release_requires_both_sides"},
    support_blockers:["object_above_table"],bottom_height_m:.02,support_requirements:{maximum_bottom_height_m:.003}});
  c.gripper_edge.left.state="CONFIRMING";c.gripper_edge.right.state="WAIT_BOTH";h.app.renderState();
  assert.match(h.element("left-interaction").textContent,/无需落桌或停手.*空中也可开爪/);
  assert.match(h.element("right-interaction").textContent,/另一侧拳→掌切换/);
  assert.match(h.element("control-gripper-state").textContent,/无需落桌或停手/);
  assert.doesNotMatch(h.element("operation-help").innerHTML,/仍需桌面支撑|停稳.*才/);
  assert.doesNotMatch(h.element("grasp-state").textContent,/正在.*开爪/);
  c.gripper_edge.left.state="CONFIRMING";c.release_progress_s={left:.4};c.release_clock="server_monotonic";h.app.renderState();
  assert.match(h.element("left-interaction").textContent,/0.40 \/ 0.70 s.*实际时间/);
  assert.match(h.element("control-status-title").textContent,/释放确认/);
  c.gripper_edge.left.state="OPENING";c.gripper_edge.right.state="OPENING";h.app.renderState();
  assert.match(h.element("grasp-state").textContent,/拳→掌已确认.*正在匀速开爪/);
});

test("partial grasp UI identifies which arm holds and which can continue alignment",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundEdgeCamera(h);
  const c=h.app.runtime.physical_contact;
  Object.assign(c,{hold_width_m:{left:.025,right:null},stable_grasps:{left:true,right:false}});
  h.app.runtime.contacts={left:true,right:false};h.app.runtime.task_state="PARTIAL_CONTACT";h.app.renderState();
  assert.equal(h.element("control-status-title").textContent,"单侧抓稳保持 · 另一侧对准");
  assert.match(h.element("notice").textContent,/左臂已抓稳.*右臂可继续对准/);
  h.app.runtime.last_command_result={command:"continuous_input",accepted:false,reason:"partial_grasp_holds_engaged_arm"};h.app.renderState();
  assert.match(h.element("notice").textContent,/左臂已抓稳.*拒绝本次双臂请求.*另一爪本次开合也未执行/);
  assert.doesNotMatch(h.element("notice").textContent,/可继续对准/);
});

test("partial grasp shows either free gripper executing while its partner remains held",async(t)=>{
  for(const held of ["left","right"]) {
    const h=makeHarness();t.after(()=>h.close());await boundEdgeCamera(h);
    const free=held==="left"?"right":"left",holdingName=held==="left"?"左":"右",freeName=free==="left"?"左":"右";
    const c=h.app.runtime.physical_contact;
    Object.assign(c,{hold_width_m:{[held]:.0094,[free]:null},stable_grasps:{[held]:true,[free]:false}});
    h.app.runtime.contacts={[held]:true,[free]:false};h.app.runtime.task_state="PARTIAL_CONTACT";
    h.app.runtime.last_command_result={command:"continuous_input",accepted:true};
    for(const state of ["CLOSING","OPENING"]) {
      c.gripper_edge[free]={...c.gripper_edge[free],state,authorized_mode:state==="CLOSING"?"Z_CLOSE":"Z_OPEN"};
      h.app.renderState();
      assert.equal(h.element("control-status-title").textContent,`${holdingName}臂保持 · ${freeName}爪开合中`);
      assert.match(h.element("notice").textContent,new RegExp(`${holdingName}臂已抓稳并保持位置；${freeName}臂可继续对准和掌↔拳开合`));
      assert.match(h.element(`${free}-interaction`).textContent,state==="CLOSING"?/正在匀速闭爪/:/正在匀速开爪/);
      assert.doesNotMatch(h.element(`${free}-interaction`).textContent,/当前请求未接受/);
    }
    c.gripper_edge[free].state="READY";h.app.renderState();
    assert.doesNotMatch(h.element("control-status-title").textContent,/开合中/);
    c.stable_grasps[held]=false;h.app.renderState();
    assert.doesNotMatch(h.element("notice").textContent,new RegExp(`${holdingName}臂已抓稳`));
  }
});

test("an accepted partial-grasp input replaces an earlier whole-request rejection in the UI",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundEdgeCamera(h);
  const c=h.app.runtime.physical_contact;
  Object.assign(c,{hold_width_m:{left:null,right:.0094},stable_grasps:{left:false,right:true}});
  Object.assign(h.app.runtime,{contacts:{left:false,right:true},task_state:"PARTIAL_CONTACT",
    last_command_result:{command:"continuous_input",eventId:"old-reject",accepted:false,reason:"partial_grasp_holds_engaged_arm"}});
  c.gripper_edge.left={...c.gripper_edge.left,state:"WAIT_BASELINE",baseline_mode:null};h.app.renderState();
  assert.match(h.element("notice").textContent,/拒绝本次双臂请求.*另一爪本次开合也未执行/);
  assert.match(h.element("left-interaction").textContent,/当前请求未接受/);
  h.enqueue("/api/dual/health",h.response(200,{...h.app.runtime,
    last_command_result:{command:"continuous_input",eventId:"accepted-projection",accepted:true,reason:null},
    physical_contact:{...c,gripper_edge:{...c.gripper_edge,left:{...c.gripper_edge.left,state:"CLOSING",authorized_mode:"Z_CLOSE",baseline_mode:"Z_CLOSE"}}}}));
  await h.app.checkHealth();
  assert.equal(h.element("control-status-title").textContent,"右臂保持 · 左爪开合中");
  assert.match(h.element("left-interaction").textContent,/正在匀速闭爪/);
  assert.doesNotMatch(h.element("notice").textContent,/拒绝|未执行/);
  assert.doesNotMatch(h.element("left-interaction").textContent,/当前请求未接受|后端：/);
  assert.equal(h.app.engine.manualPaused,false);
});

test("a short valid unknown shape sends HOLD input in Z and cannot become a hard pause",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundEdgeCamera(h);
  await deliverRecognition(h,gestureResult(["UNKNOWN","UNKNOWN"]));
  const output=h.app.output;
  assert.equal(output.command.command,"continuous_input");assert.equal(output.mode,"Z");
  assert.deepEqual([...output.command.interaction_modes],["HOLD","HOLD"]);
  assert.equal(output.telemetry.control_mode,"Z");
  assert.equal(output.reason,"rate_gesture_unqualified");
  assert.match(h.element("left-interaction").textContent,/当前手势未分配.*本帧保持爪宽/);
  for(let i=0;i<7;i++)await deliverRecognition(h,gestureResult(["FIST","FIST"]));
  assert.deepEqual([...h.app.output.command.interaction_modes],["Z_CLOSE","Z_CLOSE"]);
  await deliverRecognition(h,{landmarks:[],handedness:[],gestures:[]});
  assert.equal(h.app.output.command.command,"pause");assert.equal(h.app.output.mode,"PAUSE");
});

test("edge-trigger cards retain current confidence and transition explanations",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundEdgeCamera(h);
  await deliverRecognition(h,gestureResult(["PALM","FIST"],{scores:[.49,.79]}));h.app.renderState();
  assert.match(h.element("left-interaction").textContent,/张掌评分 49%.*至少 50%.*不产生新的开合切换/);
  assert.match(h.element("right-interaction").textContent,/握拳评分 79%.*至少 80%/);
  await deliverRecognition(h,gestureResult(["PALM","FIST"]));h.app.renderState();
  assert.match(h.element("left-interaction").textContent,/新手势确认 200 ms.*本帧保持爪宽/);
});

test("edge recording start explains the fresh baseline instead of silently reusing practice authorization",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundEdgeCamera(h);
  Object.assign(h.app.runtime,{recording:true,paused:true,pause_reason:"record_start_baseline"});
  h.app.setOutput({...h.app.output,reason:null});h.app.renderState();
  assert.equal(h.element("control-status-title").textContent,"录制已开始 · 等待开合基准");
  assert.match(h.element("notice").textContent,/位置与爪宽保持.*稳定张掌.*基准.*握拳/);
  assert.match(h.element("notice").textContent,/练习阶段.*不会带入/);
});


test("level-mode health remains connected after operation help replaces the old gripper guide", async (t) => {
  const missingIds = new Set();
  const h = makeHarness({ missingIds }); t.after(() => h.close()); await settle();
  const config = JSON.parse(readFileSync(new URL("../../../config/collection.json", import.meta.url), "utf8"));
  const runtime = { ...healthyRuntime(), config: { ...config, contact_control: { enabled: true } },
    physical_contact: { control_protocol: "gesture-rate-gripper-v3", gripper_trigger_policy: "palm_fist_level_v2" } };
  h.enqueue("/api/dual/health", h.response(200, runtime)); await h.app.checkHealth();
  assert.equal(h.app.connected, true);
  assert.equal(h.app.engine.gestureProtocol(), "gesture-rate-gripper-v3");
  assert.match(h.element("operation-help").innerHTML, /左右独立/);
  // Real innerHTML replacement removes the old child from the document.
  assert.doesNotMatch(h.element("operation-help").innerHTML, /id=["']gripper-guide/);
  missingIds.add("gripper-guide");
  for (let i = 0; i < 3; i++) {
    h.enqueue("/api/dual/health", h.response(200, runtime)); await h.app.checkHealth();
    assert.equal(h.app.connected, true, "a replaced help node must not turn healthy HTTP into disconnection");
    assert.match(h.element("connection").textContent, /已连接/);
  }
});


test("completed camera measurements survive a blocked replaceable control request",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundLevelRateCamera(h);
  await h.app.measurementJournal.flush();
  const before=h.app.measurementJournal.completed,held=deferred();
  h.enqueue("/api/dual/control",held.promise);
  for(let i=0;i<12;i++)await deliverRecognition(h,gestureResult(i<6?["UNKNOWN","UNKNOWN"]:["FIST","FIST"]));
  assert.equal(h.app.measurementJournal.completed,before+12);
  await h.app.measurementJournal.flush();
  assert.equal(h.app.measurementJournal.written,h.app.measurementJournal.completed);
  assert.equal(h.app.measurementJournal.dropped,0);
  const observations=h.requests.filter(r=>r.url.endsWith('/measurements')).flatMap(r=>r.body.rows).filter(r=>r.kind==='measurement');
  assert.ok(observations.some(r=>r.measurement.hands.some(hand=>hand.gesture==='None'&&hand.image_landmarks_21.length===21)));
  held.resolve(h.response(200,{ok:true,queued:true}));await settle();
});

test("a completed result delivered after camera stop is journalled without controlling the robot",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundLevelRateCamera(h);
  const worker=h.workers[0],frame=worker.messages.filter(m=>m.type==='frame').at(-1);
  h.app.stopCamera();await settle();
  const before=h.app.measurementJournal.completed,controls=h.controls().length;
  worker.emit({...frame,type:'result',result:gestureResult(['UNKNOWN','UNKNOWN']),inferenceMs:20});await settle();
  assert.equal(h.app.measurementJournal.completed,before+1);
  assert.equal(h.controls().length,controls);
  const row=h.app.measurementJournal.queue.find(r=>r.kind==='measurement'&&r.measurement.drop_reason==='worker_replaced');
  assert.ok(row);assert.equal(row.measurement.hands[0].image_landmarks_21.length,21);
});

for(const saveCommand of ["record_discard"]){
  test(`${saveCommand}: research upload failure must not remove save/reset/next-record workflow`,async(t)=>{
    const h=makeHarness();t.after(()=>h.close());await boundLevelRateCamera(h);
    Object.assign(h.app.runtime,{recording:true,task_state:saveCommand==="record_save"?"DONE":"PREGRASP",frames:100});
    h.enqueue("/api/dual/measurements",h.response(503,{ok:false,error:"WinError 5 manifest rename denied"}));
    h.app.renderState();h.button(saveCommand).dispatch("click",{detail:1});await settle();
    const request=h.controls().find(r=>r.body.command===saveCommand)?.body;
    assert.ok(request,"save must reach backend even if auxiliary research upload is unavailable");
    assert.ok(request.hand_journal_barrier.error,"partial research must be explicitly marked");
    const next={...h.app.runtime,recording:false,saving:false,task_state:"PREGRASP",control_epoch:1,
      contacts:{left_fingers:[false,false],right_fingers:[false,false]},
      physical_contact:{...h.app.runtime.physical_contact,state:"FREE",ever_dual:false},
      last_operation_result:{command:saveCommand,eventId:request.eventId,accepted:true,frames:100,
        control_epoch:0,completed_control_epoch:1,recording:false,research_warning:"hand_measurement_upload_incomplete"}};
    h.enqueue("/api/dual/health",h.response(200,next));await h.app.checkHealth();
    assert.equal(h.app.pendingOperation,null);assert.equal(h.button("record_start").disabled,false);
    assert.equal(h.button("reset").disabled,false);
    h.button("record_start").dispatch("click");await settle();
    assert.ok(h.controls().some(r=>r.body.command==="record_start"));
  });
}


test("a stale snapshot cannot look fresh merely because its HTTP response is new",async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await h.ready;
  await h.app.checkHealth();
  for(const name of cameraNames){
    const r=h.response(200,{},name);
    r.headers.get=(key)=>key.toLowerCase()==='x-preview-age-ms'?'1500':null;
    h.enqueue(`/api/dual/${name}-preview`,r);
  }
  await h.app.pollPreviews();
  assert.ok(imageIds.every(id=>h.element(id).hidden));
});

test('save waits for the captured worker frame and durable upload before sending the operation',async(t)=>{
  const h=makeHarness();t.after(()=>h.close());await boundLevelRateCamera(h);
  Object.assign(h.app.runtime,{recording:true,task_state:'DONE',frames:100});
  const frame=h.workers[0].messages.filter(m=>m.type==='frame').at(-1);
  const before=h.app.measurementJournal.completed;
  const upload=deferred();
  h.enqueue('/api/dual/measurements',async req=>{await upload.promise;return h.response(200,{ok:true,last_sequence:req.body.rows.at(-1).sequence});});
  h.app.renderState();h.button('record_save').dispatch('click',{detail:1});await settle();
  assert.equal(h.controls().some(r=>r.body.command==='record_save'),false);
  await deliverRecognition(h,gestureResult(['PALM','PALM']));
  assert.equal(h.app.measurementJournal.completed,before+1);
  assert.equal(h.controls().some(r=>r.body.command==='record_save'),false);
  const m=h.app.measurementJournal.queue.find(r=>r.kind==='measurement'&&r.measurement.measurement_id.endsWith(`:${frame.frameId}`));
  assert.ok(m);assert.equal(m.measurement.capture_epoch_ms,frame.capturedEpochMs);
  upload.resolve();await settle();
  const req=h.controls().find(r=>r.body.command==='record_save')?.body;
  assert.ok(req);assert.equal(req.hand_journal_barrier.pending,0);
});

for(const command of ['record_save','record_failure']) {
  test(`${command}: failed research upload blocks save, retry seals the same prefix`,async(t)=>{
    const h=makeHarness();t.after(()=>h.close());await boundLevelRateCamera(h);
    Object.assign(h.app.runtime,{recording:true,task_state:command==='record_save'?'DONE':'PREGRASP',frames:100});
    h.enqueue('/api/dual/measurements',h.response(503,{ok:false,error:'offline'}));
    h.app.renderState();h.button(command).dispatch('click',{detail:1});await settle();
    await deliverRecognition(h,gestureResult(['PALM','PALM']));
    assert.equal(h.controls().some(r=>r.body.command===command),false);
    assert.equal(h.app.pendingOperation,null);assert.equal(h.app.runtime.recording,true);
    assert.match(h.element('record-feedback').textContent,/未提交保存/);
    h.app.measurementJournal.add('measurement',{measurement:{measurement_id:'after-click',capture_epoch_ms:Date.now(),hands:[]}},{ });
    h.button(command).dispatch('click',{detail:1});await settle();
    const req=h.controls().find(r=>r.body.command===command)?.body;
    assert.ok(req);assert.equal(req.hand_journal_barrier.version,'hand-save-boundary-v1');
    assert.equal(req.hand_journal_barrier.pending,0);
    assert.equal(req.hand_journal_barrier.last_sequence,req.hand_journal_barrier.acknowledged_sequence);
    assert.ok(req.hand_journal_barrier.last_sequence<h.app.measurementJournal.sequence);
  });
}


test("success tail delays save buttons until the fixed episode boundary", async (t) => {
  const h=makeHarness(); t.after(()=>h.close()); await h.boot();
  Object.assign(h.app.runtime,{recording:true,task_state:"DONE",frames:100,
    episode_end:null,success_tail:{seconds:.5,active:true}});
  h.app.renderState();
  assert.equal(h.button("record_save").disabled,true);
  assert.equal(h.button("record_failure").disabled,true);
  assert.match(h.button("record_save").title,/收尾/);
  h.app.runtime.episode_end={simulation_time:10.5,post_success_seconds:.5};
  h.app.renderState();
  assert.equal(h.button("record_save").disabled,false);
  assert.equal(h.button("record_failure").disabled,false);
});
