/* Classic worker: the installed MediaPipe bundle exposes its global Vision. */
let recognizer = null;
let initialized = false;
let lastTimestamp = -Infinity;

self.onmessage = async ({ data }) => {
  if (data.type === "init") {
    try {
      importScripts("./node_modules/@mediapipe/tasks-vision/vision_bundle.js");
      const vision = await Vision.FilesetResolver.forVisionTasks(data.wasmRoot);
      recognizer = await Vision.GestureRecognizer.createFromOptions(vision, {
        baseOptions: { modelAssetPath: data.modelUrl, delegate: data.delegate ?? "CPU" },
        canvas: new OffscreenCanvas(640, 480), runningMode: "VIDEO", numHands: 2,
      });
      initialized = true;
      self.postMessage({ type: "ready", delegate: data.delegate ?? "CPU" });
    } catch (error) {
      self.postMessage({ type: "error", stage: "init", message: String(error?.message ?? error) });
    }
    return;
  }
  if (data.type !== "frame") return;
  const started = performance.now();
  try {
    if (!initialized || !recognizer) throw new Error("Gesture worker is not ready");
    if (data.capturedAt <= lastTimestamp) throw new Error("Out-of-order camera frame");
    lastTimestamp = data.capturedAt;
    const result = recognizer.recognizeForVideo(data.bitmap, data.capturedAt);
    self.postMessage({
      type: "result", frameId: data.frameId, capturedAt: data.capturedAt,
      capturedEpochMs: data.capturedEpochMs, control_epoch: data.control_epoch, session: data.session, inferenceMs: performance.now() - started,
      result: { landmarks: result.landmarks, handedness: result.handedness, gestures: result.gestures },
    });
  } catch (error) {
    self.postMessage({ type: "error", stage: "frame", frameId: data.frameId, message: String(error?.message ?? error) });
  } finally { data.bitmap?.close(); }
};
