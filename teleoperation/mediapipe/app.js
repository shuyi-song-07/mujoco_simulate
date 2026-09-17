import {
  DrawingUtils,
  FilesetResolver,
  GestureRecognizer,
} from "./node_modules/@mediapipe/tasks-vision/vision_bundle.mjs";

import { mapPlanarMotion, shouldStartPlanarLock } from "./interaction-state.js";

const video = document.querySelector("#webcam");
const canvas = document.querySelector("#output-canvas");
const canvasContext = canvas.getContext("2d");
const cameraButton = document.querySelector("#camera-button");
const cameraPlaceholder = document.querySelector("#camera-placeholder");
const cameraPlaceholderText = document.querySelector("#camera-placeholder-text");
const status = document.querySelector("#status");
const statusText = document.querySelector("#status-text");
const appNotice = document.querySelector("#app-notice");
const appNoticeTitle = document.querySelector("#app-notice-title");
const appNoticeText = document.querySelector("#app-notice-text");
const recognitionResults = document.querySelector("#recognition-results");
const robotStatus = document.querySelector("#robot-status");
const robotStatusText = document.querySelector("#robot-status-text");
const robotFeedback = document.querySelector("#robot-feedback");
const sidePreviewPanel = document.querySelector("#side-preview-panel");
const sidePreviewImage = document.querySelector("#side-preview-image");
const frontPreviewImage = document.querySelector("#front-preview-image");
const recordStartButton = document.querySelector("#record-start");
const recordSaveButton = document.querySelector("#record-save");
const recordDiscardButton = document.querySelector("#record-discard");
const recordStopButton = document.querySelector("#record-stop");
const robotButtons = [
  recordStartButton,
  recordSaveButton,
  recordDiscardButton,
  recordStopButton,
];

const robotApiBase = "/api";

const gestureNames = {
  Closed_Fist: "握拳",
  Open_Palm: "张开手掌",
  Thumb_Down: "拇指向下",
  Thumb_Up: "竖起拇指",
  Victory: "胜利手势",
  ILoveYou: "我爱你",
  None: "未识别手势",
};

const handednessNames = {
  Left: "左手",
  Right: "右手",
};

let gestureRecognizer;
let drawingUtils;
let mediaStream;
let animationFrameId;
let lastVideoTime = -1;
let robotConnected = false;
let lastFistY = null;
let lastOpenPalmY = null;
let lastPlanarPoint = null;
let gripperClosed = null;
let lastControlSentAt = 0;
let lastOpenSentAt = Number.NEGATIVE_INFINITY;
let lastCloseSentAt = Number.NEGATIVE_INFINITY;
let lastRobotGesture = "None";
let sidePreviewPollTimer;
let sidePreviewObjectUrl;
let frontPreviewObjectUrl;
let lastDiagonalGestureAt = Number.NEGATIVE_INFINITY;
let diagonalOpenBlocked = false;
let motionRequestInFlight = false;
let pendingMotionRequest;

const CONTROL_INTERVAL_MS = 40;
const PLANAR_SENSITIVITY = 0.03;
const VERTICAL_DEAD_ZONE = 0.015;
const PLANAR_DEAD_ZONE = 0.09;
const GRIPPER_RETRY_MS = 300;
const DIAGONAL_OPEN_GUARD_MS = 650;
const DIAGONAL_GESTURE_HOLD_MS = 180;

function updateStatus(message, state = "idle") {
  statusText.textContent = message;
  status.dataset.state = state;
}

function showNotice(title, message, type = "error") {
  appNoticeTitle.textContent = title;
  appNoticeText.textContent = message;
  appNotice.dataset.type = type;
  appNotice.hidden = false;
}

function clearNotice() {
  appNotice.hidden = true;
  appNoticeTitle.textContent = "";
  appNoticeText.textContent = "";
}

function setRobotConnected(isConnected) {
  const wasConnected = robotConnected;
  robotConnected = isConnected;
  robotStatus.dataset.state = isConnected ? "active" : "error";
  robotStatusText.textContent = isConnected ? "MuJoCo 已连接" : "MuJoCo 未连接";
  robotButtons.forEach((button) => {
    button.disabled = !isConnected;
  });
  if (!isConnected) {
    hideSidePreview();
    robotFeedback.textContent = "请先运行 MuJoCo LeRobot 录制程序。";
  } else if (!wasConnected) {
    robotFeedback.textContent = "MuJoCo 已连接，可以启用机器人控制并开始录制。";
  }
}

function hideSidePreview() {
  sidePreviewPanel.hidden = true;
  sidePreviewImage.removeAttribute("src");
  frontPreviewImage.removeAttribute("src");
  if (sidePreviewObjectUrl) {
    URL.revokeObjectURL(sidePreviewObjectUrl);
    sidePreviewObjectUrl = undefined;
  }
  if (frontPreviewObjectUrl) {
    URL.revokeObjectURL(frontPreviewObjectUrl);
    frontPreviewObjectUrl = undefined;
  }
}

async function pollSidePreview() {
  try {
    if (!robotConnected) {
      hideSidePreview();
      return;
    }
    const [sideResponse, frontResponse] = await Promise.all([
      fetch(`${robotApiBase}/side-preview`, { cache: "no-store" }),
      fetch(`${robotApiBase}/front-preview`, { cache: "no-store" }),
    ]);
    if (sideResponse.status === 204 || frontResponse.status === 204) {
      hideSidePreview();
      return;
    }
    if (!sideResponse.ok || !frontResponse.ok) throw new Error("辅助视角读取失败");

    const nextSideUrl = URL.createObjectURL(await sideResponse.blob());
    const nextFrontUrl = URL.createObjectURL(await frontResponse.blob());
    const previousSideUrl = sidePreviewObjectUrl;
    const previousFrontUrl = frontPreviewObjectUrl;
    sidePreviewObjectUrl = nextSideUrl;
    frontPreviewObjectUrl = nextFrontUrl;
    sidePreviewImage.src = nextSideUrl;
    frontPreviewImage.src = nextFrontUrl;
    sidePreviewPanel.hidden = false;
    if (previousSideUrl) URL.revokeObjectURL(previousSideUrl);
    if (previousFrontUrl) URL.revokeObjectURL(previousFrontUrl);
  } catch {
    hideSidePreview();
  } finally {
    sidePreviewPollTimer = window.setTimeout(pollSidePreview, 100);
  }
}

async function checkRobotConnection() {
  try {
    const response = await fetch(`${robotApiBase}/health`, { cache: "no-store" });
    setRobotConnected(response.ok);
  } catch {
    setRobotConnected(false);
  }
}

async function sendRobotCommand(command, extra = {}) {
  if (!robotConnected) {
    robotFeedback.textContent = "MuJoCo 尚未连接，无法发送控制命令。";
    return false;
  }

  try {
    const response = await fetch(`${robotApiBase}/control`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command, sentAt: Date.now(), ...extra }),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return true;
  } catch (error) {
    console.error(error);
    setRobotConnected(false);
    return false;
  }
}

async function flushLatestMotionRequest() {
  if (motionRequestInFlight) return;
  motionRequestInFlight = true;
  try {
    while (pendingMotionRequest) {
      const request = pendingMotionRequest;
      pendingMotionRequest = undefined;
      await sendRobotCommand(request.command, request.extra);
    }
  } finally {
    motionRequestInFlight = false;
  }
}

function sendLatestMotionCommand(command, extra = {}) {
  pendingMotionRequest = { command, extra };
  void flushLatestMotionRequest();
}

function processRobotControl(result, timestamp) {
  if (!robotConnected) {
    lastFistY = null;
    lastOpenPalmY = null;
    lastPlanarPoint = null;
    pendingMotionRequest = undefined;
    return;
  }

  if (!(result.landmarks?.length ?? 0)) {
    lastFistY = null;
    lastOpenPalmY = null;
    lastPlanarPoint = null;
    diagonalOpenBlocked = false;
    pendingMotionRequest = undefined;
    return;
  }

  const topGesture = result.gestures?.[0]?.[0];
  const gesture = topGesture?.categoryName ?? "None";
  const wrist = result.landmarks[0][0];
  const gestureChanged = gesture !== lastRobotGesture;

  if (gesture === "ILoveYou") {
    lastDiagonalGestureAt = timestamp;
    diagonalOpenBlocked = true;
  } else if (gesture !== "Open_Palm") {
    // Require a deliberate break between diagonal motion and opening. This
    // prevents ILoveYou -> Open_Palm misclassification from releasing a cube.
    diagonalOpenBlocked = false;
  }
  const openPalmControlAllowed =
    !diagonalOpenBlocked &&
    timestamp - lastDiagonalGestureAt >= DIAGONAL_OPEN_GUARD_MS;

  if (
    gesture === "Closed_Fist" &&
    (gestureChanged || timestamp - lastCloseSentAt >= GRIPPER_RETRY_MS)
  ) {
    void sendRobotCommand("close");
    gripperClosed = true;
    lastCloseSentAt = timestamp;
    robotFeedback.textContent = "握拳：夹爪闭合；上下移动手可直线升降。";
  }
  if (gesture === "Closed_Fist" && gestureChanged) {
    void sendRobotCommand("start_vertical");
  }
  if (gesture === "Closed_Fist") {
    if (lastFistY === null) {
      lastFistY = wrist.y;
    } else {
      const verticalDelta = wrist.y - lastFistY;
      if (verticalDelta < -VERTICAL_DEAD_ZONE) {
        sendLatestMotionCommand("up");
        lastFistY = wrist.y;
      } else if (verticalDelta > VERTICAL_DEAD_ZONE) {
        sendLatestMotionCommand("down");
        lastFistY = wrist.y;
      }
    }
  } else {
    lastFistY = null;
  }

  if (
    gesture === "Open_Palm" &&
    openPalmControlAllowed &&
    (gestureChanged || timestamp - lastOpenSentAt >= GRIPPER_RETRY_MS)
  ) {
    void sendRobotCommand("open");
    gripperClosed = false;
    lastOpenSentAt = timestamp;
    robotFeedback.textContent = "张开手掌：夹爪松开；上下移动手可直线升降。";
  }
  if (gesture === "Open_Palm" && openPalmControlAllowed && gestureChanged) {
    void sendRobotCommand("start_vertical");
  }
  if (gesture === "Open_Palm" && openPalmControlAllowed) {
    if (lastOpenPalmY === null) {
      lastOpenPalmY = wrist.y;
    } else {
      const verticalDelta = wrist.y - lastOpenPalmY;
      if (verticalDelta < -VERTICAL_DEAD_ZONE) {
        sendLatestMotionCommand("up");
        lastOpenPalmY = wrist.y;
      } else if (verticalDelta > VERTICAL_DEAD_ZONE) {
        sendLatestMotionCommand("down");
        lastOpenPalmY = wrist.y;
      }
    }
  } else {
    lastOpenPalmY = null;
  }

  // Brief recognition flicker must not switch a diagonal move onto a single
  // axis. Closed_Fist remains immediate so deliberate gripping is unaffected.
  const planarGesture =
    gripperClosed === true &&
    gesture !== "Closed_Fist" &&
    timestamp - lastDiagonalGestureAt <= DIAGONAL_GESTURE_HOLD_MS
      ? "ILoveYou"
      : gesture;
  const isPlanarGesture =
    planarGesture === "Victory" ||
    planarGesture === "Thumb_Up" ||
    planarGesture === "ILoveYou";
  if (isPlanarGesture) {
    const planarPoint = wrist;
    if (shouldStartPlanarLock({
      gesture: planarGesture,
      gestureChanged,
      carrying: gripperClosed === true,
    })) {
      void sendRobotCommand("start_planar", {
        lockZ: planarGesture === "ILoveYou",
      });
      robotFeedback.textContent =
        "平面移动：到运输高度后会自动锁定 Z 并切换俯视视角。";
    }

    if (lastPlanarPoint && timestamp - lastControlSentAt >= CONTROL_INTERVAL_MS) {
      const screenDx = planarPoint.x - lastPlanarPoint.x;
      const screenDy = planarPoint.y - lastPlanarPoint.y;
      let { dx, dy } = mapPlanarMotion({
        gesture: planarGesture,
        screenDx,
        screenDy,
        sensitivity: PLANAR_SENSITIVITY,
        carrying: gripperClosed === true,
      });

      const magnitude = Math.hypot(dx, dy);
      if (magnitude > 1) {
        dx /= magnitude;
        dy /= magnitude;
      }
      if (magnitude > PLANAR_DEAD_ZONE) {
        sendLatestMotionCommand("move_xy", {
          dx,
          dy,
          lockZ: planarGesture === "ILoveYou",
        });
        lastControlSentAt = timestamp;
        const messages = {
          Victory: "Victory：正在左右移动。",
          Thumb_Up: "竖起拇指：正在前后移动。",
          ILoveYou: "ILoveYou：正在斜向移动。",
        };
        robotFeedback.textContent = messages[planarGesture];
      }
    }
    lastPlanarPoint = { x: planarPoint.x, y: planarPoint.y };
  } else {
    lastPlanarPoint = null;
  }

  lastRobotGesture = gesture;
}

function setCameraLoading(isLoading, message = "正在准备摄像头…") {
  cameraPlaceholder.classList.toggle("is-loading", isLoading);
  cameraPlaceholderText.textContent = isLoading
    ? message
    : "点击下方按钮开启摄像头";
  cameraButton.classList.toggle("is-loading", isLoading);
  cameraButton.textContent = isLoading ? message : "启动摄像头";
}

async function createGestureRecognizer() {
  if (gestureRecognizer) return;

  updateStatus("正在加载识别模型", "loading");
  setCameraLoading(true, "正在加载识别模型…");

  const vision = await FilesetResolver.forVisionTasks(
    "./node_modules/@mediapipe/tasks-vision/wasm",
  );

  gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "./models/gesture_recognizer.task",
      delegate: "CPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });

  drawingUtils = new DrawingUtils(canvasContext);
}

function resizeCanvasToVideo() {
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
}

function drawHandLandmarks(result) {
  canvasContext.clearRect(0, 0, canvas.width, canvas.height);

  for (const landmarks of result.landmarks ?? []) {
    drawingUtils.drawConnectors(
      landmarks,
      GestureRecognizer.HAND_CONNECTIONS,
      { color: "#70FF9D", lineWidth: 4 },
    );
    drawingUtils.drawLandmarks(landmarks, {
      color: "#FF5C5C",
      fillColor: "#FF5C5C",
      lineWidth: 2,
      radius: 4,
    });
  }

  updateStatus(
    result.landmarks?.length ? "已检测到手部" : "请将手放入画面",
    "active",
  );
}

function createResultElement(gesture, handedness, index) {
  const confidence = Math.round((gesture?.score ?? 0) * 100);
  const gestureCode = gesture?.categoryName ?? "None";
  const handednessCode = handedness?.categoryName ?? "Unknown";

  const card = document.createElement("article");
  card.className = "hand-result";

  const header = document.createElement("div");
  header.className = "hand-result-header";

  const gestureBlock = document.createElement("div");
  const handIndex = document.createElement("span");
  handIndex.className = "hand-index";
  handIndex.textContent = `HAND ${index + 1}`;

  const gestureName = document.createElement("h3");
  gestureName.textContent = gestureNames[gestureCode] ?? gestureCode;

  const rawGestureName = document.createElement("code");
  rawGestureName.className = "gesture-code";
  rawGestureName.textContent = gestureCode;

  gestureBlock.append(handIndex, gestureName, rawGestureName);

  const handednessLabel = document.createElement("span");
  handednessLabel.className = "handedness";
  handednessLabel.textContent = handednessNames[handednessCode] ?? handednessCode;
  header.append(gestureBlock, handednessLabel);

  const confidenceRow = document.createElement("div");
  confidenceRow.className = "confidence-row";
  const confidenceLabel = document.createElement("span");
  confidenceLabel.className = "confidence-label";
  confidenceLabel.textContent = "置信度";
  const confidenceValue = document.createElement("span");
  confidenceValue.className = "confidence-value";
  confidenceValue.textContent = `${confidence}%`;
  confidenceRow.append(confidenceLabel, confidenceValue);

  const confidenceTrack = document.createElement("div");
  confidenceTrack.className = "confidence-track";
  confidenceTrack.setAttribute("role", "progressbar");
  confidenceTrack.setAttribute("aria-label", `${gestureName.textContent}置信度`);
  confidenceTrack.setAttribute("aria-valuenow", String(confidence));
  confidenceTrack.setAttribute("aria-valuemin", "0");
  confidenceTrack.setAttribute("aria-valuemax", "100");

  const confidenceFill = document.createElement("div");
  confidenceFill.className = "confidence-fill";
  confidenceFill.style.setProperty("--confidence", `${confidence}%`);
  confidenceTrack.append(confidenceFill);

  card.append(header, confidenceRow, confidenceTrack);
  return card;
}

function renderRecognitionResults(result) {
  const gestures = result.gestures ?? [];
  recognitionResults.replaceChildren();

  if (!gestures.length) {
    const emptyResult = document.createElement("p");
    emptyResult.className = "empty-result";
    emptyResult.textContent = "暂未识别到手势";
    recognitionResults.append(emptyResult);
    return;
  }

  gestures.forEach((gestureCategories, index) => {
    const topGesture = gestureCategories[0];
    const handedness = result.handedness?.[index]?.[0];
    recognitionResults.append(createResultElement(topGesture, handedness, index));
  });
}

function detectVideoFrame() {
  if (!gestureRecognizer || !mediaStream) return;

  resizeCanvasToVideo();

  if (video.currentTime !== lastVideoTime) {
    const timestamp = performance.now();
    try {
      const result = gestureRecognizer.recognizeForVideo(video, timestamp);
      drawHandLandmarks(result);
      renderRecognitionResults(result);
      processRobotControl(result, timestamp);
      lastVideoTime = video.currentTime;
    } catch (error) {
      console.error(error);
      updateStatus("识别过程出现错误", "error");
      showNotice(
        "手势识别已停止",
        "实时识别发生异常。请关闭摄像头后重新启动；如果仍然失败，请刷新页面。",
      );
      return;
    }
  }

  animationFrameId = requestAnimationFrame(detectVideoFrame);
}

async function startCamera() {
  cameraButton.disabled = true;
  clearNotice();
  cameraPlaceholder.hidden = false;
  setCameraLoading(true, "正在准备识别功能…");

  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      const unsupportedError = new Error("当前浏览器不支持摄像头访问");
      unsupportedError.name = "UnsupportedBrowserError";
      throw unsupportedError;
    }

    await createGestureRecognizer();
    updateStatus("等待摄像头授权", "loading");
    setCameraLoading(true, "请在浏览器中允许摄像头…");

    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
    });

    video.srcObject = mediaStream;
    await video.play();

    cameraPlaceholder.hidden = true;
    cameraButton.textContent = "关闭摄像头";
    cameraButton.classList.remove("is-loading");
    cameraButton.disabled = false;
    lastVideoTime = -1;
    detectVideoFrame();
  } catch (error) {
    console.error(error);
    const errors = {
      NotAllowedError: {
        title: "无法使用摄像头",
        message: "摄像头权限被拒绝。请在浏览器地址栏的网站设置中允许摄像头，然后重新点击启动。",
      },
      NotFoundError: {
        title: "没有找到摄像头",
        message: "请确认电脑摄像头可用，或连接外置摄像头后重试。",
      },
      NotReadableError: {
        title: "摄像头正被占用",
        message: "请关闭正在使用摄像头的会议或拍照软件，然后重新启动。",
      },
      UnsupportedBrowserError: {
        title: "浏览器不支持摄像头",
        message: "请使用最新版 Chrome、Edge 或 Safari，并通过 localhost 打开本项目。",
      },
    };
    const detail = errors[error.name] ?? {
      title: "启动失败",
      message: "模型或页面资源加载失败。请确认已执行 npm install，并通过 npm start 打开页面。",
    };
    updateStatus(detail.title, "error");
    showNotice(detail.title, detail.message);
    setCameraLoading(false);
    cameraButton.disabled = false;
  }
}

function stopCamera() {
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
  mediaStream?.getTracks().forEach((track) => track.stop());
  mediaStream = undefined;
  video.srcObject = null;
  canvasContext.clearRect(0, 0, canvas.width, canvas.height);
  cameraPlaceholder.hidden = false;
  setCameraLoading(false);
  cameraButton.disabled = false;
  updateStatus("等待启动");
  recognitionResults.replaceChildren();
  const emptyResult = document.createElement("p");
  emptyResult.className = "empty-result";
  emptyResult.textContent = "启动摄像头并将手放入画面";
  recognitionResults.append(emptyResult);
  lastFistY = null;
  lastOpenPalmY = null;
  lastPlanarPoint = null;
  gripperClosed = null;
  lastRobotGesture = "None";
  pendingMotionRequest = undefined;
}

cameraButton.addEventListener("click", () => {
  if (mediaStream) stopCamera();
  else startCamera();
});

recordStartButton.addEventListener("click", async () => {
  if (await sendRobotCommand("record_start")) {
    robotFeedback.textContent = "正在录制当前 demonstration。";
  }
});

recordSaveButton.addEventListener("click", async () => {
  if (await sendRobotCommand("record_save")) {
    robotFeedback.textContent = "已请求保存成功 demonstration，MuJoCo 将自动重置。";
  }
});

recordDiscardButton.addEventListener("click", async () => {
  if (await sendRobotCommand("record_discard")) {
    robotFeedback.textContent = "已丢弃失败 demonstration，MuJoCo 将自动重置。";
  }
});

recordStopButton.addEventListener("click", async () => {
  if (await sendRobotCommand("record_stop")) {
    robotFeedback.textContent = "录制已结束，正在写入数据集。";
  }
});

setRobotConnected(false);
void checkRobotConnection();
void pollSidePreview();
window.setInterval(checkRobotConnection, 2000);

window.addEventListener("pagehide", () => {
  window.clearTimeout(sidePreviewPollTimer);
  hideSidePreview();
  stopCamera();
  gestureRecognizer?.close();
});
