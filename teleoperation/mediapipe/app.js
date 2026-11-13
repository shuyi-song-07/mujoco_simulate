import {
  DrawingUtils,
  FilesetResolver,
  GestureRecognizer,
} from "./node_modules/@mediapipe/tasks-vision/vision_bundle.mjs";

import { mapPlanarMotion, pointingUpAngle, pointingControlCandidate, pointingYawFromWrist, shouldStartPlanarLock, nextPlanarAnchor, planarGestureAfterDiagonal, SingleHandRearmGate, openPalmAllowedAfterDiagonal, openPalmReadyToRelease, handClearForRelease, shouldCloseGripper } from "./interaction-state.js?v=instant-reentry-01";
import { handMeasurement } from "./hand-measurement.js?v=raw-single-arm-01";
import { MeasurementJournal } from "./measurement-journal.js?v=raw-single-arm-01";

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
const resetSceneButton = document.querySelector("#reset-scene");
const practiceHint = document.querySelector("#practice-hint");
const sidePreviewPanel = document.querySelector("#side-preview-panel");
const topPreviewFigure = document.querySelector("#top-preview-figure");
const topPreviewImage = document.querySelector("#top-preview-image");
const sidePreviewImage = document.querySelector("#side-preview-image");
const frontPreviewImage = document.querySelector("#front-preview-image");
const recordStartButton = document.querySelector("#record-start");
const recordSaveButton = document.querySelector("#record-save");
const recordFailureButton = document.querySelector("#record-failure");
const recordStopButton = document.querySelector("#record-stop");
const openPalmRule = document.querySelector("#open-palm-rule");
const closedFistRule = document.querySelector("#closed-fist-rule");
const robotHelp = document.querySelector("#robot-help");
const pickPlaceHelp = robotHelp.textContent;
const robotButtons = [
  resetSceneButton,
  recordStartButton,
  recordSaveButton,
  recordFailureButton,
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
  Pointing_Up: "向上指",
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
let robotTaskMode = "pick_place";
let robotZLocked = false;
let pushingPlanarGestureStartedAt = Number.NEGATIVE_INFINITY;
let lastFistY = null;
let lastOpenPalmY = null;
let lastPlanarPoint = null;
let lastPlanarGesture = null;
let gripperClosed = null;
let lastControlSentAt = 0;
let lastOpenSentAt = Number.NEGATIVE_INFINITY;
let lastCloseSentAt = Number.NEGATIVE_INFINITY;
let lastRobotGesture = "None";
let sidePreviewPollTimer;
let topPreviewObjectUrl;
let sidePreviewObjectUrl;
let frontPreviewObjectUrl;
let lastDiagonalGestureAt = Number.NEGATIVE_INFINITY;
let openPalmSince = null;
let motionRequestInFlight = false;
let pendingMotionRequest;
const rearmGate = new SingleHandRearmGate();
let safetyTransport = Promise.resolve();
let resumePending = false;
let resumeConfirmed = false;
let pendingResumeToken = null;
let controlGeneration = 0;
let backendContext = null;
let controlToken = null;
let sceneRecording = false;
let lastCommandTimestamp = 0;
let latestRuntime = null;
let clockPair = null;
let currentMeasurementId = null;
let cameraResearchSettings = null;
let saveFlowInFlight = false;
const measurementClientId = crypto.randomUUID();
let measurementSequence = 0;
const measurementJournal = new MeasurementJournal({clientId:measurementClientId,
  send:async batch => {
    const response = await fetch(`${robotApiBase}/measurements`, {method:"POST",
      headers:{"Content-Type":"application/json"},body:JSON.stringify(batch),signal:AbortSignal.timeout(5000)});
    const value = await response.json();
    if (!response.ok || !value.ok) throw Error(value.reason ?? "原始日志写入失败");
    return value;
  },
  changed:state => {
    const el = document.querySelector("#measurement-log-state");
    if (el) el.textContent = `原始手部记录：识别 ${state.completed} / 已落盘 ${state.written} / 待写 ${state.pending} / 丢弃 ${state.dropped}${state.error ? " · 写入异常" : ""}`;
  }});

function researchContext() {
  return {camera:cameraResearchSettings,display_mirrored:true,inference_mirrored:false,
    hand_selection:"first_detection_existing_control",model:"models/gesture_recognizer.task",delegate:"CPU",
    control_config:{interval_ms:CONTROL_INTERVAL_MS,planar_sensitivity:planarSensitivity(),
      vertical_dead_zone:verticalDeadZone(),planar_dead_zone:planarDeadZone(),
      reentry_wait_ms:0,reentry_stable_ms:0}, task_mode:robotTaskMode,
    push_control_protocol:isPushingTask() ? "closed_gripper_xy_hold_z_reposition_v1" : null};
}

function recordRawMeasurement(result, capturedAt, recognizedAt) {
  currentMeasurementId = `${measurementClientId}:${++measurementSequence}`;
  const split = backendContext?.lastIndexOf(":") ?? -1;
  const measurement = handMeasurement(result,{id:currentMeasurementId,
    session:split >= 0 ? backendContext.slice(0,split) : null,
    epoch:split >= 0 ? Number(backendContext.slice(split+1)) : null,
    width:video.videoWidth,height:video.videoHeight,capturedAt,
    capturedEpochMs:performance.timeOrigin+capturedAt,recognizedEpochMs:performance.timeOrigin+recognizedAt,
    receivedAt:recognizedAt,inferenceMs:recognizedAt-capturedAt});
  measurementJournal.add("measurement",{measurement,clock_pair:clockPair,
    video_media_time_s:video.currentTime,
    capture_time_semantics:"Browser recognition input time; not a hardware exposure timestamp",
    control:{generation:controlGeneration,armed:rearmGate.armed,
      pointing_locked:armMotionLocked,gripper_closed:gripperClosed,
      planar_anchor:lastPlanarPoint ? {...lastPlanarPoint} : null,
      vertical_anchor_fist:lastFistY,vertical_anchor_palm:lastOpenPalmY}},researchContext());
}

function flushMeasurements() {
  if (robotConnected) void measurementJournal.flush().catch(() => {});
}
let sceneResetInFlight = false;
let pendingResetContext = null;
let lastGestureReportAt = 0;
let lastPointingAngle = null;
let lastPointingSentAt = Number.NEGATIVE_INFINITY;
let armMotionLocked = false;
let gripperOrientationFinalized = false;
let pointingExitGesture = null;
let pointingExitSince = Number.NEGATIVE_INFINITY;
let pointingFinalizationPending = false;
let closedFistSince = Number.NEGATIVE_INFINITY;

function queueSafetyCommand(command, extra = {}) {
  safetyTransport = safetyTransport.catch(() => false).then(() =>
    robotConnected ? sendRobotCommand(command, extra) : false,
  );
  return safetyTransport;
}

function disarmHand(reason = "hand_lost") {
  const hadControl = rearmGate.armed || resumePending || resumeConfirmed || controlToken !== null;
  const resumePointing = reason !== "scene_reset" && armMotionLocked && !gripperOrientationFinalized;
  const retainedPointingAngle = lastPointingAngle;
  controlGeneration += 1;
  rearmGate.reset();
  controlToken = null;
  resumePending = false;
  resumeConfirmed = false;
  pendingResumeToken = null;
  lastFistY = null;
  lastOpenPalmY = null;
  lastPlanarPoint = null;
  lastPlanarGesture = null;
  lastRobotGesture = "None";
  lastPointingAngle = resumePointing ? retainedPointingAngle : null;
  lastPointingSentAt = Number.NEGATIVE_INFINITY;
  armMotionLocked = resumePointing;
  pointingExitGesture = null;
  pointingExitSince = Number.NEGATIVE_INFINITY;
  pointingFinalizationPending = false;
  closedFistSince = Number.NEGATIVE_INFINITY;
  openPalmSince = null;
  pendingMotionRequest = undefined;
  if (hadControl) void queueSafetyCommand("pause", { reason });
}

function isPushingTask() {
  return robotTaskMode === "push_t";
}

const CONTROL_INTERVAL_MS = 40;
const PLANAR_SENSITIVITY = 0.03;
const VERTICAL_DEAD_ZONE = 0.015;
const PLANAR_DEAD_ZONE = 0.09;
const GRIPPER_CLOSE_RETRY_MS = 300;
const GRIPPER_OPEN_RETRY_MS = 150;
const GRIPPER_FIST_CONFIRM_MS = 50;
const DIAGONAL_GESTURE_HOLD_MS = 180;
const POINTING_EXIT_HOLD_MS = 200;

function planarSensitivity() {
  return PLANAR_SENSITIVITY;
}

function verticalDeadZone() {
  return VERTICAL_DEAD_ZONE;
}

function planarDeadZone() {
  return PLANAR_DEAD_ZONE;
}

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
    button.disabled = !isConnected || (button === resetSceneButton && (sceneRecording || sceneResetInFlight));
  });
  if (!isConnected) {
    if (wasConnected) disarmHand("backend_disconnected");
    hideSidePreview();
    robotFeedback.textContent = "请先运行 MuJoCo 仿真采集程序。";
  } else if (!wasConnected) {
    robotFeedback.textContent =
      isPushingTask()
        ? "MuJoCo Push-T 已连接：夹爪保持闭合；张掌或握拳升降，平面手势锁住当前高度。"
        : "MuJoCo 已连接，可以启用机器人控制并开始录制。";
  }
}

function hideSidePreview() {
  sidePreviewPanel.hidden = true;
  topPreviewImage.removeAttribute("src");
  topPreviewFigure.hidden = true;
  sidePreviewImage.removeAttribute("src");
  frontPreviewImage.removeAttribute("src");
  if (topPreviewObjectUrl) {
    URL.revokeObjectURL(topPreviewObjectUrl);
    topPreviewObjectUrl = undefined;
  }
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
    const feeds = [
      { path: "top-preview", image: topPreviewImage, getUrl: () => topPreviewObjectUrl, setUrl: (url) => { topPreviewObjectUrl = url; } },
      { path: "side-preview", image: sidePreviewImage, getUrl: () => sidePreviewObjectUrl, setUrl: (url) => { sidePreviewObjectUrl = url; } },
      { path: "front-preview", image: frontPreviewImage, getUrl: () => frontPreviewObjectUrl, setUrl: (url) => { frontPreviewObjectUrl = url; } },
    ];
    await Promise.all(feeds.map(async ({ path, image, getUrl, setUrl }) => {
      try {
        const response = await fetch(`${robotApiBase}/${path}`, { cache: "no-store" });
        if (!response.ok || response.status === 204) return;
        const nextUrl = URL.createObjectURL(await response.blob());
        const previousUrl = getUrl();
        setUrl(nextUrl);
        image.src = nextUrl;
        if (previousUrl) URL.revokeObjectURL(previousUrl);
      } catch (error) {
        console.warn(`辅助视角 ${path} 暂不可用`, error);
      }
    }));
    topPreviewFigure.hidden = !topPreviewObjectUrl;
    sidePreviewPanel.hidden = !(topPreviewObjectUrl || sidePreviewObjectUrl || frontPreviewObjectUrl);
  } catch {
    hideSidePreview();
  } finally {
    sidePreviewPollTimer = window.setTimeout(pollSidePreview, 100);
  }
}

async function checkRobotConnection() {
  const generationAtRequest = controlGeneration;
  const tokenAtRequest = controlToken;
  try {
    const requestEpochMs = Date.now();
    const response = await fetch(`${robotApiBase}/health`, { cache: "no-store" });
    if (response.ok) {
      const runtime = await response.json();
      latestRuntime = runtime;
      clockPair = {browser_request_epoch_ms:requestEpochMs,browser_receive_epoch_ms:Date.now(),
        server_epoch_ms:runtime.server_epoch_ms,server_monotonic_ns:runtime.server_monotonic_ns};
      resetSceneButton.hidden = false;
      sceneRecording = runtime.recording === true || runtime.saving_episode === true;
      practiceHint.hidden = sceneRecording;
      const nextContext = `${runtime.session ?? ""}:${runtime.control_epoch ?? ""}`;
      if (backendContext !== null && backendContext !== nextContext) disarmHand("scene_reset");
      if (backendContext !== nextContext) gripperOrientationFinalized = false;
      if (runtime.gripper_yaw_finalized === true) {
        gripperOrientationFinalized = true;
        armMotionLocked = false;
        pointingFinalizationPending = false;
      }
      gripperClosed = runtime.gripper_closed === true;
      backendContext = nextContext;
      if (rearmGate.armed && tokenAtRequest &&
          generationAtRequest === controlGeneration && tokenAtRequest === controlToken &&
          (runtime.paused || runtime.control_token !== controlToken)) {
        // A late pause or dropped resume must not leave the UI armed forever.
        const candidate = rearmGate.candidate;
        disarmHand("backend_control_mismatch");
        rearmGate.candidate = candidate;
        robotFeedback.textContent = "正在重新同步控制；无需换手势或重新入镜。";
      }
      if (sceneResetInFlight && nextContext !== pendingResetContext) {
        sceneResetInFlight = false;
        pendingResetContext = null;
      }
      if (runtime.recording_error) {
        sceneResetInFlight = false;
        pendingResetContext = null;
        robotFeedback.textContent = `保存未完成：${runtime.recording_error}。当前录制保留，请重试保存。`;
      }
      document.querySelector("#preview-grid").dataset.backend = "MuJoCo";
      document.querySelector("#side-preview-title").textContent = "俯视 + 侧视 + 正面";
      robotTaskMode = runtime.task_mode ?? "pick_place";
      robotZLocked = runtime.z_locked === true;
      const xyErrorCm = ((runtime.grasp_xy_error_m ?? 0) * 100).toFixed(1);
      const controlHint = runtime.paused ? "控制暂停／等待手部识别" : "控制已恢复";
      const graspHint = runtime.gripper_closed && runtime.holding_cube
        ? "双侧接触红块，可手动抬升"
        : runtime.gripper_closed ? "夹爪已关闭；握拳上下移动控制升降"
        : `手动抓取，距红块中心 ${xyErrorCm} cm；没有自动对齐`;
      document.querySelector("#preview-help").textContent =
        `${controlHint}；${runtime.motion_limit_reason || graspHint}`;
      document.querySelector("#preview-grid").dataset.grasp = runtime.gripper_approaching_cube ? "near" : "far";
      if (robotTaskMode === "push_t") {
        robotHelp.textContent = "Push-T：夹爪始终关闭。ILoveYou 控制 XY；Victory 左右、Thumb_Up 前后，平面移动时锁住当前高度。张掌或握拳时，上下移动手控制升降，XY 保持；抬起后切回平面手势绕到新的接触点，再下降继续推。最低下降到初始接触高度。";
        openPalmRule.textContent = "夹爪保持闭合 / 手上下控制升降";
        closedFistRule.textContent = "夹爪保持闭合 / 手上下控制升降";
      } else {
        robotHelp.textContent = pickPlaceHelp;
        openPalmRule.textContent = "松开夹爪 / 手上下控制升降";
        closedFistRule.textContent = "保持 50 ms 闭爪 / 手上下控制升降";
      }
    }
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

  if (!armMotionLocked && !gripperOrientationFinalized &&
      controlToken && ["open", "close", "start_vertical", "start_planar", "move_xy",
        "up", "down", "left", "right", "forward", "backward"].includes(command)) {
    // The first manipulation command ends the initial orientation window.
    // Do this before the request so a delayed health poll cannot reopen it.
    gripperOrientationFinalized = true;
  }

  try {
    lastCommandTimestamp = Math.max(Date.now(), lastCommandTimestamp + 0.001);
    const packet = {
      command, sentAt:lastCommandTimestamp,event_id:crypto.randomUUID(),
      measurement_id:currentMeasurementId,source_context:backendContext,...extra,
      ...(controlToken && ["open","close","start_vertical","start_planar","move_xy","set_gripper_yaw","finalize_gripper_yaw","up","down","left","right","forward","backward"].includes(command) ? {controlToken} : {}),
    };
    measurementJournal.add("dispatch",{event_id:packet.event_id,measurement_id:packet.measurement_id,
      command:packet.command,packet},researchContext());
    const response = await fetch(`${robotApiBase}/control`, {
      method: "POST",
      signal: AbortSignal.timeout(400),
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(packet),
    });
    const result = await response.json();
    measurementJournal.add("control_result",{event_id:packet.event_id,http_status:response.status,
      response:result,semantics:"HTTP queued acknowledgement is not physical execution"},researchContext());
    if (!response.ok || result.ok === false || result.accepted === false) {
      robotFeedback.textContent = `操作未执行：${result.reason ?? result.error ?? response.status}`;
      return false;
    }
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
      if (request.generation !== controlGeneration) continue;
      await sendRobotCommand(request.command, request.extra);
    }
  } finally {
    motionRequestInFlight = false;
  }
}

function sendLatestMotionCommand(command, extra = {}) {
  pendingMotionRequest = { command, extra:{measurement_id:currentMeasurementId,...extra}, generation: controlGeneration };
  void flushLatestMotionRequest();
}

function processRobotControl(result, timestamp) {
  if (robotConnected && timestamp - lastGestureReportAt >= 30) {
    const recognized = result.gestures?.[0]?.[0];
    const visible = Boolean(result.landmarks?.length);
    const rawGesture = recognized?.categoryName ?? "None";
    const wrist = result.landmarks?.[0]?.[0];
    const motionGesture = isPushingTask()
      ? ["Victory", "Thumb_Up", "ILoveYou"].includes(rawGesture)
      : ["Closed_Fist", "Open_Palm", "Victory", "Thumb_Up", "ILoveYou"].includes(rawGesture);
    void sendRobotCommand("gesture_state", {
      gesture: visible ? rawGesture : "None",
      gestureScore: visible ? (recognized?.score ?? 0) : 0,
      handVisible: visible,
      wristUv: Number.isFinite(wrist?.x) && Number.isFinite(wrist?.y)
        ? [wrist.x, wrist.y] : null,
      wristCapturedAt: performance.timeOrigin + timestamp,
      handMotionReady: Boolean(rearmGate.armed && controlToken && !armMotionLocked && motionGesture),
      handMotionSegment: controlGeneration,
      handMotionToken: controlToken,
      pointingControlActive: visible && rearmGate.armed && !gripperOrientationFinalized && !pointingFinalizationPending && gripperClosed !== true && pointingControlCandidate(rawGesture, result.landmarks[0], { active: armMotionLocked }),
    });
    lastGestureReportAt = timestamp;
  }
  if (!robotConnected) {
    disarmHand("backend_disconnected");
    lastFistY = null;
    lastOpenPalmY = null;
    lastPlanarPoint = null;
    lastPlanarGesture = null;
    pendingMotionRequest = undefined;
    return;
  }
  if (sceneResetInFlight) {
    pendingMotionRequest = undefined;
    return;
  }

  // Re-entry is eligible immediately. The backend must still acknowledge a
  // fresh token, and the first visible position establishes a new motion anchor.
  let resumedThisFrame = false;
  const visibleWrist = result.landmarks?.[0]?.[0];
  const handVisibleForControl = Number.isFinite(visibleWrist?.x) && Number.isFinite(visibleWrist?.y);
  if (!handVisibleForControl) {
    // gesture_state(handVisible=false) invalidates the backend token, even
    // during the entry grace period. Invalidate our acknowledgement as well.
    // Preserve only the entry clock across a short recognition dropout.
    const waitingCandidate = !rearmGate.armed ? rearmGate.candidate : null;
    disarmHand();
    rearmGate.candidate = waitingCandidate;
  }
  const safety = rearmGate.observe(result, timestamp);
  if (safety.state !== "armed") {
    if (safety.state === "lost") {
      disarmHand();
      robotFeedback.textContent = "手已离开画面：机械臂暂停。";
    } else {
      if (handVisibleForControl && !resumePending && !resumeConfirmed && rearmGate.candidate) {
        const generation = controlGeneration;
        const candidate = rearmGate.candidate;
        const nextToken = crypto.randomUUID();
        resumePending = true;
        void queueSafetyCommand("resume", { controlToken: nextToken, controlContext: backendContext }).then((accepted) => {
          if (generation !== controlGeneration) return;
          if (accepted && rearmGate.candidate === candidate) {
            resumePending = false;
            resumeConfirmed = true;
            pendingResumeToken = nextToken;
          } else {
            disarmHand("resume_failed");
          }
        });
      }
      if (safety.state === "ready" && resumeConfirmed) {
        controlToken = pendingResumeToken;
        resumeConfirmed = false;
        pendingResumeToken = null;
        rearmGate.arm();
        resumedThisFrame = true;
        // Treat the visible gesture as a fresh gesture after re-entry. The
        // Re-entry and the existing gesture confirmation run concurrently;
        // do not impose another open/close cooldown on the same held gesture.
        lastRobotGesture = "None";
        const entryGesture = rearmGate.candidate.gesture;
        const entrySince = rearmGate.candidate.gestureSince;
        closedFistSince = entryGesture === "Closed_Fist" ? entrySince : Number.NEGATIVE_INFINITY;
        openPalmSince = null;
        if (armMotionLocked && entryGesture !== "Pointing_Up" && entryGesture !== "None") {
          pointingExitGesture = entryGesture;
          pointingExitSince = entrySince;
        }
        lastOpenSentAt = timestamp - GRIPPER_OPEN_RETRY_MS;
        lastCloseSentAt = timestamp - GRIPPER_CLOSE_RETRY_MS;
        robotFeedback.textContent = "控制已恢复；保持当前手势即可继续操作。";
      } else if (safety.state === "ready") {
        robotFeedback.textContent = "已检测到手，正在恢复仿真控制…";
      } else {
        robotFeedback.textContent = "等待手部识别，机械臂暂停。";
      }
    }
    if (!rearmGate.armed) {
      lastFistY = null;
      lastOpenPalmY = null;
      lastPlanarPoint = null;
      lastPlanarGesture = null;
      pendingMotionRequest = undefined;
      pushingPlanarGestureStartedAt = Number.NEGATIVE_INFINITY;
      return;
    }
    // Process the current gesture on the confirmation frame itself.
  }

  const topGesture = result.gestures?.[0]?.[0];
  const rawGesture = topGesture?.categoryName ?? "None";
  const wrist = result.landmarks[0][0];
  const pointingCandidate = !gripperOrientationFinalized && !pointingFinalizationPending && gripperClosed !== true && pointingControlCandidate(rawGesture, result.landmarks[0], { active: armMotionLocked });
  const gesture = pointingCandidate ? "Pointing_Up" : rawGesture;
  const gestureChanged = gesture !== lastRobotGesture;
  if (pointingCandidate) {
    armMotionLocked = true;
    pointingExitGesture = null;
  } else if (armMotionLocked && !pointingFinalizationPending && rawGesture !== "None" && rawGesture !== "Pointing_Up") {
    if (pointingExitGesture !== rawGesture) {
      pointingExitGesture = rawGesture;
      pointingExitSince = timestamp;
    } else if (timestamp - pointingExitSince >= POINTING_EXIT_HOLD_MS) {
      pointingFinalizationPending = true;
      pointingExitGesture = null;
      pendingMotionRequest = undefined;
      const generation = controlGeneration;
      void sendRobotCommand("finalize_gripper_yaw", { angle: lastPointingAngle ?? 0 }).then((accepted) => {
        if (generation !== controlGeneration) return;
        pointingFinalizationPending = false;
        if (accepted) {
          gripperOrientationFinalized = true;
          armMotionLocked = false;
          robotFeedback.textContent = "夹爪朝向已确认，本场景保持固定；当前手势可控制机械臂移动。";
        }
      });
    }
  } else {
    pointingExitGesture = null;
  }
  if (isPushingTask()) {
    processPushingControl({ gesture, gestureChanged, wrist, timestamp, reentered: resumedThisFrame });
    lastRobotGesture = gesture;
    return;
  }

  if (armMotionLocked && pointingCandidate) {
    if (lastPointingAngle === null) pendingMotionRequest = undefined;
    const fingerAngle = pointingUpAngle(result.landmarks?.[0]);
    const angle = pointingYawFromWrist(fingerAngle);
    if (angle !== null && (lastPointingAngle === null || Math.abs(angle - lastPointingAngle) >= 0.025 || timestamp - lastPointingSentAt >= 150)) {
      sendLatestMotionCommand("set_gripper_yaw", { angle });
      lastPointingAngle = angle;
      lastPointingSentAt = timestamp;
      robotFeedback.textContent = `Pointing_Up：夹爪旋转 ${Math.round(angle * 180 / Math.PI)}°；换成任一稳定手势后固定朝向并恢复移动。`;
    }
  } else if (!armMotionLocked) {
    lastPointingAngle = null;
    lastPointingSentAt = Number.NEGATIVE_INFINITY;
  }

  if (gesture === "ILoveYou") {
    lastDiagonalGestureAt = timestamp;
  }
  const releaseHandClear = handClearForRelease(result.landmarks?.[0], topGesture?.score);
  if (gesture === "Open_Palm" && releaseHandClear) {
    if (openPalmSince === null) openPalmSince = timestamp;
  } else {
    openPalmSince = null;
  }
  const openPalmControlAllowed =
    gesture === "Open_Palm" &&
    openPalmAllowedAfterDiagonal({ now: timestamp, lastDiagonalAt: lastDiagonalGestureAt });

  if (gesture === "Closed_Fist" && !armMotionLocked &&
      (!Number.isFinite(closedFistSince))) {
    closedFistSince = timestamp;
    void sendRobotCommand("start_vertical");
  }
  if (gesture === "Closed_Fist" && !armMotionLocked) {
    if (lastFistY === null) {
      lastFistY = wrist.y;
    } else {
      const verticalDelta = wrist.y - lastFistY;
      if (verticalDelta < -verticalDeadZone()) {
        sendLatestMotionCommand("up");
        lastFistY = wrist.y;
      } else if (verticalDelta > verticalDeadZone()) {
        sendLatestMotionCommand("down");
        lastFistY = wrist.y;
      }
    }
  } else {
    lastFistY = null;
    closedFistSince = Number.NEGATIVE_INFINITY;
  }

  // Closing never changes the arm target; position is always operator-controlled.
  const fistConfirmed = timestamp - closedFistSince >= GRIPPER_FIST_CONFIRM_MS;
  if (gesture === "Closed_Fist" && gripperClosed !== true) {
    if (!armMotionLocked && !fistConfirmed) {
      robotFeedback.textContent = "握拳已识别：保持 50 ms 后尝试抓取。";
    }
  }
  if (
    !armMotionLocked &&
    timestamp - lastCloseSentAt >= GRIPPER_CLOSE_RETRY_MS &&
    shouldCloseGripper({
      taskMode: robotTaskMode,
      gesture,
      gripperClosed,
      fistConfirmed,
    })
  ) {
    lastCloseSentAt = timestamp;
    void sendRobotCommand("close").then((accepted) => {
      if (accepted) {
        robotFeedback.textContent = "握拳：闭爪指令已发送，等待仿真确认。";
      }
    });
  }

  if (
    gesture === "Open_Palm" &&
    !armMotionLocked &&
    openPalmControlAllowed &&
    gripperClosed !== false && releaseHandClear &&
    openPalmReadyToRelease({ now: timestamp, openPalmSince }) &&
    (lastOpenPalmY === null || timestamp - lastOpenSentAt >= GRIPPER_OPEN_RETRY_MS)
  ) {
    const generation = controlGeneration;
    void sendRobotCommand("open").then((accepted) => {
      if (accepted && generation === controlGeneration) gripperClosed = false;
    });
    lastOpenSentAt = timestamp;
    robotFeedback.textContent = "张开手掌：夹爪松开；上下移动手可直线升降。";
  }
  // Releasing the gripper keeps its diagonal-transition debounce, while
  // vertical motion starts as soon as Open_Palm is recognized. Otherwise the
  // first movement would otherwise be lost on the gesture switch.
  if (gesture === "Open_Palm" && lastOpenPalmY === null && !armMotionLocked) {
    void sendRobotCommand("start_vertical");
  }
  if (gesture === "Open_Palm" && !armMotionLocked) {
    if (lastOpenPalmY === null) {
      lastOpenPalmY = wrist.y;
    } else {
      const verticalDelta = wrist.y - lastOpenPalmY;
      if (verticalDelta < -verticalDeadZone()) {
        sendLatestMotionCommand("up");
        lastOpenPalmY = wrist.y;
      } else if (verticalDelta > verticalDeadZone()) {
        sendLatestMotionCommand("down");
        lastOpenPalmY = wrist.y;
      }
    }
  } else {
    lastOpenPalmY = null;
  }

  // Brief recognition flicker must not switch a diagonal move onto a single
  // axis. Explicit palm/fist vertical gestures bypass this label hold;
  // gripper open/close confirmation is handled independently above.
  const planarGesture = !armMotionLocked
    ? planarGestureAfterDiagonal(gesture, timestamp, lastDiagonalGestureAt, DIAGONAL_GESTURE_HOLD_MS)
    : gesture;
  const isPlanarGesture =
    planarGesture === "Victory" ||
    planarGesture === "Thumb_Up" ||
    planarGesture === "ILoveYou";
  if (isPlanarGesture) {
    const planarPoint = wrist;
    const planarGestureChanged = planarGesture !== lastPlanarGesture;
    if (planarGestureChanged) {
      lastPlanarPoint = { x: planarPoint.x, y: planarPoint.y };
    }
    if (shouldStartPlanarLock({
      gesture: planarGesture,
      gestureChanged: planarGestureChanged,
      carrying: gripperClosed === true,
    })) {
      void sendRobotCommand("start_planar", {
        lockZ: planarGesture === "ILoveYou" && gesture === "ILoveYou",
      });
      robotFeedback.textContent = "平面移动已接管：移动手控制方向，夹爪保持当前高度。";
    }

    let moved = false;
    if (lastPlanarPoint && timestamp - lastControlSentAt >= CONTROL_INTERVAL_MS) {
      const screenDx = planarPoint.x - lastPlanarPoint.x;
      const screenDy = planarPoint.y - lastPlanarPoint.y;
      let { dx, dy } = mapPlanarMotion({
        gesture: planarGesture,
        screenDx,
        screenDy,
        sensitivity: planarSensitivity(),
        carrying: gripperClosed === true,
        reverseVictory: true,
        allowEmptyDiagonal: true,
      });

      const magnitude = Math.hypot(dx, dy);
      if (magnitude > 1) {
        dx /= magnitude;
        dy /= magnitude;
      }
      if (magnitude > planarDeadZone()) {
        sendLatestMotionCommand("move_xy", {
          dx,
          dy,
          lockZ: planarGesture === "ILoveYou" && gesture === "ILoveYou",
        });
        lastControlSentAt = timestamp;
        moved = true;
        const messages = {
          Victory: "Victory：正在左右移动。",
          Thumb_Up: "竖起拇指：正在前后移动。",
          ILoveYou: "ILoveYou：正在斜向移动。",
        };
        robotFeedback.textContent = messages[planarGesture];
      }
    }
    lastPlanarPoint = nextPlanarAnchor(
      lastPlanarPoint, planarPoint, moved, true,
    );
    lastPlanarGesture = planarGesture;
  } else {
    lastPlanarPoint = null;
    lastPlanarGesture = null;
  }

  lastRobotGesture = gesture;
}

function processPushingControl({ gesture, gestureChanged, wrist, timestamp, reentered = false }) {
  // Palm/fist change height with the same closed pushing surface. Returning
  // to a planar gesture captures the new height, not the initial contact plane.
  const isVerticalGesture = gesture === "Open_Palm" || gesture === "Closed_Fist";
  if (isVerticalGesture) {
    lastPlanarPoint = null;
    pushingPlanarGestureStartedAt = Number.NEGATIVE_INFINITY;
    if (gestureChanged || reentered || lastOpenPalmY === null) {
      pendingMotionRequest = undefined;
      void sendRobotCommand("start_vertical");
      robotZLocked = false;
      lastOpenPalmY = wrist.y;
    } else if (timestamp - lastControlSentAt >= CONTROL_INTERVAL_MS) {
      const delta = wrist.y - lastOpenPalmY;
      if (Math.abs(delta) > verticalDeadZone()) {
        sendLatestMotionCommand(delta < 0 ? "up" : "down");
        lastOpenPalmY = wrist.y;
        lastControlSentAt = timestamp;
      }
    }
    robotFeedback.textContent = "Push-T：夹爪保持关闭，上下移动手调整高度；抬起后可换接触点。";
    return;
  }
  lastOpenPalmY = null;

  const isPlanarGesture =
    gesture === "Victory" || gesture === "Thumb_Up" || gesture === "ILoveYou";
  if (!isPlanarGesture) {
    lastPlanarPoint = null;
    pushingPlanarGestureStartedAt = Number.NEGATIVE_INFINITY;
    if (robotZLocked) {
      robotFeedback.textContent = "Push-T：固定高度已锁定，等待平面移动手势。";
    }
    return;
  }

  if (gestureChanged) {
    pendingMotionRequest = undefined;
    void sendRobotCommand("start_planar", { lockZ: true });
    robotZLocked = true;
    // A resumed hand establishes a fresh anchor without another entry delay.
    // Ordinary in-frame gesture changes retain their existing confirmation.
    pushingPlanarGestureStartedAt = reentered ? Number.NEGATIVE_INFINITY : timestamp;
    lastPlanarPoint = { x: wrist.x, y: wrist.y };
  }
  if (lastPlanarPoint && timestamp - lastControlSentAt >= CONTROL_INTERVAL_MS) {
    const screenDx = wrist.x - lastPlanarPoint.x;
    const screenDy = wrist.y - lastPlanarPoint.y;
    let { dx, dy } = mapPlanarMotion({
      gesture,
      screenDx,
      screenDy,
      sensitivity: planarSensitivity(),
      carrying: true,
      reverseVictory: true,
    });
    const magnitude = Math.hypot(dx, dy);
    if (magnitude > 1) {
      dx /= magnitude;
      dy /= magnitude;
    }
    const gestureIsStable = timestamp - pushingPlanarGestureStartedAt >= 250;
    if (gestureIsStable && magnitude > planarDeadZone()) {
      if (!robotZLocked) {
        void sendRobotCommand("start_planar", { lockZ: true });
        robotZLocked = true;
      }
      sendLatestMotionCommand("move_xy", { dx, dy, lockZ: true });
      lastControlSentAt = timestamp;
    }
  }
  lastPlanarPoint = { x: wrist.x, y: wrist.y };
  const messages = robotZLocked
    ? {
        Victory: "Push-T：固定高度已锁定，正在左右移动。",
        Thumb_Up: "Push-T：固定高度已锁定，正在前后移动。",
        ILoveYou: "Push-T：固定高度已锁定，正在斜向移动。",
      }
    : {
        Victory: "Push-T：正在左右移动。",
        Thumb_Up: "Push-T：正在前后移动。",
        ILoveYou: "Push-T：正在斜向移动。",
      };
  robotFeedback.textContent = messages[gesture];
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
      recordRawMeasurement(result, timestamp, performance.now());
      drawHandLandmarks(result);
      renderRecognitionResults(result);
      processRobotControl(result, timestamp);
      lastVideoTime = video.currentTime;
    } catch (error) {
      measurementJournal.add("camera_event",{event:"recognition_error",message:String(error)},researchContext());
      disarmHand("recognition_error");
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
    const settings = mediaStream.getVideoTracks()[0].getSettings();
    cameraResearchSettings = Object.fromEntries(["width","height","frameRate","facingMode","resizeMode"].map(k=>[k,settings[k]??null]));
    measurementJournal.add("camera_event",{event:"camera_started",camera:cameraResearchSettings},researchContext());

    cameraPlaceholder.hidden = true;
    cameraButton.textContent = "关闭摄像头";
    cameraButton.classList.remove("is-loading");
    cameraButton.disabled = false;
    lastVideoTime = -1;
    disarmHand("camera_started");
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
  measurementJournal.add("camera_event",{event:"camera_stopped",measurement_id:currentMeasurementId},researchContext());
  currentMeasurementId = null;
  flushMeasurements();
  disarmHand("camera_stopped");
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
  lastPlanarGesture = null;
  gripperClosed = null;
  lastRobotGesture = "None";
  pendingMotionRequest = undefined;
}

cameraButton.addEventListener("click", () => {
  if (mediaStream) stopCamera();
  else startCamera();
});

resetSceneButton.addEventListener("click", async () => {
  if (sceneRecording || sceneResetInFlight || !robotConnected) return;
  sceneResetInFlight = true;
  pendingResetContext = backendContext;
  resetSceneButton.disabled = true;
  disarmHand("scene_reset");
  await safetyTransport.catch(() => false);
  let accepted = false;
  try {
    accepted = await sendRobotCommand("reset");
    if (accepted) {
      for (let attempt = 0; attempt < 50 && sceneResetInFlight; attempt += 1) {
        await new Promise((resolve) => window.setTimeout(resolve, 100));
        await checkRobotConnection();
      }
      robotFeedback.textContent = sceneResetInFlight
        ? "重置请求已发送，等待仿真确认。"
        : "场景已重置。保持手在画面中 0.5 秒后可继续练习。";
    }
  } finally {
    if (sceneResetInFlight && (!robotConnected || !accepted)) {
      sceneResetInFlight = false;
      pendingResetContext = null;
    }
    resetSceneButton.disabled = !robotConnected || sceneRecording;
  }
});

recordStartButton.addEventListener("click", async () => {
  if (sceneResetInFlight) return;
  if (await sendRobotCommand("record_start")) {
    sceneRecording = true;
    resetSceneButton.disabled = true;
    practiceHint.hidden = true;
    robotFeedback.textContent = "正在录制当前 demonstration。";
  }
});

async function saveWithResearch(outcome) {
  if (saveFlowInFlight) return false;
  saveFlowInFlight = true;
  try {
    // Do not reuse a previous episode's seal from the periodic health cache.
    const current = await fetch(`${robotApiBase}/health`,{cache:"no-store",signal:AbortSignal.timeout(1500)});
    if (!current.ok) throw Error("无法确认当前录制状态");
    latestRuntime = await current.json();
    if (!latestRuntime.recording) throw Error("当前没有正在录制的 demonstration");
    if (latestRuntime?.hand_journal_schema !== "single-hand-journal-v1") throw Error("请先重启采集程序并刷新网页");
    let seal = latestRuntime.episode_seal;
    const saveId = seal?.save_id ?? crypto.randomUUID();
    if (seal && seal.outcome !== outcome) throw Error("本轮已冻结为另一保存结果，请按原结果重试");
    if (!seal && !await sendRobotCommand("record_seal",{save_id:saveId,outcome})) return false;
    const deadline = performance.now()+5000;
    while (performance.now()<deadline) {
      const response = await fetch(`${robotApiBase}/health`,{cache:"no-store",signal:AbortSignal.timeout(1500)});
      const runtime = await response.json();
      if (runtime.episode_seal?.save_id===saveId) {seal=runtime.episode_seal;break;}
      await new Promise(resolve=>setTimeout(resolve,40));
    }
    if (!seal) throw Error("等待录制终点超时；本轮数据保留，可重试保存");
    robotFeedback.textContent = "录制终点已固定，正在确认原始日志完整写入…";
    const boundary = measurementJournal.freeze(seal.end_epoch_ms);
    const durable = await measurementJournal.flushBoundary(boundary);
    return await sendRobotCommand(outcome==="success"?"record_save":"record_failure",
      {save_id:saveId,hand_save_boundary:durable});
  } catch (error) {
    robotFeedback.textContent = `保存未完成：${error.message}。当前录制保留，请重试。`;
    return false;
  } finally {saveFlowInFlight=false;}
}

recordSaveButton.addEventListener("click", async () => {
  if (await saveWithResearch("success")) {
    disarmHand("scene_reset");
    sceneResetInFlight = true;
    pendingResetContext = backendContext;
    sceneRecording = false;
    resetSceneButton.disabled = !robotConnected || sceneResetInFlight;
    practiceHint.hidden = false;
    robotFeedback.textContent = "已请求保存成功 demonstration，仿真将自动重置。";
  }
});

recordFailureButton.addEventListener("click", async () => {
  if (await saveWithResearch("failure")) {
    disarmHand("scene_reset");
    sceneResetInFlight = true;
    pendingResetContext = backendContext;
    sceneRecording = false;
    resetSceneButton.disabled = !robotConnected || sceneResetInFlight;
    practiceHint.hidden = false;
    robotFeedback.textContent =
      "失败 demonstration 已保存到独立失败数据集，仿真将自动重置。";
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
window.setInterval(checkRobotConnection, 200);
window.setInterval(flushMeasurements, 200);

window.addEventListener("pagehide", () => {
  window.clearTimeout(sidePreviewPollTimer);
  hideSidePreview();
  stopCamera();
  gestureRecognizer?.close();
});
