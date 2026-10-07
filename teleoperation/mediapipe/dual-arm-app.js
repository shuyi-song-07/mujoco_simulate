import { DrawingUtils, GestureRecognizer } from "./node_modules/@mediapipe/tasks-vision/vision_bundle.mjs";
import { DEFAULT_DUAL_CONFIG, DualInteractionState, detectionsFromMediaPipe, resolveOperation } from "./dual-interaction-state.js";

const $ = (id) => document.getElementById(id);
const api = "/api/dual", video = $("webcam"), canvas = $("output-canvas"), context = canvas.getContext("2d");
const gestures = { ILoveYou: "I Love You", Victory: "剪刀手", Closed_Fist: "握拳", Open_Palm: "张掌", Pointing_Up: "食指向上", Thumb_Up: "竖拇指", None: "未识别" };
const modes = { XY: "XY 平移", Z: "Z 升降", PAUSE: "暂停", IDLE: "等待", GRASP_CONFIRM: "闭合确认", RELEASE_CONFIRM: "打开确认" };
const tasks = { PREGRASP: "对准待抓", GRASP_VERIFY: "验证双侧抓取", DUAL_GRASPED: "协同搬运", PLACEMENT: "放置微调", RELEASE_VERIFY: "验证放置", DONE: "成功完成", FAIL: "失败" };
const failureLabels = { timeout: "超过正式任务时限", grasp_failure: "双侧抓取未建立", contact_loss: "搬运中失去接触", object_drop: "物体跌落", relative_pose_violation: "两臂相对位置超限", robot_robot_collision: "两臂发生碰撞", workspace_violation: "工作空间越界", joint_or_ik_failure: "关节或逆解异常" };
const viewLabels = { top: "正上方 · XY 对准（↑ −X，→ +Y）", side: "斜侧方 · 两侧接触辅助", front: "内侧近景 · 对准与高度" };
const reasons = {
  calibration_required: "双手张掌稳定 2 秒后直接绑定，无需确认。",
  binding_complete: "左右手已绑定；持续张掌保持暂停及夹爪状态。换成操作手势后开始控制。",
  binding_gesture_released: "已离开绑定手势；等待操作手势稳定后重新建立移动基准。",
  binding_released: "已解除左右手绑定，双臂暂停、夹爪保持。双手张掌稳定 2 秒可重新绑定。",
  mapping_changed: "左右标签已交换，原绑定已解除；请双手张掌稳定 2 秒重新绑定。",
  config_updated: "控制配置已更新；已有绑定保留，等待双手稳定后重新建立移动基准。",
  tracking_recovery: "双手重新出现，等待稳定追踪后重新建立移动基准。",
  tracking_recovered: "双手身份已稳定恢复，旧移动基准已清除；继续保持操作手势建立新基准。",
  tracking_stalled: "识别时间没有推进，绑定保留、动作暂停；等待新的双手画面。",
  clock_reversal: "识别时间顺序异常，绑定保留、动作暂停；等待新的双手画面。",
  hand_lost: "至少一只手离开画面；两臂暂停，夹爪保持当前状态。",
  ambiguous_handedness: "左右身份暂不明确，绑定保留、双臂暂停。请分开双手并完整入镜。",
  identity_conflict: "位置与左右身份冲突，绑定保留、双臂暂停。请分开双手，恢复正确的左右身份。",
  motion_low_confidence: "移动手势评分不足，动作暂停；请保持双手使用同一移动手势。",
  unassigned_gesture: "当前手势未分配操作，动作暂停；请双手使用 I Love You、剪刀手、握拳或张掌。",
  gesture_disagreement: "双手需要使用同一种移动手势，暂时暂停。",
  motion_confirmation: "正在确认移动模式；确认后第一帧仅建立基准。",
  gripper_confirmation: "保持双手同为握拳或张掌，等待两侧确认进度完成。",
  gripper_pending: "已提交双夹爪事件，等待仿真真实状态，不重复发送。",
  release_waiting_for_stop: "等待物体与夹爪停稳，保持双手张掌；重新确认后再尝试释放。",
  gripper_already_latched: "夹爪已经处于该状态，切换 XY / Z 手势继续操作。",
  gripper_rejected_change_gesture: "抓放未被执行。先离开当前手势，检查安全提示后再重试。",
  backend_state_unknown: "等待仿真返回真实夹爪状态。",
  backend_disconnected: "Task 3 仿真未连接，请运行 START-TASK3.cmd。",
  saving: "正在写入数据，请等待保存完成。", task_done: "任务已自动判定成功，可以保存成功示范。",
  camera_stopped: "摄像头已关闭，绑定已解除。再次启动后双手张掌稳定 2 秒重新绑定。",
  camera_error: "摄像头或识别异常，两臂暂停。", camera_stalled: "摄像头画面停止更新，两臂暂停；请检查摄像头。", user_pause: "已暂停。把双手放回舒适位置，再使用同一移动手势。",
  tab_hidden: "页面离开前台，机器人暂停。回来后重新建立移动基准。",
  operation_pending: "等待仿真完成操作；双臂暂停，暂不接受新的移动或抓放手势。",
  scene_reset: "场景已更新，旧移动基准已清除；保持双手静止后重新进入 XY / Z 模式。",
  backend_session_changed: "仿真已重新启动；绑定保留，旧指令已清除。等待双手稳定后重新建立移动基准。",
  service_stopped: "退出请求已提交，仿真连接已关闭。需要继续采集时重新运行 START-TASK3.cmd。",
  control_transport_error: "控制通道暂时繁忙或超时，两臂暂停；等待健康检查恢复后重新建立移动基准。",
  transport_recovery: "控制通道已恢复；旧动作已丢弃，请保持双手静止后重新进入移动模式。",
  stale_recognition: "识别结果延迟较大，已丢弃旧手势并暂停；等待新的摄像头画面。",
  unexpected_frame_id: "识别结果不属于当前摄像头请求，已丢弃并暂停；等待当前画面。",
  command_watchdog: "仿真未及时收到有效控制，已暂停；等待新识别结果后恢复。",
  task_motion_frozen: "仿真当前阶段暂不接受移动。",
};
let engine = new DualInteractionState(), output = engine.pause("camera_stopped");
let gestureWorker, workerReady = false, workerFrameBusy = false, workerFrameId = 0, workerGeneration = 0;
let drawing, stream, frameId, healthTimer, previewTimer;
let runtime = {}, connected = false, lastVideoTime = -1, lastPacketAt = -Infinity;
let lastHealthReceivedAt = -Infinity;
let latestPacket = null, sending = false, events = [], configSignature = "", mainView = "top";
let viewPreference = "auto";
let pendingOperation = null, controlEpoch = null, controlSession = null, serviceStopped = false, stoppedSession = null;
const previews = { top: null, side: null, front: null };
const previewsUpdatedAt = { top: -Infinity, side: -Infinity, front: -Infinity };
let lastFrameSubmittedAt = -Infinity, lastUiRenderAt = -Infinity, transportFrozen = false, transportFrozenAt = -Infinity;
let lastDetections = [], lastRecognitionIssue = null, lastReceivedResultAt = -Infinity;
let controlPumpTimer, lastContinuousPostAt = -Infinity;
const PREVIEW_GRACE_MS = 1000;
const RECOGNITION_MAX_AGE_MS = 250;
const recognitionClientId = crypto.randomUUID();

function notice(message, active = false) { $("notice").textContent = message; $("notice").dataset.state = active ? "active" : "warning"; }
function updateButtons() {
  const recording = runtime.recording === true, saving = runtime.saving === true;
  for (const button of document.querySelectorAll("[data-command]")) {
    const command = button.dataset.command;
    const cannotStart = recording || !engine.tracker.calibrated || (!["PREGRASP", "DONE", "FAIL"].includes(runtime.task_state)) || (runtime.task_state === "PREGRASP" && engine.gripperLatch !== "OPEN");
    button.disabled = !connected || transportFrozen || saving || Boolean(pendingOperation) || (command === "record_start" && cannotStart) || (["record_save", "record_failure", "record_discard"].includes(command) && !recording) || (command === "record_save" && runtime.task_state !== "DONE") || (["reset", "record_stop"].includes(command) && recording);
  }
}
function setConnected(value) {
  connected = value;
  $("connection").textContent = value ? "双臂仿真已连接" : "双臂仿真未连接";
  $("connection").dataset.state = value ? "active" : "error";
  if (!value) {
    resolvePendingOperation(false);
    latestPacket = null; events = []; engine.gripperLatch = null;
    output = engine.pause(serviceStopped ? "service_stopped" : "backend_disconnected");
    notice(reasons[output.reason]);
  } else if (serviceStopped && runtime.session !== stoppedSession) serviceStopped = false;
  updateButtons(); renderControlStatus();
}
async function postPacket(packet) {
  const response = await fetch(`${api}/control`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(packet), signal: AbortSignal.timeout(2500) });
  let data = {}; try { data = await response.json(); } catch { /* Proxy error can be plain text. */ }
  if (!response.ok || data.ok === false) {
    const error = new Error(data.error || data.reason || `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return data;
}
async function flushPackets() {
  if (sending || transportFrozen) return;
  sending = true;
  try {
    while (connected && (events.length || latestPacket)) {
      const continuous = events.length === 0;
      if (continuous && performance.now() - lastContinuousPostAt < engine.config.control_interval_ms) {
        clearTimeout(controlPumpTimer);
        controlPumpTimer = setTimeout(flushPackets, engine.config.control_interval_ms - (performance.now() - lastContinuousPostAt));
        break;
      }
      const packet = events.length ? events.shift() : latestPacket;
      if (packet === latestPacket) latestPacket = null;
      if (packet.session !== undefined && packet.session !== runtime.session) continue;
      if (packet.command === "dual_motion" && Date.now() - packet.sentAt > (runtime.config?.control?.max_command_age_ms ?? 150)) continue;
      if (continuous) lastContinuousPostAt = performance.now();
      await postPacket(packet);
      if (pendingOperation && packet.eventId && pendingOperation.eventId === packet.eventId) pendingOperation.queued = true;
    }
  } catch (error) {
    // A rejected/slow control request does not mean that the health service died.
    latestPacket = null; events = [];
    clearTimeout(controlPumpTimer);
    transportFrozen = true; transportFrozenAt = performance.now();
    output = engine.pause("control_transport_error");
    notice(`控制已暂停：${error.message}。正在确认连接，旧动作不会重放。`);
    renderState();
  }
  finally { sending = false; }
}
function queueOutput(next, timestamp, immediate = false) {
  output = next;
  if (!connected) return;
  if (transportFrozen) return;
  if (next.command?.command === "dual_gripper") {
    latestPacket = null;
    events.push({ ...next.command, sentAt: next.measurementAt ?? Date.now(), control_epoch: runtime.control_epoch, session: runtime.session });
    lastPacketAt = timestamp;
    void flushPackets();
  } else {
    const incoming = { ...(next.command ?? { command: "telemetry", telemetry: next.telemetry }), sentAt: next.measurementAt ?? Date.now(), control_epoch: runtime.control_epoch, session: runtime.session };
    const pendingMotionFresh = latestPacket?.command === "dual_motion" && Date.now() - latestPacket.sentAt <= (runtime.config?.control?.max_command_age_ms ?? 150);
    if (incoming.command === "telemetry" && pendingMotionFresh && incoming.telemetry?.control_mode?.toLowerCase() === latestPacket.mode) {
      latestPacket.telemetry = { ...incoming.telemetry, user_delta: latestPacket.telemetry?.user_delta, training_recordable: true };
    } else latestPacket = incoming;
    if (immediate || next.command?.command === "dual_motion" || timestamp - lastPacketAt >= engine.config.control_interval_ms) {
      lastPacketAt = timestamp;
      void flushPackets();
    }
  }
}
function pause(reason) { queueOutput(engine.pause(reason), performance.now(), true); renderState(); }
function resolvePendingOperation(isConnected = true) {
  const result = resolveOperation(pendingOperation, runtime, performance.now(), isConnected);
  pendingOperation = result.pending;
  if (!result.outcome) return;
  if (result.outcome === "accepted") {
    $("record-feedback").textContent = { record_start: "录制已开始。", record_save: "仿真已确认成功保存。", record_failure: "失败示范已独立保存。", record_discard: "本轮未进入训练集，研究日志与快照保留为 rejected。", reset: "场景已重置，请重新建立移动基准。", record_stop: "采集程序已确认正常结束。" }[result.command];
    if (["reset", "record_save", "record_failure", "record_discard"].includes(result.command)) engine.resetScene();
    if (result.command === "record_stop") { serviceStopped = true; stoppedSession = runtime.session; }
  } else if (result.outcome === "rejected") $("record-feedback").textContent = `操作未执行：${result.reason ?? "条件不满足"}`;
  else if (result.outcome === "connection_closed") { serviceStopped = true; stoppedSession = runtime.session; $("record-feedback").textContent = "退出请求已提交，仿真连接已关闭。"; }
  else $("record-feedback").textContent = "等待仿真确认超时，请检查任务状态和日志，不要重复保存同一轮。";
}
async function checkHealth() {
  const requestStartedAt = performance.now();
  try {
    const response = await fetch(`${api}/health`, { cache: "no-store", signal: AbortSignal.timeout(2500) });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    runtime = await response.json();
    lastHealthReceivedAt = performance.now();
    const sessionChanged = controlSession !== null && runtime.session !== controlSession;
    if (sessionChanged) {
      latestPacket = null; events = [];
      clearTimeout(controlPumpTimer);
      if (pendingOperation) $("record-feedback").textContent = "仿真已重新启动；上次未确认的操作已清除，请检查当前录制状态。";
      pendingOperation = null;
      engine.resetScene();
      output = engine.pause("backend_session_changed");
    }
    resolvePendingOperation();
    const control = runtime.config?.control ?? {};
    const config = Object.fromEntries(Object.keys(DEFAULT_DUAL_CONFIG).filter((key) => key in control).map((key) => [key, control[key]]));
    const signature = JSON.stringify(config);
    if (signature !== configSignature) {
      configSignature = signature;
      latestPacket = null; events = events.filter((packet) => packet.command !== "dual_gripper");
      output = engine.reconfigure(config);
    }
    engine.syncHealth(runtime, performance.now());
    setConnected(true);
    if (transportFrozen && requestStartedAt >= transportFrozenAt) {
      transportFrozen = false;
      latestPacket = null; events = [];
      queueOutput(engine.pause("transport_recovery"), performance.now(), true);
    }
    if (sessionChanged || (controlEpoch !== null && runtime.control_epoch !== controlEpoch)) {
      latestPacket = null; events = [];
      let reset = engine.resetScene();
      if (sessionChanged) reset = engine.pause("backend_session_changed");
      queueOutput(reset, performance.now(), true);
    }
    controlEpoch = runtime.control_epoch;
    controlSession = runtime.session;
    renderState();
  } catch { setConnected(false); }
  finally { healthTimer = setTimeout(checkHealth, 350); }
}
function assignImage(id, placeholder, url) {
  const image = $(id);
  if (url) { if (image.src !== url) image.src = url; image.hidden = false; $(placeholder).hidden = true; }
  else { image.hidden = true; image.removeAttribute("src"); $(placeholder).hidden = false; }
}
function renderViews() {
  if (viewPreference !== "auto") mainView = viewPreference;
  else {
    if (output.mode === "XY") mainView = "top";
    if (output.mode === "Z") mainView = "front";
  }
  $("main-view-title").textContent = viewLabels[mainView];
  const visible = (view) => performance.now() - previewsUpdatedAt[view] <= PREVIEW_GRACE_MS ? previews[view] : null;
  assignImage("main-preview", "main-placeholder", visible(mainView));
  const auxiliary = ["top", "side", "front"].filter((view) => view !== mainView);
  assignImage("side-preview", "side-placeholder", visible(auxiliary[0]));
  assignImage("front-preview", "front-placeholder", visible(auxiliary[1]));
  $("auxiliary-view-one").textContent = viewLabels[auxiliary[0]];
  $("auxiliary-view-two").textContent = viewLabels[auxiliary[1]];
}
async function decodePreview(url) {
  const image = new Image();
  image.src = url;
  await image.decode();
}
async function pollPreviews() {
  if (connected) {
    const requestedEpoch = runtime.control_epoch, requestedSession = runtime.session;
    const status = await Promise.all(["top", "side", "front"].map(async (view) => {
      try {
        const response = await fetch(`${api}/${view}-preview`, { cache: "no-store", signal: AbortSignal.timeout(2000) });
        if (!response.ok || response.status === 204) throw new Error("preview unavailable");
        const next = URL.createObjectURL(await response.blob()), previous = previews[view];
        try {
          await decodePreview(next);
          if (runtime.control_epoch !== requestedEpoch || runtime.session !== requestedSession) throw new Error("scene changed during preview request");
        } catch (error) { URL.revokeObjectURL(next); throw error; }
        previews[view] = next;
        previewsUpdatedAt[view] = performance.now();
        renderViews();
        if (previous) URL.revokeObjectURL(previous);
        return true;
      } catch {
        if (performance.now() - previewsUpdatedAt[view] > PREVIEW_GRACE_MS) {
          const previous = previews[view];
          previews[view] = null;
          renderViews();
          if (previous) URL.revokeObjectURL(previous);
        }
        return false;
      }
    }));
    $("preview-state").textContent = status.every(Boolean) ? "三路仿真画面" : runtime.preview_error ? `渲染异常：${runtime.preview_error}` : "画面等待更新；最多短暂保留 1 秒";
  } else $("preview-state").textContent = "等待仿真连接";
  renderViews(); renderControlStatus(); previewTimer = setTimeout(pollPreviews, 150);
}
const percent = (value) => `${Math.floor((Number.isFinite(value) ? value : 0) * 1000 + 1e-8) / 10}%`;
const gripperThreshold = (action) => engine.config[`gripper_${action}_confidence_min`] ?? engine.config.gripper_confidence_min;
const gripperInstructions = () => `握拳闭合各至少 ${percent(gripperThreshold("close"))}，张掌打开各至少 ${percent(gripperThreshold("open"))}；双手保持约 ${(engine.config.gripper_confirm_ms / 1000).toFixed(1)} 秒。`;
function gripperDetail() {
  const failure = runtime.task_state === "FAIL" ? `本轮失败：${failureLabels[runtime.failure_reason] ?? runtime.failure_reason ?? "安全条件不满足"}。` : "";
  if (failure && engine.gripperLatch !== "CLOSED") return `${failure}禁止继续移动或抓取；${runtime.recording ? "请保存失败或放弃本轮，再重置场景。" : "请点击“重置场景”后重试。"}`;
  const explain = (message) => `${failure ? `${failure}仍可张掌松爪。` : ""}${message}`;
  if (runtime.task_state === "DONE") return runtime.recording ? "任务已成功完成，请保存成功；本轮不再接受开合手势。" : "任务已完成，请重置场景后再练习。";
  if (!stream) return explain(`启动摄像头并绑定后，${gripperInstructions()}`);
  if (!engine.tracker.calibrated) return explain("请先双手张掌稳定 2 秒完成绑定；未绑定时不开合夹爪。");
  if (lastRecognitionIssue) return explain(calibrationDetail(engine.tracker.gestureCalibrationStatus()));
  if (engine.tracker.recoveryPending) return explain("绑定保留，等待双手身份恢复后再确认抓放。");
  if (engine.pendingGripper) return explain("抓放已提交，等待仿真确认真实夹爪状态。");
  if (engine.tracker.palmBindingGuard) return explain("绑定时的张掌不触发松爪；先换操作手势，再正常握拳／张掌开合。");
  if (!engine.gripperLatch) return explain("等待后台同步当前夹爪状态。");
  if (engine.gripperBlocked) {
    const result = runtime.last_command_result;
    const reason = result?.command === "dual_gripper" && result.accepted === false ? result.reason : null;
    const message = {
      release_rejected_stop_before_opening: "物体或夹爪尚未停稳。保持双掌，停稳并重新确认后再释放。",
      episode_ended_reset_required: "本轮已结束，不能再次抓取；请重置场景。",
      stale_command: "抓放指令到达时已过期，请先换手势再重试。",
      task_motion_frozen: "当前任务阶段禁止该操作。",
    }[reason];
    return explain(message ?? (reason ? `抓放未执行：${reason}。先换手势再重试。` : "抓放尚未得到执行确认。先换手势，再重新确认抓放。"));
  }
  const action = engine.gripperLatch === "OPEN" ? "close" : "open";
  const wanted = action === "close" ? "Closed_Fist" : "Open_Palm";
  const label = action === "close" ? "闭合" : "打开";
  const status = ["left", "right"].map((side) => {
    const name = side === "left" ? "左手" : "右手", hand = engine.tracker.hands[side];
    if (!hand.visible) return `${name}未可靠入镜`;
    if (hand.gesture !== wanted) return `${name}需${gestures[wanted]}（当前${gestures[hand.gesture] ?? hand.gesture}）`;
    if (!Number.isFinite(hand.confidence) || hand.confidence < gripperThreshold(action)) return `${name}评分 ${percent(hand.confidence)} < ${percent(gripperThreshold(action))}`;
    return `${name} ${Math.round(engine.evidence[action][side])} / ${engine.config.gripper_confirm_ms} ms`;
  });
  const waitingForStop = engine.gripperRetryReason === "release_rejected_stop_before_opening";
  const pendingStop = waitingForStop ? "等待物体与夹爪停稳，保持双掌。" : "";
  return explain(`${pendingStop}${label}：${status.join("；")}。`);
}
function calibrationDetail(calibration) {
  if (!stream) return "两只手需要同时完整入镜、掌心朝向摄像头。启动后会显示具体等待原因。";
  if (lastRecognitionIssue) {
    if (["scene_reset", "backend_session_changed"].includes(lastRecognitionIssue.code)) return "仿真或场景已切换，绑定保留；正在等待属于当前仿真的新摄像头帧。";
    if (lastRecognitionIssue.code === "unexpected_frame_id") return reasons.unexpected_frame_id;
    if (!Number.isFinite(lastRecognitionIssue.actual) || lastRecognitionIssue.actual < 0) return "识别结果时间戳无效，已暂停；等待新的摄像头画面。";
    return `识别结果延迟 ${Math.round(lastRecognitionIssue.actual)} ms，超过当前允许的 ${Math.round(lastRecognitionIssue.required)} ms；旧结果不用于绑定或控制。`;
  }
  if (performance.now() - lastReceivedResultAt > (engine.tracker.calibrated ? engine.config.hand_lost_grace_ms : engine.config.calibration_sample_gap_ms)) return engine.tracker.calibrated ? "绑定保留，动作暂停。正在等待新的识别结果；夹爪保持当前状态。" : "正在等待新的识别结果；摄像头停顿或识别未完成时不增加稳定计时。";
  if (engine.tracker.calibrated) {
    if (engine.tracker.palmBindingGuard && !engine.tracker.recoveryPending) return "已完成绑定。继续张掌保持暂停和夹爪状态；换成 I Love You、剪刀手或握拳等操作手势后开始控制。";
    if (engine.tracker.recoveryPending) {
      const recovery = calibration.tracking_recovery;
      return `绑定保留。${reasons[output.reason] ?? "动作暂停，等待双手可靠识别。"} 恢复 ${recovery.valid_samples} / ${recovery.required_samples} 次 · ${(recovery.stable_ms / 1000).toFixed(2)} / ${(recovery.required_ms / 1000).toFixed(2)} 秒；不用重新张掌绑定。`;
    }
    return "绑定已保留。放下双手可点击录制；只有“解除绑定”、关闭摄像头或交换左右标签会清除绑定。";
  }
  const blocker = calibration.blocker;
  const side = { left: "左手", right: "右手", both: "双手", unknown: "手部" }[blocker?.side] ?? "手部";
  const target = "张掌";
  const codes = {
    waiting_for_hands: `目前检测到 ${calibration.detection_count ?? 0} 只手，需要两只手同时完整入镜。`,
    hand_lost: `目前检测到 ${calibration.detection_count ?? 0} 只手；请让双手同时完整入镜。`,
    unknown_handedness: "识别到了手，但左右身份未知；请分开双手，掌心朝向摄像头。",
    duplicate_handedness: "两只手被识别为同一侧；请分开双手，掌心朝向摄像头，避免重叠。",
    handedness_confidence_low: `${side}左右身份评分 ${percent(blocker?.actual)}，需要至少 ${percent(blocker?.required)}；请保持整只手入镜。`,
    invalid_wrist: `${side}手腕关键点无效；请让整只手入镜。`,
    identity_conflict: "左右身份与之前位置冲突，已重置；请分开双手重新张掌。",
    gesture_confidence_low: `${side}${target}评分 ${percent(blocker?.actual)}，需要至少 ${percent(blocker?.required)}；短暂抖动不增加计时，请保持手势。`,
    gesture_mismatch: `双手需同时${target}；短暂分类变化不增加计时，请保持手势。`,
    wrist_motion: `${side}手腕位置移动较大，稳定计时已重置；请在舒适位置保持双手。`,
    sample_gap: `识别帧间隔 ${Math.round(blocker?.actual ?? 0)} ms 超过 ${Math.round(blocker?.required ?? 0)} ms，计时已重置。`,
    clock_reversal: "识别时间顺序异常，正在等待新帧并重新计时。",
    insufficient_samples: `双手张掌时间已达标，还需稳定识别样本：${blocker?.actual ?? 0} / ${blocker?.required ?? 5}；请继续保持。`,
  };
  const resetLabels = { hand_lost: "至少一只手离开画面", unknown_handedness: "左右身份未知", duplicate_handedness: "两手被识别为同一侧", handedness_confidence_low: "左右身份评分不足", invalid_wrist: "手腕关键点无效", identity_conflict: "左右身份与位置冲突", gesture_confidence_low: "手势评分不足持续过久", gesture_mismatch: "其他手势持续过久", wrist_motion: "手腕位置移动过大", sample_gap: "识别间隔过长", clock_reversal: "识别时间顺序异常" };
  const previousReset = resetLabels[calibration.last_reset_reason];
  const suffix = previousReset && calibration.last_reset_reason !== blocker?.code ? ` 最近一次重置：${previousReset}。` : "";
  if (blocker) return (codes[blocker.code] ?? "正在等待双手稳定识别；请检查两侧手势和身份评分。") + suffix;
  return `双手识别合格，正在累计有效稳定时间。绑定手势至少 ${percent(engine.config.calibration_gesture_confidence_min)}，左右身份至少 ${percent(engine.config.handedness_confidence_min)}；短暂分类抖动只暂停计时。${suffix}`;
}
function currentControlStatus() {
  const paused = (detail, title = "已暂停") => ({ title, detail, state: "paused" });
  if (!connected) return paused(serviceStopped ? reasons.service_stopped : reasons.backend_disconnected);
  if (transportFrozen) return paused(reasons.control_transport_error);
  if (runtime.saving || pendingOperation) return paused(reasons[runtime.saving ? "saving" : "operation_pending"]);
  const failure = runtime.task_state === "FAIL";
  const failureText = failure ? `本轮失败：${failureLabels[runtime.failure_reason] ?? runtime.failure_reason ?? "安全条件不满足"}。` : "";
  const failureRecovery = `${engine.gripperLatch === "CLOSED" ? "仍可双手张掌松爪。" : ""}${runtime.recording ? "请保存失败或放弃本轮，再重置场景。" : "请重置场景后继续练习。"}`;
  if (failure && engine.gripperLatch !== "CLOSED") return paused(failureText + failureRecovery, "已暂停 · 本轮失败");
  if (runtime.task_state === "DONE") return { title: "本轮已完成", detail: reasons.task_done, state: "complete" };
  if (!stream) return paused(failure ? failureText + failureRecovery : "摄像头未启动，双臂暂停。启动后双手张掌 2 秒绑定。");
  if (!engine.tracker.calibrated) return { title: "等待绑定", detail: calibrationDetail(engine.tracker.gestureCalibrationStatus()), state: "waiting" };
  if (lastRecognitionIssue) return paused(calibrationDetail(engine.tracker.gestureCalibrationStatus()), {
    unexpected_frame_id: "已暂停 · 识别请求不匹配", backend_session_changed: "已暂停 · 仿真同步", scene_reset: "已暂停 · 场景同步",
  }[lastRecognitionIssue.code] ?? "已暂停 · 识别结果过期");
  if (performance.now() - lastReceivedResultAt > engine.config.hand_lost_grace_ms) return paused(reasons.camera_stalled, "已暂停 · 等待新画面");
  if (engine.tracker.recoveryPending) {
    const recovering = output.reason === "tracking_recovery";
    return { title: recovering ? "恢复追踪" : "已暂停 · 丢手或身份不明确", detail: calibrationDetail(engine.tracker.gestureCalibrationStatus()), state: recovering ? "recovering" : "paused" };
  }
  if (engine.tracker.palmBindingGuard) return paused(reasons.binding_complete, "已绑定 · 等待操作手势");
  if (engine.pendingGripper || ["GRASP_CONFIRM", "RELEASE_CONFIRM"].includes(output.mode)) {
    const opening = engine.pendingGripper?.action === "open" || output.mode === "RELEASE_CONFIRM";
    const wanted = opening ? "Open_Palm" : "Closed_Fist";
    if (!engine.pendingGripper && ["left", "right"].some((side) => engine.tracker.hands[side].gesture === wanted && engine.tracker.hands[side].confidence < gripperThreshold(opening ? "open" : "close"))) return paused(gripperDetail(), "已暂停 · 抓放评分不足");
    return { title: `${opening ? "松爪" : "握拳"}确认${failure ? " · 本轮失败" : ""}`, detail: gripperDetail(), state: "confirming" };
  }
  if (failure) return paused(failureText + failureRecovery, "已暂停 · 本轮失败");
  if (engine.gripperBlocked) return paused(gripperDetail(), "已暂停 · 抓放未执行");
  if (["GRASP_VERIFY", "RELEASE_VERIFY"].includes(runtime.task_state)) return paused(`${tasks[runtime.task_state]}中，仿真暂不接受移动，请等待验证结果。`, runtime.task_state === "GRASP_VERIFY" ? "验证抓取" : "验证放置");
  if (output.reason === "motion_low_confidence") {
    const low = ["left", "right"].filter((side) => engine.tracker.hands[side].confidence < engine.config.motion_confidence_min)
      .map((side) => `${side === "left" ? "左手" : "右手"} ${percent(engine.tracker.hands[side].confidence)} < ${percent(engine.config.motion_confidence_min)}`);
    return paused(`移动手势评分不足：${low.join("；")}。`, "已暂停 · 评分不足");
  }
  if (!["XY", "Z"].includes(output.mode)) return paused(reasons[output.reason] ?? "尚未进入移动模式，夹爪保持当前状态。", output.reason === "motion_confirmation" ? "确认移动模式" : "已暂停");
  // A fresh recognized gesture expresses intent. Only a fresh backend reply
  // can establish that the simulation is accepting that movement mode.
  if (performance.now() - lastHealthReceivedAt > 1000) return { title: "等待仿真状态更新", detail: "仿真状态尚未及时更新，当前是否可移动暂未确认。", state: "waiting" };
  if (runtime.paused === true || runtime.control_mode === "PAUSE") return paused(reasons[runtime.pause_reason] ?? `仿真已暂停${runtime.pause_reason ? `：${runtime.pause_reason}` : ""}；等待新控制确认。`, "已暂停 · 仿真保持");
  if (runtime.control_mode !== output.mode) return { title: "等待控制同步", detail: `手势已进入 ${modes[output.mode]}，等待仿真确认该模式。`, state: "waiting" };
  return { title: output.mode === "XY" ? "XY 可平移" : "Z 可升降", detail: `${runtime.task_state === "DUAL_GRASPED" ? "双手输入融合为共同平移。" : "左右手分别控制对应机械臂。"}${engine.config.motion_mapping === "rate" ? "回到中心停止输入。" : "停手停止新的位移输入。"}放下双手会暂停，重新入镜后恢复。`, state: "active" };
}
function renderControlStatus() {
  const status = currentControlStatus();
  $("control-status").dataset.state = status.state;
  $("control-status-title").textContent = status.title;
  $("control-gripper-state").textContent = engine.pendingGripper ? "夹爪等待仿真确认" : engine.gripperLatch === "CLOSED" ? "夹爪保持闭合" : engine.gripperLatch === "OPEN" ? "夹爪保持打开" : "夹爪状态待确认";
  notice(status.detail, ["active", "complete"].includes(status.state));
  $("control-recording-note").hidden = !runtime.recording;
  const captureMessages = {
    not_recording: "未录制。",
    boundary_event: "录制中：保留抓放与边界事件。",
    physical_change: "录制中：实际物理变化保留到训练数据。",
    operator_control: "录制中：保留有效控制帧。",
    stationary_pause: "录制中：静止暂停只留研究日志；出现物理变化时继续保留训练帧。",
    gesture_confirmation: "录制中：抓放确认的静止等待不入训练；实际物理变化仍保留。",
    idle: "录制中：静止等待只留研究日志。",
  };
  $("control-recording-note").textContent = captureMessages[runtime.capture_status?.state] ?? "录制中：静止暂停跳过训练帧；残余物理运动仍保留。";
}
function renderState() {
  const now = performance.now();
  $("mode").textContent = modes[output.mode] ?? output.mode;
  $("task-state").textContent = tasks[runtime.task_state] ?? runtime.task_state ?? "—";
  $("gripper").textContent = engine.gripperLatch === "CLOSED" ? "双侧闭合" : engine.gripperLatch === "OPEN" ? "双侧打开" : "等待同步";
  $("recording-state").textContent = runtime.saving ? "保存中" : runtime.recording ? `进行中 · ${runtime.frames ?? 0} 帧` : "未开始";
  $("session-mode").textContent = runtime.recording ? `正式录制 · 限时 ${runtime.config?.evaluation?.max_episode_seconds ?? 60} 秒` : runtime.practice_mode ? "练习 · 不限时" : "未录制";
  $("task-elapsed").textContent = Number.isFinite(runtime.recording_wall_seconds) && runtime.recording ? `${runtime.recording_wall_seconds.toFixed(1)} s` : runtime.task_timer_started ? `仿真 ${Number(runtime.task_elapsed_seconds ?? 0).toFixed(1)} s` : "未开始";
  $("dataset-counts").textContent = `成功 ${runtime.saved_success ?? 0} / 失败 ${runtime.saved_failure ?? 0}`;
  $("motion-mapping").value = engine.config.motion_mapping;
  $("motion-speed").value = String(engine.config.motion_speed_scale);
  $("motion-mapping").disabled = $("motion-speed").disabled = Boolean(runtime.recording || runtime.saving || pendingOperation);
  $("motion-detail").textContent = engine.config.motion_mapping === "rate"
    ? `回中速度：进入手势时建立中心，偏移约 ${Math.round(engine.config.rate_motion_range * 100)}% 画面并保持可持续移动，回到中心停止；放下双手可暂停重定位。`
    : "随手位移，无需回中。放下双手会暂停，挪回舒适位置重新入镜后恢复。";
  $("gripper-guide").textContent = `${gripperInstructions()}张掌时先停止移动；等待真实状态更新。`;
  for (const side of ["left", "right"]) {
    const observed = lastDetections.find((detection) => detection.side === side);
    $(`${side}-gesture`).textContent = observed ? `${gestures[observed.gesture] ?? observed.gesture} · ${percent(observed.confidence)}` : "未检测到该侧手";
    $(`${side}-identity`).textContent = observed ? `左右身份 ${percent(observed.handedness_score)} / 要求 ${percent(engine.config.handedness_confidence_min)}` : "左右身份待确认";
    $(`${side}-evidence`).value = engine.evidence[output.mode === "RELEASE_CONFIRM" ? "open" : "close"][side] / engine.config.gripper_confirm_ms;
  }
  $("unbind-hands").disabled = !stream || !engine.tracker.calibrated;
  const calibration = engine.tracker.gestureCalibrationStatus(now);
  $("calibration-progress").value = engine.tracker.calibrated ? 1 : calibration.palm_progress;
  $("calibration-progress").hidden = engine.tracker.calibrated;
  if (engine.tracker.calibrated) $("calibration-state").textContent = "左右手已绑定 · 丢手只暂停，绑定保留。";
  else if (!stream) $("calibration-state").textContent = "启动摄像头后：双手张掌稳定 2 秒即绑定，无需确认。";
  else $("calibration-state").textContent = `请分开双手并张掌保持稳定 ${(calibration.palm_progress * engine.config.calibration_ms / 1000).toFixed(2)} / ${(engine.config.calibration_ms / 1000).toFixed(1)} 秒，达到后直接绑定。`;
  $("calibration-detail").textContent = calibrationDetail(calibration);
  $("calibration-detail").hidden = engine.tracker.calibrated && !engine.tracker.recoveryPending && !lastRecognitionIssue;
  $("gripper-detail").textContent = gripperDetail();
  if (!pendingOperation && runtime.last_command_result?.accepted === false && ["record_start", "record_save", "record_failure", "record_discard", "reset", "record_stop"].includes(runtime.last_command_result.command)) $("record-feedback").textContent = `操作未执行：${runtime.last_command_result.reason ?? "安全条件不满足"}`;
  renderControlStatus();
  updateButtons(); renderViews();
}
function drawHands(result) {
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) { canvas.width = video.videoWidth; canvas.height = video.videoHeight; }
  context.clearRect(0, 0, canvas.width, canvas.height);
  const detections = detectionsFromMediaPipe(result, $("swap-labels").checked);
  (result.landmarks ?? []).forEach((landmarks, i) => {
    if (!Number.isFinite(landmarks[0]?.x) || !Number.isFinite(landmarks[0]?.y)) return;
    const side = detections[i]?.side, color = side === "left" ? "#72e0b4" : "#7bb6ff";
    const neutral = engine.tracker.hands[side]?.anchor;
    if (engine.config.motion_mapping === "rate" && ["XY", "Z"].includes(output.mode) && neutral) {
      const x = neutral[0] * canvas.width, y = neutral[1] * canvas.height;
      context.save(); context.strokeStyle = color; context.lineWidth = 2;
      context.beginPath(); context.ellipse(x, y, engine.config.rate_dead_zone * canvas.width, engine.config.rate_dead_zone * canvas.height, 0, 0, Math.PI * 2); context.stroke();
      context.beginPath(); context.moveTo(x - 7, y); context.lineTo(x + 7, y); context.moveTo(x, y - 7); context.lineTo(x, y + 7); context.stroke();
      context.globalAlpha = 0.5; context.beginPath(); context.moveTo(x, y); context.lineTo(landmarks[0].x * canvas.width, landmarks[0].y * canvas.height); context.stroke(); context.restore();
    }
    drawing.drawConnectors(landmarks, GestureRecognizer.HAND_CONNECTIONS, { color, lineWidth: 2 });
    drawing.drawLandmarks(landmarks, { color, radius: 2 });
    context.save(); context.translate(landmarks[0].x * canvas.width, landmarks[0].y * canvas.height - 18); context.scale(-1, 1);
    context.fillStyle = color; context.font = "bold 16px sans-serif"; context.fillText(side === "left" ? "LEFT 左手" : side === "right" ? "RIGHT 右手" : "身份未知", -36, 0); context.restore();
  });
  return detections;
}
function recognitionDiagnostic(data, detections, now, dropReason) {
  const finiteOrNull = (value) => Number.isFinite(value) ? value : null;
  const sourceId = data.frameId ?? `capture-${data.session ?? "unknown"}-${data.control_epoch ?? "unknown"}-${data.capturedAt ?? "unknown"}`;
  return {
    frame_id: `${recognitionClientId}:${sourceId}`,
    source_session: data.session ?? null,
    source_control_epoch: data.control_epoch ?? null,
    captured_at_ms: finiteOrNull(data.capturedEpochMs),
    received_at_ms: Date.now(),
    inference_ms: finiteOrNull(data.inferenceMs),
    result_age_ms: finiteOrNull(now - data.capturedAt),
    accepted: dropReason === null,
    drop_reason: dropReason,
    observed_hands: detections.map((hand) => ({
      label: hand.side ?? "unknown", gesture: hand.gesture ?? "None",
      score: finiteOrNull(hand.confidence), identity_score: finiteOrNull(hand.handedness_score),
    })),
  };
}
function handleRecognition(data) {
  if (!stream) return;
  const now = performance.now();
  const detections = detectionsFromMediaPipe(data.result, $("swap-labels").checked);
  const maxAge = engine.tracker.calibrated
    ? Math.min(RECOGNITION_MAX_AGE_MS, runtime.config?.control?.max_command_age_ms ?? 150)
    : Math.min(500, engine.config.calibration_max_result_age_ms);
  const mismatchedFrame = data.frameId !== workerFrameId;
  const staleContext = data.session !== runtime.session ? "backend_session_changed" : data.control_epoch !== runtime.control_epoch ? "scene_reset" : null;
  const dropReason = mismatchedFrame ? "unexpected_frame_id" : staleContext ?? (!Number.isFinite(data.capturedAt) || !Number.isFinite(data.capturedEpochMs) || now - data.capturedAt < 0 || now - data.capturedAt > maxAge ? "stale_recognition" : null);
  const diagnostic = recognitionDiagnostic(data, detections, now, dropReason);
  // A mismatched reply must not release the slot of the current Worker request.
  // Diagnostics are observations only: rejected hands never enter the tracker.
  if (!mismatchedFrame) {
    workerFrameBusy = false;
    lastReceivedResultAt = now;
    drawHands(data.result);
    lastDetections = detections;
    $("camera-state").textContent = `后台识别 · ${Math.round(data.inferenceMs)} ms`;
  }
  if (dropReason) {
    lastRecognitionIssue = { code: dropReason, actual: diagnostic.result_age_ms, required: maxAge };
    const rejected = engine.pause(dropReason);
    rejected.telemetry.recognition_result = diagnostic;
    // Use the current command envelope for a safe pause; the old camera
    // context remains inside the diagnostic rather than refreshing identity.
    queueOutput(rejected, now);
    if (now - lastUiRenderAt >= 100) { renderState(); lastUiRenderAt = now; }
    return;
  }
  lastRecognitionIssue = null;
  let next;
  if (!connected || transportFrozen || pendingOperation || runtime.saving) {
    engine.tracker.update(detections, data.capturedAt);
    next = engine.pause(!connected ? (serviceStopped ? "service_stopped" : "backend_disconnected") : transportFrozen ? "control_transport_error" : runtime.saving ? "saving" : "operation_pending");
  } else next = engine.update(detections, data.capturedAt);
  next.measurementAt = next.command?.command === "dual_motion" || next.command?.command === "dual_gripper" ? data.capturedEpochMs : Date.now();
  Object.assign(next.telemetry, { capture_time_ms: data.capturedEpochMs, inference_ms: data.inferenceMs, recognition_age_ms: now - data.capturedAt, recognition_result: diagnostic });
  queueOutput(next, now);
  if (now - lastUiRenderAt >= 100 || next.command?.command === "dual_gripper") { renderState(); lastUiRenderAt = now; }
}
async function createGestureWorker() {
  if (gestureWorker && workerReady) return;
  if (typeof Worker === "undefined" || typeof createImageBitmap === "undefined") throw new Error("当前浏览器不支持后台识别，请使用 Edge 或 Chrome。");
  const worker = new Worker("./dual-gesture-worker.js");
  gestureWorker = worker;
  const generation = ++workerGeneration;
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("后台识别模型加载超时")), 30000);
    worker.onmessage = ({ data }) => {
      if (worker !== gestureWorker || generation !== workerGeneration) return;
      if (data.type === "ready") { clearTimeout(timer); workerReady = true; resolve(); }
      else if (data.type === "result") handleRecognition(data);
      else if (data.type === "error") {
        clearTimeout(timer); workerFrameBusy = false;
        if (!workerReady) reject(new Error(data.message));
        else { stopCamera("camera_error"); notice(`后台识别已暂停：${data.message}`); }
      }
    };
    worker.onerror = (event) => {
      clearTimeout(timer); workerFrameBusy = false;
      if (!workerReady) reject(new Error(event.message || "识别 Worker 加载失败"));
      else { stopCamera("camera_error"); notice(`后台识别已停止：${event.message || "Worker 异常"}`); }
    };
    worker.postMessage({ type: "init", modelUrl: new URL("./models/gesture_recognizer.task", location.href).href, wasmRoot: new URL("./node_modules/@mediapipe/tasks-vision/wasm", location.href).href, delegate: "CPU" });
  });
  drawing = new DrawingUtils(context);
}
async function submitCameraFrame(now) {
  const worker = gestureWorker, generation = workerGeneration;
  workerFrameBusy = true;
  lastFrameSubmittedAt = now;
  lastVideoTime = video.currentTime;
  const capturedEpochMs = Date.now(), epoch = runtime.control_epoch, session = runtime.session;
  try {
    const bitmap = await createImageBitmap(video);
    if (!stream || worker !== gestureWorker || generation !== workerGeneration) { bitmap.close(); return; }
    worker.postMessage({ type: "frame", frameId: ++workerFrameId, capturedAt: now, capturedEpochMs, control_epoch: epoch, session, bitmap }, [bitmap]);
  } catch (error) { workerFrameBusy = false; stopCamera("camera_error"); notice(`摄像头画面读取失败：${error.message}`); }
}
function detectFrame() {
  if (!stream) return;
  const now = performance.now();
  if (workerReady && !workerFrameBusy && video.readyState >= 2 && video.currentTime !== lastVideoTime && now - lastFrameSubmittedAt >= engine.config.control_interval_ms) void submitCameraFrame(now);
  const stallLimit = engine.tracker.calibrated ? engine.config.hand_lost_grace_ms : engine.config.calibration_sample_gap_ms;
  if (now - lastReceivedResultAt > stallLimit && Number.isFinite(lastReceivedResultAt)) {
    queueOutput(engine.pause("camera_stalled"), now);
    if (now - lastUiRenderAt >= 100) { renderState(); lastUiRenderAt = now; }
  }
  frameId = requestAnimationFrame(detectFrame);
}
async function startCamera() {
  $("camera-button").disabled = true; $("camera-state").textContent = "加载模型 / 等待授权";
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("请通过 localhost 使用 Edge 或 Chrome。");
    await createGestureWorker();
    stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" } });
    video.srcObject = stream; await video.play(); output = engine.unbind("calibration_required"); lastVideoTime = -1; lastReceivedResultAt = performance.now(); lastFrameSubmittedAt = -Infinity;
    lastDetections = []; lastRecognitionIssue = null;
    $("camera-placeholder").hidden = true; $("camera-button").textContent = "关闭摄像头"; $("camera-state").textContent = "双手识别中"; detectFrame();
  } catch (error) {
    if (stream) stopCamera("camera_error");
    else { gestureWorker?.terminate(); gestureWorker = null; workerReady = workerFrameBusy = false; workerGeneration += 1; }
    const detail = { NotAllowedError: "权限被拒绝，请在地址栏允许摄像头。", NotFoundError: "未发现摄像头，请检查台式机摄像头连接。", NotReadableError: "摄像头被占用，请关闭会议或拍照软件。" }[error.name];
    notice(detail ?? `启动失败：${error.message}`); $("camera-state").textContent = "启动失败";
  } finally { $("camera-button").disabled = false; }
}
function stopCamera(reason = "camera_stopped") {
  if (frameId) cancelAnimationFrame(frameId);
  stream?.getTracks().forEach((track) => track.stop()); stream = null; video.srcObject = null; context.clearRect(0, 0, canvas.width, canvas.height);
  gestureWorker?.terminate(); gestureWorker = null; workerReady = workerFrameBusy = false; workerGeneration += 1;
  lastDetections = []; lastRecognitionIssue = null; lastReceivedResultAt = -Infinity;
  clearHandCommands(); queueOutput(engine.unbind(reason), performance.now(), true); renderState(); $("camera-placeholder").hidden = false; $("camera-button").textContent = "启动摄像头"; $("camera-state").textContent = "未启动";
}
$("camera-button").addEventListener("click", () => { if (stream) stopCamera(); else void startCamera(); });
$("pause-button").addEventListener("click", () => pause("user_pause"));
$("view-select").addEventListener("change", () => { viewPreference = ["auto", "top", "side", "front"].includes($("view-select").value) ? $("view-select").value : "auto"; renderViews(); });
function updateMotionSettings() {
  if (runtime.recording || runtime.saving || pendingOperation) { renderState(); return; }
  const mapping = $("motion-mapping").value;
  const speed = Number($("motion-speed").value);
  if (!["rate", "incremental"].includes(mapping) || ![0.3, 0.7, 1].includes(speed)) { renderState(); return; }
  clearHandCommands();
  queueOutput(engine.reconfigure({ ...engine.config, motion_mapping: mapping, motion_speed_scale: speed }), performance.now(), true);
  renderState();
}
$("motion-mapping").addEventListener("change", updateMotionSettings);
$("motion-speed").addEventListener("change", updateMotionSettings);
function clearHandCommands() {
  latestPacket = null;
  events = events.filter((packet) => packet.command !== "dual_gripper");
  clearTimeout(controlPumpTimer);
}
function unbindHands(reason = "binding_released") {
  clearHandCommands();
  queueOutput(engine.unbind(reason), performance.now(), true);
  renderState();
}
$("unbind-hands").addEventListener("click", () => unbindHands());
$("swap-labels").addEventListener("change", () => unbindHands("mapping_changed"));
for (const button of document.querySelectorAll("[data-command]")) {
  button.addEventListener("click", () => {
    if (!connected || pendingOperation) return;
    const command = button.dataset.command;
    pause("user_pause");
    const eventId = `operation-${Date.now()}-${crypto.randomUUID()}`;
    pendingOperation = { command, eventId, startedAt: performance.now() };
    events.push({ command, eventId, sentAt: Date.now(), control_epoch: runtime.control_epoch, session: runtime.session, telemetry: engine.telemetry(false) });
    $("record-feedback").textContent = "已提交操作，等待仿真执行确认。";
    engine.clearAnchors();
    updateButtons(); void flushPackets();
  });
}
window.addEventListener("keydown", (event) => { if (["Space", "Escape"].includes(event.code) && !["INPUT", "TEXTAREA", "BUTTON"].includes(event.target.tagName)) { event.preventDefault(); pause("user_pause"); } });
document.addEventListener("visibilitychange", () => { if (document.hidden) pause("tab_hidden"); });
window.addEventListener("pagehide", () => {
  clearTimeout(healthTimer); clearTimeout(previewTimer); clearTimeout(controlPumpTimer); latestPacket = null; events = [];
  if (connected) void fetch(`${api}/control`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ command: "pause", reason: "page_closed", sentAt: Date.now(), control_epoch: runtime.control_epoch, session: runtime.session, telemetry: engine.telemetry(false) }), keepalive: true }).catch(() => {});
  if (frameId) cancelAnimationFrame(frameId); stream?.getTracks().forEach((track) => track.stop()); gestureWorker?.terminate(); Object.values(previews).filter(Boolean).forEach((url) => URL.revokeObjectURL(url));
});
setConnected(false); void checkHealth(); void pollPreviews();
