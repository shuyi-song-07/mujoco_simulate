import { DrawingUtils, GestureRecognizer } from "./node_modules/@mediapipe/tasks-vision/vision_bundle.mjs";
import { DEFAULT_DUAL_CONFIG, DualInteractionState, detectionsFromMediaPipe, resolveOperation } from "./dual-interaction-state.js";
import { MeasurementJournal, controlObservation } from "./measurement-journal.js";
import { handMeasurement } from "./hand-measurement.js";
import { ContinuousInteractionState } from "./continuous-interaction-state.js";

const $ = (id) => document.getElementById(id);
const commandClientId = `browser-${Date.now()}-${Math.random().toString(36).slice(2)}`;
let commandSequence = 0;
function transportPacket(packet) {
  if (runtime.command_protocol !== "task3-command-v2") return packet;
  const sequence = ++commandSequence;
  return { ...packet, protocol: "task3-command-v2", client_id: commandClientId, sequence,
    eventId: packet.eventId ?? `${commandClientId}-${sequence}` };
}
const api = "/api/dual", video = $("webcam"), canvas = $("output-canvas"), context = canvas.getContext("2d");
const gestures = { ILoveYou: "I Love You", Victory: "剪刀手", Closed_Fist: "握拳", Open_Palm: "张掌", Pointing_Up: "食指向上", Thumb_Up: "竖拇指", None: "未识别" };
const modes = { XY: "XY 平移", Z: "Z 升降", PAUSE: "暂停", IDLE: "等待", GRASP_CONFIRM: "闭合确认", RELEASE_CONFIRM: "打开确认" };
const tasks = { PREGRASP: "对准待抓", GRASP_VERIFY: "验证双侧抓取", DUAL_GRASPED: "协同搬运", PLACEMENT: "放置微调", RELEASE_VERIFY: "验证放置", DONE: "成功完成", FAIL: "失败" };
const failureLabels = { timeout: "超过正式任务时限", grasp_failure: "双侧抓取未建立", contact_loss: "搬运中失去接触", object_drop: "物体跌落", relative_pose_violation: "两臂相对位置超限", robot_robot_collision: "两臂发生碰撞", workspace_violation: "工作空间越界", joint_or_ik_failure: "关节或逆解异常" };
const viewLabels = { top: "正上方 · XY 对准（↑ −X，→ +Y）", oblique: "斜视 · 模型输入 oblique", front: "内侧近景 · 对准与高度" };
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
  hand_near_frame_edge: "手部进入画面边缘保护区，动作和夹爪保持。请把完整双手移回内框，再匹配开度恢复；无需重新标定。",
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
  backend_disconnected: "Task 3 仿真未连接，请运行 START-PAPERA.cmd。",
  saving: "正在写入数据，请等待保存完成。", task_done: "任务已自动判定成功，可以保存成功示范。",
  camera_stopped: "摄像头已关闭，绑定已解除。再次启动后双手张掌稳定 2 秒重新绑定。",
  camera_error: "摄像头或识别异常，两臂暂停。", camera_stalled: "摄像头画面停止更新，两臂暂停；请检查摄像头。", user_pause: "已暂停。把双手放回舒适位置，再使用同一移动手势。",
  tab_hidden: "页面离开前台，机器人暂停。回来后重新建立移动基准。",
  operation_pending: "等待仿真完成操作；双臂暂停，暂不接受新的移动或抓放手势。",
  scene_reset: "场景已更新，旧移动基准已清除；保持双手静止后重新进入 XY / Z 模式。",
  backend_session_changed: "仿真已重新启动；绑定保留，旧指令已清除。等待双手稳定后重新建立移动基准。",
  service_stopped: "退出请求已提交，仿真连接已关闭。需要继续采集时重新运行 START-PAPERA.cmd。",
  control_transport_error: "控制通道暂时繁忙或超时，两臂暂停；等待健康检查恢复后重新建立移动基准。",
  transport_recovery: "控制通道已恢复；旧动作已丢弃，请保持双手静止后重新进入移动模式。",
  stale_recognition: "识别结果延迟较大，已丢弃旧手势并暂停；等待新的摄像头画面。",
  unexpected_frame_id: "识别结果不属于当前摄像头请求，已丢弃并暂停；等待当前画面。",
  command_watchdog: "仿真未及时收到有效控制，已暂停；等待新识别结果后恢复。",
  task_motion_frozen: "仿真当前阶段暂不接受移动。",
  recording_requires_clean_control_cycle: "本轮练习已有释放授权记录。请先重置场景，再开始录制，确保新示范包含完整的授权来源。",
  record_start_baseline: "录制已开始，位置与爪宽保持。先稳定张掌建立本轮开合基准，再换握拳闭合；练习阶段的开合授权不会带入本轮。",
};
let engine = new DualInteractionState(), output = engine.pause("camera_stopped");
let gestureWorker, workerReady = false, workerFrameBusy = false, workerFrameId = 0, workerGeneration = 0;
let drawing, stream, frameId, healthTimer, previewTimer;
let runtime = {}, connected = false, lastVideoTime = -1, lastPacketAt = -Infinity;
let lastHealthReceivedAt = -Infinity;
let latestPacket = null, sending = false, events = [], configSignature = "", mainView = "top";
let viewPreference = "auto";
let pendingOperation = null, controlEpoch = null, controlSession = null, serviceStopped = false, stoppedSession = null;
let lastOperationKey = null;
const previews = { top: null, oblique: null, front: null };
const previewsUpdatedAt = { top: -Infinity, oblique: -Infinity, front: -Infinity };
let lastFrameSubmittedAt = -Infinity, lastUiRenderAt = -Infinity, transportFrozen = false, transportFrozenAt = -Infinity;
let lastDetections = [], lastRecognitionIssue = null, lastReceivedResultAt = -Infinity;
let controlPumpTimer, lastContinuousPostAt = -Infinity;
const PREVIEW_GRACE_MS = 1000;
const RECOGNITION_MAX_AGE_MS = 250;
const recognitionClientId = crypto.randomUUID();
let cameraResearchSettings = null;
let sealingHandBoundary = false;
let retryHandBoundary = null;
function researchContext() {
  return {camera:cameraResearchSettings,display_mirrored:true,inference_mirrored:false,
    swap_labels:$("swap-labels").checked,frame_crop:"none_in_application",
    filter_and_mapping:engine.config,gesture_parameters:engine.gestureParameters??null,
    control_protocol:engine.gestureProtocol?.()??null,config_hash:runtime.config_hash??null,
    model:"models/gesture_recognizer.task",delegate:"CPU"};
}
const measurementJournal = new MeasurementJournal({clientId:recognitionClientId,
  send:async batch=>{
    const response=await fetch(`${api}/measurements`,{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify(batch),signal:AbortSignal.timeout(5000)});
    const result=await response.json();if(!response.ok)throw Error(result.error??`research HTTP ${response.status}`);return result;
  },changed:status=>{
    const el=$("measurement-log-state");if(el)el.textContent=`人手记录：完成 ${status.completed} / 落盘 ${status.written} / 待写记录 ${status.pending} / 丢弃记录 ${status.dropped}（测量 ${status.dropped_measurements}）${status.error?` · 异常 ${status.error}`:""}`;
  }});
function flushMeasurements() {void measurementJournal.flush().catch(()=>{});}
setInterval(flushMeasurements,200);
function journalMeasurement(diagnostic,next,before) {
  measurementJournal.add("measurement",{measurement:diagnostic.measurement,
    control:controlObservation(engine,next,diagnostic.measurement,before),clock_pair:runtime.clock_pair??null},researchContext());
}


function notice(message, active = false) { $("notice").textContent = message; $("notice").dataset.state = active ? "active" : "warning"; }
function rateGestures() {return engine.gestureProtocol?.()==="gesture-rate-gripper-v3";}
function edgeTriggeredGrippers() {
  return rateGestures()&&["palm_fist_edge_v1","palm_fist_edge_v2"].includes(runtime.physical_contact?.gripper_trigger_policy??runtime.config?.gesture_control?.gripper_trigger_policy);
}
function resumableGrippers() {
  return (runtime.physical_contact?.gripper_trigger_policy??runtime.config?.gesture_control?.gripper_trigger_policy)==="palm_fist_edge_v2";
}
function levelTriggeredGrippers() {
  return rateGestures()&&(runtime.physical_contact?.gripper_trigger_policy??runtime.config?.gesture_control?.gripper_trigger_policy)==="palm_fist_level_v2";
}
function rateModeConfirmation() {
  const duration=engine.gestureParameters?.mode_confirm_ms??200;
  return duration>0?`新手势稳定 ${duration} ms` : "识别后直接进入手势模式";
}
function lastSavedOperation() {
  const history = Array.isArray(runtime.operation_results) ? runtime.operation_results : [];
  return [...history, runtime.last_operation_result].reverse().find(result => result?.accepted === true
    && ["record_save", "record_failure"].includes(result.command)
    && result.completed_control_epoch === runtime.control_epoch);
}
function updateButtons() {
  const recording = runtime.recording === true, saving = runtime.saving === true;
  // Binding survives moving a hand to the mouse. Free, empty jaws need not
  // remain fully open while a solo operator clicks; record their real width.
  const freeLevelStart = levelTriggeredGrippers() && runtime.physical_contact?.state === "FREE"
    && runtime.physical_contact?.ever_dual === false
    && ["left","right"].every(s => runtime.contacts?.[`${s}_fingers`]?.length === 2
      && runtime.contacts[`${s}_fingers`].every(v => v === false));
  const startWidthsReady = freeLevelStart || (!levelTriggeredGrippers()
    && ["left","right"].every(s => runtime.grippers?.[s]?.command_width_m >= .079));
  const successFinishing = runtime.recording && runtime.task_state === "DONE" && !runtime.episode_end && runtime.success_tail?.seconds > 0;
  for (const button of document.querySelectorAll("[data-command]")) {
    const command = button.dataset.command;
    const cannotStart = engine.continuous ? (recording || !engine.tracker.calibrated || (!rateGestures()&&engine.apertureCalibration.phase !== "READY") || runtime.task_state !== "PREGRASP" || !startWidthsReady || runtime.config?.recording?.schema_version !== "task3-papera-v2") : recording || !engine.tracker.calibrated || (!["PREGRASP", "DONE", "FAIL"].includes(runtime.task_state)) || (runtime.task_state === "PREGRASP" && engine.gripperLatch !== "OPEN");
    button.disabled = !connected || transportFrozen || saving || Boolean(pendingOperation) || (successFinishing && ["record_save", "record_failure"].includes(command)) || (command === "record_start" && cannotStart) || (["record_save", "record_failure", "record_discard"].includes(command) && !recording) || (command === "record_save" && runtime.task_state !== "DONE") || (["reset", "record_stop"].includes(command) && recording);
    button.title = !button.disabled ? "" : !connected ? "仿真未连接，等待重新连接。"
      : transportFrozen ? "控制通道正在恢复，等待连接确认。"
      : saving ? "正在保存和生成视频，完成后按钮恢复。"
      : pendingOperation ? "上一项操作正在等待仿真确认。"
      : successFinishing && ["record_save", "record_failure"].includes(command) ? "正在记录成功后的开爪收尾，结束后即可保存。"
      : ["record_save", "record_failure", "record_discard"].includes(command) && !recording ? "本轮未录制，无法保存或放弃录制；请重置场景，再点击开始录制。"
      : command === "record_save" && runtime.task_state !== "DONE" ? "任务尚未判定成功；录制中仍可保存失败或放弃本轮。"
      : ["reset", "record_stop"].includes(command) && recording ? "请先保存或放弃正在录制的本轮。"
      : levelTriggeredGrippers() ? "开始录制需要先绑定双手，处于未抓取、未接触物体的待抓阶段；绑定后可以移手点击鼠标。已抓取过请先重置。"
      : "开始录制需要完成双手绑定、处于对准待抓阶段且两爪全开。";
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
    if (engine.continuous) renderContinuousRobotReadings();
  } else if (serviceStopped && runtime.session !== stoppedSession) serviceStopped = false;
  updateButtons(); renderControlStatus();
}
async function postPacket(packet) {
  packet = transportPacket(packet);
  measurementJournal.add("dispatch",{event_id:packet.eventId??null,session:packet.session,
    control_epoch:packet.control_epoch,measurement_id:packet.measurement_id??packet.telemetry?.recognition_result?.measurement?.measurement_id??null,
    command:packet.command,delta_task_m:packet.delta_task_m??null,reason:packet.reason??null},researchContext());
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
      if (["dual_motion", "continuous_input"].includes(packet.command) && Date.now() - packet.sentAt > (runtime.config?.control?.max_command_age_ms ?? 150)) continue;
      if (continuous) lastContinuousPostAt = performance.now();
      // A stop is an independent safety event, not a reused hand action.
      // Preserve its request time but stamp actual dispatch so a slow earlier
      // POST cannot make cancellation expire. Measurement capture stays intact.
      if (packet.command === "pause") packet.sentAt = Date.now();
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
  if (next.command?.command === "pause") {
    // Latest telemetry may replace latest motion, but cannot erase a stop.
    // Keep one unsent priority pause; discard unsent hand actuation before it.
    latestPacket = null;
    events = events.filter(packet => !["pause", "dual_motion", "continuous_input", "dual_gripper"].includes(packet.command));
    events.unshift({ ...next.command, requestedAt: Date.now(), sentAt: Date.now(), control_epoch: runtime.control_epoch, session: runtime.session });
    lastPacketAt = timestamp;
    void flushPackets();
  } else if (next.command?.command === "dual_gripper") {
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
function pauseUntrustedTracking(reason) {
  // A rejected result or short camera stall may never enter tracker.update().
  // Still require new evidence; clearing motion anchors alone is insufficient.
  if (engine.continuous) engine.tracker.requireRecovery(reason);
  return engine.pause(reason);
}
function resolvePendingOperation(isConnected = true) {
  const result = resolveOperation(pendingOperation, runtime, performance.now(), isConnected);
  pendingOperation = result.pending;
  if (!result.outcome) return;
  if (result.outcome === "accepted") {
    $("record-feedback").textContent = { record_start: "录制已开始。", record_save: "仿真已确认成功保存。", record_failure: "失败示范已独立保存。", record_discard: "本轮未进入训练集，研究日志与快照保留为 rejected。", reset: engine.continuous?"已切换下一摆放，请重新建立移动基准。":"场景已重置，请重新建立移动基准。", record_stop: "采集程序已确认正常结束。" }[result.command];
    if (["reset", "record_save", "record_failure", "record_discard"].includes(result.command)) engine.resetScene();
    if (result.command === "record_stop") { serviceStopped = true; stoppedSession = runtime.session; }
    if (["record_save", "record_failure"].includes(result.command) && Number.isInteger(result.operation?.frames))
      $("record-feedback").textContent = `${result.command === "record_save" ? "成功" : "失败"}示范已保存（${result.operation.frames} 帧）。当前新场景尚未录制，请再次点击“开始录制”后操作。`;
    if (result.operation?.research_warning) $("record-feedback").textContent += ` 人手研究记录需复查：${result.operation.research_warning}`;
  } else if (result.outcome === "rejected") $("record-feedback").textContent = `操作未执行：${reasons[result.reason] ?? result.reason ?? "条件不满足"}`;
  else if (result.outcome === "connection_closed") { serviceStopped = true; stoppedSession = runtime.session; $("record-feedback").textContent = "退出请求已提交，仿真连接已关闭。"; }
  else $("record-feedback").textContent = "等待仿真确认超时，请检查任务状态和日志，不要重复保存同一轮。";
}
async function checkHealth() {
  const requestStartedAt = performance.now();
  const requestEpochMs = Date.now();
  try {
    const response = await fetch(`${api}/health`, { cache: "no-store", signal: AbortSignal.timeout(2500) });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    runtime = await response.json();
    runtime.clock_pair = {browser_request_epoch_ms:requestEpochMs,browser_receive_epoch_ms:Date.now(),
      server_epoch_ms:runtime.server_epoch_ms,server_monotonic_ns:runtime.server_monotonic_ns,
      round_trip_ms:performance.now()-requestStartedAt};
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
    if (runtime.config?.contact_control?.enabled && !engine.continuous) { engine = new ContinuousInteractionState(runtime.config.control); viewPreference = "top"; $("view-select").value = "top"; }
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
    if (["DONE", "FAIL"].includes(runtime.task_state)) {
      latestPacket = null;
      events = events.filter(packet => !["pause", "dual_motion", "continuous_input", "dual_gripper"].includes(packet.command));
      clearTimeout(controlPumpTimer);
      output = engine.pause(runtime.task_state === "DONE" ? "task_done" : runtime.failure_reason ?? "task_failed");
    }
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
  const auxiliary = ["top", "oblique", "front"].filter((view) => view !== mainView);
  assignImage("auxiliary-preview-one", "auxiliary-placeholder-one", visible(auxiliary[0]));
  assignImage("auxiliary-preview-two", "auxiliary-placeholder-two", visible(auxiliary[1]));
  $("auxiliary-view-one").textContent = viewLabels[auxiliary[0]];
  $("auxiliary-view-two").textContent = viewLabels[auxiliary[1]];
}
async function decodePreview(url) {
  const image = new Image();
  image.src = url;
  await image.decode();
}
async function pollPreviews() {
  const pollStartedAt = performance.now();
  if (connected) {
    const requestedEpoch = runtime.control_epoch, requestedSession = runtime.session;
    const status = await Promise.all(["top", "oblique", "front"].map(async (view) => {
      try {
        const requestStartedAt = performance.now();
        const response = await fetch(`${api}/${view}-preview`, { cache: "no-store", signal: AbortSignal.timeout(2000) });
        if (!response.ok || response.status === 204) throw new Error("preview unavailable");
        const sourceAge = Number(response.headers.get("X-Preview-Age-Ms") ?? 0);
        const sourceEpoch = response.headers.get("X-Preview-Epoch");
        if (!Number.isFinite(sourceAge) || sourceAge < 0 || sourceAge > PREVIEW_GRACE_MS
            || (sourceEpoch !== null && Number(sourceEpoch) !== requestedEpoch)) throw new Error("stale preview snapshot");
        const next = URL.createObjectURL(await response.blob()), previous = previews[view];
        try {
          await decodePreview(next);
          if (runtime.control_epoch !== requestedEpoch || runtime.session !== requestedSession) throw new Error("scene changed during preview request");
        } catch (error) { URL.revokeObjectURL(next); throw error; }
        previews[view] = next;
        previewsUpdatedAt[view] = requestStartedAt - sourceAge;
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
  renderViews(); renderControlStatus(); previewTimer = setTimeout(pollPreviews, Math.max(20, 100 - (performance.now() - pollStartedAt)));
}
const percent = (value) => `${Math.floor((Number.isFinite(value) ? value : 0) * 1000 + 1e-8) / 10}%`;
const gripperThreshold = (action) => engine.config[`gripper_${action}_confidence_min`] ?? engine.config.gripper_confidence_min;
const gripperInstructions = () => `握拳闭合各至少 ${percent(gripperThreshold("close"))}，张掌打开各至少 ${percent(gripperThreshold("open"))}；双手保持约 ${(engine.config.gripper_confirm_ms / 1000).toFixed(1)} 秒。`;
const rateGestureThreshold = (gesture, parameters, side) => ({
  ILoveYou: parameters.motion_score_min??.75,
  Open_Palm: parameters.open_score_min??.5,
  Closed_Fist: side === "left" ? parameters.left_close_score_min??parameters.close_score_min??.8 : parameters.close_score_min??.8,
})[gesture];
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
  if (engine.continuous) return continuousControlStatus();
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
  if(engine.continuous) {
    const angle=Number.isFinite(runtime.target_orientation_error_rad)?` · 朝向差 ${(runtime.target_orientation_error_rad*180/Math.PI).toFixed(1)}°`:"";
    const support=levelTriggeredGrippers()?"张掌匀速开 / 握拳匀速合 · 左右独立":edgeTriggeredGrippers()?"拳→掌确认释放 · 无需落桌或停手":runtime.physical_contact?.support_ready?(rateGestures()?"桌面支撑已稳定 · 仍需张掌与停稳确认":"桌面支撑已稳定"):
      rateGestures()?rateSupportDetail():"释放需桌面支撑";
    $("control-gripper-state").textContent=connected?support+angle:"仿真未连接 · 支撑状态不可用";
    notice(status.detail,["active","complete"].includes(status.state));
    $("control-recording-note").hidden=!connected;
    $("control-recording-note").textContent=runtime.saving?"正在按录制配置从物理快照生成视频，请等待完成。":runtime.recording
      ?runtime.episode_end?"成功放置已确认：本条记录已截止并暂停，等待保存成功或失败；保存前会收齐截止时刻之前的人手日志。":"保留完整 30 Hz 原始物理时间轴，人手测量独立记录；训练过滤另行派生。"
      :["DONE","FAIL"].includes(runtime.task_state)
        ?"练习已结束 · 本轮未录制，无法保存为示范。请点击“重置场景”，再点击“开始录制”后操作。"
        :"当前为练习，本轮未录制。请在操作前点击“开始录制”；每次保存或放弃后，新场景都需重新开始录制。";
    const saved = lastSavedOperation();
    if (!runtime.recording && !runtime.saving && saved)
      $("control-recording-note").textContent = `上一轮${saved.command === "record_save" ? "成功" : "失败"}示范已保存（${saved.frames} 帧）。当前是新的未录制场景。${$("control-recording-note").textContent}`;
    return;
  }
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

const gestureWaitTitles = {
  user_pause:"手动暂停", tracking_recovery:"追踪恢复", tracking_recovered:"追踪已恢复 · 建立基准",
  hand_lost:"丢手暂停", ambiguous_handedness:"左右身份待确认", identity_conflict:"左右身份待确认",
  stale_recognition:"识别结果过期", tracking_stalled:"等待新识别画面", camera_stalled:"等待新摄像头画面",
  gesture_wrist_not_still:"停稳确认", release_waiting_for_stop:"释放等待 · 请停稳", gesture_mode_confirmation:"手形模式确认",
  gesture_geometry_hold:"手形待确认", aperture_takeover:"匹配开度", operation_lock_hold:"握拳操作锁 · 爪宽保持",
  mixed_motion_planes:"双手移动平面不一致", pinch_baseline:"开度已接管 · 首帧保持", gesture_motion_baseline:"移动基准已建立",
  release_cycle_hold:"释放完成 · 等待退出周期", release_cycle_exit:"释放周期已退出", opening_cancelled:"自动开爪已中止",
  release_requires_stable_table_support:"释放等待 · 桌面支撑", release_requires_stationary_targets:"释放等待 · 请停稳",
  release_requires_both_sides:"释放等待 · 另一侧确认", command_watchdog:"控制输入中断", control_transport_error:"控制通道恢复中",
  rate_gesture_unqualified:"手势未达门槛 · 保持", binding_complete:"身份已绑定 · 张掌保护", binding_gesture_released:"绑定保护已解除",
  record_start_baseline:"录制已开始 · 等待开合基准",
};
const gestureReleaseLabels = {
  release_requires_both_sides:"等待另一侧完整张掌确认", release_requires_stable_table_support:"等待桌面稳定支撑",
  release_requires_stationary_targets:"等待物体与夹爪目标停稳", release_needs_relaxed_intent:"先离开张掌再重新完整张掌确认",
  release_input_interrupted:"输入中断；停稳后重新完整张掌确认", release_mode_changed:"模式已切换；需重新完整张掌确认",
};
function continuousBackendRejection() {
  // The latest execution ACK replaces earlier rejects. Never reuse an old
  // pause_reason when a newer input was accepted, or present an old health
  // snapshot as evidence of the current control state.
  if(performance.now()-lastHealthReceivedAt>1000)return null;
  const result=runtime.last_command_result;
  return result?.command==="continuous_input"&&result.accepted===false?result.reason??"command_rejected":null;
}
function rateSupportDetail() {
  const c=runtime.physical_contact??{},r=c.support_requirements??{},m=c.support_measurements??{};
  const surface=c.support_surface==="placement_platform"?"高台":"桌面";
  const value=(v,scale=1,unit="")=>Number.isFinite(v)?`${(v*scale).toFixed(1)}${unit}`:"—";
  const labels={
    insufficient_table_support:`${surface}承载不足（${value(c.support_force_n??c.table_support_force_n,1," N")} / 至少 ${value(r.minimum_table_force_n,1," N")}）`,
    object_above_table:`物体还高于${surface}（底部 ${value(c.bottom_height_m,1000," mm")} / 允许 ≤ ${value(r.maximum_bottom_height_m,1000," mm")}）`,
    excessive_table_penetration:`物体压入${surface}过深（${value(-c.bottom_height_m,1000," mm")} / 允许 ≤ ${value(-r.minimum_bottom_height_m,1000," mm")}），请稍抬起`,
    object_linear_motion:`物体仍在平移（${value(m.object_linear_speed_mps,1000," mm/s")} / 上限 ${value(r.max_object_linear_speed_mps,1000," mm/s")}）`,
    object_angular_motion:`物体仍在转动（${value(m.object_angular_speed_radps,180/Math.PI,"°/s")} / 上限 ${value(r.max_object_angular_speed_radps,180/Math.PI,"°/s")}）`,
    end_effector_motion:`夹爪仍在运动（${value(m.max_ee_speed_mps,1000," mm/s")} / 上限 ${value(r.max_ee_speed_mps,1000," mm/s")}）`,
    support_stabilizing:`支撑稳定计时 ${(m.stable_seconds??0).toFixed(2)} / ${(r.stable_seconds??.12).toFixed(2)} s（物理时间）`,
  };
  const blockers=Array.isArray(c.support_blockers)?c.support_blockers:[];
  return blockers.length?blockers.map(code=>labels[code]??`等待支撑检查：${code}`).join("；"):`等待${surface}稳定支撑`;
}
function rateReleaseConfirmation(side) {
  const c=runtime.physical_contact??{};
  const clock=c.release_clock==="server_monotonic"?"实际时间":c.release_clock==="simulation"?"仿真时间":"后端计时";
  if(edgeTriggeredGrippers())return `张掌确认 ${(c.release_progress_s?.[side]??0).toFixed(2)} / ${(c.release_confirm_seconds??.7).toFixed(2)} s（${clock}） · 无需落桌或停手，空中也可开爪`;
  return `释放确认 ${(c.release_progress_s?.[side]??0).toFixed(2)} / ${(c.release_confirm_seconds??.7).toFixed(2)} s（${clock}） · 保持张掌并停手，确认期间保持位置`;
}
function rateHandReading(side) {
  const measurement=engine.measurement,hand=measurement?.hands?.find(h=>h.bound_side===side);
  const age=performance.now()-measurement?.capture_browser_monotonic_ms;
  const limit=engine.tracker.calibrated?(runtime.config?.control?.max_command_age_ms??150):Math.min(500,engine.config.calibration_max_result_age_ms);
  // Rejects never enter the state engine. Its retained measurement is history,
  // so do not present that old score as a newly trusted camera observation.
  const fresh=Boolean(measurement&&!lastRecognitionIssue&&measurement.accepted_recognition!==false&&age>=0&&age<=limit);
  return {hand:fresh?hand:null,stale:Boolean(measurement&&!fresh)};
}
function ratePartialGraspSide() {
  const c=runtime.physical_contact??{};
  const held=["left","right"].filter(side=>Number.isFinite(c.hold_width_m?.[side])&&c.stable_grasps?.[side]===true&&runtime.contacts?.[side]===true);
  return held.length===1?held[0]:null;
}
function ratePartialGraspDetail() {
  const side=ratePartialGraspSide();
  if(!side)return null;
  if(edgeTriggeredGrippers()) {
    const holding=side==="left"?"左":"右",free=side==="left"?"右":"左";
    return `${holding}臂已抓稳并保持位置；${free}臂可继续对准和掌↔拳开合，已抓侧手腕微动不会阻止另一爪开合；双侧抓稳后再共同抬升。`;
  }
  return side==="left"?"左臂已抓稳并保持位置，右臂可继续对准；双侧抓稳后再共同抬升。":"右臂已抓稳并保持位置，左臂可继续对准；双侧抓稳后再共同抬升。";
}
function rateEdgeStatus(side) {
  // These are executor states, not inferred opening/closing from the current
  // hand shape. Z_OPEN can legitimately mean moving in Z with width held.
  const c=runtime.physical_contact??{},edge=c.gripper_edge?.[side],name=side==="left"?"左侧":"右侧";
  const result=(kind,detail)=>({name,kind,detail});
  if(!edge||performance.now()-lastHealthReceivedAt>1000)return result("edge_wait","等待仿真返回掌拳切换状态；当前手形不代表夹爪已执行");
  const reason=c.release_reason?.[side],held=Number.isFinite(c.hold_width_m?.[side]);
  const stable=held&&c.stable_grasps?.[side]===true&&runtime.contacts?.[side]===true;
  if(edge.state==="WAIT_SUPPORT")return result("release",`后端仍报告旧释放限制：${reason==="release_requires_stationary_targets"?"等待停稳":rateSupportDetail()}；当前切换策略应允许空中释放，请检查运行版本`);
  if(edge.state==="WAIT_BOTH")return result("release","本次张开请求等待另一侧拳→掌切换及双掌确认；当前爪宽保持");
  if(edge.state==="CONFIRMING")return result("release",rateReleaseConfirmation(side));
  if(edge.state==="OPENING")return result("edge_open","本次拳→掌已确认，正在匀速开爪；无需落桌或停手，空中也可开爪；换手势、暂停或丢手停止行程");
  if(edge.state==="CLOSING")return result("edge_close","本次掌→拳已确认，正在匀速闭爪；保持握拳，可 Z 升降，真实抓稳后锁宽");
  if(edge.state==="ABORTED")return result("interrupted","本次开合已中止，爪宽保持；重新做掌↔拳切换才会再次开合，当前保持不代表已抓稳");
  const hold=stable?"已真实双指抓稳，保持接触爪宽，不继续加压；":held?"抓持保护保持爪宽，稳定接触待恢复；":"";
  if(edge.state==="WAIT_BASELINE")return result("edge_baseline",`${hold}暂停／丢手后，首次可信张掌或握拳只建立基准，不开合；随后换成相反手形才触发`);
  if(edge.state==="COMPLETE")return result("edge_complete",`${hold}本次开合已结束；保持同一手形不重复触发，可继续移动${!stable&&edge.baseline_mode==="Z_CLOSE"?"。闭合行程已结束但未确认抓稳，请张掌打开后重新对准":""}`);
  if(edge.state==="HOLD")return result("edge_hold",`${hold}${edge.authorized_mode?(resumableGrippers()?"识别暂缓，未完成的开合已保留；恢复原手形后继续，无需再做掌↔拳；真丢手或手动暂停仍会取消":"本次开合暂时保持；短暂识别过渡恢复原手形可继续，完整暂停后需重新切换"):"爪宽保持；I Love You 仅控制 XY，记住最近掌／拳，返回同一手形不续开合"}`);
  const baseline=edge.baseline_mode==="Z_OPEN"?"张掌":edge.baseline_mode==="Z_CLOSE"?"握拳":null;
  const next=edge.baseline_mode==="Z_OPEN"?"握拳触发闭爪":"张掌触发开爪";
  return result("edge_ready",`${hold}${baseline?`当前${baseline}基准已建立；保持同一手形不开合，换成${next}`:"等待可信掌／拳建立切换基准"}${edge.baseline_source==="initial_open_state"?"。初始夹爪全开，首次握拳可触发闭合":""}`);
}
function rateSideStatus(side) {
  const state=engine.interaction?.[side]??{},contact=runtime.physical_contact??{};
  const wire=engine.lastWireModes?.[side==="left"?0:1]??"HOLD",name=side==="left"?"左侧":"右侧";
  const held=Number.isFinite(contact.hold_width_m?.[side]),reason=contact.release_reason?.[side];
  const interrupted=contact.rate_interrupted_release_hold?.[side]===true;
  const stableContact=contact.stable_grasps?.[side]===true&&runtime.contacts?.[side]===true;
  const reading=rateHandReading(side);
  if(reading.stale)return {name,kind:"tracking",detail:"等待新的可信识别画面，当前结果已过期或被拒绝；位置与爪宽保持"};
  if(!reading.hand)return {name,kind:"tracking",detail:"未检测到此侧手；位置与爪宽保持，完整入镜后等待可信恢复"};
  if(engine.tracker.palmBindingGuard)return {name,kind:"binding",detail:"绑定张掌不会开爪；双手先换 I Love You 或握拳，再开始控制"};
  if(edgeTriggeredGrippers()) {
    const p=engine.gestureParameters??{},threshold=rateGestureThreshold(reading.hand.gesture,p,side);
    if(Number.isFinite(threshold)&&reading.hand.gesture_score<threshold)return {name,kind:"hold",detail:`${gestures[reading.hand.gesture]}评分 ${percent(reading.hand.gesture_score)}，需要至少 ${percent(threshold)}；本帧保持，不产生新的开合切换`};
    if(wire==="HOLD"&&state.candidate&&state.candidate!=="HOLD")return {name,kind:"confirmation",detail:"新手势确认 200 ms；本帧保持爪宽，确认后再由后端判断是否构成掌↔拳切换"};
    if(wire==="HOLD"&&state.candidate==="HOLD")return {name,kind:"hold",detail:resumableGrippers()?"当前手势未分配，本帧保持爪宽；追踪可信时保留未完成开合，恢复原手形可继续":"当前手势未分配，本帧保持爪宽；短暂过渡恢复原手形可继续未过期的开合授权"};
    return rateEdgeStatus(side);
  }
  if(levelTriggeredGrippers()) {
    const p=engine.gestureParameters??{},threshold=rateGestureThreshold(reading.hand.gesture,p,side);
    if(Number.isFinite(threshold)&&reading.hand.gesture_score<threshold)return {name,kind:"hold",detail:`${gestures[reading.hand.gesture]}评分 ${percent(reading.hand.gesture_score)}，需要至少 ${percent(threshold)}；当前位置与爪宽保持`};
    if(wire==="HOLD"&&state.candidate&&state.candidate!=="HOLD")return {name,kind:"confirmation",detail:`${rateModeConfirmation()}；过渡期间保持位置和已下发爪宽，不继续旧开合`};
    if(wire==="XY_HOLD")return {name,kind:"xy",detail:"I Love You 控制 XY；保持当前已下发爪宽"};
    if(wire==="Z_CLOSE")return {name,kind:"close",detail:held?(stableContact?"握拳可 Z 升降；已真实抓稳，保持接触爪宽，不继续加压":"抓持保护保持爪宽；稳定接触待恢复"):"握拳持续匀速闭爪；手腕上下控制 Z，停腕仍会闭爪"};
    if(wire==="Z_OPEN")return {name,kind:"open",detail:"张掌持续匀速开爪，同时可控制 Z；此侧独立释放，无需额外释放确认。换 I Love You、未知手势或丢手即保持当前爪宽"};
    return {name,kind:"hold",detail:"当前手势未分配或暂未使能；位置与爪宽保持，再次识别张掌／握拳可继续对应开合"};
  }
  if(interrupted)return {name,kind:"interrupted",detail:"释放已中止 · 爪宽保持；放稳后握拳可重夹，张掌重新确认继续开爪；此保持不代表已抓稳"};
  if(wire==="XY_HOLD")return {name,kind:"xy",detail:"I Love You 控制 XY；保持当前已下发爪宽"};
  if(wire==="Z_CLOSE")return {name,kind:"close",detail:held?(stableContact?"握拳可 Z 升降；已真实抓稳，保持接触爪宽，不继续加压":"抓持保护保持爪宽；当前双指稳定接触待恢复，是否允许移动由仿真检查"):"握拳持续匀速闭爪；手腕上下控制 Z，停腕仍会闭爪"};
  if(wire==="Z_OPEN") {
    const waiting=reason==="release_requires_stable_table_support"?rateSupportDetail():
      reason==="release_confirmation"?rateReleaseConfirmation(side):gestureReleaseLabels[reason];
    if(contact.release_opening?.[side])return {name,kind:"open",detail:"正在匀速开爪；保持张掌并停手，换手势或丢手立即停止剩余行程"};
    return {name,kind:held&&!contact.release_opening?.[side]?"release":"open",detail:held&&!contact.release_opening?.[side]?
      `${reason==="release_confirmation"?"":"张掌可 Z 下降；"}${waiting??"等待放稳及释放确认"}`:
      "张掌持续匀速开爪，同时可控制 Z；换手势或丢手立即停止剩余开爪"};
  }
  if(state.candidate&&state.candidate!=="HOLD")return {name,kind:"confirmation",detail:"新手势确认 200 ms，当前爪宽保持；确认后重建手腕基准"};
  const hand=engine.measurement?.hands?.find(h=>h.bound_side===side),p=engine.gestureParameters??{};
  const threshold=rateGestureThreshold(hand?.gesture,p,side);
  if(Number.isFinite(threshold)&&hand.gesture_score<threshold)return {name,kind:"hold",detail:`${gestures[hand.gesture]}评分 ${percent(hand.gesture_score)}，需要至少 ${percent(threshold)}；当前位置与爪宽保持`};
  return {name,kind:"hold",detail:"手势未分配或评分不足，位置与爪宽保持；请用 I Love You、张掌或握拳"};
}
function gestureSideStatus(side) {
  const state=engine.interaction?.[side]??{},contact=runtime.physical_contact??{};
  const wire=engine.lastWireModes?.[side==="left"?0:1]??"HOLD", name=side==="left"?"左侧":"右侧";
  const physicalLock=Number.isFinite(contact.hold_width_m?.[side]), operationLock=Number.isFinite(contact.operation_hold_width_m?.[side]);
  const releaseReason=contact.release_reason?.[side], locked=physicalLock||operationLock;
  if(contact.release_opening?.[side]||wire==="OPENING")return {name,kind:"opening",detail:"自动开爪；保持可信追踪与停手，无需持续张掌"};
  if(wire==="RELEASE"||locked&&gestureReleaseLabels[releaseReason])return {name,kind:"release",detail:gestureReleaseLabels[releaseReason]??`完整张掌释放确认 ${(contact.release_progress_s?.[side]??0).toFixed(1)} / ${(contact.release_confirm_seconds??.7).toFixed(1)} s`};
  if(locked) {
    const width=operationLock?contact.operation_hold_width_m[side]:contact.hold_width_m[side];
    return {name,kind:"locked",detail:physicalLock?"真实双指抓持保持；支撑停稳后完整张掌确认释放":`握拳锁定当时宽度 ${(width*1000).toFixed(1)} mm${width>=.079?"（全开也会锁定）":""}；完整张掌确认解锁，三指捏合不能解锁`};
  }
  if(wire==="PINCH")return {name,kind:"pinch",detail:continuousBackendRejection()?"前端已接管，但仿真当前拒绝请求，尚不能开合":performance.now()-lastHealthReceivedAt>1000?"前端已接管；当前仿真使能状态待更新":contact.aperture_direct_enabled?.[side]===true?"连续捏合已使能；拇指食指控制此侧，末端保持":"连续捏合已请求；等待仿真确认使能"};
  if(state.mode==="PINCH"&&state.takeover) {
    // desired_width_m may belong to an older PINCH frame. Use the current
    // trusted hand measurement only, and compare against the issued width.
    const hand=engine.measurement?.hands?.find(h=>h.bound_side===side);
    const age=performance.now()-engine.measurement?.capture_browser_monotonic_ms;
    const fresh=age>=0&&age<=(runtime.config?.control?.max_command_age_ms??150)&&performance.now()-lastHealthReceivedAt<=1000;
    const u=fresh&&hand?.valid&&engine.measurement?.accepted_recognition!==false?engine.apertureCalibration.normalize(side,hand.aperture_ratio_image):null;
    const width=runtime.grippers?.[side]?.command_width_m;
    if(Number.isFinite(u)&&Number.isFinite(width)) {
      const target=width/.08, difference=Math.abs(u-target)*100;
      const direction=difference<.05?"已匹配，停住手腕等待接管":u<target?"请张大拇指食指间距":"请收小拇指食指间距";
      return {name,kind:"takeover",detail:`待匹配开度：手 ${Math.round(u*100)}% / 爪 ${Math.round(target*100)}% · 差 ${Math.round(difference)} 个百分点；${direction}`};
    }
    return {name,kind:"takeover",detail:"待匹配开度；等待可信的当前人手与夹爪读数"};
  }
  if(wire==="Z_APPROACH")return {name,kind:"approach",detail:`仅 Z 接近、爪宽保持；停稳 ${Math.round(state.still_ms??0)} / ${engine.gestureParameters.wrist_still_ms} ms 后进入捏合接管`};
  if(wire==="XY_HOLD")return {name,kind:"xy",detail:"I Love You 控制 XY；爪宽保持"};
  if(wire==="Z_HOLD")return {name,kind:"z",detail:"握拳 Z 升降；锁定当前宽度，等待后端确认锁存"};
  if(state.candidate&&state.candidate!=="HOLD"&&state.mode!==state.candidate)return {name,kind:"confirmation",detail:`新手形模式确认中；腕${state.moving?"仍在移动":"已停稳"}，当前爪宽保持`};
  if(state.mode==="PINCH"&&!state.takeover)return {name,kind:"baseline",detail:"开度已接管；首帧保持，随后两指开合控制此侧"};
  if(state.release_cycle_guard)return {name,kind:"cycle",detail:"释放周期已完成；先换 I Love You，再三指伸展重新接管"};
  if(state.profile&&!state.profile.pinch&&!state.profile.xy&&!state.profile.fist&&!state.profile.full_palm)return {name,kind:"geometry",detail:"手形暂不可信；连续捏合需中指、无名指、小指完整伸展"};
  return {name,kind:"hold",detail:state.moving?"腕运动中；连续捏合需先停稳":"保持当前目标；等待可信操作手形"};
}
function continuousControlStatus() {
  const gestureV2=engine.gestureProtocol?.()==="gesture-grasp-release-v2";
  const gestureV3=rateGestures();
  const labels = {palm_clutch:"手掌轻斜进入暂停区，可在画面内挪回舒适位置；直掌恢复 XY，横掌恢复 Z。",
    aperture_takeover:"请调整拇指与食指间距，使手的开度条接近夹爪当前目标，然后恢复控制。",
    continuous_mode_confirmation:"正在确认移动平面，夹爪保持。", continuous_baseline:"基准已建立，停手不移动；移动手腕并捏合可同时控制。",
    partial_grasp_holds_engaged_arm:"单侧已抓住：该侧保持，另一侧继续对准；双侧抓稳后才能抬升。",
    cooperative_axial_conflict:"两手沿物体长轴拉扯，动作已拒绝。双手同向平移，差分前后移动用于转向。",
    release_requires_stable_table_support:"物体尚未稳定支撑在桌面，开爪请求已拒绝。先放稳再慢慢张开。",
    contact_recovery_required:"正在等待有效接触恢复，夹爪保持。", contact_loss:"搬运接触丢失，保持夹爪；持续异常会将本轮标记失败。",
    robot_obstacle_collision:"机械臂碰到挡块或放置台，本轮失败。保存失败或放弃后重置；搬运前先抬高，避免撞台。",
    predicted_robot_obstacle_collision:"预计会碰到挡块，目标已拒绝。可抬起退让，先推移托盘再对准抓耳。",
    predicted_robot_robot_collision:"预计两臂会相碰，目标已拒绝。请先退开，重新调整两臂位置。",
    tracking_recovery:"等待至少 5 个可信新样本并稳定 200 ms。", mode_changed:"移动平面已改变，等待追踪恢复并重建基准。",
    user_pause:"已手动暂停。点击“恢复控制”，再匹配当前夹爪开度。", joint_limit:"接近关节限位，目标已限制。",
    workspace_limit:"接近工作空间边界，整体目标已限制。", ik_unreachable:"当前目标无法到达，保持原目标。",
    gesture_mode_confirmation:"先保持爪宽，等待新手形稳定 200 ms；过渡不会继续旧的捏合。",
    gesture_motion_baseline:"新的手腕基准已建立。移动阶段保持爪宽，停手不补发旧位移。",
    pinch_baseline:"开度接管完成，本帧保持。随后拇指与食指的变化才控制对应夹爪。",
    gesture_geometry_hold:"手形暂不可确认，位置与爪宽保持。三根闲置手指需完整伸展且骨架可信。",
    mixed_motion_planes:"两手移动平面不同，已保持。XY 需双方 I Love You；Z 中可一侧握拳、一侧捏合。",
    operation_lock_hold:"操作锁或真实抓持保持中。三指伸展不能解锁；放稳停手后用完整张掌确认释放。",
    opening_cancelled:"自动开爪已请求中止，保持当时宽度；需停稳并重新整掌确认，不能自动续开。",
    release_cycle_hold:"本次释放已结束，保持开度。先换 I Love You 退出周期，再进入三指预抓并重新接管。",
    release_cycle_exit:"已退出释放周期，本帧保持；新的捏合仍需停稳和开度接管。"};
  labels.gesture_wrist_not_still="后端尚未确认腕部停稳；位置和爪宽保持。先停住手腕，再做两指开合。";
  if(gestureV2) {
    labels.aperture_takeover="停住手腕，调整拇指食指间距以匹配当前夹爪开度；每次移动结束都重新接管，不补发移动期间的开度漂移。";
    labels.continuous_baseline=labels.gesture_motion_baseline;
    labels.release_requires_stable_table_support="真实持物尚未得到稳定支撑。先放稳并停手，再完整张掌确认；空爪操作锁不要求伪造支撑。";
  }
  if(gestureV3)Object.assign(labels,{
    user_pause:"已手动暂停，位置与爪宽保持。将双手放回舒适位置，点击恢复控制即可，无需匹配开度。",
    hand_near_frame_edge:"完整双手移回内框后，等待可信追踪恢复并重建手腕基准；无需开度标定。",
    binding_complete:"身份已绑定。绑定用的张掌保持爪宽；双手先换 I Love You 或握拳，再开始控制。",
    binding_gesture_released:"绑定保护已解除；操作手势稳定 200 ms 后建立新手腕基准。",
    mixed_motion_planes:"XY 需要双手同时 I Love You；Z 中可一侧张掌、一侧握拳。当前两侧位置与爪宽都保持。",
    gesture_mode_confirmation:"新手势确认 200 ms；过渡期间停止旧开合，保持当前爪宽。",
    gesture_motion_baseline:"新手腕基准已建立，首帧不移动；张掌持续开爪，握拳持续闭爪，I Love You 保持爪宽。",
    rate_gesture_unqualified:"当前手势未知或评分不足，保持目标；只使用 I Love You、张掌或握拳。",
    release_requires_stable_table_support:`${rateSupportDetail()}。夹爪保持；桌面支撑且停稳后，张掌确认释放。`
  });
  if(gestureV3&&ratePartialGraspDetail())labels.partial_grasp_holds_engaged_arm=ratePartialGraspDetail();
  if(levelTriggeredGrippers())Object.assign(labels,{
    binding_gesture_released:`绑定保护已解除；${rateModeConfirmation()}后建立新手腕基准。`,
    gesture_mode_confirmation:`${rateModeConfirmation()}；过渡期间停止旧开合，保持当前爪宽。`,
    record_start_baseline:"录制已开始，等待可信手势。张掌匀速开爪，握拳匀速闭爪，I Love You 控制 XY 并保持爪宽。"
  });
  if(edgeTriggeredGrippers())Object.assign(labels,{
    user_pause:"已手动暂停，位置与爪宽保持。将双手放回舒适位置后点击恢复控制；首次可信掌／拳只建立基准，再换相反手形才开合，无需匹配开度。",
    hand_near_frame_edge:"完整双手移回内框，等待可信追踪恢复。首次可信掌／拳只建立基准，不开合；再换相反手形触发。",
    tracking_recovery:"等待至少 5 个可信新样本并稳定 200 ms。恢复后的首次掌／拳只建立基准，再换相反手形才开合。",
    gesture_motion_baseline:"新手腕基准已建立，首帧不移动。掌→拳触发一次闭合，拳→掌触发一次张开；同一手形不重复触发。"
  });
  if(edgeTriggeredGrippers()&&ratePartialGraspSide()) {
    const holding=ratePartialGraspSide()==="left"?"左":"右";
    labels.partial_grasp_holds_engaged_arm=`${holding}臂已抓稳，保护器拒绝本次双臂请求；另一爪本次开合也未执行。等待新请求被接受后，先建立掌／拳基准，再切换开合。`;
  }
  if (!connected) return {title:"等待连接",detail:reasons.backend_disconnected,state:"waiting"};
  if (runtime.diagnostics?.error || runtime.diagnostics?.dropped || runtime.research_log_dropped) return {title:"日志异常",detail:"诊断或原始研究日志出现丢弃或写入错误。请暂停并保留当前记录，查看高级诊断。",state:"paused"};
  if(runtime.task_state==="FAIL")return {title:"本轮失败 · 保持夹爪",detail:labels[runtime.failure_reason]??failureLabels[runtime.failure_reason]??runtime.failure_reason,state:"paused"};
  if(runtime.saving)return {title:"正在保存",detail:`正在从物理快照生成 ${(runtime.config?.recording?.cameras??["oblique"]).join(" / ")} 视频，请等待保存确认。`,state:"waiting"};
  if(runtime.task_state==="DONE") {
    if(runtime.recording && !runtime.episode_end && runtime.success_tail?.seconds > 0)return {title:"放置成功 · 记录开爪收尾",detail:`成功已确认，再记录 ${runtime.success_tail.seconds} 秒；机械臂位置保持，只继续已确认的开爪动作。结束后保存按钮恢复。`,state:"waiting"};
    if(!runtime.recording)return {title:"练习放置完成",detail:"真实接触、支撑和目标条件已满足；本轮未录制，无法保存为示范。请点击“重置 · 下一摆放”，再点击“开始录制”后操作。",state:"complete"};
    if(pendingOperation)return {title:"放置完成 · 等待操作确认",detail:"本轮已录制，上一项操作仍在等待仿真确认；确认后再选择保存。",state:"waiting"};
    if(transportFrozen)return {title:"放置完成 · 控制通道恢复中",detail:"本轮已录制，正在确认连接；恢复后可保存，请勿重复提交。",state:"waiting"};
    return {title:"放置完成",detail:"真实接触、支撑和目标条件已满足，本轮已录制，可以保存本轮。",state:"complete"};
  }
  if(!stream)return {title:"启动本机摄像头",detail:gestureV3?"双掌保持 2 秒绑定，无需捏合标定。绑定后双手先换 I Love You 或握拳，解除绑定张掌保护。":"双掌保持 2 秒绑定，随后按提示分别标定舒适闭合和张开。",state:"waiting"};
  if(!engine.tracker.calibrated)return {title:"等待双掌绑定",detail:calibrationDetail(engine.tracker.gestureCalibrationStatus()),state:"waiting"};
  const phase=engine.apertureCalibration.phase;
  if(output.reason==="hand_near_frame_edge")return {title:"边缘保护 · 保持夹爪",detail:labels.hand_near_frame_edge??reasons.hand_near_frame_edge,state:"paused"};
  if(!gestureV3&&phase!=="READY")return {title:phase==="CLOSE"?"标定 1/2 · 舒适捏合":"标定 2/2 · 舒适张开",
    detail:phase==="CLOSE"?"两手各用拇指轻触食指，稳定保持 1.5 秒。机器人保持，无需点击。":"两手舒适地张开拇指与食指，稳定保持 1.5 秒。无需用力握拳或伸直全部手指。",state:"waiting"};
  if(engine.manualPaused||output.reason==="user_pause")return {title:"手动暂停",detail:labels.user_pause,state:"paused"};
  if(transportFrozen)return {title:"控制通道恢复中",detail:reasons.control_transport_error,state:"paused"};
  if(performance.now()-lastHealthReceivedAt>1000)return {title:"等待仿真状态更新",detail:"当前执行状态尚未确认；旧拒绝或旧使能状态不能代表现在，正在等待新的仿真回复。",state:"waiting"};
  const rejection=continuousBackendRejection();
  if(rejection)return {title:gestureWaitTitles[rejection]??"仿真暂未接受控制",detail:`仿真当前拒绝：${labels[rejection]??reasons[rejection]??rejection}`,state:"paused"};
  const reason=output.reason??(runtime.paused===true?runtime.pause_reason:null);
  if(gestureV3) {
    const globalHold=["hand_lost","tracking_recovery","tracking_recovered","ambiguous_handedness","identity_conflict","tracking_stalled","stale_recognition","camera_stalled","mixed_motion_planes","binding_complete","binding_gesture_released","record_start_baseline"];
    if(globalHold.includes(reason))return {title:gestureWaitTitles[reason]??"控制保持",detail:labels[reason]??reasons[reason]??reason,state:reason.includes("recover")?"recovering":"waiting"};
    const wire=engine.lastWireModes??["HOLD","HOLD"],sides=["left","right"].map(rateSideStatus);
    const partial=ratePartialGraspDetail(),detail=[partial,...sides.map(s=>`${s.name}：${s.detail}`)].filter(Boolean).join("；");
    if(sides.some(side=>side.kind==="tracking"))return {title:"等待新的可信识别画面",detail,state:"waiting"};
    if(sides.some(side=>side.kind==="release")&&(Object.values(runtime.physical_contact?.release_reason??{}).includes("release_confirmation")||Object.values(runtime.physical_contact?.gripper_edge??{}).some(edge=>edge.state==="CONFIRMING")))return {title:edgeTriggeredGrippers()?"释放确认 · 保持张掌":"释放确认 · 保持张掌并停手",detail,state:"waiting"};
    if(sides.some(side=>side.kind==="release"))return {title:edgeTriggeredGrippers()?"Z 升降 · 开爪等待确认":"Z 升降 · 开爪等待支撑 / 确认",detail,state:"waiting"};
    if(partial) {
      const held=ratePartialGraspSide(),free=held==="left"?"right":"left";
      const freeStatus=sides[free==="left"?0:1];
      if((edgeTriggeredGrippers()&&["edge_open","edge_close"].includes(freeStatus.kind))||(levelTriggeredGrippers()&&["open","close"].includes(freeStatus.kind)))
        return {title:`${held==="left"?"左":"右"}臂保持 · ${free==="left"?"左":"右"}爪开合中`,detail,state:"active"};
      return {title:"单侧抓稳保持 · 另一侧对准",detail,state:"waiting"};
    }
    if(wire.every(mode=>mode==="XY_HOLD"))return {title:"XY 移动 · 爪宽保持",detail,state:"active"};
    if(edgeTriggeredGrippers()) {
      if(sides.some(side=>side.kind==="edge_open"||side.kind==="edge_close"))return {title:"Z 升降 · 本次开合执行中",detail,state:"active"};
      if(sides.some(side=>side.kind==="edge_baseline"))return {title:"开合等待新基准",detail,state:"waiting"};
      if(sides.some(side=>side.kind==="interrupted"))return {title:"开合已中止 · 等待再次切换",detail,state:"waiting"};
      return {title:wire.some(mode=>mode==="Z_OPEN"||mode==="Z_CLOSE")?"Z 升降 · 爪宽保持":"爪宽保持 · 等待掌拳切换",detail,state:"waiting"};
    }
    if(wire.some(mode=>mode==="Z_OPEN"||mode==="Z_CLOSE"))return {title:"Z 升降 · 张掌开 / 握拳闭",detail,state:"active"};
    return {title:gestureWaitTitles[reason]??"等待操作手势",detail,state:"waiting"};
  }
  if(!gestureV2&&reason&&reason!=="continuous_baseline")return {title:gestureWaitTitles[reason]??"控制等待",detail:labels[reason]??reasons[reason]??reason,state:"paused"};
  if(gestureV2) {
    const globalHold=["hand_lost","tracking_recovery","tracking_recovered","ambiguous_handedness","identity_conflict","tracking_stalled","stale_recognition","camera_stalled","gesture_wrist_not_still","mixed_motion_planes"];
    if(globalHold.includes(reason))return {title:gestureWaitTitles[reason],detail:labels[reason]??reasons[reason]??reason,state:reason.includes("recover")?"recovering":"paused"};
    const wire=engine.lastWireModes??["HOLD","HOLD"];
    const sides=["left","right"].map(gestureSideStatus),detail=sides.map(s=>`${s.name}：${s.detail}`).join("；");
    if(wire.includes("OPENING"))return {title:"自动开爪",detail,state:"active"};
    if(wire.includes("RELEASE"))return {title:"整掌释放确认 / 释放等待",detail,state:"waiting"};
    const active=sides.filter(s=>s.kind==="pinch");
    if(active.length) {
      const enabled=active.some(s=>runtime.physical_contact?.aperture_direct_enabled?.[s.name==="左侧"?"left":"right"]===true);
      return {title:`${active.length===2?"双侧":active[0].name}连续捏合${enabled?"":" · 待仿真确认"}`,detail,state:enabled?"active":"waiting"};
    }
    if(wire.includes("Z_APPROACH"))return {title:"仅 Z 接近 · 爪宽保持",detail,state:"active"};
    if(wire.every(v=>v==="XY_HOLD"))return {title:"XY 移动 · 爪宽保持",detail,state:"active"};
    if(wire.includes("Z_HOLD"))return {title:"Z 升降 · 操作锁宽",detail,state:"active"};
    return {title:gestureWaitTitles[reason]??(sides.some(s=>s.kind==="locked")?"握拳操作锁 / 抓持保持":"等待操作手形"),detail:detail+(labels[reason]?`。${labels[reason]}`:""),state:"waiting"};
  }
  return {title:`${output.mode} · 连续控制`,detail:runtime.task_state==="DUAL_GRASPED"?"双侧真实抓稳。同向移动平移，差分前后移动小幅转向；空中保持抓持。":"手腕控制位置，拇指与食指独立控制对应夹爪；停手停止新增位移。",state:"active"};
}
function renderContinuousRobotReadings() {
  if (!connected) {
    $("gripper-detail").textContent="仿真未连接 · 请求 / 执行 / 实测开度不可用";
    $("grasp-state").textContent="仿真未连接 · 抓持与释放状态不可用";
    $("diagnostic-health").textContent="仿真未连接 · 日志状态不可用";
    $("task-state").textContent="—";
    $("gripper").textContent="状态不可用";
    $("recording-state").textContent="状态不可用";
    $("task-elapsed").textContent="—";
    $("session-mode").textContent="等待仿真状态";
    return;
  }
  const contact=runtime.physical_contact??{}, mm=v=>Number.isFinite(v)?`${(1000*v).toFixed(1)} mm`:"—";
  const labels={release_needs_relaxed_intent:"先收回张开意图，再主动张开释放",grasp_hold:"抓持保持，可继续移动",
    release_requires_both_sides:"等待双侧张开意图",release_requires_stable_table_support:"等待桌面稳定支撑",
    release_requires_stationary_targets:"等待停止移动",release_intent_inactive:"抓持保持",release_input_interrupted:"输入中断，抓持保持",
    release_mode_changed:"切换模式，重新确认释放",release_completed:"释放已接受"};
  const rows=[],states=[];
  for(const side of ["left","right"]) {
    const g=runtime.grippers?.[side]??{},name=side==="left"?"左":"右",reason=contact.release_reason?.[side];
    const fingers=runtime.contacts?.[`${side}_fingers`],count=Array.isArray(fingers)?fingers.filter(Boolean).length:null;
    const force=runtime.contacts?.finger_normal_force_n?.[side];
    const forceText=Array.isArray(force)&&force.length===2&&force.every(Number.isFinite)?` · 指力 ${force.map(f=>f.toFixed(2)).join(" / ")} N`:"";
    const contactText=contact.stable_grasps?.[side]===true&&runtime.contacts?.[side]?"真实双指抓稳":count===1?"仅单指接触，未抓稳":runtime.contacts?.[side]?"双指接触，等待抓稳确认":"无双指接触";
    rows.push(`${name}爪 请求 ${mm(g.requested_width_m)} · 执行 ${mm(g.command_width_m)} · 实测 ${mm(g.actual_width_m)} · ${contactText}${forceText}`);
    const held=Number.isFinite(contact.hold_width_m?.[side]),operationHeld=Number.isFinite(contact.operation_hold_width_m?.[side]);
    let state=contact.control_protocol==="gesture-grasp-release-v2"?(held?"真实抓持保护":operationHeld?`操作锁宽 ${mm(contact.operation_hold_width_m[side])} · 无真实抓持锁`:"未锁定 · 需停稳接管"):(held?"抓持保持，可继续移动":"自由开合");
    if(held&&reason==="release_confirmation")state=`释放确认 ${(contact.release_progress_s?.[side]??0).toFixed(1)} / ${(contact.release_confirm_seconds??.7).toFixed(1)} s，保持张开并停手`;
    else if((held||reason==="release_completed")&&labels[reason])state=labels[reason];
    if(contact.release_opening?.[side])state=contact.control_protocol==="gesture-grasp-release-v2"?"自动开爪 · 保持追踪与停手，无需持续张掌":"正在开爪，请保持张开并停手，直到完成";
    else if(contact.control_protocol==="gesture-grasp-release-v2"&&reason==="release_completed")state="自动开爪完成 · 保持全开，退出周期后再接管";
    if(contact.control_protocol==="gesture-rate-gripper-v3") {
      const mode=contact.interaction_modes?.[side];
      const stableContact=contact.stable_grasps?.[side]===true&&runtime.contacts?.[side]===true;
      state=held?(stableContact?"真实双指抓稳 · 保持接触爪宽":"抓持保护保持爪宽 · 稳定接触待恢复"):mode==="Z_CLOSE"?"匀速闭爪 · 可 Z 升降":mode==="Z_OPEN"?"匀速开爪 · 可 Z 升降":mode==="XY_HOLD"?"XY 移动 · 爪宽保持":"保持当前爪宽";
      if(held&&mode==="Z_OPEN")state=reason==="release_requires_stable_table_support"?rateSupportDetail():labels[reason]??"等待支撑 / 释放确认";
      if(held&&reason==="release_confirmation")state=rateReleaseConfirmation(side);
      if(contact.release_opening?.[side])state="匀速开爪 · 持续张掌，换手势即停止";
      if(contact.rate_interrupted_release_hold?.[side]===true)state="释放已中止 · 爪宽保持；放稳后握拳可重夹，张掌重新确认继续开爪";
      if(levelTriggeredGrippers())state=rateSideStatus(side).detail;
      if(edgeTriggeredGrippers())state=rateEdgeStatus(side).detail;
    }
    states.push(`${name}爪 · ${state}`);
  }
  $("gripper-detail").textContent=rows.join("\n");
  $("grasp-state").textContent=states.join("\n");
  $("diagnostic-health").textContent=runtime.diagnostics?`日志 ${runtime.diagnostics.written} 行 / 丢弃 ${runtime.diagnostics.dropped} / ${runtime.diagnostics.error??"正常"}`:"等待后端日志状态";
}
function renderContinuousState() {
  document.body?.classList?.add("continuous-ui");
  for(const n of document.querySelectorAll(".paper-only"))n.hidden=false;
  for(const n of document.querySelectorAll(".legacy-only"))n.hidden=true;
  const sceneNumber=Number.isInteger(runtime.scene_index)?` · 第 ${runtime.scene_index+1} 组摆放`:"";
  $("scenario-name").textContent=`任务 ${runtime.task_id??runtime.config?.task_binding?.task_id??"—"} · 场景 ${runtime.config?.scene?.scenario??"—"}${sceneNumber} · 种子 ${runtime.seed??"—"}`;
  $("scene-reset-help").textContent=connected?`未录制时“重置”切换下一摆放；保存或放弃也会切换。会话种子 ${runtime.session_seed??"—"}；当前摆放种子 ${runtime.seed??"—"}。`:"仿真未连接 · 摆放与种子状态待同步。";
  $("mode").textContent=output.mode;
  $("task-state").textContent=runtime.task_state==="PARTIAL_CONTACT"?"单侧/部分接触":tasks[runtime.task_state]??runtime.task_state;
  $("gripper").textContent=edgeTriggeredGrippers()?"掌↔拳切换开合":rateGestures()?"张掌开 / 握拳闭":"连续开合 / 抓持保持";
  $("mode-label").textContent="当前移动模式";
  $("recording-state").textContent=runtime.saving?"保存中":runtime.recording?`${runtime.episode_end?"已截止 · 待保存":"录制中"} · ${runtime.frames} 帧`:"练习 · 未录制";
  $("session-mode").textContent=runtime.recording?(runtime.episode_end?"已截止 · 待保存":"录制中"):"练习 · 不限时";
  $("task-elapsed").textContent=runtime.recording?`${(runtime.recording_wall_seconds??0).toFixed(1)} s`:"—";
  $("dataset-counts").textContent=`成功 ${runtime.saved_success??0} / 失败 ${runtime.saved_failure??0}`;
  const frozen=Boolean(runtime.recording||runtime.saving||pendingOperation);
  $("motion-speed").value=String(engine.config.motion_speed_scale);$("motion-speed").disabled=frozen;
  $("recalibrate-aperture").disabled=frozen||!engine.tracker.calibrated;
  const gestureV2=engine.gestureProtocol?.()==="gesture-grasp-release-v2";
  const gestureV3=rateGestures();
  $("recalibrate-aperture").hidden=gestureV3;
  $("motion-detail").textContent=gestureV3?"I Love You：XY 移动且保持爪宽；张掌：匀速开爪 + Z；握拳：匀速闭爪 + Z。停腕仍会开合，换手势停止。":gestureV2?"I Love You：XY 保持；握拳：锁宽 / Z。三指伸展：腕运动仅 Z，停稳后接管捏合。两手同属 Z 可独立分工。":"历史连续协议 · 当前使能见状态与诊断。";
  if(levelTriggeredGrippers()) {
    $("motion-detail").textContent=`I Love You：XY 移动且保持爪宽；张掌：Z + 匀速开爪；握拳：Z + 匀速闭爪。${rateModeConfirmation()}；左右独立，无额外释放确认。`;
  }
  if(edgeTriggeredGrippers())$("motion-detail").textContent="I Love You：XY 移动且锁宽；张掌／握拳：Z 移动。确认掌→拳触发一次闭合，拳→掌触发一次张开；同一手形不重复触发。";
  for(const side of ["left","right"]){
    const reading=gestureV3?rateHandReading(side):null;
    const h=gestureV3?reading.hand:engine.measurement?.hands?.find(h=>h.bound_side===side);
    const aperture=h?.valid?engine.apertureCalibration.normalize(side,h.aperture_ratio_image):null;
    const validity=h?.invalid_reason==="hand_near_frame_edge"?"边缘保护":h?.valid?"可信":"无效";
    $(`${side}-gesture`).textContent=h?(gestureV3?`${validity} · ${gestures[h.gesture]??h.gesture} · 评分 ${percent(h.gesture_score)}`:`${validity} · ${gestures[h.gesture]??h.gesture} / 开度 ${aperture===null?(engine.apertureCalibration.phase==="READY"?"暂不可用":"待标定"):percent(aperture)}`):reading?.stale?"识别已过期或被拒绝 · 等待新画面":"未检测到";
    $(`${side}-identity`).textContent=h?`原标签 ${h.raw_label} → ${h.bound_side} · ${percent(h.identity_score)}${h.near_edge?" · 接近画面边缘":""}`:"等待测量";
    const interaction=engine.interaction?.[side],modeLabels={XY_HOLD:"XY · 爪宽保持",Z_OPEN:"Z · 匀速开爪",Z_CLOSE:"Z · 匀速闭爪",Z_HOLD:"Z · 操作锁宽",Z_APPROACH:"仅 Z 接近 · 爪宽保持",PINCH:"停稳 · 连续捏合",RELEASE:"整掌确认",OPENING:"自动开爪",HOLD:"保持"};
    if(edgeTriggeredGrippers()){modeLabels.Z_OPEN="Z · 张掌";modeLabels.Z_CLOSE="Z · 握拳";}
    const wire=engine.lastWireModes?.[side==="left"?0:1]??"HOLD";
    const accepted=runtime.physical_contact?.interaction_modes?.[side];
    const sideStatus=gestureV3?rateSideStatus(side):gestureV2?gestureSideStatus(side):null,rejection=continuousBackendRejection();
    const actual=performance.now()-lastHealthReceivedAt>1000?"状态待更新":rejection?"当前请求未接受":modeLabels[accepted]??accepted??"待同步";
    $(`${side}-interaction`).textContent=gestureV2||gestureV3?`请求 ${modeLabels[wire]??wire} · 仿真实际 ${actual} · ${sideStatus.detail}${sideStatus.kind==="pinch"?" · 已接管":""}${rejection?` · 后端：${gestureWaitTitles[rejection]??rejection}`:""}`:"历史连续协议";
    $(`${side}-evidence`).value=aperture??0;
    $(`${side}-evidence`).hidden=gestureV3||aperture===null;
    $(`${side}-evidence`).setAttribute?.("aria-label",`${side} 人手连续开度`);
  }
  $("unbind-hands").disabled=!engine.tracker.calibrated||frozen;
  const calibrated=engine.tracker.calibrated,ac=engine.apertureCalibration.status(),bc=engine.tracker.gestureCalibrationStatus();
  $("calibration-state").textContent=!calibrated?`双掌绑定 ${(bc.palm_progress*2).toFixed(1)}/2 秒`:gestureV3?"身份已绑定 · 无需开度标定":ac.phase==="READY"?"身份已绑定 · 左右开度已标定":`舒适${ac.phase==="CLOSE"?"捏合":"张开"} ${(ac.progress*1.5).toFixed(1)}/1.5 秒`;
  $("calibration-progress").hidden=calibrated&&(gestureV3||ac.phase==="READY");$("calibration-progress").value=calibrated?ac.progress:bc.palm_progress;
  $("calibration-detail").hidden=true;
  $("pause-button").textContent=engine.manualPaused?"恢复控制":"暂停 / 重定位";
  renderControlStatus();
  renderContinuousRobotReadings();
  const helpKey=levelTriggeredGrippers()?`gesture-palm-fist-level-v2-${engine.gestureParameters?.mode_confirm_ms??200}`:edgeTriggeredGrippers()?"gesture-palm-fist-edge-v1":gestureV3?"gesture-rate-v3":gestureV2?"gesture-v2":"legacy";
  if($("operation-help").dataset.continuous!==helpKey){
    $("operation-help").dataset.continuous=helpKey;
    $("operation-help").innerHTML="<p><strong>绑定与标定</strong>双掌 2 秒绑定，再按提示分别标定拇食舒适捏合和张开；这些步骤保持机器人，不执行抓放。</p><p><strong>I Love You / 握拳</strong>I Love You 控制 XY 并保持爪宽；握拳控制 Z 并永久操作锁住当前宽度，不会强制全闭，也不是已真实抓稳。</p><p><strong>三指伸展与接管</strong>中指、无名指、小指伸展，拇指与食指自由开合。腕运动阶段仅 Z；停稳并重新匹配开度后，位置保持、两指独立捏合。None 标签仍可由可信三指几何使能，未知骨架不会继续旧开合。</p><p><strong>整掌释放与自动开爪</strong>锁定后放稳停手，完整张掌约 700 ms，由后端核对支撑与双侧意图。授权后自动全开，无需持续张掌或精确大指距；暂停、丢手、握拳、I Love You 或明显腕移动中止，恢复后重新确认。开完先换 I Love You 退出周期，再预抓接管。</p><p><strong>双手分工</strong>XY 需双方 I Love You；同属 Z 可一侧握拳保持、另一侧捏合。XY / Z 混合会保持两臂。推物可先捏到需要的爪宽，再切移动手形保持。</p>";
    if(!gestureV2)$("operation-help").innerHTML="<p><strong>历史连续协议</strong>拇食开度跟随及物理抓持保持按当前后端状态执行；此兼容分支不具有新版手形分工。主动释放需张开指距确认，并持续保持到开爪完成；输入中断停止行程。</p>";
    if(gestureV3)$("operation-help").innerHTML="<p><strong>双掌绑定</strong>双手张掌稳定 2 秒后绑定，无需开度标定。绑定张掌只保持；双手先换 I Love You 或握拳解除绑定保护。</p><p><strong>I Love You · XY</strong>双手同时 I Love You，手腕增量控制 XY，夹爪保持当前已下发宽度。切换时不继续旧的开合行程。</p><p><strong>张掌 / 握拳 · Z 与开合</strong>张掌持续匀速打开，握拳持续匀速闭合，两者手腕上下都控制 Z。左右夹爪独立，可一侧张掌另一侧握拳；新手势确认 200 ms，首帧只重建移动基准。停腕不会停止开合，未知手势、切换、暂停或丢手会停止剩余行程。</p><p><strong>抓稳与释放保护</strong>每爪双指真实抓稳后保持接触宽度，握拳不继续加压。持物张掌可以下降，但必须桌面支撑、停稳并确认约 700 ms 后才打开；双抓时两侧都需张掌。继续保持张掌完成开爪，中断后重新确认。</p><p><strong>恢复与提示</strong>暂停 / 重定位和 Esc 可停止动作。丢手只暂停，放回内框等待可信恢复，随后重建基准；无需匹配指间距。手势未知或评分不足时查看左右卡片，XY / Z 混合会保持双方。</p>";
    if(edgeTriggeredGrippers())$("operation-help").innerHTML="<p><strong>双掌绑定</strong>双手张掌稳定 2 秒绑定，无需指距标定。绑定张掌只保持；双手先换 I Love You 或握拳。初始夹爪确认全开时，首次握拳可闭合。</p><p><strong>移动与开合分开</strong>双手 I Love You 控制 XY 并保持爪宽；张掌／握拳均控制 Z。新手形稳定 200 ms 后，掌→拳触发一次匀速闭合，拳→掌触发一次匀速张开；两侧独立，同一手形不重复触发。I Love You 停止剩余行程并记住最近掌／拳，返回同一手形不续开合。</p><p><strong>抓稳与释放</strong>主动闭爪后的双指真实抓稳才锁宽；先抓稳的一侧保持位置，另一侧仍可对准和掌↔拳开合，已抓侧手腕微动不会阻止另一爪开合。拳→掌触发释放，仍需约 700 ms 张掌确认；双抓需双方都完成切换。无需落桌或停手，空中也可开爪；是否已放稳由操作者判断。保持张掌直到行程完成。</p><p><strong>暂停与恢复</strong>短暂识别过渡停止行程，恢复原手形可继续本次授权。手动暂停、丢手、过期或重连会清除开合授权：恢复后首次可信掌／拳只建立基准，不开合；再换相反手形才触发。Esc／空格只暂停，恢复请用按钮；不需要匹配指距。</p>";
    if(levelTriggeredGrippers())$("operation-help").innerHTML=`<p><strong>双掌绑定</strong>双手张掌稳定 2 秒绑定，无需指距标定。绑定张掌只保持；双手先换 I Love You 或握拳解除绑定保护。</p><p><strong>I Love You · XY</strong>双手同时 I Love You，手腕增量控制 XY，并保持各自当前已下发爪宽。</p><p><strong>张掌 / 握拳 · Z 与开合</strong>${rateModeConfirmation()}后，张掌持续匀速打开，握拳持续匀速闭合，两者手腕上下均控制 Z。左右独立，一侧张掌即可释放对应爪；无需额外释放确认。双指真实抓稳后保持接触宽度。</p><p><strong>过渡与重定位</strong>候选手势尚未稳定、未知手势、I Love You、暂停或丢手都会停止旧开合并保持当时爪宽。再次张掌或握拳可继续对应开合，不要求先做相反手形。</p><p><strong>暂停与恢复</strong>Esc／空格只暂停，恢复请用按钮。丢手后保留身份绑定，完整入镜等待至少 5 个可信新样本并稳定 200 ms；恢复时重建手腕基准。XY / Z 混合仍保持双方。</p>`;
    $("recovery-help").textContent="S1 为桌面搬运，S2 为有挡块的推抓，S3 为搬到高5厘米的放置台。S3 请先抬高托盘底面越过台顶，再平移、对齐、下降松爪。切换前保存/放弃并退出，运行 START-PAPERA.cmd，在任务菜单中选择对应任务。默认不录真人视频。";
  }
  updateButtons();renderViews();
}
function renderState() {
  if (engine.continuous) { renderContinuousState(); return; }
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
  if (!pendingOperation && runtime.last_command_result?.accepted === false && ["record_start", "record_save", "record_failure", "record_discard", "reset", "record_stop"].includes(runtime.last_command_result.command)) $("record-feedback").textContent = `操作未执行：${reasons[runtime.last_command_result.reason] ?? runtime.last_command_result.reason ?? "安全条件不满足"}`;
  renderControlStatus();
  updateButtons(); renderViews();
}
function drawHands(result) {
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) { canvas.width = video.videoWidth; canvas.height = video.videoHeight; }
  context.clearRect(0, 0, canvas.width, canvas.height);
  if(engine.continuous) {
    const edge=(result.landmarks??[]).some(points=>points.some(p=>p.x<.06||p.x>.94||p.y<.06||p.y>.94));
    context.save();context.strokeStyle=edge?"#f0a43c":"#8aa9ba";context.lineWidth=edge?5:2;
    context.strokeRect(.06*canvas.width,.06*canvas.height,.88*canvas.width,.88*canvas.height);context.restore();
  }
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
    measurement: handMeasurement(data.result, {id: `${recognitionClientId}:${sourceId}`, session: data.session, epoch: data.control_epoch,
      capturedAt: data.capturedAt, capturedEpochMs: data.capturedEpochMs, inferenceMs: data.inferenceMs, receivedAt: now,
      recognitionCompletedEpochMs: data.recognitionCompletedEpochMs, workerCompletedAt: data.workerCompletedAt, workerTimeOriginMs: data.workerTimeOriginMs,
      width: data.width ?? video.videoWidth, height: data.height ?? video.videoHeight, swapLabels: $("swap-labels").checked, dropReason,gestureParameters:runtime.action_contract?.gesture_parameters}),
    observed_hands: detections.map((hand) => ({
      label: hand.side ?? "unknown", gesture: hand.gesture ?? "None",
      score: finiteOrNull(hand.confidence), identity_score: finiteOrNull(hand.handedness_score),
    })),
  };
}
function handleRecognition(data) {
  const now = performance.now();
  const detections = detectionsFromMediaPipe(data.result, $("swap-labels").checked);
  const maxAge = engine.tracker.calibrated
    ? Math.min(RECOGNITION_MAX_AGE_MS, runtime.config?.control?.max_command_age_ms ?? 150)
    : Math.min(500, engine.config.calibration_max_result_age_ms);
  const mismatchedFrame = data.frameId !== workerFrameId;
  const staleContext = data.session !== runtime.session ? "backend_session_changed" : data.control_epoch !== runtime.control_epoch ? "scene_reset" : null;
  const dropReason = !stream ? "camera_stopped" : mismatchedFrame ? "unexpected_frame_id" : staleContext ?? (!Number.isFinite(data.capturedAt) || !Number.isFinite(data.capturedEpochMs) || now - data.capturedAt < 0 || now - data.capturedAt > maxAge ? "stale_recognition" : null);
  const diagnostic = recognitionDiagnostic(data, detections, now, dropReason);
  const before=Object.fromEntries(["left","right"].map(side=>[side,engine.interaction?.[side]?.motionAnchor?[...engine.interaction[side].motionAnchor]:null]));
  if(!stream) {journalMeasurement(diagnostic,{telemetry:{},reason:"camera_stopped"},before);return;}
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
    const rejected = pauseUntrustedTracking(dropReason);
    rejected.telemetry.recognition_result = diagnostic;
    journalMeasurement(diagnostic,rejected,before);
    // Use the current command envelope for a safe pause; the old camera
    // context remains inside the diagnostic rather than refreshing identity.
    queueOutput(rejected, now);
    if (now - lastUiRenderAt >= 100) { renderState(); lastUiRenderAt = now; }
    return;
  }
  lastRecognitionIssue = null;
  engine.setMeasurement?.(diagnostic.measurement);
  let next;
  if (!connected || transportFrozen || pendingOperation || runtime.saving) {
    engine.tracker.update(detections, data.capturedAt);
    next = engine.pause(!connected ? (serviceStopped ? "service_stopped" : "backend_disconnected") : transportFrozen ? "control_transport_error" : runtime.saving ? "saving" : "operation_pending");
  } else next = engine.update(detections, data.capturedAt);
  next.measurementAt = ["dual_motion", "dual_gripper", "continuous_input"].includes(next.command?.command) ? data.capturedEpochMs : Date.now();
  Object.assign(next.telemetry, { capture_time_ms: data.capturedEpochMs, inference_ms: data.inferenceMs, recognition_age_ms: now - data.capturedAt, recognition_result: diagnostic, clock_pair:runtime.clock_pair });
  journalMeasurement(diagnostic,next,before);
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
      if (worker !== gestureWorker || generation !== workerGeneration) {
        if(data.type==="result")journalMeasurement(recognitionDiagnostic(data,
          detectionsFromMediaPipe(data.result,$("swap-labels").checked),performance.now(),"worker_replaced"),
          {telemetry:{},reason:"worker_replaced"},{});
        return;
      }
      if (data.type === "ready") { clearTimeout(timer); workerReady = true; resolve(); }
      else if (data.type === "result") handleRecognition(data);
      else if (data.type === "error") {
        measurementJournal.add("camera_event",{event:"worker_error",frame_id:data.frameId??null,
          reason:data.message,stage:data.stage??null},researchContext());
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
  if(sealingHandBoundary)return;
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
  if (!sealingHandBoundary && workerReady && !workerFrameBusy && video.readyState >= 2 && video.currentTime !== lastVideoTime && now - lastFrameSubmittedAt >= engine.config.control_interval_ms) void submitCameraFrame(now);
  const stallLimit = engine.tracker.calibrated ? engine.config.hand_lost_grace_ms : engine.config.calibration_sample_gap_ms;
  if (now - lastReceivedResultAt > stallLimit && Number.isFinite(lastReceivedResultAt)) {
    queueOutput(pauseUntrustedTracking("camera_stalled"), now);
    if (now - lastUiRenderAt >= 100) { renderState(); lastUiRenderAt = now; }
  }
  frameId = requestAnimationFrame(detectFrame);
}
async function startCamera() {
  $("camera-button").disabled = true; $("camera-state").textContent = "加载模型 / 等待授权";
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("请通过 localhost 使用 Edge 或 Chrome。");
    await createGestureWorker();
    const constraints={width:{ideal:640},height:{ideal:480},facingMode:"user"};
    if(navigator.mediaDevices.getSupportedConstraints?.().resizeMode)constraints.resizeMode="none";
    stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: constraints });
    video.srcObject = stream; await video.play(); output = engine.unbind("calibration_required"); lastVideoTime = -1; lastReceivedResultAt = performance.now(); lastFrameSubmittedAt = -Infinity;
    const settings=stream.getVideoTracks?.()[0]?.getSettings?.()??{};
    cameraResearchSettings={...settings,video_width:video.videoWidth,video_height:video.videoHeight};
    measurementJournal.add("camera_event",{event:"camera_started",session:runtime.session,control_epoch:runtime.control_epoch},researchContext());
    const resize=settings.resizeMode==="none"?"浏览器未裁剪":settings.resizeMode==="crop-and-scale"?"浏览器裁剪 / 缩放":"浏览器裁剪信息未知";
    $("camera-format").textContent=`实际 ${settings.width??video.videoWidth} × ${settings.height??video.videoHeight}${Number.isFinite(settings.frameRate)?` · ${Math.round(settings.frameRate)} fps`:""} · ${resize}。网页和识别读取完整视频帧；内框仅用于保护。`;
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
  measurementJournal.add("camera_event",{event:"camera_stopped",reason,session:runtime.session,
    control_epoch:runtime.control_epoch,inference_in_flight:workerFrameBusy},researchContext());
  flushMeasurements();
  if (frameId) cancelAnimationFrame(frameId);
  stream?.getTracks().forEach((track) => track.stop()); stream = null; video.srcObject = null; context.clearRect(0, 0, canvas.width, canvas.height);
  gestureWorker?.terminate(); gestureWorker = null; workerReady = workerFrameBusy = false; workerGeneration += 1;
  lastDetections = []; lastRecognitionIssue = null; lastReceivedResultAt = -Infinity;
  clearHandCommands(); queueOutput(engine.unbind(reason), performance.now(), true); renderState(); $("camera-placeholder").hidden = false; $("camera-button").textContent = "启动摄像头"; $("camera-state").textContent = "未启动";
  $("camera-format").textContent="实际摄像头格式：未启动";
}
$("camera-button").addEventListener("click", () => {
  if(pendingOperation?.waitingForJournal) {notice("正在收齐本条日志，请等待完成后再关闭摄像头。");return;}
  if (stream) stopCamera(); else void startCamera();
});
$("pause-button").addEventListener("click", () => {
  if(pendingOperation || runtime.saving)return;
  if (engine.continuous && engine.manualPaused) {
    retryHandBoundary=null; // Continuing this attempt requires a later save boundary.
    queueOutput(engine.resume(), performance.now(), true);
  } else pause("user_pause");
  renderState();
});
$("view-select").addEventListener("change", () => { viewPreference = ["auto", "top", "oblique", "front"].includes($("view-select").value) ? $("view-select").value : "auto"; renderViews(); });
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
  button.addEventListener("click", async (event) => {
    if (!connected || pendingOperation) return;
    const command = button.dataset.command;
    const operationSession=runtime.session,operationEpoch=runtime.control_epoch;
    pause(engine.continuous ? "operation_pending" : "user_pause");
    const eventId = `operation-${Date.now()}-${crypto.randomUUID()}`;
    pendingOperation = { command, eventId, startedAt: performance.now() };
    let handBarrier;
    if(["record_save","record_failure"].includes(command)) {
      pendingOperation.waitingForJournal=true;
      $("record-feedback").textContent="正在收齐本条人手日志，完成后保存并重置；请勿关闭页面。";
      updateButtons();
      try {
        const sameBoundary=retryHandBoundary?.session===operationSession&&retryHandBoundary?.epoch===operationEpoch
          && (!runtime.episode_end || retryHandBoundary.barrier.end_capture_epoch_ms>=runtime.episode_end.wall_time_ms);
        if(!sameBoundary) {
          const endCaptureMs=Date.now();
          sealingHandBoundary=true;
          // Finish the already captured frame before freezing the sequence.
          // No new camera frame enters this brief boundary-drain phase.
          const deadline=performance.now()+5000;
          while(workerFrameBusy) {
            if(performance.now()>deadline)throw Error('recognition_boundary_timeout');
            await new Promise(resolve=>setTimeout(resolve,20));
          }
          retryHandBoundary={session:operationSession,epoch:operationEpoch,
            barrier:measurementJournal.freeze(endCaptureMs)};
        }
        sealingHandBoundary=false;
        let timer;
        try {
          handBarrier=await Promise.race([measurementJournal.flushBoundary(retryHandBoundary.barrier),
            new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error('research_boundary_timeout')),15000);})]);
        } finally {clearTimeout(timer);}
      } catch(error) {
        sealingHandBoundary=false;
        if(pendingOperation?.eventId===eventId)pendingOperation=null;
        $("record-feedback").textContent=`人手日志尚未收齐，本条未提交保存、场景未重置。请恢复连接后重试保存：${error.message}`;
        notice($("record-feedback").textContent);updateButtons();return;
      }
      if(!connected||pendingOperation?.eventId!==eventId||runtime.session!==operationSession||runtime.control_epoch!==operationEpoch) {
        if(pendingOperation?.eventId===eventId)pendingOperation=null;
        notice("等待研究记录期间会话或操作状态已变化，请核对当前轮次后重新操作。");updateButtons();return;
      }
    }
    if(["record_discard","record_stop"].includes(command)) {
      try {await measurementJournal.flush();}
      catch(error) {notice(`人手研究记录未完整写入；保留会话原件供排查：${error.message}`);}
      if(!pendingOperation || runtime.session!==operationSession || runtime.control_epoch!==operationEpoch)return;
    }
    pendingOperation.waitingForJournal=false;
    pendingOperation.startedAt=performance.now();
    events.push({ command, eventId, hand_journal_barrier:handBarrier??{client_id:recognitionClientId,...measurementJournal.status()}, sentAt: Date.now(), control_epoch: runtime.control_epoch, session: runtime.session, telemetry: engine.telemetry(false),
      ui_trigger: { version:"recording-operation-ui-v1", click_detail:event.detail??null,
        pointer_type:event.pointerType??null, is_trusted:event.isTrusted??null,
        recent_key:lastOperationKey&&performance.now()-lastOperationKey.at<1500?lastOperationKey.code:null,
        command, recording:runtime.recording===true, task_state:runtime.task_state } });
    $("record-feedback").textContent = "已提交操作，等待仿真执行确认。";
    engine.clearAnchors();
    updateButtons(); void flushPackets();
  });
}
window.addEventListener("keydown", (event) => {
  if (["Space", "Escape", "Enter"].includes(event.code) && event.target.tagName === "BUTTON")
    lastOperationKey = {code:event.code, at:performance.now()};
  if (["Space", "Escape"].includes(event.code) && !["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) {
    event.preventDefault(); pause("user_pause");
  }
});
document.addEventListener("visibilitychange", () => { if (document.hidden) pause("tab_hidden"); });
window.addEventListener("beforeunload",event=>{
  if(measurementJournal.queue.length||runtime.recording){event.preventDefault();event.returnValue="";flushMeasurements();}
});
window.addEventListener("pagehide", () => {
  flushMeasurements();
  clearTimeout(healthTimer); clearTimeout(previewTimer); clearTimeout(controlPumpTimer); latestPacket = null; events = [];
  if (connected) void fetch(`${api}/control`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(transportPacket({ command: "pause", reason: "page_closed", sentAt: Date.now(), control_epoch: runtime.control_epoch, session: runtime.session, telemetry: engine.telemetry(false) })), keepalive: true }).catch(() => {});
  if (frameId) cancelAnimationFrame(frameId); stream?.getTracks().forEach((track) => track.stop()); gestureWorker?.terminate(); Object.values(previews).filter(Boolean).forEach((url) => URL.revokeObjectURL(url));
});
setConnected(false); void checkHealth(); void pollPreviews();

$("recalibrate-aperture").addEventListener("click", () => {
  if(rateGestures())return;
  if (!engine.continuous || runtime.recording || runtime.saving) return;
  engine.apertureCalibration.reset(); queueOutput(engine.resetInput("aperture_close"), performance.now(), true); renderState();
});
