import { handMeasurement } from './hand-measurement.js';
import { SingleInteractionState, recordingButtons, continuousPacket, singleTelemetry,
  createSingleJournal, logSingleControlResult, operationExplicitlyRejected } from './single-interaction-state.js';

const $=id=>document.getElementById(id),api='/api/dual';
const video=$('webcam'),canvas=$('output-canvas'),ctx=canvas.getContext('2d');
const clientId=`single-${crypto.randomUUID()}`;
const engine=new SingleInteractionState();
let runtime={},connected=false,sequence=0,stream=null,worker=null,workerReady=false,workerBusy=false;
let cameraSettings=null,cameraGeneration=0,frameId=0,inFlightId=null,workerSentAt=0,lastVideoTime=-1;
let lastDiagnostic=null,lastNext=null,lastResultAt=-Infinity,packetFlight=false,latestInput=null;
let operation=null,lastSaved=null,operationError='',shutdown=false,healthFlight=false,previewFlight=false;
const eventQueue=[],views=['oblique','top','front'],previews={};
const labels={oblique:'斜视 · 模型输入 oblique',top:'正上方 · XY 对准',front:'辅助侧视 · 高度'};
const reasonText={binding_required:'请将唯一一只控制手张掌，完整留在内框内。',binding_in_progress:'保持张掌和手腕稳定，正在绑定。',binding_complete:'已绑定。先换为 I Love You 或握拳，解除绑定保护。',binding_gesture_released:'绑定完成，正在建立控制基准。',operator_pause:'手动暂停。可移动手回中；点击恢复后重新建立基准。',tracking_recovery:'正在确认连续可信的新画面，夹爪与移动目标保持。',gesture_mode_confirmation:'等待当前手势稳定。',gesture_motion_baseline:'已建立手腕基准，继续移动即可。',unknown_gesture:'未分配手势，保持不动；可以将手移回画面中央。',hand_lost:'控制手未可信识别，保持不动；完整入镜后恢复。',stale_recognition:'识别画面过期，保持不动。',out_of_order_measurement:'识别顺序异常，等待新画面。',camera_stalled:'摄像头识别中断，已暂停。',xy_motion:'I Love You：手腕控制 XY，夹爪保持闭合。',z_motion:'手腕上下控制 Z，夹爪保持闭合。',backend_session_changed:'仿真已重连，请重新张掌绑定。',scene_reset:'场景已更新，请重新张掌绑定。'};
const gestureText={ILoveYou:'I Love You',Open_Palm:'张掌',Closed_Fist:'握拳',None:'未知手势'};

function context() {return {camera:cameraSettings,display_mirrored:true,inference_mirrored:false,frame_crop:'none_in_application',
  hand_selection:'single_bound_raw_label',bound_raw_label:engine.boundLabel,arm_order:['left'],
  control_protocol:'gesture-rate-gripper-v3',gripper_trigger_policy:'push_locked_v1',
  filter_and_mapping:{...engine.parameters,motion_speed_scale:Number($('motion-speed').value)},
  config_hash:runtime.config_hash??null,model:'models/gesture_recognizer.task',delegate:'CPU'};}
async function jsonRequest(path,body,timeout=5000) {
  const response=await fetch(`${api}/${path}`,{method:body?'POST':'GET',cache:'no-store',headers:body?{'Content-Type':'application/json'}:{},body:body?JSON.stringify(body):undefined,signal:AbortSignal.timeout(timeout)});
  const result=await response.json();
  if(!response.ok) {const error=Error(result.error??result.reason??`HTTP ${response.status}`);error.status=response.status;error.response=result;throw error;}
  return result;
}
const journal=createSingleJournal({clientId,send:batch=>jsonRequest('measurements',batch),changed:s=>{
  $('measurement-log-state').textContent=`测量 ${s.completed} · 已写 ${s.written} · 待写 ${s.pending} · 丢弃 ${s.dropped}${s.error?' · '+s.error:''}`;
}});
const flushJournal=()=>journal.flush().catch(()=>{});

function telemetry(diagnostic=lastDiagnostic) {
  return singleTelemetry(engine,lastNext,diagnostic,$('motion-speed').value);
}
function stamped(packet) {return {...packet,protocol:'task3-command-v2',client_id:clientId,sequence:++sequence,eventId:packet.eventId??crypto.randomUUID(),session:runtime.session,control_epoch:runtime.control_epoch};}
function queueEvent(packet) {if(!connected)return;latestInput=null;eventQueue.push(packet);void pump();}
async function pump() {
  if(packetFlight||!connected||shutdown)return;
  packetFlight=true;
  try {
    while(eventQueue.length||latestInput) {
      const input=eventQueue.shift()??latestInput;if(input===latestInput)latestInput=null;
      if(input.session&&input.session!==runtime.session||input.control_epoch!==undefined&&input.control_epoch!==runtime.control_epoch)continue;
      if(input.command==='continuous_input'&&Date.now()-input.sentAt>engine.parameters.maximum_age_ms)continue;
      // Manual operations are created now; their separately frozen journal
      // boundary is unchanged. Camera actions retain the original capture time.
      // Two absent event IDs must not classify camera input as a button event.
      const manual=input.command==='pause'||Boolean(operation?.eventId&&operation.eventId===input.eventId);
      const packet=stamped(manual?{...input,sentAt:Date.now()}:input);
      journal.add('dispatch',{event_id:packet.eventId,measurement_id:packet.measurement_id??null,command:packet.command,packet},context());
      let response;
      try {response=await jsonRequest('control',packet,3000);}
      catch(error) {
        logSingleControlResult(journal,{event_id:packet.eventId,error:String(error)},context());
        if(operation?.eventId===packet.eventId) {
          if(operationExplicitlyRejected(error)) {
            operationError=`操作未受理：${error.response?.reason??error.message}。核对状态后可重试。`;operation=null;
          } else {operation.uncertain=true;operationError='请求结果暂不确定，正在核对后端。请勿重复保存。';}
        }
        else {engine.interrupt('tracking_recovery');latestInput=null;}
        break;
      }
      logSingleControlResult(journal,{event_id:packet.eventId,response},context());
      if(operation?.eventId===packet.eventId)operation.queued=true;
      if(response.ok===false||response.accepted===false) {
        operationError=response.reason??response.error??'请求未接受';
        if(operation?.eventId===packet.eventId)operation=null;
      }
    }
  } finally {packetFlight=false;render();}
}
function pause(reason='operator_pause') {
  engine.setPaused(true);engine.reason=reason;latestInput=null;
  queueEvent({command:'pause',reason,sentAt:Date.now(),telemetry:telemetry()});render();
}
function clearContext(reason) {
  engine.reset();engine.manualPaused=true;engine.reason=reason;lastDiagnostic=null;lastNext=null;
  latestInput=null;eventQueue.length=0;lastSaved=null;
  for(const view of views) {if(previews[view])URL.revokeObjectURL(previews[view].url);delete previews[view];}
}
function resolveOperation() {
  if(!operation?.eventId)return;
  const results=[...(runtime.operation_results??[]),runtime.last_operation_result].filter(Boolean);
  const result=results.find(r=>r.eventId===operation.eventId);
  if(result) {
    const name=operation.command;
    if(result.accepted===true||result.ok===true) {
      if(['record_save','record_failure'].includes(name))lastSaved={kind:name==='record_save'?'成功':'失败',at:Date.now()};
      operationError=name==='record_start'?'已确认开始录制。':name==='record_discard'?'本轮已放弃。':name==='reset'?'场景已重置。':lastSaved?`已保存${lastSaved.kind}示范。重置场景后可开始下一轮。`:'操作已完成。';
    } else operationError=`操作未执行：${result.reason??result.error??'后端拒绝'}`;
    operation=null;return;
  }
  // Confirm state changes independently if the result history has rotated.
  if(operation.command==='record_start'&&runtime.recording) {operation=null;operationError='已确认开始录制。';return;}
  if(operation.command==='reset'&&runtime.control_epoch!==operation.epoch) {operation=null;operationError='场景已重置。';return;}
  if(operation.command==='record_save'&&runtime.saved_success>operation.successBefore) {lastSaved={kind:'成功'};operation=null;operationError='已确认保存成功。重置场景后开始下一轮。';return;}
  if(operation.command==='record_failure'&&runtime.saved_failure>operation.failureBefore) {lastSaved={kind:'失败'};operation=null;operationError='已确认保存失败示范。';}
}
async function pollHealth() {
  if(healthFlight||shutdown)return;healthFlight=true;
  const requestEpoch=Date.now();
  try {
    const next=await jsonRequest('health');
    if(next.config?.task_binding?.task_id!=='push_t'&&next.task_id!=='push_t')throw Error('此页面需要启动 push_t 任务');
    if(runtime.session&&(runtime.session!==next.session||runtime.control_epoch!==next.control_epoch))clearContext(runtime.session!==next.session?'backend_session_changed':'scene_reset');
    runtime={...next,clock_pair:{browser_request_epoch_ms:requestEpoch,browser_receive_epoch_ms:Date.now(),server_epoch_ms:next.server_epoch_ms??next.wall_time_ms??null,server_monotonic_ns:next.server_monotonic_ns??null}};
    engine.parameters={...engine.parameters,...(runtime.action_contract?.gesture_parameters??runtime.config?.gesture_control??{})};
    connected=true;resolveOperation();
  } catch(error) {connected=false;latestInput=null;engine.interrupt('backend_session_changed');$('notice').textContent=String(error.message);}
  finally {healthFlight=false;render();}
}
function render() {
  $('connection').textContent=connected?'仿真已连接':'仿真未连接';$('connection').dataset.state=connected?'active':'error';
  $('task-state').textContent={PREGRASP:'准备推动',PUSHING:'推动中',DONE:'成功完成',FAIL:'失败'}[runtime.task_state]??runtime.task_state??'—';
  $('recording-state').textContent=runtime.saving?'正在保存':runtime.recording?`录制中 · ${runtime.frames??0} 帧`:lastSaved?`本轮已保存${lastSaved.kind}`:'练习 · 未录制';
  $('scenario-name').textContent=connected?`第 ${(runtime.scene_index??0)+1} 组 · 种子 ${runtime.seed??runtime.session_seed??'—'}`:'';
  $('dataset-counts').textContent=`成功 ${runtime.saved_success??0} / 失败 ${runtime.saved_failure??0}`;
  const flags=recordingButtons(runtime,connected,Boolean(operation));
  document.querySelectorAll('[data-command]').forEach(b=>{b.disabled=!flags[b.dataset.command];});
  $('pause-button').disabled=!connected||runtime.saving;
  $('pause-button').textContent=engine.manualPaused?'恢复控制':'暂停 / 重定位';
  $('unbind-hands').disabled=!engine.boundLabel||runtime.recording||runtime.saving;
  $('hand-select').disabled=Boolean(engine.boundLabel)||runtime.recording||runtime.saving;
  $('hand-identity').textContent=engine.boundLabel?`已绑定：MediaPipe ${engine.boundLabel==='left'?'左':'右'}手 → 单臂`:'控制手尚未绑定';
  $('binding-progress').value=engine.bindingProgress;
  $('hand-gesture').textContent=lastNext?.hand?`${gestureText[lastNext.hand.gesture]??lastNext.hand.gesture} · ${Math.round(lastNext.hand.gesture_score*100)}%`:'未可信识别控制手';
  $('hand-interaction').textContent=reasonText[engine.reason]??engine.reason;
  const terminal=['DONE','FAIL'].includes(runtime.task_state);
  const titles={XY_HOLD:'XY 平移',Z_OPEN:'Z 升降',Z_CLOSE:'Z 升降',HOLD:'保持 / 等待'};
  $('control-status-title').textContent=!connected?'等待连接':runtime.saving?'正在保存':terminal?(runtime.task_state==='DONE'?'推动完成':'任务失败'):engine.manualPaused?'手动暂停':runtime.paused?'后端保持 / 等待':titles[lastNext?.mode??'HOLD'];
  $('control-status').dataset.state=terminal?'complete':!connected?'waiting':engine.manualPaused?'paused':lastNext?.mode!=='HOLD'&&lastNext?.mode?'active':'waiting';
  if(connected)$('notice').textContent=terminal?'任务结束，请保存或放弃本轮；未录制练习请重置场景。':runtime.paused&&!engine.manualPaused&&runtime.pause_reason?(reasonText[runtime.pause_reason]??`后端暂停：${runtime.pause_reason}`):(reasonText[engine.reason]??engine.reason);
  $('control-recording-note').textContent=runtime.recording?'正在录制，完成后仍需手动点击保存。':lastSaved?'本轮数据已写入；并非退回“未录制练习”。':terminal?'这一轮未录制，无法事后保存为示范。':'';
  if(operation)$('record-feedback').textContent=runtime.saving?'正在写入视频和日志，请稍候。':operation.uncertain?operationError:'正在等待后端确认操作…';
  else if(operationError)$('record-feedback').textContent=operationError;
  $('backend-detail').textContent=`后端状态：${runtime.pause_reason??runtime.backend_input?.reason??'正常'}${runtime.preview_error?' · '+runtime.preview_error:''}`;
}

function renderViews() {
  const main=$('view-select').value,aux=views.filter(v=>v!==main);$('main-view-title').textContent=labels[main];
  [['main',main],['auxiliary',aux[0],'one'],['auxiliary',aux[1],'two']].forEach(([kind,view,suffix])=>{
    const id=kind==='main'?'main':`auxiliary`,end=suffix?`-${suffix}`:'';
    const img=$(`${id}-preview${end}`),placeholder=$(`${id}-placeholder${end}`),valid=previews[view]&&performance.now()-previews[view].time<1500;
    img.hidden=!valid;placeholder.hidden=Boolean(valid);if(valid)img.src=previews[view].url;
    if(suffix)$(`auxiliary-view-${suffix}`).textContent=labels[view];
  });
}
async function pollPreviews() {
  if(previewFlight||!connected||shutdown)return;previewFlight=true;const epoch=runtime.control_epoch,session=runtime.session;
  try {
    await Promise.all(views.map(async view=>{
      try {
        const r=await fetch(`${api}/${view}-preview`,{cache:'no-store',signal:AbortSignal.timeout(1500)});
        if(!r.ok||r.status===204)throw Error('No image');
        const capturedEpoch=r.headers.get('X-Preview-Epoch');
        if(capturedEpoch!==null&&Number(capturedEpoch)!==epoch)throw Error('Old image');
        const blob=await r.blob();if(epoch!==runtime.control_epoch||session!==runtime.session)return;
        const old=previews[view];previews[view]={url:URL.createObjectURL(blob),time:performance.now()-Number(r.headers.get('X-Preview-Age-Ms')??0)};
        if(old)URL.revokeObjectURL(old.url);
      } catch { /* The control path is independent of preview rendering. */ }
    }));
    $('preview-state').textContent=views.every(v=>previews[v]&&performance.now()-previews[v].time<1500)?'三路仿真画面':'等待画面更新';renderViews();
  } finally {previewFlight=false;}
}

function draw(result) {
  canvas.width=video.videoWidth;canvas.height=video.videoHeight;ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.strokeStyle='#8cf0b4';ctx.lineWidth=2;ctx.strokeRect(canvas.width*.06,canvas.height*.06,canvas.width*.88,canvas.height*.88);
  for(const hand of result.landmarks??[]) {
    ctx.strokeStyle='#68c6ff';ctx.fillStyle='#eaf8ff';
    for(const chain of [[0,1,2,3,4],[0,5,6,7,8],[5,9,10,11,12],[9,13,14,15,16],[13,17,18,19,20],[17,0]]) {
      ctx.beginPath();chain.forEach((n,i)=>{const p=hand[n];i?ctx.lineTo(p.x*canvas.width,p.y*canvas.height):ctx.moveTo(p.x*canvas.width,p.y*canvas.height);});ctx.stroke();
    }
    for(const p of hand){ctx.beginPath();ctx.arc(p.x*canvas.width,p.y*canvas.height,2.5,0,Math.PI*2);ctx.fill();}
  }
}
function recognition(data) {
  const now=performance.now(),dropReason=!stream?'camera_stopped':data.session!==runtime.session?'backend_session_changed':data.control_epoch!==runtime.control_epoch?'scene_reset':data.frameId!==inFlightId?'unexpected_frame_id':now-data.capturedAt>engine.parameters.maximum_age_ms?'stale_recognition':null;
  const measurement=handMeasurement(data.result,{id:`${clientId}:${data.frameId}`,session:data.session,epoch:data.control_epoch,capturedAt:data.capturedAt,capturedEpochMs:data.capturedEpochMs,
    inferenceMs:data.inferenceMs,receivedAt:now,recognitionCompletedEpochMs:data.recognitionCompletedEpochMs,workerCompletedAt:data.workerCompletedAt,workerTimeOriginMs:data.workerTimeOriginMs,
    width:data.width,height:data.height,swapLabels:false,dropReason});
  const next=engine.update(measurement,Date.now(),Number($('motion-speed').value));
  const diagnostic={frame_id:measurement.measurement_id,source_session:data.session,source_control_epoch:data.control_epoch,captured_at_ms:data.capturedEpochMs,
    received_at_ms:Date.now(),inference_ms:data.inferenceMs,result_age_ms:now-data.capturedAt,accepted:dropReason===null,drop_reason:dropReason,measurement};
  lastDiagnostic=diagnostic;lastNext=next;lastResultAt=now;
  journal.add('measurement',{measurement,clock_pair:runtime.clock_pair??null,control:{mode:next.mode,pause_reason:next.reason,
    calibrated:next.calibrated,bound_raw_label:engine.boundLabel,motion_anchor:engine.anchor,proposed_delta_task_m:{left:next.delta}}},context());
  if(stream)draw(data.result);
  $('camera-state').textContent=`后台识别 ${Math.round(data.inferenceMs)} ms · 结果年龄 ${Math.round(now-data.capturedAt)} ms`;
  if(connected&&!operation&&!runtime.saving&&!['DONE','FAIL'].includes(runtime.task_state)) {
    let packet;
    if(next.canSendControl)packet=continuousPacket(next,telemetry(diagnostic));
    else packet={command:next.hand?.valid?'telemetry':'pause',reason:next.reason,sentAt:Date.now(),telemetry:telemetry(diagnostic)};
    latestInput={...packet,session:runtime.session,control_epoch:runtime.control_epoch};void pump();
  }
  render();
}
async function cameraFrame() {
  if(shutdown)return;
  if(stream&&workerReady&&!workerBusy&&video.readyState>=2&&video.currentTime!==lastVideoTime&&connected&&!operation) {
    const generation=cameraGeneration;lastVideoTime=video.currentTime;workerBusy=true;
    const capture={type:'frame',frameId:++frameId,capturedAt:performance.now(),capturedEpochMs:Date.now(),session:runtime.session,control_epoch:runtime.control_epoch};
    inFlightId=capture.frameId;workerSentAt=performance.now();
    try {
      const bitmap=await createImageBitmap(video);
      if(generation!==cameraGeneration||!worker){bitmap.close();workerBusy=false;requestAnimationFrame(cameraFrame);return;}
      worker.postMessage({...capture,bitmap},[bitmap]);
    } catch(error) {workerBusy=false;pause('camera_stalled');$('camera-state').textContent=String(error);}
  }
  if(stream&&workerReady&&(workerBusy&&performance.now()-workerSentAt>1000||performance.now()-lastResultAt>1000)&&!engine.manualPaused)pause('camera_stalled');
  requestAnimationFrame(cameraFrame);
}
async function startCamera() {
  $('camera-button').disabled=true;
  try {
    stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480},frameRate:{ideal:30}},audio:false});
    cameraGeneration++;video.srcObject=stream;await video.play();cameraSettings=stream.getVideoTracks()[0].getSettings();
    $('camera-format').textContent=`实际画面 ${video.videoWidth}×${video.videoHeight} · ${Math.round(cameraSettings.frameRate??0)} fps；应用未裁剪。`;
    $('camera-placeholder').hidden=true;lastVideoTime=-1;lastResultAt=performance.now();
    worker=new Worker('./dual-gesture-worker.js');
    worker.onmessage=({data})=>{
      if(data.type==='ready'){workerReady=true;lastResultAt=performance.now();return;}
      if(data.type==='result') {recognition(data);if(data.frameId===inFlightId)workerBusy=false;return;}
      if(data.type==='error'){workerBusy=false;pause('camera_stalled');$('camera-state').textContent=data.message;}
    };
    worker.onerror=()=>{workerBusy=false;pause('camera_stalled');$('camera-state').textContent='识别进程异常，请停止后重新启动摄像头。';};
    worker.postMessage({type:'init',wasmRoot:new URL('./node_modules/@mediapipe/tasks-vision/wasm/',location.href).href,modelUrl:new URL('./models/gesture_recognizer.task',location.href).href,delegate:'CPU'});
    stream.getVideoTracks()[0].addEventListener('ended',()=>{if(stream)stopCamera();});
    journal.add('camera_event',{event:'camera_started',session:runtime.session,control_epoch:runtime.control_epoch},context());
    $('camera-button').textContent='停止摄像头';
  } catch(error) {stopCamera();$('camera-state').textContent=`摄像头启动失败：${error.message}`;}
  finally {$('camera-button').disabled=false;}
}
function stopCamera() {
  cameraGeneration++;const old=stream;stream=null;old?.getTracks().forEach(track=>track.stop());video.srcObject=null;
  worker?.terminate();worker=null;workerReady=false;workerBusy=false;
  engine.reset();pause('camera_stalled');ctx.clearRect(0,0,canvas.width,canvas.height);
  $('camera-placeholder').hidden=false;$('camera-button').textContent='启动摄像头';
  journal.add('camera_event',{event:'camera_stopped',session:runtime.session,control_epoch:runtime.control_epoch},context());void flushJournal();
}

async function operate(command) {
  if(!recordingButtons(runtime,connected,Boolean(operation))[command])return;
  const pending={command,eventId:crypto.randomUUID(),epoch:runtime.control_epoch,session:runtime.session,successBefore:runtime.saved_success??0,failureBefore:runtime.saved_failure??0};
  operation=pending;operationError='';latestInput=null;eventQueue.length=0;engine.resetMotion();render();
  try {
    // Freeze capture before flushing, so continuous webcam input cannot extend
    // the save boundary forever. Already dispatched worker work must finish.
    if(command!=='record_start') {
      engine.manualPaused=true;
      if(workerBusy) {const deadline=performance.now()+2000;while(workerBusy&&performance.now()<deadline)await new Promise(r=>setTimeout(r,20));if(workerBusy)throw Error('识别未结束，请停止摄像头后重试保存。');}
    }
    const boundary=journal.freeze(Date.now());
    const barrier=await journal.flushBoundary(boundary);
    if(runtime.session!==pending.session||runtime.control_epoch!==pending.epoch)throw Error('场景已改变，请确认当前状态后再操作。');
    queueEvent({command,eventId:pending.eventId,hand_journal_barrier:barrier,sentAt:Date.now(),telemetry:telemetry(),source:'single_arm_button',session:pending.session,control_epoch:pending.epoch});
  } catch(error) {operationError=String(error.message);operation=null;render();}
}
document.querySelectorAll('[data-command]').forEach(button=>button.addEventListener('click',()=>void operate(button.dataset.command)));
$('camera-button').addEventListener('click',()=>stream?stopCamera():void startCamera());
$('pause-button').addEventListener('click',()=>{
  if(engine.manualPaused){engine.setPaused(false);lastNext=null;render();}else pause();
});
$('unbind-hands').addEventListener('click',()=>{engine.reset();pause('binding_required');});
$('hand-select').addEventListener('change',()=>{engine.preference=$('hand-select').value;engine.reset();pause('binding_required');});
$('motion-speed').addEventListener('change',()=>{engine.resetMotion();});
$('view-select').addEventListener('change',renderViews);
document.addEventListener('keydown',event=>{
  if(event.repeat||['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName))return;
  if(event.code==='Space'||event.key==='Escape'){event.preventDefault();pause();}
});
window.addEventListener('beforeunload',event=>{if(runtime.recording||journal.queue.length){event.preventDefault();event.returnValue='';}});
window.addEventListener('pagehide',()=>{
  if(connected)void fetch(`${api}/control`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(stamped({command:'pause',reason:'page_closed',sentAt:Date.now()})),keepalive:true}).catch(()=>{});
  shutdown=true;stream?.getTracks().forEach(t=>t.stop());worker?.terminate();
});
setInterval(()=>void pollHealth(),350);setInterval(()=>void pollPreviews(),180);setInterval(()=>void flushJournal(),200);
void pollHealth();requestAnimationFrame(cameraFrame);render();
