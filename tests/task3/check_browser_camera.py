"""Exercise the real camera UI with Chrome's fake webcam and an isolated backend.

No physical camera, robot movement or recording is used. A single rejected
control request must not make a healthy simulation appear disconnected.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import aiohttp
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Backend:
    def __init__(self, render_scene=False, fail_after_close=False):
        self.config = json.loads((ROOT / "config/task3_dual_arm.json").read_text(encoding="utf-8"))
        self.controls = 0
        self.rejections = 0
        self.reject_next = False
        self.available = True
        self.timestamps = []
        self.commands = []
        self.payloads = []
        self.gripper_latch = "OPEN"
        self.task_state = "PREGRASP"
        self.control_mode = "PAUSE"
        self.paused = True
        self.fail_after_close = fail_after_close
        buffer = io.BytesIO()
        Image.new("RGB", (480, 360), "#477b66").save(buffer, "JPEG")
        self.jpeg = buffer.getvalue()
        self.scene_images = {}
        if render_scene:
            from simulation.mujoco.dual_arm.dual_arm_task_controller import DualArmTaskController
            from simulation.mujoco.dual_arm.dual_arm_render_workers import MultiCameraRenderer
            controller = DualArmTaskController(config=self.config)
            renderer = MultiCameraRenderer(controller.model)
            try:
                images = renderer.render(controller.data, operator_view=controller.config)
                for name, pixels in images.items():
                    buffer = io.BytesIO()
                    Image.fromarray(pixels).save(buffer, "JPEG")
                    self.scene_images[name] = buffer.getvalue()
            finally:
                renderer.close()

    def health(self):
        return {"ok": True, "service": "mujoco-task3-recorder", "task_mode": "dual_arm",
                "session": "isolated-browser-check", "control_epoch": 0, "config": self.config,
                "gripper_latch": self.gripper_latch, "task_state": self.task_state, "recording": False, "practice_mode": True,
                "control_mode": self.control_mode, "paused": self.paused,
                "saving": False, "frames": 0, "saved_success": 0, "saved_failure": 0,
                "release_ready": True,
                "capture_status": {"state": "not_recording", "training_eligible": False, "last_frame_recorded": False, "frames": 0},
                "failure_reason": "grasp_failure" if self.task_state == "FAIL" else None}

    def handler(self):
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def send(self, status, data, content_type="application/json"):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if not backend.available:
                    self.send(503, b'{"ok":false}')
                elif self.path == "/health":
                    self.send(200, json.dumps(backend.health()).encode())
                elif self.path.endswith("-preview"):
                    name = self.path.rsplit("/", 1)[-1].removesuffix("-preview")
                    self.send(200, backend.scene_images.get(name, backend.jpeg), "image/jpeg")
                else:
                    self.send(404, b"{}")

            def do_POST(self):
                if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                    blocks = []
                    while True:
                        size = int(self.rfile.readline().split(b";", 1)[0], 16)
                        if size == 0:
                            self.rfile.readline()
                            break
                        blocks.append(self.rfile.read(size))
                        self.rfile.read(2)
                    data = b"".join(blocks)
                else:
                    data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                payload = json.loads(data)
                backend.controls += 1
                backend.commands.append(payload["command"])
                backend.payloads.append(payload)
                backend.timestamps.append(time.monotonic())
                if backend.reject_next:
                    backend.reject_next = False
                    backend.rejections += 1
                    self.send(429, b'{"ok":false,"error":"Test control backpressure"}')
                else:
                    if payload["command"] in {"pause", "telemetry", "dual_motion"}:
                        mode = str(payload.get("mode", payload.get("telemetry", {}).get("control_mode", "PAUSE"))).upper()
                        self_pause = payload["command"] == "pause" or mode not in {"XY", "Z"} or backend.task_state == "FAIL"
                        backend.control_mode = "PAUSE" if self_pause else mode
                        backend.paused = self_pause
                    if payload["command"] == "dual_gripper":
                        backend.gripper_latch = "CLOSED" if payload["action"] == "close" else "OPEN"
                        if payload["action"] == "close" and backend.fail_after_close:
                            backend.task_state = "FAIL"
                    self.send(200, json.dumps({"ok": True, "queued": True, "eventId": payload.get("eventId")}).encode())

        return Handler


class CDP:
    def __init__(self, ws, app_source=None):
        self.ws, self.app_source = ws, app_source
        self.index = 0
        self.pending = {}
        self.errors = []
        self.reader = asyncio.create_task(self.read())

    async def read(self):
        async for item in self.ws:
            if item.type != aiohttp.WSMsgType.TEXT:
                continue
            value = json.loads(item.data)
            if "id" in value:
                future = self.pending.pop(value["id"], None)
                if future and not future.done():
                    future.set_result(value)
            elif value.get("method") == "Fetch.requestPaused" and self.app_source:
                asyncio.create_task(self.call("Fetch.fulfillRequest", {
                    "requestId": value["params"]["requestId"], "responseCode": 200,
                    "responseHeaders": [{"name": "Content-Type", "value": "text/javascript"}],
                    "body": base64.b64encode(self.app_source.encode()).decode()}))
            elif value.get("method") == "Runtime.exceptionThrown":
                self.errors.append(value["params"]["exceptionDetails"])

    async def call(self, method, params=None):
        self.index += 1
        index = self.index
        future = asyncio.get_running_loop().create_future()
        self.pending[index] = future
        await self.ws.send_json({"id": index, "method": method, "params": params or {}})
        result = await asyncio.wait_for(future, 30)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result.get("result", {})

    async def evaluate(self, expression):
        result = await self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True})
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return result.get("result", {}).get("value")


async def exercise(args, backend, port, cdp_port):
    async with aiohttp.ClientSession() as session:
        for _ in range(40):
            try:
                async with session.get(f"http://127.0.0.1:{port}/dual_arm.html") as page:
                    assert page.status == 200
                    assert "text/html" in page.headers.get("Content-Type", "")
                    assert "camera-button" in await page.text()
                break
            except aiohttp.ClientError:
                await asyncio.sleep(0.15)
        deadline = time.monotonic() + 20
        while True:
            try:
                async with session.get(f"http://127.0.0.1:{cdp_port}/json/list") as response:
                    targets = await response.json()
                pages = [target for target in targets if target.get("type") == "page"]
                if pages:
                    break
            except (aiohttp.ClientError, OSError):
                if time.monotonic() > deadline:
                    raise TimeoutError("Isolated Chrome did not start")
                await asyncio.sleep(0.15)
        app_source = args.app_source.read_text(encoding="utf-8") if args.app_source else None
        # Chrome publishes CDP before its initial about:blank navigation has
        # settled; avoid racing that navigation with the test page.
        await asyncio.sleep(2)
        async with session.ws_connect(pages[0]["webSocketDebuggerUrl"]) as ws:
            cdp = CDP(ws, app_source)
            await cdp.call("Runtime.enable")
            await cdp.call("Page.enable")
            await cdp.call("Emulation.setDeviceMetricsOverride", {"width": 1365, "height": 1100, "deviceScaleFactor": 1, "mobile": False})
            if app_source:
                await cdp.call("Fetch.enable", {"patterns": [{"urlPattern": "*dual-arm-app.js", "requestStage": "Request"}]})
            await cdp.call("Page.addScriptToEvaluateOnNewDocument", {"source": """
                window.__cameraCheck = {connection: [], longTasks: []};
                new PerformanceObserver(items => {
                  for (const entry of items.getEntries()) window.__cameraCheck.longTasks.push(entry.duration);
                }).observe({entryTypes: ['longtask']});
                window.addEventListener('DOMContentLoaded', () => {
                  const node = document.getElementById('connection');
                  const save = () => window.__cameraCheck.connection.push({at:performance.now(),text:node.textContent});
                  new MutationObserver(save).observe(node,{childList:true,characterData:true,subtree:true}); save();
                  const control = document.getElementById('control-status-title');
                  window.__cameraCheck.controlStates = [];
                  if (control) new MutationObserver(() => {
                    const title = control.textContent;
                    if (window.__cameraCheck.controlStates.at(-1) !== title) window.__cameraCheck.controlStates.push(title);
                  }).observe(control,{childList:true,characterData:true,subtree:true});
                });
            """})
            if args.exercise_calibration:
                # The real Worker and fake webcam still run. Replace only the
                # classifier labels with a deterministic two-hand sequence;
                # this verifies page wiring, not visual recognition accuracy.
                await cdp.call("Page.addScriptToEvaluateOnNewDocument", {"source": """
                    const NativeWorker = window.Worker;
                    window.Worker = class extends NativeWorker {
                      set onmessage(handler) {
                        let firstCapture = null;
                        super.onmessage = event => {
                          if (event.data.type === 'result') {
                            firstCapture ??= event.data.capturedAt;
                            const elapsed = event.data.capturedAt - firstCapture;
                            const phase = elapsed < 3500 ? 'palms' : elapsed < 4000 ? 'gesture_changed' : elapsed < 4500 ? 'partial_low_confidence' : elapsed < 5000 ? 'duplicate_identity' : 'hands_lowered';
                            const gesture = phase === 'palms' ? 'Open_Palm' : phase === 'gesture_changed' ? 'None' : 'Closed_Fist';
                            const flicker = phase === 'palms' && (window.__cameraCheck.scriptedFrames = (window.__cameraCheck.scriptedFrames ?? 0) + 1) % 7 === 0;
                            window.__cameraCheck.classifierFlickers = (window.__cameraCheck.classifierFlickers ?? 0) + Number(flicker);
                            window.__cameraCheck.scriptedPhases ??= [];
                            if (!window.__cameraCheck.scriptedPhases.includes(phase)) window.__cameraCheck.scriptedPhases.push(phase);
                            const landmarks = x => Array.from({length:21}, (_,i) => ({x:x+(i%4)*0.002,y:0.5+Math.floor(i/4)*0.002,z:0}));
                            event.data.result = {
                              landmarks:[landmarks(0.2),landmarks(0.8)],
                              handedness:[[{categoryName:'Left',score:0.99}],[{categoryName:'Right',score:0.99}]],
                              gestures:[[{categoryName:flicker?'None':gesture,score:flicker?0.4:0.65}],[{categoryName:flicker?'None':gesture,score:flicker?0.4:0.65}]]
                            };
                            if (phase === 'partial_low_confidence') {
                              event.data.result.landmarks = [landmarks(0.2)];
                              event.data.result.handedness = [[{categoryName:'Left',score:0.6}]];
                              event.data.result.gestures = [[{categoryName:'Closed_Fist',score:0.98}]];
                            } else if (phase === 'duplicate_identity') event.data.result.handedness[1][0].categoryName = 'Left';
                            else if (phase === 'hands_lowered') event.data.result = {landmarks:[],handedness:[],gestures:[]};
                          }
                          handler(event);
                        };
                      }
                      get onmessage() { return super.onmessage; }
                    };
                """})
            if args.exercise_controls:
                await cdp.call("Page.addScriptToEvaluateOnNewDocument", {"source": """
                    const NativeWorker = window.Worker;
                    window.Worker = class extends NativeWorker {
                      set onmessage(handler) {
                        let firstCapture = null;
                        super.onmessage = event => {
                          if (event.data.type === 'result') {
                            firstCapture ??= event.data.capturedAt;
                            const t = event.data.capturedAt - firstCapture;
                            window.__cameraCheck.sequenceStartedWall ??= Date.now() - t;
                            const phase = t < 2500 ? 'palms' : t < 2900 ? 'neutral' : t < 3500 ? 'moving' : t < 4450 ? 'held_position' : t < 4900 ? 'unassigned_return' : t < 5250 ? 'resumed' : t < 6450 ? 'fists' : t < 7900 ? 'release_palms' : t < 8300 ? 'unassigned_pause' : 'hands_lowered';
                            window.__cameraCheck.scriptedPhases ??= [];
                            if (!window.__cameraCheck.scriptedPhases.includes(phase)) window.__cameraCheck.scriptedPhases.push(phase);
                            const gesture = phase === 'palms' || phase === 'release_palms' ? 'Open_Palm' : phase === 'fists' ? 'Closed_Fist' : phase.startsWith('unassigned') ? 'None' : 'ILoveYou';
                            const score = phase === 'release_palms' ? 0.57 : phase === 'fists' ? 0.82 : 0.9;
                            const offset = phase === 'moving' ? 0.03 * (t - 2900) / 600 : phase === 'held_position' ? 0.03 : 0;
                            const landmarks = x => Array.from({length:21}, (_,i) => ({x:x+offset+(i%4)*0.002,y:0.5-offset+Math.floor(i/4)*0.002,z:0}));
                            event.data.result = {landmarks:[landmarks(0.2),landmarks(0.8)],handedness:[[{categoryName:'Left',score:0.99}],[{categoryName:'Right',score:0.99}]],gestures:[[{categoryName:gesture,score}],[{categoryName:gesture,score}]]};
                            if (phase === 'hands_lowered') event.data.result = {landmarks:[],handedness:[],gestures:[]};
                            window.__cameraCheck.currentPhase = phase;
                          }
                          handler(event);
                        };
                      }
                      get onmessage() { return super.onmessage; }
                    };
                """})
            navigation = await cdp.call("Page.navigate", {"url": f"http://127.0.0.1:{port}/dual_arm.html"})
            if navigation.get("errorText"):
                await asyncio.sleep(0.5)
                navigation = await cdp.call("Page.navigate", {"url": f"http://127.0.0.1:{port}/dual_arm.html"})
                if navigation.get("errorText"):
                    detail = await cdp.evaluate("({url:location.href,body:document.body?.innerText})")
                    raise RuntimeError({"navigation":navigation,"page":detail})
            deadline = time.monotonic() + 20
            while not await cdp.evaluate("Boolean(document.getElementById('camera-button'))"):
                if time.monotonic() > deadline:
                    detail = await cdp.evaluate("({url:location.href,body:document.body?.innerText})")
                    raise TimeoutError(f"Task3 page did not load: {detail}")
                await asyncio.sleep(0.15)
            await asyncio.sleep(0.5)
            await cdp.evaluate("document.getElementById('camera-button').click()")
            deadline = time.monotonic() + 35
            while True:
                state = await cdp.evaluate("({camera:document.getElementById('camera-state').textContent,notice:document.getElementById('notice').textContent})")
                if (state["camera"] == "双手识别中" or state["camera"].startswith("后台识别")) and backend.controls >= 3:
                    break
                if "失败" in state["camera"] or time.monotonic() > deadline:
                    raise RuntimeError(f"Fake camera could not start: {state}; JS errors={cdp.errors}")
                await asyncio.sleep(0.25)
            await cdp.evaluate("window.__cameraCheck.connection=[]; window.__cameraCheck.longTasks=[]")
            if not args.exercise_calibration and not args.exercise_controls:
                backend.reject_next = True
            early_progress = None
            if args.exercise_calibration:
                await asyncio.sleep(min(1.5, args.seconds))
                early_progress = await cdp.evaluate("({value:document.getElementById('calibration-progress').value,text:document.getElementById('calibration-state').textContent,detail:document.getElementById('calibration-detail').textContent,left:document.getElementById('left-gesture').textContent,right:document.getElementById('right-gesture').textContent})")
                clip = await cdp.evaluate("(()=>{const b=document.querySelector('.camera-panel').getBoundingClientRect();return {x:b.x+scrollX,y:b.y+scrollY,width:b.width,height:b.height,scale:1};})()")
                shot = await cdp.call("Page.captureScreenshot", {"captureBeyondViewport": True, "clip": clip})
                (args.output / "calibration-progress.png").write_bytes(base64.b64decode(shot["data"]))
                await asyncio.sleep(max(0, args.seconds - 1.5))
            else:
                await asyncio.sleep(args.seconds)
            result = await cdp.evaluate("({connection:window.__cameraCheck.connection,longTasks:window.__cameraCheck.longTasks,classifier_phases:window.__cameraCheck.scriptedPhases??[],classifier_flickers:window.__cameraCheck.classifierFlickers??0,camera:document.getElementById('camera-state').textContent,calibration:document.getElementById('calibration-state').textContent,calibration_detail:document.getElementById('calibration-detail').textContent,previews:['main-preview','side-preview','front-preview'].map(id=>({id,visible:!document.getElementById(id).hidden,loaded:document.getElementById(id).naturalWidth>0}))})")
            result.update(control_requests=backend.controls, injected_control_rejections=backend.rejections,
                          physical_camera_access=False, javascript_errors=cdp.errors)
            result["connected_through_rejected_control"] = not any("未连接" in row["text"] for row in result["connection"])
            result["valid"] = result["connected_through_rejected_control"] and all(row["visible"] and row["loaded"] for row in result["previews"]) and not cdp.errors and backend.rejections == (0 if args.exercise_calibration or args.exercise_controls else 1)
            if args.exercise_calibration:
                result["hands_free_binding"] = result["calibration"].startswith("左右手已绑定")
                result["early_progress"] = early_progress
                result["moderate_confidence_with_flicker"] = result["classifier_flickers"] > 0 and 0 < early_progress["value"] < 1
                result["binding_retained_after_partial_hand_loss"] = result["hands_free_binding"] and result["classifier_phases"] == ["palms", "gesture_changed", "partial_low_confidence", "duplicate_identity", "hands_lowered"]
                result["scripted_classifier_labels"] = True
                result["confirmation_removed"] = await cdp.evaluate("!document.getElementById('confirm-calibration') && !document.getElementById('recalibrate') && !document.getElementById('unbind-hands').disabled")
                await cdp.evaluate("document.getElementById('unbind-hands').click()")
                await asyncio.sleep(0.25)
                result["explicit_unbind"] = await cdp.evaluate("({disabled:document.getElementById('unbind-hands').disabled,progress:document.getElementById('calibration-progress').value,text:document.getElementById('calibration-state').textContent})")
                result["explicit_unbind_succeeded"] = result["explicit_unbind"]["disabled"] and result["explicit_unbind"]["progress"] == 0
                result["unexpected_actions"] = [command for command in backend.commands if command not in {"telemetry", "pause"}]
                result["valid"] = result["valid"] and result["binding_retained_after_partial_hand_loss"] and result["moderate_confidence_with_flicker"] and result["confirmation_removed"] and result["explicit_unbind_succeeded"] and not result["unexpected_actions"]
            if args.exercise_controls:
                moves = [item for item in backend.payloads if item["command"] == "dual_motion"]
                grips = [item["action"] for item in backend.payloads if item["command"] == "dual_gripper"]
                result["motion_requests"] = len(moves)
                result["gripper_actions"] = grips
                result["no_recording_or_save"] = not any(item.startswith("record_") for item in backend.commands)
                result["incremental_mode"] = all(item["telemetry"]["operator_control"]["motion_mapping"] == "incremental" for item in moves)
                sequence_start = await cdp.evaluate("window.__cameraCheck.sequenceStartedWall")
                result["stationary_hold_motion_requests"] = sum(4100 <= item["sentAt"] - sequence_start < 4430 for item in moves)
                result["unassigned_return_motion_requests"] = sum(4500 <= item["sentAt"] - sequence_start < 5220 for item in moves)
                result["motion_direction_correct"] = bool(moves) and all(item[side]["dx"] < 0 and item[side]["dy"] < 0 and item[side]["dz"] == 0 for item in moves for side in ("left", "right"))
                result["motion_within_speed_limit"] = all(sum(item[side][key] ** 2 for key in ("dx", "dy", "dz")) ** 0.5 <= 0.7 + 1e-9 for item in moves for side in ("left", "right"))
                await cdp.evaluate("document.getElementById('view-select').value='front'; document.getElementById('view-select').dispatchEvent(new Event('change'))")
                result["manual_view"] = await cdp.evaluate("({title:document.getElementById('main-view-title').textContent,urls:['main-preview','side-preview','front-preview'].map(id=>document.getElementById(id).src),mapping:document.getElementById('motion-mapping').value,speed:document.getElementById('motion-speed').value})")
                result["three_distinct_views"] = len(set(result["manual_view"]["urls"])) == 3
                result["bindings_retained"] = result["calibration"].startswith("左右手已绑定")
                result["obsolete_guidance_removed"] = await cdp.evaluate("!document.getElementById('alignment-values') && !document.getElementById('alignment-note') && !document.body.innerText.includes('蓝叉') && !document.body.innerText.includes('黄圈')")
                result["release_after_failure"] = backend.fail_after_close and backend.task_state == "FAIL" and backend.gripper_latch == "OPEN"
                opens = [item for item in backend.payloads if item["command"] == "dual_gripper" and item["action"] == "open"]
                closes = [item for item in backend.payloads if item["command"] == "dual_gripper" and item["action"] == "close"]
                result["release_palm_confidence"] = {side: opens[0]["telemetry"]["hands"][side]["confidence"] for side in ("left", "right")} if opens else {}
                result["close_fist_confidence"] = {side: closes[0]["telemetry"]["hands"][side]["confidence"] for side in ("left", "right")} if closes else {}
                result["moderate_confidence_release"] = result["release_palm_confidence"] == {"left": 0.57, "right": 0.57}
                result["moderate_confidence_close"] = result["close_fist_confidence"] == {"left": 0.82, "right": 0.82}
                result["pointing_pause_removed"] = await cdp.evaluate("!document.body.textContent.includes('食指向上')")
                result["thumb_pause_guidance_removed"] = await cdp.evaluate("!document.body.textContent.includes('竖拇指') && !document.body.textContent.includes('双拇指')")
                diagnostic_packets = [item for item in backend.payloads if isinstance(item.get("telemetry", {}).get("recognition_result"), dict)]
                diagnostics = [item["telemetry"]["recognition_result"] for item in diagnostic_packets]
                diagnostic_keys = {"frame_id", "source_session", "source_control_epoch", "captured_at_ms", "received_at_ms", "inference_ms", "result_age_ms", "accepted", "drop_reason", "observed_hands"}
                def diagnostic_shape_valid(item):
                    if not diagnostic_keys <= item.keys() or not isinstance(item["frame_id"], str) or not item["frame_id"]:
                        return False
                    if not isinstance(item["accepted"], bool) or (item["drop_reason"] is None) != item["accepted"]:
                        return False
                    numeric = (item[key] for key in ("captured_at_ms", "received_at_ms", "inference_ms", "result_age_ms"))
                    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in numeric):
                        return False
                    hands = item["observed_hands"]
                    return isinstance(hands, list) and all(isinstance(hand, dict) and set(hand) == {"label", "gesture", "score", "identity_score"}
                        and hand["label"] in {"left", "right", "unknown"} and isinstance(hand["gesture"], str)
                        and all(isinstance(hand[key], (int, float)) and math.isfinite(hand[key]) and 0 <= hand[key] <= 1 for key in ("score", "identity_score")) for hand in hands)
                actuations = [item for item in backend.payloads if item["command"] in {"dual_motion", "dual_gripper"}]
                max_age = backend.config["control"]["max_command_age_ms"]
                def action_diagnostic_fresh(item):
                    diagnostic = item.get("telemetry", {}).get("recognition_result", {})
                    # A newer heartbeat can be merged into queued motion while
                    # its original sentAt remains unchanged; both must be fresh
                    # and belong to the same backend scene, not necessarily the
                    # identical camera frame.
                    return (diagnostic.get("accepted") is True and diagnostic.get("drop_reason") is None
                        and diagnostic.get("source_session") == item.get("session")
                        and diagnostic.get("source_control_epoch") == item.get("control_epoch")
                        and 0 <= diagnostic.get("result_age_ms", float("inf")) <= max_age
                        and 0 <= diagnostic.get("captured_at_ms", -float("inf")) - item["sentAt"] <= max_age)
                result["recognition_diagnostics"] = {
                    "packets": len(diagnostics), "unique_frames": len({item.get("frame_id") for item in diagnostics}),
                    "rejected_results": sum(item.get("accepted") is False for item in diagnostics),
                    "shape_valid": bool(diagnostics) and all(diagnostic_shape_valid(item) for item in diagnostics),
                    "action_context_fresh": bool(actuations) and all(action_diagnostic_fresh(item) for item in actuations),
                    "actuation_count": len(actuations),
                    "actuation_exact_capture_matches": sum(item.get("telemetry", {}).get("recognition_result", {}).get("captured_at_ms") == item["sentAt"] for item in actuations),
                    "gripper_scores_match": all(item["telemetry"].get("recognition_result", {}).get("observed_hands") == [
                        {"label": side, "gesture": "Closed_Fist" if item["action"] == "close" else "Open_Palm",
                         "score": 0.82 if item["action"] == "close" else 0.57, "identity_score": 0.99}
                        for side in ("left", "right")] for item in closes + opens),
                }
                result["control_status"] = await cdp.evaluate("({state:document.getElementById('control-status').dataset.state,title:document.getElementById('control-status-title').textContent,reason:document.getElementById('notice').textContent,gripper:document.getElementById('control-gripper-state').textContent})")
                result["hand_loss_pause_visible"] = result["control_status"]["state"] == "paused" and "暂停" in result["control_status"]["title"] and ("丢手" in result["control_status"]["title"] or "离开画面" in result["control_status"]["reason"])
                result["visible_control_states"] = await cdp.evaluate("window.__cameraCheck.controlStates")
                shot = await cdp.call("Page.captureScreenshot", {"captureBeyondViewport": True})
                (args.output / "controls-page.png").write_bytes(base64.b64decode(shot["data"]))
                result["scripted_classifier_labels"] = True
                result["valid"] = result["valid"] and len(moves) >= 3 and grips == ["close", "open"] and result["no_recording_or_save"] and result["incremental_mode"] and result["stationary_hold_motion_requests"] == 0 and result["unassigned_return_motion_requests"] == 0 and result["motion_direction_correct"] and result["motion_within_speed_limit"] and result["three_distinct_views"] and result["bindings_retained"] and result["obsolete_guidance_removed"] and result["pointing_pause_removed"] and result["thumb_pause_guidance_removed"] and (not backend.fail_after_close or result["release_after_failure"])
                result["valid"] = result["valid"] and result["moderate_confidence_release"] and result["moderate_confidence_close"] and (backend.fail_after_close or result["hand_loss_pause_visible"])
                result["valid"] = result["valid"] and all(result["recognition_diagnostics"][key] for key in ("shape_valid", "action_context_fresh", "gripper_scores_match"))
            await cdp.evaluate("document.getElementById('camera-button').click()")
            cdp.reader.cancel()
            return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".local/task3-ui-check/browser")
    parser.add_argument("--app-source", type=Path, help="Optional previous app source for before/after reproduction")
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--allow-reproduced-failure", action="store_true")
    parser.add_argument("--exercise-calibration", action="store_true", help="Inject palms, gesture changes, unreliable identities and hand loss; verify palms-only binding and explicit unbinding without robot or gripper actions")
    parser.add_argument("--exercise-controls", action="store_true", help="Inject incremental movement, holding still, unassigned-gesture repositioning and grasp/release; verify the fake backend commands without moving a robot")
    parser.add_argument("--render-scene", action="store_true", help="Render static real MuJoCo operator views with top-only robot transparency; commands still go only to the fake backend")
    parser.add_argument("--fail-after-close", action="store_true", help="With --exercise-controls, inject FAIL after closing and verify explicit palms still release without clearing FAIL")
    args = parser.parse_args()
    if args.exercise_calibration and args.exercise_controls:
        parser.error("Choose one scripted classifier sequence")
    if args.exercise_controls and args.seconds < 9:
        parser.error("The control workflow requires --seconds at least 9")
    if args.fail_after_close and not args.exercise_controls:
        parser.error("--fail-after-close requires --exercise-controls")
    args.output.mkdir(parents=True, exist_ok=False)
    backend = Backend(render_scene=args.render_scene, fail_after_close=args.fail_after_close)
    server = ThreadingHTTPServer(("127.0.0.1", 0), backend.handler())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port, cdp_port = free_port(), free_port()
    conda = Path(sys.executable).parent
    chrome = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    processes = []
    try:
        with (args.output / "process.log").open("w", encoding="utf-8") as log:
            env = dict(os.environ, PORT=str(port), DUAL_BACKEND_PORT=str(server.server_port))
            processes.append(subprocess.Popen([str(conda / "node.exe"), "teleoperation/mediapipe/server.js"], cwd=ROOT, env=env,
                             stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW))
            processes.append(subprocess.Popen([str(chrome), "--headless=new", "--no-first-run", "--no-default-browser-check",
                             "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", "--disable-background-networking",
                             f"--remote-debugging-port={cdp_port}", f"--user-data-dir={args.output.resolve() / 'profile'}", "about:blank"],
                             stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW))
            result = asyncio.run(exercise(args, backend, port, cdp_port))
            (args.output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({key:value for key,value in result.items() if key not in {'connection','longTasks','javascript_errors'}}, ensure_ascii=False))
            if not result["valid"] and not args.allow_reproduced_failure:
                raise SystemExit("Camera UI regression check failed; inspect report.json")
    finally:
        server.shutdown()
        server.server_close()
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)


if __name__ == "__main__":
    main()
