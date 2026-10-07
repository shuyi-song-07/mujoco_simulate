#!/usr/bin/env node
/**
 * Read-only replay of recorded grasp/release confirmation windows through the
 * production DualInteractionState. Emits JSON to stdout; never writes inputs.
 *
 * This is fixed-recorded-input counterfactual analysis, not a physics rollout
 * or a prediction of how an operator would react to an earlier gripper event.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { DualInteractionState } from "../../teleoperation/mediapipe/dual-interaction-state.js";

const projectRoot = fileURLToPath(new URL("../../", import.meta.url));
const sides = ["left", "right"];
const round = (value, places = 3) => Math.round(value * 10 ** places) / 10 ** places;

function argumentsFrom(argv) {
  const options = { episode: 0, config: resolve(projectRoot, "config/task3_dual_arm.json") };
  for (let index = 0; index < argv.length; index++) {
    const option = argv[index];
    if (option === "--help") return { help: true };
    if (!["--dataset", "--episode", "--config"].includes(option) || index + 1 >= argv.length) throw new Error(`Unknown or missing argument: ${option}`);
    options[option.slice(2)] = argv[++index];
  }
  options.episode = Number(options.episode);
  if (!options.dataset || !Number.isSafeInteger(options.episode) || options.episode < 0) throw new Error("Use --dataset PATH [--episode INDEX] [--config CANDIDATE_JSON]");
  options.dataset = resolve(options.dataset);
  options.config = resolve(options.config);
  return options;
}

function independentSamples(rows) {
  const samples = [], seen = new Set();
  for (const [index, row] of rows.entries()) {
    const telemetry = row.telemetry, hands = telemetry?.hands;
    if (!sides.every((side) => hands?.[side]?.visible && Number.isFinite(hands[side].last_seen_ms))) continue;
    // Each control tick can repeat one recognition result. Do not grant that
    // repetition extra gesture evidence or count it as a new camera sample.
    const key = JSON.stringify(sides.map((side) => hands[side].last_seen_ms));
    if (seen.has(key)) continue;
    seen.add(key);
    if (hands.left.last_seen_ms !== hands.right.last_seen_ms) throw new Error(`Unequal trusted pair timestamps at raw line ${index + 1}`);
    samples.push({ index, row, now: hands.left.last_seen_ms, detections: sides.map((side) => ({
      side, gesture: hands[side].gesture, confidence: hands[side].confidence,
      handedness_score: hands[side].handedness_score, wrist: [...hands[side].wrist_raw],
    })) });
  }
  return samples;
}

function confirmationWindow(rows, samples, eventIndex, action) {
  const mode = action === "close" ? "GRASP_CONFIRM" : "RELEASE_CONFIRM";
  let start = eventIndex;
  while (start > 0 && rows[start - 1].telemetry?.control_mode === mode) start--;
  assert.equal(rows[start].telemetry?.control_mode, mode, `No confirmation window for ${action} at line ${eventIndex + 1}`);
  const firstSampleIndex = samples.findIndex((sample) => sample.index >= start);
  assert.ok(firstSampleIndex > 0, "Replay requires a preceding trusted transition sample");
  const initial = samples[firstSampleIndex - 1];
  const calibration = initial.row.telemetry.calibration;
  assert.equal(initial.row.telemetry.calibrated, true, "Only already bound confirmation windows can be reconstructed");
  assert.equal(calibration.palm_binding_guard, false, "Binding-palm guard is active at the replay boundary");
  assert.equal(calibration.tracking_recovery.pending, false, "Tracking recovery is active at the replay boundary");
  assert.equal(initial.row.telemetry.gripper_pending, null, "Another gripper event is pending at the replay boundary");
  assert.ok(sides.every((side) => initial.row.telemetry.evidence_ms[action][side] === 0), "This confirmation window starts with existing action evidence; choose a complete window");
  return { start, eventIndex, initial, confirmationStart: samples[firstSampleIndex],
    samples: samples.filter((sample) => sample.index > initial.index && sample.index <= eventIndex) };
}

function replay(window, control, action, firstWallTime) {
  const engine = new DualInteractionState(control);
  const initial = window.initial;
  // Restore only the observed trusted binding context. The recorded source
  // starts after binding, so inventing an operator calibration would be false.
  // Every candidate result after this boundary goes through the real tracker,
  // identity checks, evidence decay, task gate and discrete-event logic.
  Object.assign(engine.tracker, {
    calibrated: true,
    calibrationSource: initial.row.telemetry.calibration.source,
    calibrationConfirmedAt: initial.row.telemetry.calibration.confirmed_at_monotonic_ms,
    palmBindingGuard: false,
    recoveryPending: false,
    controlResumeRequired: false,
    validFrames: Math.max(5, engine.config.recovery_frames),
  });
  engine.syncHealth({ task_state: initial.row.task_state, gripper_latch: initial.row.gripper_latch }, initial.now);
  engine.update(initial.detections, initial.now);
  assert.equal(engine.pendingGripper, null);
  const reasons = {};
  let processed = 0;
  for (const sample of window.samples) {
    processed++;
    const age = sample.row.telemetry.recognition_age_ms;
    // Result freshness is checked by the browser outside the FSM. Recreate
    // that gate without treating stale recorded data as accepted observations.
    const limit = Math.min(150, control.max_command_age_ms ?? 150);
    const result = Number.isFinite(age) && age > limit
      ? engine.pause("recognition_stale")
      : engine.update(sample.detections, sample.now);
    if (result.reason) reasons[result.reason] = (reasons[result.reason] ?? 0) + 1;
    if (result.command?.command !== "dual_gripper") continue;
    assert.equal(result.command.action, action, "The replay emitted a different gripper action");
    return {
      emitted: true, source_line: sample.index + 1,
      source_wall_s: round((sample.row.wall_time_ms - firstWallTime) / 1000),
      confirmation_ms: round(sample.now - window.confirmationStart.now),
      including_transition_ms: round(sample.now - initial.now),
      thresholds: result.telemetry.gripper_thresholds,
      independent_samples_processed: processed,
      recognition_reasons: reasons,
    };
  }
  return { emitted: false, thresholds: engine.telemetry().gripper_thresholds, recognition_reasons: reasons };
}

function main() {
  const options = argumentsFrom(process.argv.slice(2));
  if (options.help) {
    console.log("Read-only production-FSM replay: node tests/task3/check_gesture_replay.mjs --dataset PATH [--episode 0] [--config CANDIDATE_JSON]\nJSON goes to stdout; redirect outside the dataset if an artifact is needed.");
    return;
  }
  const stem = `episode_${String(options.episode).padStart(6, "0")}`;
  const rawPath = resolve(options.dataset, "research", `${stem}.jsonl`);
  const frozenPath = resolve(options.dataset, "research", `${stem}_snapshots`, "task3.json");
  const rows = readFileSync(rawPath, "utf8").trim().split(/\r?\n/).map((line) => JSON.parse(line));
  const frozen = JSON.parse(readFileSync(frozenPath, "utf8"));
  const candidate = JSON.parse(readFileSync(options.config, "utf8"));
  assert.ok(rows.length > 0 && frozen.config?.control && candidate.control, "Missing raw rows or control configuration");
  const samples = independentSamples(rows), actions = [];
  for (const [eventIndex, row] of rows.entries()) {
    for (const event of row.events ?? []) {
      if (event.type !== "gripper" || !["close", "open"].includes(event.action)) continue;
      const window = confirmationWindow(rows, samples, eventIndex, event.action);
      const recorded = replay(window, frozen.config.control, event.action, rows[0].wall_time_ms);
      const proposed = replay(window, candidate.control, event.action, rows[0].wall_time_ms);
      actions.push({
        action: event.action, actual_event_line: eventIndex + 1,
        actual_event_wall_s: round((row.wall_time_ms - rows[0].wall_time_ms) / 1000),
        confirmation_start_line: window.confirmationStart.index + 1,
        transition_sample_line: window.initial.index + 1,
        distinct_window_samples: window.samples.length,
        recorded, candidate: proposed,
        recorded_event_reproduced: recorded.emitted && recorded.source_line === eventIndex + 1,
      });
    }
  }
  const result = {
    dataset: options.dataset, episode: options.episode, raw_rows: rows.length,
    independent_trusted_pairs: samples.length, recorded_version: frozen.config.version,
    candidate_version: candidate.version, candidate_config: options.config,
    method: "production DualInteractionState; de-duplicate trusted pairs by last_seen_ms; restore observed binding and pre-event task/latch for each complete confirmation window",
    scope: "fixed-recorded-input counterfactual; no MuJoCo rollout, no new recognition, no operator response, no data writes",
    timing: "confirmation_ms starts at the first recorded confirmation sample; including_transition_ms also includes the preceding trusted transition frame",
    limitations: "A recording can omit camera results. Restoration uses an already bound, recovered, zero-evidence boundary. Earlier hypothetical events would alter later human/physics behavior; replay timing is not a closed-loop guarantee or false-trigger-rate estimate.",
    actions,
    valid: actions.length > 0 && actions.every((action) => action.recorded_event_reproduced && action.candidate.emitted),
  };
  console.log(JSON.stringify(result, null, 2));
  if (!result.valid) process.exitCode = 1;
}

try { main(); }
catch (error) { console.error(error.stack ?? error.message); process.exitCode = 1; }
