/* Independent fixtures for the human review display contract. Run with Node;
 * no browser, source recording, simulator, or package installation is needed.
 */
'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const metrics = require('../../dataset/task3/review_metrics.js');

test('actual review playback controls accept single-view and legacy three-view media', () => {
  const fs = require('node:fs'), vm = require('node:vm'), path = require('node:path');
  const html = fs.readFileSync(path.join(__dirname, '../../dataset/task3/review.html'), 'utf8');
  const fn = html.slice(html.indexOf('function playbackControls(){'), html.indexOf('function showOriginal(){'));
  for (const count of [0, 1, 3]) for (const loading of [false, true]) for (const previewing of [false, true]) {
    const nodes = {};
    const $ = id => nodes[id] ??= {setAttribute(){}, classList:{toggle(){}}};
    vm.runInNewContext(`${fn}\nplaybackControls();`, {
      $, document:{querySelector:()=>({classList:{toggle(){}}})},
      videos:Array.from({length:count},()=>({})), loading, playing:false,
      preview:previewing?{kept_frames:20}:null, previewBusy:false, manifest:{read_only:false},
      episode:()=>({}), frameCount:()=>20, playCount:()=>20,
    });
    assert.equal(nodes.play.disabled, loading || count===0,
      `videos=${count}, loading=${loading}, preview=${previewing}`);
  }
});

test('intent policy display distinguishes automatic evidence from human effective keep', () => {
  const source = {frame_role:'GRASP_PROTECTED', policy_decision:'keep', keep:false,
    accepted_left_intent:'Z_CLOSE', accepted_right_intent:'HOLD',
    policy_reason:['actual_grasp_execution'], stagnant_run_id:0, grasp_protected:true,
    release_protected:false, review_required:true, task_segment_id:2,
    contiguous_fragment_id:null, soft_curation_gap:false};
  const before = JSON.stringify(source), summary = metrics.policySummary(source);
  assert.equal(summary.automaticKeep, true);
  assert.equal(summary.label, '抓取执行／验证保护');
  assert.equal(summary.left, 'Z升降／握拳模式');
  assert.equal(summary.right, '保持／中性状态');
  assert.equal(summary.stagnantRun, 0);
  assert.equal(summary.grasp, true);
  assert.equal(summary.reviewRequired, true);
  assert.equal(JSON.stringify(source), before);
});
test('intent policy display does not invent evidence for historical rows', () => {
  assert.equal(metrics.policySummary({keep:true}), null);
  const summary = metrics.policySummary({frame_role:'AMBIGUOUS_IDLE', policy_decision:'keep'});
  assert.equal(summary.left, '无可靠记录');
  assert.equal(summary.grasp, false);
  assert.equal(summary.stagnantRun, null);
});
test('intent policy display retains both protection kinds and invalid status', () => {
  const summary = metrics.policySummary({frame_role:'INVALID', policy_decision:'exclude',
    grasp_protected:true, release_protected:true, review_required:true,
    policy_reason:['invalid_snapshot']});
  assert.equal(summary.invalid, true);
  assert.equal(summary.automaticKeep, false);
  assert.equal(summary.grasp, true);
  assert.equal(summary.release, true);
  assert.deepEqual(summary.reasons, ['invalid_snapshot']);
});

const DEG = 180 / Math.PI;
const options = {
  fps: 30,
  stability: {position_m: 0.0001, rotation_rad: 0.001,
    joint_rad: 0.001, aperture_m: 0.0001},
  alignment: {maximum_frame_gap_s: 0.05, maximum_next_control_gap_s: 0.03},
};

function row(index, overrides = {}) {
  return {
    source_episode: 'test-session/0', session: 'test-session', episode_index: 0,
    raw_frame_index: index, source_control_tick: index * 2, control_epoch: 1,
    original_time_s: index / 30, recorded_video_time_s: index / 30,
    keep: true, segment_id: 0,
    action_recorded: [0.1, 0.2, 0.3, 0, 0, 0, 0.04,
      -0.1, -0.2, 0.4, 0, 0, 0, 0.06],
    state_recorded: [0, 0, 0, 0, 0, 0, 0, 0.035,
      0, 0, 0, 0, 0, 0, 0, 0.055],
    status_summary: {paused: false, pause_reason: null},
    auxiliary: {
      left: {category: 'Open_Palm', category_score: 0.95,
        category_mask: true, identity_score: 0.99, age_ms: 20,
        measurement_id: 'measurement-left', reason: 'trusted_category', raw_label: 'Left'},
      right: {category: 'Closed_Fist', category_score: 0.92,
        category_mask: true, identity_score: 0.98, age_ms: 23,
        measurement_id: 'measurement-right', reason: 'trusted_category', raw_label: 'Right'},
    },
    ...overrides,
  };
}

function near(actual, expected, tolerance = 1e-9) {
  assert.equal(typeof actual, 'number', `Expected numeric ${expected}, got ${actual}`);
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
}

function frame(previous = row(0), current = row(1), extra = {}) {
  return metrics.compareFrame([previous, current], current.raw_frame_index, {...options, ...extra});
}

function allFields(result) {
  return ['action', 'state'].flatMap(kind => ['left', 'right'].flatMap(side => result[kind][side].fields));
}

function expectNoDeltas(result) {
  assert.equal(result.boundary.comparable, false);
  for (const field of allFields(result)) {
    assert.equal(field.delta, null);
    assert.equal(field.change, 'unavailable');
  }
}

function nextTarget(source, action = source.action_recorded.slice(), overrides = {}) {
  return {action, source_tick: source.source_control_tick + 1,
    sim_time: source.original_time_s + 1 / 60, control_epoch: source.control_epoch,
    alignment: 'next_observed_control_tick_after_post_step_snapshot', ...overrides};
}

function freezeDeep(value) {
  if (value && typeof value === 'object') {
    Object.freeze(value);
    for (const child of Object.values(value)) freezeDeep(child);
  }
  return value;
}

test('jumping around the player always compares raw frame n with raw frame n-1', () => {
  const rows = [row(7), row(8), row(9), row(10)];
  rows[0].action_recorded[0] = -0.5;
  rows[1].action_recorded[0] = -0.4;
  rows[2].action_recorded[0] = 0.12;
  rows[3].action_recorded[0] = 0.123;
  metrics.compareFrame(rows, 7, options);
  const tenth = metrics.compareFrame(rows, 10, options);
  assert.equal(tenth.previous.raw_frame_index, 9);
  near(tenth.action.left.fields[0].delta, 3);
  const eighth = metrics.compareFrame(rows, 8, options);
  assert.equal(eighth.previous.raw_frame_index, 7);
  near(eighth.action.left.fields[0].delta, 100);
});

test('array order cannot substitute for raw frame identity', () => {
  const previous = row(8), current = row(9);
  current.action_recorded[1] += 0.002;
  const result = metrics.compareFrame([current, row(30), previous], 9, options);
  assert.equal(result.previous.raw_frame_index, 8);
  near(result.action.left.fields[1].delta, 2);
});

test('the first raw frame has current values but no manufactured zero difference', () => {
  const result = metrics.compareFrame([row(0)], 0, options);
  expectNoDeltas(result);
  near(result.action.left.fields[0].current, 100);
  assert.equal(result.previous, null);
});

test('missing immediate raw predecessor cannot fall back to an earlier kept frame', () => {
  const result = metrics.compareFrame([row(0), row(2)], 2, options);
  expectNoDeltas(result);
  assert.equal(result.previous, null);
});

test('a missing selected frame has no fabricated current values or differences', () => {
  const result = metrics.compareFrame([row(0), row(2)], 1, options);
  expectNoDeltas(result);
  assert.equal(result.current, null);
  for (const field of allFields(result)) assert.equal(field.current, null);
});

test('duplicate current raw indices cannot silently select one of two physical records', () => {
  const result = metrics.compareFrame([row(0), row(1), row(1)], 1, options);
  expectNoDeltas(result);
  assert.equal(result.current, null);
});

test('duplicate previous raw indices cannot silently select one predecessor', () => {
  const result = metrics.compareFrame([row(0), row(0), row(1)], 1, options);
  expectNoDeltas(result);
  assert.equal(result.previous, null);
});

test('automatic keep and segment boundaries do not hide physical adjacent-frame changes', () => {
  const previous = row(0, {keep: false, segment_id: null});
  const current = row(1, {keep: true, segment_id: 9, cut_reasons: ['automatic_wait_cut']});
  current.action_recorded[2] += 0.004;
  const result = frame(previous, current);
  assert.equal(result.boundary.comparable, true);
  near(result.action.left.fields[2].delta, 4);
});

test('a14 fields separate both arms and convert metres to millimetres and radians to degrees', () => {
  const previous = row(0), current = row(1);
  current.action_recorded = [0.101, 0.198, 0.303, Math.PI / 2, -Math.PI / 4, Math.PI / 6, 0.045,
    -0.103, -0.196, 0.395, -Math.PI / 2, Math.PI / 4, -Math.PI / 6, 0.058];
  const result = frame(previous, current);
  assert.equal(result.action.left.fields.length, 7);
  assert.equal(result.action.right.fields.length, 7);
  const left = result.action.left.fields, right = result.action.right.fields;
  [1, -2, 3, 90, -45, 30, 5].forEach((value, i) => near(left[i].delta, value));
  [-3, 4, -5, -90, 45, -30, -2].forEach((value, i) => near(right[i].delta, value));
  assert.deepEqual(left.map(f => f.unit), ['mm', 'mm', 'mm', '°', '°', '°', 'mm']);
  near(left[6].current, 45);
  near(right[6].current, 58);
});

test('s16 is seven measured joint angles and measured total aperture per arm, not xyz', () => {
  const previous = row(0), current = row(1);
  current.state_recorded = [0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07, 0.038,
    -0.08, 0.09, -0.10, 0.11, -0.12, 0.13, -0.14, 0.051];
  const result = frame(previous, current);
  assert.equal(result.state.left.fields.length, 8);
  assert.equal(result.state.right.fields.length, 8);
  for (let i = 0; i < 7; ++i) {
    near(result.state.left.fields[i].delta, current.state_recorded[i] * DEG);
    near(result.state.right.fields[i].delta, current.state_recorded[i + 8] * DEG);
  }
  near(result.state.left.fields[7].delta, 3);
  near(result.state.right.fields[7].delta, -4);
  assert.deepEqual(result.state.left.fields.map(f => f.unit), ['°', '°', '°', '°', '°', '°', '°', 'mm']);
});

test('RPY crossing +pi to -pi uses the short signed angular difference', () => {
  const previous = row(0), current = row(1);
  previous.action_recorded[5] = 179 / DEG;
  current.action_recorded[5] = -179 / DEG;
  previous.action_recorded[12] = -179 / DEG;
  current.action_recorded[12] = 179 / DEG;
  const result = frame(previous, current);
  near(result.action.left.fields[5].delta, 2);
  near(result.action.right.fields[5].delta, -2);
});

test('actual joint coordinates do not borrow periodic RPY wrapping', () => {
  const previous = row(0), current = row(1);
  previous.state_recorded[0] = 179 / DEG;
  current.state_recorded[0] = -179 / DEG;
  const result = frame(previous, current);
  near(result.state.left.fields[0].delta, -358);
});

test('a held target with changing measured joints is not reported as a wholly static robot', () => {
  const previous = row(0), current = row(1);
  current.state_recorded[8] = 0.02;
  const result = frame(previous, current);
  assert.equal(result.action.left.summary, 'unchanged');
  assert.equal(result.action.right.summary, 'unchanged');
  assert.equal(result.state.right.summary, 'changed');
  near(result.state.right.fields[0].delta, 0.02 * DEG);
});

test('real zero coordinates are numeric; unavailable values are not coerced to zero', () => {
  const previous = row(0), current = row(1);
  previous.action_recorded[0] = current.action_recorded[0] = 0;
  current.action_recorded[1] = null;
  current.state_recorded[0] = Number.NaN;
  current.state_recorded[8] = Number.POSITIVE_INFINITY;
  const result = frame(previous, current);
  assert.equal(result.action.left.fields[0].current, 0);
  assert.equal(result.action.left.fields[0].delta, 0);
  for (const field of [result.action.left.fields[1], result.state.left.fields[0], result.state.right.fields[0]]) {
    assert.equal(field.current, null);
    assert.equal(field.delta, null);
    assert.equal(field.change, 'unavailable');
  }
});

test('wrong action dimensions cannot shift a right-arm field into a left-arm comparison', () => {
  const previous = row(0), current = row(1);
  current.action_recorded = current.action_recorded.slice(1);
  const result = frame(previous, current);
  assert.equal(result.action.left.validCount, 0);
  assert.equal(result.action.right.validCount, 0);
  assert.equal(result.state.left.validCount, 8);
});

test('wrong measured-state dimensions are unavailable while valid action values remain usable', () => {
  const previous = row(0), current = row(1);
  current.state_recorded = [...current.state_recorded, 42];
  const result = frame(previous, current);
  assert.equal(result.state.left.validCount, 0);
  assert.equal(result.state.right.validCount, 0);
  assert.equal(result.action.left.validCount, 7);
});

test('missing action/state fields do not borrow next targets or the other vector', () => {
  const previous = row(0), current = row(1);
  current.next_control_target = nextTarget(current);
  delete current.action_recorded;
  delete current.state_recorded;
  const result = frame(previous, current);
  for (const field of allFields(result)) {
    assert.equal(field.current, null);
    assert.equal(field.delta, null);
  }
});

for (const [name, modify] of [
  ['changed control epoch', r => { r.control_epoch = 2; }],
  ['missing control epoch', r => { delete r.control_epoch; }],
  ['changed source session', r => { r.session = 'another-session'; }],
  ['changed source episode', r => { r.source_episode = 'test-session/1'; }],
  ['missing physical timestamp', r => { delete r.original_time_s; }],
  ['equal physical timestamps', r => { r.original_time_s = 0; }],
  ['backwards physical timestamp', r => { r.original_time_s = -0.01; }],
  ['long physical time gap', r => { r.original_time_s = 0.1; }],
]) {
  test(`${name} cannot manufacture zero differences across a discontinuity`, () => {
    const previous = row(0), current = row(1);
    modify(current);
    const result = frame(previous, current);
    expectNoDeltas(result);
    near(result.action.left.fields[0].current, 100);
  });
}

test('small genuine differences remain visible and do not authorize a deletion or quality decision', () => {
  const previous = row(0), current = row(1);
  current.action_recorded[0] += 0.00001;
  current.state_recorded[0] += 0.0001;
  const rows = freezeDeep([previous, current]);
  const original = JSON.stringify(rows);
  const result = metrics.compareFrame(rows, 1, options);
  near(result.action.left.fields[0].delta, 0.01);
  near(result.state.left.fields[0].delta, 0.0001 * DEG);
  assert.equal(result.action.left.fields[0].change, 'tiny');
  assert.equal(result.state.left.fields[0].change, 'tiny');
  assert.equal(result.action.left.changedCount, 0);
  assert.equal(result.action.left.tinyCount, 1);
  assert.equal(current.keep, true);
  assert.equal(JSON.stringify(rows), original);
  assert.equal(Object.hasOwn(result, 'delete'), false);
  assert.equal(Object.hasOwn(result, 'training_approved'), false);
  assert.equal(Object.hasOwn(result, 'quality'), false);
});

test('difference emphasis respects configured physical-unit thresholds', () => {
  const previous = row(0), current = row(1);
  current.action_recorded[0] += 0.0002;
  const normal = frame(previous, current);
  const wider = frame(previous, current, {stability: {...options.stability, position_m: 0.001}});
  assert.equal(normal.action.left.fields[0].change, 'changed');
  assert.equal(wider.action.left.fields[0].change, 'tiny');
  near(normal.action.left.fields[0].delta, wider.action.left.fields[0].delta);
});

test('next-control training targets and recorded current targets are different selectable quantities', () => {
  const previous = row(0), current = row(1);
  previous.next_control_target = nextTarget(previous);
  current.next_control_target = nextTarget(current);
  current.next_control_target.action[0] += 0.006;
  const recorded = frame(previous, current);
  const next = frame(previous, current, {actionSource: 'next_control'});
  near(recorded.action.left.fields[0].delta, 0);
  near(next.action.left.fields[0].delta, 6);
  assert.equal(next.actionSource, 'next_control');
  assert.equal(next.state.left.summary, recorded.state.left.summary);
});

test('a missing next-control target never silently falls back to a recorded target', () => {
  const previous = row(0), current = row(1);
  previous.next_control_target = nextTarget(previous);
  const result = frame(previous, current, {actionSource: 'next_control'});
  assert.equal(result.action.left.validCount, 0);
  for (const field of result.action.left.fields) assert.equal(field.delta, null);
  assert.equal(result.state.left.validCount, 8);
});

for (const [name, override] of [
  ['wrong epoch', {control_epoch: 99}],
  ['target preceding its image', {sim_time: -0.1}],
  ['target too far after its image', {sim_time: 0.5}],
  ['nonadjacent control tick', {source_tick: 100}],
]) {
  test(`next-control ${name} is unavailable without hiding valid measured-state differences`, () => {
    const previous = row(0), current = row(1);
    previous.next_control_target = nextTarget(previous);
    current.next_control_target = nextTarget(current, current.action_recorded.slice(), override);
    const result = frame(previous, current, {actionSource: 'next_control'});
    assert.equal(result.action.left.validCount, 0);
    assert.equal(result.action.right.validCount, 0);
    assert.equal(result.state.left.validCount, 8);
  });
}

test('gesture category is not confused with handedness or the accepted controller mode', () => {
  const current = row(1, {original_control_labels: {interaction_modes: {left: 'HOLD', right: 'Z_CLOSE'}}});
  const gesture = metrics.gestureSummary(current, 'left');
  assert.equal(gesture.category, 'Open_Palm');
  near(gesture.score, 0.95);
  assert.notEqual(gesture.label, 'Left');
  assert.notEqual(gesture.category, 'HOLD');
  assert.equal(gesture.mode, 'HOLD');
});

test('a missing gesture measurement is explicitly missing rather than Unknown or zero confidence', () => {
  const current = row(1, {auxiliary: {}});
  const gesture = metrics.gestureSummary(current, 'left');
  assert.equal(gesture.trust, 'missing');
  assert.equal(gesture.category, null);
  assert.equal(gesture.score, null);
  assert.equal(gesture.supervision, false);
});

for (const category of ['None', 'Unknown']) {
  test(`recorded ${category} is an unknown hand shape, not an absent measurement`, () => {
    const current = row(1);
    current.auxiliary.left = {...current.auxiliary.left, category, category_score: 0.91,
      category_mask: false, reason: 'unknown_or_low_score'};
    const gesture = metrics.gestureSummary(current, 'left');
    assert.equal(gesture.category, category);
    assert.notEqual(gesture.trust, 'missing');
    assert.equal(gesture.supervision, false);
    near(gesture.score, 0.91);
  });
}

test('a low-confidence recognized palm remains visible but is not promoted to trusted supervision', () => {
  const current = row(1);
  current.auxiliary.left = {...current.auxiliary.left, category_score: 0.12,
    category_mask: false, reason: 'unknown_or_low_score'};
  const gesture = metrics.gestureSummary(current, 'left');
  assert.equal(gesture.category, 'Open_Palm');
  near(gesture.score, 0.12);
  assert.equal(gesture.trust, 'untrusted');
  assert.equal(gesture.supervision, false);
});

test('zero gesture confidence is a recorded zero, not missing information', () => {
  const current = row(1);
  current.auxiliary.left = {...current.auxiliary.left, category_score: 0,
    category_mask: false, reason: 'unknown_or_low_score'};
  const gesture = metrics.gestureSummary(current, 'left');
  assert.equal(gesture.score, 0);
  assert.notEqual(gesture.trust, 'missing');
});

test('pause masks current training supervision even when the recorded category itself is trusted', () => {
  const current = row(1, {status_summary: {paused: true, pause_reason: 'manual_pause'}});
  const gesture = metrics.gestureSummary(current, 'left');
  assert.equal(gesture.category, 'Open_Palm');
  assert.equal(gesture.paused, true);
  assert.equal(gesture.supervision, false);
});

test('an unknown category does not itself assert that the hand left the camera frame', () => {
  const current = row(1);
  current.auxiliary.left = {...current.auxiliary.left, category: 'None',
    category_mask: false, reason: 'unknown_or_low_score'};
  const gesture = metrics.gestureSummary(current, 'left');
  assert.equal(gesture.globalHandLost, false);
  assert.equal(gesture.paused, false);
});

test('left and right gesture records remain independent', () => {
  const current = row(1);
  current.auxiliary.right = {category: null, category_score: null, category_mask: false,
    reason: 'measurement_unavailable_or_not_causal'};
  const result = metrics.compareFrame([row(0), current], 1, options);
  assert.equal(result.gestures.left.category, 'Open_Palm');
  assert.equal(result.gestures.left.supervision, true);
  assert.equal(result.gestures.right.trust, 'missing');
  assert.equal(result.gestures.right.supervision, false);
});

test('explicit category-only policy preserves a low-score paused label without claiming confidence', () => {
  const current = row(1, {category_policy: 'recorded_category_no_score_v1',
    status_summary: {paused: true, pause_reason: 'manual_pause'}});
  current.auxiliary.left.category_score = 0.01;
  current.auxiliary.left.identity_score = 0.01;
  const gesture = metrics.gestureSummary(current, 'left');
  assert.equal(gesture.category, 'Open_Palm');
  assert.equal(gesture.trust, 'recorded');
  assert.equal(gesture.recordedWithoutScores, true);
  assert.equal(gesture.supervision, true);
  assert.equal(gesture.paused, true);
  assert.match(gesture.description, /不按手势／身份分数/);
});

test('category-only policy cannot enable absent or unknown category targets', () => {
  const current = row(1, {category_policy: 'recorded_category_no_score_v1'});
  current.auxiliary.left = {...current.auxiliary.left, category: 'Unknown', category_mask: false};
  current.auxiliary.right = {category: null, category_mask: false,
    reason: 'measurement_unavailable_or_not_causal'};
  assert.equal(metrics.gestureSummary(current, 'left').supervision, false);
  assert.equal(metrics.gestureSummary(current, 'right').supervision, false);
  assert.equal(metrics.gestureSummary(current, 'right').trust, 'missing');
});

test('s/a selection exposes distinct actual and target aperture thresholds', () => {
  const result = metrics.compareFrame([row(0), row(1)], 1, {...options,
    selection: {thresholds: {state_aperture_m: 0.0002, action_aperture_m: 0.0008}}});
  assert.equal(result.state.left.fields[7].threshold, 0.2);
  assert.equal(result.action.left.fields[6].threshold, 0.8);
});

test('reposition wait has a clear Chinese label and preserves automatic decision', () => {
  const p = metrics.policySummary({frame_role: 'RECENTER_IDLE', keep: false,
    policy_decision: 'exclude', policy_reason: ['stationary_reposition_same_intent'],
    accepted_left_intent: 'HOLD', accepted_right_intent: 'XY_HOLD'});
  assert.equal(p.label, '回中／追踪恢复等待');
  assert.equal(p.automaticKeep, false);
  assert.equal(p.reviewRequired, false);
  assert.deepEqual(p.reasons, ['stationary_reposition_same_intent']);
});


// Exercise the actual HTML renderer, not a parallel test-only label template.
function renderPolicyFixture(source) {
  const fs = require('node:fs'), vm = require('node:vm'), path = require('node:path');
  const html = fs.readFileSync(path.join(__dirname, '../../dataset/task3/review.html'), 'utf8');
  const reasons = html.slice(html.indexOf('const reasonNames='), html.indexOf('const pauseNames='));
  const label = html.match(/^const labelReasons=.*$/m)?.[0];
  const begin = html.indexOf('function renderPolicy(row){');
  const render = html.slice(begin, html.indexOf('\nfunction updateTime()', begin));
  assert.ok(reasons && label && begin >= 0 && render, 'The real policy renderer must be available');
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
  const elements = new Map(ids.map(id => [id, {textContent: '', className: '', hidden: false,
    classList: {toggle(name, on) { elements.get(id)[name] = on; }}}]));
  const context = {M: metrics, $: id => { assert.ok(elements.has(id), `Missing HTML element ${id}`); return elements.get(id); },
    episode: () => ({}), current: source.raw_frame_index ?? 0, source};
  vm.runInNewContext(`${reasons}\n${label}\n${render}\nrenderPolicy(source);`, context);
  return Object.fromEntries([...elements.entries()].map(([id, element]) => [id, element]));
}

function transitionWait(overrides = {}) {
  return {
    raw_frame_index: 357, frame_role: 'INTENT_TRANSITION_IDLE', policy_decision: 'exclude',
    policy_reason: ['stationary_intent_transition'], accepted_left_intent: 'HOLD',
    accepted_right_intent: 'Z_OPEN', stagnant_run_id: 8, task_segment_id: 0,
    pre_intent: ['XY_HOLD', 'XY_HOLD'], post_intent: ['Z_OPEN', 'Z_OPEN'],
    phase_pause_interpretation: {
      phase: 'transition', pre_intent_evidence: {modes: ['XY_HOLD', 'XY_HOLD']},
      post_intent_evidence: {modes: ['Z_OPEN', 'Z_OPEN']}, retained_frame_budget: 10,
      retained_frames: 10, minimum_duration_s: 1.0,
    }, ...overrides,
  };
}

test('stationary mode transition displays the real role, reason and before/after intent', () => {
  const source = freezeDeep(transitionWait()), before = JSON.stringify(source);
  const summary = metrics.policySummary(source), rendered = renderPolicyFixture(source);
  assert.equal(summary.label, '换模式期间静止等待');
  assert.equal(summary.automaticKeep, false);
  assert.match(rendered.policyRole.textContent, /换模式期间静止等待 · 自动删除/);
  assert.match(rendered.policyReason.textContent, /前后控制模式不同/);
  assert.doesNotMatch(rendered.policyReason.textContent, /证据不足|其他记录/);
  assert.match(rendered.policyWait.textContent, /等待前：左 XY移动，保持爪宽；右 XY移动，保持爪宽/);
  assert.match(rendered.policyWait.textContent, /等待后：左 Z升降／张掌模式；右 Z升降／张掌模式/);
  assert.match(rendered.policyWait.textContent, /保留预算：10 帧（不是毫秒）/);
  assert.match(rendered.policyWait.textContent, /实际保留 10 帧/);
  assert.match(rendered.policyWait.textContent, /至少 1 秒仿真时间/);
  assert.equal(rendered.policyWait.hidden, false);
  assert.equal(JSON.stringify(source), before);
});

for (const [reason, expected] of [
  ['missing_pre_intent_context', /等待前可比较的可靠稳定意图/],
  ['missing_post_intent_context', /等待后可比较的可靠稳定意图/],
  ['unexplained_mode_excursion', /额外控制模式.*返回旧模式/],
  ['physical_event_in_wait_context', /抓放执行、取消、接触变化或抓放保护/],
  ['no_verified_stationary_core', /没有找到可确认连续停稳的部分/],
  ['tracking_pause_without_hold_evidence', /暂停记录与HOLD／未知模式证据不一致/],
]) {
  test(`wait rejection ${reason} is specific in the actual page`, () => {
    const source = transitionWait({frame_role: 'AMBIGUOUS_IDLE', policy_decision: 'keep',
      policy_reason: [reason], review_required: true, phase_pause_interpretation: {
        reason, pre_intent_evidence: reason === 'missing_pre_intent_context' ? null : {modes: ['XY_HOLD', 'XY_HOLD']},
        post_intent_evidence: reason === 'missing_post_intent_context' ? null : {modes: ['Z_OPEN', 'Z_OPEN']},
      }, pre_intent: null, post_intent: null});
    const rendered = renderPolicyFixture(source);
    assert.match(rendered.policyReason.textContent, expected);
    assert.match(rendered.policyWait.textContent, expected);
    assert.doesNotMatch(rendered.policyReason.textContent, /其他记录/);
    assert.match(rendered.policyRole.textContent, /自动保留 · 需人工复查/);
    if (reason === 'missing_pre_intent_context') assert.match(rendered.policyWait.textContent, /等待前：未找到可靠意图/);
    if (reason === 'missing_post_intent_context') assert.match(rendered.policyWait.textContent, /等待后：未找到可靠意图/);
  });
}

test('retained context after residual settling may exceed the static-wait frame budget', () => {
  const source = transitionWait(); source.phase_pause_interpretation.retained_frames = 24;
  const summary = metrics.waitSummary(source), rendered = renderPolicyFixture(source);
  assert.equal(summary.retainedFrameBudget, 10); assert.equal(summary.retainedFrames, 24);
  assert.match(rendered.policyWait.textContent, /保留预算：10 帧/);
  assert.match(rendered.policyWait.textContent, /实际保留 24 帧，可能另含残余收敛/);
});

test('diagnostic reason is shown even when the frame role remains an intent-transition hold', () => {
  const rendered = renderPolicyFixture(transitionWait({frame_role: 'INTENT_TRANSITION_HOLD',
    policy_decision: 'keep', policy_reason: ['accepted_intent_transition_context'],
    phase_pause_interpretation: {reason: 'unexplained_mode_excursion',
      pre_intent_evidence: {modes: ['XY_HOLD', 'XY_HOLD']},
      post_intent_evidence: {modes: ['Z_OPEN', 'Z_OPEN']}}}));
  assert.match(rendered.policyWait.textContent, /未压缩原因：等待中出现额外控制模式/);
});

test('wait details preserve valid zero values and never invent missing timing or budgets', () => {
  const source = transitionWait({phase_pause_interpretation: {retained_frame_budget: 0,
    retained_frames: 0, minimum_duration_s: 0}});
  const summary = metrics.waitSummary(source);
  assert.equal(summary.retainedFrameBudget, 0); assert.equal(summary.retainedFrames, 0);
  assert.equal(summary.minimumDurationSeconds, 0);
  assert.equal(metrics.waitSummary({frame_role: 'PREFIX_IDLE', phase_pause_interpretation: {phase: 'prefix'}}), null);
  const invalid = transitionWait({phase_pause_interpretation: {retained_frame_budget: NaN,
    retained_frames: -1, minimum_duration_s: Infinity}});
  const result = metrics.waitSummary(invalid);
  assert.equal(result.retainedFrameBudget, null); assert.equal(result.retainedFrames, null);
  assert.equal(result.minimumDurationSeconds, null);
  assert.doesNotMatch(result.description, /预算|毫秒|仿真时间|NaN|Infinity/);
});

test('frames without wait evidence hide the new wait-detail paragraph', () => {
  const rendered = renderPolicyFixture({frame_role: 'ACTIVE_MOTION', policy_decision: 'keep',
    policy_reason: ['state_or_action_changed']});
  assert.equal(rendered.policyWait.hidden, true);
  assert.equal(rendered.policyWait.textContent, '');
});


function heightRow(left = .388, right = .389, protectedValue = true, extra = {}) {
  return {raw_frame_index: 7, frame_role: protectedValue ? 'LOW_HEIGHT_PROTECTED' : 'SAME_INTENT_IDLE',
    policy_decision: 'keep', policy_reason: protectedValue ? ['both_actual_ee_below_height_threshold'] : [],
    height_protected: protectedValue, grasp_protected: false, release_protected: false,
    height_evidence: {available: true, threshold_m: .390, left_z_m: left, right_z_m: right,
      comparison: 'both_strictly_below'}, ...extra};
}

test('actual world-site heights show both millimetre values, strict AND and protected status', () => {
  const source = freezeDeep(heightRow()), original = JSON.stringify(source);
  const p = metrics.policySummary(source), rendered = renderPolicyFixture(source);
  assert.equal(p.heightProtected, true);
  assert.equal(p.height.leftMm, 388); assert.equal(p.height.rightMm, 389);
  assert.match(rendered.policyRole.textContent, /双侧末端低位保护/);
  assert.match(rendered.policyProtection.textContent, /左 388\.000 mm；右 389\.000 mm/);
  assert.match(rendered.policyProtection.textContent, /左、右都严格小于 390\.000 mm.*AND/);
  assert.match(rendered.policyProtection.textContent, /人工排除保护帧需在审核理由中解释/);
  assert.match(rendered.policyReason.textContent, /低位静止等待也保留/);
  assert.equal(JSON.stringify(source), original);
});

for (const [name, left, right] of [['left-only low', .380, .400], ['right-only low', .400, .380], ['equal threshold', .380, .390]]) {
  test(`height display does not turn ${name} into recorded protection`, () => {
    const p = metrics.policySummary(heightRow(left, right, false));
    const rendered = renderPolicyFixture(heightRow(left, right, false));
    assert.equal(p.heightProtected, false);
    assert.match(rendered.policyProtection.textContent, /当前高度保护：否/);
    assert.match(rendered.policyProtection.textContent, /只低一侧或等于阈值不满足/);
  });
}

test('missing actual height is not displayed as zero or reconstructed from a target', () => {
  const source = heightRow(null, .388, false, {frame_role:'AMBIGUOUS_IDLE', review_required:true,
    policy_reason:['missing_actual_ee_height'], action_recorded:Array(14).fill(0),
    height_evidence:{available:false, threshold_m:.390, left_z_m:null, right_z_m:.388}});
  const p = metrics.policySummary(source), rendered = renderPolicyFixture(source);
  assert.equal(p.height.available, false); assert.equal(p.height.leftMm, null);
  assert.match(rendered.policyProtection.textContent, /左 缺少记录；右 388\.000 mm/);
  assert.match(rendered.policyReason.textContent, /缺少左右实际末端高度依据/);
  assert.match(rendered.policyRole.textContent, /需人工复查/);
});

test('execution phase is diagnostic while historical grasp and release protection still render', () => {
  const source = heightRow(.42, .42, false, {execution_grasp_phase:true, execution_release_phase:true});
  const p = metrics.policySummary(source), rendered = renderPolicyFixture(source);
  assert.equal(p.grasp, false); assert.equal(p.release, false);
  assert.match(rendered.policyProtection.textContent, /抓取 是／释放 是（仅诊断，不代表长期保护）/);
  assert.doesNotMatch(rendered.policyProtection.textContent, /人工排除保护帧需/);
  const old = renderPolicyFixture({frame_role:'GRASP_PROTECTED',grasp_protected:true,release_protected:true,
    policy_decision:'keep',policy_reason:['grasp_protected']});
  assert.match(old.policyProtection.textContent, /抓取保护：是 · 释放保护：是/);
  assert.match(old.policyProtection.textContent, /人工排除需/);
});

test('verified terminal tail explains why low heights no longer protect an excluded frame', () => {
  const source = heightRow(.380,.381,false,{frame_role:'TAIL_IDLE',policy_decision:'exclude',
    policy_reason:['after_verified_release_tail_budget'],terminal_tail:true,
    terminal_completion_raw_frame:600,terminal_post_release_frames:3,terminal_tail_excluded:true,
    height_protection_overridden_by_terminal:true});
  const p=metrics.policySummary(source), rendered=renderPolicyFixture(source);
  assert.equal(p.heightProtected,false);assert.equal(p.terminal.excluded,true);
  assert.match(rendered.policyProtection.textContent,/低位保护由完成后尾段规则覆盖/);
  assert.match(rendered.policyTerminal.textContent,/完成这一帧保留，其后再保留 3 个原始帧/);
  assert.match(rendered.policyTerminal.textContent,/本帧已超出尾段预算，自动排除/);
  assert.match(rendered.policyReason.textContent,/超出完成后的短上下文/);
});

test('release context and completion evidence missing have separate visible explanations', () => {
  const context=renderPolicyFixture({frame_role:'RELEASE_TAIL_CONTEXT',policy_decision:'keep',
    policy_reason:['verified_release_post_context'],terminal_tail:true,
    terminal_completion_raw_frame:600,terminal_post_release_frames:3,terminal_tail_excluded:false});
  assert.match(context.policyRole.textContent,/释放完成后的短上下文/);
  assert.match(context.policyTerminal.textContent,/本帧属于完成后的保留上下文/);
  const missing=renderPolicyFixture({frame_role:'AMBIGUOUS_IDLE',policy_decision:'keep',
    review_required:true,policy_reason:['missing_actual_release_completion']});
  assert.equal(missing.policyTerminal.hidden,true);
  assert.match(missing.policyReason.textContent,/缺少真实释放完成依据/);
});

function renderEligibilityFixture(ep) {
  const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
  const html=fs.readFileSync(path.join(__dirname,'../../dataset/task3/review.html'),'utf8');
  const start=html.indexOf('function renderEligibility(ep){');
  const render=html.slice(start,html.indexOf('\nfunction renderEpisodeSummary()',start));
  assert.ok(start>=0&&html.includes('renderEligibility(ep);'),'The actual page must call eligibility rendering');
  const notice={textContent:'',hidden:false,classList:{toggle(name,on){notice[name]=on;}}};
  const quality={value:'pending',options:['pending','A','B','C','rejected'].map(value=>({value,disabled:false}))};
  vm.runInNewContext(`${render}\nrenderEligibility(ep);`,{M:metrics,ep,$:id=>id==='quality'?quality:notice});
  return {notice,quality};
}

test('user-excluded episode is clearly marked and cannot select A or B in the actual page', () => {
  const source=freezeDeep({quality_exclusion_reason:'用户明确排除399mm真实释放',training_candidate:false});
  const eligibility=metrics.trainingEligibility(source),{notice,quality}=renderEligibilityFixture(source);
  assert.equal(eligibility.excluded,true);assert.equal(notice.hidden,false);
  assert.match(notice.textContent,/用户已排除训练候选：用户明确排除399mm真实释放/);
  for(const option of quality.options)assert.equal(option.disabled,['A','B'].includes(option.value));
  assert.equal(quality.value,'pending');
});

for(const value of [undefined,null,'','  ']){
  test(`historical or nonexcluded manifest remains eligible without a reason (${String(value)})`,()=>{
    const {notice,quality}=renderEligibilityFixture({quality_exclusion_reason:value});
    assert.equal(notice.hidden,true);assert.equal(notice.textContent,'');
    assert.ok(quality.options.every(option=>option.disabled===false));
  });
}

test('new low-height fallback and historical blanket protection remain distinguishable',()=>{
  const source=heightRow(.38,.381,false,{height_evidence:{available:true,threshold_m:.39,
    left_z_m:.38,right_z_m:.381,fallback_required:false,physical_evidence_missing:[]}});
  const result=renderPolicyFixture(source);
  assert.match(result.policyProtection.textContent,/物理证据完整，低位本身不强制保留静止等待/);
  assert.match(result.policyProtection.textContent,/真实执行完整保护/);
  assert.match(metrics.heightSummary(heightRow()).description,/历史结果/);
  source.height_protected=true;source.frame_role='LOW_HEIGHT_PROTECTED';
  source.height_evidence.fallback_required=true;
  source.policy_reason=['both_actual_ee_low_missing_physical_evidence_fallback'];
  const missing=renderPolicyFixture(source);
  assert.match(missing.policyRole.textContent,/低位兜底保护/);
  assert.match(missing.policyReason.textContent,/关键物理证据缺失/);
});

test('virtual sequence display separates certified seams, hard boundaries and historical data',()=>{
  const source={frame_role:'SAME_INTENT_IDLE',keep:true,training_frame_index:21,
    training_time_s:.7,train_sequence_id:2,sequence_frame_index:5,stitch_gap_before:0,
    causal_target_valid:true};
  const safe=renderPolicyFixture(source);
  assert.match(safe.policySegments.textContent,/虚拟时间 0.700 s（不是原始仿真时间）/);
  assert.match(safe.policySegments.textContent,/缺口 0 已通过安全拼接核验/);
  const hard=renderPolicyFixture({...source,sequence_frame_index:0,causal_target_valid:false});
  assert.match(hard.policySegments.textContent,/硬边界.*不能跨越/);
  assert.match(hard.policySegments.textContent,/动作标签不可用/);
  assert.equal(metrics.sequenceSummary({}),null);
});

test('single-side intent evidence is displayed without inventing evidence for the other hand',()=>{
  const row=transitionWait({phase_pause_interpretation:{
    pre_intent_evidence:{modes:['XY_HOLD',null]},post_intent_evidence:{modes:['XY_HOLD',null]},
    comparable_sides:['left'],retained_frame_budget:3,retained_frames:3}});
  assert.match(metrics.waitSummary(row).description,/同侧前后可靠比较：左手；另一侧仍核对实际运动与执行/);
  assert.match(metrics.waitSummary(row).description,/右 无可靠记录/);
});
