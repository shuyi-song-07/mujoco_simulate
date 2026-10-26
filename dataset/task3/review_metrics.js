/* Pure display calculations. Nothing here approves, removes, or rewrites frames. */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.ReviewFrameMetrics = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const present = value => value !== null && value !== undefined;
  const DEGREES = 180 / Math.PI;
  const defaults = Object.freeze({position_m: 0.0001, rotation_rad: 0.001,
    joint_rad: 0.001, aperture_m: 0.0001});
  const reasonText = {
    comparable: '与上一原始帧比较', first_frame: '首帧没有上一帧',
    current_missing: '当前原始帧索引缺失', previous_missing: '上一原始帧索引缺失',
    duplicate_frame: '原始帧号重复，不能确定唯一前后帧',
    source_changed: '跨记录来源，不能比较', epoch_missing: '缺少控制会话编号，无法确认连续性',
    epoch_changed: '控制会话已变化，不能跨会话比较', time_missing: '缺少原始仿真时间，无法确认连续性',
    time_not_increasing: '原始仿真时间未递增，不能比较',
    time_gap: '原始仿真时间间隔超过审核配置，不能作为连续帧比较',
    next_missing: '缺少下一控制步目标；不以当前目标代替',
    next_epoch: '下一控制步目标的会话不匹配',
    next_timing: '下一控制步目标的时间对齐无效',
    next_tick: '下一控制步目标不是紧接当前快照的控制步'
  };
  function thresholds(options) {
    const supplied = options.stability || {};
    return Object.fromEntries(Object.entries(defaults).map(([key, fallback]) =>
      [key, finite(supplied[key]) && supplied[key] >= 0 ? supplied[key] : fallback]));
  }
  function maxGap(options, key, fallback) {
    const value = options.alignment?.[key];
    return finite(value) && value > 0 ? value : fallback;
  }
  function boundaryFor(current, previous, rawIndex, options, duplicate) {
    let reason = 'comparable', deltaSeconds = null;
    if (duplicate) reason = 'duplicate_frame';
    else if (!current) reason = 'current_missing';
    else if (rawIndex === 0) reason = 'first_frame';
    else if (!previous) reason = 'previous_missing';
    else if (['source_episode', 'session', 'episode_index'].some(key =>
      present(current[key]) && present(previous[key]) && current[key] !== previous[key])) reason = 'source_changed';
    else if (!present(current.control_epoch) || !present(previous.control_epoch)) reason = 'epoch_missing';
    else if (current.control_epoch !== previous.control_epoch) reason = 'epoch_changed';
    else if (!finite(current.original_time_s) || !finite(previous.original_time_s)) reason = 'time_missing';
    else {
      deltaSeconds = current.original_time_s - previous.original_time_s;
      if (deltaSeconds <= 0) reason = 'time_not_increasing';
      else if (deltaSeconds > maxGap(options, 'maximum_frame_gap_s', 0.05) + 1e-9) reason = 'time_gap';
    }
    return {comparable: reason === 'comparable', reason,
      description: reasonText[reason], deltaSeconds};
  }
  function actionFields(t) {
    return [
      ...['x', 'y', 'z'].map(key => ({key, label: key.toUpperCase(), unit: 'mm', scale: 1000,
        threshold: t.position_m, wrap: false})),
      ...[['roll', 'Roll'], ['pitch', 'Pitch'], ['yaw', 'Yaw']].map(([key, label]) =>
        ({key, label, unit: '°', scale: DEGREES, threshold: t.rotation_rad, wrap: true})),
      {key: 'aperture', label: '目标总爪宽', unit: 'mm', scale: 1000, threshold: t.aperture_m, wrap: false}
    ];
  }
  function stateFields(t) {
    return [...Array.from({length: 7}, (_, i) => ({key: `joint${i + 1}`, label: `关节 ${i + 1}`,
      unit: '°', scale: DEGREES, threshold: t.joint_rad, wrap: false})),
    {key: 'aperture', label: '实际总爪宽', unit: 'mm', scale: 1000, threshold: t.aperture_m, wrap: false}];
  }
  function wrapAngle(value) {
    return Math.atan2(Math.sin(value), Math.cos(value));
  }
  function compareVectors(current, previous, specs, comparable, expectedLength, offset) {
    const hasCurrent = Array.isArray(current) && current.length === expectedLength;
    const hasPrevious = Array.isArray(previous) && previous.length === expectedLength;
    const fields = specs.map((spec, i) => {
      const now = hasCurrent && finite(current[offset + i]) ? current[offset + i] : null;
      const before = hasPrevious && finite(previous[offset + i]) ? previous[offset + i] : null;
      const rawDelta = comparable && now !== null && before !== null
        ? (spec.wrap ? wrapAngle(now - before) : now - before) : null;
      const absolute = rawDelta === null ? null : Math.abs(rawDelta);
      const change = absolute === null ? 'unavailable' : absolute <= 1e-12 ? 'unchanged'
        : absolute > spec.threshold + 1e-12 ? 'changed' : 'tiny';
      return {key: spec.key, label: spec.label, unit: spec.unit,
        current: now === null ? null : now * spec.scale,
        previous: before === null ? null : before * spec.scale,
        delta: rawDelta === null ? null : rawDelta * spec.scale,
        threshold: spec.threshold * spec.scale, change};
    });
    const changedCount = fields.filter(f => f.change === 'changed').length;
    const tinyCount = fields.filter(f => f.change === 'tiny').length;
    const validCount = fields.filter(f => f.delta !== null).length;
    const summary = changedCount ? 'changed' : validCount !== fields.length ? 'unavailable'
      : tinyCount ? 'tiny' : 'unchanged';
    return {summary, changedCount, tinyCount, validCount, fields};
  }
  const gestureNames = {ILoveYou: 'ILoveYou · XY 手势', Open_Palm: '张掌',
    Closed_Fist: '握拳', Victory: 'V 手势', Thumb_Up: '竖拇指', Pointing_Up: '食指向上',
    None: '未知手势', Unknown: '未知手势', unknown: '未知手势'};
  function gestureSummary(row, side) {
    const aux = row?.auxiliary?.[side] || {}, status = row?.status_summary || row?.status || {};
    const categoryPolicy = row?.category_policy || null;
    const recordedWithoutScores = categoryPolicy === 'recorded_category_no_score_v1';
    const category = typeof aux.category === 'string' && aux.category.length ? aux.category : null;
    const unknown = ['None', 'Unknown', 'unknown'].includes(category);
    const trusted = category !== null && !unknown && aux.category_mask === true;
    const paused = status.paused === true;
    const reason = aux.reason || 'no_record';
    let description = '无当前手势记录；不能据此断定丢手';
    if (trusted) description = recordedWithoutScores
      ? '采用当前记录类别；不按手势／身份分数筛选，不代表高置信标签。'
      : '识别标签可信';
    else if (reason === 'measurement_unavailable_or_not_causal') description = '无新鲜有效测量，或测量与此帧时序不符';
    else if (reason === 'hand_untrusted') description = '本侧手部测量不可信，可能涉及身份、边缘或有效性';
    else if (unknown) description = '未知类别，不等于手移出镜头';
    else if (category !== null) description = recordedWithoutScores
      ? '记录了类别，但缺少当前因果测量或不是本实验的已知类别；不按分数筛选。'
      : '记录了类别，但未通过标签可信条件';
    return {category, label: category === null ? '无当前手势记录' : gestureNames[category] || category,
      score: finite(aux.category_score) ? aux.category_score : null,
      identityScore: finite(aux.identity_score) ? aux.identity_score : null,
      ageMs: finite(aux.age_ms) ? aux.age_ms : null,
      trust: trusted ? (recordedWithoutScores ? 'recorded' : 'trusted') : category === null ? 'missing' : 'untrusted',
      reason, description, supervision: trusted && (recordedWithoutScores || !paused), paused,
      categoryPolicy, recordedWithoutScores,
      globalHandLost: String(status.pause_reason || '').includes('hand_lost'),
      mode: row?.original_control_labels?.interaction_modes?.[side] ?? null};
  }
  function nextValidity(row, options) {
    const next = row?.next_control_target;
    if (!next || !Array.isArray(next.action)) return 'next_missing';
    if (!present(next.control_epoch) || next.control_epoch !== row.control_epoch) return 'next_epoch';
    if (!Number.isInteger(next.source_tick) || !Number.isInteger(row.source_control_tick)
        || next.source_tick !== row.source_control_tick + 1) return 'next_tick';
    const gap = next.sim_time - row.original_time_s;
    if (!finite(next.sim_time) || !finite(row.original_time_s) || gap <= 0 ||
        gap > maxGap(options, 'maximum_next_control_gap_s', 0.03) + 1e-9) return 'next_timing';
    return null;
  }
  function compareFrame(rows, rawIndex, options = {}) {
    // Always match raw n and n-1, regardless of sort order, keep flag, or segment.
    const matches = (Array.isArray(rows) ? rows : []).filter(row => row?.raw_frame_index === rawIndex);
    const prior = (Array.isArray(rows) ? rows : []).filter(row => row?.raw_frame_index === rawIndex - 1);
    const current = matches.length === 1 ? matches[0] : null;
    const previous = prior.length === 1 ? prior[0] : null;
    const boundary = boundaryFor(current, previous, rawIndex, options, matches.length > 1 || prior.length > 1);
    const t = thresholds(options), actionSource = options.actionSource === 'next_control' ? 'next_control' : 'recorded';
    const actionThresholds = {...t}, stateThresholds = {...t};
    const selection = options.selection?.thresholds || {};
    for (const [target, name, key] of [
      [actionThresholds, 'position_m', 'action_position_m'],
      [actionThresholds, 'rotation_rad', 'action_rotation_rad'],
      [actionThresholds, 'aperture_m', 'action_aperture_m'],
      [stateThresholds, 'joint_rad', 'state_joint_rad'],
      [stateThresholds, 'aperture_m', 'state_aperture_m']]) {
      if (finite(selection[key]) && selection[key] > 0) target[name] = selection[key];
    }
    const actionNow = actionSource === 'recorded' ? current?.action_recorded : current?.next_control_target?.action;
    const actionBefore = actionSource === 'recorded' ? previous?.action_recorded : previous?.next_control_target?.action;
    const nextIssue = actionSource === 'next_control'
      ? nextValidity(current, options) || (previous ? nextValidity(previous, options) : null) : null;
    const actionComparable = boundary.comparable && !nextIssue;
    const action = {}, state = {};
    ['left', 'right'].forEach((side, i) => {
      action[side] = compareVectors(actionNow, actionBefore, actionFields(actionThresholds), actionComparable, 14, i * 7);
      state[side] = compareVectors(current?.state_recorded, previous?.state_recorded,
        stateFields(stateThresholds), boundary.comparable, 16, i * 8);
    });
    return {current, previous, boundary, thresholds: t, actionThresholds, stateThresholds, actionSource,
      actionIssue: nextIssue ? {reason: nextIssue, description: reasonText[nextIssue]} : null,
      action, state, gestures: {left: gestureSummary(current, 'left'), right: gestureSummary(current, 'right')}};
  }
  const roleNames = Object.freeze({
    ACTIVE_MOTION: '实际运动或目标变化', PREFIX_IDLE: '任务开始前等待',
    TAIL_IDLE: '完成并稳定后等待', SAME_INTENT_IDLE: '相同意图下的冗余等待',
    RECENTER_IDLE: '回中／追踪恢复等待',
    INTENT_TRANSITION_IDLE: '换模式期间静止等待',
    LOW_HEIGHT_PROTECTED: '双侧末端低位保护',
    RELEASE_TAIL_CONTEXT: '释放完成后的短上下文',
    INTENT_TRANSITION_HOLD: '控制意图转换', PRE_GRASP_HOLD: '抓取前观察',
    GRASP_PROTECTED: '抓取执行／验证保护', PRE_RELEASE_HOLD: '释放前观察',
    GRASP_EXECUTION: '真实抓取行程／接触验证', RELEASE_EXECUTION: '真实释放行程／落稳验证',
    RELEASE_PROTECTED: '释放执行／落稳保护', SEMANTIC_PHYSICAL_HOLD: '接触或物理事件',
    AMBIGUOUS_IDLE: '证据不足，保留复查', INVALID: '数据无效，不能人工保留'
  });
  const intentNames = Object.freeze({XY_HOLD:'XY移动，保持爪宽',XY:'XY移动',
    Z_OPEN:'Z升降／张掌模式',Z_CLOSE:'Z升降／握拳模式',
    Z_HOLD:'Z升降，保持爪宽',HOLD:'保持／中性状态',PINCH:'捏合模式'});
  const waitReasonNames = Object.freeze({
    after_verified_release_tail_budget: '已确认真实释放、物体放稳及任务成功；超出完成后的短上下文，裁剪尾部',
    verified_release_post_context: '已确认真实释放完成，保留配置指定的少量后续原始帧',
    verified_bilateral_release_complete: '双侧真实释放、物体放稳及任务成功已确认，保留完成这一帧',
    missing_actual_release_completion: '仅有任务完成标签，缺少真实释放完成依据，尾部继续保留复查',
    both_actual_ee_below_height_threshold: '左右实际末端世界Z都严格低于高度阈值，整帧保留；低位静止等待也保留',
    both_actual_ee_low_missing_physical_evidence_fallback: '关键物理证据缺失且双侧实际末端低于阈值，启用低位兜底保护；证据完整的低位静止仍可压缩',
    complete_physical_execution_phase: '从真实开闭爪执行到实际物理验证结束，完整保留',
    missing_physical_or_execution_evidence: '关键物理或执行授权证据缺失，保留待审',
    missing_actual_ee_height: '缺少左右实际末端高度依据，不能应用高度规则，保留待复查',
    stationary_intent_transition: '前后控制模式不同，但等待中没有实际任务动作；裁剪已稳定部分，保留必要上下文',
    tracking_pause_without_hold_evidence: '暂停记录与HOLD／未知模式证据不一致，保留复查',
    safety_pause_requires_review: '存在安全暂停，需人工复查',
    incomplete_source_evidence: '来源记录不完整，无法核验这段等待，保留复查',
    missing_pre_intent_context: '搜索范围内未找到等待前可比较的可靠稳定意图；保留待复查',
    missing_post_intent_context: '搜索范围内未找到等待后可比较的可靠稳定意图；保留待复查',
    unexplained_mode_excursion: '等待中出现额外控制模式，或切到新模式后又返回旧模式；保留转换过程',
    physical_event_in_wait_context: '等待前后关联范围存在抓放执行、取消、接触变化或抓放保护；不能按纯等待删除',
    no_verified_stationary_core: '没有找到可确认连续停稳的部分；可能仍在收敛或缺少物理依据，继续保留'
  });
  function waitSummary(row) {
    const context = row?.phase_pause_interpretation;
    if (!context || typeof context !== 'object') return null;
    const describePair = pair => Array.isArray(pair) && pair.length === 2
      ? `左 ${intentNames[pair[0]] || pair[0] || '无可靠记录'}；右 ${intentNames[pair[1]] || pair[1] || '无可靠记录'}`
      : '未找到可靠意图';
    const before = context.pre_intent_evidence?.modes ?? row.pre_intent;
    const after = context.post_intent_evidence?.modes ?? row.post_intent;
    const budget = Number.isInteger(context.retained_frame_budget) && context.retained_frame_budget >= 0
      ? context.retained_frame_budget : null;
    const retained = Number.isInteger(context.retained_frames) && context.retained_frames >= 0
      ? context.retained_frames : null;
    const minimum = finite(context.minimum_duration_s) && context.minimum_duration_s >= 0
      ? context.minimum_duration_s : null;
    const diagnostic = typeof context.reason === 'string' ? context.reason : null;
    const hasIntents = present(before) || present(after) || diagnostic !== null;
    const lines = [];
    if (hasIntents) lines.push(`等待前：${describePair(before)}。等待后：${describePair(after)}。`);
    if (Array.isArray(context.comparable_sides) && context.comparable_sides.length)
      lines.push(`同侧前后可靠比较：${context.comparable_sides.map(s => s === 'left' ? '左手' : s === 'right' ? '右手' : s).join('、')}；另一侧仍核对实际运动与执行。`);
    if (budget !== null) lines.push(`稳定等待保留预算：${budget} 帧（不是毫秒）${retained !== null
      ? `；本停滞区间实际保留 ${retained} 帧，可能另含残余收敛或必要上下文` : ''}。`);
    else if (retained !== null) lines.push(`本停滞区间实际保留 ${retained} 帧。`);
    if (minimum !== null) lines.push(`连续稳定部分至少 ${minimum} 秒仿真时间才考虑压缩。`);
    if (diagnostic) lines.push(`未压缩原因：${waitReasonNames[diagnostic] || diagnostic}。`);
    return lines.length ? {before: describePair(before), after: describePair(after),
      retainedFrameBudget: budget, retainedFrames: retained, minimumDurationSeconds: minimum,
      diagnosticReason: diagnostic, description: lines.join(' ')} : null;
  }
  function heightSummary(row) {
    const evidence = row?.height_evidence;
    if (!evidence || typeof evidence !== 'object') return null;
    const mm = value => finite(value) ? value * 1000 : null;
    const left = mm(evidence.left_z_m), right = mm(evidence.right_z_m), threshold = mm(evidence.threshold_m);
    const available = evidence.available === true && left !== null && right !== null && threshold !== null;
    const value = number => number === null ? '缺少记录' : `${number.toFixed(3)} mm`;
    const fallbackPolicy = typeof evidence.fallback_required === 'boolean';
    const policy = fallbackPolicy
      ? (evidence.fallback_required ? '当前规则：关键物理证据缺失时才启用双侧低位兜底。' : '当前规则：物理证据完整，低位本身不强制保留静止等待。')
      : '历史结果：使用当时双侧低位整帧保护规则。';
    const rule = threshold === null ? '高度阈值缺少记录，无法核验。'
      : `规则：左、右都严格小于 ${value(threshold)} 才满足低位条件（AND）；只低一侧或等于阈值不满足。`;
    return {available, protected: row.height_protected === true, fallbackPolicy,
      leftMm: left, rightMm: right, thresholdMm: threshold,
      description: `左右末端实测世界 Z：左 ${value(left)}；右 ${value(right)}。${rule} ${policy} 当前高度保护：${row.height_protected === true ? '是' : '否'}。${row.height_protection_overridden_by_terminal === true ? '已确认释放完成，低位保护由完成后尾段规则覆盖。' : ''}${available ? '' : '高度证据缺失，不能当作零高度。'}`};
  }
  function sequenceSummary(row) {
    if (!Number.isInteger(row?.training_frame_index) || row.training_frame_index < 0 ||
        !Number.isInteger(row?.train_sequence_id) || row.train_sequence_id < 0 ||
        !Number.isInteger(row?.sequence_frame_index) || row.sequence_frame_index < 0 ||
        !finite(row?.training_time_s)) return null;
    const gap = Number.isInteger(row.stitch_gap_before) && row.stitch_gap_before >= 0;
    const seam = !gap ? '' : row.sequence_frame_index === 0
      ? `此前缺口 ${row.stitch_gap_before} 为硬边界，未来动作块不能跨越。`
      : `此前缺口 ${row.stitch_gap_before} 已通过安全拼接核验。`;
    return {sequenceId:row.train_sequence_id,frameIndex:row.training_frame_index,timeSeconds:row.training_time_s,
      safeStitch:gap && row.sequence_frame_index>0,hardBoundary:gap && row.sequence_frame_index===0,
      description:`虚拟训练序列 ${row.train_sequence_id} · 保留帧 ${row.training_frame_index} · 虚拟时间 ${row.training_time_s.toFixed(3)} s（不是原始仿真时间）。${seam}${row.causal_target_valid===false?'此观察缺少合法下一控制步目标，动作标签不可用。':''} 人工区间修改后需在 Prepared 导出时重新核验。`};
  }
  function terminalSummary(row) {
    if (!Number.isInteger(row?.terminal_completion_raw_frame) || row.terminal_completion_raw_frame < 0) return null;
    const count = Number.isInteger(row.terminal_post_release_frames) && row.terminal_post_release_frames >= 0
      ? row.terminal_post_release_frames : null;
    const action = row.terminal_tail_excluded === true ? '本帧已超出尾段预算，自动排除。'
      : row.terminal_tail === true ? '本帧属于完成后的保留上下文。' : '这是释放完成确认帧或其之前的帧。';
    return {completionRawFrame: row.terminal_completion_raw_frame, postReleaseFrames: count,
      excluded: row.terminal_tail_excluded === true,
      description: `真实释放完成确认：原始帧 ${row.terminal_completion_raw_frame}。${count === null ? '' : `完成这一帧保留，其后再保留 ${count} 个原始帧（不是毫秒）。`}${action}`};
  }
  function trainingEligibility(episode) {
    const reason = typeof episode?.quality_exclusion_reason === 'string'
      ? episode.quality_exclusion_reason.trim() : '';
    return {excluded: reason.length > 0, reason,
      label: reason ? `用户已排除训练候选：${reason}。不能标为 A / B；仍可查看原片并保存待审核或排除意见。` : ''};
  }
  function policySummary(row) {
    if (!row?.frame_role) return null;
    const mode = value => value === null || value === undefined ? '无可靠记录' : intentNames[value] || value;
    const reasons = Array.isArray(row.policy_reason) ? row.policy_reason : [row.policy_reason].filter(Boolean);
    return {
      role: row.frame_role, label: row.frame_role==='LOW_HEIGHT_PROTECTED' && typeof row.height_evidence?.fallback_required==='boolean'
        ? '物理证据不足的双侧低位兜底保护' : roleNames[row.frame_role] || row.frame_role,
      automaticKeep: row.policy_decision ? row.policy_decision === 'keep' : row.keep === true,
      left: mode(row.accepted_left_intent), right: mode(row.accepted_right_intent),
      stagnantRun: Number.isInteger(row.stagnant_run_id) ? row.stagnant_run_id : null,
      grasp: row.grasp_protected === true, release: row.release_protected === true,
      heightProtected: row.height_protected === true, height: heightSummary(row),
      executionGrasp: row.execution_grasp_phase === true, executionRelease: row.execution_release_phase === true,
      reviewRequired: row.review_required === true, invalid: row.frame_role === 'INVALID',
      reasons, wait: waitSummary(row), terminal: terminalSummary(row), sequence: sequenceSummary(row), taskSegment: row.task_segment_id ?? null,
      fragment: row.contiguous_fragment_id ?? row.segment_id ?? null,
      softGap: row.soft_curation_gap === true
    };
  }
  return Object.freeze({compareFrame, compareVectors, gestureSummary, wrapAngle, defaults,
    policySummary, heightSummary, terminalSummary, sequenceSummary, trainingEligibility, waitSummary, waitReasonNames, roleNames, intentNames});
});
