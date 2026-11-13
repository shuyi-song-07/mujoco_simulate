/** Camera measurements are estimates, never robot/world absolute positions. */
const sides = ["left", "right"];
const clean = (n) => Number.isFinite(n) ? n : null;
const points = (a) => Array.isArray(a) && a.length === 21 ? a.map(p => [clean(p.x), clean(p.y), clean(p.z)]) : null;
const finitePoints = (a) => a?.length === 21 && a.every(p => p.length === 3 && p.every(Number.isFinite));
const median = (xs) => [...xs].sort((a, b) => a-b)[Math.floor(xs.length/2)];

export const DEFAULT_GESTURE_PARAMETERS = Object.freeze({
  mode_confirm_ms:200, wrist_still_ms:200, wrist_motion_threshold:.008,
  aperture_takeover_tolerance_m:.012, palm_score_min:.6, motion_score_min:.75,
  extended_angle_deg:150, flexed_angle_deg:130, extended_straightness:.80,
  flexed_straightness:.65, tip_mcp_ratio_min:1.1, min_bone_px:3,
  thumb_angle_deg:145, thumb_straightness:.78,
});

/** Rotation-invariant projected geometry. Unknown/foreshortened bones do not
 * become extension evidence; these engineering thresholds need human QA. */
export function fingerGeometry(image, width, height, parameters={}) {
  const p={...DEFAULT_GESTURE_PARAMETERS,...parameters}, fingers={};
  const chains={index:[5,6,7,8],middle:[9,10,11,12],ring:[13,14,15,16],little:[17,18,19,20],thumb:[1,2,3,4]};
  const complete=finitePoints(image)&&width>0&&height>0;
  const pixel=i=>[image[i][0]*width,image[i][1]*height];
  const d=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
  const angle=(a,b,c)=>{const u=[a[0]-b[0],a[1]-b[1]],v=[c[0]-b[0],c[1]-b[1]];const cosine=(u[0]*v[0]+u[1]*v[1])/(Math.hypot(...u)*Math.hypot(...v));return Math.acos(Math.max(-1,Math.min(1,cosine)))*180/Math.PI;};
  for(const [name,chain]of Object.entries(chains)) {
    const ps=complete?chain.map(pixel):null, lengths=ps?[d(ps[0],ps[1]),d(ps[1],ps[2]),d(ps[2],ps[3])]:[];
    const valid=Boolean(complete&&lengths.every(v=>v+1e-9>=p.min_bone_px));
    const pip=valid?angle(ps[0],ps[1],ps[2]):null,dip=valid?angle(ps[1],ps[2],ps[3]):null;
    const straight=valid?d(ps[0],ps[3])/lengths.reduce((a,b)=>a+b,0):null;
    const ratio=valid?d(ps[0],ps[3])/lengths[0]:null;
    const threshold=name==="thumb"?p.thumb_angle_deg:p.extended_angle_deg;
    const extended=valid&&pip+1e-9>=threshold&&dip+1e-9>=threshold&&straight+1e-9>=(name==="thumb"?p.thumb_straightness:p.extended_straightness)&&ratio+1e-9>=p.tip_mcp_ratio_min;
    const flexed=valid&&(pip<=p.flexed_angle_deg+1e-9||dip<=p.flexed_angle_deg+1e-9||straight<=p.flexed_straightness+1e-9);
    fingers[name]={pip_angle_deg:clean(pip),dip_angle_deg:clean(dip),straightness:clean(straight),tip_mcp_ratio:clean(ratio),valid,state:extended?"EXTENDED":flexed?"FLEXED":"UNKNOWN"};
  }
  const is=(name,state)=>fingers[name].state===state;
  return {method:"image-pixel-angles-v1",valid:Object.values(fingers).every(f=>f.valid),fingers,
    idle_extended:["middle","ring","little"].every(n=>is(n,"EXTENDED")),
    full_palm:Object.keys(chains).every(n=>is(n,"EXTENDED")),
    fist:["index","middle","ring","little"].every(n=>is(n,"FLEXED")),
    iloveyou:is("index","EXTENDED")&&is("little","EXTENDED")&&is("middle","FLEXED")&&is("ring","FLEXED")};
}

export function gestureProfile(hand, width, height, parameters={}) {
  const p={...DEFAULT_GESTURE_PARAMETERS,...parameters};
  const geometry=fingerGeometry(hand.image_landmarks_21,width,height,p);
  const score=hand.gesture_score??0;
  return {geometry, xy:hand.gesture==="ILoveYou"&&score>=p.motion_score_min&&geometry.iloveyou,
    fist:hand.gesture==="Closed_Fist"&&score>=p.motion_score_min&&geometry.fist,
    full_palm:hand.gesture==="Open_Palm"&&score>=p.palm_score_min&&geometry.full_palm,
    pinch:geometry.idle_extended};
}

export function handMeasurement(result, meta) {
  const width = meta.width, height = meta.height;
  const detections = (result.landmarks ?? []).map((lm, index) => {
    const image = points(lm), world = points(result.worldLandmarks?.[index]);
    const original = result.handedness?.[index]?.[0], gesture = result.gestures?.[index]?.[0];
    const rawLabel = original?.categoryName?.toLowerCase() ?? "unknown";
    const bound = meta.swapLabels && sides.includes(rawLabel) ? (rawLabel === "left" ? "right" : "left") : rawLabel;
    const complete = finitePoints(image), inFrame = complete && image.every(p => p[0] >= 0 && p[0] <= 1 && p[1] >= 0 && p[1] <= 1);
    // H03: fully in-frame landmarks can still collapse near a crop boundary.
    // Keep raw geometry for diagnosis; the inner 6% margin gates control only.
    const nearEdge = complete && image.some(p => p[0] < .06 || p[0] > .94 || p[1] < .06 || p[1] > .94);
    const d2 = (a, b) => Math.hypot((image[a][0]-image[b][0])*width, (image[a][1]-image[b][1])*height);
    const palm = complete ? d2(5, 17) : 0;
    const geometry = Boolean(inFrame && width > 0 && height > 0 && palm >= 12);
    const worldValid = Boolean(finitePoints(world));
    const d3 = (a, b) => Math.hypot(...world[a].map((v, j) => v-world[b][j]));
    const quality = clean(original?.score) ?? 0;
    return {raw_label: rawLabel, bound_side: bound, identity_score: quality,
      gesture: gesture?.categoryName ?? "None", gesture_score: clean(gesture?.score) ?? 0,
      image_landmarks_21: image, hand_center_landmarks_m_21: world,
      image_valid: geometry, world_valid: worldValid, valid: Boolean(geometry && !nearEdge && sides.includes(bound) && quality >= .8 && !meta.dropReason),
      invalid_reason: meta.dropReason ?? (!complete ? "incomplete_landmarks" : !inFrame ? "hand_out_of_frame" : nearEdge ? "hand_near_frame_edge" : palm < 12 ? "palm_too_small" : !sides.includes(bound) ? "unknown_handedness" : quality < .8 ? "identity_low" : null),
      palm_width_px: clean(palm), aperture_ratio_image: geometry ? d2(4, 8)/palm : null,
      aperture_ratio_world: worldValid && d3(5, 17) > .005 ? d3(4, 8)/d3(5, 17) : null,
      palm_roll_rad: geometry ? Math.atan2((image[9][0]-image[0][0])*width, -(image[9][1]-image[0][1])*height) : null,
      near_edge: nearEdge, finger_geometry:fingerGeometry(image,width,height,meta.gestureParameters)};
  });
  for (const side of sides) {
    const duplicates = detections.filter(d => d.bound_side === side);
    if (duplicates.length > 1) for (const d of duplicates) {d.valid = false; d.invalid_reason = "duplicate_identity";}
  }
  return {schema_version: "task3-hand-v1", measurement_id: meta.id,
    source_session: meta.session, source_control_epoch: meta.epoch,
    capture_browser_monotonic_ms: clean(meta.capturedAt), capture_epoch_ms: clean(meta.capturedEpochMs),
    recognition_epoch_ms: clean(meta.recognitionCompletedEpochMs), recognition_worker_monotonic_ms: clean(meta.workerCompletedAt),
    worker_time_origin_epoch_ms: clean(meta.workerTimeOriginMs),
    received_browser_monotonic_ms: clean(meta.receivedAt), inference_ms: clean(meta.inferenceMs),
    width, height, coordinate_semantics: {image: "xy normalized image; z relative model depth, not metres", hand_center: "model-estimated hand-centred metres; not camera/robot absolute coordinates"},
    accepted_recognition: !meta.dropReason, drop_reason: meta.dropReason ?? null, hands: detections};
}

export class ApertureCalibration {
  constructor() { this.reset(); }
  reset() {this.phase = "CLOSE"; this.samples = []; this.last = null; this.started = null; this.endpoints = {}; this.progress = 0; this.reason = "pinch_comfortably";}
  clearWindow() { this.samples = []; this.started = null; this.progress = 0; }
  update(measurement, bound) {
    const now = measurement.capture_browser_monotonic_ms;
    if (this.phase === "READY") return this.status();
    const pair = sides.map(side => measurement.hands.find(h => h.bound_side === side));
    if (!bound || !Number.isFinite(now) || (this.last !== null && (now <= this.last || now-this.last > 250)) || !pair.every(h => h?.valid)) {
      this.clearWindow(); this.last = now; this.reason = "fresh_valid_pair_required"; return this.status();
    }
    this.last = now;
    const values = pair.map(h => h.aperture_ratio_image);
    const eligible = this.phase === "CLOSE" ? values.every(v => v < .45) : values.every((v, i) => v > .8 && v-this.endpoints[sides[i]].closed > .4);
    if (!eligible) {this.clearWindow(); this.reason = this.phase === "CLOSE" ? "pinch_comfortably" : "open_comfortably"; return this.status();}
    this.started ??= now; this.samples.push(values);
    if (sides.some((_, i) => Math.max(...this.samples.map(v => v[i]))-Math.min(...this.samples.map(v => v[i])) > .12)) {
      this.clearWindow(); this.reason = "hold_aperture_steady"; return this.status();
    }
    this.progress = Math.min(1, (now-this.started)/1500);
    if (this.progress >= 1 && this.samples.length >= 20) {
      sides.forEach((s, i) => { this.endpoints[s] ??= {}; this.endpoints[s][this.phase === "CLOSE" ? "closed" : "open"] = median(this.samples.map(v => v[i])); });
      this.phase = this.phase === "CLOSE" ? "OPEN" : "READY"; this.clearWindow();
      this.reason = this.phase === "READY" ? "calibrated" : "open_comfortably";
    }
    return this.status();
  }
  normalize(side, ratio) {
    const e = this.endpoints[side];
    return this.phase === "READY" && Number.isFinite(ratio) ? Math.max(0, Math.min(1, (ratio-e.closed)/(e.open-e.closed))) : null;
  }
  status() { return {phase: this.phase, progress: this.progress, reason: this.reason, endpoints: structuredClone(this.endpoints)}; }
}
