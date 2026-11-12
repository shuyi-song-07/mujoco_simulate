/** Raw MediaPipe estimates, independent of single-arm control eligibility. */
const clean = (x) => Number.isFinite(x) ? x : null;
const points = (a) => Array.isArray(a) && a.length === 21
  ? a.map(p => [clean(p.x), clean(p.y), clean(p.z)]) : null;
const finite = (a) => a?.length === 21 && a.every(p => p.every(Number.isFinite));

export function handMeasurement(result, meta) {
  const width = meta.width, height = meta.height;
  const hands = (result.landmarks ?? []).map((lm, index) => {
    const image = points(lm), world = points(result.worldLandmarks?.[index]);
    const identity = result.handedness?.[index]?.[0], gesture = result.gestures?.[index]?.[0];
    const complete = finite(image);
    const inFrame = complete && image.every(p => p[0] >= 0 && p[0] <= 1 && p[1] >= 0 && p[1] <= 1);
    const palm = complete ? Math.hypot((image[5][0]-image[17][0])*width, (image[5][1]-image[17][1])*height) : null;
    const nearEdge = complete && image.some(p => p[0] < .06 || p[0] > .94 || p[1] < .06 || p[1] > .94);
    const valid = Boolean(inFrame && width > 0 && height > 0 && palm >= 12 && !meta.dropReason);
    return {detection_index:index, selected_for_control:index === 0,
      bound_side:index === 0 ? "control_hand" : null,
      raw_label:identity?.categoryName?.toLowerCase() ?? "unknown",
      identity_score:clean(identity?.score),
      gesture:gesture?.categoryName ?? "None", gesture_score:clean(gesture?.score),
      image_landmarks_21:image, hand_center_landmarks_m_21:world,
      image_valid:valid, world_valid:Boolean(finite(world)), valid,
      invalid_reason:meta.dropReason ?? (!complete ? "incomplete_landmarks" : !inFrame ? "hand_out_of_frame" : palm < 12 ? "palm_too_small" : null),
      near_edge:Boolean(nearEdge), palm_width_px:clean(palm),
      quality_semantics:"Observation-only flags; never change the existing control hand or motion gate"};
  });
  return {schema_version:"single-arm-hand-v1", measurement_id:meta.id,
    source_session:meta.session, source_control_epoch:meta.epoch,
    capture_browser_monotonic_ms:clean(meta.capturedAt), capture_epoch_ms:clean(meta.capturedEpochMs),
    recognition_epoch_ms:clean(meta.recognizedEpochMs), received_browser_monotonic_ms:clean(meta.receivedAt),
    inference_ms:clean(meta.inferenceMs), width, height,
    accepted_recognition:!meta.dropReason, drop_reason:meta.dropReason ?? null,
    coordinate_semantics:{image:"xy normalized image before display mirroring; z relative model depth, not metres",
      hand_center:"model-estimated hand-centred metres, not camera/robot/world absolute position"}, hands};
}
