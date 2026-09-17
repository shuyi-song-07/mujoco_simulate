export function mapPlanarMotion({
  gesture,
  screenDx,
  screenDy,
  sensitivity,
  carrying,
}) {
  const clamp = (value) => Math.min(1, Math.max(-1, value));
  // 摄像头画面是镜像的：画面中的右/上分别对应原始坐标的 -X/-Y。
  const visualRight = clamp(-screenDx / sensitivity);
  const visualForward = clamp(-screenDy / sensitivity);

  if (gesture === "ILoveYou" && carrying) {
    return { dx: visualForward, dy: -visualRight };
  }
  if (gesture === "Victory") {
    return { dx: 0, dy: visualRight };
  }
  if (gesture === "Thumb_Up") {
    return { dx: visualForward, dy: 0 };
  }
  return { dx: 0, dy: 0 };
}

export function shouldStartPlanarLock({ gesture, gestureChanged, carrying }) {
  const planarGestures = new Set(["Victory", "Thumb_Up", "ILoveYou"]);
  return planarGestures.has(gesture) && gestureChanged && carrying;
}
