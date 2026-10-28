"""Loopback-only, read-only-video review UI for a frozen curation manifest.

Only reviews.json is persisted. Cut previews use disposable viewing copies.
Intervals are half-open [start, end); source videos and indexes stay unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import socket
import tempfile
import threading
from urllib.parse import urlsplit

from dataset.task3.review_preview import PreviewCache, preview_plan


REVIEW_VERSION = "papera-curation-review-v1"
PAGE_VERSION = "papera-cut-preview-v1"
QUALITIES = {"pending", "A", "B", "C", "rejected"}
CURRENT_POLICY = "papera-intent-aware"


class ReadOnlyReviewError(PermissionError):
    """Historical outputs can be inspected but never revised by this service."""


def validate_frame_overrides(rows, review):
    """Validate human overrides against immutable automatic frame evidence.

    This is shared by HTTP saves, draft previews and Prepared export. It does
    not change a row or infer a human grade from an automatic decision.
    """
    excludes = review.get("exclude_intervals", [])
    keeps = review.get("keep_intervals", [])
    for row in rows:
        raw = row["raw_frame_index"]
        invalid = (row.get("frame_role") == "INVALID"
                   or "invalid_numeric" in row.get("reason", []))
        if invalid and any(a <= raw < b for a, b in keeps):
            raise ValueError(f"原始帧 {raw} 为 INVALID / invalid numeric，人工保留不能覆盖数据错误")
        protected = (row.get("height_protected") is True or row.get("grasp_protected") is True
                     or row.get("release_protected") is True)
        if protected and any(a <= raw < b for a, b in excludes) and not review.get("notes", "").strip():
            raise ValueError(f"排除低位／抓取／释放保护帧 {raw}，必须在审核理由中说明原因")


def service_identity():
    elevated = None
    if os.name == "nt":
        import ctypes
        elevated = bool(ctypes.windll.shell32.IsUserAnAdmin())
    return {"pid": os.getpid(), "elevated": elevated, "version": "papera-review-service-v2"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def review_binding(manifest, manifest_path=None):
    """Bind human decisions to exactly one prepared source/configuration view."""
    return {
        "manifest_sha256": (hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
                            if manifest_path else _digest(manifest)),
        "run_id": manifest.get("run_id"),
        "source_root": str(Path(manifest["source_root"]).resolve()),
        "config_sha256": _digest(manifest.get("config", {})),
        "source_fingerprints_sha256": manifest.get("source_fingerprints_sha256") or _digest(manifest.get("source_fingerprints", [
            {"key": ep["key"], "fingerprints": ep.get("fingerprints", ep.get("source_hashes"))}
            for ep in manifest["episodes"]])),
    }


def _frame_count(episode):
    metrics = episode.get("metrics", {})
    for value in (metrics.get("raw_frames"), metrics.get("frame_count"),
                  metrics.get("frames"), episode.get("frame_count"), episode.get("frames")):
        if type(value) is int and value >= 0:
            return value
    raise ValueError("manifest 缺少非负整数 raw_frames/frame_count，不能确认裁剪边界")


def _intervals(value, count):
    if not isinstance(value, list) or len(value) > 500:
        raise ValueError("区间必须是列表，最多 500 段")
    result = []
    for pair in value:
        if (not isinstance(pair, list) or len(pair) != 2 or
                any(type(n) is not int for n in pair) or not 0 <= pair[0] < pair[1] <= count):
            raise ValueError(f"区间必须满足 0 ≤ 起始帧 < 结束帧 ≤ {count}，结束帧不包含")
        result.append(pair[:])
    result.sort()
    if any(left[1] > right[0] for left, right in zip(result, result[1:])):
        raise ValueError("同类区间不能重叠")
    return result


def validate_review(review, episode):
    """Validate editable decisions; never infer success from a human grade."""
    if not isinstance(review, dict) or review.get("quality") not in QUALITIES:
        raise ValueError("未知质量级别")
    allowed = {"quality", "reviewer", "watched_full", "notes", "exclude_intervals",
               "keep_intervals", "reviewed_at"}
    if set(review) - allowed:
        raise ValueError("审核含有未知字段；不能手动设置 training_ready 或结构通过")
    result = {"quality": review["quality"]}
    for field, limit in (("reviewer", 120), ("notes", 8000)):
        text = review.get(field, "")
        if not isinstance(text, str) or len(text) > limit:
            raise ValueError(f"{field} 必须是至多 {limit} 字符的文本")
        result[field] = text.strip()
    if type(review.get("watched_full", False)) is not bool:
        raise ValueError("watched_full 必须是布尔值")
    result["watched_full"] = review.get("watched_full", False)
    if result["quality"] != "pending" and not result["reviewer"]:
        raise ValueError("正式分级需要审核者姓名")
    if result["quality"] in {"A", "B"} and not result["watched_full"]:
        raise ValueError("A/B 分级需要完整观看确认；未看完请保存为待审核")
    if result["quality"] in {"C", "rejected"} and not result["notes"]:
        raise ValueError("异常或排除记录必须填写审核理由")
    if result["quality"] in {"A", "B"} and episode.get("structural_pass") is not True:
        raise ValueError("结构检查未通过，不能标为 A/B；人工确认不能绕过结构错误")
    exclusion_reason = episode.get("quality_exclusion_reason")
    if result["quality"] in {"A", "B"} and isinstance(exclusion_reason, str) and exclusion_reason.strip():
        raise ValueError(f"用户已排除训练候选，不能标为 A/B：{exclusion_reason.strip()}")
    count = _frame_count(episode)
    for field in ("exclude_intervals", "keep_intervals"):
        result[field] = _intervals(review.get(field, []), count)
    if any(max(a[0], b[0]) < min(a[1], b[1])
           for a in result["exclude_intervals"] for b in result["keep_intervals"]):
        raise ValueError("人工排除区间与必要保留区间相互冲突")
    reviewed_at = review.get("reviewed_at")
    if reviewed_at is not None:
        if not isinstance(reviewed_at, str):
            raise ValueError("reviewed_at 必须是 ISO 时间")
        try:
            parsed = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("reviewed_at 必须是 ISO 时间") from exc
        if parsed.tzinfo is None:
            raise ValueError("审核时间必须包含时区")
        result["reviewed_at"] = reviewed_at
    return result


def validate_reviews(document, manifest, manifest_path=None):
    """Import helper: reject stale reviews or decisions for another manifest."""
    if not isinstance(document, dict) or document.get("version") != REVIEW_VERSION:
        raise ValueError("审核文件版本不匹配")
    if document.get("binding") != review_binding(manifest, manifest_path):
        raise ValueError("审核绑定不匹配：原始清单、配置或来源指纹已经变化，请重新审核")
    episodes = {ep["key"]: ep for ep in manifest["episodes"]}
    if len(episodes) != len(manifest["episodes"]):
        raise ValueError("manifest 存在重复 episode key")
    decisions = document.get("episodes")
    if not isinstance(decisions, dict) or set(decisions) - set(episodes):
        raise ValueError("审核含有不属于当前清单的 episode")
    if type(document.get("revision")) is not int or document["revision"] < 0:
        raise ValueError("审核 revision 非法")
    normalized = {}
    for key, review in decisions.items():
        normalized[key] = validate_review(review, episodes[key])
        if review.get("quality") != "pending" and not review.get("reviewed_at"):
            raise ValueError("正式分级缺少审核时间")
    return dict(document) | {"episodes": normalized}


def load_reviews(output_dir, manifest=None):
    output = Path(output_dir).resolve()
    manifest_path = output / "manifest.json"
    manifest = manifest or json.loads(manifest_path.read_text(encoding="utf-8"))
    path = output / "reviews.json"
    if path.is_symlink() or path.resolve().parent != output:
        raise ValueError("reviews.json 不能指向输出目录外")
    if path.exists():
        return validate_reviews(json.loads(path.read_text(encoding="utf-8")), manifest, manifest_path)
    return {"version": REVIEW_VERSION, "binding": review_binding(manifest, manifest_path),
            "revision": 0, "episodes": {}}


def _inside(path, root):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("文件路径超出允许目录")
    return resolved


REVIEW_CAMERAS = ("top", "front", "side")
VIDEO_PREFIX = "observation.images."


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def linked_review_sources(derived_root, original_root, episode):
    """Resolve original three-camera review media for any indexed episode."""
    from dataset.task3.review_sources import review_sources
    return review_sources(derived_root, int(episode["episode_index"]),
                          expected_frames=_frame_count(episode), override_root=original_root)


class ReviewStore:
    def __init__(self, output_dir, *, read_only=None, review_source=None, frozen_preview=False):
        self.output = Path(output_dir).resolve()
        self.manifest_path = self.output / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if read_only is not None and type(read_only) is not bool:
            raise ValueError("read_only 必须为布尔值或 None")
        self.frozen_preview = bool(frozen_preview)
        self.read_only = (self.frozen_preview or self.manifest.get("selection_policy") != CURRENT_POLICY
                          or read_only is True)
        self.document = load_reviews(self.output, self.manifest)
        self.source_root = Path(self.manifest["source_root"]).resolve()
        self.episodes = self.manifest["episodes"]
        if len({ep["key"] for ep in self.episodes}) != len(self.episodes):
            raise ValueError("manifest 存在重复 episode key")
        self.media = {}
        self.video_urls = []
        self.review_video_sources = []
        self.indexes = []
        review_root = Path(review_source).resolve() if review_source is not None else None
        for number, episode in enumerate(self.episodes):
            dataset = _inside(episode["dataset_root"], self.source_root)
            from dataset.task3.review_sources import provenance_path
            if review_root is not None or provenance_path(dataset) is not None:
                original = review_root
                if original is not None and not (original / "meta/info.json").is_file():
                    original = original / dataset.name
                sources = linked_review_sources(dataset, original, episode)
            else:
                from dataset.task3.review_sources import select_review_sources
                sources = select_review_sources(episode.get("review_video_sources") or episode.get("video_sources", []))
            if not sources:
                raise ValueError("Review requires top/front/oblique, historical top/front/side, or a native oblique-only recording; linked derivatives require their original views")
            videos = []
            registered = []
            for camera_no, source in enumerate(sources):
                external = review_root is not None or provenance_path(dataset) is not None
                allowed_root = Path(source.get("allowed_root", self.source_root) if external else self.source_root).resolve()
                raw_path = Path(source["path"])
                path = _inside(raw_path if raw_path.is_absolute() else dataset / raw_path, allowed_root)
                if not path.is_file() and episode.get("structural_pass") is not True:
                    continue
                if path.suffix.lower() != ".mp4" or not path.is_file():
                    raise ValueError(f"清单视频不存在或不是 mp4：{path.name}")
                media_id = f"{number}-{camera_no}"
                self.media[media_id] = (path, allowed_root)
                registered.append(dict(source) | {"path": str(path)})
                videos.append({k: v for k, v in source.items() if k != "path"} |
                              {"url": f"/media/{media_id}"})
            self.video_urls.append(videos)
            self.review_video_sources.append(registered)
            index_path = Path(episode["index_path"])
            self.indexes.append(_inside(index_path if index_path.is_absolute() else self.output / index_path,
                                        self.output))
        self.lock = threading.RLock()
        self.token = secrets.token_urlsafe(32)
        self.previews = PreviewCache()

    def manifest_view(self):
        return {"version": self.manifest.get("version"), "run_id": self.manifest.get("run_id"),
                "selection_policy": self.manifest.get("selection_policy"),
                "read_only": self.read_only, "frozen_preview": self.frozen_preview, "training_ready": False,
                "training_compatibility": "export_prepared_then_verified_safe_stitch_pack",
                "service": service_identity() | {"read_only": self.read_only, "training_ready": False},
                "binding": self.document["binding"], "revision": self.document["revision"],
                "frame_review": {"version": PAGE_VERSION,
                                 "selection": self.manifest.get("config", {}).get("sa",
                                              self.manifest.get("config", {}).get("selection")),
                                 "stability": self.manifest.get("config", {}).get("physical_motion",
                                              self.manifest.get("config", {}).get("stability", {})),
                                 "alignment": self.manifest.get("config", {}).get("alignment", {}),
                                 "nominal_fps": self.manifest.get("config", {}).get("fps", 30)},
                "episodes": [{k: v for k, v in ep.items() if k not in
                              {"dataset_root", "video_sources", "review_video_sources", "index_path", "fingerprints", "source_hashes"}} |
                             {"videos": self.video_urls[n], "review": self.document["episodes"].get(ep["key"])}
                             for n, ep in enumerate(self.episodes)]}

    def index(self, number):
        if not 0 <= number < len(self.indexes):
            raise KeyError("episode 不存在")
        path = _inside(self.indexes[number], self.output)
        expected = self.episodes[number].get("index_sha256")
        content = path.read_bytes()
        if expected and hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("原始自动索引指纹已变化，请重新生成版本；不能继续审核旧来源")
        rows = []
        for line in content.decode("utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def save(self, payload):
        if self.read_only:
            raise ReadOnlyReviewError("历史版本只读：可以查看原片和已有审核，不能保存；请用当前流程生成新版本")
        with self.lock:
            if payload.get("binding") != self.document["binding"]:
                raise ValueError("页面与当前清单不匹配，请刷新页面")
            if review_binding(self.manifest, self.manifest_path) != self.document["binding"]:
                raise ValueError("清单已被重新生成，请重启审核服务")
            if payload.get("revision") != self.document["revision"]:
                raise RuntimeError("另一页面已修改审核结果，请刷新后再保存")
            saved = self.output / "reviews.json"
            if saved.exists() and load_reviews(self.output, self.manifest) != self.document:
                raise RuntimeError("审核文件已由其他进程修改，请重启审核服务")
            key = payload.get("episode_key")
            episode = next((ep for ep in self.episodes if ep["key"] == key), None)
            if episode is None:
                raise ValueError("episode 不属于当前清单")
            review = validate_review(payload.get("review"), episode)
            sequence_validation = None
            overrides = review["exclude_intervals"] or review["keep_intervals"]
            if overrides or episode.get("structural_pass") is True:
                rows = self.index(self.episodes.index(episode))
                if overrides and (len(rows) != _frame_count(episode)
                                  or any(row.get("raw_frame_index") != n for n, row in enumerate(rows))):
                    raise ValueError("原始帧索引不完整，不能验证人工区间")
                validate_frame_overrides(rows, review)
                if self.manifest.get('sequence_contract'):
                    from dataset.task3.prepare_paper_dataset import apply_review
                    from dataset.task3.sequence_stitching import build_sequence_contract
                    sequence_validation = build_sequence_contract(
                        apply_review(rows, review), self.manifest['config'])
                    sequence_validation.pop('rows')
            review["reviewed_at"] = datetime.now(timezone.utc).isoformat()
            document = dict(self.document)
            document["episodes"] = dict(document["episodes"]) | {key: review}
            document["revision"] += 1
            target = self.output / "reviews.json"
            if target.is_symlink() or target.resolve().parent != self.output:
                raise ValueError("拒绝写入指向输出目录外的审核文件")
            descriptor, temporary = tempfile.mkstemp(prefix=".reviews-", suffix=".tmp", dir=self.output)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                    json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
            finally:
                if Path(temporary).exists():
                    Path(temporary).unlink()
            self.document = document
            return {"revision": document["revision"], "review": review,
                    'sequence_validation': sequence_validation}

    def preview(self, payload):
        if self.read_only and not self.frozen_preview:
            raise ReadOnlyReviewError("历史版本只读：不能生成新的裁剪预览，请用当前流程生成新版本")
        if (payload.get("binding") != self.document["binding"] or
                review_binding(self.manifest, self.manifest_path) != self.document["binding"]):
            raise ValueError("预览来源清单已变化，请重新打开审核服务")
        number = next((n for n, ep in enumerate(self.episodes) if ep["key"] == payload.get("episode_key")), None)
        if number is None:
            raise ValueError("预览记录不属于当前清单")
        episode = self.episodes[number]
        if episode.get("structural_pass") is not True:
            raise ValueError("结构检查尚未通过，无法保证裁剪预览对齐；请先检查原片及结构问题")
        # Draft cuts are allowed, but generating a preview must not approve or
        # save a review, regardless of the grade currently selected in the UI.
        review = validate_review({"quality": "pending", "exclude_intervals": payload.get("exclude_intervals", []),
                                  "keep_intervals": payload.get("keep_intervals", []),
                                  "notes": payload.get("notes", "")}, episode)
        rows = self.index(number)
        validate_frame_overrides(rows, review)
        if self.frozen_preview and (review['exclude_intervals'] or review['keep_intervals']):
            raise ValueError('只读裁剪预览不接受人工删留区间')
        plan = preview_plan(rows, review, _frame_count(episode),
                            float(self.manifest.get("config", {}).get("fps", 30)),
                            config=self.manifest['config'] if self.manifest.get('sequence_contract') and not self.frozen_preview else None,
                            frozen=self.frozen_preview)
        sources = self.review_video_sources[number]
        return self.previews.create(plan, sources)


def _byte_range(header, length):
    if header is None:
        return 0, length - 1, False
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
    if not match or not any(match.groups()) or length == 0:
        raise ValueError("非法 Range")
    first, last = match.groups()
    if not first:
        if int(last) == 0:
            raise ValueError("空 Range")
        start, end = max(0, length - int(last)), length - 1
    else:
        start, end = int(first), min(int(last) if last else length - 1, length - 1)
    if start >= length or end < start:
        raise ValueError("Range 超出文件")
    return start, end, True


def make_review_server(output_dir, host="127.0.0.1", port=8023, *,
                       read_only=None, review_source=None, frozen_preview=False):
    """Create a testable server; only IPv4 loopback is allowed."""
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("审核服务仅允许本机 127.0.0.1")
    store = ReviewStore(output_dir, read_only=read_only, review_source=review_source, frozen_preview=frozen_preview)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def _local(self, post=False):
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            host_header = self.headers.get("Host", "")
            if self.client_address[0] != "127.0.0.1" or host_header not in allowed:
                self.send_error(403, "Only local Host is accepted")
                return False
            if post:
                if self.headers.get("Origin") != f"http://{host_header}" or not secrets.compare_digest(
                        self.headers.get("X-Review-Token", ""), store.token):
                    self.send_error(403, "Same-origin review token required")
                    return False
            return True

        def _headers(self, status, kind, length):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                             "style-src 'self' 'unsafe-inline'; media-src 'self'; frame-ancestors 'none'")

        def _json(self, value, status=200):
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self._headers(status, "application/json; charset=utf-8", len(encoded))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if not self._local():
                return
            path = urlsplit(self.path).path
            try:
                if path == "/":
                    encoded = HTML.replace("__REVIEW_TOKEN__", json.dumps(store.token)).encode("utf-8")
                    self._headers(200, "text/html; charset=utf-8", len(encoded))
                    self.end_headers()
                    self.wfile.write(encoded)
                elif path == "/review_metrics.js":
                    encoded = Path(__file__).with_name("review_metrics.js").read_bytes()
                    self._headers(200, "text/javascript; charset=utf-8", len(encoded))
                    self.end_headers()
                    self.wfile.write(encoded)
                elif path == "/api/manifest":
                    self._json(store.manifest_view())
                elif path == "/api/reviews":
                    self._json(store.document)
                elif re.fullmatch(r"/api/index/\d+", path):
                    self._json(store.index(int(path.rsplit("/", 1)[1])))
                elif re.fullmatch(r"/preview-media/[0-9a-f]{64}/(top|front|side|oblique)", path):
                    _, _, digest, camera = path.split("/")
                    video, root = store.previews.media(digest, camera)
                    self._video(video, root)
                elif path.startswith("/media/") and path[7:] in store.media:
                    video_path, video_root = store.media[path[7:]]
                    self._video(video_path, video_root)
                else:
                    self.send_error(404)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            except PermissionError:
                self._json({"error": "当前 Windows 用户无权读取此文件，请保留待审核并检查文件读取权限；不代表数据损坏。"}, 403)
            except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, 400)

        def _video(self, registered, root=None):
            path = _inside(registered, root or store.source_root)
            length = path.stat().st_size
            try:
                start, end, partial = _byte_range(self.headers.get("Range"), length)
            except ValueError:
                self._headers(416, "video/mp4", 0)
                self.send_header("Content-Range", f"bytes */{length}")
                self.end_headers()
                return
            count = max(0, end - start + 1)
            # Check read access before sending video headers so permission failures
            # return one complete error response rather than a corrupt MP4 stream.
            with path.open("rb") as stream:
                stream.seek(start)
                self._headers(206 if partial else 200, "video/mp4", count)
                self.send_header("Accept-Ranges", "bytes")
                if partial:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{length}")
                self.end_headers()
                while count:
                    chunk = stream.read(min(count, 256 * 1024))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    count -= len(chunk)

        def do_POST(self):
            if not self._local(post=True):
                return
            if self.path not in {"/api/review", "/api/preview"}:
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                if not 0 < length <= 131072 or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("仅接受不超过 128 KiB 的 JSON")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("请求必须是 JSON 对象")
                self._json(store.preview(payload) if self.path == "/api/preview" else store.save(payload))
            except ReadOnlyReviewError as exc:
                self._json({"error": str(exc), "read_only": True}, 403)
            except PermissionError:
                message = ("当前 Windows 用户无权读取视频，无法生成预览。请检查读取权限；原数据未改动。"
                           if self.path == "/api/preview" else
                           "当前 Windows 用户无权保存审核文件。请检查输出目录写入权限；原数据未改动。")
                self._json({"error": message}, 403)
            except RuntimeError as exc:
                self._json({"error": str(exc)}, 409)
            except (ValueError, OSError) as exc:
                self._json({"error": str(exc)}, 400)

    class ReviewServer(ThreadingHTTPServer):
        # Windows SO_REUSEADDR can let a second process bind this exact live
        # endpoint. A newly elevated service must never share the old listener.
        allow_reuse_address = os.name != "nt"

        def server_bind(self):
            if os.name == "nt":
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            super().server_bind()

        def server_close(self):
            super().server_close()
            store.previews.close()

    server = ReviewServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.review_store = store
    return server


def serve_review(output_dir, host="127.0.0.1", port=8023, *,
                 read_only=None, review_source=None, frozen_preview=False, open_browser=False):
    try:
        server = make_review_server(output_dir, host, port, read_only=read_only,
                                    review_source=review_source, frozen_preview=frozen_preview)
    except OSError as exc:
        if getattr(exc, "winerror", None) in {10013, 10048} or exc.errno in {98, 48}:
            raise SystemExit(f"审核端口 {port} 已被占用。请先关闭旧审核服务，或用 --port 指定空闲端口；"
                             "管理员启动不会替换已运行的普通权限服务。") from exc
        raise
    print(f"审核页面：http://127.0.0.1:{server.server_port}/ （原始数据只读；Ctrl+C 退出）", flush=True)
    identity = service_identity()
    label = "管理员" if identity["elevated"] else "普通权限"
    print(f"当前服务 PID {identity['pid']} · {label}。网页顶部应显示同一个 PID。", flush=True)
    if open_browser:
        import webbrowser
        webbrowser.open(f'http://127.0.0.1:{server.server_port}/')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


HTML = Path(__file__).with_name("review.html").read_text(encoding="utf-8")
