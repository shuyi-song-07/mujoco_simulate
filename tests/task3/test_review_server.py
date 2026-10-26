"""Review decisions cannot override source identity, structure, or local scope."""
import copy
import http.client
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from dataset.task3.review_server import (
    HTML, REVIEW_VERSION, ReviewStore, load_reviews, make_review_server,
    review_binding, validate_review, validate_reviews, validate_frame_overrides,
)


class CurationReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.output = self.root / "prepared"
        self.source.mkdir()
        self.output.mkdir()
        self.video = self.source / "three_episodes.mp4"
        self.video.write_bytes(bytes(range(100)))
        index = self.output / "index.jsonl"
        index.write_text("\n".join(json.dumps({"raw_frame_index": n, "sim_time": n / 30,
                            "keep": n >= 2, "segment": 0 if n >= 2 else None,
                            "status": {"left_valid": n != 4}}) for n in range(10)), encoding="utf-8")
        self.manifest = {"version": "test-v1", "run_id": "test-run", "source_root": str(self.source),
                         "selection_policy": "papera-intent-aware",
                         "config": {"wait_s": 1}, "source_fingerprints_sha256": "source-hash",
                         "episodes": [{"key": "session/0", "session": "session", "episode_index": 0,
                             "dataset_root": str(self.source), "structural_pass": True,
                             "metrics": {"raw_frames": 10}, "errors": [], "warnings": [],
                             "video_sources": [{"camera": camera, "path": str(self.video),
                                                "start_s": 20, "end_s": 20 + 10 / 30}
                                               for camera in ("top", "front", "side")],
                             "index_path": str(index), "segments": [], "cuts": []}]}
        self.write_manifest()
        self.servers = []

    def tearDown(self):
        for server, thread in self.servers:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.temp.cleanup()

    def write_manifest(self):
        (self.output / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def server(self):
        server = make_review_server(self.output, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.servers.append((server, thread))
        return server

    def test_four_camera_collection_prefers_model_view_in_three_review_views(self):
        self.manifest['episodes'][0]['video_sources'].append({
            'camera': 'oblique', 'path': str(self.video), 'start_s': 20, 'end_s': 20 + 10 / 30})
        self.write_manifest()
        store = ReviewStore(self.output)
        self.assertEqual([v['camera'] for v in store.manifest_view()['episodes'][0]['videos']],
                         ['top', 'front', 'oblique'])

    def test_current_three_camera_collection_can_be_reviewed(self):
        self.manifest['episodes'][0]['video_sources'][2]['camera']='oblique'
        self.write_manifest()
        store=ReviewStore(self.output)
        self.assertEqual([v['camera'] for v in store.manifest_view()['episodes'][0]['videos']],
                         ['top','front','oblique'])

    def test_native_single_camera_collection_can_be_reviewed(self):
        episode = self.manifest['episodes'][0]
        episode['video_sources'] = [dict(episode['video_sources'][0], camera='oblique')]
        self.write_manifest()
        store = ReviewStore(self.output)
        self.assertEqual([v['camera'] for v in store.manifest_view()['episodes'][0]['videos']], ['oblique'])

    def request(self, server, path, method="GET", data=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        request_headers = headers or {}
        body = json.dumps(data).encode() if data is not None else None
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    def review(self, **overrides):
        return {"quality": "B", "reviewer": " reviewer ", "watched_full": True,
                "notes": "合理修正 <script>alert(1)</script>", "exclude_intervals": [[0, 2]],
                "keep_intervals": [[2, 4]], **overrides}

    def payload(self, store, **overrides):
        return {"episode_key": "session/0", "binding": store.document["binding"],
                "revision": store.document["revision"], "review": self.review(), **overrides}

    def post_headers(self, server):
        return {"Origin": f"http://127.0.0.1:{server.server_port}", "Content-Type": "application/json",
                "X-Review-Token": server.review_store.token}

    def test_second_service_cannot_bind_an_active_review_port(self):
        first = self.server()
        other = None
        try:
            with self.assertRaises(OSError):
                other = make_review_server(self.output, port=first.server_port)
        finally:
            if other is not None:
                other.server_close()
        self.assertEqual(self.request(first, "/api/manifest")[0], 200)

    def test_empty_reviews_are_pending_and_sources_unchanged_after_save(self):
        originals = {p: p.read_bytes() for p in self.source.rglob("*") if p.is_file()}
        store = ReviewStore(self.output)
        self.assertEqual(store.document["episodes"], {})
        result = store.save(self.payload(store))
        self.assertEqual(result["review"]["reviewer"], "reviewer")
        self.assertEqual(result["revision"], 1)
        self.assertIn("+00:00", result["review"]["reviewed_at"])
        self.assertEqual(load_reviews(self.output)["episodes"]["session/0"]["quality"], "B")
        self.assertFalse(any("training_ready" in value for value in result["review"]))
        self.assertEqual(originals, {p: p.read_bytes() for p in originals})

    def test_structural_errors_and_unwatched_review_cannot_pass(self):
        ep = self.manifest["episodes"][0]
        for change in ({"watched_full": False}, {"reviewer": ""}, {"training_ready": True},
                       {"quality": "success"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_review(self.review(**change), ep)
        ep["structural_pass"] = False
        for grade in ("A", "B"):
            with self.assertRaisesRegex(ValueError, "结构"):
                validate_review(self.review(quality=grade), ep)
        validate_review(self.review(quality="C"), ep)
        validate_review(self.review(quality="pending", watched_full=False, reviewer=""), ep)

    def test_intervals_half_open_bounded_and_disjoint(self):
        ep = self.manifest["episodes"][0]
        for intervals in ([[0, 11]], [[-1, 2]], [[1, 1]], [[True, 2]], [[1.1, 2]], [[0, 3], [2, 4]]):
            with self.subTest(intervals=intervals), self.assertRaises(ValueError):
                validate_review(self.review(exclude_intervals=intervals, keep_intervals=[]), ep)
        with self.assertRaisesRegex(ValueError, "冲突"):
            validate_review(self.review(exclude_intervals=[[0, 3]]), ep)
        result = validate_review(self.review(exclude_intervals=[[5, 10], [0, 2]]), ep)
        self.assertEqual(result["exclude_intervals"], [[0, 2], [5, 10]])

    def test_unreadable_episode_can_be_rejected_without_claiming_full_view(self):
        ep = copy.deepcopy(self.manifest["episodes"][0])
        ep.update(structural_pass=False, metrics={"raw_frames": 0})
        review = self.review(quality="rejected", watched_full=False,
                             exclude_intervals=[], keep_intervals=[], notes="视频无法解码")
        result = validate_review(review, ep)
        self.assertFalse(result["watched_full"])
        with self.assertRaisesRegex(ValueError, "理由"):
            validate_review(review | {"notes": ""}, ep)

    def test_import_bound_to_exact_manifest_config_and_sources(self):
        store = ReviewStore(self.output)
        store.save(self.payload(store))
        for field, value in (("run_id", "other"), ("config", {"wait_s": 2}),
                             ("source_fingerprints_sha256", "changed")):
            altered = dict(self.manifest) | {field: value}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "绑定"):
                validate_reviews(store.document, altered, store.manifest_path)
        document = copy.deepcopy(store.document)
        document["episodes"]["unknown"] = self.review()
        with self.assertRaisesRegex(ValueError, "不属于"):
            validate_reviews(document, self.manifest, store.manifest_path)
        self.manifest["episodes"][0]["warnings"].append("new warning")
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "变化"):
            load_reviews(self.output)
        with self.assertRaisesRegex(ValueError, "清单已"):
            store.save(self.payload(store))

    def test_import_normalizes_without_mutating_input(self):
        document = load_reviews(self.output)
        document["episodes"]["session/0"] = self.review(reviewed_at="2026-10-16T00:00:00+00:00")
        normalized = validate_reviews(document, self.manifest, self.output / "manifest.json")
        self.assertEqual(normalized["episodes"]["session/0"]["reviewer"], "reviewer")
        self.assertEqual(document["episodes"]["session/0"]["reviewer"], " reviewer ")

    def test_media_is_registry_only_and_supports_shard_ranges(self):
        server = self.server()
        status, _, body = self.request(server, "/api/manifest")
        manifest = json.loads(body)
        self.assertEqual(status, 200)
        video = manifest["episodes"][0]["videos"][0]
        self.assertEqual(video["start_s"], 20)
        self.assertNotIn("path", video)
        status, headers, body = self.request(server, video["url"], headers={"Range": "bytes=10-19"})
        self.assertEqual((status, headers["Content-Range"], body), (206, "bytes 10-19/100", bytes(range(10, 20))))
        self.assertEqual(self.request(server, video["url"], headers={"Range": "bytes=-3"})[2], bytes(range(97, 100)))
        for value in ("bytes=200-", "bytes=0-1,3-4", "bytes=-0", "bytes=9-1"):
            self.assertEqual(self.request(server, video["url"], headers={"Range": value})[0], 416)
        for path in ("/media/../../manifest.json", "/media/%2e%2e%2fmanifest.json", "/api/index/-1", "/reviews.json"):
            self.assertEqual(self.request(server, path)[0], 404)
        self.assertEqual(len(json.loads(self.request(server, "/api/index/0")[2])), 10)

    def test_only_loopback_same_origin_and_token_can_write(self):
        with self.assertRaisesRegex(ValueError, "本机"):
            make_review_server(self.output, host="0.0.0.0")
        server = self.server()
        headers = self.post_headers(server)
        payload = self.payload(server.review_store)
        for change in ({"Origin": "https://evil.example"}, {"Origin": "null"},
                       {"X-Review-Token": "wrong"}, {"Host": "evil.example"}):
            self.assertEqual(self.request(server, "/api/review", "POST", payload, headers | change)[0], 403)
        self.assertEqual(self.request(server, "/api/manifest", headers={"Host": "evil.example"})[0], 403)
        status, _, body = self.request(server, "/api/review", "POST", payload, headers)
        self.assertEqual(status, 200, body)
        self.assertEqual(self.request(server, "/api/review", "POST", payload, headers)[0], 409)

    def test_unreadable_registered_video_returns_one_json_error_before_success_headers(self):
        original = self.video.read_bytes()
        server = self.server()
        manifest = json.loads(self.request(server, "/api/manifest")[2])
        url = manifest["episodes"][0]["videos"][0]["url"]
        original_open = Path.open

        def deny_only_video(path, *args, **kwargs):
            if path.resolve() == self.video.resolve():
                raise PermissionError("synthetic registered-video read denied")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", new=deny_only_video):
            for headers in ({}, {"Range": "bytes=10-19"}):
                with self.subTest(headers=headers):
                    status, response_headers, body = self.request(server, url, headers=headers)
                    self.assertEqual(status, 403)
                    self.assertEqual(response_headers["Content-Type"], "application/json; charset=utf-8")
                    self.assertNotIn("Content-Range", response_headers)
                    self.assertNotIn("Accept-Ranges", response_headers)
                    self.assertEqual(int(response_headers["Content-Length"]), len(body))
                    error = json.loads(body)["error"]
                    self.assertIn("读取权限", error)
                    self.assertIn("不代表数据损坏", error)
        self.assertEqual(self.video.read_bytes(), original)

    def test_manifest_cannot_register_arbitrary_source_or_index(self):
        outside = self.root / "private.mp4"
        outside.write_bytes(b"secret")
        for part, value in (("video", str(outside)), ("index", str(outside))):
            altered = copy.deepcopy(self.manifest)
            if part == "video":
                altered["episodes"][0]["video_sources"][0]["path"] = value
            else:
                altered["episodes"][0]["index_path"] = value
            (self.output / "manifest.json").write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "超出"):
                ReviewStore(self.output)

    def test_html_uses_text_content_for_untrusted_episode_and_notes(self):
        self.assertNotIn("innerHTML", HTML)
        self.assertRegex(HTML, r"\.textContent\s*=")
        server = self.server()
        status, headers, body = self.request(server, "/")
        self.assertEqual(status, 200)
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        self.assertNotIn(b"__REVIEW_TOKEN__", body)
        self.assertIn(server.review_store.token.encode(), body)

    def test_frame_review_metadata_and_raw_action_state_are_read_only(self):
        self.manifest["config"] = {
            "fps": 30, "stability": {"position_m": .0001, "joint_rad": .001},
            "alignment": {"maximum_frame_gap_s": .05},
        }
        self.write_manifest()
        path = self.output / "index.jsonl"
        row = {"raw_frame_index": 0, "state_recorded": list(range(16)),
               "action_recorded": list(range(14)),
               "next_control_target": {"action": list(range(10, 24))},
               "auxiliary": {"left": {"category": "Open_Palm", "category_score": .7}}}
        path.write_text(json.dumps(row), encoding="utf-8")
        before = path.read_bytes()
        server = self.server()
        status, _, body = self.request(server, "/api/manifest")
        view = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(view["frame_review"]["version"], "papera-cut-preview-v1")
        self.assertEqual(view["service"]["pid"], os.getpid())
        self.assertEqual(view["service"]["version"], "papera-review-service-v2")
        if os.name == "nt":
            self.assertIsInstance(view["service"]["elevated"], bool)
        self.assertEqual(view["frame_review"]["stability"], self.manifest["config"]["stability"])
        self.assertEqual(view["frame_review"]["alignment"], self.manifest["config"]["alignment"])
        self.assertEqual(json.loads(self.request(server, "/api/index/0")[2]), [row])
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse((self.output / "reviews.json").exists())

    def test_only_named_review_script_is_exposed(self):
        server = self.server()
        status, headers, body = self.request(server, "/review_metrics.js")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/javascript; charset=utf-8")
        self.assertIn(b"ReviewFrameMetrics", body)
        for path in ("/review_server.py", "/review.html", "/../review_server.py",
                     "/%2e%2e/review_server.py"):
            self.assertEqual(self.request(server, path)[0], 404)

    def test_historical_service_is_read_only_even_when_explicitly_disabled(self):
        self.manifest["selection_policy"] = "papera-sa-window-v1"
        self.write_manifest()
        store = ReviewStore(self.output, read_only=False)
        self.assertTrue(store.read_only)
        server = self.server()
        view = json.loads(self.request(server, "/api/manifest")[2])
        self.assertTrue(view["read_only"])
        self.assertTrue(view["service"]["read_only"])
        self.assertFalse(view["training_ready"])
        for endpoint in ("/api/review", "/api/preview"):
            status, _, body = self.request(server, endpoint, "POST",
                                           self.payload(server.review_store), self.post_headers(server))
            self.assertEqual(status, 403)
            self.assertIn("只读", json.loads(body)["error"])
        self.assertEqual(self.request(server, "/media/0-0")[0], 200)
        self.assertFalse((self.output / "reviews.json").exists())

    def test_explicit_read_only_can_only_restrict_current_policy(self):
        store = ReviewStore(self.output, read_only=True)
        with self.assertRaisesRegex(PermissionError, "只读"):
            store.save(self.payload(store))
        self.assertFalse(ReviewStore(self.output).read_only)

    def test_protected_exclusion_requires_reason_and_preserves_automatic_evidence(self):
        path = self.output / "index.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
        rows[0].update(frame_role="GRASP_PROTECTED", grasp_protected=True,
                       policy_reason=["actual_grasp_execution"], policy_decision="keep")
        path.write_text("\n".join(map(json.dumps, rows)), encoding="utf8")
        before = path.read_bytes()
        server = self.server()
        payload = self.payload(server.review_store, review=self.review(notes=" "))
        status, _, body = self.request(server, "/api/review", "POST", payload, self.post_headers(server))
        self.assertEqual(status, 400)
        self.assertIn("保护", json.loads(body)["error"])
        payload["review"]["notes"] = "工程测试：此段为错误抓空，排除但保留证据"
        self.assertEqual(self.request(server, "/api/review", "POST", payload, self.post_headers(server))[0], 200)
        self.assertEqual(path.read_bytes(), before)

    def test_invalid_role_cannot_be_kept_even_without_legacy_reason_code(self):
        row = {"raw_frame_index": 5, "frame_role": "INVALID", "reason": ["new_evidence_error"]}
        with self.assertRaisesRegex(ValueError, "INVALID"):
            validate_frame_overrides([row], {"keep_intervals": [[5, 6]], "notes": "仍不能覆盖"})
        validate_frame_overrides([row], {"exclude_intervals": [[5, 6]]})

    def test_index_fingerprint_is_checked_before_review_save(self):
        self.manifest["episodes"][0]["index_sha256"] = "0" * 64
        self.write_manifest()
        store = ReviewStore(self.output)
        with self.assertRaisesRegex(ValueError, "指纹"):
            store.save(self.payload(store))
        self.assertFalse((self.output / "reviews.json").exists())

    def test_new_sa_configuration_reaches_display_without_changing_legacy_defaults(self):
        self.manifest["config"] = {"sa": {"window_frames": 3, "thresholds": {"action_position_m": .0005}},
                                   "physical_motion": {"position_m": .0001}}
        self.write_manifest()
        view = ReviewStore(self.output).manifest_view()
        self.assertEqual(view["frame_review"]["selection"], self.manifest["config"]["sa"])
        self.assertFalse(view["training_ready"])

    def test_failed_structure_can_be_rejected_without_unreadable_frame_overrides(self):
        self.manifest["episodes"][0]["structural_pass"] = False
        self.write_manifest()
        (self.output / "index.jsonl").unlink()
        store = ReviewStore(self.output)
        review = self.review(quality="rejected", watched_full=False, exclude_intervals=[],
                             keep_intervals=[], notes="工程夹具缺失索引，排除整条")
        result = store.save(self.payload(store, review=review))
        self.assertEqual(result["review"]["quality"], "rejected")
        self.assertFalse(result["review"]["watched_full"])


    def test_height_protected_exclusion_requires_note_and_keeps_source_bytes(self):
        path = self.output / "index.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
        rows[0].update(frame_role="LOW_HEIGHT_PROTECTED", height_protected=True,
                       grasp_protected=False, release_protected=False, keep=True,
                       height_evidence={"available": True, "threshold_m": .390,
                                        "left_z_m": .388, "right_z_m": .389},
                       policy_reason=["both_actual_ee_below_height_threshold"], policy_decision="keep")
        path.write_text("\n".join(map(json.dumps, rows)), encoding="utf8")
        before = path.read_bytes()
        server = self.server()
        payload = self.payload(server.review_store, review=self.review(notes=" "))
        status, _, body = self.request(server, "/api/review", "POST", payload, self.post_headers(server))
        self.assertEqual(status, 400)
        self.assertIn("低位", json.loads(body)["error"])
        payload["review"]["notes"] = "工程测试：明确排除此低位帧，自动证据不改写"
        self.assertEqual(self.request(server, "/api/review", "POST", payload, self.post_headers(server))[0], 200)
        self.assertEqual(path.read_bytes(), before)

    def test_execution_diagnostics_and_overridden_terminal_height_are_not_manual_protection(self):
        rows = [{"raw_frame_index": 0, "height_protected": False,
                 "execution_grasp_phase": True, "execution_release_phase": True,
                 "height_protection_overridden_by_terminal": True,
                 "terminal_tail_excluded": True}]
        validate_frame_overrides(rows, {"exclude_intervals": [[0, 1]], "notes": ""})

    def test_user_quality_exclusion_blocks_a_b_but_does_not_invent_watched_status(self):
        ep = copy.deepcopy(self.manifest["episodes"][0])
        ep.update(quality_exclusion_reason="用户明确排除399mm真实释放", training_candidate=False)
        for grade in ("A", "B"):
            with self.subTest(grade=grade), self.assertRaisesRegex(ValueError, "用户已排除训练候选"):
                validate_review(self.review(quality=grade), ep)
        for grade in ("pending", "rejected"):
            result = validate_review(self.review(quality=grade, watched_full=False), ep)
            self.assertFalse(result["watched_full"])
        for reason in (None, "", " "):
            self.assertEqual(validate_review(self.review(), ep | {"quality_exclusion_reason": reason})["quality"], "B")

    def test_quality_exclusion_is_served_and_enforced_by_http_save(self):
        self.manifest["episodes"][0].update(quality_exclusion_reason="用户明确排除此条", training_candidate=False)
        self.write_manifest()
        server = self.server()
        manifest = json.loads(self.request(server, "/api/manifest")[2])
        self.assertEqual(manifest["episodes"][0]["quality_exclusion_reason"], "用户明确排除此条")
        self.assertFalse(manifest["episodes"][0]["training_candidate"])
        payload = self.payload(server.review_store)
        status, _, body = self.request(server, "/api/review", "POST", payload, self.post_headers(server))
        self.assertEqual(status, 400)
        self.assertIn("用户已排除训练候选", json.loads(body)["error"])
        self.assertFalse((self.output / "reviews.json").exists())
        payload["review"] = self.review(quality="pending", watched_full=False)
        status, _, body = self.request(server, "/api/review", "POST", payload, self.post_headers(server))
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)["review"]["watched_full"])

if __name__ == "__main__":
    unittest.main()
