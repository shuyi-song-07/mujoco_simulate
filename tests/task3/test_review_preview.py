import copy
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import av
import numpy as np

from dataset.task3.review_preview import PreviewCache, preview_plan, render_video
from dataset.task3.review_server import make_review_server
from dataset.task3.prepare_paper_dataset import apply_review


def rows_fixture():
    return [{"raw_frame_index": n, "keep": 2 <= n < 10, "segment_id": 0 if n < 7 else 1,
             "reason": [], "control_epoch": 1, "session": "synthetic", "original_time_s": n / 30}
            for n in range(12)]


class PreviewPlanTests(unittest.TestCase):
    def test_frozen_plan_never_recomputes_cuts_or_stitching(self):
        rows=rows_fixture(); before=copy.deepcopy(rows)
        review={'quality':'pending','exclude_intervals':[],'keep_intervals':[]}
        with patch('dataset.task3.prepare_paper_dataset.apply_review', side_effect=AssertionError('must not reapply')), \
             patch('dataset.task3.sequence_stitching.build_sequence_contract', side_effect=AssertionError('must not restitch')):
            result=preview_plan(rows,review,12,30,frozen=True)
        self.assertEqual(result['frame_map'],[r['raw_frame_index'] for r in rows if r['keep']])
        self.assertEqual(rows,before)
        with self.assertRaises(ValueError):preview_plan(rows,review,12,30,config={},frozen=True)
        with self.assertRaises(ValueError):preview_plan(rows,{**review,'exclude_intervals':[[4,6]]},12,30,frozen=True)

    def test_plan_matches_export_with_draft_overrides_and_keeps_boundaries(self):
        rows = rows_fixture()
        original = copy.deepcopy(rows)
        review = {"quality": "pending", "exclude_intervals": [[4, 6]], "keep_intervals": [[0, 1], [10, 12]]}
        plan = preview_plan(rows, review, 12, 30)
        self.assertEqual(plan["frame_map"], [0, 2, 3, 6, 7, 8, 9, 10, 11])
        expected = [r["raw_frame_index"] for r in apply_review(rows, review) if r["keep"]]
        self.assertEqual(plan["frame_map"], expected)
        self.assertEqual([s["preview_start"] for s in plan["segments"]], [0, 1, 3, 4])
        self.assertEqual(plan["duration_s"], 9 / 30)
        self.assertEqual(rows, original)
        self.assertNotIn("curation_approved", rows[0])

    def test_all_cut_and_invalid_keep(self):
        review = {"quality": "pending", "exclude_intervals": [[0, 12]], "keep_intervals": []}
        plan = preview_plan(rows_fixture(), review, 12, 30)
        self.assertEqual(plan["frame_map"], [])
        with PreviewCacheContext() as cache:
            self.assertEqual(cache.create(plan, [])["videos"], [])
        rows = rows_fixture()
        rows[0]["reason"] = ["invalid_numeric"]
        with self.assertRaisesRegex(ValueError, "invalid numeric"):
            preview_plan(rows, dict(review, exclude_intervals=[], keep_intervals=[[0, 1]]), 12, 30)

    def test_missing_duplicate_or_unsorted_frames_are_rejected(self):
        rows = rows_fixture()
        review = {"quality": "pending", "exclude_intervals": [], "keep_intervals": []}
        for bad in (rows[:-1], rows[::-1], [rows[0]] + rows[:-1]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                preview_plan(bad, review, 12, 30)


class PreviewCacheContext(PreviewCache):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class PreviewHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.output = self.root / "audit"
        self.source.mkdir(); self.output.mkdir()
        self.video = self.source / "shard.mp4"
        with av.open(str(self.video), "w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
            for n in range(45):
                pixels = np.full((48, 64, 3), n * 5, np.uint8)
                for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        index = self.output / "frames.jsonl"
        index.write_text("\n".join(json.dumps(r) for r in rows_fixture()), encoding="utf8")
        episode = {"key": "synthetic/0", "session": "synthetic", "episode_index": 0,
                   "dataset_root": str(self.source), "index_path": str(index), "structural_pass": True,
                   "metrics": {"raw_frames": 12}, "video_sources": [
                       {"camera": name, "path": str(self.video), "start_s": 1, "end_s": 1.4, "fps": 30}
                       for name in ("top", "front", "side")]}
        manifest = {"source_root": str(self.source), "episodes": [episode], "config": {"fps": 30},
                    "selection_policy": "papera-intent-aware"}
        (self.output / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
        self.before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.server = make_review_server(self.output, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)
        for path, original in self.before.items():
            self.assertEqual(path.read_bytes(), original)
        self.assertFalse((self.output / "reviews.json").exists())
        self.temp.cleanup()

    def request(self, path, payload=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=20)
        default = {"Origin": f"http://127.0.0.1:{self.server.server_port}", "Content-Type": "application/json",
                   "X-Review-Token": self.server.review_store.token}
        connection.request("POST" if payload is not None else "GET", path,
                           json.dumps(payload).encode() if payload is not None else None, default | (headers or {}))
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    def payload(self, **overrides):
        return {"episode_key": "synthetic/0", "binding": self.server.review_store.document["binding"],
                "exclude_intervals": [[4, 6]], "keep_intervals": [[10, 12]], **overrides}

    def test_three_decodable_videos_exact_frame_order_offset_and_cache(self):
        status, _, body = self.request("/api/preview", self.payload())
        self.assertEqual(status, 200, body)
        plan = json.loads(body)
        self.assertEqual(plan["frame_map"], [2, 3, 6, 7, 8, 9, 10, 11])
        with av.open(str(self.video)) as container:
            original = [f.to_ndarray(format="rgb24").mean() for f in container.decode(video=0)]
        for source in plan["videos"]:
            status, _, encoded = self.request(source["url"])
            self.assertEqual(status, 200)
            import io
            with av.open(io.BytesIO(encoded)) as container:
                frames = list(container.decode(video=0))
            self.assertEqual(len(frames), len(plan["frame_map"]))
            for n, (frame, raw) in enumerate(zip(frames, plan["frame_map"])):
                self.assertAlmostEqual(float(frame.pts * frame.time_base), n / 30, places=5)
                self.assertLess(abs(frame.to_ndarray(format="rgb24").mean() - original[30 + raw]), 4)
            status, headers, partial = self.request(source["url"], headers={"Range": "bytes=0-15"})
            self.assertEqual((status, partial), (206, encoded[:16]))
            self.assertIn("Content-Range", headers)
        cached = json.loads(self.request("/api/preview", self.payload())[2])
        self.assertEqual(plan, cached)
        self.assertEqual(len(self.server.review_store.previews.entries), 1)
        self.assertEqual(self.server.review_store.document["revision"], 0)

    def test_frozen_preview_allows_video_but_rejects_edits(self):
        from dataset.task3.review_server import ReviewStore
        store=ReviewStore(self.output,frozen_preview=True)
        self.assertTrue(store.read_only)
        self.assertTrue(store.manifest_view()['frozen_preview'])
        self.server.review_store.frozen_preview=True
        self.server.review_store.read_only=True
        payload=self.payload(exclude_intervals=[],keep_intervals=[])
        status,_,body=self.request('/api/preview',payload)
        self.assertEqual(status,200,body)
        self.assertEqual(json.loads(body)['frame_map'],list(range(2,10)))
        self.assertEqual(self.request('/api/review',payload)[0],403)
        self.assertEqual(self.request('/api/preview',self.payload())[0],400)

    def test_bad_intervals_binding_identity_and_auth_do_not_generate(self):
        for override in ({"exclude_intervals": [[4, 13]]}, {"keep_intervals": [[5, 8]]},
                         {"exclude_intervals": [[4, 4]]}, {"binding": {}}, {"episode_key": "elsewhere"}):
            self.assertEqual(self.request("/api/preview", self.payload(**override))[0], 400)
        self.assertEqual(self.request("/api/preview", self.payload(), {"X-Review-Token": "bad"})[0], 403)
        self.assertEqual(self.request("/preview-media/../../manifest.json")[0], 404)
        self.assertEqual(len(self.server.review_store.previews.entries), 0)

    def test_single_oblique_preview_decodes_and_http_serves_registered_camera(self):
        source = dict(self.server.review_store.review_video_sources[0][0], camera='oblique')
        self.server.review_store.review_video_sources[0] = [source]
        status, _, body = self.request('/api/preview', self.payload())
        self.assertEqual(status, 200, body)
        plan = json.loads(body)
        self.assertEqual([v['camera'] for v in plan['videos']], ['oblique'])
        status, _, encoded = self.request(plan['videos'][0]['url'])
        self.assertEqual(status, 200)
        import io
        with av.open(io.BytesIO(encoded)) as container:
            self.assertEqual(len(list(container.decode(video=0))), len(plan['frame_map']))
        self.assertEqual(self.request(plan['videos'][0]['url'].replace('/oblique', '/top'))[0], 400)

    def test_current_three_preview_streams_decode_without_side(self):
        import io
        sources=self.server.review_store.review_video_sources[0]
        sources[2]=dict(sources[2],camera='oblique')
        status,_,body=self.request('/api/preview',self.payload())
        self.assertEqual(status,200,body)
        plan=json.loads(body)
        self.assertEqual([v['camera'] for v in plan['videos']],['top','front','oblique'])
        for source in plan['videos']:
            status,_,encoded=self.request(source['url'])
            self.assertEqual(status,200)
            with av.open(io.BytesIO(encoded)) as container:
                self.assertEqual(len(list(container.decode(video=0))),len(plan['frame_map']))

    def test_all_excluded_has_no_video_and_wrong_pts_cannot_be_previewed(self):
        result = self.request("/api/preview", self.payload(exclude_intervals=[[0, 12]], keep_intervals=[]))
        self.assertEqual(result[0], 200)
        self.assertEqual(json.loads(result[2])["videos"], [])
        source = {"path": self.video, "start_s": 1.01, "end_s": 1.41, "fps": 30}
        with self.assertRaisesRegex(ValueError, "时间戳"):
            render_video(source, self.root / "invalid.mp4", [0, 1], 12, 30)

    def test_draft_exclusion_of_protection_requires_notes_without_saving_review(self):
        rows = rows_fixture()
        rows[4].update(frame_role="RELEASE_PROTECTED", release_protected=True,
                       policy_reason=["release_settling"], policy_decision="keep")
        with patch.object(self.server.review_store, "index", return_value=rows):
            status, _, body = self.request("/api/preview", self.payload(notes=" "))
            self.assertEqual(status, 400)
            self.assertIn("理由", json.loads(body)["error"])
            self.assertEqual(len(self.server.review_store.previews.entries), 0)
            status, _, body = self.request("/api/preview", self.payload(notes="工程夹具：排除错误释放片段"))
            self.assertEqual(status, 200, body)
            self.assertNotIn(4, json.loads(body)["frame_map"])
        self.assertEqual(self.server.review_store.document["revision"], 0)
        self.assertEqual(rows[4]["policy_reason"], ["release_settling"])

    def test_invalid_role_is_not_resurrected_by_draft_keep(self):
        rows = rows_fixture()
        rows[0].update(frame_role="INVALID", policy_reason=["missing_snapshot"])
        with patch.object(self.server.review_store, "index", return_value=rows):
            status, _, body = self.request("/api/preview", self.payload(keep_intervals=[[0, 1]]))
        self.assertEqual(status, 400)
        self.assertIn("INVALID", json.loads(body)["error"])
        self.assertEqual(len(self.server.review_store.previews.entries), 0)


if __name__ == "__main__":
    unittest.main()
