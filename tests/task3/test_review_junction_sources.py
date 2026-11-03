"""Selected session aliases must not broaden the review media boundary."""
import copy
import http.client
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest

import av
import numpy as np

from dataset.task3.review_server import make_review_server


class SelectedSessionPreviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="papera-review-junction-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "selected_sources"
        self.session = self.root / "datasets" / "recorded_session"
        self.output = self.root / "curation"
        self.source.mkdir()
        self.session.mkdir(parents=True)
        self.output.mkdir()
        self.video = self.session / "oblique.mp4"
        with av.open(str(self.video), "w") as container:
            encoder = container.add_stream("libx264", rate=30)
            encoder.width, encoder.height, encoder.pix_fmt = 64, 48, "yuv420p"
            for number in range(12):
                pixels = np.full((48, 64, 3), number * 15, dtype=np.uint8)
                for packet in encoder.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                    container.mux(packet)
            for packet in encoder.encode():
                container.mux(packet)
        rows = [{"raw_frame_index": number, "keep": 2 <= number < 10,
                 "segment_id": 0, "reason": []} for number in range(12)]
        (self.output / "frames.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        self.manifest = {
            "version": "test-selected-session-v1", "run_id": "selected-session",
            "source_root": str(self.source), "selection_policy": "papera-intent-aware",
            "config": {"fps": 30}, "episodes": [{
                "key": "recorded_session/episode_000000", "session": "recorded_session",
                "episode_index": 0, "dataset_root": str(self.session),
                "index_path": "frames.jsonl", "structural_pass": True,
                "metrics": {"raw_frames": 12}, "video_sources": [{
                    "camera": "oblique", "path": str(self.video),
                    "start_s": 0, "end_s": .4, "fps": 30,
                }],
            }],
        }

    def directory_alias(self, alias, target):
        """Use actual junction resolution on Windows, a symlink elsewhere."""
        alias, target = Path(alias), Path(target).resolve()
        self.assertTrue(alias.parent.resolve().is_relative_to(self.root))
        self.assertTrue(target.is_relative_to(self.root))
        if os.name == "nt":
            quote = lambda value: "'" + str(value).replace("'", "''") + "'"
            subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                 "New-Item -ItemType Junction -Path " + quote(alias)
                 + " -Target " + quote(target) + " | Out-Null"],
                check=True, capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            self.addCleanup(alias.rmdir)
        else:
            alias.symlink_to(target, target_is_directory=True)
            self.addCleanup(alias.unlink)
        self.assertEqual(alias.resolve(), target)
        return alias

    def server(self, manifest=None):
        manifest = self.manifest if manifest is None else manifest
        (self.output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        server = make_review_server(self.output, port=0, frozen_preview=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def close():
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.addCleanup(close)
        return server

    def request(self, server, path, payload=None):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        headers = {"Origin": f"http://127.0.0.1:{server.server_port}",
                   "Content-Type": "application/json", "X-Review-Token": server.review_store.token}
        connection.request("GET" if payload is None else "POST", path,
                           None if payload is None else json.dumps(payload).encode(), headers)
        response = connection.getresponse()
        result = response.status, response.read()
        connection.close()
        return result

    def test_selected_session_junction_opens_media_and_frozen_cut_preview(self):
        self.directory_alias(self.source / self.session.name, self.session)
        original_video = self.video.read_bytes()
        server = self.server()
        status, encoded = self.request(server, "/api/manifest")
        self.assertEqual(status, 200, encoded)
        episode = json.loads(encoded)["episodes"][0]
        status, encoded = self.request(server, episode["videos"][0]["url"])
        self.assertEqual((status, encoded), (200, original_video))
        status, encoded = self.request(server, "/api/preview", {
            "episode_key": self.manifest["episodes"][0]["key"],
            "binding": server.review_store.document["binding"],
            "exclude_intervals": [], "keep_intervals": [],
        })
        self.assertEqual(status, 200, encoded)
        plan = json.loads(encoded)
        self.assertEqual(plan["frame_map"], [2, 3, 4, 5, 6, 7, 8, 9])
        status, encoded = self.request(server, plan["videos"][0]["url"])
        self.assertEqual(status, 200, encoded)
        with av.open(io.BytesIO(encoded)) as video:
            self.assertEqual(len(list(video.decode(video=0))), 8)
        self.assertEqual(self.video.read_bytes(), original_video)
        self.assertFalse((self.output / "reviews.json").exists())

    def test_unselected_external_session_is_rejected(self):
        with self.assertRaises(ValueError):
            self.server()

    def test_same_named_junction_to_another_session_is_rejected(self):
        other = self.root / "other_dataset" / self.session.name
        other.mkdir(parents=True)
        self.directory_alias(self.source / self.session.name, other)
        with self.assertRaises(ValueError):
            self.server()

    def test_nested_session_alias_is_not_a_direct_selection(self):
        nested = self.source / "nested"
        nested.mkdir()
        self.directory_alias(nested / self.session.name, self.session)
        with self.assertRaises(ValueError):
            self.server()

    def test_selected_session_cannot_register_another_selected_sessions_video(self):
        self.directory_alias(self.source / self.session.name, self.session)
        other = self.root / "datasets" / "other_session"
        other.mkdir()
        self.directory_alias(self.source / other.name, other)
        outside = other / "oblique.mp4"
        outside.write_bytes(b"another session must stay private")
        altered = copy.deepcopy(self.manifest)
        source = altered["episodes"][0]["video_sources"][0]
        source.update(path=str(outside), allowed_root=str(other.parent))
        with self.assertRaises(ValueError):
            self.server(altered)

    def test_video_cannot_escape_through_a_junction_inside_the_selected_session(self):
        self.directory_alias(self.source / self.session.name, self.session)
        outside = self.root / "private_media"
        outside.mkdir()
        (outside / "private.mp4").write_bytes(b"outside the selected session")
        alias = self.directory_alias(self.session / "escaped_media", outside)
        altered = copy.deepcopy(self.manifest)
        altered["episodes"][0]["video_sources"][0]["path"] = str(alias / "private.mp4")
        with self.assertRaises(ValueError):
            self.server(altered)

    def test_selected_session_does_not_authorize_an_external_frame_index(self):
        self.directory_alias(self.source / self.session.name, self.session)
        index = self.session / "private_index.jsonl"
        index.write_text("{}\n", encoding="utf-8")
        altered = copy.deepcopy(self.manifest)
        altered["episodes"][0]["index_path"] = str(index)
        with self.assertRaises(ValueError):
            self.server(altered)


if __name__ == "__main__":
    unittest.main()
