"""Deleting a live dataset must not trigger remote resume or kill the capture."""
from pathlib import Path
import tempfile
import unittest
from simulation.mujoco.common.teleop_service import DualArmRecorder


class MissingMetadataTests(unittest.TestCase):
    def test_missing_saved_dataset_preserves_capture_before_worker_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            for outcome in ('success', 'failure'):
                recorder=object.__new__(DualArmRecorder)
                recorder.recording=True;recorder.frames=1228
                recorder.saved_success=recorder.saved_failure=1
                recorder.roots={outcome:Path(tmp)/outcome}
                events=[];recorder.diagnostic=lambda *args,**kw:events.append((args,kw))
                with self.assertRaisesRegex(ValueError,'本地已保存的数据目录缺失'):
                    recorder.save_recording(outcome)
                self.assertTrue(recorder.recording)
                self.assertEqual(recorder.frames,1228)
                self.assertEqual(events[0][0][0],'save_local_metadata_missing')


if __name__=='__main__':unittest.main()
