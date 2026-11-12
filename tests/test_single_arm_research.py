"""Raw evidence survives retries, preserves unknown detections, and seals episodes."""
import hashlib
import json
from pathlib import Path
import queue
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np

from simulation.mujoco.measurement_journal import SingleArmResearchJournal
from simulation.mujoco.record_mujoco_panda import GestureServer, MujocoPandaRecorder


class SingleArmResearchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.journal = SingleArmResearchJournal(self.root / 'session', 's', {'test': True})
        self.addCleanup(self.journal.close)

    def batch(self):
        return {'client_id':'c', 'report':{'completed':1,'dropped':0,'pending':1}, 'rows':[
            {'sequence':1,'kind':'measurement','browser_epoch_ms':1000,'context':{'gain':1},
             'measurement':{'measurement_id':'c:1','hands':[{'gesture':'None',
                'image_landmarks_21':[[.2,.3,0]]*21}],
                'source_session':'s','source_control_epoch':2,'capture_epoch_ms':1000}}]}

    def boundary(self):
        return {'version':'hand-save-boundary-v1','client_id':'c','last_sequence':1,
                'acknowledged_sequence':1,'end_capture_epoch_ms':1100,
                'pending':0,'error':None,'dropped':0}

    def test_durable_ack_is_idempotent_and_rejects_changed_retry(self):
        batch=self.batch()
        self.assertTrue(self.journal.append(batch, {})['ok'])
        self.journal.append(batch, {})
        self.assertEqual(len(self.journal.path.read_text().splitlines()),1)
        batch['rows'][0]['measurement']['hands'][0]['gesture']='ILoveYou'
        with self.assertRaisesRegex(ValueError,'retry_content_changed'):
            self.journal.append(batch,{})
        self.assertEqual(self.journal.validate_boundary(self.boundary()),1100)
        bad={**self.boundary(),'last_sequence':2,'acknowledged_sequence':2}
        with self.assertRaisesRegex(ValueError,'not_durable'):
            self.journal.validate_boundary(bad)

    def test_episode_export_filters_epoch_and_keeps_source_and_configuration(self):
        self.journal.append(self.batch(),{})
        for name in ('success','failure'):
            result=self.journal.export_episode(self.root/name,0,2,900,1100,self.boundary())
            path=self.root/name/'research/episode_000000_hands.jsonl'
            row=json.loads(path.read_text())
            self.assertEqual(row['measurement']['hands'][0]['gesture'],'None')
            self.assertEqual(result['configurations'][row['configuration_id']],{'gain':1})
            self.assertEqual(result['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertTrue(result['boundary_verified'])
            self.assertFalse(result['requires_review'])
        result=self.journal.export_episode(self.root/'other',0,3,900,1100,self.boundary())
        self.assertEqual(result['measurements'],0)
        self.assertTrue(result['requires_review'])

    def test_event_snapshot_does_not_mutate_and_keyboard_save_is_marked(self):
        qpos=np.array([1.,2.]);start=time.time()*1000-10
        self.journal.record('robot_frame',control_epoch=2,qpos=qpos)
        qpos[0]=999
        result=self.journal.export_episode(self.root/'keyboard',0,2,start,time.time()*1000+10)
        row=json.loads((self.root/'keyboard/research/episode_000000_events.jsonl').read_text())
        self.assertEqual(row['qpos'],[1.,2.])
        self.assertTrue(result['requires_review'])
        self.assertFalse(result['boundary_verified'])

    def test_http_channel_ack_and_control_are_independent(self):
        commands=queue.Queue()
        server=GestureServer(commands,0,lambda:None,lambda:None,lambda:None,
            lambda:{'session':'s','control_epoch':2}, lambda:self.journal)
        self.addCleanup(server.server.server_close)
        client=server.server.app.test_client()
        response=client.post('/measurements',json=self.batch())
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json['last_sequence'],1)
        self.assertTrue(commands.empty())
        response=client.post('/control',json={'command':'move_xy','event_id':'e','measurement_id':'c:1'})
        self.assertEqual(response.status_code,200)
        self.assertTrue(response.json['queued'])
        self.assertEqual(commands.get_nowait()['measurement_id'],'c:1')

    def test_failed_boundary_preserves_robot_buffer_and_seal_stops_new_frames(self):
        recorder=object.__new__(MujocoPandaRecorder)
        recorder.research=self.journal
        recorder.recording=True;recorder.episode_frames=5;recorder.episode_seal=None
        recorder.control_epoch=2;recorder.episode_started_epoch_ms=900
        recorder.saved_episodes=0;recorder.failed_episodes=0
        recorder.args=SimpleNamespace(dataset_root=self.root/'success',failure_dataset_root=self.root/'failure')
        recorder.training_queue=queue.Queue()
        recorder.handle_gesture({'command':'record_seal','save_id':'save','outcome':'success'})
        recorder.record_frame()  # needs no robot state: sealing must stop at the same frame
        self.assertEqual(recorder.episode_frames,5)
        recorder.save_episode({'save_id':'save'})
        self.assertTrue(recorder.recording)
        self.assertEqual(recorder.episode_frames,5)
        self.assertTrue(recorder.training_queue.empty())
        self.assertIn('保存确认',recorder.last_recording_error)


if __name__=='__main__':
    unittest.main()
