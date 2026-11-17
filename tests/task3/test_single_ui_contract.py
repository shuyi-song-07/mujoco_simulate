"""Actual JS UI transport -> Python journal/hand guards; synthetic, never human."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import queue
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from tasks.registry import get
from simulation.mujoco.single_arm.task_controller import SingleArmTaskController
from simulation.mujoco.dual_arm.command_protocol import CommandGate
from simulation.mujoco.dual_arm.hand_measurement import HandInputGuard, RATE_PROTOCOL
from simulation.mujoco.dual_arm.measurement_journal import MeasurementJournal

ROOT=Path(__file__).resolve().parents[2]


class SingleUIContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        node=shutil.which('node')
        if node is None:raise unittest.SkipTest('Node is required to exercise real browser helpers')
        cls.node=node
        process=subprocess.run([node,str(Path(__file__).with_name('single_ui_contract_fixture.mjs'))],
            cwd=ROOT,capture_output=True,text=True,encoding='utf-8',check=True)
        cls.fixture=json.loads(process.stdout);cls.config=get('push_t').config()

    def test_browser_operation_and_automatic_recovery_lifecycle(self):
        result=subprocess.run([shutil.which('node'),'--test',str(ROOT/'collection/push_t/web/single-arm-app.test.js')],
            cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_dispatch_keeps_capture_time_and_refreshes_only_manual_operations(self):
        self.assertTrue(self.fixture['packets'])
        for row in self.fixture['packets']:
            packet=row['packet']
            self.assertEqual(packet['sentAt'],packet['telemetry']['recognition_result']['measurement']['capture_epoch_ms'])
        self.assertEqual([p['command'] for p in self.fixture['manualPackets']],['pause','record_start'])
        for packet in self.fixture['manualPackets']:
            self.assertEqual(packet['sentAt'],self.fixture['dispatchEpochMs'])

    def test_real_dispatch_queue_preserves_capture_time(self):
        process=subprocess.run([self.node,'--test',
            str(ROOT/'collection/push_t/web/single-dispatch.test.js')],
            cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(process.returncode,0,process.stdout+process.stderr)

    def test_actual_journal_batches_ack_and_seal_recording_interval(self):
        fixture=self.fixture;checkroot=ROOT/'.local/checks/single_arm_push_t';checkroot.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='ui-contract-',dir=checkroot) as temporary:
            journal=MeasurementJournal(Path(temporary)/'journal','ui-contract',{'input_kind':'scripted_test'})
            try:
                for batch in fixture['batches']:
                    self.assertLessEqual(len(batch['rows']),4)
                    ack=journal.append(batch,{'session':'ui-contract','control_epoch':2,'tick':0,'sim_time':0.})
                    self.assertTrue(ack['ok']);self.assertEqual(ack['last_sequence'],batch['rows'][-1]['sequence'])
                start=fixture['start'];end=fixture['saved']
                self.assertEqual(journal.validate_boundary(start),start['end_capture_epoch_ms'])
                self.assertEqual(journal.validate_boundary(end),end['end_capture_epoch_ms'])
                destination=Path(temporary)/'episode_hands.jsonl'
                summary=journal.export(destination,2,start['end_capture_epoch_ms'],end['end_capture_epoch_ms'],end)
                rows=[json.loads(line) for line in destination.read_text(encoding='utf-8').splitlines()]
                ids=[row['measurement']['measurement_id'] for row in rows if row['kind']=='measurement']
                self.assertEqual(ids,fixture['expectedRecordingIds'])
                self.assertEqual(summary['measurements'],len(ids));self.assertEqual(journal.status()['error'],None)
                self.assertTrue(any(row['kind']=='camera_event' and row.get('event')=='control_result' for row in rows))
                self.assertNotIn(fixture['measurements'][-1]['measurement_id'],ids)
                with self.assertRaisesRegex(ValueError,'incomplete'):
                    journal.validate_boundary({**end,'pending':1})
            finally:journal.close()

    def test_wrist_directions_match_actual_single_arm_top_camera(self):
        for row in self.fixture['directionPackets']:
            with self.subTest(direction=row['direction']):
                controller=SingleArmTaskController(self.config)
                camera=controller.model.camera('top').id
                rotation=controller.data.cam_xmat[camera].reshape(3,3)
                before=controller.left.command_position.copy()
                packet=row['packet']
                answer=controller.apply_command(packet,now_ms=packet['sentAt']+10)
                self.assertTrue(answer['accepted'],answer)
                world_delta=controller.left.command_position-before
                # Camera columns are screen right, screen up and backward.
                projected=rotation[:,:2].T@world_delta
                self.assertGreater(np.linalg.norm(projected),0)
                np.testing.assert_allclose(projected/np.linalg.norm(projected),row['displayDelta'],atol=1e-6)

    def test_actual_ui_fields_pass_python_admission_guard_and_controller(self):
        controller=SingleArmTaskController(self.config);guard=HandInputGuard(('left',));gate=CommandGate(self.config['control'])
        accepted=[];recording_started=False;observed_pause=False;accepted_palm=False
        for row in self.fixture['packets']:
            p=row['packet'];clock=p['sentAt']+10
            if row['frame']>=self.fixture['startFrame'] and not recording_started:
                before=controller.action_vector_m();controller.begin_recording();recording_started=True
                self.assertIsNone(controller.mode_event_id);np.testing.assert_array_equal(before,controller.action_vector_m())
            ticket,reason=gate.admit(p,session='ui-contract',epoch=2,now_ms=clock,monotonic=clock/1000,epoch_started_ms=100000)
            self.assertIsNone(reason);self.assertIsNone(gate.consume(ticket,session='ui-contract',epoch=2,now_ms=clock,monotonic=clock/1000))
            reason=guard.check(p,session='ui-contract',epoch=2,now_ms=clock,control_protocol=RATE_PROTOCOL)
            if reason:controller.pause(reason);continue
            controller.update_telemetry(p['telemetry'])
            answer=controller.apply_command(p,now_ms=clock)
            if answer['accepted']:
                accepted.append(row['frame'])
                self.assertTrue(p['telemetry']['hands']['left']['visible']);self.assertNotIn('right',p['telemetry']['hands'])
                self.assertEqual(controller.action_vector_m()[-1],0.)
                accepted_palm|=p['interaction_modes']==['Z_OPEN']
            else:
                self.assertEqual(p['interaction_modes'],['HOLD'],answer)
                self.assertTrue(controller.paused);observed_pause=True
            controller.step(1/60)
        self.assertTrue(recording_started);self.assertTrue(observed_pause);self.assertTrue(accepted_palm)
        self.assertGreater(sum(i<self.fixture['startFrame'] for i in accepted),5)
        self.assertGreater(sum(i>=self.fixture['startFrame'] for i in accepted),20)
        self.assertGreater(max(accepted),self.fixture['endFrame']-5)

    def test_shared_record_start_clears_practice_sources_before_first_capture(self):
        # Exercise the actual recorder boundary. Rendering IPC is replaced by
        # an acknowledged test sink; full video encoding has separate checks.
        from simulation.mujoco.dual_arm import teleop_service
        controller=SingleArmTaskController(self.config)
        p=next(row['packet'] for row in reversed(self.fixture['packets'])
               if row['frame']<self.fixture['startFrame'] and row['packet']['interaction_modes']==['XY_HOLD'])
        self.assertTrue(controller.apply_command(p,now_ms=p['sentAt']+10)['accepted'])
        record=teleop_service.DualArmRecorder.__new__(teleop_service.DualArmRecorder)
        record.controller=controller;record.config=self.config;record.paper_recording=True;record.recording=False
        record.args=SimpleNamespace(input_kind='scripted_test');record.telemetry=p['telemetry']
        record.session='single-ui-contract';record.attempt=0;record.control_epoch=2
        record.paper_events=queue.Queue();record.gripper_audits=queue.Queue();record.record_commands=queue.Queue()
        record.metadata=lambda outcome:{'outcome':outcome,'input_kind':'scripted_test'}
        record.wait_reply=lambda expected:{'type':expected};record.latest_observation=None
        record.action_event_id=p['eventId'];record.action_measurement_id=p['measurement_id']
        captured=[];record.capture=lambda **kw:captured.append(controller.snapshot())
        checkroot=ROOT/'.local/checks/single_arm_push_t';checkroot.mkdir(parents=True,exist_ok=True)
        before=controller.action_vector_m();qpos=controller.data.qpos.copy()
        with tempfile.TemporaryDirectory(prefix='record-start-',dir=checkroot) as temporary:
            try:
                with patch.object(teleop_service,'ROOT',Path(temporary)):
                    record.start_recording()
                self.assertTrue(record.recording);self.assertEqual(len(captured),1)
                self.assertIsNone(record.action_event_id);self.assertIsNone(record.action_measurement_id)
                self.assertIsNone(captured[0]['status']['physical_contact']['interaction_mode_event_id']['left'])
                self.assertTrue(captured[0]['status']['paused'])
                np.testing.assert_array_equal(before,controller.action_vector_m())
                np.testing.assert_array_equal(qpos,controller.data.qpos)
            finally:
                if getattr(record,'raw_file',None):record.raw_file.close()


if __name__=='__main__':unittest.main()
