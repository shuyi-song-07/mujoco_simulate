import json
from pathlib import Path
import tempfile
import unittest
from simulation.mujoco.dual_arm.measurement_journal import MeasurementJournal


class JournalTests(unittest.TestCase):
    def test_sealed_boundary_requires_durable_prefix_and_preserves_capture_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal=MeasurementJournal(Path(tmp)/'journal','s',{})
            barrier={'version':'hand-save-boundary-v1','client_id':'c','last_sequence':2,
                     'acknowledged_sequence':2,'pending':0,'dropped':0,'end_capture_epoch_ms':105}
            with self.assertRaisesRegex(ValueError,'not_durable'):journal.validate_boundary(barrier)
            rows=[{'sequence':i,'kind':'measurement','browser_epoch_ms':200+i,
                   'measurement':{'measurement_id':f'm{i}','source_session':'s','source_control_epoch':0,
                                  'capture_epoch_ms':100+i,'hands':[]}} for i in range(1,4)]
            journal.append({'client_id':'c','rows':rows,'report':{'completed':3}}, {'tick':90})
            out=Path(tmp)/'hands.jsonl'
            summary=journal.export(out,0,100,999,barrier)
            self.assertEqual(summary['measurements'],2)
            self.assertEqual(summary['end_ms'],105)
            data=[json.loads(l) for l in out.read_text().splitlines()]
            self.assertEqual([x['measurement']['capture_epoch_ms'] for x in data],[101,102])
            self.assertEqual([x['receipt']['tick'] for x in data],[90,90])
            self.assertGreater(data[0]['server_epoch_ms'],105)
            with self.assertRaisesRegex(ValueError,'incomplete'):journal.validate_boundary({**barrier,'pending':1})
            journal.close()

    def test_retry_unknown_missing_hand_and_episode_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal=MeasurementJournal(Path(tmp)/'journal','s',{'model_sha256':'frozen'})
            rows=[]
            for i in range(1,5):
                rows.append({'kind':'measurement','sequence':i,'browser_epoch_ms':100+i,'context':{'width':640,'height':480},
                    'measurement':{'measurement_id':f'c:{i}','source_session':'s','source_control_epoch':0,
                        'capture_epoch_ms':100+i,'hands':[] if i==4 else [{'gesture':'None','valid':True,
                            'image_landmarks_21':[[.2,.3,0]]*21}]}})
            batch={'client_id':'c','rows':rows,'report':{'completed':4,'dropped_measurements':0}}
            first=journal.append(batch,{'tick':1});retry=journal.append(batch,{'tick':2})
            self.assertEqual(first,retry);self.assertEqual(journal.written,4)
            summary=journal.export(Path(tmp)/'episode_hands.jsonl',0,102,105)
            self.assertEqual(summary['measurements'],3)
            persisted=[json.loads(line) for line in (Path(tmp)/'episode_hands.jsonl').read_text().splitlines()]
            self.assertEqual(persisted[0]['measurement']['hands'][0]['gesture'],'None')
            self.assertEqual(len(persisted[0]['measurement']['hands'][0]['image_landmarks_21']),21)
            self.assertEqual(persisted[-1]['measurement']['hands'],[])
            self.assertEqual(journal.clients['c']['unaccounted_measurements'],0)
            batch['rows'][0]['measurement']['measurement_id']='changed'
            with self.assertRaisesRegex(ValueError,'retry_content_changed'):journal.append(batch,{})
            journal.close()

    def test_old_session_and_epoch_remain_in_journal_but_not_new_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal=MeasurementJournal(Path(tmp)/'journal','s',{})
            row={'sequence':1,'kind':'measurement','browser_epoch_ms':110,'measurement':{'measurement_id':'late',
                 'source_session':'old','source_control_epoch':0,'capture_epoch_ms':105,'hands':[]}}
            journal.append({'client_id':'c','rows':[row],'report':{'completed':1}}, {})
            self.assertEqual(journal.export(Path(tmp)/'hands.jsonl',0,0,200)['measurements'],0)
            self.assertIn('late',journal.path.read_text());journal.close()

    def test_nonfinite_and_invalid_batch_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal=MeasurementJournal(Path(tmp)/'journal','s',{})
            with self.assertRaises(ValueError):journal.append({'client_id':'c','rows':[],'report':{}},{})
            row={'sequence':1,'kind':'dispatch','browser_epoch_ms':float('nan')}
            with self.assertRaises(ValueError):journal.append({'client_id':'c','rows':[row],'report':{}},{})
            self.assertEqual(journal.written,0);journal.close()


class ManifestRecoveryTests(unittest.TestCase):
    def test_temporary_manifest_rename_denial_does_not_poison_retries(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            journal=MeasurementJournal(Path(tmp)/'journal','s',{})
            row={'sequence':1,'kind':'measurement','browser_epoch_ms':100,
                 'measurement':{'measurement_id':'m1','hands':[]}}
            payload={'client_id':'c','rows':[row],'report':{'completed':1}}
            with patch('simulation.mujoco.dual_arm.measurement_journal.os.replace',side_effect=PermissionError('WinError 5')):
                with self.assertRaises(PermissionError):journal.append(payload,{})
            ack=journal.append(payload,{})
            self.assertTrue(ack['ok']);self.assertEqual(journal.written,1)
            self.assertEqual(len(journal.path.read_text().splitlines()),1)
            self.assertIsNone(journal.status()['error']);journal.close()


class OperationExportTests(unittest.TestCase):
    def test_failed_auxiliary_export_returns_explicit_recovery_marker(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            journal=MeasurementJournal(Path(tmp)/'journal','s',{})
            try:
                with patch.object(journal,'export',side_effect=PermissionError('export denied')):
                    result=journal.export_for_operation(Path(tmp)/'hands.jsonl',3,100,200,{'pending':4})
                self.assertTrue(result['requires_review'])
                self.assertIn('export denied',result['research_warning'])
                self.assertEqual(result['path'],str(journal.root))
                self.assertIsNone(result['sha256'])
                marker=json.loads((Path(tmp)/'hands.manifest.json').read_text())
                self.assertEqual(marker['source_control_epoch'],3)
            finally:journal.close()

    def test_pending_upload_marks_export_for_review_without_claiming_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal=MeasurementJournal(Path(tmp)/'journal','s',{})
            try:
                result=journal.export_for_operation(Path(tmp)/'hands.jsonl',0,100,200,{'pending':4,'error':'HTTP 503'})
                self.assertTrue(result['requires_review'])
                self.assertEqual(result['research_warning'],'hand_measurement_upload_incomplete')
            finally:journal.close()
