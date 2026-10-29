"""Run retained algorithm and migration regressions; no human camera or training."""
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'tests/task3',ROOT/'tests/papera'):sys.path.insert(0,str(path))
MODULES=[
    'tests.test_task_registry',
    *['tests.task3.'+n for n in (
        'test_sequence_stitching','test_adjacent_stationarity','test_wait_compression',
        'test_height_protection','test_terminal_tail','test_frame_evidence',
        'test_curation_rules','test_intent_curation','test_curation_safety',
        'test_single_view_prepared_pack','test_measurement_journal','test_hand_measurement','test_review_server','test_review_preview',
        'test_gesture_hand_guard','test_rate_hand_guard','test_rate_config',
        'test_level_gripper_control','test_operator_view','test_training_camera','test_raised_placement',
        'test_highwall_platform','test_platform_success')],
    *['tests.papera.'+n for n in ('test_safe_stitch_data','test_act_model','test_action_chunks',
        'test_policy_executor','test_training_loop')]
]
if __name__=='__main__':
    import json
    import subprocess
    collection_only='--collection-only' in sys.argv
    names=['tests.task3.test_action_contract', 'tests.task3.test_level_gripper_control',
           'tests.task3.test_control_computation','tests.task3.test_episode_completion'] if collection_only else MODULES
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(names))
    failed=not result.wasSuccessful()
    if not collection_only:
        env=json.loads((ROOT/'config/environments.json').read_text(encoding='utf-8'))
        failed |= subprocess.call([env['collection_python'],'-X','utf8',str(Path(__file__).resolve()),'--collection-only'],cwd=ROOT)!=0
    raise SystemExit(failed)
