"""Run maintained regressions without human camera or formal training.

--with-upstream also checks an existing, independent ACT source checkout.
It never downloads upstream source, benchmark datasets, or checkpoints.
"""
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'tests/task3',ROOT/'tests/papera'):sys.path.insert(0,str(path))
MODULES=[
    'tests.test_task_registry',
    'tests.test_dataset_navigation',
    'tests.test_recorded_collection',
    'tests.test_native_source_paths',
    'tests.test_baseline_campaign',
    *['tests.task3.'+n for n in (
        'test_sequence_stitching','test_adjacent_stationarity','test_wait_compression',
        'test_height_protection','test_terminal_tail','test_frame_evidence',
        'test_curation_rules','test_curation_io','test_intent_curation','test_curation_safety','test_review_junction_sources',
        'test_merge_prepared','test_training_pack_union',
        'test_single_view_prepared_pack','test_measurement_journal','test_hand_measurement','test_review_server','test_review_preview',
        'test_gesture_hand_guard','test_rate_hand_guard','test_rate_config','test_recorded_rate_audit',
        'test_level_gripper_control','test_operator_view','test_training_camera','test_raised_placement',
        'test_highwall_platform','test_platform_success')],
    *['tests.papera.'+n for n in ('test_safe_stitch_data','test_act_model','test_action_chunks',
        'test_policy_executor','test_training_loop','test_training_engine','test_act_reference',
        'test_reference_compatibility','test_checkpoint_candidates','test_check_training_pack','test_diffusion_model',
        'test_diffusion_data','test_diffusion_training','test_diffusion_rollout','test_diffusion_official',
        'test_rollout_collection_sources','test_rollout_checkpoint_selection',
        'test_rollout_candidates','test_gesture_auxiliary','test_gesture_continuation',
        'test_loss_reporting','test_training_plots','test_policy_comparison','test_smoke_continuation_guard','test_rollout_reporting')]
]
if __name__=='__main__':
    import json
    import subprocess
    collection_only='--collection-only' in sys.argv
    names=['tests.task3.test_action_contract', 'tests.task3.test_level_gripper_control',
           'tests.task3.test_control_computation','tests.task3.test_episode_completion',
           'tests.task3.test_preview_cameras', 'tests.task3.test_h_push_success',
           'tests.h_frame_push_transport', 'tests.push_r_transport'] if collection_only else MODULES
    if '--with-upstream' in sys.argv and not collection_only:
        names = [*names, 'tests.papera.test_act_official']
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(names))
    failed=not result.wasSuccessful()
    if not collection_only:
        env=json.loads((ROOT/'config/environments.json').read_text(encoding='utf-8'))
        failed |= subprocess.call([env['collection_python'],'-X','utf8',str(Path(__file__).resolve()),'--collection-only'],cwd=ROOT)!=0
    raise SystemExit(failed)
