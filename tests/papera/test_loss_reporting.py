"""Rollout reports the actual optimized objective, including DP auxiliary CE."""
import json
from pathlib import Path
import tempfile
import unittest
from rollout.rollout import training_loss_summary


class LossReportingTests(unittest.TestCase):
    def summary(self, rows, **policy):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'metrics.jsonl').write_text(
                ''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
            return training_loss_summary(directory, {'checkpoint_step': 100, **policy})

    def test_dp_gesture_total_is_not_noise_only(self):
        result=self.summary([
            {'event':'train','step':100,'noise_mse':.02,'gesture_loss':1.,'gesture_weight':.1,'total_loss':.12},
            {'event':'validation','step':100,'noise_mse':.03,'gesture_loss':.9}],
            algorithm='diffusion_policy',training_variant='gesture',variant='diffusion_gesture')
        self.assertEqual(result['training_metric'],'total_loss')
        self.assertEqual(result['checkpoint_training']['value'],.12)
        self.assertEqual(result['checkpoint_training']['components']['noise_mse'],.02)
        self.assertEqual(result['checkpoint_validation']['value'],.03)
        self.assertEqual(result['validation_metric'],'ema_noise_mse')

    def test_dp_baseline_historical_noise_is_total(self):
        result=self.summary([{'event':'train','step':100,'noise_mse':.02}],algorithm='diffusion_policy')
        self.assertEqual(result['training_metric'],'noise_mse')
        self.assertEqual(result['checkpoint_training']['value'],.02)

    def test_act_total_and_prior_validation_remain_distinct(self):
        result=self.summary([{'event':'train','step':100,'total':2.5,'action':.5,'kl':.2},
                             {'event':'validation','step':100,'prior_action_l1':.8}],variant='baseline')
        self.assertEqual(result['checkpoint_training']['value'],2.5)
        self.assertEqual(result['checkpoint_validation']['value'],.8)

    def test_missing_total_does_not_fabricate_gesture_objective(self):
        result=self.summary([{'event':'train','step':100,'noise_mse':.02}],
                            algorithm='diffusion_policy',training_variant='gesture')
        self.assertIsNone(result['checkpoint_training'])


if __name__=='__main__': unittest.main()
