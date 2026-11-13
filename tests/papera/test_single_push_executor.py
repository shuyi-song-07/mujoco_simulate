"""Real single-Panda physics with the same A7 command guard used for collection."""
import unittest
import numpy as np

from tasks.registry import get
from tasks.runtime import create_controller
from rollout.executor import create_policy_executor, SinglePushPolicyExecutor
from simulation.mujoco.dual_arm.task_states import TaskState


class SinglePushExecutorTests(unittest.TestCase):
    def setUp(self):
        self.controller = create_controller(get('push_t').config())
        self.executor = create_policy_executor(self.controller)

    def test_factory_uses_one_real_arm_and_neutral_target_runs_physics(self):
        c,e = self.controller,self.executor
        self.assertIsInstance(e,SinglePushPolicyExecutor)
        self.assertEqual(c.state_vector().shape,(9,))
        self.assertEqual(c.action_vector_m().shape,(7,))
        self.assertEqual(len(c.arms),1)
        target=c.action_vector_m().copy()
        for _ in range(12):
            result=e.apply(target);self.assertTrue(result['accepted'],result)
            c.step(1/60)
        self.assertTrue(np.isfinite(c.data.qpos).all())
        self.assertEqual(result['accepted_target'][6],0.)
        self.assertEqual(len(result['actual_state']),9)

    def test_open_width_is_projected_to_task_closed_constraint_and_logged(self):
        c,e=self.controller,self.executor
        target=c.action_vector_m().copy();target[6]=.04
        result=e.apply(target)
        self.assertTrue(result['accepted'],result)
        self.assertEqual(result['projected_target'][6],0.)
        self.assertTrue(result['intervened']);self.assertIn(6,result['intervened_fields'])
        self.assertEqual(c.left.gripper_goal_width_m,0.)

    def test_bad_dimension_nan_orientation_and_reset_do_not_execute(self):
        c,e=self.controller,self.executor
        before=c.action_vector_m().copy()
        invalid=before.copy();invalid[0]=np.nan
        turned=before.copy();turned[3]+=.7
        for value in (np.zeros(14),invalid,turned):
            result=e.apply(value)
            self.assertFalse(result['accepted'])
            np.testing.assert_array_equal(c.action_vector_m(),before)
        c.task_state=TaskState.DONE
        self.assertFalse(e.apply(before)['accepted'])
        c.reset(3)
        with self.assertRaisesRegex(RuntimeError,'reset'):
            e.apply(c.action_vector_m())


if __name__ == '__main__':
    unittest.main()
