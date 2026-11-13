"""One real arm throughout ACT posterior, auxiliary labels and rollout targets."""
from dataclasses import replace
import unittest

import numpy as np
import torch

from tasks.data_schema import SINGLE
from training.papera.act_model import ACTConfig, ACTPolicy, compute_loss
from training.papera.data import fit_statistics, Normalizer
from rollout.action_chunks import ActionChunkScheduler, InvalidActionChunk
from rollout.policy import validate_statistics, prepare_observation
from rollout.rollout import measure_intervention


class SingleArmACTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def config(self):
        return ACTConfig(state_dim=9, action_dim=7, num_cameras=1, variant='gesture_skeleton',
            hidden_dim=32, heads=4, feedforward_dim=64, encoder_layers=1,
            decoder_layers=1, posterior_layers=1, latent_dim=4, chunk_size=5, dropout=0.)

    def test_s9_a7_posterior_forward_auxiliary_loss_and_update(self):
        torch.manual_seed(51)
        model = ACTPolicy(self.config()).train()
        images, state = torch.rand(2,1,3,64,64), torch.rand(2,9)
        actions = torch.rand(2,5,7)
        pad = torch.tensor([[False,False,False,True,True],[False]*5])
        actions[pad] = float('nan')
        output = model(images,state,actions,pad)
        self.assertEqual(output['action'].shape, (2,5,7))
        self.assertEqual(output['gesture_logits'].shape, (2,1,3))
        self.assertEqual(output['skeleton'].shape, (2,1,21,3))
        self.assertEqual(model.model.encoder_action_proj.in_features,7)
        self.assertEqual(model.model.encoder_joint_proj.in_features,9)
        targets = {'action':actions,'gesture':torch.tensor([[0],[2]]),'skeleton':torch.rand(2,1,21,3)}
        loss = compute_loss(output,targets,{'action':~pad,'gesture':torch.ones(2,1,dtype=torch.bool),
            'skeleton':torch.ones(2,1,dtype=torch.bool)})
        self.assertEqual(loss['action_count'].item(),8*7)
        self.assertEqual(loss['gesture_count'].item(),2)
        before=model.action_head.weight.detach().clone()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        loss['total'].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        optimizer.step()
        self.assertFalse(torch.equal(before,model.action_head.weight))
        model.eval()
        first=model(images,state)
        second=model(images,state,actions)
        torch.testing.assert_close(first['action'],second['action'],atol=0,rtol=0)
        self.assertTrue(torch.equal(first['mu'],torch.zeros_like(first['mu'])))

    def test_dual_checkpoint_cannot_load_into_single_arm(self):
        single=ACTPolicy(self.config())
        dual=ACTPolicy(replace(self.config(),state_dim=16,action_dim=14))
        with self.assertRaisesRegex(RuntimeError,'size mismatch'):
            single.load_state_dict(dual.state_dict(),strict=True)

    def test_normalization_schema_and_full_frame_observation(self):
        state=np.arange(36,dtype=float).reshape(4,9)/20
        action=np.zeros((4,7));action[:,3]=[3.13,-3.13,3.12,-3.12]
        episodes={'train':{'state':state,'action':action,'action_mask':np.ones(4,bool)}}
        stats=fit_statistics(episodes,['train'],schema=SINGLE)
        validate_statistics(stats,{'train':['train']},schema=SINGLE)
        with self.assertRaises(ValueError):
            validate_statistics(stats,{'train':['train']})
        normalizer=Normalizer(stats)
        decoded=normalizer.decode_action(normalizer.encode_action(action))
        np.testing.assert_allclose(np.sin(decoded[:,3]),np.sin(action[:,3]),atol=1e-6)
        rgb=np.full((480,640,3),127,np.uint8)
        image,s=prepare_observation({'oblique':rgb},state[0],normalizer,cameras=('oblique',),image_size=(64,48))
        self.assertEqual(image.shape,(1,3,48,64));self.assertEqual(s.shape,(9,))
        with self.assertRaises(ValueError):
            prepare_observation({'oblique':rgb},np.zeros(16),normalizer,cameras=('oblique',))

    def test_single_action_chunks_and_intervention_do_not_invent_second_arm(self):
        scheduler=ActionChunkScheduler('temporal_ensemble',chunk_size=5,n_action_steps=2,action_dim=7)
        chunk=np.zeros((5,7));chunk[:,3]=np.pi
        for i in range(8):
            action,report=scheduler.next_action(i,lambda:chunk)
            self.assertEqual(action.shape,(7,))
            self.assertAlmostEqual(abs(action[3]),np.pi)
        bad=ActionChunkScheduler(chunk_size=5,n_action_steps=2,action_dim=7)
        with self.assertRaises(InvalidActionChunk):
            bad.next_action(0,lambda:np.zeros((5,14)))
        requested=chunk[0];issued=requested.copy();issued[3]=-np.pi
        material,residual=measure_intervention(requested,issued,{})
        self.assertFalse(material);self.assertEqual(len(residual),7)


if __name__ == '__main__':
    unittest.main()
