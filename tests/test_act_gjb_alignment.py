"""Single-arm adaptation, padding math, prior-only validation and exact resume."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from training.ACT.aligned_data import SingleArmData
from training.ACT.aligned_model import ACTConfig, ACTPolicy, compute_loss
from training.ACT.aligned_state import StatefulBatchSampler
from training.ACT.aligned_training import config_for, run_aligned
from training.ACT.train_act_variation_baseline import make_parser


class AlignmentTest(unittest.TestCase):
    def test_gjb_padding_reduction_is_full_mean_not_valid_mean(self):
        prediction=torch.ones(1,2,7,requires_grad=True)
        output={'action':prediction,'mu':torch.zeros(1,2),'logvar':torch.zeros(1,2),
                'posterior_mask':torch.tensor([True])}
        loss=compute_loss(output,{'action':torch.zeros_like(prediction)},
                          {'action':torch.tensor([[True,False]])},variant='baseline')
        self.assertEqual(loss['action'].item(),.5)
        self.assertEqual(loss['action_valid_l1'].item(),1.)
        loss['total'].backward()
        self.assertEqual(prediction.grad[:,1].abs().sum().item(),0.)

    def test_single_arm_projections_and_eval_ignore_future_actions(self):
        config=ACTConfig(hidden_dim=32,heads=4,feedforward_dim=64,encoder_layers=1,
                         decoder_layers=1,posterior_layers=1,latent_dim=8,chunk_size=2)
        model=ACTPolicy(config).eval()
        image=torch.zeros(1,1,3,32,32);state=torch.zeros(1,9)
        with torch.no_grad():
            a=model(image,state)['action']
            b=model(image,state,torch.randn(1,2,7),torch.zeros(1,2,dtype=torch.bool))['action']
        self.assertEqual(tuple(a.shape),(1,2,7))
        self.assertTrue(torch.equal(a,b))
        self.assertEqual(model.model.encoder_joint_proj.in_features,9)
        self.assertEqual(model.model.encoder_action_proj.in_features,7)
        self.assertEqual(model.model.action_head.out_features,7)

    def test_sampler_commits_consumed_not_prefetched_batches(self):
        sampler=StatefulBatchSampler(9,2,10,prefetch_safe=True)
        stream=iter(sampler);next(stream);sampler.mark_consumed()
        state=copy.deepcopy(sampler.state_dict())
        expected=next(stream);next(stream)  # issued beyond the checkpoint
        restored=StatefulBatchSampler(9,2,999,prefetch_safe=True)
        restored.load_state_dict(state)
        self.assertEqual(next(iter(restored)),expected)

    def test_real_reader_checkpoint_and_resume_match_uninterrupted_updates(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from training.ACT.aligned_inference import SingleArmPolicy
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dataset=root/'scripted'
            source=LeRobotDataset.create('local/scripted_test',fps=30,root=dataset,use_videos=False,
                features={'observation.state':{'dtype':'float32','shape':(9,), 'names':None},
                    'action':{'dtype':'float32','shape':(7,), 'names':None},
                    'observation.images.overview':{'dtype':'image','shape':(32,32,3), 'names':['height','width','channels']}})
            for episode in range(2):
                for frame in range(2):
                    source.add_frame({'observation.state':np.full(9,episode*100+frame,dtype=np.float32),
                        'action':np.array([.5,.1,.5,0,0,0,255],dtype=np.float32)+frame*.001,
                        'observation.images.overview':np.full((32,32,3),100+frame,dtype=np.uint8),
                        'task':'scripted engineering test; no human/policy success claim'})
                source.save_episode()
            source.finalize()
            data=SingleArmData(dataset,.5)
            self.assertEqual(data.stats['state_mean'],[.5]*9)
            self.assertTrue(all(x>=.01 for x in data.stats['action_std']))
            parser=make_parser()
            args=parser.parse_args(['--dataset-root',str(dataset),'--output-dir',str(root/'full'),
                '--device','cpu','--no-pretrained-backbone','--steps','2',
                '--batch-size','1','--eval-steps','1','--save-freq','1','--cpu-threads','1'])
            config=config_for(args)
            config['model'].update(hidden_dim=32,heads=4,feedforward_dim=64,
                                   encoder_layers=1,decoder_layers=1,posterior_layers=1,latent_dim=8)
            with patch('training.ACT.aligned_training.config_for',return_value=copy.deepcopy(config)):
                run_aligned(args)
                args.output_dir=root/'resumed';args.stop_after=1
                run_aligned(args)
                args.resume=root/'resumed/checkpoints/last.pt';args.stop_after=None
                run_aligned(args)
            full=torch.load(root/'full/checkpoints/last.pt',weights_only=True,map_location='cpu')
            resumed=torch.load(args.resume,weights_only=True,map_location='cpu')
            self.assertEqual(full['step'],resumed['step'])
            for key,value in full['model'].items():
                self.assertTrue(torch.equal(value,resumed['model'][key]),key)
            policy=SingleArmPolicy(args.resume,'cpu',2)
            action=policy.select_action({'observation.state':torch.zeros(9),
                'observation.images.overview':torch.zeros(3,32,32)})
            self.assertEqual(tuple(action.shape),(7,))
            self.assertTrue(torch.isfinite(action).all())
            policy.reset();self.assertEqual(len(policy.queue),0)


if __name__=='__main__':unittest.main()
