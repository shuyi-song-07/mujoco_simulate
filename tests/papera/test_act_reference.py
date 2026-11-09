"""Reference profile interface checks without data downloads or long training."""
import copy
import io
import json
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from training.papera.act_model import ACTConfig, ACTPolicy
from training.papera.reference import adapt_batch, BatchAdapter, posterior_validation, run_reference
from training.papera.train import main, validate_config


def original_batch(count, target=1., length=400):
    return (torch.zeros(count,1,3,32,32), torch.zeros(count,14),
            torch.full((count,length,14),float(target)), torch.zeros(count,length,dtype=torch.bool))


class ReferenceProfileTests(unittest.TestCase):
    def test_adapter_does_not_shift_actions_or_invent_robot_state(self):
        raw=list(original_batch(2))
        raw[2]=torch.arange(2*400*14,dtype=torch.float32).reshape(2,400,14)
        raw[3][1,20:]=True
        actual=adapt_batch(raw)
        self.assertIs(actual['images'],raw[0])
        self.assertIs(actual['state'],raw[1])
        torch.testing.assert_close(actual['action'],raw[2][:,:100],rtol=0,atol=0)
        torch.testing.assert_close(actual['is_pad'],raw[3][:,:100],rtol=0,atol=0)
        self.assertFalse(actual['gesture_mask'].any())
        self.assertFalse(actual['skeleton_mask'].any())
        with self.assertRaises(ValueError):
            adapt_batch((raw[0],torch.zeros(2,16),raw[2],raw[3]))

    def test_adapter_keeps_original_loader_lazy_rng_timing(self):
        class Loader:
            dataset=object()
            calls=0
            def __len__(self): return 1
            def __iter__(self):
                self.calls+=1
                yield original_batch(2)
        source=Loader(); wrapped=BatchAdapter(source)
        self.assertIs(wrapped.dataset,source.dataset)
        iterator=iter(wrapped)
        self.assertEqual(source.calls,0)
        next(iterator)
        self.assertEqual(source.calls,1)

    def test_posterior_validation_uses_equal_batch_mean_and_restores_mode(self):
        class PosteriorFixture(nn.Module):
            def __init__(self): super().__init__(); self.calls=0
            def forward(self,images,state,action,is_pad,*,posterior_in_eval=False):
                if self.training or not posterior_in_eval:
                    raise AssertionError('Reference validation must explicitly request an eval-mode posterior')
                self.calls+=1
                return {'action':torch.zeros_like(action),'mu':state.new_zeros(len(state),2),
                        'logvar':state.new_zeros(len(state),2),
                        'posterior_mask':torch.ones(len(state),dtype=torch.bool)}
        model=PosteriorFixture();model.train()
        batches=[adapt_batch(original_batch(8,1)),adapt_batch(original_batch(2,3))]
        mean=lambda rows:{key:sum(row[key]/len(rows) for row in rows) for key in rows[0]}
        result=posterior_validation(model,batches,torch.device('cpu'),mean)
        self.assertAlmostEqual(result['l1'],2.)
        self.assertAlmostEqual(result['loss'],2.)
        self.assertEqual(model.calls,2)
        self.assertTrue(model.training)

    def test_s14_reference_does_not_relax_formal_pack_contract(self):
        ACTConfig(state_dim=14,action_dim=14,num_cameras=1,chunk_size=100)
        root=Path(__file__).resolve().parents[2]
        config=json.loads((root/'config/training_baseline.json').read_text(encoding='utf-8'))
        config['pack_path']='unused'
        config['model']['state_dim']=14
        with self.assertRaisesRegex(ValueError,'S16'):
            validate_config(config)

    def test_eval_prior_remains_default_and_posterior_is_explicit(self):
        previous=torch.get_num_threads();torch.set_num_threads(2)
        try:
            config=ACTConfig(state_dim=14,action_dim=14,num_cameras=1,chunk_size=3,
                hidden_dim=32,heads=4,feedforward_dim=64,encoder_layers=1,
                decoder_layers=2,posterior_layers=1,latent_dim=8,dropout=.5)
            model=ACTPolicy(config).eval()
            image=torch.rand(2,1,3,64,64);state=torch.rand(2,14)
            action=torch.rand(2,3,14);pad=torch.zeros(2,3,dtype=torch.bool)
            with torch.no_grad():
                prior=model(image,state,action,pad)
                posterior=model(image,state,action,pad,posterior_in_eval=True)
            self.assertFalse(prior['posterior_mask'].any())
            self.assertEqual(int(prior['mu'].count_nonzero()),0)
            self.assertTrue(posterior['posterior_mask'].all())
            self.assertGreater(int(posterior['mu'].count_nonzero()),0)
            self.assertFalse(model.training)
            self.assertTrue(all(not layer.training for layer in model.modules() if isinstance(layer,nn.Dropout)))
        finally:
            torch.set_num_threads(previous)

    def test_reference_cli_rejects_task_overrides(self):
        prefix=['--official-reference','unused','--output','.local/checks/reference-unit']
        for override in (['--pack','unused'],['--resume','unused'],['--steps','5'],
                         ['--variant','gesture'],['--seed','1'],['--task','tray_to_platform']):
            with self.subTest(override=override), patch('training.papera.reference.run_reference') as runner:
                with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    main(prefix+override)
                runner.assert_not_called()
        with patch('training.papera.reference.run_reference',return_value=Path('result')) as runner:
            main(prefix+['--smoke-test'])
            self.assertEqual(runner.call_args.kwargs['epochs'],2)

    def test_reference_cannot_write_formal_outputs(self):
        root=Path(__file__).resolve().parents[2]
        with self.assertRaisesRegex(ValueError,'isolated'):
            run_reference('unused',None,root/'outputs/training/not-a-reference',epochs=2)


if __name__=='__main__':
    unittest.main()
