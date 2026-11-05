"""Independent pinned-source checks for the official ACT task adapter.

The original snapshot is a read-only test reference, never a runtime import.
PAPERA_ACT_UPSTREAM can point at another checkout of the pinned ACT repository.
Without that optional snapshot, adapter/loss checks remain in test_act_model.
"""
from __future__ import annotations

import ast
from dataclasses import replace
import os
from pathlib import Path
import sys
import types
import unittest

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from training.papera.act_model import ACTConfig, ACTPolicy, compute_loss


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path(os.environ.get('PAPERA_ACT_UPSTREAM',
    ROOT/'.local/checks/official_reproduction_20261105/act/official'))


def _reference_modules():
    """Execute original modules separately; relocate imports, omit debug shell."""
    folder = UPSTREAM/'detr/models'
    if not (folder/'detr_vae.py').is_file():
        raise unittest.SkipTest('Optional fixed original ACT checkout is unavailable')
    package_name = '_papera_act_original_742c753'
    package = types.ModuleType(package_name)
    package.__path__ = [str(folder)]
    sys.modules[package_name] = package
    result = {}
    for name in ('transformer', 'position_encoding', 'backbone', 'detr_vae'):
        path = folder/f'{name}.py'
        source = path.read_text(encoding='utf-8')
        source = '\n'.join(line for line in source.splitlines()
            if line.strip() not in ('import IPython', 'e = IPython.embed'))
        # These annotation/distributed helpers have no role in tested forward
        # math. Models, attention, spatial encoding and CVAE come from the
        # independent original files, not the production vendor classes.
        source = source.replace('from util.misc import ',
                                'from training.papera.official.misc import ')
        module = types.ModuleType(f'{package_name}.{name}')
        module.__package__, module.__file__ = package_name, str(path)
        sys.modules[module.__name__] = module
        exec(compile(source, str(path), 'exec'), module.__dict__)
        result[name] = module
    # Execute the authors' actual policy loss methods, avoiding their CLI and
    # optional IPython imports. The constructor is intentionally not called.
    policy_tree = ast.parse((UPSTREAM/'policy.py').read_text(encoding='utf-8'))
    nodes = [node for node in policy_tree.body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in ('ACTPolicy', 'kl_divergence')]
    scope = {'torch': torch, 'nn': nn, 'F': F,
             'transforms': torchvision.transforms}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(UPSTREAM/'policy.py'), 'exec'), scope)
    result['policy_class'] = scope['ACTPolicy']
    return result


def _reference_model(config, modules):
    """Original graph with only qpos projection dimensions adapted to S16."""
    t = modules['transformer']
    b = modules['backbone']
    p = modules['position_encoding']
    d = modules['detr_vae']
    backbone = torchvision.models.resnet18(weights=None, norm_layer=b.FrozenBatchNorm2d)
    body = b.BackboneBase(backbone, True, 512, False)
    joiner = b.Joiner(body, p.PositionEmbeddingSine(config.hidden_dim//2, normalize=True))
    joiner.num_channels = 512
    transformer = t.Transformer(
        d_model=config.hidden_dim, nhead=config.heads,
        num_encoder_layers=config.encoder_layers,
        num_decoder_layers=config.decoder_layers,
        dim_feedforward=config.feedforward_dim, dropout=config.dropout,
        activation='relu', normalize_before=False, return_intermediate_dec=True)
    layer = t.TransformerEncoderLayer(config.hidden_dim, config.heads,
        config.feedforward_dim, config.dropout, 'relu', False)
    encoder = t.TransformerEncoder(layer, config.posterior_layers, norm=None)
    model = d.DETRVAE([joiner], transformer, encoder, 14, config.chunk_size,
                     ['oblique'] if config.num_cameras == 1 else ['top', 'front', 'oblique'])
    model.input_proj_robot_state = nn.Linear(config.state_dim, config.hidden_dim)
    model.encoder_joint_proj = nn.Linear(config.state_dim, config.hidden_dim)
    if config.latent_dim != 32:
        model.latent_dim = config.latent_dim
        model.latent_proj = nn.Linear(config.hidden_dim, 2*config.latent_dim)
        model.latent_out_proj = nn.Linear(config.latent_dim, config.hidden_dim)
    return model


class OfficialACTSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = _reference_modules()
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(81)
        self.config = ACTConfig(num_cameras=1, hidden_dim=32, heads=4,
            feedforward_dim=64, encoder_layers=1, decoder_layers=2,
            posterior_layers=1, latent_dim=4, chunk_size=30, dropout=.1,
            backbone_weights=None)
        self.policy = ACTPolicy(self.config)
        self.reference = _reference_model(self.config, self.modules)
        self.reference.load_state_dict(self.policy.model.state_dict(), strict=True)
        self.images = torch.rand(2, 1, 3, 64, 64)
        self.state = torch.rand(2, 16)
        self.actions = torch.rand(2, 30, 14)
        self.padding = torch.zeros(2, 30, dtype=torch.bool)
        self.padding[0, 21:] = True
        self.padding[1, 28:] = True

    def test_original_postln_relu_graph_and_parameter_names_match(self):
        # This compares classes loaded from the separate upstream files.
        self.assertNotEqual(type(self.reference).__module__, type(self.policy.model).__module__)
        self.assertEqual(set(self.reference.state_dict()), set(self.policy.model.state_dict()))
        encoder = self.policy.model.transformer.encoder
        self.assertIsNone(encoder.norm)
        self.assertFalse(encoder.layers[0].normalize_before)
        self.assertIs(encoder.layers[0].activation, F.relu)
        self.assertEqual(self.policy.model.encoder_joint_proj.in_features, 16)
        self.assertEqual(self.policy.model.action_head.out_features, 14)

    def test_original_training_actions_posterior_and_rng_match(self):
        self.policy.train(); self.reference.train()
        clean = self.actions.masked_fill(self.padding[..., None], 0)
        torch.manual_seed(933)
        actual = self.policy(self.images, self.state, self.actions, self.padding)
        actual_rng = torch.get_rng_state()
        torch.manual_seed(933)
        image = torchvision.transforms.functional.normalize(self.images,
            [.485, .456, .406], [.229, .224, .225])
        action, _, (mu, logvar) = self.reference(self.state, image, None, clean, self.padding)
        for expected, received in ((action, actual['action']), (mu, actual['mu']), (logvar, actual['logvar'])):
            torch.testing.assert_close(received, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_rng, torch.get_rng_state(), rtol=0, atol=0)

    def test_original_prior_zero_eval_matches_even_if_future_labels_are_supplied(self):
        self.policy.eval(); self.reference.eval()
        actual = self.policy(self.images, self.state,
            torch.full_like(self.actions, float('nan')), self.padding)
        image = torchvision.transforms.functional.normalize(self.images,
            [.485, .456, .406], [.229, .224, .225])
        expected, _, (mu, logvar) = self.reference(self.state, image, None)
        self.assertIsNone(mu); self.assertIsNone(logvar)
        torch.testing.assert_close(actual['action'], expected, rtol=0, atol=0)
        self.assertFalse(actual['posterior_mask'].any())
        torch.testing.assert_close(self.policy.predict(self.images, self.state), expected, rtol=0, atol=0)

    def test_original_policy_l1_kl_and_core_backward_match(self):
        self.policy.train()
        cls = self.modules['policy_class']
        ref_policy = cls.__new__(cls)
        nn.Module.__init__(ref_policy)
        ref_policy.model, ref_policy.kl_weight = self.reference, 10.
        ref_policy.train()
        clean = self.actions.masked_fill(self.padding[..., None], 0)
        torch.manual_seed(98)
        actual = compute_loss(self.policy(self.images, self.state, clean, self.padding),
            {'action': clean}, {'action': ~self.padding}, {'action': 1., 'kl': 10.})
        torch.manual_seed(98)
        expected = ref_policy(self.state, self.images, clean, self.padding)
        for key, original in (('action', 'l1'), ('kl', 'kl'), ('total', 'loss')):
            torch.testing.assert_close(actual[key], expected[original], rtol=1e-6, atol=1e-6)
        valid = (~self.padding).sum()*14
        torch.testing.assert_close(actual['action_valid_l1'], actual['action']*self.actions.numel()/valid,
                                   rtol=1e-6, atol=1e-6)
        actual['total'].backward(); expected['loss'].backward()
        grads = dict(self.reference.named_parameters())
        for name, parameter in self.policy.model.named_parameters():
            reference_gradient = grads[name].grad
            if reference_gradient is None:
                self.assertIsNone(parameter.grad, name)
            else:
                self.assertIsNotNone(parameter.grad, name)
                torch.testing.assert_close(parameter.grad, reference_gradient, rtol=2e-5, atol=2e-6, msg=name)

    def test_rectangular_rgb_and_three_camera_configuration_use_original_graph(self):
        policy = ACTPolicy(replace(self.config, num_cameras=3)).eval()
        reference = _reference_model(policy.config, self.modules).eval()
        reference.load_state_dict(policy.model.state_dict(), strict=True)
        rgb = torch.rand(1, 3, 3, 48, 64)
        state = torch.rand(1, 16)
        expected, _, _ = reference(state, torchvision.transforms.functional.normalize(rgb,
            [.485, .456, .406], [.229, .224, .225]), None)
        torch.testing.assert_close(policy.predict(rgb, state), expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
