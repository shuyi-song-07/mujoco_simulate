"""Combine task physics/criteria with the selected algorithm's execution profile.

Only the listed algorithm fields may come from the shared DP profile. In
particular, tray success conditions never replace a Push task's conditions.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DP_FIELDS = ('policy_algorithm', 'execution', 'n_action_steps',
             'checkpoint_selection', 'expected_checkpoint_step')


def task_evaluation(task, algorithm='act'):
    if algorithm not in ('act', 'diffusion'):
        raise ValueError('Evaluation algorithm must be act or diffusion')
    path = ROOT / task.manifest['evaluation_config']
    settings = json.loads(path.read_text(encoding='utf-8-sig'))
    if algorithm == 'diffusion':
        profile = json.loads((ROOT / 'config/evaluation_diffusion.json').read_text(encoding='utf-8-sig'))
        settings.update({key: profile[key] for key in DP_FIELDS})
    return settings
