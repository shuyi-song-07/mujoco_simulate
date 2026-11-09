"""Retained DP checkpoints: metadata and file I/O only, never training math.

The index binds each full checkpoint to one run/config/source. It does not
infer candidates from filenames or run extra validation. Small metadata is
digested; checkpoint contents are verified by the normal inference loader.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re


VERSION = 'papera-diffusion-checkpoint-candidates-v1'


def run_signature(run_dir, task_id, pack_path, config, sources):
    binding = {'run_dir': str(Path(run_dir).resolve()), 'task_id': task_id,
               'pack_path': str(Path(pack_path).resolve()), 'config': config,
               'source_sha256': sources}
    encoded = json.dumps(binding, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def candidate_due(config, step):
    every = config.get('candidate_save_every')
    return bool(every and step > 0 and (step % every == 0 or step == config['steps']))


def _index(run_dir, signature):
    path = Path(run_dir)/'checkpoints/candidates.json'
    if not path.exists():
        raise ValueError('DP candidate requires a run-bound checkpoints/candidates.json')
    value = json.loads(path.read_text(encoding='utf-8'))
    if (value.get('version') != VERSION or value.get('run_signature') != signature
            or not isinstance(value.get('candidates'), list)):
        raise ValueError('DP candidate index belongs to a different run/config/source')
    return value


def _write_index(run_dir, value):
    path = Path(run_dir)/'checkpoints/candidates.json'
    temporary = path.with_suffix('.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def retain_candidate(run_dir, save, config, step, signature, validation=None):
    if not candidate_due(config, step):
        return None
    if validation is not None and (validation.get('step') != step
            or validation.get('event') != 'validation' or validation.get('ema') is not True):
        raise ValueError('DP candidate validation must describe this exact EMA update')
    path = Path(run_dir)/'checkpoints'/f'step_{step:06d}.pt'
    if path.exists():
        raise ValueError(f'DP candidate already exists and will not be overwritten: {path}')
    index_path = path.parent/'candidates.json'
    index = (_index(run_dir, signature) if index_path.exists()
             else {'version': VERSION, 'run_signature': signature, 'candidates': []})
    if any(row.get('step') == step for row in index['candidates']):
        raise ValueError('DP candidate index already contains this step')
    save(path.name)  # The same full EMA/optimizer/sampler/RNG payload as last.pt.
    entry = {'checkpoint': path.name, 'step': step, 'saved_step': step,
             'file_size_bytes': path.stat().st_size,
             'validation_step': step if validation is not None else None,
             'validation_source': 'metrics.jsonl:validation' if validation is not None else None,
             'validation': copy.deepcopy(validation), 'selection': 'not_selected'}
    index['candidates'].append(entry)
    _write_index(run_dir, index)
    return entry


def reconcile_resume_candidates(run_dir, signature, restored_step):
    """Withdraw uncommitted future entries, keeping their original files.

    Normal saves commit last.pt before candidates, so this is primarily a
    guard for an explicitly restored earlier checkpoint. An orphan file is
    never selected or overwritten automatically.
    """
    if not (Path(run_dir)/'checkpoints/candidates.json').exists():
        return
    index = _index(run_dir, signature)
    if any(type(row.get('saved_step')) is not int for row in index['candidates']):
        raise ValueError('DP candidate index has an invalid saved step')
    future = [row for row in index['candidates'] if row['saved_step'] > restored_step]
    if future:
        index['candidates'] = [row for row in index['candidates'] if row['saved_step'] <= restored_step]
        index.setdefault('retired_candidates', []).extend(
            dict(row, retirement_reason='after_restored_checkpoint', restored_step=restored_step)
            for row in future)
        _write_index(run_dir, index)


def candidate_record(run_dir, checkpoint, signature, config):
    if checkpoint in ('best', 'last'):
        return None
    if not isinstance(checkpoint, str) or re.fullmatch(r'step_[0-9]{6}', checkpoint) is None:
        raise ValueError('Select best, last, or an indexed DP step_NNNNNN candidate')
    step = int(checkpoint[5:])
    if step > config['steps'] or not candidate_due(config, step):
        raise ValueError('DP candidate is outside the frozen retention schedule')
    index = _index(run_dir, signature)
    entries = [row for row in index['candidates'] if row.get('checkpoint') == f'{checkpoint}.pt']
    if len(entries) != 1:
        raise ValueError('DP candidate needs exactly one matching index record')
    record = entries[0]
    if record.get('step') != step or record.get('saved_step') != step:
        raise ValueError('DP candidate name and saved step differ')
    path = Path(run_dir)/'checkpoints'/record['checkpoint']
    size = record.get('file_size_bytes')
    if type(size) is not int or size <= 0 or not path.is_file() or path.stat().st_size != size:
        raise ValueError('DP candidate file is missing or differs from the indexed size')
    validation = record.get('validation')
    if validation is None:
        if record.get('validation_step') is not None or record.get('validation_source') is not None:
            raise ValueError('DP candidate has inconsistent validation evidence')
    elif (record.get('validation_step') != step or validation.get('step') != step
          or validation.get('event') != 'validation' or validation.get('ema') is not True
          or record.get('validation_source') != 'metrics.jsonl:validation'):
        raise ValueError('DP candidate validation must describe its saved EMA step')
    return record
