"""Group compatible frozen collection configurations for the curation engine.

This module only selects sessions and establishes an explicit source scope.
It does not change frame evidence, cuts, Safe Stitch, or original metadata.
"""
from __future__ import annotations

from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from dataset.task3.curation_io import discover_episodes
from tasks.recorded_collection import (ALLOWED_CONFIG_PATHS, EQUIVALENCE_FIELD,
    config_differences, configuration_equivalence, recorded_collection_scope)

ALLOWED_FIELDS = ALLOWED_CONFIG_PATHS


def recorded_groups(workflow, source=None, *, refs=None):
    """Check all sessions before making any grouped output or selecting data."""
    source = Path(source or workflow.source).resolve()
    refs = list(refs) if refs is not None else discover_episodes(source, engineering=workflow.engineering)
    if not refs:
        raise ValueError(f'No saved episodes: {source}')
    keys, session_names = set(), {}
    for ref in refs:
        key = os.path.normcase(ref['key'])
        if key in keys:
            raise ValueError(f'Duplicate episode key in selected sources: {ref["key"]}')
        keys.add(key)
        session = Path(ref['dataset_root']).resolve()
        name = os.path.normcase(session.name)
        if name in session_names and session_names[name] != session:
            raise ValueError(f'Duplicate session basename in selected sources: {session.name}')
        session_names[name] = session
    current = workflow.task.binding()
    metadata = [(ref, json.loads(Path(ref['metadata_path']).read_text(encoding='utf-8'))) for ref in refs]
    reference_source = next(((ref, m) for ref, m in metadata if m.get('task_binding') == current), None)
    if reference_source is None:
        from collection.config import recording_config
        reference = recording_config(workflow.task)
    else:
        reference = reference_source[1]['config']
    reference_config = copy.deepcopy(reference)
    reference = copy.deepcopy(reference_config)
    reference.pop('task_binding', None)
    groups, session_groups, frozen_configs = {}, {}, {}
    for ref, meta in metadata:
        binding = meta.get('task_binding')
        if (not isinstance(binding, dict) or meta.get('task_id') != workflow.task.id
                or meta.get('config', {}).get('task_binding') != binding):
            raise ValueError(f'Frozen task/config identity mismatch: {ref["metadata_path"]}')
        comparable = copy.deepcopy(binding)
        try:
            comparable['sources']['config/collection.json'] = current['sources']['config/collection.json']
        except KeyError as exc:
            raise ValueError('Missing collection source in frozen task') from exc
        if comparable != current:
            raise ValueError(f'Only historical collection settings are supported; task/assets differ: {ref["key"]}')
        config = copy.deepcopy(meta['config'])
        config.pop('task_binding', None)
        differences = config_differences(reference, config)
        for difference in differences:
            if difference['path'] not in ALLOWED_FIELDS:
                raise ValueError(f'Unapproved frozen setting difference: {ref["key"]}: {difference["path"]}')
            for field in ('current', 'recorded'):
                value = difference[field]
                if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError(f'Invalid gesture threshold: {difference}')
        group_id = binding['sources']['config/collection.json']
        if binding == current and differences:
            raise ValueError(f'Current-bound recordings disagree in frozen settings: {ref["key"]}')
        group = groups.setdefault(group_id, {'binding': binding, 'refs': [], 'sessions': {},
            'provenance': {'config_differences': differences, 'metadata_paths': []},
            'current': binding == current})
        if group['binding'] != binding or group['provenance']['config_differences'] != differences:
            raise ValueError('One recorded collection source contains inconsistent frozen settings')
        if group_id in frozen_configs and config_differences(frozen_configs[group_id], meta['config']):
            raise ValueError('One recorded collection source contains inconsistent frozen settings')
        frozen_configs[group_id] = copy.deepcopy(meta['config'])
        session_root = str(Path(ref['dataset_root']).resolve())
        if session_root in session_groups and session_groups[session_root] != group_id:
            raise ValueError('A session mixes historical collection bindings')
        session_groups[session_root] = group_id
        group['sessions'][Path(session_root).name] = session_root
        group['refs'].append(ref)
        group['provenance']['metadata_paths'].append(str(Path(ref['metadata_path']).resolve()))
    for group_id, group in groups.items():
        if not group['current'] and not group['provenance']['config_differences']:
            group['provenance'][EQUIVALENCE_FIELD] = configuration_equivalence(
                reference_config, frozen_configs[group_id],
                reference_source[0]['metadata_path'] if reference_source is not None else None)
    return sorted(groups.values(), key=lambda item: not item['current'])


def select_group(group, directory):
    """Create directory references, preserving original session/data paths."""
    directory = Path(directory).resolve()
    if directory.exists():
        if (not directory.is_dir()
                or {os.path.normcase(p.name) for p in directory.iterdir()}
                != {os.path.normcase(name) for name in group['sessions']}):
            raise ValueError('Selected session directory entries differ from the frozen selection')
        for name, target in group['sessions'].items():
            link = directory / name
            if not link.is_dir() or link.resolve() != Path(target).resolve():
                raise ValueError('Selected session reference points to another source')
        return directory
    directory.mkdir(parents=True, exist_ok=False)
    for name, target in group['sessions'].items():
        link = directory / name
        if os.name == 'nt':
            quote = lambda value: "'" + str(value).replace("'", "''") + "'"
            command = f'New-Item -ItemType Junction -Path {quote(link)} -Target {quote(target)} | Out-Null'
            subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command], check=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            link.symlink_to(target, target_is_directory=True)
        if link.resolve() != Path(target).resolve():
            raise ValueError('Session reference points to another source')
    return directory


def _selection_id(workflow, group):
    value = {'version': 'papera-native-selection-v1', 'task_id': workflow.task.id,
             'engineering': bool(workflow.engineering), 'binding': group['binding'],
             'sessions': sorted((name, str(Path(path).resolve())) for name, path in group['sessions'].items()),
             'episode_keys': sorted(ref['key'] for ref in group['refs'])}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def process_recorded(workflow, *, source=None, config=None, fresh=False, workers=2):
    """Run the same official process separately for each validated source group."""
    groups = recorded_groups(workflow, source)
    folder = workflow.root / '.local/dataset_selections' / workflow.task.id
    outputs = []
    for number, group in enumerate(groups, 1):
        selected = select_group(group, folder / _selection_id(workflow, group))
        print(f'Collection group {number}/{len(groups)}: {len(group["refs"])} episodes, '
              f'{len(group["sessions"])} sessions; current={group["current"]}', flush=True)
        scope = nullcontext() if group['current'] else recorded_collection_scope(group['binding'], group['provenance'])
        with scope:
            outputs.extend(workflow.process(source=selected, config=config, fresh=fresh, workers=workers))
    workflow._remember(outputs)
    return outputs


def prepare_recorded(workflow, runs, *, review_mode='auto', output=None,
                     split_file=None, allow_recorded=False, workers=2):
    """Export and join selected native Curation sources through formal stages.

    Source parts keep their capture bindings. The merged learning-domain
    artifact stores every original binding and episode source explicitly.
    """
    from dataset.manage_dataset import stamp, write_navigation
    from dataset.task3.merge_prepared import merge_prepared

    runs = [Path(p).resolve() for p in runs]
    if not runs or len(set(runs)) != len(runs):
        raise ValueError('Select one or more distinct Curation directories')
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('prepare workers must be within 1..4')
    validated = []
    for run in runs:
        manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('task_id') != workflow.task.id:
            raise ValueError('Cross-task Curation selection rejected')
        refs = [ep['source_ref'] for ep in manifest.get('episodes', [])]
        groups = recorded_groups(workflow, refs=refs)
        if len(groups) != 1 or groups[0]['binding'] != manifest.get('task_binding'):
            raise ValueError('Curation binding disagrees with its actual source metadata')
        group = groups[0]
        if not group['current'] and not allow_recorded:
            raise ValueError('Historical collection thresholds require --recorded-collection')
        scope = nullcontext() if group['current'] else recorded_collection_scope(group['binding'], group['provenance'])
        with scope:
            workflow._version(run)
        validated.append((run, group))
    identifier = stamp()
    target = Path(output).resolve() if output else workflow.output/'prepared'/f'prepared_{identifier}'
    if target.exists():
        raise FileExistsError(f'Choose a new Prepared output: {target}')
    source_folder = workflow.output/'prepared_sources'/f'groups_{identifier}'
    def export(item):
        number, (run, group) = item
        scope = nullcontext() if group['current'] else recorded_collection_scope(group['binding'], group['provenance'])
        with scope:
            part = workflow.prepare(run, review_mode, output=source_folder/f'group_{number}', candidate=True)
        return Path(part)
    with ThreadPoolExecutor(max_workers=min(workers,len(validated))) as pool:
        parts = list(pool.map(export,enumerate(validated,1)))
    merge_prepared(parts, target, split_file=split_file)
    write_navigation(workflow.output/'PREPARED_CURRENT.json', {
        'path': target.as_posix(), 'source_curations': [p.as_posix() for p in runs],
        'source_prepared': [p.as_posix() for p in parts], 'review_mode': review_mode,
        'split_file': str(Path(split_file).resolve()) if split_file else None,
        'training_started': False})
    print(f'联合 Prepared 已完成：{target}', flush=True)
    return target
