"""Small, explicitly synthetic formal Prepared sources for union tests."""
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dataset.task3 import prepare_paper_dataset as prepared
from dataset.task3.curation_config import effective_config
from tasks.recorded_collection import recorded_collection_scope
from tasks.registry import get
from tests.task3.test_single_view_prepared_pack import fixture


@contextmanager
def source_prepared_fixture(root, excluded=()):
    """Yield two source Prepared inputs plus their shared whole-session split."""
    root = Path(root)
    task = get('tray_to_platform')
    canonical = task.binding()
    collection = task.config()
    historical = copy.deepcopy(canonical)
    historical['sources']['config/collection.json'] = '0' * 64
    stub = SimpleNamespace(id=task.id, binding=lambda: copy.deepcopy(canonical))
    roots = []
    with patch('tasks.registry.get', return_value=stub), \
            patch.object(prepared, 'load_episode', return_value={'load_errors': [], 'measurements': {}}):
        for label, binding in (('current', canonical), ('recorded', historical)):
            home = root / label
            home.mkdir(parents=True)
            run, manifest, _ = fixture(home)
            manifest['task_binding'] = copy.deepcopy(binding)
            manifest['config']['quality']['excluded_episodes'] = [
                {'episode_key': key, 'reason': 'explicit synthetic user exclusion'} for key in excluded]
            manifest['config_sha256'] = prepared.digest(manifest['config'])
            manifest['effective_config_sha256'] = prepared.digest(effective_config(manifest['config']))
            frozen_config = copy.deepcopy(collection)
            frozen_config['task_binding'] = copy.deepcopy(binding)
            if label == 'recorded':
                frozen_config['gesture_control'].update(close_score_min=.6, open_score_min=.5, left_close_score_min=.5)
            metadata_paths = []
            for ep in manifest['episodes']:
                ep['session'] = f'{label}_{ep["session"]}'
                ep['key'] = ep['session'] + '/episode_000000'
                rows = prepared.read_rows(run / ep['index_path'])
                for row in rows:
                    row.update(session=ep['session'], source_episode=ep['key'])
                    row['evidence']['next_control']['session'] = ep['session']
                prepared.write_rows(run / ep['index_path'], rows)
                ep['index_sha256'] = hashlib.sha256((run / ep['index_path']).read_bytes()).hexdigest()
                dataset = home / 'raw' / ep['session']
                archive = dataset / 'research/episode_000000_snapshots'
                archive.mkdir(parents=True)
                metadata = archive / 'episode_metadata.json'
                prepared.write_json(metadata, {'task_id': task.id, 'task_binding': binding,
                    'scene_id': binding['scene_id'], 'session': ep['session'], 'config': frozen_config})
                ep['source_ref'] = {'key': ep['key'], 'dataset_root': str(dataset.resolve()),
                    'metadata_path': str(metadata.resolve()), 'episode_index': 0, 'session': ep['session']}
                ep['source_fingerprints'].append({'path': str(metadata.resolve()),
                    'sha256': hashlib.sha256(metadata.read_bytes()).hexdigest(),
                    'size': metadata.stat().st_size, 'mtime_ns': metadata.stat().st_mtime_ns})
                metadata_paths.append(str(metadata.resolve()))
            prepared.write_json(run / 'manifest.json', manifest)
            proof = {'metadata_paths': metadata_paths, 'config_differences': [
                {'path': 'gesture_control.close_score_min', 'current': .8, 'recorded': .6},
                {'path': 'gesture_control.left_close_score_min', 'current': None, 'recorded': .5},
                {'path': 'gesture_control.open_score_min', 'current': .55, 'recorded': .5},
            ]}
            destination = home / 'prepared'
            with recorded_collection_scope(binding, proof) as checked_proof:
                if checked_proof is not None:
                    manifest['recorded_collection_compatibility'] = checked_proof
                    prepared.write_json(run / 'manifest.json', manifest)
                prepared.prepare(run, destination, review_mode='auto')
            roots.append(destination)
        split = root / 'master_split.json'
        prepared.write_json(split, {'train_sessions': ['current_train_session', 'recorded_train_session'],
                                   'val_sessions': ['current_val_session', 'recorded_val_session']})
        yield {'roots': roots, 'split': split, 'canonical_binding': canonical,
               'historical_binding': historical, 'reference_collection_config': collection}
