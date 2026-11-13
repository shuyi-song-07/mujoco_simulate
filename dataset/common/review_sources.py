"""Resolve original review videos independently of the single training view."""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from dataset.task3.source_paths import dataset_path

REVIEW_CAMERAS = ('top', 'front', 'side')
CURRENT_REVIEW_CAMERAS = ('top', 'front', 'oblique')
PREFIX = 'observation.images.'
DERIVED_SCHEMA = 'papera-oblique-derived-raw-v1'


def select_review_sources(sources):
    """Keep historical three-view review; accept native single-view recordings.

    A partial multi-view set is an error, not permission to hide a missing view.
    Linked derivatives resolve their original dataset before calling this path.
    """
    names = [s.get('camera') for s in sources]
    if len(set(names)) != len(names):
        return []
    for layout in (CURRENT_REVIEW_CAMERAS, REVIEW_CAMERAS):
        if set(layout).issubset(names):
            by_name={s['camera']:s for s in sources}
            return [by_name[name] for name in layout]
    return list(sources) if names == ['oblique'] else []


def _inside(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Review source path escapes its declared dataset root')
    return path


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def provenance_path(dataset):
    root = Path(dataset).resolve()
    for relative in ('meta/oblique_provenance.json', 'provenance/oblique_render.json'):
        path = _inside(root, root / relative)
        if path.is_file():
            return path
    return None


def review_sources(dataset, episode_index, *, expected_frames=None, override_root=None):
    """Resolve any indexed episode, including a span of a shared video shard.

    A derivative may link external original videos only through its declared
    provenance. Matching frozen task metadata and the episode's original raw
    log prevent a same-length, wrong-demonstration video override. Reading
    these three display streams is independent of training RGB verification.
    """
    from dataset.task3.curation_io import fingerprint_file

    derived = Path(dataset).resolve()
    linked = provenance_path(derived)
    original = derived
    if linked:
        p = _json(linked)
        schema = p.get('schema', p.get('version'))
        if schema == DERIVED_SCHEMA:
            value = p.get('source_dataset_root')
        elif schema == 'task3-oblique-rgb-only-v1':
            if p.get('episode_index') != episode_index:
                raise ValueError('Pilot provenance refers to a different episode')
            value = p.get('source')
        else:
            raise ValueError('Unknown oblique provenance; review source cannot be inferred')
        if not isinstance(value, str) or not value.strip() or not Path(value).is_absolute():
            raise ValueError('Review source requires an absolute source_dataset_root')
        original = Path(value).resolve()
        if original == derived:
            raise ValueError('Derived review source cannot refer to itself')
        for relative in ('meta/task3.json', f'research/episode_{episode_index:06d}.jsonl'):
            paths = [_inside(root, root / relative) for root in (original, derived)]
            if fingerprint_file(paths[0])['sha256'] != fingerprint_file(paths[1])['sha256']:
                raise ValueError(f'Review nonvisual source mismatch: {relative}')
    if override_root is not None and Path(override_root).resolve() != original:
        raise ValueError('Review override does not match the declared original dataset')
    info = _json(_inside(original, original / 'meta/info.json'))
    available = {k[len(PREFIX):] for k in info.get('features', {}) if k.startswith(PREFIX)}
    cameras = next((layout for layout in (CURRENT_REVIEW_CAMERAS, REVIEW_CAMERAS)
                    if set(layout).issubset(available)), ())
    if not cameras and available == {'oblique'} and not linked:cameras=('oblique',)
    if not cameras:
        return []
    indexes = [row for path in sorted((original / 'meta/episodes').rglob('*.parquet'))
               for row in pq.read_table(_inside(original, path)).to_pylist()
               if row['episode_index'] == episode_index]
    if len(indexes) != 1:
        raise ValueError('Original review episode index is missing or duplicated')
    episode = indexes[0]
    if expected_frames is not None and int(episode['length']) != expected_frames:
        raise ValueError('Review episode length differs from the derived frame map')
    result = []
    for camera in cameras:
        key = PREFIX + camera
        prefix = 'videos/' + key
        path = dataset_path(original, info['video_path'], video_key=key,
                            chunk_index=episode[prefix + '/chunk_index'],
                            file_index=episode[prefix + '/file_index'])
        path = _inside(original, path)
        if not path.is_file() or path.suffix.lower() != '.mp4':
            raise ValueError(f'Original review video is missing: {path}')
        start, end, fps = float(episode[prefix + '/from_timestamp']), float(episode[prefix + '/to_timestamp']), float(info['fps'])
        if fps <= 0 or abs(end - start - int(episode['length']) / fps) > .001:
            raise ValueError('Original review video timing disagrees with episode length')
        result.append({'camera': camera, 'video_key': key, 'path': str(path),
                       'allowed_root': str(original), 'start_s': start, 'end_s': end,
                       'fps': fps, 'frame_count': int(episode['length'])})
    return result
