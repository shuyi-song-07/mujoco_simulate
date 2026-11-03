"""Merge verified Prepared inputs without changing their frozen frame choices.

The global task binding describes the current learning domain. Original
collection bindings remain explicit on every episode and in the source ledger.
Only recorded gesture confidence thresholds may vary between collections.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil

from dataset.task3.curation_config import effective_config, validate_config
from tasks.recorded_collection import (ALLOWED_CONFIG_PATHS, SOURCE, EQUIVALENCE_FIELD,
    config_differences as _differences, configuration_equivalence, recorded_collection_scope)

VERSION = 'papera-prepared-collection-union-v1'
UNION_FIELDS = ('source_task_bindings', 'source_prepared_manifests', 'dataset_collection_compatibility')


class ValidatedManifest(dict):
    """A JSON-compatible manifest returned after the union checks below."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _same_domain(binding, canonical):
    if not isinstance(binding, dict) or not isinstance(binding.get('sources'), dict):
        return False
    if not isinstance(binding['sources'].get(SOURCE), str) or not re.fullmatch(r'[0-9a-f]{64}', binding['sources'][SOURCE]):
        return False
    normalized = copy.deepcopy(binding)
    normalized['sources'][SOURCE] = canonical['sources'][SOURCE]
    return normalized == canonical


def _collection_differences(reference, recorded):
    left, right = copy.deepcopy(reference), copy.deepcopy(recorded)
    left.pop('task_binding', None)
    right.pop('task_binding', None)
    differences = _differences(left, right)
    for difference in differences:
        if difference['path'] not in ALLOWED_CONFIG_PATHS:
            raise ValueError(f'Collection scene/camera/physical setting differs: {difference["path"]}')
        for field in ('current', 'recorded'):
            value = difference[field]
            if value is None and difference['path'] == 'gesture_control.left_close_score_min':
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid recorded gesture confidence threshold')
    return differences


def _curation_math(config):
    result = effective_config(validate_config(config))
    result.pop('quality')  # Exclusion ledgers select episodes, never recut frames.
    return result


def _excluded(config, selection):
    result = {item['episode_key']: item['reason'] for item in config['quality']['excluded_episodes']}
    for decision in selection:
        if decision.get('included') is False:
            result.setdefault(decision['key'], decision.get('reason') or 'source_prepared_exclusion')
    return result


def validate_manifest(manifest):
    """Validate frozen union declarations without Raw, video or live task access.

    This does not grant acceptance to an arbitrary legacy binding. The merge
    builder verifies actual metadata before writing this explicit source ledger;
    later Pack readers check the same ledger and episode associations unchanged.
    """
    from tasks.registry import SCHEMA

    if not isinstance(manifest, dict) or not all(field in manifest for field in UNION_FIELDS):
        raise ValueError('Missing explicit Prepared collection union provenance')
    canonical = manifest.get('task_binding')
    proof = manifest['dataset_collection_compatibility']
    if (manifest.get('pipeline_stage') != 'prepared' or not manifest.get('complete')
            or not isinstance(canonical, dict) or canonical.get('schema') != SCHEMA
            or manifest.get('cameras') != ['oblique']
            or manifest.get('state_dim') != 16 or manifest.get('action_dim') != 14
            or canonical.get('task_id') != manifest.get('task_id')
            or not isinstance(proof, dict) or proof.get('version') != VERSION
            or proof.get('learning_domain_task_binding') != canonical
            or proof.get('frame_selection') != 'unchanged_frozen_prepared'):
        raise ValueError('Invalid learning-domain Prepared collection union')
    bindings = manifest['source_task_bindings']
    if (not isinstance(bindings, list) or not bindings
            or len({_digest(binding) for binding in bindings}) != len(bindings)
            or any(not _same_domain(binding, canonical) for binding in bindings)):
        raise ValueError('Source task bindings differ outside collection configuration')
    binding_ids = {_digest(binding) for binding in bindings}
    reference = proof.get('reference_collection_config')
    if not isinstance(reference, dict) or reference.get('task_binding') != canonical:
        raise ValueError('Missing canonical frozen collection configuration')
    groups = proof.get('groups')
    if not isinstance(groups, list) or {_digest(g.get('source_task_binding')) for g in groups} != binding_ids:
        raise ValueError('Collection groups do not cover the original task bindings')
    if len(groups) != len(bindings):
        raise ValueError('Duplicate source collection group')
    group_by_binding = {}
    for group in groups:
        binding = group['source_task_binding']
        config = group.get('frozen_collection_config')
        if not isinstance(config, dict) or config.get('task_binding') != binding:
            raise ValueError('Frozen collection configuration task identity mismatch')
        if group.get('config_differences') != _collection_differences(reference, config):
            raise ValueError('Collection compatibility differences are not the frozen configuration differences')
        fingerprints = group.get('metadata_fingerprints')
        if (not isinstance(fingerprints, list) or not fingerprints
                or len({f.get('path') for f in fingerprints}) != len(fingerprints)
                or set(group.get('metadata_paths', [])) != {f.get('path') for f in fingerprints}
                or any(not isinstance(f.get('path'), str) or not Path(f['path']).is_absolute()
                       or not isinstance(f.get('sha256'), str) or len(f['sha256']) != 64
                       or type(f.get('size')) is not int or f['size'] <= 0 for f in fingerprints)):
            raise ValueError('Collection metadata evidence is missing or malformed')
        group_by_binding[_digest(binding)] = group
    sources = manifest['source_prepared_manifests']
    if not isinstance(sources, list) or not sources:
        raise ValueError('Missing source Prepared manifests')
    source_by_path, prohibited, expected_keys = {}, {}, set()
    config = validate_config(manifest['config'])
    if (_digest(config) != manifest.get('config_sha256')
            or _digest(effective_config(config)) != manifest.get('effective_config_sha256')):
        raise ValueError('Merged frozen curation configuration hash mismatch')
    mathematics = _curation_math(config)
    for source in sources:
        path = source.get('path')
        if not isinstance(path, str) or not Path(path).is_absolute() or path in source_by_path:
            raise ValueError('Source Prepared paths must be unique absolute paths')
        if not isinstance(source.get('manifest_sha256'), str) or len(source['manifest_sha256']) != 64:
            raise ValueError('Missing source Prepared manifest fingerprint')
        if _digest(source.get('task_binding')) not in binding_ids:
            raise ValueError('Source Prepared binding is absent from source task bindings')
        config = validate_config(source['config'])
        if (_digest(config) != source.get('config_sha256')
                or _digest(effective_config(config)) != source.get('effective_config_sha256')
                or _curation_math(config) != mathematics):
            raise ValueError('Source Prepared Safe Stitch/curation configuration differs')
        keys = source.get('episode_keys')
        if not isinstance(keys, list) or len(set(keys)) != len(keys) or set(keys) & expected_keys:
            raise ValueError('Duplicate episode keys across source Prepared inputs')
        expected_keys.update(keys)
        selection = source.get('selection')
        if not isinstance(selection, list):
            raise ValueError('Missing original source Prepared selection')
        selected = {d['key'] for d in selection if d.get('included') is True}
        if selected != set(keys):
            raise ValueError('Source Prepared episode keys differ from its frozen selection')
        prohibited.update(_excluded(config, selection))
        source_by_path[path] = source
    if prohibited.keys() & expected_keys:
        raise ValueError('An excluded episode cannot re-enter a merged Prepared')
    merged_excluded = {x['episode_key']: x['reason'] for x in manifest['config']['quality']['excluded_episodes']}
    if not prohibited.keys() <= merged_excluded.keys():
        raise ValueError('Merged configuration lost a source episode exclusion')
    if manifest.get('selection') != [item for source in sources for item in source['selection']]:
        raise ValueError('Merged selection differs from original Prepared selections')
    episodes = manifest.get('episodes')
    if (not isinstance(episodes, list) or {e['key'] for e in episodes} != expected_keys
            or len(episodes) != len(expected_keys)):
        raise ValueError('Merged Prepared episode membership differs from its sources')
    sessions = {}
    for episode in episodes:
        source = source_by_path.get(episode.get('source_prepared'))
        if (source is None or episode['key'] not in source['episode_keys']
                or episode.get('source_task_binding') != source['task_binding']
                or episode.get('source_prepared_manifest_sha256') != source['manifest_sha256']):
            raise ValueError('Episode source Prepared/task binding association differs')
        group = group_by_binding[_digest(source['task_binding'])]
        metadata = str(Path((episode.get('source_ref') or {}).get('metadata_path', '')).resolve())
        frozen_fingerprints = [f for f in episode.get('source_fingerprints', [])
                               if str(Path(f['path']).resolve()) == metadata]
        declared = next((f for f in group['metadata_fingerprints'] if f['path'] == metadata), None)
        if (declared is None or len(frozen_fingerprints) != 1
                or any(declared[k] != frozen_fingerprints[0].get(k) for k in ('sha256', 'size'))):
            raise ValueError('Episode metadata is absent from its verified collection evidence')
        if (not episode.get('training_candidate') or episode.get('quality_exclusion_reason')
                or episode.get('review', {}).get('quality') in ('C', 'rejected')):
            raise ValueError('Excluded/non-candidate episode in merged Prepared')
        binding_id = _digest(episode['source_task_binding'])
        if sessions.setdefault(episode['session'], binding_id) != binding_id:
            raise ValueError('A collection session mixes source task bindings')
    for source in sources:
        paths = {str(Path(episode['source_ref']['metadata_path']).resolve()) for episode in episodes
                 if episode['source_prepared'] == source['path']}
        if set(source.get('metadata_paths', [])) != paths:
            raise ValueError('Source Prepared metadata membership differs from its episodes')
    for group in groups:
        paths = {str(Path(episode['source_ref']['metadata_path']).resolve()) for episode in episodes
                 if episode['source_task_binding'] == group['source_task_binding']}
        if set(group['metadata_paths']) != paths:
            raise ValueError('Collection metadata membership differs from its episodes')
    split = proof.get('split')
    if split is not None:
        train, val = set(split['train']), set(split['val'])
        train_sessions, val_sessions = set(split['train_sessions']), set(split['val_sessions'])
        if train & val or train | val != expected_keys or train_sessions & val_sessions:
            raise ValueError('Merged frozen split overlaps or omits episodes')
        for episode in episodes:
            side = train_sessions if episode['key'] in train else val_sessions
            if episode['session'] not in side:
                raise ValueError('Merged episode/session split assignment disagrees')
    return ValidatedManifest(copy.deepcopy(manifest))


def _metadata_evidence(entry, binding):
    ref = entry.get('source_ref') or {}
    path = Path(ref.get('metadata_path', '')).resolve()
    dataset = Path(ref.get('dataset_root', '')).resolve()
    index = ref.get('episode_index')
    if (type(index) is not int or index < 0 or ref.get('key') != entry['key']
            or entry['key'] != f'{dataset.name}/episode_{index:06d}'
            or path != dataset / f'research/episode_{index:06d}_snapshots/episode_metadata.json'):
        raise ValueError('Prepared episode source reference is missing or mismatched')
    matches = [f for f in entry.get('source_fingerprints', []) if Path(f['path']).resolve() == path]
    if len(matches) != 1:
        raise ValueError('Original metadata lacks its frozen source fingerprint')
    payload = path.read_bytes()
    fingerprint = {'path': str(path), 'sha256': hashlib.sha256(payload).hexdigest(), 'size': len(payload)}
    if any(fingerprint[k] != matches[0].get(k) for k in ('sha256', 'size')):
        raise ValueError('Original episode metadata changed since Prepared audit')
    metadata = json.loads(payload.decode('utf-8'))
    if (metadata.get('task_id') != binding['task_id'] or metadata.get('task_binding') != binding
            or metadata.get('config', {}).get('task_binding') != binding
            or metadata.get('session') != entry['session']
            or metadata.get('scene_id') != binding['scene_id']):
        raise ValueError('Original metadata task/config/scene/session identity mismatch')
    return metadata['config'], fingerprint


def merge_prepared(prepared_roots, output, *, split_file=None):
    """Copy/link immutable Prepared episodes into one explicit learning domain.

    Only Prepared metadata and its NPZ/index/sequence files are read. Original
    video and raw research streams are neither decoded nor rehashed here.
    """
    from dataset.task3.training_pack import _load_prepared, _prepared_episode, session_split
    from tasks.registry import get, validate_binding

    roots = [Path(path).resolve() for path in prepared_roots]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError('Provide nonempty distinct Prepared inputs')
    output = Path(output).resolve()
    if output.exists() or any(output == root or root in output.parents or output in root.parents for root in roots):
        raise ValueError('Merged output must be a new directory separate from all inputs')
    manifests = [_load_prepared(root) for root in roots]
    if any(any(field in manifest for field in UNION_FIELDS) for manifest in manifests):
        raise ValueError('Merge original Prepared sources, not an already merged union')
    canonical = get(manifests[0]['task_id']).binding()
    common = ('task_id', 'engineering_only', 'sequence_contract', 'cameras', 'state_dim', 'action_dim',
              'categories', 'hand_order', 'geometry_normalization', 'action_alignment', 'review_mode')
    for manifest in manifests:
        if (manifest.get('cameras') != ['oblique'] or manifest.get('state_dim') != 16
                or manifest.get('action_dim') != 14
                or any(manifest.get(k) != manifests[0].get(k) for k in common)
                or not _same_domain(manifest.get('task_binding'), canonical)):
            raise ValueError('Prepared task/scene/camera/schema/collection domain differs')
        if _curation_math(manifest['config']) != _curation_math(manifests[0]['config']):
            raise ValueError('Prepared Safe Stitch/curation configuration differs')
    groups, sources, entries, prohibited = {}, [], [], {}
    for root, manifest in zip(roots, manifests):
        binding = manifest['task_binding']
        source_sha = _sha(root / 'manifest.json')
        keys = [entry['key'] for entry in manifest['episodes']]
        if set(keys) & {entry['key'] for entry in entries}:
            raise ValueError('Duplicate episode keys across source Prepared inputs')
        prohibited.update(_excluded(manifest['config'], manifest.get('selection', [])))
        sources.append({'path': str(root), 'manifest_sha256': source_sha,
            'task_binding': copy.deepcopy(binding), 'config': copy.deepcopy(manifest['config']),
            'config_sha256': manifest['config_sha256'], 'effective_config_sha256': manifest['effective_config_sha256'],
            'selection': copy.deepcopy(manifest.get('selection')), 'episode_keys': keys,
            'metadata_paths': [],
            'audit_run': manifest.get('audit_run'), 'review_binding': copy.deepcopy(manifest.get('review_binding')),
            'reviews_sha256': manifest.get('reviews_sha256'),
            'source_fingerprints_sha256': manifest.get('source_fingerprints_sha256'),
            'implementation_sha256': copy.deepcopy(manifest.get('implementation_sha256')),
            'recorded_collection_compatibility': copy.deepcopy(manifest.get('recorded_collection_compatibility'))})
        group = groups.setdefault(_digest(binding), {'source_task_binding': copy.deepcopy(binding),
            'metadata_paths': [], 'metadata_fingerprints': []})
        for entry in manifest['episodes']:
            config, fingerprint = _metadata_evidence(entry, binding)
            if 'frozen_collection_config' in group and group['frozen_collection_config'] != config:
                raise ValueError('One source binding contains inconsistent frozen collection settings')
            group['frozen_collection_config'] = copy.deepcopy(config)
            group['metadata_paths'].append(fingerprint['path'])
            group['metadata_fingerprints'].append(fingerprint)
            sources[-1]['metadata_paths'].append(fingerprint['path'])
            copied = copy.deepcopy(entry)
            copied.update(source_task_binding=copy.deepcopy(binding), source_prepared=str(root),
                          source_prepared_manifest_sha256=source_sha)
            entries.append(copied)
    if prohibited.keys() & {entry['key'] for entry in entries}:
        raise ValueError('An excluded episode cannot re-enter a merged Prepared')
    reference_group = next((g for g in groups.values() if g['source_task_binding'] == canonical), None)
    if reference_group is None:
        from collection.config import recording_config
        reference = recording_config(get(canonical['task_id']))
    else:
        reference = reference_group['frozen_collection_config']
    for group in groups.values():
        binding = group['source_task_binding']
        group['config_differences'] = _collection_differences(reference, group['frozen_collection_config'])
        if binding != canonical:
            for source in sources:
                if source['task_binding'] != binding:
                    continue
                evidence = source['recorded_collection_compatibility']
                if (not isinstance(evidence, dict) or evidence.get('task_id') != binding['task_id']
                        or evidence.get('recorded_collection_sha256') != binding['sources'][SOURCE]
                        or evidence.get('current_collection_sha256') != canonical['sources'][SOURCE]
                        or evidence.get('config_differences') != group['config_differences']
                        or not set(source['metadata_paths']) <= set(evidence.get('metadata_paths', []))):
                    raise ValueError('Historical Prepared provenance differs from actual metadata evidence')
        proof = {'config_differences': group['config_differences'], 'metadata_paths': group['metadata_paths']}
        if binding != canonical and not group['config_differences']:
            proof[EQUIVALENCE_FIELD] = configuration_equivalence(reference, group['frozen_collection_config'],
                reference_group['metadata_paths'][0] if reference_group is not None else None)
        with recorded_collection_scope(binding, proof):
            validate_binding(binding, canonical['task_id'])
    # Reuse the formal causal/array/sequence checks without loading RGB.
    for root, manifest in zip(roots, manifests):
        for entry in manifest['episodes']:
            _, _, _, rows = _prepared_episode(root, entry, manifest['config'])
            if any(row.get('session') != entry['session'] or row.get('source_episode') != entry['key'] for row in rows):
                raise ValueError('Prepared frame rows refer to another original episode/session')
    combined = copy.deepcopy(manifests[0])
    for field in ('recorded_collection_compatibility', 'audit_run', 'review_binding', 'reviews_sha256',
                  'source_manifest_sha256', 'source_fingerprints_sha256'):
        combined.pop(field, None)
    combined.update(task_binding=canonical, source_task_bindings=[g['source_task_binding'] for g in groups.values()],
                    source_prepared_manifests=sources, episodes=entries,
                    created_at=datetime.now(timezone.utc).isoformat(), human_reviewed=all(m['human_reviewed'] for m in manifests),
                    curation_approved=all(m['curation_approved'] for m in manifests),
                    implementation_sha256={'dataset/task3/merge_prepared.py': _sha(__file__)})
    combined['config']['quality']['excluded_episodes'] = [
        {'episode_key': key, 'reason': reason} for key, reason in sorted(prohibited.items())]
    combined['config_sha256'] = _digest(combined['config'])
    combined['effective_config_sha256'] = _digest(effective_config(combined['config']))
    combined['selection'] = [copy.deepcopy(item) for source in sources for item in source['selection']]
    combined['dataset_collection_compatibility'] = {
        'version': VERSION, 'learning_domain_task_binding': canonical,
        'reference_collection_config': reference, 'groups': list(groups.values()),
        'frame_selection': 'unchanged_frozen_prepared',
        'split': session_split(entries, split_file) if split_file is not None else None,
    }
    validate_manifest(combined)
    for entry in entries:
        dataset = Path(entry['source_ref']['dataset_root']).resolve()
        if output == dataset or dataset in output.parents or output in dataset.parents:
            raise ValueError('Merged output must be separate from immutable Raw inputs')
    output.mkdir(parents=True, exist_ok=False)
    for entry in combined['episodes']:
        source_root = Path(entry['source_prepared'])
        original = next(ep for source in manifests for ep in source['episodes'] if ep['key'] == entry['key'])
        directory = (output / 'episodes' / entry['key']).resolve()
        if not directory.is_relative_to(output):
            raise ValueError('Episode key escapes the merged output')
        directory.mkdir(parents=True, exist_ok=False)
        for field, filename, digest_field in (('arrays_path', 'arrays.npz', 'arrays_sha256'),
                ('index_path', 'frames.jsonl', 'index_sha256'), ('sequence_path', 'sequence.json', 'sequence_sha256')):
            old, new = (source_root / original[field]).resolve(), directory / filename
            try:
                os.link(old, new)
            except OSError:
                shutil.copy2(old, new)
            if _sha(new) != entry[digest_field]:
                raise ValueError('Prepared episode file changed during merge')
            entry[field] = new.relative_to(output).as_posix()
    for source in sources:
        if _sha(Path(source['path']) / 'manifest.json') != source['manifest_sha256']:
            raise ValueError('Source Prepared manifest changed during merge')
    for name, value in (('parameters.json', combined['config']), ('manifest.json', combined)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return ValidatedManifest(combined)
