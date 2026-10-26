"""Formal Prepared -> multi-episode oblique pack, with an independent reader.

This module writes data and diagnostics only. It does not choose ACT windows,
padding, losses, normalization or a model architecture. No second frame filter
is applied: every retained Prepared frame has exactly one cache row.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from dataset.task3.prepare_paper_dataset import CATEGORIES, digest, read_rows
from dataset.task3.video_cache import cache_video


VERSION = 'papera-single-view-training-pack-v2'
LEGACY_VERSION = 'papera-single-view-training-pack-v1'
PREPARED_VERSION = 'papera-prepared-single-view-v2'
ALIGNMENT = 'next_control_tick_at_raw_observation_anchors'
STAGES = ('free_stationary', 'free_motion', 'contact_hold', 'contact_motion',
          'partial_grasp', 'dual_grasp_hold', 'dual_grasp_motion', 'releasing',
          'terminal_success', 'terminal_failure')


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def within(root, relative):
    root = Path(root).resolve()
    path = (root/relative).resolve()
    if root not in path.parents:
        raise ValueError(f'Path escapes data directory: {relative}')
    return path


def file_record(path, root):
    path = Path(path)
    return {'path': str(path.relative_to(root)), 'size': path.stat().st_size, 'sha256': sha256(path)}


def verify_files(root, files):
    for record in files:
        path = within(root, record['path'])
        if not path.is_file() or path.stat().st_size != record['size'] or sha256(path) != record['sha256']:
            raise ValueError(f'Cache file changed or hash mismatch: {path}')


def session_split(entries, split_file):
    """Project the frozen session assignment; do not re-randomize subsets."""
    source = json.loads(Path(split_file).read_text(encoding='utf-8'))
    groups = {side: set(source[side+'_sessions']) for side in ('train', 'val')}
    if groups['train'] & groups['val']:
        raise ValueError('Frozen split has overlapping train/validation sessions')
    if set(source.get('train', [])) & set(source.get('val', [])):
        raise ValueError('Frozen split has overlapping episode keys')
    result = {'version': 'frozen-session-membership-v1', 'unit': 'whole collection session',
              'source': str(Path(split_file).resolve()), 'source_sha256': sha256(split_file)}
    for ep in entries:
        session = ep.get('session')
        if session not in groups['train'] | groups['val']:
            raise ValueError(f'Session absent from frozen split: {session}')
        for side in ('train','val'):
            if ep['key'] in source.get(side, []) and session not in groups[side]:
                raise ValueError(f'Episode/session split assignment disagrees: {ep["key"]}')
    for side in ('train','val'):
        result[side] = [ep['key'] for ep in entries if ep['session'] in groups[side]]
        result[side+'_sessions'] = sorted({ep['session'] for ep in entries if ep['session'] in groups[side]})
    # Subsets are legitimate engineering packs, but cannot pretend to contain an
    # independent validation experiment when only one frozen side is available.
    result['independent_validation_available'] = bool(result['train'] and result['val'])
    return result


def _load_prepared(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('version') != PREPARED_VERSION or not manifest.get('complete')
            or manifest.get('pipeline_stage') != 'prepared'
            or manifest.get('review_mode') not in ('auto','reviewed')):
        raise ValueError('Expected a complete formal auto/reviewed single-view Prepared')
    from dataset.task3.sequence_stitching import VERSION as SEQUENCE_VERSION
    if manifest.get('sequence_contract') != SEQUENCE_VERSION:
        raise ValueError('Prepared lacks the verified virtual sequence contract; export a new candidate')
    _validate_frozen_config(manifest)
    if json.loads((root/'parameters.json').read_text(encoding='utf-8')) != manifest['config']:
        raise ValueError('Prepared parameters differ from the frozen curation configuration')
    episodes = manifest.get('episodes', [])
    if not episodes or len({e['key'] for e in episodes}) != len(episodes):
        raise ValueError('Prepared episodes are empty or duplicate')
    for ep in episodes:
        if (not ep.get('training_candidate') or ep.get('quality_exclusion_reason')
                or ep.get('review', {}).get('quality') in ('C','rejected')):
            raise ValueError(f'Excluded/non-candidate Prepared episode: {ep["key"]}')
        views = ep.get('training_video_sources', ep.get('video_sources', []))
        if len(views) != 1 or views[0].get('camera') != 'oblique':
            raise ValueError('Only one oblique training camera is permitted')
    return manifest


def _validate_frozen_config(manifest):
    from dataset.task3.curation_config import effective_config, validate_config
    config = validate_config(manifest['config'])
    if (digest(config) != manifest.get('config_sha256')
            or digest(effective_config(config)) != manifest.get('effective_config_sha256')):
        raise ValueError('Frozen curation configuration hash mismatch')


def _validate_pack_contract(pack, manifest):
    """Small metadata gate shared by direct reads and full independent checks."""
    frozen_path = pack/'prepared.manifest.json'
    if sha256(frozen_path) != manifest['binding']['prepared_manifest_sha256']:
        raise ValueError('Pack frozen Prepared manifest differs from build binding')
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    if manifest['version'] == VERSION:
        _validate_frozen_config(frozen)
        if (manifest.get('curation_config') != frozen.get('config')
                or manifest.get('sequence_contract') != frozen.get('sequence_contract')):
            raise ValueError('Pack sequence/config contract differs from its frozen Prepared')


def derive_arrays(arrays, rows, *, fps, sequence=None, config=None):
    """Validate the kept-frame join and causal target without changing values."""
    kept = [row for row in rows if row.get('keep')]
    ids = np.asarray(arrays['raw_frame_index'])
    n = len(kept)
    if not n or not np.array_equal(ids, [r['raw_frame_index'] for r in kept]) or np.any(np.diff(ids) <= 0):
        raise ValueError('Prepared arrays and retained original frame indexes disagree')
    required = {'state': (n,16), 'action_recorded': (n,14), 'next_control_target': (n,14),
        'next_control_target_mask': (n,), 'category_id': (n,2), 'category_mask': (n,2),
        'hand_image': (n,2,21,3), 'hand_model_world': (n,2,21,3), 'hand_normalized': (n,2,21,3),
        'geometry_mask': (n,2), 'task_geometry_mask': (n,2), 'raw_geometry_present': (n,2),
        'source_control_tick': (n,), 'control_epoch': (n,), 'original_time_s': (n,),
        'source_control_tick_mask': (n,), 'control_epoch_mask': (n,),
        'task_segment_id': (n,), 'contiguous_fragment_id': (n,), 'segment_id': (n,)}
    sequence_fields = ('training_frame_index', 'training_time_s', 'train_sequence_id',
                       'sequence_frame_index', 'sequence_time_s', 'causal_target_valid')
    has_sequence = any(name in arrays for name in sequence_fields)
    if sequence is not None:
        if config is None:
            raise ValueError('Sequence validation requires frozen curation parameters')
        from dataset.task3.sequence_stitching import verify_sequence_contract
        verify_sequence_contract(rows, sequence, config)
        required.update({name: (n,) for name in sequence_fields})
    elif has_sequence or any(row.get('train_sequence_id') is not None for row in rows):
        raise ValueError('Virtual timeline arrays require their complete raw-evidence sequence certificate')
    for name, shape in required.items():
        if name not in arrays or arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError(f'Invalid/nonfinite Prepared array {name}; expected {shape}')
    target_mask = np.asarray(arrays['next_control_target_mask'], dtype=bool)
    if not target_mask.any():
        raise ValueError('Episode has no valid causal next control targets')
    ticks, times, stages, modes = [], [], [], []
    for i, row in enumerate(kept):
        for field in ('state_recorded','action_recorded'):
            name = 'state' if field == 'state_recorded' else field
            if not np.array_equal(arrays[name][i], np.asarray(row[field], dtype=arrays[name].dtype)):
                raise ValueError(f'Prepared {name} differs from original frame row')
        for field in ('source_control_tick','control_epoch','original_time_s','task_segment_id',
                      'contiguous_fragment_id','segment_id'):
            recorded = row.get(field)
            if recorded is None and field in ('source_control_tick','control_epoch'):
                recorded = -1
            if arrays[field][i] != recorded:
                raise ValueError(f'Prepared {field} differs from frame index')
        if sequence is not None:
            for field in sequence_fields:
                if arrays[field][i] != row.get(field):
                    raise ValueError(f'Prepared {field} differs from verified virtual timeline')
        for field in ('source_control_tick','control_epoch'):
            if bool(arrays[field+'_mask'][i]) != (row.get(field) is not None):
                raise ValueError(f'Prepared {field}_mask differs from original source presence')
        target = row.get('next_control_target')
        if bool(target) != bool(target_mask[i]):
            raise ValueError('Causal next control target mask differs from row')
        if target:
            tick, epoch = row.get('source_control_tick'), row.get('control_epoch')
            period = (row.get('evidence') or {}).get('control_period_s')
            # Earlier evidence schemas nest physical facts under evidence.
            if period is None:
                period = ((row.get('evidence') or {}).get('source') or {}).get('control_period_s')
            delta = float(target['sim_time'])-float(row['original_time_s'])
            if (tick is None or epoch is None or target.get('source_tick') != tick+1
                    or target.get('control_epoch') != epoch or not 0 < delta <= .03+1e-9):
                raise ValueError('Target is not the causal next control tick in this epoch')
            if period is not None and not 0 < delta <= float(period)+.002+1e-9:
                raise ValueError('Causal target exceeds frozen control period')
            source_line, target_line = row.get('source_line'), target.get('source_line')
            if (source_line is not None and target_line is not None
                    and (type(source_line) is not int or type(target_line) is not int or target_line <= source_line)):
                raise ValueError('Causal next target violates original log source-line order')
            if not np.array_equal(arrays['next_control_target'][i],
                                  np.asarray(target['action'], dtype=arrays['next_control_target'].dtype)):
                raise ValueError('Prepared causal action differs from frame source')
        ticks.append(target['source_tick'] if target else -1)
        times.append(target['sim_time'] if target else -1.)
        if row.get('robot_phase') not in STAGES:
            raise ValueError('Unknown robot phase; cannot invent supervision')
        stages.append(STAGES.index(row['robot_phase']))
        modes.append([row.get('accepted_left_intent'), row.get('accepted_right_intent')])
        for side_index, side in enumerate(('left','right')):
            aux = row['auxiliary'][side]
            category = aux.get('category')
            category_id = CATEGORIES.index(category) if category in CATEGORIES else -1
            if arrays['category_id'][i,side_index] != category_id:
                raise ValueError('Gesture class no longer matches the original hand/frame')
            for mask_name in ('category_mask','geometry_mask','task_geometry_mask'):
                if bool(arrays[mask_name][i,side_index]) != bool(aux[mask_name]):
                    raise ValueError(f'Auxiliary {mask_name} no longer matches its original frame')
    result = {name: np.asarray(value).copy() for name,value in arrays.items()}
    # A syntactically present next target is insufficient when its source proof
    # fails. Preserve the raw mask, and expose the verified training validity.
    action_mask = target_mask & arrays['causal_target_valid'] if sequence is not None else target_mask
    if not action_mask.any():
        raise ValueError('Episode has no independently verified causal control targets')
    result.update(action=arrays['next_control_target'].copy(), action_mask=action_mask.copy(),
        action_control_tick=np.asarray(ticks, np.int64), action_time_s=np.asarray(times, np.float64),
        stage_id=np.asarray(stages, np.int64),
        paused=np.asarray([r.get('status_summary',{}).get('paused') is True for r in kept]),
        actual_done=np.asarray([r.get('terminal',{}).get('actual_done') is True for r in kept]),
        success=np.asarray([r.get('terminal',{}).get('success') is True for r in kept]))
    if np.any((result['category_id'] < 0) & result['category_mask']):
        raise ValueError('Unknown gestures cannot have a valid category target mask')
    # Preserve exact masks, including paused/unknown semantics. This pack does
    # not reinterpret human intent or introduce a confidence-based filter.
    report = temporal_diagnostics(result, kept, fps=fps, modes=modes)
    if sequence is not None:
        report['virtual_sequence'] = sequence
        report['clock'] = ('original_time_s preserves physics; training_time_s is a verified 30Hz '
                           'retiming, usable only within each train_sequence_id')
    return result, report


def temporal_diagnostics(arrays, rows, *, fps, modes=None):
    n = len(rows); fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid source observation FPS')
    ids, times = arrays['raw_frame_index'], arrays['original_time_s']
    gaps, boundaries = [], [0]
    transitions = []
    for i in range(1,n):
        same_epoch = arrays['control_epoch'][i] == arrays['control_epoch'][i-1]
        dt = float(times[i]-times[i-1])
        if same_epoch and dt <= 0:
            raise ValueError('Original observation time does not increase within control epoch')
        causes = []
        if ids[i] != ids[i-1]+1: causes.append('removed_raw_frames')
        if not same_epoch: causes.append('control_epoch_change')
        # Collection's fixed phase may differ by one discrete control step.
        if not 0 < dt <= 1/fps+.02+1e-9: causes.append('actual_time_gap')
        if rows[i].get('hard_boundary') or ((rows[i].get('evidence') or {}).get('source') or {}).get('hard_boundary'):
            causes.append('source_hard_boundary')
        # Segment numbers describe source-backed boundaries; a phase label or
        # unexplained renumbering is not evidence that physical time broke.
        for field, cause in (('task_segment_id','task_segment_change'),
                             ('contiguous_fragment_id','retained_fragment_change')):
            if arrays[field][i] != arrays[field][i-1]:
                if not causes:
                    raise ValueError(f'Unexplained {field} boundary in continuous original frames')
                causes.append(cause)
        if rows[i].get('gap_before',{}):
            if not causes:
                raise ValueError('Recorded gap has no original frame/time/epoch/hard-boundary evidence')
            causes.append('recorded_gap')
        if causes:
            boundaries.append(i)
            gaps.append({'previous_raw_frame': int(ids[i-1]), 'next_raw_frame': int(ids[i]),
                'removed_frames': int(ids[i]-ids[i-1]-1), 'original_delta_s': dt,
                'causes': causes, 'gap_evidence': rows[i].get('gap_before')})
        previous_mode = modes[i-1] if modes else [rows[i-1].get('accepted_left_intent'), rows[i-1].get('accepted_right_intent')]
        mode = modes[i] if modes else [rows[i].get('accepted_left_intent'), rows[i].get('accepted_right_intent')]
        stage_changed = rows[i].get('robot_phase') != rows[i-1].get('robot_phase')
        if stage_changed or mode != previous_mode:
            transitions.append({'previous_raw_frame': int(ids[i-1]), 'raw_frame': int(ids[i]),
                'from_phase': rows[i-1].get('robot_phase'), 'to_phase': rows[i].get('robot_phase'),
                'from_intents': previous_mode, 'to_intents': mode,
                'crosses_gap': bool(causes), 'original_delta_s': dt})
    runs = [{'start': a, 'stop': b, 'start_raw_frame': int(ids[a]),
             'end_raw_frame': int(ids[b-1]), 'frames': b-a}
            for a,b in zip(boundaries, boundaries[1:]+[n])]
    # These are diagnostic counts, not a training-window implementation.
    lengths = (1,5,10,30)
    coverage = {str(length): {'full_contiguous_anchors': sum(max(0,r['frames']-length+1) for r in runs),
        'all_retained_anchors': n} for length in lengths}
    return {'frames': n, 'gaps': gaps, 'contiguous_runs': runs, 'transitions': transitions,
        'window_length_diagnostics_only': coverage,
        'gesture_valid_hands': int(arrays['category_mask'].sum()),
        'geometry_valid_hands': int(arrays['task_geometry_mask'].sum()),
        'causal_action_valid_frames': int(arrays['action_mask'].sum()),
        'frame_filter_applied': False, 'clock': 'original simulation time; preview playback is not an action clock'}


def _prepared_episode(prepared, entry, config):
    for name, hash_name in (('arrays_path','arrays_sha256'), ('index_path','index_sha256'),
                           ('sequence_path','sequence_sha256')):
        path = within(prepared, entry[name])
        if sha256(path) != entry[hash_name]:
            raise ValueError(f'Prepared file hash mismatch: {path}')
    with np.load(within(prepared,entry['arrays_path']), allow_pickle=False) as bundle:
        arrays = {name: bundle[name] for name in bundle.files}
    rows = read_rows(within(prepared,entry['index_path']))
    source = entry.get('training_video_sources', entry['video_sources'])[0]
    sequence = json.loads(within(prepared,entry['sequence_path']).read_text(encoding='utf-8'))
    arrays, report = derive_arrays(arrays, rows, fps=source['fps'], sequence=sequence, config=config)
    if len(arrays['state']) != entry['kept_frames']:
        raise ValueError('Prepared retained frame count differs from manifest')
    return source, arrays, report, rows


def _source_oblique_hash(entry, source):
    path = Path(source['path']).resolve()
    frozen = [f for f in entry['source_fingerprints'] if Path(f['path']).resolve() == path]
    if len(frozen) != 1:
        raise ValueError(f'Source oblique video lacks its frozen audit fingerprint: {path}')
    current = sha256(path)
    if current != frozen[0]['sha256'] or path.stat().st_size != frozen[0]['size']:
        raise ValueError(f'Source oblique video changed since Prepared audit: {path}')
    return current


def build(prepared, output, *, split_file, width=640, height=480, resume=False, continue_on_error=False):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    if not isinstance(width,int) or not isinstance(height,int) or min(width,height) < 32:
        raise ValueError('Cache image dimensions must be integers >=32')
    if output == prepared or prepared in output.parents or output in prepared.parents:
        raise ValueError('Training pack output must be separate from Prepared')
    manifest = _load_prepared(prepared)
    entries = manifest['episodes']
    # Validate every action join before allocating large RGB files.
    for entry in entries:
        source, _, _, _ = _prepared_episode(prepared,entry,manifest['config'])
        _source_oblique_hash(entry,source)
    split = session_split(entries,split_file)
    binding = {'prepared': str(prepared), 'prepared_manifest_sha256': sha256(prepared/'manifest.json'),
        'split_source': str(Path(split_file).resolve()), 'split_source_sha256': sha256(split_file),
        'config': {'image_width': width, 'image_height': height, 'cameras': ['oblique']},
        'implementation_sha256': {name: sha256(Path(__file__).with_name(name))
            for name in ('training_pack.py','video_cache.py','sequence_stitching.py',
                         'curation_rules.py','curation_config.py')}}
    if output.exists() and not resume:
        raise FileExistsError(f'Choose a new output directory or use resume: {output}')
    output.mkdir(parents=True,exist_ok=resume)
    if (output/'build.json').exists():
        if json.loads((output/'build.json').read_text(encoding='utf-8'))['binding'] != binding:
            raise ValueError('Resume binding changed: Prepared, split, code or image configuration')
    else:
        if any(output.iterdir()):
            raise ValueError('Resume target has no formal build binding')
        write_json(output/'build.json', {'binding': binding, 'complete': False})
    if (output/'manifest.json').exists():
        return verify(output)['manifest']
    results, errors = [], []
    for entry in entries:
        try:
            directory = within(output,Path('episodes')/entry['key'])
            result_path = directory/'result.json'
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding='utf-8'))
                if result.get('prepared_arrays_sha256') != entry['arrays_sha256']:
                    raise ValueError('Resume cached episode differs from Prepared')
                verify_files(output,result['files'])
                results.append(result)
                continue
            source, arrays, diagnostics, rows = _prepared_episode(prepared,entry,manifest['config'])
            source_hash = _source_oblique_hash(entry,source)
            directory.mkdir(parents=True,exist_ok=True)
            arrays_path, rgb_path = directory/'arrays.npz', directory/'rgb_oblique.npy'
            np.savez_compressed(arrays_path,**arrays)
            check = cache_video(source,arrays['raw_frame_index'],rgb_path,width,height)
            if sha256(source['path']) != source_hash:
                raise ValueError('Source oblique video changed while caching')
            # Keep all raw rows and causal/auxiliary IDs, including rows removed
            # upstream. This preserves evidence; only arrays/images are kept rows.
            index_path = directory/'frames.jsonl'
            from dataset.task3.prepare_paper_dataset import write_rows
            write_rows(index_path,rows)
            diagnostics_path = directory/'diagnostics.json'; write_json(diagnostics_path,diagnostics)
            result = {'key': entry['key'], 'session': entry['session'], 'frames': len(arrays['state']),
                'review_mode': entry['review_mode'], 'review_status': entry['review_status'],
                'human_reviewed': entry['human_reviewed'], 'training_candidate': not manifest.get('engineering_only', False),
                'source_ref': entry['source_ref'], 'source_fingerprints': entry['source_fingerprints'],
                'review_video_sources': entry.get('review_video_sources', []),
                'prepared_arrays_sha256': entry['arrays_sha256'],
                'arrays_path': str(arrays_path.relative_to(output)),
                'index_path': str(index_path.relative_to(output)),
                'diagnostics_path': str(diagnostics_path.relative_to(output)),
                'image_paths': {'oblique': str(rgb_path.relative_to(output))},
                'source_video': dict(source,sha256=source_hash), 'video_check': check,
                'files': [file_record(p,output) for p in (arrays_path,rgb_path,index_path,diagnostics_path)]}
            write_json(result_path,result); results.append(result)
            print(f'Packed {entry["key"]}: {result["frames"]} original-index-aligned frames',flush=True)
        except Exception as exc:
            errors.append({'key': entry['key'], 'error': f'{type(exc).__name__}: {exc}'})
            write_json(output/'errors.json',{'complete': False,'errors': errors})
            if not continue_on_error:
                raise
    if errors:
        raise ValueError(f'{len(errors)} pack episodes failed; resume after correcting the recorded causes')
    if sha256(prepared/'manifest.json') != binding['prepared_manifest_sha256'] or sha256(split_file) != binding['split_source_sha256']:
        raise ValueError('Prepared/split changed while building pack; completion refused')
    write_json(output/'split.json',split)
    (output/'prepared.manifest.json').write_bytes((prepared/'manifest.json').read_bytes())
    from tasks.artifacts import fields
    record = {**fields(manifest), 'version': VERSION,'pipeline_stage': 'training_pack','complete': True,
        'sequence_contract': manifest['sequence_contract'], 'curation_config': manifest['config'],
        'data_ready': True,'training_ready': False, 'training_interface': 'training.papera.pack_data.SafeStitchDataset',
        'remaining_gate': 'Data contract verified; quality approval and explicit training invocation remain separate. Engineering data is never formal training data.',
        'created_at': datetime.now(timezone.utc).isoformat(),'binding': binding,
        'review_mode': manifest['review_mode'],'human_reviewed': manifest['human_reviewed'],
        'cameras': ['oblique'],'state_dim': 16,'action_dim': 14,
        'config': binding['config'], 'action_alignment': ALIGNMENT,
        'categories': manifest['categories'],'hand_order': manifest['hand_order'],'stages': list(STAGES),
        'episodes': results,'episode_count': len(results),'frame_count': sum(e['frames'] for e in results),
        'frame_filter_applied': False,'normalization_applied': False,
        'split': split, 'files': [file_record(output/name,output)
            for name in ('split.json','prepared.manifest.json')],
        'time_semantics': 'Original IDs/ticks/epochs/times preserved. Virtual 30Hz adjacency is valid only inside verified train_sequence_id; hard boundaries cannot be crossed.'}
    _verify_contents(output,record)
    write_json(output/'manifest.json',record)
    write_json(output/'build.json',{'binding': binding,'complete': True})
    if (output/'errors.json').exists():
        write_json(output/'errors.json',{'complete': True,'errors': []})
    return record


def _pack_manifest(pack):
    manifest = json.loads((pack/'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('version') not in (VERSION, LEGACY_VERSION) or not manifest.get('complete')
            or manifest.get('cameras') != ['oblique'] or manifest.get('action_alignment') != ALIGNMENT):
        raise ValueError('Incomplete/incompatible single-view pack')
    _validate_pack_contract(pack, manifest)
    return manifest


def _episode_storage(pack, entry):
    """One storage path shared by independent verification and verified workers."""
    with np.load(within(pack,entry['arrays_path']),allow_pickle=False) as bundle:
        arrays = {name: bundle[name] for name in bundle.files}
    image = np.load(within(pack,entry['image_paths']['oblique']),mmap_mode='r',allow_pickle=False)
    return arrays, image


def _read_episode(pack,key,manifest,*,verify_files=True,include_evidence=True):
    """Return exact stored arrays and a read-only RGB memmap, never windows."""
    pack = Path(pack).resolve()
    entries = [e for e in manifest['episodes'] if e['key'] == key]
    if len(entries) != 1:
        raise ValueError(f'Unknown/duplicate pack episode: {key}')
    entry = entries[0]
    if verify_files:
        globals()['verify_files'](pack,entry['files'])
    arrays, image = _episode_storage(pack, entry)
    saved = json.loads(within(pack,entry['diagnostics_path']).read_text(encoding='utf-8'))
    config = manifest['config']
    if image.dtype != np.uint8 or image.shape != (entry['frames'],config['image_height'],config['image_width'],3):
        image._mmap.close()
        raise ValueError('RGB/robot cache shape mismatch')
    if not include_evidence:
        return {'key': key,'entry': entry,'arrays': arrays,'images': {'oblique': image},
                'diagnostics': saved,'rows': None}
    rows = read_rows(within(pack,entry['index_path']))
    sequence = saved.get('virtual_sequence')
    if manifest.get('version') == VERSION:
        from dataset.task3.sequence_stitching import VERSION as SEQUENCE_VERSION
        if manifest.get('sequence_contract') != SEQUENCE_VERSION or not sequence:
            raise ValueError('Pack lacks a complete verified sequence contract')
    expected, diagnostic = derive_arrays(arrays,rows,fps=entry['source_video']['fps'],
        sequence=sequence, config=manifest.get('curation_config'))
    for name,value in expected.items():
        if name not in arrays or not np.array_equal(arrays[name],value):
            raise ValueError(f'Cached array violates original Prepared relation: {name}')
    if saved != diagnostic:
        raise ValueError('Temporal diagnostics disagree with stored frame evidence')
    return {'key': key,'entry': entry,'arrays': arrays,'images': {'oblique': image},
            'diagnostics': diagnostic,'rows': rows}


class VerifiedPack(dict):
    """JSON-compatible verification report with an in-memory reader capability.

    JSON round-tripping intentionally loses the capability. A new process or
    resumed run must call verify again; Windows workers may pickle the verified
    report, which contains only metadata, never RGB arrays or frame evidence.
    """


def _file_states(pack, manifest):
    records = [{'path': 'manifest.json'}, *manifest['files']]
    records.extend(record for entry in manifest['episodes'] for record in entry['files'])
    result = {}
    for record in records:
        info = within(pack, record['path']).stat()
        result[record['path']] = (info.st_size, info.st_mtime_ns, info.st_ino)
    return result


def _checked_manifest(pack, checked, key):
    if not isinstance(checked, VerifiedPack) or getattr(checked, '_verified_root', None) != str(pack):
        raise ValueError('A live report returned by verify(pack) is required for a lightweight read')
    if sha256(pack/'manifest.json') != checked._manifest_sha256:
        raise ValueError('Verified pack manifest changed')
    manifest = checked['manifest']
    if digest(manifest) != checked._manifest_digest:
        raise ValueError('Verified pack report was modified')
    entries = [entry for entry in manifest['episodes'] if entry['key'] == key]
    if len(entries) != 1:
        raise ValueError(f'Unknown/duplicate pack episode: {key}')
    records = [{'path': 'manifest.json'}, *manifest['files'], *entries[0]['files']]
    for record in records:
        info = within(pack, record['path']).stat()
        if (info.st_size, info.st_mtime_ns, info.st_ino) != checked._file_states[record['path']]:
            raise ValueError(f'File changed since full verification: {record["path"]}')
    return manifest


def read_episode(pack,key,*,verify_files=True,checked=None,include_evidence=True):
    """Read through one official interface; workers may reuse full verification.

    The default independently checks all episode evidence. A lightweight read
    requires the live, manifest/file-bound report returned by verify; a plain
    dict loaded from a prior JSON report cannot bypass verification.
    """
    pack = Path(pack).resolve()
    if checked is not None:
        manifest = _checked_manifest(pack, checked, key)
        return _read_episode(pack,key,manifest,verify_files=False,include_evidence=include_evidence)
    if not include_evidence:
        raise ValueError('Lightweight reading requires a current full verify report')
    return _read_episode(pack,key,_pack_manifest(pack),verify_files=verify_files)


def _verify_contents(pack,manifest):
    verify_files(pack,manifest['files'])
    from tasks.artifacts import fields
    frozen_prepared=json.loads((pack/'prepared.manifest.json').read_text(encoding='utf-8'))
    if fields(manifest)!=fields(frozen_prepared):
        raise ValueError('Pack/Prepared task or engineering provenance mismatch')
    _validate_pack_contract(pack, manifest)
    if len({e['key'] for e in manifest['episodes']}) != len(manifest['episodes']):
        raise ValueError('Duplicate pack episodes')
    split = json.loads((pack/'split.json').read_text(encoding='utf-8'))
    if split != manifest['split'] or set(split['train_sessions']) & set(split['val_sessions']):
        raise ValueError('Pack split mismatch or session overlap')
    keys = {e['key'] for e in manifest['episodes']}
    if set(split['train']) & set(split['val']) or set(split['train']) | set(split['val']) != keys:
        raise ValueError('Pack split leaks or omits episodes')
    frame_count = 0; gap_count = 0
    for entry in manifest['episodes']:
        value = _read_episode(pack,entry['key'],manifest)
        side = 'train' if entry['key'] in split['train'] else 'val'
        if entry['session'] not in split[side+'_sessions']:
            raise ValueError('Episode assigned across session boundary')
        frame_count += len(value['arrays']['state']); gap_count += len(value['diagnostics']['gaps'])
        del value
    if frame_count != manifest['frame_count'] or len(keys) != manifest['episode_count']:
        raise ValueError('Pack totals mismatch')
    return {'valid': True,'episodes': len(keys),'frames': frame_count,'gaps': gap_count,
        'training_started': False,'training_interface': 'training.papera.pack_data.SafeStitchDataset','manifest': manifest}


def verify(pack):
    pack = Path(pack).resolve()
    manifest_sha256 = sha256(pack/'manifest.json')
    manifest = _pack_manifest(pack)
    before = _file_states(pack, manifest)
    result = _verify_contents(pack,manifest)
    after = _file_states(pack, manifest)
    if before != after or sha256(pack/'manifest.json') != manifest_sha256:
        raise ValueError('Pack files changed during full verification')
    report = VerifiedPack(result)
    report._verified_root = str(pack)
    report._manifest_sha256 = manifest_sha256
    report._manifest_digest = digest(manifest)
    report._file_states = after
    return report
