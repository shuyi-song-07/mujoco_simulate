"""Explicit all-training membership and immutable cache reuse for a new Pack."""
from pathlib import Path
import os
import shutil

VERSION = 'papera-all-train-v1'


def all_train(episodes):
    keys = [e['key'] for e in episodes]
    sessions = {e.get('session', e.get('session_id')) for e in episodes}
    if not keys or len(keys) != len(set(keys)) or any(not isinstance(s, str) or not s for s in sessions):
        raise ValueError('All-train requires nonempty unique episodes with source sessions')
    return {'version': VERSION, 'unit': 'whole eligible episode', 'train': keys, 'val': [],
            'train_sessions': sorted(sessions), 'val_sessions': [], 'independent_validation_available': False}


def validate_all_train(split, episodes):
    if split.get('version') == VERSION and split != all_train(episodes):
        raise ValueError('All-train membership differs from eligible Pack episodes')


def clone_cache(source, output):
    """New metadata, hard-linked immutable arrays. No video decode or recuration.

    Explicitly refuse a cross-volume clone instead of silently copying tens of GB.
    All files remain inside the new Pack; it has no dependency on the old path.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('New Pack must be separate from its source')
    if output.exists():
        raise FileExistsError(f'Choose a new Pack directory: {output}')
    def copy_file(src, dst):
        if Path(src).suffix in ('.npy', '.npz'):
            try:
                os.link(src, dst)
            except OSError as exc:
                raise OSError('Cache reuse requires a new directory on the same filesystem; '
                              'choose another output or build a fresh Pack') from exc
        else:
            shutil.copy2(src, dst)
        return dst
    shutil.copytree(source, output, copy_function=copy_file,
                    ignore=lambda directory, names: [n for n in names
                        if n in ('manifest.json', 'split.json', 'build.json', 'errors.json')
                        and Path(directory).resolve() == source])
    return output
