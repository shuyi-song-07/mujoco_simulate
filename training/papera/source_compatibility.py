"""Exact sources, with narrowly reviewed trainer-interface migrations.

The reviewed pair adds optional checkpoint files and their index. It changes
no network, losses, sampler, optimizer or schedule. Missing retention settings
remain disabled in historical configs; their original signatures are retained.
The second migration permits same-variant Gesture continuation with the exact
existing checkpoint-LR hold path. It changes no training math or old Baseline
execution. The third migration only rejects completed engineering smoke runs
at the continuation boundary; it changes no full-data training computation.
The explicit reference-profile migration below can additionally relocate the
unchanged Panda update loop to engine.py. Its model/trainer/engine bundle must
be registered together; no future engine edit is covered by that approval.
"""
import copy
from pathlib import Path
import hashlib


# Exact byte hashes, not a blanket exemption for train.py.
REVIEWED_TRAINER_PAIRS = {
    ("6f3725da5e997ba73e524ae017bdf511635c223d39e409a20e57fef23ed690f4",
     "d8762c00feef82771eff9fc7bc85bf555ef699be7dd9a3a59e7652ea12bb9f1d"):
        "optional-candidate-checkpoint-retention-20261104",
    ("d8762c00feef82771eff9fc7bc85bf555ef699be7dd9a3a59e7652ea12bb9f1d",
     "f0f856e53e6d454ff30e48fa577ff294cadb2a1286bc7882d2e83b6de67dc8ff"):
        "same-variant-gesture-continuation-interface-20261104",
    ("6f3725da5e997ba73e524ae017bdf511635c223d39e409a20e57fef23ed690f4",
     "f0f856e53e6d454ff30e48fa577ff294cadb2a1286bc7882d2e83b6de67dc8ff"):
        "optional-retention-and-same-variant-gesture-continuation-interface-20261104",
    ("f0f856e53e6d454ff30e48fa577ff294cadb2a1286bc7882d2e83b6de67dc8ff",
     "6868d5dcb29ca2d568723095de1bffa34459e597ae044c48c79749446f7d3358"):
        "smoke-continuation-guard-only-20261104",
    ("d8762c00feef82771eff9fc7bc85bf555ef699be7dd9a3a59e7652ea12bb9f1d",
     "6868d5dcb29ca2d568723095de1bffa34459e597ae044c48c79749446f7d3358"):
        "same-variant-continuation-and-smoke-guard-interface-20261104",
    ("6f3725da5e997ba73e524ae017bdf511635c223d39e409a20e57fef23ed690f4",
     "6868d5dcb29ca2d568723095de1bffa34459e597ae044c48c79749446f7d3358"):
        "optional-retention-same-variant-continuation-and-smoke-guard-interface-20261104",
}


# Populated only after the exact default-Panda refactor has been reviewed.
# A source pair alone is insufficient: the complete engine bundle below is
# required, so this cannot authorize a network or later engine replacement.
REVIEWED_SOURCE_PAIRS = {
    ('act_model.py', 'e3856d7fff452e1a871419cb9146990d9e796350dbe639030df18753c49677af',
     'f0bc54faef6f1915f186c1775e8c736451364542e533934f7d690ba9b726a6dc'):
        'explicit-official-reference-profile-shared-panda-engine-20261106',
    ('train.py', '5cb3237833acff87389fe2cd627f61e7e7bfe7172175d97ef3d8dfcdb4024629',
     '5e575c61464650a96e51c3eb8890a0c3ba402294bbba78d0cd23873d0d22af72'):
        'explicit-official-reference-profile-shared-panda-engine-20261106',
}
REVIEWED_ENGINE_MIGRATIONS = {
    ('5cb3237833acff87389fe2cd627f61e7e7bfe7172175d97ef3d8dfcdb4024629',
     '5e575c61464650a96e51c3eb8890a0c3ba402294bbba78d0cd23873d0d22af72'): {
        'old_model_sha256': 'e3856d7fff452e1a871419cb9146990d9e796350dbe639030df18753c49677af',
        'new_model_sha256': 'f0bc54faef6f1915f186c1775e8c736451364542e533934f7d690ba9b726a6dc',
        'engine_sha256': 'bf179ab08cad2314387465c34d67e64a33d250cabb5def88616a21c8eb93cde0',
        'reason': 'explicit-official-reference-profile-shared-panda-engine-20261106',
    },
}


def _source_state(directory):
    fields = {'act_model.py': 'model_source_sha256', 'pack_data.py': 'reader_source_sha256',
              'data.py': 'normalizer_source_sha256', 'train.py': 'trainer_source_sha256'}
    if (directory / 'engine.py').is_file():
        fields['engine.py'] = 'engine_source_sha256'
    actual = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in fields}
    return fields, actual


def _reference_migration(provenance, actual):
    pair = (provenance.get('trainer_source_sha256'), actual['train.py'])
    migration = REVIEWED_ENGINE_MIGRATIONS.get(pair)
    if migration is None:
        return None
    if (provenance.get('model_source_sha256') != migration['old_model_sha256']
            or actual['act_model.py'] != migration['new_model_sha256']):
        return None
    if actual.get('engine.py') != migration['engine_sha256']:
        raise ValueError('Frozen training source changed: engine.py')
    for name, old, new in (
        ('train.py', *pair),
        ('act_model.py', migration['old_model_sha256'], migration['new_model_sha256']),
    ):
        if (name, old, new) not in REVIEWED_SOURCE_PAIRS:
            return None
    # This one migration starts before engine.py was a frozen input. A run
    # which already recorded an engine must match it through its own metadata.
    if 'engine_source_sha256' in provenance:
        return None
    return migration


def verify_training_sources(provenance):
    directory = Path(__file__).resolve().parent
    pairs, actual = _source_state(directory)
    migration = _reference_migration(provenance, actual)
    reviewed = []
    for name, key in pairs.items():
        if name == 'engine.py' and key not in provenance and migration is not None:
            reviewed.append(migration['reason'])
            continue
        if actual[name] != provenance.get(key):
            reason = REVIEWED_TRAINER_PAIRS.get((provenance.get(key), actual[name])) if name == 'train.py' else None
            if migration is not None:
                reason = REVIEWED_SOURCE_PAIRS.get((name, provenance.get(key), actual[name]), reason)
            if reason is None:
                raise ValueError(f'Frozen training source changed: {name}')
            reviewed.append(reason)
    if 'engine_source_sha256' in provenance and 'engine.py' not in actual:
        raise ValueError('Frozen training source changed: engine.py')
    # New official-core runs freeze the small vendored source files as well.
    # Historical fixture metadata without this field still follows its
    # existing model/trainer checks; it cannot pass the new checkpoint gate.
    official = provenance.get('official_sources_sha256', {})
    if not isinstance(official, dict):
        raise ValueError('Invalid frozen official ACT sources')
    for relative, expected in official.items():
        path = (directory / relative).resolve()
        if not path.is_relative_to((directory / 'official').resolve()) or not path.is_file():
            raise ValueError('Invalid official ACT source path')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen official ACT source changed: {relative}')
    mode = ('exact-current-sources' if not reviewed else
            'reviewed-checkpoint-retention-only' if all(reason == 'optional-candidate-checkpoint-retention-20261104'
                                                      for reason in reviewed) else
            'reviewed-training-interface-only')
    return {'compatibility': mode,
            'sha256': actual, 'reviewed_changes': reviewed}


def compatible_resume_provenance(current, previous):
    if current == previous:
        verify_training_sources(current)
        return current
    pair = (previous.get('trainer_source_sha256'), current.get('trainer_source_sha256'))
    directory = Path(__file__).resolve().parent
    _, actual = _source_state(directory)
    migration = _reference_migration(previous, actual)
    if migration is not None:
        verify_training_sources(current)
        verify_training_sources(previous)
        compared = copy.deepcopy(current)
        for key in ('trainer_source_sha256', 'model_source_sha256'):
            compared[key] = previous[key]
        compared.pop('engine_source_sha256', None)
        if compared == previous:
            return copy.deepcopy(previous)
    if pair in REVIEWED_TRAINER_PAIRS:
        verify_training_sources(previous)
        compared = copy.deepcopy(current)
        compared['trainer_source_sha256'] = previous['trainer_source_sha256']
        if compared == previous:
            # Frozen run/checkpoint signatures must still describe their source.
            return copy.deepcopy(previous)
    raise ValueError('Checkpoint provenance changed; exact resume requires frozen code/config/data')
