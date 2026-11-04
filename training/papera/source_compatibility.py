"""Exact sources, with narrowly reviewed trainer-interface migrations.

The reviewed pair adds optional checkpoint files and their index. It changes
no network, losses, sampler, optimizer or schedule. Missing retention settings
remain disabled in historical configs; their original signatures are retained.
The second migration permits same-variant Gesture continuation with the exact
existing checkpoint-LR hold path. It changes no training math or old Baseline
execution. The third migration only rejects completed engineering smoke runs
at the continuation boundary; it changes no full-data training computation.
No future trainer edit is covered by these exact pairs.
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


def verify_training_sources(provenance):
    directory = Path(__file__).resolve().parent
    pairs = {'act_model.py': 'model_source_sha256', 'pack_data.py': 'reader_source_sha256',
             'data.py': 'normalizer_source_sha256', 'train.py': 'trainer_source_sha256'}
    actual = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in pairs}
    reviewed = []
    for name, key in pairs.items():
        if actual[name] != provenance.get(key):
            reason = REVIEWED_TRAINER_PAIRS.get((provenance.get(key), actual[name])) if name == 'train.py' else None
            if reason is None:
                raise ValueError(f'Frozen training source changed: {name}')
            reviewed.append(reason)
    mode = ('exact-current-sources' if not reviewed else
            'reviewed-checkpoint-retention-only' if all(reason == 'optional-candidate-checkpoint-retention-20261104'
                                                      for reason in reviewed) else
            'reviewed-training-interface-only')
    return {'compatibility': mode,
            'sha256': actual, 'reviewed_changes': reviewed}


def compatible_resume_provenance(current, previous):
    if current == previous:
        return current
    pair = (previous.get('trainer_source_sha256'), current.get('trainer_source_sha256'))
    if pair in REVIEWED_TRAINER_PAIRS:
        verify_training_sources(previous)
        compared = copy.deepcopy(current)
        compared['trainer_source_sha256'] = previous['trainer_source_sha256']
        if compared == previous:
            # Frozen run/checkpoint signatures must still describe their source.
            return copy.deepcopy(previous)
    raise ValueError('Checkpoint provenance changed; exact resume requires frozen code/config/data')
