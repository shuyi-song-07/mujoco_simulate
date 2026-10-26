"""Exact-source resume for the independent Paper A project."""
from pathlib import Path
import hashlib

def verify_training_sources(provenance):
    directory=Path(__file__).resolve().parent
    pairs={'act_model.py':'model_source_sha256','pack_data.py':'reader_source_sha256',
           'data.py':'normalizer_source_sha256','train.py':'trainer_source_sha256'}
    actual={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in pairs}
    for name,key in pairs.items():
        if actual[name]!=provenance.get(key):
            raise ValueError(f'Frozen training source changed: {name}')
    return {'compatibility':'exact-current-sources','sha256':actual}

def compatible_resume_provenance(current, previous):
    if current != previous:
        raise ValueError('Checkpoint provenance changed; exact resume requires frozen code/config/data')
    return current
