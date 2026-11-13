"""Shared stage operations; task adapters choose or override these entry points.

Lazy module calls keep one implementation and preserve test instrumentation.
This layer does not select frames, change labels, or alter split membership.
"""


def audit(*args, **kwargs):
    from . import prepare_paper_dataset
    return prepare_paper_dataset.audit(*args, **kwargs)


def reclip(*args, **kwargs):
    from . import rebuild_curation
    return rebuild_curation.build(*args, **kwargs)


def prepare(*args, **kwargs):
    from . import prepare_paper_dataset
    return prepare_paper_dataset.prepare(*args, **kwargs)


def pack(*args, **kwargs):
    from . import training_pack
    return training_pack.build(*args, **kwargs)


def verify_pack(*args, **kwargs):
    from . import training_pack
    return training_pack.verify(*args, **kwargs)
