"""Model support and new-run defaults for push_t; shared model math is untouched."""
from training.common import task_adapter
SUPPORTED_MODELS = ('act',)


def validate_model(algorithm):
    return task_adapter.validate_model(algorithm, SUPPORTED_MODELS)


def configure_defaults(config, task, algorithm='act'):
    return task_adapter.configure_defaults(config, task, algorithm, SUPPORTED_MODELS)
