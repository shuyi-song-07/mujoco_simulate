"""Contained raw shard path resolution shared by audit and review."""
from pathlib import Path

def dataset_path(root: Path, template: str, **values) -> Path:
    """Never follow a dataset-supplied path outside the dataset root."""
    path = (root / template.format(**values)).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Dataset path escapes its root: {template}")
    return path
