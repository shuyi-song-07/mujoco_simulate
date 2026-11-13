"""push_r dataset stage: task evidence plus the unchanged common pipeline."""
from dataset.common.pipeline import audit, reclip, prepare, pack, verify_pack
from .evidence import dataset_adapter as adapter_factory
