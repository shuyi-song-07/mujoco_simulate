"""Shared dual-arm dataset stage for tasks without a custom evidence adapter."""
from dataset.common.pipeline import audit, reclip, prepare, pack, verify_pack


class DualPandaEvidenceAdapter:
    version = 'papera-tray-evidence-v1'

    def validate(self, metadata):
        if metadata.get('schema_version') != 'task3-papera-v2':
            raise ValueError('Dual-arm evidence adapter requires task3-papera-v2')

    def build_evidence(self, episode, curation_config):
        from dataset.common.frame_evidence import build_frame_evidence
        return build_frame_evidence(episode, curation_config)


def adapter_factory():
    return DualPandaEvidenceAdapter()
