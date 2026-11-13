"""Shared single-arm push evidence without task-specific success metrics."""
from dataset.common.pipeline import audit, reclip, prepare, pack, verify_pack


class SinglePandaPushEvidenceAdapter:
    version = 'papera-single-push-evidence-v1'

    def validate(self, metadata):
        if metadata.get('schema_version') != 'single-panda-push-v1':
            raise ValueError('Single-arm push evidence requires single-panda-push-v1')

    def build_evidence(self, episode, curation_config):
        from dataset.common.frame_evidence import build_frame_evidence
        self.validate(episode['metadata'])
        return build_frame_evidence(episode, curation_config)


def adapter_factory():
    return SinglePandaPushEvidenceAdapter()
