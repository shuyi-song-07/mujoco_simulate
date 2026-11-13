"""Tray recording evidence; shared physical and curation rules remain unchanged."""
class TrayEvidenceAdapter:
    version = 'papera-tray-evidence-v1'
    def validate(self, metadata):
        if metadata.get('schema_version') != 'task3-papera-v2':
            raise ValueError('Tray evidence adapter requires task3-papera-v2')
    def build_evidence(self, episode, curation_config):
        from dataset.task3.frame_evidence import build_frame_evidence
        return build_frame_evidence(episode, curation_config)

def dataset_adapter():
    return TrayEvidenceAdapter()
