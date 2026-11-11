"""Push-R recording validation with the shared planar-alignment evidence."""
from tasks.h_frame_push_align.evidence import PushAlignmentEvidenceAdapter


class PushREvidenceAdapter(PushAlignmentEvidenceAdapter):
    version = 'papera-push-r-alignment-evidence-v1'

    def validate(self, metadata):
        if (metadata.get('task_id') != 'push_r'
                or metadata.get('schema_version') != 'task3-papera-v2'):
            raise ValueError('Push-R evidence requires push_r task3-papera-v2 recordings')


def dataset_adapter():
    return PushREvidenceAdapter()
