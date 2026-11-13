"""Expose single-arm physical facts without inventing a second arm or grasp."""
class PushTEvidenceAdapter:
    version='papera-single-push-t-evidence-v1'

    def validate(self, metadata):
        if metadata.get('task_id')!='push_t' or metadata.get('schema_version')!='single-panda-push-v1':
            raise ValueError('Push-T requires its registered single-panda-push-v1 schema')

    def build_evidence(self, episode, curation_config):
        from dataset.task3.frame_evidence import build_frame_evidence
        self.validate(episode['metadata'])
        result=build_frame_evidence(episode,curation_config)
        for row,frame in zip(result['rows'],episode['frames'],strict=True):
            status=frame.get('status') or {}
            row['evidence']['push_alignment']={'available': 'push_t_coverage' in status,
                'coverage': status.get('push_t_coverage'), 'criterion': 'upstream-push-t-coverage-v1'}
        return result


def dataset_adapter():
    return PushTEvidenceAdapter()
