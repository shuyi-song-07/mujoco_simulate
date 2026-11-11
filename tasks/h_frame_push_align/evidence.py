"""Recorded H-frame alignment facts, using the shared S16/A14 evidence schema."""
import numpy as np

from .success import alignment_errors


class PushAlignmentEvidenceAdapter:
    version = 'papera-h-frame-alignment-evidence-v1'

    def validate(self, metadata):
        if (metadata.get('task_id') != 'h_frame_push_align'
                or metadata.get('schema_version') != 'task3-papera-v2'):
            raise ValueError('H-frame evidence requires h_frame_push_align task3-papera-v2 recordings')

    def build_evidence(self, episode, curation_config):
        from dataset.task3.frame_evidence import build_frame_evidence

        self.validate(episode['metadata'])
        result = build_frame_evidence(episode, curation_config)
        for row, frame in zip(result['rows'], episode['frames'], strict=True):
            status = frame.get('status') or {}
            evidence = row['evidence']
            obj = evidence['object']
            position = np.asarray(obj['position'], dtype=float)
            rotation = np.asarray(obj['rotation'], dtype=float)
            target = np.asarray(status.get('target_position'), dtype=float)
            yaw = status.get('target_yaw_rad')
            available = (position.shape == (3,) and rotation.shape == (3, 3) and target.shape == (3,)
                         and all(np.isfinite(x).all() for x in (position, rotation, target))
                         and type(yaw) in (int, float) and np.isfinite(yaw))
            # A pose match is not terminal success. Keep the real DONE,
            # contact, gripper and motion evidence used by Safe Stitch.
            evidence['push_alignment'] = {
                'available': bool(available),
                'target_error': alignment_errors(position, rotation, target, yaw) if available else None,
            }
        return result


def dataset_adapter():
    return PushAlignmentEvidenceAdapter()
