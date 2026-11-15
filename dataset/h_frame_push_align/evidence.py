"""Recorded H-frame alignment facts, using the shared S16/A14 evidence schema."""
import numpy as np

from tasks.h_frame_push_align.success import alignment_errors


class PushAlignmentEvidenceAdapter:
    version = 'papera-h-frame-alignment-evidence-v2'
    criterion = 'papera-h-frame-push-align-v1'

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
            evidence['task_semantics'] = {'contact': 'nonprehensile_push', 'completion': 'task_monitor'}
            physical = status.get('physical_contact') or {}
            completion = physical.get('task_completion') or {}
            limits = completion.get('completion_requirements') or {}
            conditions = completion.get('final_conditions') or {}
            support = physical.get('support_checks') or {}
            required = ('position_xy', 'yaw', 'upright', 'inside_table', 'table_support',
                        'linear_speed', 'angular_speed', 'not_failed')
            keys = ('position_tolerance_m', 'yaw_tolerance_rad', 'max_tilt_rad', 'stable_seconds')
            valid_limits = all(type(limits.get(k)) in (int, float) and np.isfinite(limits[k])
                               and limits[k] > 0 for k in keys)
            error = evidence['push_alignment']['target_error']
            verified = bool(available and valid_limits
                and completion.get('task_success_criterion') == self.criterion
                and completion.get('task_success') is True
                and all(conditions.get(k) is True for k in required)
                and all(support.get(k) is True for k in ('table_load', 'clearance', 'penetration'))
                and error['xy_m'] <= limits['position_tolerance_m']
                and error['orientation_rad'] <= limits['yaw_tolerance_rad']
                and np.arccos(np.clip(rotation[2, 2], -1, 1)) <= limits['max_tilt_rad']
                and completion.get('alignment_stable_seconds', -1) >= limits['stable_seconds'] - 1e-9)
            # Historical records lacking monitor evidence stay conservative;
            # DONE alone must never invent a physical success certificate.
            evidence['task_completion'] = {'kind': 'push_alignment', 'verified': verified,
                'criterion': self.criterion, 'monitor': completion}
        return result


def dataset_adapter():
    return PushAlignmentEvidenceAdapter()
