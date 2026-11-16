"""One task-evidence route for Audit and Curation; no raw metadata rewriting."""
import math


def build_task_evidence(episode, config):
    frozen = episode.get('metadata', {}).get('config', {})
    if frozen.get('task_binding'):
        from tasks.registry import hook
        result = hook(frozen, 'dataset_adapter')().build_evidence(episode, config)
    else:
        from dataset.task3.frame_evidence import build_frame_evidence
        result = build_frame_evidence(episode, config)
    frames = episode.get('frames')
    if frames is None:
        return result
    for row, frame in zip(result['rows'], frames, strict=True):
        facts = row['evidence']
        alignment = facts.get('push_alignment', {})
        status = frame.get('status') or {}
        # Single-arm completion uses its coverage/dwell monitor, not a grasp.
        if alignment.get('criterion') == 'upstream-push-t-coverage-v1':
            monitor = status.get('push_success_evidence') or {}
            dwell = frozen.get('scene', {}).get('push_t', {}).get('dwell_seconds')
            elapsed = monitor.get('dwell_elapsed_s')
            verified = (monitor.get('criterion') == alignment['criterion']
                        and monitor.get('task_success') is True
                        and type(dwell) in (int, float) and math.isfinite(dwell) and dwell > 0
                        and type(elapsed) in (int, float) and math.isfinite(elapsed)
                        and elapsed >= dwell - 1e-9)
            facts['task_semantics'] = {'contact': 'nonprehensile_push', 'completion': 'task_monitor'}
            facts['task_completion'] = {'kind': 'push_coverage', 'verified': verified,
                'criterion': alignment['criterion'], 'monitor': monitor}
        if facts.get('task_semantics', {}).get('completion') == 'task_monitor':
            completion = facts.get('task_completion', {})
            completion['recorded_success_time_s'] = (status.get('success_tail') or {}).get('success_time')
    return result
