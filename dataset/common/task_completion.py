"""Push-only tail budget on the raw clock. Grasp release rules stay separate."""
import math


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def terminal_task_completion(rows, cfg, *, sides, window, stable_width, object_settled):
    if not rows:
        return None
    final_segment = rows[-1]['task_segment_id']
    first_good = next((i for i, row in enumerate(rows)
        if row['task_segment_id'] == final_segment and row['frame_role'] != 'INVALID'
        and row['evidence'].get('source', {}).get('valid') is True
        and row['evidence'].get('source', {}).get('available') is True
        and row['evidence'].get('task', {}).get('done') is True
        and row['evidence'].get('task', {}).get('success') is True
        and row['evidence'].get('task_completion', {}).get('verified') is True
        and _finite(row.get('original_time_s'))), None)
    if first_good is None:
        return None
    i = first_good
    evidence = rows[i]['evidence']['task_completion']
    observed_time = rows[i]['original_time_s']
    recorded_time = evidence.get('recorded_success_time_s')
    # Never reuse a stale success from a previous segment or a future timestamp.
    previous_time = rows[i-1]['original_time_s'] if i else observed_time
    exact = (_finite(recorded_time) and _finite(previous_time)
             and previous_time <= recorded_time <= observed_time)
    success_time = recorded_time if exact else observed_time
    seconds = cfg['waits'].get('post_push_success_seconds', .5)
    cutoff = success_time + seconds
    last = i
    for j in range(i, len(rows)):
        time = rows[j].get('original_time_s')
        if not _finite(time) or time > cutoff + 1e-9:
            break
        last = j
    # Keep anomaly evidence even when its images are outside the training tail.
    anomalies = {
        'monitor_outside_after_success': sum(
            r['evidence'].get('task_completion', {}).get('verified') is not True for r in rows[i+1:]),
        'later_failure_frames': sum(r['evidence'].get('task', {}).get('state') == 'FAIL' for r in rows[i+1:]),
    }
    return {'kind': 'task_completion', 'policy': 'push_done_plus_seconds_v1',
        'completion_index': i, 'completion_raw_frame': rows[i]['raw_frame_index'],
        'completion_time_s': success_time, 'observed_completion_time_s': observed_time,
        'clock_source': 'recorded_success_time' if exact else 'first_verified_done_sample',
        'task_segment_id': final_segment, 'release_event': None,
        'post_success_seconds': seconds, 'cutoff_time_s': cutoff,
        'retained_post_frames': last-i, 'last_kept_index': last,
        'last_kept_raw_frame': rows[last]['raw_frame_index'],
        'excluded_frames': len(rows)-last-1, 'tail_anomalies': anomalies,
        'evidence': evidence}
