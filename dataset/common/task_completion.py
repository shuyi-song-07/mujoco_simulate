"""Consume task-adapter completion evidence; never infer success from gestures."""
import math


def terminal_task_completion(rows, cfg, *, sides, window, stable_width, object_settled):
    final_segment = rows[-1]['task_segment_id']
    def servo_complete(facts):
        for side in sides:
            g = facts.get('grippers', {}).get(side, {})
            actual, goal = g.get('actual_width_m'), g.get('goal_width_m')
            if not (type(actual) in (int, float) and type(goal) in (int, float)
                    and math.isfinite(actual) and math.isfinite(goal)
                    and abs(actual-goal) <= cfg['protection']['aperture_stability_m']):
                return False
        return True

    last_execution = max((i for i, r in enumerate(rows) if r['task_segment_id'] == final_segment
        and any(e.get('kind') in ('gripper_open_executed', 'gripper_close_executed')
                for e in r['evidence'].get('events', []))), default=-1)
    good_suffix = True
    suffix_start = len(rows)
    for i in range(len(rows) - 1, -1, -1):
        row = rows[i]
        if row['task_segment_id'] != final_segment:
            break
        f = row['evidence']
        good_suffix &= (row['frame_role'] != 'INVALID'
            and f.get('source', {}).get('valid') is True
            and f.get('source', {}).get('available') is True
            and f.get('task', {}).get('done') is True
            and f.get('task', {}).get('success') is True
            and f.get('task_completion', {}).get('verified') is True and servo_complete(f))
        if not good_suffix:
            break
        suffix_start = i
    first_good = None
    for i in range(max(suffix_start, last_execution), len(rows)):
        sample = window(rows, i, cfg['protection']['settle_seconds'])
        if (not sample or sample[0]['task_segment_id'] != final_segment or i < last_execution
                or (last_execution >= 0 and sample[0]['raw_frame_index'] < rows[last_execution]['raw_frame_index'])):
            continue
        # Completion does not truncate an opening/closing still in progress.
        settled = all(stable_width(sample, s, cfg['protection']['aperture_stability_m']) for s in sides)
        for r in sample:
            facts = r['evidence']
            settled &= (facts.get('source', {}).get('valid') is True
                        and facts.get('source', {}).get('available') is True and servo_complete(facts))
        if settled and object_settled(sample, cfg) and object_settled(rows[i:], cfg):
            first_good = i
            break
    if first_good is None:
        return None
    i = first_good
    last = min(len(rows) - 1, i + cfg['waits']['post_release_frames'])
    return {'kind': 'task_completion', 'completion_index': i,
        'completion_raw_frame': rows[i]['raw_frame_index'], 'completion_time_s': rows[i]['original_time_s'],
        'task_segment_id': final_segment, 'release_event': None,
        'post_release_frames': cfg['waits']['post_release_frames'],
        'retained_post_frames': last-i, 'last_kept_index': last,
        'last_kept_raw_frame': rows[last]['raw_frame_index'], 'excluded_frames': len(rows)-last-1,
        'evidence': rows[i]['evidence']['task_completion']}
