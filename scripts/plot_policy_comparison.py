"""Plot logged ACT/DP objectives without conflating their loss definitions.

Use the collection environment (matplotlib). This command only reads runs;
continuation histories are joined by recorded lineage, never by directory age.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_policy_comparison import training_identity


def read_metrics(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError('Cyclic training continuation lineage')
    seen.add(path)
    run = json.loads((path/'run.json').read_text(encoding='utf-8'))
    parent = run.get('provenance', {}).get('continued_from', {})
    rows = []
    if parent:
        rows = [r for r in read_metrics(parent['run'], seen) if r['step'] <= parent['parent_step']]
    text = (path/'metrics.jsonl').read_text(encoding='utf-8')
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # Active trainers append a line at a time; ignore only an unfinished
            # final write, never a malformed completed/interior log record.
            if index != len(lines)-1 or text.endswith('\n'):
                raise
    # A resumed run can repeat a validation at its saved step. Keep the last.
    return sorted({(r['event'], r['step']): r for r in rows}.values(), key=lambda r: r['step'])


def prepare_rows(rows, algorithm, variant, config):
    """Never label epsilon MSE alone as a gesture policy's total objective."""
    prepared = [dict(row) for row in rows]
    for row in prepared:
        if algorithm != 'dp' or row['event'] != 'train':
            continue
        if variant == 'gesture':
            if not isinstance(row.get('gesture_loss'), (int, float)):
                raise ValueError('DP gesture total requires its logged auxiliary CE')
            weight = row.get('gesture_weight', config['model'].get('gesture_weight', .1))
            expected = row['noise_mse'] + weight * row['gesture_loss']
            if 'total_loss' in row and not np.isclose(row['total_loss'], expected, rtol=1e-5, atol=1e-7):
                raise ValueError('Logged DP total differs from noise MSE plus weighted auxiliary CE')
            row['total'] = row.get('total_loss', expected)
        else:
            row['total'] = row.get('total_loss', row['noise_mse'])
    return prepared


def draw(ax, rows, event, key, label, color, smooth=1):
    points = [(r['step'], r[key]) for r in rows if r['event'] == event
              and isinstance(r.get(key), (int, float)) and np.isfinite(r[key])]
    if not points:
        return
    x, y = np.array(points, float).T
    if smooth > 1 and len(y) >= smooth:
        # Descriptive trailing mean only; it never enters checkpoint selection.
        y = np.convolve(y, np.ones(smooth)/smooth, mode='valid')
        x = x[smooth-1:]
    ax.plot(x, y, label=label, color=color, linewidth=1.5)


def draw_closed_loop(path, output):
    """Visualize an audited comparison; never choose or alter model outcomes."""
    result = json.loads(Path(path).read_text(encoding='utf-8'))
    if result.get('version') != 'papera-four-group-comparison-v1':
        raise ValueError('Expected the four-group audit output')
    if result.get('engineering_self_check') or result.get('not_four_group_results'):
        raise ValueError('Engineering self-checks are not four-group evaluation results')
    names = ('act_baseline', 'act_gesture', 'dp_baseline', 'dp_gesture')
    labels = ('ACT baseline', 'ACT + gesture', 'DP baseline', 'DP + gesture')
    groups = [result['groups'][name] for name in names]
    count = len(result['seeds'])
    if count != 50 or len(set(result['seeds'])) != count:
        raise ValueError('Expected the same 50 unique initial states')
    for group in groups:
        if (group['successes'] + group['failures'] != count
                or not group.get('initial_physics_equal')
                or not group.get('cuda_and_nvidia_checked')):
            raise ValueError('Incomplete physical evaluation audit')
    rates = np.array([group['task_success_rate'] for group in groups]) * 100
    intervals = np.array([group['wilson95_given_checkpoint'] for group in groups]) * 100
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    x = np.arange(4)
    colors = ['#2465a8', '#c75d22'] * 2
    axes[0].bar(x, rates, color=colors, alpha=.85,
                yerr=np.maximum(0., np.stack((rates-intervals[:, 0], intervals[:, 1]-rates))), capsize=5)
    for i, group in enumerate(groups):
        axes[0].text(i, min(97, intervals[i, 1]+2),
                     f"{group['successes']}/50 ({rates[i]:.0f}%)", ha='center', fontsize=9)
    axes[0].set(ylim=(0, 105), ylabel='Success rate (%)', title='Success and 95% Wilson interval')
    bottom = np.zeros(4)
    for key, color in (('success', '#2b8c6b'), ('timeout', '#ddad48'),
                       ('contact_loss', '#bb594d'), ('other_failure', '#777d87')):
        values = []
        for group in groups:
            failures = group['failure_reasons']
            values.append(group['successes'] if key == 'success' else
                          sum(n for reason, n in failures.items() if reason not in ('timeout', 'contact_loss'))
                          if key == 'other_failure' else failures.get(key, 0))
        axes[1].bar(x, values, bottom=bottom, label=key.replace('_', ' '), color=color)
        bottom += np.array(values)
    if not np.all(bottom == count):
        raise ValueError('Failure categories do not add up to evaluated cases')
    axes[1].set(ylim=(0, 55), ylabel='Episodes', title='All outcomes, including failures')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=15)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Paper A: four policies on the same 50 unseen initial states\n'
                 'One training seed; intervals do not include training-seed uncertainty', fontsize=12)
    for extension in ('png', 'pdf'):
        fig.savefig(output/f'closed_loop_comparison.{extension}', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('act-baseline', 'act-gesture', 'dp-baseline', 'dp-gesture'):
        p.add_argument('--'+name, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--comparison', type=Path, help='Optional JSON from summarize_policy_comparison.py')
    args = p.parse_args()
    has_runs = any(getattr(args, n) for n in ('act_baseline','act_gesture','dp_baseline','dp_gesture'))
    if not has_runs and args.comparison is None:
        p.error('Specify at least one actual run or an audited comparison')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.comparison is not None:
        draw_closed_loop(args.comparison, args.output)
    if not has_runs:
        print(args.output.resolve())
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    report = {'curves': {}, 'train_smoothing_logged_points': 20,
              'note': 'Different losses and training posterior/validation prior are not numerically comparable.'}
    for row, algorithm in enumerate(('act','dp')):
        axes[row,0].set_yscale('log')
        for variant, color in (('baseline','#2465a8'), ('gesture','#c75d22')):
            path = getattr(args, algorithm+'_'+variant)
            if not path:
                continue
            identity = training_identity(path)
            if (identity['algorithm'], identity['variant']) != (algorithm, variant):
                raise ValueError(f'{algorithm}_{variant}: supplied run has a different algorithm/variant')
            rows = prepare_rows(read_metrics(path), algorithm, variant, identity['config'])
            draw(axes[row,0], rows, 'train', 'total', variant, color, smooth=20)
            key = 'prior_action_l1' if algorithm == 'act' else 'noise_mse'
            draw(axes[row,1], rows, 'validation', key, variant, color)
            if variant == 'gesture':
                draw(axes[row,2], rows, 'validation', 'gesture_loss', 'valid-label CE', color)
            valid = [r for r in rows if r['event'] == 'validation']
            report['curves'][algorithm+'_'+variant] = {
                'run': str(path.resolve()), 'final_logged_step': max((r['step'] for r in rows), default=None),
                'training_budget_steps': identity['config']['steps'],
                'curve_scope': 'All complete log records currently available, including frozen continuation history',
                'last_validation': valid[-1] if valid else None,
                'best_primary_validation': min((r for r in valid if key in r),
                                                key=lambda r:r[key], default=None)}
        titles = ['ACT posterior training total (log)', 'ACT z=0 validation action L1', 'ACT gesture validation CE'] if row == 0 else [
                  'DP training total (log)', 'DP EMA validation noise MSE', 'DP gesture validation CE']
        for ax, title in zip(axes[row], titles):
            ax.set_title(title)
            ax.set_xlabel('Cumulative optimizer updates')
            ax.grid(alpha=.2)
            if ax.lines:
                ax.legend()
            else:
                ax.text(.5,.5,'No completed data supplied',ha='center',transform=ax.transAxes)
    fig.suptitle('Paper A: fixed-data baseline / gesture comparison\nLoss curves are diagnostics; task success requires closed-loop evaluation', fontsize=14)
    for extension in ('png','pdf'):
        fig.savefig(args.output/f'training_curves.{extension}',dpi=160)
    plt.close(fig)
    (args.output/'training_curve_metrics.json').write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
