"""Run ACT/DP baselines, then evaluate their validation-best models on 50 cases.

Uses the existing training and rollout CLIs, never substitutes a model, Pack,
or task criterion. An already running ACT may be attached by its verified PID.
All campaign snapshots, commands, logs and failures remain in one output folder.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def resolve(path):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_evaluations(act, dp):
    """Execution differs by algorithm; physical task and all 50 draws do not."""
    allowed = {'execution', 'n_action_steps', 'checkpoint_selection', 'policy_algorithm', '_help'}
    require({k: v for k, v in act.items() if k not in allowed} ==
            {k: v for k, v in dp.items() if k not in allowed}, 'ACT/DP physical evaluation settings differ')
    require(len(act['test_seeds']) == len(set(act['test_seeds'])) == 50, 'Exactly 50 unique seeds are required')
    require(act['max_seconds'] == dp['max_seconds'] == 60, 'This campaign requires a 60-second simulation deadline')
    require(act['observation_hz'] == 30 and act['control_hz'] == 60, 'Expected 30 Hz model / 60 Hz controller')
    require(dp.get('policy_algorithm') == 'diffusion_policy' and dp['execution'] == 'chunked'
            and dp['n_action_steps'] == 8, 'DP requires its explicit H2 / action-horizon-8 execution config')


def validate_run(run, algorithm, plan):
    value = read(Path(run) / 'run.json')
    if algorithm == 'act':
        source = value['provenance']
        require(source['variant'] == 'baseline', 'Attached ACT must be a baseline without gesture supervision')
        config, pack, task = source['config'], source['pack'], source['task_id']
    else:
        require(value['algorithm'] == 'diffusion_policy', 'Expected a DP run')
        config, pack, task = value['config'], value['pack_path'], value['task_id']
        require(all(variant == 'baseline' for variant in (
            value.get('variant', 'baseline'), config.get('variant', 'baseline'),
            config.get('model', {}).get('variant', 'baseline'))),
            'DP campaign requires a baseline without gesture supervision')
    require(task == plan['task_id'] and resolve(pack) == resolve(plan['pack']), 'Training run task/Pack differs from campaign')
    expected = plan['training_configs'][algorithm]
    if algorithm == 'dp':
        # The CLI materializes --variant in model; compare equivalent defaults
        # without rewriting either the frozen plan or the run's configuration.
        config, expected = dict(config), dict(expected)
        for item in (config, expected):
            if 'model' in item:
                item['model'] = {'variant': 'baseline', **item['model']}
    for key, wanted in expected.items():
        if key not in {'pack_path', 'output_root'}:
            require(config.get(key) == wanted, f'{algorithm} frozen training setting differs: {key}')
    return value


def training_complete(run, algorithm, plan):
    validate_run(run, algorithm, plan)
    path = Path(run) / 'completion.json'
    if not path.is_file():
        return False
    value = read(path)
    if algorithm == 'act':
        ok = value.get('completed') is True and value['step'] == value['target_steps'] == plan['training_configs']['act']['steps']
        ok &= value.get('engineering_smoke') is False and not value.get('engineering_overfit', False)
    else:
        ok = value.get('status') == 'complete' and value['step'] == value['budget_steps'] == plan['training_configs']['dp']['steps']
        ok &= value.get('engineering') is False
    require(ok, f'{algorithm} training stopped without completing its frozen budget: {path}')
    require((Path(run) / 'checkpoints/best.pt').is_file(), f'{algorithm} best checkpoint is missing')
    return True


def validate_act_command(arguments, run):
    require('-m' in arguments and arguments[arguments.index('-m') + 1] == 'training.papera', 'PID is not the ACT trainer')
    require('--output' in arguments and resolve(arguments[arguments.index('--output') + 1]) == resolve(run),
            'PID does not own the selected ACT output directory')


class AttachedACT:
    """Keep a Windows process handle, so PID reuse cannot prolong an old run."""
    def __init__(self, pid, run):
        require(os.name == 'nt', '--act-pid attachment currently supports this Windows workstation')
        import ctypes
        from ctypes import wintypes
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.kernel.OpenProcess(0x100000 | 0x1000, False, pid)
        if not self.handle:
            raise OSError(f'Cannot open ACT PID {pid}; check its completion before attaching')
        try:
            command = ("[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding; "
                       f"Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}' | Select-Object -ExpandProperty CommandLine")
            output = subprocess.check_output(['powershell.exe', '-NoProfile', '-Command', command], encoding='utf-8', errors='replace')
            shell = ctypes.WinDLL('shell32', use_last_error=True)
            shell.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
            shell.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
            count = ctypes.c_int()
            pointer = shell.CommandLineToArgvW(output.strip(), ctypes.byref(count))
            if not pointer:
                raise OSError('Cannot parse the attached ACT command line')
            try:
                self.arguments = [pointer[i] for i in range(count.value)]
            finally:
                self.kernel.LocalFree.argtypes = [wintypes.HLOCAL]
                self.kernel.LocalFree(pointer)
            require(self.running(), 'ACT exited while its PID was being verified; check completion and retry')
            validate_act_command(self.arguments, run)
        except BaseException:
            self.close()
            raise

    def running(self):
        status = self.kernel.WaitForSingleObject(self.handle, 0)
        if status == 0xFFFFFFFF:
            raise OSError('Cannot query attached ACT process state')
        return status == 258

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def validate_protocol_pair(act, dp):
    require(act['seeds'] == dp['seeds'], 'ACT/DP protocol seeds differ')
    require(act['source'] == dp['source'], 'ACT/DP protocol source scenes differ')
    a = [(r['seed'], r['initial_state']) for r in act['preflight']['cases']]
    b = [(r['seed'], r['initial_state']) for r in dp['preflight']['cases']]
    require(a == b, 'ACT/DP protocol initial states differ')


def gpu_preview(output):
    """A 2 s timeout is an expected model outcome, not an infrastructure error."""
    import numpy as np
    rows = list((Path(output) / 'episodes').glob('seed_*/result.json'))
    require(len(rows) == 1, 'GPU preview did not finish exactly one case')
    row, runtime = read(rows[0]), read(rows[0].with_name('runtime.json'))
    require(not row.get('infrastructure_error'), 'GPU preview had an infrastructure error')
    require(runtime['device'] == 'cuda' and 'nvidia' in str(runtime['gpu']).lower(), 'Policy inference did not use NVIDIA CUDA')
    gl = runtime['opengl']
    require('nvidia' in (str(gl['vendor']) + str(gl['renderer'])).lower(), 'MuJoCo rendering did not use NVIDIA OpenGL')
    for key in ('mean_inference_ms', 'p95_inference_ms'):
        require(np.isfinite(row[key]) and row[key] > 0, 'GPU preview did not record model inference timing')
    return {k: runtime[k] for k in ('device', 'gpu', 'opengl')} | {
        k: row[k] for k in ('mean_inference_ms', 'p95_inference_ms', 'failure_reason')}


def evaluation_result(output):
    import numpy as np
    output = Path(output)
    summary, metadata = read(output / 'summary.json'), read(output / 'run.json')
    require(summary.get('complete') is True and summary['planned_episodes'] == summary['completed_episodes'] == 50,
            'Evaluation is not a complete 50-case result')
    require(not summary.get('infrastructure_errors'), 'Evaluation has infrastructure failures')
    require(metadata['device'] == 'cuda' and metadata['max_seconds'] == 60, 'Evaluation device/deadline differs')
    require(len(metadata['seeds']) == len(set(metadata['seeds'])) == 50, 'Evaluation did not retain all 50 unique seeds')
    latencies = []
    successes = 0
    for seed in metadata['seeds']:
        case = output / 'episodes' / f'seed_{seed}'
        result, runtime = read(case / 'result.json'), read(case / 'runtime.json')
        require(result['seed'] == seed and result.get('infrastructure_error') is False, 'A case is incomplete or mismatched')
        require(runtime['device'] == 'cuda' and 'nvidia' in str(runtime['gpu']).lower(), 'A batch case did not use NVIDIA CUDA')
        require('nvidia' in (runtime['opengl']['vendor'] + runtime['opengl']['renderer']).lower(), 'A batch case did not use NVIDIA OpenGL')
        successes += int(result['success'])
        with np.load(case / 'trajectory.npz', allow_pickle=False) as arrays:
            latencies.extend(arrays['inference_ms'][arrays['prediction_query_mask'].astype(bool)].tolist())
    require(successes == summary['successes'] and summary['failures'] == 50 - successes,
            'Summary success counts differ from the 50 completed results')
    require(latencies and np.isfinite(latencies).all(), 'Missing inference latency evidence')
    return {k: summary[k] for k in ('successes', 'failures', 'task_success_rate', 'failure_reasons', 'wilson95_given_checkpoint')} | {
        'episodes': 50, 'mean_inference_ms': float(np.mean(latencies)),
        'p95_inference_ms': float(np.quantile(latencies, .95)), 'output': str(output)}


class Campaign:
    def __init__(self, directory, plan):
        self.directory, self.plan = Path(directory), plan
        self.directory.mkdir(parents=True, exist_ok=False)
        self.status = {'version': 'papera-baseline-campaign-v1', 'state': 'running', 'stages': [],
                       'plan': str(self.directory / 'plan.json'), 'owner_pid': os.getpid()}
        write(self.directory / 'plan.json', plan)
        for kind, configs in (('training', plan['training_configs']), ('evaluation', plan['evaluation_configs'])):
            for name, config in configs.items():
                write(self.directory / 'configs' / f'{kind}_{name}.json', config)
        self.save()

    def save(self):
        self.status['updated_at'] = datetime.now(timezone.utc).isoformat()
        write(self.directory / 'status.json', self.status)

    def command(self, name, arguments):
        if 'pack_manifest' in self.plan:
            require(read(Path(self.plan['pack']) / 'manifest.json') == self.plan['pack_manifest'],
                    'Campaign Pack manifest changed; refusing to switch training or evaluation inputs')
            for kind, configs in (('training', self.plan['training_configs']), ('evaluation', self.plan['evaluation_configs'])):
                for algorithm, expected in configs.items():
                    require(read(self.directory / 'configs' / f'{kind}_{algorithm}.json') == expected,
                            'Campaign configuration snapshot changed')
        log = self.directory / 'logs' / f'{name}.log'
        log.parent.mkdir(exist_ok=True)
        command = [self.plan['python'], '-u', '-X', 'utf8', *map(str, arguments)]
        stage = {'name': name, 'state': 'starting', 'command': command, 'stdout': str(log)}
        self.status['stages'].append(stage)
        self.save()
        environment = dict(os.environ, PYTHONUTF8='1', PYTHONNOUSERSITE='1', PYTHONPATH='')
        print(f'[{name}] {subprocess.list2cmdline(command)}\nLog: {log}', flush=True)
        with log.open('w', encoding='utf-8') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            stage.update(pid=process.pid, state='running')
            self.save()
            try:
                code = process.wait()
            except BaseException:
                # Let the formal trainer checkpoint its own completed updates.
                stage['state'] = 'supervisor_interrupted_child_may_still_run'
                self.save()
                raise
        stage.update(exit_code=code, state='complete' if code == 0 else 'failed')
        self.save()
        if code != 0:
            raise RuntimeError(f'{name} exited with code {code}; see {log}')

    def wait_act(self, run, pid, poll_seconds):
        if (Path(run) / 'run.json').is_file() and training_complete(run, 'act', self.plan):
            self.status['stages'].append({'name': 'reuse_completed_act', 'state': 'complete', 'run': str(run),
                'completion': str(Path(run) / 'completion.json'), 'stdout': self.plan.get('attached_act_log')})
            self.save()
            return
        require(pid is not None, 'Incomplete --act-run requires its actual --act-pid; no duplicate trainer will be started')
        process = AttachedACT(pid, run)
        stage = {'name': 'attach_act', 'state': 'running', 'pid': pid, 'command': process.arguments,
                 'stdout': self.plan.get('attached_act_log'), 'run': str(run)}
        self.status['stages'].append(stage)
        self.save()
        print(f'[attach_act] Waiting for verified PID {pid}: {run}', flush=True)
        try:
            while process.running():
                if (Path(run) / 'run.json').is_file():
                    validate_run(run, 'act', self.plan)
                time.sleep(poll_seconds)
            require(training_complete(run, 'act', self.plan), 'Attached ACT exited without a complete training result')
            stage['state'] = 'complete'
            self.save()
        finally:
            process.close()

    def execute(self, attached_act=None, act_pid=None, poll_seconds=15):
        p, outputs = self.plan, self.plan['outputs']
        try:
            if attached_act:
                self.wait_act(attached_act, act_pid, poll_seconds)
            else:
                self.command('train_act', ['-m', 'training.papera', '--task', p['task_id'], '--variant', 'baseline',
                    '--pack', p['pack'], '--config', self.directory / 'configs/training_act.json', '--output', outputs['act_run']])
                require(training_complete(outputs['act_run'], 'act', p), 'ACT budget was not completed')
            self.command('train_dp', ['-m', 'training.diffusion', '--task', p['task_id'], '--variant', 'baseline',
                '--pack', p['pack'], '--config', self.directory / 'configs/training_dp.json', '--output', outputs['dp_run']])
            require(training_complete(outputs['dp_run'], 'dp', p), 'DP budget was not completed')
            for algorithm in ('act', 'dp'):
                common = ['-m', 'rollout', '--task', p['task_id'], '--config',
                          self.directory / f'configs/evaluation_{algorithm}.json']
                protocol = self.directory / 'protocols' / f'{algorithm}.json'
                self.command(f'prepare_{algorithm}', [*common, '--mode', 'prepare', '--pack', p['pack'], '--protocol', protocol])
            validate_protocol_pair(read(self.directory / 'protocols/act.json'), read(self.directory / 'protocols/dp.json'))
            results = {}
            for algorithm in ('act', 'dp'):
                common = ['-m', 'rollout', '--task', p['task_id'], '--config',
                          self.directory / f'configs/evaluation_{algorithm}.json', '--run', outputs[f'{algorithm}_run'],
                          '--checkpoint', 'best', '--device', 'cuda', '--workers', '1']
                self.command(f'check_{algorithm}', [*common, '--mode', 'check'])
                preview = self.directory / 'gpu_previews' / algorithm
                self.command(f'gpu_preview_{algorithm}', [*common, '--mode', 'preview', '--headless', '--seed', '910001',
                    '--max-seconds', '2', '--output', preview])
                evidence = gpu_preview(preview)
                self.status.setdefault('gpu_evidence', {})[algorithm] = evidence
                self.save()
                self.command(f'evaluate_{algorithm}', [*common, '--mode', 'batch', '--protocol',
                    self.directory / 'protocols' / f'{algorithm}.json', '--output', outputs[f'{algorithm}_evaluation']])
                results[algorithm] = evaluation_result(outputs[f'{algorithm}_evaluation'])
                write(self.directory / 'results.json', results)
            self.status.update(state='complete', results=results)
            self.save()
            print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
        except BaseException as error:
            self.status.update(state='failed', error=f'{type(error).__name__}: {error}')
            self.save()
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='New campaign folder; existing folders are never overwritten')
    parser.add_argument('--act-run', type=Path, help='Reuse a completed ACT baseline or attach its active trainer')
    parser.add_argument('--act-pid', type=int, help='Actual running ACT trainer PID; required when attached ACT is incomplete')
    parser.add_argument('--act-log', type=Path, help='Existing attached ACT stdout log for navigation')
    parser.add_argument('--poll-seconds', type=float, default=15)
    parser.add_argument('--dp-evaluation-config', type=Path, default=ROOT / 'config/evaluation_diffusion.json')
    args = parser.parse_args(argv)
    require(1 <= args.poll_seconds <= 60, 'poll-seconds must be between 1 and 60')
    require(args.act_pid is None or args.act_run is not None, '--act-pid requires --act-run')
    require(args.act_pid is None or args.act_pid > 0, '--act-pid must be positive')
    from tasks.registry import get
    from tasks.artifacts import check_artifact
    task, pack = get(args.task), resolve(args.pack)
    manifest = check_artifact(pack, args.task, engineering=False)
    require(manifest.get('complete') is True and manifest.get('data_ready') is True, 'Pack is not complete')
    require(manifest.get('cameras') == ['oblique'] and manifest.get('state_dim') == 16 and manifest.get('action_dim') == 14,
            'Expected single oblique / S16 / A14 Pack')
    configs = {'act': read(ROOT / 'config/training_baseline.json'), 'dp': read(ROOT / 'config/training_diffusion.json')}
    for field in ('seed', 'steps', 'batch_size', 'image_size'):
        require(configs['act'][field] == configs['dp'][field], f'Baseline training budgets differ: {field}')
    evaluations = {'act': read(resolve(task.manifest['evaluation_config'])), 'dp': read(resolve(args.dp_evaluation_config))}
    validate_evaluations(evaluations['act'], evaluations['dp'])
    identifier = datetime.now().strftime('baselines_%Y%m%d_%H%M%S_%f')
    output = resolve(args.output or ROOT / 'outputs/campaigns' / task.id / identifier)
    runs = {'act_run': str(resolve(args.act_run)) if args.act_run else str(ROOT / 'outputs/training' / task.id / 'baseline' / identifier),
            'dp_run': str(ROOT / 'outputs/training' / task.id / 'diffusion' / identifier)}
    runs.update({f'{a}_evaluation': str(ROOT / 'outputs/rollout' / task.id / variant / identifier)
                 for a, variant in (('act', 'baseline'), ('dp', 'diffusion'))})
    plan = {'version': 'papera-baseline-campaign-v1', 'task_id': task.id, 'task_binding': task.binding(), 'pack': str(pack),
            'pack_manifest': manifest, 'training_configs': configs, 'evaluation_configs': evaluations,
            'python': read(ROOT / 'config/environments.json')['training_python'], 'outputs': runs,
            'attached_act_log': str(resolve(args.act_log)) if args.act_log else None,
            'checkpoint': 'best', 'episodes_per_model': 50, 'max_seconds': 60, 'device': 'cuda'}
    campaign = Campaign(output, plan)
    print(f'Campaign: {output}', flush=True)
    campaign.execute(resolve(args.act_run) if args.act_run else None, args.act_pid, args.poll_seconds)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
