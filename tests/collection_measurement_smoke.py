"""Short synthetic-recognition/real-MuJoCo/real-save journal regression."""
from pathlib import Path
import json
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from simulation.mujoco.dual_arm.teleop_service import DualArmRecorder


def main():
    folder=ROOT/'.local/checks/hand_journal'/str(time.time_ns());folder.mkdir(parents=True)
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    args=SimpleNamespace(task='tray_to_platform',seed=0,dataset_root=folder/'raw',no_previews=True,
        headless=True,port=port,diagnostics_root=folder/'logs',operator_id='scripted-engineering',
        input_kind='scripted_test',max_seconds=100)
    recorder=DualArmRecorder(args)
    inject_export_error='--inject-hand-export-error' in sys.argv
    if inject_export_error:
        original_export=recorder.measurement_journal.export
        calls=0
        def export_once_failed(*a,**kw):
            nonlocal calls
            calls+=1
            if calls==1:raise PermissionError('injected auxiliary export failure')
            return original_export(*a,**kw)
        recorder.measurement_journal.export=export_once_failed
    node=json.loads((ROOT/'config/environments.json').read_text())['node_executable']
    with (folder/'frontend.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen([node,str(ROOT/'tests/collection_measurement_smoke.mjs'),
            f'http://127.0.0.1:{port}',str(folder/'frontend.json')],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        try:recorder.run()
        finally:recorder.close()
        code=process.wait(timeout=10)
    if code:raise RuntimeError((folder/'frontend.log').read_text())
    front=json.loads((folder/'frontend.json').read_text())
    journal=recorder.measurement_journal
    rows=[json.loads(line) for line in journal.path.read_text(encoding='utf-8').splitlines()]
    measures=[r for r in rows if r['kind']=='measurement']
    assert front['journal']['completed']==front['journal']['written']==len(measures)
    assert front['journal']['dropped']==front['journal']['pending']==0
    research=recorder.roots['failure']/'research'
    hands_path=research/'episode_000000_hands.jsonl'
    hands=[json.loads(l) for l in hands_path.read_text(encoding='utf-8').splitlines()] if hands_path.exists() else []
    entries=[json.loads(l) for l in (research/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(entries)==2 and front['saved_failure']==2
    assert entries[1]['scene_index']==entries[0]['scene_index']+1
    assert hands and not entries[0]['hand_measurements']['requires_review']
    if inject_export_error:
        assert next(s for s in front['stages'] if s['name']=='sealed_save')['retries']==1
    for entry in entries:
        exported=research/f"episode_{entry['episode_index']:06d}_hands.jsonl"
        selected=[json.loads(l) for l in exported.read_text(encoding='utf-8').splitlines()]
        meta=json.loads(exported.with_suffix('.manifest.json').read_text(encoding='utf-8'))
        assert meta['save_barrier']['version']=='hand-save-boundary-v1'
        assert meta['save_barrier']['pending']==0
        assert not any(r.get('stage')=='after_save_boundary' for r in selected)
        for row in selected:
            if row['kind']=='measurement':
                assert meta['start_ms']<=row['measurement']['capture_epoch_ms']<=meta['end_ms']
                original=next(r for r in measures if r['measurement']['measurement_id']==row['measurement']['measurement_id'])
                assert original==row  # IDs, capture clocks, tick receipt and control trace unchanged.
    assert not entries[1]['hand_measurements']['requires_review']
    timeline=[json.loads(l) for l in (research/'episode_000000.jsonl').read_text(encoding='utf-8').splitlines()]
    executions=[r for r in timeline if r['kind']=='request_executed']
    accepts=[r for r in executions if r.get('result',{}).get('accepted') and r.get('request',{}).get('command')=='continuous_input']
    assert accepts
    unknown=[r for r in measures if any(h['gesture']=='None' for h in r['measurement']['hands'])]
    assert unknown and all(len(h['image_landmarks_21'])==21 for r in unknown for h in r['measurement']['hands'])
    assert any(not r['measurement']['hands'] for r in measures)
    report={'folder':str(folder),'dataset':str(recorder.roots['failure']),'completed':len(measures),
        'episode_measurements':sum(r['kind']=='measurement' for r in hands),'unknown_observations':len(unknown),
        'accepted_continuous_requests':len(accepts),'frontend':front,'human_camera_validation':False,'saved_episodes':len(entries),'injected_export_error':inject_export_error}
    (folder/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='frontend'},ensure_ascii=False))


if __name__=='__main__':main()
