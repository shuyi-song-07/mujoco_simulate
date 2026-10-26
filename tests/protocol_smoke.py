"""Freeze/verify 50 static initial cases from engineering data; run zero policies."""
from pathlib import Path
import copy
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tasks.registry import get
from tasks.artifacts import check_artifact
from rollout.rollout import prepare_protocol, verify_protocol

if __name__=='__main__':
    task=get('tray_transfer')
    pointer=ROOT/'.local/checks/pipeline/outputs/dataset/tray_transfer/TRAINING_PACK_CURRENT.json'
    pack=json.loads(pointer.read_text(encoding='utf-8'))['path']
    check_artifact(pack,task.id,True)
    settings=json.loads((ROOT/task.manifest['evaluation_config']).read_text(encoding='utf-8'))
    settings.update(task_id=task.id,task_binding=task.binding(),scenario=task.manifest['scene_id'],pack=pack,
        protocol_path=f'outputs/rollout/{task.id}/protocols/default/protocol.json',output_root=f'outputs/rollout/{task.id}')
    path=ROOT/'.local/checks/protocol/protocol.json'
    protocol=prepare_protocol(settings,path)
    assert verify_protocol(path,settings)['signature']==protocol['signature']
    wrong=copy.deepcopy(settings);wrong['task_id']='push'
    try:verify_protocol(path,wrong)
    except ValueError:pass
    else:raise AssertionError('cross-task protocol accepted')
    report={'engineering_only':True,'policy_simulations_started':0,'static_cases':len(protocol['seeds']),
            'cross_task_rejected':True,'summary':protocol['preflight']['summary'],'protocol':str(path)}
    (ROOT/'.local/checks/protocol_smoke.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))
