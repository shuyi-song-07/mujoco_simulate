"""Generate editor shortcuts from the same environment config; no training starts."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
env=json.loads((ROOT/'config/environments.json').read_text(encoding='utf-8'))
folder=ROOT/'.vscode';folder.mkdir(exist_ok=True)
def launch(name,module,args,kind='training'):
    return {'name':name,'type':'debugpy','request':'launch','module':module,
            'python':env[kind+'_python'],'cwd':'${workspaceFolder}','console':'integratedTerminal',
            'justMyCode':True,'args':args,'env':{'PYTHONUTF8':'1','PYTHONNOUSERSITE':'1','PYTHONPATH':''}}
task=['--task','${input:task}']
doc={'version':'0.2.0','configurations':[
    launch('Paper A · Collection','collection',['start',*task],kind='collection'),
    launch('Paper A · Baseline 20000 steps','training.papera',task+['--variant','baseline','--pack','${input:pack}']),
    launch('Paper A · Gesture 20000 steps','training.papera',task+['--variant','gesture','--pack','${input:pack}']),
    launch('Paper A · Resume interrupted run','training.papera',task+['--resume','${input:last}']),
    launch('Paper A · Rollout check','rollout',task+['--mode','check','--run','${input:run}']),
    launch('Paper A · Rollout preview 60 s','rollout',task+['--mode','preview','--run','${input:run}','--seed','910001','--max-seconds','60'])
], 'inputs':[
    {'id':'task','type':'promptString','description':'Registered task_id (python -m collection list)','default':'tray_transfer'},
    {'id':'pack','type':'promptString','description':'Verified Training Pack absolute path; same Pack for both variants'},
    {'id':'last','type':'promptString','description':'Exact checkpoints/last.pt path; keeps frozen original budget'},
    {'id':'run','type':'promptString','description':'Training run directory containing run.json (not a .pt file)'}]}
(folder/'launch.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(folder/'settings.json').write_text(json.dumps({'python.defaultInterpreterPath':env['training_python'],
    'python.terminal.activateEnvironment':False},indent=2)+'\n',encoding='utf-8')
print('VS Code shortcuts updated; no processes started.')
