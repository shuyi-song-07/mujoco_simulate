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
    launch('ACT 训练','training.papera',task+['--variant','baseline','--config','config/training_baseline.json','--pack','${input:pack}']),
    launch('DP 训练','training.diffusion',task+['--variant','baseline','--config','config/training_diffusion.json','--pack','${input:pack}']),
    launch('ACT + 手势辅助监督训练','training.papera',task+['--variant','gesture','--config','config/training_act_gesture.json','--pack','${input:pack}']),
    launch('DP + 手势辅助监督训练','training.diffusion',task+['--variant','gesture','--config','config/training_diffusion_gesture.json','--pack','${input:pack}'])
], 'inputs':[
    {'id':'task','type':'promptString','description':'Registered task_id (python -m collection list)','default':'tray_to_platform'},
    {'id':'pack','type':'promptString','description':'Formal Training Pack absolute path; share Pack/split for ACT and DP comparisons'}]}
(folder/'launch.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(folder/'settings.json').write_text(json.dumps({'python.defaultInterpreterPath':env['training_python'],
    'python.terminal.activateEnvironment':False},indent=2)+'\n',encoding='utf-8')
print('VS Code shortcuts updated; no processes started.')
