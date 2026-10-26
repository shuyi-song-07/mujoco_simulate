"""Select a registered task and run the existing recorder plus local web UI."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from tasks.registry import ROOT, discover, get
from collection.config import recording_config

def environment():
    return json.loads((ROOT/'config/environments.json').read_text(encoding='utf-8'))

def free_port(port):
    with socket.socket() as sock:
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            sock.bind(('127.0.0.1', port))
        except OSError as exc:
            raise RuntimeError(f'Port {port} is occupied. Close its owning service after saving, or select another port.') from exc

def list_tasks():
    found=discover()
    for task in found.values():
        print(f"{task.id:26} {task.manifest['status']:12} {task.manifest['display_name']}")
    return found

def stop_idle_backend(port):
    """Use the recorder's existing exit command; never discard an episode here."""
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as response:
        status=json.load(response)
    if status.get('recording') or status.get('saving'):
        return False
    body=json.dumps({'command':'record_stop','session':status['session']}).encode()
    request=urllib.request.Request(f'http://127.0.0.1:{port}/control',data=body,
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=3) as response:
        json.load(response)
    return True

def start(args):
    task=get(args.task)
    config=recording_config(task)  # Resolve all resources before creating any subprocess.
    env=environment()
    python=Path(env['collection_python'])
    if not python.is_file(): raise FileNotFoundError(python)
    node=env.get('node_executable') or shutil.which('node')
    if not node or not Path(node).is_file(): raise RuntimeError('Configure node_executable in config/environments.json')
    for port in (args.web_port,args.backend_port): free_port(port)
    if args.web_port==args.backend_port: raise ValueError('Web/backend ports must differ')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    logs=ROOT/'logs'/task.id/stamp;logs.mkdir(parents=True)
    (logs/'task_binding.json').write_text(json.dumps(config['task_binding'],ensure_ascii=False,indent=2),encoding='utf-8')
    child_env=os.environ.copy()
    prefix=str(python.parent)
    child_env['PATH']=os.pathsep.join([prefix,str(python.parent/'Library/bin'),str(python.parent/'Scripts'),child_env.get('PATH','')])
    child_env.update(PYTHONUTF8='1',PYTHONNOUSERSITE='1',PORT=str(args.web_port),DUAL_BACKEND_PORT=str(args.backend_port))
    child_env.pop('PYTHONPATH',None)
    command=[str(python),'-m','simulation.mujoco.dual_arm.teleop_service','--task',task.id,
             '--port',str(args.backend_port),'--seed',str(args.seed),'--operator-id',args.operator_id,
             '--diagnostics-root',str(logs/'diagnostics')]
    if args.headless:command+=['--headless']
    handles=[];processes=[]
    try:
        for name,cmd in [('backend',command),('web',[node,str(ROOT/'collection/teleoperation/mediapipe/server.js')])]:
            handle=(logs/f'{name}.log').open('w',encoding='utf-8');handles.append(handle)
            processes.append(subprocess.Popen(cmd,cwd=ROOT,env=child_env,stdout=handle,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0))
        url=f'http://127.0.0.1:{args.web_port}/dual_arm.html'
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            if any(p.poll() is not None for p in processes): raise RuntimeError(f'Service exited; inspect {logs}')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{args.backend_port}/health',timeout=2) as response:
                    status=json.load(response)
                if status:break
            except (OSError,ValueError):time.sleep(.3)
        else:raise RuntimeError(f'Backend did not become ready; inspect {logs}')
        print(f'Task: {task.id}\nPage: {url}\nRaw: {ROOT / "datasets" / task.id}\nLogs: {logs}',flush=True)
        print('Keep this terminal open. Save/discard the episode before Ctrl+C.',flush=True)
        if not args.no_browser:webbrowser.open(url)
        while all(p.poll() is None for p in processes):
            try: time.sleep(.5)
            except KeyboardInterrupt:
                if stop_idle_backend(args.backend_port):
                    processes[0].wait(timeout=30)
                    break
                print('Episode active: save/discard in the web page, then press Ctrl+C again.',flush=True)
    except KeyboardInterrupt:pass
    finally:
        if processes and processes[0].poll() is None:
            try:
                if stop_idle_backend(args.backend_port):processes[0].wait(timeout=30)
            except (OSError,ValueError,KeyError,subprocess.TimeoutExpired):pass
        for p in processes:
            if p.poll() is None:
                if os.name=='nt':
                    subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True)
                else:p.terminate()
        for p in processes:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for handle in handles:handle.close()

def main(argv=None):
    env=environment()
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('list')
    sub.add_parser('menu')
    s=sub.add_parser('start');s.add_argument('--task',required=True)
    s.add_argument('--seed',type=int,default=None)
    s.add_argument('--operator-id',default='anonymous')
    s.add_argument('--web-port',type=int,default=env['web_port'])
    s.add_argument('--backend-port',type=int,default=env['backend_port'])
    s.add_argument('--headless',action='store_true');s.add_argument('--no-browser',action='store_true')
    args=p.parse_args(argv)
    if args.command=='list':list_tasks();return
    if args.command=='menu':
        list_tasks();return main(['start','--task',input('Task ID: ').strip()])
    if args.seed is None:args.seed=int.from_bytes(os.urandom(4),'big')%(2**31)
    start(args)
