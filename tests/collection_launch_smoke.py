"""Exercise the actual launcher/proxy/assets, then request an idle recorder exit."""
from pathlib import Path
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.request
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from collection.cli import main, stop_idle_backend

def port():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));return s.getsockname()[1]

def run(task='tray_transfer', browser_check=False):
    from tasks.registry import get
    selected = get(task)
    web,backend=port(),port()
    errors=[]
    def launch():
        try:main(['start','--task',selected.id,'--seed','731999','--headless','--no-browser',
                  '--web-port',str(web),'--backend-port',str(backend)])
        except Exception as exc:errors.append(repr(exc))
    thread=threading.Thread(target=launch);thread.start()
    base=f'http://127.0.0.1:{web}'
    report={}
    try:
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            try:
                with urllib.request.urlopen(base+'/api/dual/health',timeout=2) as r:status=json.load(r)
                break
            except OSError:
                if errors:raise RuntimeError(errors)
                time.sleep(.3)
        else:raise TimeoutError('collection launch/proxy')
        assert Path(status['project_root']).resolve()==ROOT
        assert not status['recording']
        for path in ['/dual_arm.html','/dual-arm-app.js','/models/gesture_recognizer.task',
                     '/node_modules/@mediapipe/tasks-vision/vision_bundle.mjs']:
            with urllib.request.urlopen(base+path,timeout=10) as r:
                report[path]={'status':r.status,'bytes':len(r.read())}
        # The proxy may answer before the launcher's 300 ms readiness poll returns.
        time.sleep(1)
        if browser_check:
            subprocess.run([sys.executable, '-X', 'utf8', str(ROOT/'tests/collection_browser_smoke.py'),
                            '--url', base+'/dual_arm.html'], cwd=ROOT, check=True, timeout=60)
            report['browser_connected'] = True
        # Exercise real preview rendering, source age and proxy header forwarding.
        previews={}
        for camera in ('top','front','oblique'):
            deadline=time.monotonic()+10
            while time.monotonic()<deadline:
                with urllib.request.urlopen(base+f'/api/dual/{camera}-preview',timeout=3) as response:
                    if response.status==200:
                        image=response.read()
                        assert image.startswith(b'\xff\xd8')
                        age=float(response.headers['X-Preview-Age-Ms'])
                        assert 0<=age<=1000
                        assert int(response.headers['X-Preview-Epoch'])==status['control_epoch']
                        previews[camera]={'bytes':len(image),'source_age_ms':age}
                        break
                time.sleep(.1)
            else:raise AssertionError(f'{camera} preview unavailable')
        report['previews']=previews
        assert stop_idle_backend(backend)
        thread.join(45)
        assert not thread.is_alive() and not errors,(errors,'launcher did not exit')
        report.update(task_id=selected.id,project_root=status['project_root'],recording_started=False,camera_opened=False,
                      backend_and_proxy_stopped=True)
        out=ROOT/f'.local/checks/collection_launch_{selected.id}.json'
        out.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report))
    finally:
        if thread.is_alive():
            try:stop_idle_backend(backend)
            except OSError:pass
            thread.join(45)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',default='tray_transfer')
    parser.add_argument('--browser-check',action='store_true',help='Also inspect the real headless browser DOM; no camera')
    args=parser.parse_args()
    run(args.task,args.browser_check)
