"""HTTP preview of the newly recorded engineering dataset, without human approval."""
from pathlib import Path
import io
import json
import sys
import threading
import urllib.request
import av
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dataset.task3.review_server import make_review_server

def run():
    pointer=ROOT/'.local/checks/pipeline/outputs/dataset/tray_transfer/CURATION_CURRENT.json'
    source=json.loads(pointer.read_text(encoding='utf-8'))['selected_version']
    server=make_review_server(source,port=0)
    thread=threading.Thread(target=server.serve_forever);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        with urllib.request.urlopen(base+'/api/manifest') as r:manifest=json.load(r)
        store=server.review_store
        payload={'episode_key':manifest['episodes'][0]['key'],'binding':store.document['binding'],
                 'exclude_intervals':[],'keep_intervals':[],'notes':''}
        request=urllib.request.Request(base+'/api/preview',json.dumps(payload).encode(),
            {'Content-Type':'application/json','Origin':base,'X-Review-Token':store.token})
        with urllib.request.urlopen(request,timeout=60) as r:plan=json.load(r)
        decoded={}
        for video in plan['videos']:
            with urllib.request.urlopen(base+video['url']) as r:encoded=r.read()
            with av.open(io.BytesIO(encoded)) as container:
                count=sum(1 for frame in container.decode(video=0))
            assert count==plan['kept_frames']
            decoded[video['camera']]=count
        report={'source':source,'episode':payload['episode_key'],'decoded':decoded,
                'kept_frames':plan['kept_frames'],'human_review_saved':False}
        (ROOT/'.local/checks/review_smoke.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report))
    finally:
        server.shutdown();thread.join();server.server_close()

if __name__=='__main__':run()
