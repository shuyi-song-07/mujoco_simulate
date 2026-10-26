"""Generate isolated, explicitly scripted real-physics recordings (never success demos)."""
from pathlib import Path
import json
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tests/task3'))
from rate_fixture import packet, CALIBRATION
from simulation.mujoco.dual_arm.teleop_service import DualArmRecorder

def capture(task, seed, cameras=None):
    folder=ROOT/'.local/checks/pipeline'
    args=SimpleNamespace(task=task,seed=seed,dataset_root=None,no_previews=True,headless=True,port=0,
        diagnostics_root=folder/'diagnostics'/task,operator_id='scripted-engineering',input_kind='scripted_test',max_seconds=0)
    # A caller-selected isolated path does not alter the recorded scene/config.
    args.dataset_root=folder/'datasets'/task/f'smoke_{seed}_{time.time_ns()}'
    if cameras is None:
        rec=DualArmRecorder(args)
    else:
        from unittest.mock import patch
        from collection.config import recording_config
        def configured(task):
            config=recording_config(task)
            config['recording']['cameras']=list(cameras)
            return config
        with patch('collection.config.recording_config',side_effect=configured):
            rec=DualArmRecorder(args)
    results=[]
    try:
        rec.telemetry={'aperture_calibration':CALIBRATION,'calibrated':True}
        rec.start_recording()
        for i in range(420):
            began=time.monotonic()
            if i%3==0:
                # Explicit synthetic observations exercise the genuine hand guard and action recorder.
                modes=('XY_HOLD','XY_HOLD') if i<300 else ('Z_CLOSE','Z_CLOSE') if i<360 else ('Z_OPEN','Z_OPEN')
                p=packet(i,modes=modes)
                now=time.time()*1000
                p.update(session=rec.session,control_epoch=rec.control_epoch,sentAt=now)
                m=p['telemetry']['recognition_result']['measurement']
                m.update(source_session=rec.session,source_control_epoch=rec.control_epoch,
                         capture_epoch_ms=now,recognition_epoch_ms=now)
                if 25<i<70 or 255<i<290:
                    p['delta_task_m']={'left':[.001,0.,0.],'right':[.001,0.,0.]}
                rec.latest_observation={'measurement':m,'received_at_ms':now}
                rec.diagnostic('hand_measurement',measurement=m,schema_valid=True,received_at_ms=now)
                rec.handle(p)
                results.append(rec._execution_result)
            rec.controller.step(1/60)
            rec.tick+=1;rec.capture();rec.controller.consume_gripper_event()
            time.sleep(max(0,1/60-(time.monotonic()-began)))
        rec.save_recording('failure')
        return {'task_id':task,'dataset':str(rec.roots['failure']),'session_id':rec.session,
                'input_kind':'scripted_test','outcome':'failure','task_success_claimed':False,
                'accepted':sum(bool(r and r.get('accepted')) for r in results),'requests':len(results),
                'rejections':sorted({r.get('reason','unknown') for r in results if r and not r.get('accepted')})}
    finally:rec.close()

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',default='tray_transfer')
    p.add_argument('--sessions',type=int,default=2)
    p.add_argument('--cameras',nargs='+',choices=('top','front','side','oblique'))
    a=p.parse_args()
    if not 1<=a.sessions<=2:raise ValueError('Only 1-2 engineering sessions')
    result=[capture(a.task,731000+i,a.cameras) for i in range(a.sessions)]
    path=ROOT/'.local/checks/pipeline/capture_report.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
