"""Isolated single-arm recorder/hand-guard check; never a human demonstration."""
from pathlib import Path
import json
import os
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests/task3'))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def capture(seconds=10, seed=731001, batch='frozen_v1'):
    from rate_fixture import packet, CALIBRATION
    from simulation.mujoco.dual_arm.teleop_service import DualArmRecorder
    folder = ROOT/'.local/checks/single_arm_push_t'
    args = SimpleNamespace(task='push_t', seed=seed,
        dataset_root=folder/'datasets'/batch/f'smoke_{time.time_ns()}',
        no_previews=True, headless=True, port=0, diagnostics_root=folder/'diagnostics',
        operator_id='scripted-engineering', input_kind='scripted_test', max_seconds=0)
    rec = DualArmRecorder(args)
    results=[]
    try:
        rec.telemetry={'aperture_calibration':CALIBRATION,'calibrated':True}
        rec.start_recording()
        start_sim=rec.controller.data.time
        for i in range(int(seconds*60)):
            began=time.monotonic()
            if i % 3 == 0:
                p=packet(i, modes=('XY_HOLD','XY_HOLD'))
                now=time.time()*1000
                p.update(session=rec.session,control_epoch=rec.control_epoch,sentAt=now,
                         interaction_modes=['XY_HOLD'],release_intent=[False],widths_m=[0.])
                p['delta_task_m']={'left':[.0004 if 100<i<160 or 350<i<410 else 0.,0.,0.]}
                m=p['telemetry']['recognition_result']['measurement']
                m['hands']=[h for h in m['hands'] if h.get('bound_side')=='left']
                m.update(source_session=rec.session,source_control_epoch=rec.control_epoch,
                         capture_epoch_ms=now,recognition_epoch_ms=now)
                rec.latest_observation={'measurement':m,'received_at_ms':now}
                rec.diagnostic('hand_measurement',measurement=m,schema_valid=True,received_at_ms=now)
                rec.handle(p);results.append(rec._execution_result)
            rec.controller.step(1/60);rec.tick+=1;rec.capture();rec.controller.consume_gripper_event()
            time.sleep(max(0,1/60-(time.monotonic()-began)))
        frames=rec.frames;span=rec.controller.data.time-start_sim
        rec.save_recording('failure')
        report={'dataset':str(rec.roots['failure']),'input_kind':'scripted_test',
            'frames':frames,'sim_span_s':span,'expected_intervals':round(span*30),
            'accepted':sum(bool(r and r.get('accepted')) for r in results),
            'rejections':sorted({str(r.get('reason')) for r in results if r and not r.get('accepted')}),
            'task_success_claimed':False}
        (folder/f'capture_report_{seed}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))
        return report
    finally:
        rec.close()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=float,default=10)
    parser.add_argument('--seed',type=int,default=731001)
    parser.add_argument('--sessions',type=int,choices=(1,2),default=1)
    parser.add_argument('--batch',default='frozen_v1')
    args=parser.parse_args()
    if not args.batch.isidentifier():parser.error('batch must be a simple directory name')
    reports=[capture(args.seconds,args.seed+i,args.batch) for i in range(args.sessions)]
    (ROOT/'.local/checks/single_arm_push_t/capture_report.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
