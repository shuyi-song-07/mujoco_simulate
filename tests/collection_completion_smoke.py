"""Real contact DONE -> frozen episode -> late journal flush -> save/reset."""
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tests.platform_transport import run
from simulation.mujoco.dual_arm.teleop_service import DualArmRecorder


def main():
    folder=ROOT/'.local/checks/completion_boundary'/str(time.time_ns())
    folder.mkdir(parents=True)
    reports=[]
    for case,yaw_offset in enumerate((0.,.1)):
        seed=0  # Known reachable fixture; this is not random-layout coverage.
        args=SimpleNamespace(task='tray_to_platform',seed=seed,dataset_root=folder/'raw'/f'session{case}',
            no_previews=True,headless=True,port=0,diagnostics_root=folder/'logs',
            operator_id='scripted-engineering',input_kind='scripted_test',max_seconds=0)
        r=DualArmRecorder(args)
        try:
            r.start_recording()
            done_frame_counts = []
            def step():
                r.tick+=1;r.capture();r.controller.consume_gripper_event()
                if r.controller.get_status()['task_state']=='DONE' and not done_frame_counts:
                    done_frame_counts.append(r.frames)
            run(seed=seed,output=folder/f'physics{case}',controller=r.controller,on_step=step,
                placement_yaw_offset=yaw_offset)
            if yaw_offset:assert r.controller.get_status()['target_orientation_error_rad']>.06
            end=dict(r.episode_end);frames=r.frames
            assert r.controller.get_status()['paused']
            assert r.recording and r.capture_status['state']=='awaiting_save'
            assert r.controller.data.time>end['simulation_time']+.4
            assert abs(end['simulation_time']-end['success']['simulation_time']-.5)<.018
            assert 14 <= frames-done_frame_counts[0] <= 16
            r.capture(force=True);assert r.frames==frames
            # Recognition captured before DONE may finish/upload afterwards.
            rows=[{'sequence':i,'kind':'measurement','browser_epoch_ms':time.time()*1000,
                'measurement':{'measurement_id':f'late-{i}','source_session':r.session,
                    'source_control_epoch':r.control_epoch,'capture_epoch_ms':end['wall_time_ms']+offset,
                    'hands':[]}} for i,offset in ((1,-1),(2,1))]
            r.measurement_journal.append({'client_id':'engineering','rows':rows,
                'report':{'completed':2,'written':0,'pending':2}}, {'tick':r.tick})
            barrier={'version':'hand-save-boundary-v1','client_id':'engineering','last_sequence':2,
                'acknowledged_sequence':2,'pending':0,'dropped':0,'end_capture_epoch_ms':time.time()*1000}
            r.save_recording('success',barrier)
            assert not r.recording and r.control_epoch==1
            research=r.roots['success']/'research'
            terminal=json.loads((research/'episode_000000_snapshots/terminal.json').read_text(encoding='utf-8'))
            assert terminal['time']==end['simulation_time'] and terminal['episode_end']==end
            hands=research/'episode_000000_hands.jsonl'
            selected=[json.loads(l) for l in hands.read_text(encoding='utf-8').splitlines()]
            assert [x['measurement']['measurement_id'] for x in selected]==['late-1']
            meta=json.loads(hands.with_suffix('.manifest.json').read_text(encoding='utf-8'))
            assert meta['end_ms']==end['wall_time_ms']
            timeline=[json.loads(l) for l in (research/'episode_000000.jsonl').read_text(encoding='utf-8').splitlines()]
            endings=[x for x in timeline if x.get('kind')=='episode_capture_end']
            assert len(endings)==1 and endings[0]['boundary']==end
            assert not any(x.get('sim_time',0)>end['simulation_time']+1e-9 for x in timeline)
            # Saving advances scene and the next recording clears the old end.
            r.start_recording();assert r.episode_end is None and r.frames>0
            r.discard_recording('smoke_next_episode')
            reports.append({'seed':seed,'yaw_offset_rad':yaw_offset,'dataset':str(r.roots['success']),'frames':frames,
                'post_success_frames':frames-done_frame_counts[0],
                'episode_end':end,'late_pre_cutoff_measurement_retained':True,'post_cutoff_measurement_excluded':True})
        finally:r.close()
    report={'scripted_test':True,'human_camera_validated':False,'episodes':reports}
    (folder/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('VERIFIED',folder)


if __name__=='__main__':main()
