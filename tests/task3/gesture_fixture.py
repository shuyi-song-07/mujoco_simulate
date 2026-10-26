"""Explicitly synthetic pixel skeletons for gesture-interface engineering tests."""
import copy

PROTOCOL = "gesture-grasp-release-v2"


def hand(shape="palm", side="left", dx=0., dy=0.):
    cx = 170 if side == "left" else 470
    pixels = [[cx, 330] for _ in range(21)]
    bases = {1:(cx-35,310),5:(cx-20,270),9:(cx,265),13:(cx+18,270),17:(cx+38,280)}
    for start, (x,y) in bases.items():
        folded = (shape == "fist" and start != 1) or (shape == "ily" and start in (9,13)) or (shape == "pinch" and start == 5)
        offsets = [(0,0),(0,-18),(12,-10),(4,3)] if folded else [(0,0),(0,-18),(0,-36),(0,-54)]
        if start == 1:
            offsets = [(0,0),(-12,-10),(-24,-20),(-36,-30)]
        for j,(u,v) in enumerate(offsets):
            pixels[start+j] = [x+u,y+v]
    return {"bound_side":side,"raw_label":side,"valid":True,"identity_score":.99,
            "gesture":{"palm":"Open_Palm","pinch":"None","fist":"Closed_Fist","ily":"ILoveYou"}.get(shape,"None"),
            "gesture_score":.9,"image_landmarks_21":[[x/640+dx,y/480+dy,0.] for x,y in pixels],
            "hand_center_landmarks_m_21":None}


def packet(i=0, shapes=("palm","palm"), modes=("HOLD","HOLD"), *, dx=0., dy=0.):
    stamp=100000+i*50
    m={"schema_version":"task3-hand-v1","measurement_id":f"gesture:{i}","source_session":"session",
       "source_control_epoch":0,"capture_epoch_ms":stamp,"capture_browser_monotonic_ms":i*50,
       "accepted_recognition":True,"width":640,"height":480,
       "hands":[hand(shape,side,dx,dy) for shape,side in zip(shapes,("left","right"))]}
    return {"command":"continuous_input","control_protocol":PROTOCOL,"interaction_modes":list(modes),
            "mode":"XY" if "XY_HOLD" in modes else "Z", "release_intent":[v=="RELEASE" for v in modes],
            "delta_task_m":{"left":[0.,0.,0.],"right":[0.,0.,0.]},"widths_m":[.08,.08],
            "measurement_id":m["measurement_id"],"sentAt":stamp,"eventId":f"gesture-event:{i}",
            "telemetry":{"calibrated":True,"aperture_calibration":{"phase":"READY","endpoints":{
                s:{"closed":.15,"open":3.0} for s in ("left","right")}},"recognition_result":{"measurement":m}}}


def clone(value):
    return copy.deepcopy(value)
