import time
import mujoco
import mujoco.viewer

MODEL_PATH = "mujoco_menagerie/franka_emika_panda/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Panda 初始姿态
data.ctrl[:7] = [
    0.0,
    -0.5,
    0.0,
    -2.0,
    0.0,
    1.5,
    0.8
]

def key_callback(keycode):

    # A
    if keycode == ord('A') or keycode == ord('a'):
        data.ctrl[0] += 0.2
        print("左转，目标 =", data.ctrl[0])

    # D
    elif keycode == ord('D') or keycode == ord('d'):
        data.ctrl[0] -= 0.2
        print("右转，目标 =", data.ctrl[0])


with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback
) as viewer:

    print("A = 左转")
    print("D = 右转")

    while viewer.is_running():

        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(model.opt.timestep)