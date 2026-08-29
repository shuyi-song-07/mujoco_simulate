import time
import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = "mujoco_menagerie/franka_emika_panda/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Panda 初始姿态
data.qpos[:7] = [
    0.0,
    -0.5,
    0.0,
    -2.0,
    0.0,
    1.5,
    0.8
]

data.ctrl[:7] = data.qpos[:7]

mujoco.mj_forward(model, data)

# 找到 hand
hand_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "hand"
)

print("hand id =", hand_id)

jacp = np.zeros((3, model.nv))

STEP = 0.02


def move_hand(dx, dy, dz):

    # 当前 hand 的位置
    current_pos = data.xpos[hand_id].copy()

    # 希望移动的距离
    error = np.array([dx, dy, dz])

    # 计算 hand 的 Jacobian
    mujoco.mj_jacBody(
        model,
        data,
        jacp,
        None,
        hand_id
    )

    # Panda 7 个机械臂关节
    J = jacp[:, :7]

    # 末端位移 -> 关节角变化
    dq = np.linalg.pinv(J) @ error

    # 更新关节目标
    data.ctrl[:7] += dq

    # 限制关节范围
    for i in range(7):
        low, high = model.actuator_ctrlrange[i]
        data.ctrl[i] = np.clip(data.ctrl[i], low, high)

    print(
        "hand:",
        np.round(current_pos, 3),
        "→ ctrl:",
        np.round(data.ctrl[:7], 2)
    )


def key_callback(keycode):

    # ---------- 前后左右 ----------

    # ↑ 前
    if keycode == 265:
        move_hand(STEP, 0, 0)
        print("前")

    # ↓ 后
    elif keycode == 264:
        move_hand(-STEP, 0, 0)
        print("后")

    # ← 左
    elif keycode == 263:
        move_hand(0, STEP, 0)
        print("左")

    # → 右
    elif keycode == 262:
        move_hand(0, -STEP, 0)
        print("右")

    # ---------- 上下 ----------

    # I：上
    elif keycode == ord('I') or keycode == ord('i'):
        move_hand(0, 0, STEP)
        print("上")

    # K：下
    elif keycode == ord('K') or keycode == ord('k'):
        move_hand(0, 0, -STEP)
        print("下")

    # ---------- 夹爪 ----------

    # J：张开
    elif keycode == ord('J') or keycode == ord('j'):
        data.ctrl[7] = 255
        print("夹爪张开")

    # L：闭合
    elif keycode == ord('L') or keycode == ord('l'):
        data.ctrl[7] = 0
        print("夹爪闭合")
        
with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback
) as viewer:

    print("====== Panda 键盘控制 ======")
    print("↑ ↓ ← → : 前后左右")
    print("I / K     : 上 / 下")
    print("J / L     : 张开 / 闭合夹爪")
    print("===========================")
    while viewer.is_running():

        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(model.opt.timestep)