# MuJoCo Panda 手势遥操作、数据采集与 Single-Arm Pushing 项目说明

> 更新日期：2026-09-29
> 本文以当前工作目录中的实际代码为准，用于后续恢复项目、换电脑、向 mentor 交接以及继续开发 ACT baseline。

## 1. 项目目标

本项目使用电脑摄像头和 MediaPipe 识别操作者手势，通过网页将控制指令发送给 MuJoCo 中的 Franka Emika Panda 机械臂，同时把机器人的视觉观测、状态和动作保存成 LeRobot Dataset。

当前包含两个独立任务：

1. **Pick-and-place**：抓住红色方块，抬起并搬运到白色圆盘；
2. **Push-T**：夹爪闭合并固定在T形刚体半高处，把随机初始位置和角度的红色T形刚体推入白色T形目标区域。

两个任务共用 MediaPipe 网页、Panda 模型、运动控制、相机渲染和 LeRobot 数据管线，但采用不同的任务规则和启动入口，数据集不会混在一起。

## 2. 当前项目位置

```text
/Users/susilyeon/Desktop/git/mujoco_simulate_sync
```

项目结构：

```text
mujoco_simulate_sync/
├── simulation/
│   └── mujoco/
│       ├── record_mujoco_panda.py       # Pick-and-place 主程序及共享控制/录制能力
│       ├── record_mujoco_push_t.py       # Push-T 独立启动入口
│       ├── render_workers.py            # 三路训练相机、辅助视角和视频编码
│       └── assets/franka_emika_panda/
│           ├── panda.xml                # Franka Panda 模型
│           ├── scene.xml                # Pick-and-place 场景
│           └── push_t_scene.xml          # Push-T 专用场景
├── teleoperation/
│   └── mediapipe/
│       ├── index.html                   # 网页结构
│       ├── app.js                       # 摄像头、手势识别和任务模式控制
│       ├── interaction-state.js         # 平面移动方向映射
│       ├── interaction-state.test.js    # 网页控制测试
│       ├── server.js                    # localhost 网页服务器及 API 代理
│       └── models/gesture_recognizer.task
├── dataset/demonstrations/              # 数据说明及 action 分析工具
├── evaluation/                          # LeRobot 数据完整性检查
├── training/ACT/                        # ACT 的 MPS/CUDA 训练脚本
├── datasets/                            # 本地数据，默认不上传 GitHub
├── outputs/                             # 训练输出，默认不上传 GitHub
├── requirements.txt
└── README.md
```

## 3. 机器人和软件来源

仿真机器人是 **Franka Emika Panda**：7 个机械臂关节，加一个两指夹爪。模型来自 MuJoCo Menagerie 的 Panda 资源，项目在此基础上增加任务场景、手势遥操作、逆运动学、Z 锁定、多相机录制和 LeRobot 数据输出。

主要软件：

- Python 3.12；
- MuJoCo；
- MediaPipe Gesture Recognizer；
- LeRobot；
- PyAV/FFmpeg 视频编码；
- Flask 本地控制接口；
- Node.js 本地网页服务器。

Python 3.12 环境位于：

```text
/Users/susilyeon/Desktop/git/lerobot/.venv
```

## 4. 系统数据流

```text
Mac 摄像头
  ↓
Safari/浏览器 getUserMedia
  ↓
MediaPipe 识别手势与手腕位置
  ↓
JavaScript 计算移动方向
  ↓ HTTP/JSON
Node.js :8000 代理 → Flask :5001
  ↓
末端 XYZ 增量 + 姿态约束 + Z 锁定
  ↓
阻尼最小二乘逆运动学
  ↓
Panda 关节目标与夹爪目标
  ↓
MuJoCo 仿真
  ├── 主操作窗口
  ├── 网页侧视/正视辅助画面
  └── 三路 640×480 训练视频 + state + action
          ↓
     LeRobot Dataset
          ↓
      ACT baseline
```

所有摄像头和控制服务都在本机运行。摄像头画面由浏览器本地处理，不上传到外部服务器。

## 5. 首次安装

### 5.1 Python 依赖

```bash
cd /Users/susilyeon/Desktop/git/mujoco_simulate_sync
source ../lerobot/.venv/bin/activate
pip install -r requirements.txt
```

### 5.2 网页依赖

必须在 MediaPipe 网页自己的目录安装，不能只在上级目录安装：

```bash
cd /Users/susilyeon/Desktop/git/mujoco_simulate_sync/teleoperation/mediapipe
npm install
```

如果这里缺少 `node_modules`，网页虽然能显示，但 MediaPipe JavaScript 会返回 404，表现为摄像头按钮无效、MuJoCo 一直显示未连接。

## 6. 每次运行

需要两个终端窗口。

### 终端一：启动网页

```bash
cd /Users/susilyeon/Desktop/git/mujoco_simulate_sync/teleoperation/mediapipe
npm start
```

浏览器打开：

```text
http://localhost:8000
```

允许 localhost 使用摄像头。

### 终端二：选择一个任务

Pick-and-place：

```bash
cd /Users/susilyeon/Desktop/git/mujoco_simulate_sync
source ../lerobot/.venv/bin/activate
mjpython simulation/mujoco/record_mujoco_panda.py
```

Push-T：

```bash
cd /Users/susilyeon/Desktop/git/mujoco_simulate_sync
source ../lerobot/.venv/bin/activate
mjpython simulation/mujoco/record_mujoco_push_t.py
```

一次只运行一个任务。网页会从 `/api/health` 自动读取 `task_mode`，不需要在网页手动选择。

## 7. 通用录制操作

网页按钮和键盘快捷键：

| 操作 | 网页 | 键盘 |
|---|---|---|
| 开始一条 demonstration | 开始录制 | `S` |
| 成功并保存 | 成功并保存 | `N` |
| 失败并保存到独立数据集 | 失败并保存 | `R` |
| 结束整个录制进程 | 结束录制 | `Q` |

规则：

- 每次点击成功保存，当前 episode 写入数据集，然后随机重置场景；
- 失败并保存会把当前 episode 的三路视频、state、action 和初始位姿完整写入独立失败数据集，然后随机重置；
- 结束录制会 finalize 数据集；
- 未点击“成功并保存”或“失败并保存”的未完成 episode，在程序关闭时仍会丢弃；
- 不要把 pick-and-place 与 Push-T 录到同一个数据集目录。

## 8. Pick-and-place 任务规则

原有规则保持不变：

| 手势 | 作用 |
|---|---|
| `Open_Palm` | 松开夹爪；上下移动手控制末端升降 |
| `Closed_Fist` | 闭合夹爪；上下移动手控制末端升降 |
| `Victory` | 左右移动 |
| `Thumb_Up` | 前后移动 |
| `ILoveYou` | 斜向移动 |

任务阶段：

1. 接近并抓住方块；
2. 方块抬到约 `0.50 m` 的运输高度后锁定 Z；
3. 锁定期间切换俯视操作视角，只进行平面运输；
4. 方块进入圆盘中心附近并稳定后解锁 Z；
5. 下降、松开方块并保存成功 episode；
6. 如果仍夹着方块离开圆盘，则重新锁定 Z。

## 9. Push-T 任务规则

普通方块 pushing 任务已经移除。Push-T 是当前唯一的 pushing 任务，使用闭合夹爪作为固定推动面，并从 episode 开始就锁定推动高度。

| 手势 | Push-T 中的作用 |
|---|---|
| `Open_Palm` 或 `Closed_Fist` | 不改变夹爪，也不改变高度；夹爪始终闭合 |
| `Victory` | 左右移动 |
| `Thumb_Up` | 前后移动 |
| `ILoveYou` | 斜向移动 |

固定状态：

- 夹爪 actuator target 全程固定为 `0/255`，每帧 action 的最后一维也记录为 `0`；
- T 刚体底面位于桌面 `Z=0.400 m`，高度为 `30 mm`，中线为 `Z=0.415 m`；
- 闭合指尖接触垫中心固定在 `Z=0.415 m`；
- Panda `hand` 原点相对指尖接触中心高 `0.1084 m`，所以程序实际锁定的末端原点高度为 `Z=0.5234 m`；
- 启动、保存后重置或失败重录后，都自动恢复到该闭合夹爪和固定高度，不需要人工升降；
- Z 全程锁定，只允许 XY 平面运动。

物体、目标和成功判定：

- 红色T形刚体由同一 body 下的两个 box geom 组成，因此是一个不会散开的刚体；
- 总质量为 `0.1 kg`，桌面和刚体滑动摩擦参数为 `0.45`；
- T形平面外轮廓约为 `100×100 mm`，横杆为 `100×30 mm`，竖杆为 `70×30 mm`；
- 白色目标T在边界各增加 `5 mm` 容差，整体约为 `110×110 mm`；
- 目标是不可碰撞的纯视觉区域，不会阻挡物体；
- 物体初始 yaw 在 `-30°～30°` 随机，目标方向第一阶段固定；
- T形物体在目标中的二维覆盖率达到 `90%` 并稳定 `0.5 s` 后判定到达；
- 覆盖率跌到 `85%` 以下时退出成功状态，形成判定滞回；
- 仍由操作者点击“成功并保存”，不会自动保存。

目标比物体略大，但不是只比较中心距离。覆盖率会同时约束最终位置和方向：完全对齐时为 `100%`；在中心一致的测试中，约 `15°` 方向误差接近 `90%` 阈值。

## 10. Push-T 场景物理设置

Push-T 使用独立场景 `push_t_scene.xml`，不会修改 pick-and-place 场景。接触参数经过加硬处理，T 刚体底面与桌面顶面一致，目标T不参与碰撞。

## 11. Push-T 随机布局与可达范围

每次首次启动、成功保存或失败重录时，T 刚体和T目标都会重新随机生成。

```text
T 刚体：
X = 0.46 ～ 0.54 m
Y = -0.10 ～ 0.10 m

T 目标：
X = 0.57 ～ 0.66 m
Y = -0.14 ～ 0.14 m

刚体与目标中心最小距离：0.14 m
```

这是为固定底座 Panda 选择的保守平面工作区，给闭合夹爪从 T 刚体后方接近和推动留出空间。

当前第一阶段主要让目标位于 T 刚体前方区域，重点先验证稳定 planar pushing。后续若需要研究不同推动方向，可在验证逆运动学和碰撞安全后扩展目标分布。

## 12. 相机设计

训练数据包含三路固定第三人称 RGB：

```text
observation.images.overview
observation.images.camera_2
observation.images.camera_3
```

共同规格：

- `640×480`；
- `30 FPS`；
- 三个方位约相隔 `120°`；
- 全程与 state/action 同步保存。

网页侧视图和正视图仅用于操作者定位，不写入训练数据。每个成功 episode 还会在 `episode_videos/episode_xxxxxx/` 下生成三段独立 MP4，方便人工检查。

## 13. LeRobot 数据字段

每帧包含：

### `observation.state`：8维

```text
[joint1, joint2, joint3, joint4, joint5, joint6, joint7, finger_width_m]
```

### `action`：7维

```text
[
  target_x_base_m,
  target_y_base_m,
  target_z_base_m,
  target_rx_base_rad,
  target_ry_base_rad,
  target_rz_base_rad,
  gripper_target_0_255
]
```

action 的 XYZ 和 RPY 均以 Panda 底座 `link0` 坐标系表示。Pick-and-place 与 Push-T 保持相同的 state/action 维度，方便继续使用 LeRobot ACT；但两种任务的行为分布不同，应分别训练或明确采用多任务设计，不能无说明地混合。

## 14. 数据保存位置

默认在项目根目录生成时间戳目录：

```text
datasets/mujoco_panda_pick_YYYYMMDD_HHMMSS/
datasets/mujoco_panda_pick_failures_YYYYMMDD_HHMMSS/
datasets/mujoco_panda_push_t_YYYYMMDD_HHMMSS/
datasets/mujoco_panda_push_t_failures_YYYYMMDD_HHMMSS/
```

主要内容：

```text
data/              # state、action、索引等
meta/              # 数据集元信息、任务和 episode 信息
videos/            # LeRobot 正式视频
episode_videos/    # 每个 episode 的独立检查视频
```

`episode_videos/initial_positions.jsonl` 保存每条 episode 的物体和目标初始 XY 位置。Push-T 还会记录物体和目标的初始 yaw。

成功和失败数据集结构相同，但目录完全分离。manifest 中额外保存 `outcome: "success"` 或 `outcome: "failure"`，便于后续质量分析；训练 ACT baseline 时默认只使用成功目录，除非后续明确设计了使用失败轨迹的训练方法。

## 15. 数据验证

录制后运行：

```bash
cd /Users/susilyeon/Desktop/git/mujoco_simulate_sync
source ../lerobot/.venv/bin/activate
python evaluation/validate_multicam_dataset.py datasets/你的数据集目录
```

重点检查：

- episode 数和帧数一致；
- 三路视频帧数与数据帧数一致；
- 分辨率为 `640×480`；
- state shape 为 `[8]`；
- action shape 为 `[7]`；
- 没有 NaN/Inf；
- 时间戳连续；
- action 没有异常跳变。

验证器中的 `all_episodes_have_gripper_close_and_open=false` 对 Push-T 是正常结果，因为 Push-T 的夹爪全程闭合；该检查主要面向 pick-and-place。

## 16. ACT baseline

现有训练脚本：

```text
training/ACT/train_act_baseline_mps.sh
training/ACT/train_act_baseline_cuda.sh
```

Mac MPS 示例：

```bash
cd /Users/susilyeon/Desktop/git/mujoco_simulate_sync
DATASET_ROOT="$PWD/datasets/act_50eps" \
OUTPUT_DIR="$PWD/outputs/act_panda_baseline" \
PYTHON_BIN="$PWD/../lerobot/.venv/bin" \
zsh training/ACT/train_act_baseline_mps.sh
```

NVIDIA CUDA 电脑先做 smoke test：

```bash
DATASET_ROOT="$PWD/datasets/act_50eps" \
PYTHON_BIN="$PWD/../lerobot/.venv/bin" \
STEPS=1000 \
bash training/ACT/train_act_baseline_cuda.sh
```

确认能够读数据、计算 loss、保存 checkpoint 后，再把 `STEPS` 提高。当前 CUDA 脚本默认使用 ACT、ResNet18 ImageNet 初始化、chunk size 100 和 batch size 8。

注意：现有训练脚本默认指向上传过的 pick-and-place 数据集 `shuyisong07/act_50eps`。Push-T 正式录制完成后，应为 Push-T 建立独立 repo id 和训练输出目录，不能直接让脚本误读旧数据。

## 17. 控制稳定性保护

当前程序包含以下保护：

- 只处理队列中最新的移动指令，防止旧指令积压；
- 丢弃超过约 `250 ms` 的过期运动指令；
- 限制单步末端位移；
- 限制单步关节目标变化；
- 限制关节目标领先真实关节的位置差；
- 固定末端姿态，降低平移过程中机械臂乱扭；
- Z 锁定后持续纠正末端高度；
- 训练相机和辅助相机放在独立进程中，减少渲染阻塞控制循环。

如果机械臂再次出现乱甩，不应首先扩大步长。优先检查浏览器是否重复发送旧命令、相机预览是否过载、目标是否超出稳定工作区，以及关节目标是否领先真实状态过多。

## 18. 常见故障

### 网页可以打开，但摄像头按钮无效、MuJoCo 显示未连接

原因通常是网页目录缺少 Node 依赖，MediaPipe 模块返回 404。

```bash
cd /Users/susilyeon/Desktop/git/mujoco_simulate_sync/teleoperation/mediapipe
npm install
npm start
```

然后刷新 Safari 页面并允许 localhost 使用摄像头。

### 网页显示 MuJoCo 未连接

检查：

```bash
curl http://127.0.0.1:8000/api/health
```

正常会返回 `"ok": true` 和当前 `task_mode`。

### 修改 XML 后没有变化

MuJoCo 只在启动时载入场景。必须退出当前 MuJoCo，再重新执行对应的 `mjpython` 命令。

### `AVFFrameReceiver` / `AVFAudioReceiver` 重复类警告

这是 PyAV 与 Homebrew FFmpeg 同时加载时可能出现的 macOS 警告。当前代码已把视频依赖放入独立渲染进程并固定使用 PyAV backend。不要随意删除系统 FFmpeg 或虚拟环境文件。

### Push-T 一开始就锁定 Z

这是当前规定行为：程序启动时夹爪已经闭合，指尖接触中心已经位于 T 刚体中线 `Z=0.415 m`，因此 Z 从 episode 开始全程锁定。

### T 刚体推不动或陷入桌面

确认运行的是 `record_mujoco_push_t.py`，并在修改场景后重启 MuJoCo。Push-T 场景使用加硬接触参数，刚体底面应与桌面顶面一致。

## 19. 已验证内容

截至本文更新时间，已经完成：

- MediaPipe 网页 JavaScript 语法及控制映射测试；
- 三路视频、8维 state、7维 action 数据验证；
- 原 pick-and-place 两帧回归测试；
- Safari 摄像头和 MuJoCo 本地连接验证；
- Push-T XML、总质量、桌面接触和覆盖率判定验证；
- Push-T 三路视频、8维 state、7维 action smoke test；
- Push-T 闭合夹爪和半高接触姿态验证；
- 新增 Push-T 后再次完成原 pick-and-place 回归测试。

## 20. 当前边界与下一步

当前已实现的是 demonstration 采集系统，不等于模型已经能够自主完成 Push-T。

接下来的合理顺序：

1. 人工试录少量 Push-T episodes；
2. 检查推动高度、方向、摩擦、Z 锁定和成功判定；
3. 删除失败或异常轨迹；
4. 固定最终任务规则后正式采集更多 demonstrations；
5. 为 Push-T 建立独立 Hugging Face Dataset；
6. 修改 ACT 训练配置指向 Push-T 数据；
7. 训练 ACT baseline；
8. 编写 ACT→MuJoCo rollout，使模型接管仿真机械臂；
9. 统计成功率、最终位置误差和随机初始位置泛化能力；
10. 根据老师和 mentor 的 roadmap 再进行真机接口、安全限制和部署。

Push-T 会通过T形覆盖率同时约束位置和方向。当前不包含障碍物、不同摩擦/质量、双臂或自动真机迁移，这些属于后续研究扩展。

## 21. 修改原则

后续开发必须遵守：

1. 不用 pushing 修改破坏 pick-and-place 规则；
2. 新任务优先使用独立入口和独立场景；
3. 改控制规则前先写清楚手势、状态转换和解锁条件；
4. 数据字段一旦开始正式采集，不应中途改变维度或坐标定义；
5. 每次正式大量录制前，先做少量 smoke test 和数据验证；
6. 数据集、模型 checkpoint、虚拟环境和 `node_modules` 不提交 GitHub；
7. 真机部署前必须另外加入坐标标定、速度/工作空间限制、通信失效停止和急停保护。

---

这份文档描述的是 2026-09-27 的当前实现。后续修改任务规则、数据字段、随机范围、相机或 ACT 配置时，应同步更新本文档，避免代码与交接说明不一致。
