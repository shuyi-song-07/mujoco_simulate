# 手势遥操作 MuJoCo Panda、LeRobot 数据采集与 ACT Baseline 项目全解

> 本文档按照当前项目的实际代码与数据编写，目标是让项目成员能够理解、运行、讲解、复查并独立复现整个系统。

## 1. 项目概述

本项目构建了一个在 Mac 上运行的机器人学习数据采集系统。操作者通过电脑摄像头做手势，网页使用 MediaPipe 识别手势，Python 后端把手势转换为 Panda 机械臂末端运动，MuJoCo 执行物理仿真，同时将机器人状态、动作目标和三个固定相机的 RGB 图像同步保存为 LeRobot Dataset v3 数据。

系统当前完成的任务是：

1. Panda 机械臂接近并抓住桌面上的方块；
2. 将方块抬升到运输高度；
3. 锁定末端执行器 Z 高度，在水平面内搬运方块；
4. 到达白色圆盘中心附近后解除 Z 锁定；
5. 下降并松开夹爪，将方块放在圆盘上；
6. 将整段过程保存为一条 demonstration（一个 episode）。

它包含三个层次：

- **复现**：运行并理解 MediaPipe、MuJoCo、LeRobot 和 ACT 的开源能力；
- **任务适配**：把手势识别接到 MuJoCo Panda，并定义抓取放置任务规则；
- **二次开发**：增加 Z 锁定、视角切换、多相机同步、随机布局、独立 episode 视频和控制积压治理。

## 2. 完整技术路线

```text
Mac 摄像头
    ↓
浏览器 getUserMedia 获取视频
    ↓
MediaPipe Gesture Recognizer（CPU）
    ↓
手势名称 + 21 个手部关键点
    ↓
JavaScript 将手腕位移映射为控制方向
    ↓ HTTP/JSON
Python Flask 控制接口（127.0.0.1:5001）
    ↓
末端 XYZ 增量 + 固定姿态约束
    ↓
阻尼最小二乘逆运动学
    ↓
Panda 7 个关节目标 + 夹爪目标
    ↓
MuJoCo 物理仿真
    ├── 操作者主窗口与辅助视角
    └── 30 FPS 同步记录 state、action、三路 RGB
            ↓
      LeRobot Dataset v3
            ↓
      ACT baseline 训练与推理
```

## 3. 项目位置与目录结构

项目根目录：

```text
/Users/susilyeon/Desktop/git/lerobot
```

核心文件位于：

```text
examples/mujoco_panda/
├── PROJECT_COMPLETE_GUIDE_CN.md       # 本文档
├── README_CN.md                       # 简明运行说明
├── record_mujoco_panda.py             # 主控制、仿真、录制与本地 API
├── render_workers.py                  # 训练相机、辅助相机和视频编码进程
├── validate_multicam_dataset.py       # 数据集完整性检查
├── analyze_action_trajectory.py       # action 轨迹分析
├── train_act_baseline_mps.sh          # Mac MPS 上的 ACT baseline 配置
├── assets/franka_emika_panda/         # Panda 模型、场景、网格与 XML
└── web/
    ├── index.html                     # 页面结构
    ├── style.css                      # 页面外观和布局
    ├── app.js                         # 摄像头、MediaPipe、控制与录制按钮
    ├── interaction-state.js           # 平面手势到 XY 的映射
    ├── interaction-state.test.js      # 网页控制映射测试
    ├── server.js                      # localhost:8000 网页服务器和 API 代理
    ├── package.json                   # Node.js 依赖与命令
    └── models/gesture_recognizer.task # MediaPipe 手势模型
```

数据位于：

```text
datasets/
├── mujoco_panda_pick_20260908_220959/       # 正式 50 episodes
└── mujoco_panda_pick_act_20260903_53eps/    # 保留的旧 53 episodes
```

正式 50 条数据已经上传到：

```text
https://huggingface.co/datasets/shuyisong07/act_50eps
```

## 4. 环境与版本

当前集成项目使用独立的 Python 3.12 虚拟环境：

```text
Python 3.12.14
LeRobot 0.6.2
MuJoCo 3.12.0
NumPy 2.2.6
Node.js 22.22.3
```

虚拟环境位于：

```text
/Users/susilyeon/Desktop/git/lerobot/.venv
```

原来的 Python 3.11 和旧 MuJoCo `.venv` 没有被覆盖。采用多版本并存是为了满足 LeRobot 对较新 Python 的要求，同时保护旧项目环境。

macOS 上运行 MuJoCo GUI 使用 `mjpython`，而不是普通 `python`。PyAV 自带的 FFmpeg 与 Homebrew FFmpeg 同时载入时可能出现 `AVFFrameReceiver` 或 `AVFAudioReceiver` 重复类警告。当前实现将视频依赖限制在独立渲染进程，并固定使用 PyAV backend，以降低库冲突和崩溃风险。

## 5. 系统组成

### 5.1 网页前端

网页运行在 `http://localhost:8000`，职责包括：

- 请求并显示 Mac 摄像头；
- 加载 MediaPipe Gesture Recognizer；
- 绘制手部关键点；
- 显示手势名称和置信度；
- 根据手腕在屏幕中的位移计算移动方向；
- 将控制命令发送给 MuJoCo 后端；
- 显示 MuJoCo 连接状态；
- 显示侧视和正视辅助画面；
- 提供开始、保存、丢弃和结束录制按钮。

MediaPipe 使用：

```text
runningMode: VIDEO
numHands: 2
delegate: CPU
输入摄像头理想分辨率: 640×480
```

网页每 2 秒检查一次 MuJoCo 是否在线，并持续按需请求辅助画面。

### 5.2 本地 HTTP 接口

Python 后端监听：

```text
http://127.0.0.1:5001
```

接口包括：

| 接口 | 方法 | 作用 |
|---|---|---|
| `/control` | POST | 接收手势和录制命令 |
| `/health` | GET | 返回连接、录制、Z 锁定和高度状态 |
| `/side-preview` | GET | 返回侧视辅助 JPEG |
| `/front-preview` | GET | 返回正视辅助 JPEG |

网页的 `server.js` 把 `/api/...` 请求代理到 Python，因此浏览器只需要访问 8000 端口。

### 5.3 MuJoCo 主控制进程

主进程负责：

- 60 Hz 物理和 UI 控制循环；
- 处理网页命令与备用键盘输入；
- 逆运动学；
- Panda 关节与夹爪控制；
- Z 锁定和目标区域状态机；
- MuJoCo 主窗口视角切换；
- 按 30 FPS 产生录制快照。

### 5.4 训练数据渲染进程

训练渲染独立于主控制进程，负责：

- 按顺序接收每一帧 MuJoCo 状态快照；
- 渲染三台固定训练相机；
- 写入 LeRobotDataset；
- 编码训练视频；
- 为每个 episode 生成三个独立 MP4。

### 5.5 辅助视角渲染进程

辅助进程只服务网页操作，不进入训练数据。它交替渲染侧视和正视，保留最新快照，丢弃过期请求，避免渲染积压拖慢机械臂。

## 6. 手势与操作规则

### 6.1 当前有效规则

| 手势 | 操作 | 附加规则 |
|---|---|---|
| `Open_Palm` 张开手掌 | 松开夹爪；手上下移动时控制末端升降 | ILoveYou 后短时间内会屏蔽误识别出来的张掌 |
| `Closed_Fist` 握拳 | 闭合夹爪；手上下移动时控制末端升降 | 进入竖直移动时固定 XY，避免升降时横向漂移 |
| `Victory` | 左右移动 | 摄像头镜像方向已经校正；仅这个左右规则使用对应映射 |
| `Thumb_Up` | 前后移动 | 由手腕在画面中的上下位移决定前后方向 |
| `ILoveYou` | 斜向二维移动 | 只有夹爪处于闭合/搬运状态时有效，并强制锁定 Z |

`Pointing_Up` 控制功能已删除。`Thumb_Down` 可能仍能被 MediaPipe 显示为识别结果，但没有绑定机器人控制动作。

### 6.2 不是“看到一个手势就连续移动”

系统使用手腕位置变化控制方向和幅度。以平面移动为例：

```text
当前手腕位置 - 上一时刻手腕位置
→ 屏幕 dx/dy
→ 镜像修正
→ 死区过滤
→ 归一化
→ 机器人底座坐标系中的 XY 增量
```

关键网页参数：

```text
控制间隔                 40 ms
平面灵敏度               0.03
竖直死区                 0.015
平面死区                 0.09
夹爪命令重试间隔         300 ms
ILoveYou 张掌保护时间     650 ms
斜走识别短暂保持         180 ms
```

### 6.3 摄像头镜像

自拍摄像头画面通常是镜像的。代码先把屏幕位移转换为视觉方向：

```text
visualRight   = -screenDx / sensitivity
visualForward = -screenDy / sensitivity
```

再按手势映射：

```text
Victory  → (dx=0,             dy=visualRight)
Thumb_Up → (dx=visualForward, dy=0)
ILoveYou → (dx=visualForward, dy=-visualRight)
```

### 6.4 防止斜走时松开夹爪

`ILoveYou` 容易在手指形态变化时被短暂识别成 `Open_Palm`。系统采用两层保护：

1. ILoveYou 后 650 ms 内不接受张掌命令；
2. 必须出现一个明确的非张掌、非 ILoveYou 手势或手离开画面，才允许真正松开。

这保证斜走不会因为单帧误识别而掉落方块。

## 7. 坐标系、state 与 action

### 7.1 坐标原点

action 使用 Panda 底座 `link0` 坐标系：

- 原点：机器人底座；
- 平移：X、Y、Z，单位为米；
- 旋转：RX、RY、RZ，单位为弧度；
- 姿态表示：ZYX 约定下的 roll、pitch、yaw。

即使未来把机器人整体移动或旋转，代码也会先通过底座旋转矩阵进行坐标转换，而不是默认世界坐标永远等于底座坐标。

### 7.2 observation.state

每帧状态为 8 维：

```text
[joint1, joint2, joint3, joint4, joint5, joint6, joint7, finger_width_m]
```

前 7 项是 Panda 实际关节位置；最后一项是两根手指关节位置之和，即实际夹爪宽度。

### 7.3 action

每帧 action 为 7 维：

```text
[target_x, target_y, target_z, target_rx, target_ry, target_rz, gripper_target]
```

它不是 7 个关节目标，而是底座坐标系中的完整末端目标：

- XYZ：3 维位置；
- RX/RY/RZ：3 维姿态；
- gripper：1 维夹爪目标，范围 0–255。

因此它是 7 维。机器人内部仍会把末端目标通过逆运动学转换为 7 个关节控制目标。

## 8. 逆运动学与稳定控制

### 8.1 为什么需要逆运动学

用户想表达的是“夹爪向前、向左或向上移动”，但 Panda 执行器接收的是关节目标。逆运动学解决：

```text
希望的末端位移和姿态
→ 求解 7 个关节应该怎样改变
```

### 8.2 当前解法

系统使用位置 Jacobian 和旋转 Jacobian组成 6×7 任务 Jacobian，并用阻尼最小二乘求解：

```text
dq = Jᵀ (J Jᵀ + λ²I)⁻¹ Δtask
```

当前阻尼 `λ = 0.03`。阻尼用于降低接近奇异位形时关节突然大幅旋转的风险。

### 8.3 防止机械臂乱甩的措施

1. **从命令目标计算下一步**：不用滞后的真实关节状态重复累积同一个姿态误差；
2. **单次关节步长限制**：默认最大 `0.08 rad`；
3. **目标领先限制**：命令目标最多领先真实关节 `0.12 rad`；
4. **统一缩放 7 维关节增量**：保留笛卡尔运动方向，不逐关节破坏轨迹；
5. **平面单步限制**：默认最大 `0.015 m`；
6. **丢弃过期命令**：超过 `250 ms` 的移动命令不执行；
7. **只执行队列中最新移动命令**：防止卡顿后回放旧动作；
8. **网页请求合并**：同一时刻只发送一个请求，积压时只保留最新动作；
9. **渲染进程隔离**：视频编码不阻塞物理控制循环。

## 9. Z 锁定与视角状态机

### 9.1 为什么锁定 Z

老师要求搬运阶段减少自由度：方块到达固定高度后，运动变成二维平面控制。这样操作者只需处理前后左右，方块不会在运输过程中上下漂移。

### 9.2 进入锁定

默认运输高度：

```text
cube_z >= 0.50 m
```

抓住方块并达到该高度后：

- 记录一次末端的命令高度 `locked_z`；
- 将 `z_locked` 设为真；
- MuJoCo 操作窗口切到 top-down；
- 网页显示侧视和正视辅助视角；
- 所有平面命令的 `dz` 被替换为返回锁定平面的高度误差。

ILoveYou 平面移动还会显式发送 `lockZ=true`，避免抓取检测短暂抖动导致锁定失效。

### 9.3 锁定高度只能记录一次

一个重要 bug 曾经是：每个 ILoveYou 请求都用当前实际高度重新覆盖 `locked_z`。机械臂的实际位置存在微小滞后，于是锁定平面会一层层向下移动。

当前规则是：

```text
未锁定 → 锁定：记录一次命令高度
已经锁定 → 再次收到锁定：保持原值，不覆盖
解除锁定：清空 locked_z
```

主循环即使没有新的移动手势，也会持续调用平面保持逻辑。

### 9.4 解除锁定

只在方块中心满足以下条件时解除：

```text
方块与圆盘中心 XY 距离 ≤ 0.05 m
并连续稳定 ≥ 0.25 s
```

解除后：

- 恢复斜视操作相机；
- 隐藏网页辅助视角；
- 允许操作者下降并放置。

如果仍抓着方块并移出圆盘边界，系统会重新锁定 Z 并恢复俯视操作。

单纯张开夹爪不能绕过目标区域规则，也不能在圆盘之外提前解除 Z。

## 10. 夹爪、方块和圆盘规则

### 10.1 夹爪

- 最大张开宽度：8 cm；
- 张开控制目标：255；
- 闭合控制目标：0；
- 方块边长：6 cm（半尺寸 0.030 m）；
- 实际抓住后，碰撞约束使夹爪停在方块宽度附近。

抓取状态综合判断：

- 已发送闭合状态；
- 实际夹爪宽度位于合理区间；
- 夹爪与方块的空间距离足够近。

### 10.2 随机初始位置

每次成功保存或失败丢弃后，方块和圆盘都会重新随机生成：

```text
方块 X: 0.44–0.51 m
方块 Y: -0.07–0.07 m
圆盘 X: 0.54–0.62 m
圆盘 Y: -0.13–0.09 m
二者中心距离至少: 0.16 m
```

这个范围的目的：

- 增加数据多样性；
- 让模型学习物体位置变化，而不是记住固定轨迹；
- 保持在 Panda 固定夹爪姿态下较稳定的可达区域；
- 避免初始状态就相互重叠。

每条 episode 的实际初始位置记录在：

```text
episode_videos/initial_positions.jsonl
```

## 11. 相机系统

### 11.1 操作者相机

MuJoCo 主窗口根据任务阶段切换：

- 抓取、抬升、最终下降：斜视；
- 固定高度运输：top-down 俯视。

它用于人操作，不写入训练数据。

### 11.2 网页辅助相机

网页在运输阶段显示：

- 侧视：方位角 90°；
- 正视：方位角 180°；
- 距离：0.78；
- 仰角：-12°；
- 渲染尺寸：320×240；
- 每路约 8 FPS。

辅助相机仅用于精确定位，不进入 ACT 数据。

### 11.3 三台训练相机

训练数据始终使用固定相机，不随操作窗口切换：

| 数据字段 | 方位角 | 分辨率 | 帧率 |
|---|---:|---:|---:|
| `observation.images.overview` | 145° | 640×480 | 30 FPS |
| `observation.images.camera_2` | 25° | 640×480 | 30 FPS |
| `observation.images.camera_3` | 265° | 640×480 | 30 FPS |

三台相机围绕任务区域提供不同第三人称观察。操作视角放大、切换 top-down 或网页辅助视角变化都不会改变已经定义的训练观测。

## 12. demonstration 与 episode 生命周期

一条 demonstration 等于一个 episode：

```text
开始录制
→ 完成一次抓取放置
→ 成功保存或失败丢弃
```

操作方式：

| 网页按钮 | 键盘 | 作用 |
|---|---|---|
| 开始录制 | S | 开始当前 episode |
| 成功并保存 | N | 保存当前 episode，并随机重置 |
| 失败并重录 | R | 丢弃当前 episode，并随机重置 |
| 结束录制 | Q | 完成整个数据集并退出 |

必须对每一次成功 demonstration 单独按“成功并保存”。不能连续完成多次任务后才一起保存，因为 episode 的边界、重置和初始位置记录都在保存时确定。

如果只是想随机重置而不保存，使用“失败并重录/丢弃”即可。未开始录制时它也会重置布局。

## 13. LeRobot 数据集结构

正式数据集包含：

```text
dataset_root/
├── data/                  # Parquet：state、action、时间戳和索引
├── meta/                  # info、stats、tasks 和 episode 元数据
├── videos/                # LeRobot/ACT 正式读取的三路视频
├── episode_videos/        # 每个 episode 单独三个 MP4，供人工检查
└── analysis/              # 本地验证报告；不参与训练
```

### 13.1 训练必需目录

```text
data + meta + videos
```

### 13.2 老师要求的独立视频

为了让每条数据方便查验，额外保存：

```text
episode_videos/
├── initial_positions.jsonl
├── episode_000000/
│   ├── overview.mp4
│   ├── camera_2.mp4
│   └── camera_3.mp4
├── episode_000001/
│   └── ...
└── episode_000049/
    └── ...
```

这满足“每个 eps 对应一个单独视频”的要求，而且每个 eps 实际有三视角独立视频。它们用于人工查验，不会被 ACT 重复读取。

### 13.3 analysis

`analysis/` 由本地验证脚本生成，包括：

- JSON 总报告；
- 每条 episode 的 action 检查表；
- episode 最终画面拼图。

它不参与训练。Hugging Face 的 `act_50eps` 中已删除该目录，本地可以保留用于复查。

## 14. 正式 50 条数据的检查结果

数据集：

```text
mujoco_panda_pick_20260908_220959
```

检查结果：

```text
episodes: 50
frames: 19,682
fps: 30
action shape: [7]
state shape: [8]
训练相机: 3
每路分辨率: 640×480
所有视频帧数与数据帧数一致: 是
所有时间戳与帧序号连续: 是
所有数值有限，无 NaN/Inf: 是
所有 episode 均含夹爪闭合和松开: 是
结束画面均显示方块位于圆盘上: 是
```

这说明数据结构符合 ACT 输入要求，但“格式有效”不等于保证模型一定达到高成功率。最终效果还取决于数据覆盖范围、动作一致性、训练步数、模型设置和评估条件。

## 15. 数据验证方法

运行：

```bash
cd /Users/susilyeon/Desktop/git/lerobot
source .venv/bin/activate
python examples/mujoco_panda/validate_multicam_dataset.py \
  datasets/mujoco_panda_pick_20260908_220959
```

验证脚本检查：

- metadata 的 episode 与 frame 数；
- 每个 episode 的 `frame_index`；
- 30 FPS 时间戳；
- action 7 维和 state 8 维；
- NaN 与 Inf；
- XYZ/RPY 相邻帧跳变；
- 夹爪开合是否完整；
- 三路训练视频尺寸和总帧数；
- 每条 episode 的最终画面。

## 16. 启动完整系统

### 16.1 终端一：网页

首次安装或依赖变化时：

```bash
cd /Users/susilyeon/Desktop/git/lerobot/examples/mujoco_panda/web
npm install
npm start
```

以后只需：

```bash
cd /Users/susilyeon/Desktop/git/lerobot/examples/mujoco_panda/web
npm start
```

浏览器打开：

```text
http://localhost:8000
```

### 16.2 终端二：MuJoCo 与录制

```bash
cd /Users/susilyeon/Desktop/git/lerobot
source .venv/bin/activate
mjpython examples/mujoco_panda/record_mujoco_panda.py
```

### 16.3 网页操作

1. 等待“MuJoCo 已连接”；
2. 点击“启动摄像头”；
3. 允许浏览器摄像头权限；
4. 先测试动作，再开始录制；
5. 每次任务完成按“成功并保存”；
6. 失败按“失败并重录”；
7. 全部完成按“结束录制”。

每次运行都会在 `datasets/` 下创建新的时间戳目录，不覆盖旧数据。

## 17. ACT baseline

### 17.1 ACT 学习什么

ACT 的训练关系是：

```text
当前三路图像 + 当前机器人状态
→ 预测未来一段 7 维 action
```

ACT 不是简单逐帧预测下一步，而是一次预测一个 action chunk，以降低长任务中的误差累积。

### 17.2 baseline 的含义

复现 ACT baseline 指：

1. 使用 LeRobot 官方 ACT 实现；
2. 保持论文/官方模型结构与主要参数；
3. 把自己的 LeRobot 数据接入；
4. 跑通训练并得到 checkpoint；
5. 将 checkpoint 接回 MuJoCo；
6. 让模型替代手势自主控制；
7. 在随机初始位置上统计成功率。

不要求从头编写 ResNet 或 Transformer，也不要求从零进行大规模视觉预训练。

### 17.3 当前 baseline 配置

`train_act_baseline_mps.sh` 当前包含：

```text
policy: ACT
device: Apple MPS
vision backbone: ResNet18
backbone weights: ImageNet1K V1
chunk size: 100
action steps: 100
model dimension: 512
attention heads: 8
feedforward dimension: 3200
encoder layers: 4
decoder layers: 1
VAE: enabled
latent dimension: 32
KL weight: 10
batch size: 8
training steps: 100,000
checkpoint interval: 10,000
```

### 17.4 预训练与 ResNet18 权重

老师说“不需要预训练”通常表示不需要自行收集海量通用图像并从零训练视觉骨干。`ResNet18_Weights.IMAGENET1K_V1` 是已经在 ImageNet 上训练好的通用图像特征参数，ACT 使用它作为视觉起点，然后用本项目 demonstrations 学习机器人任务。

使用已有 ResNet18 权重不等于自己进行大规模预训练。

### 17.5 本机算力

当前 Mac 没有 NVIDIA CUDA GPU，但不是完全没有算力：

- 可以使用 CPU/MPS 检查数据加载；
- 可以运行少量 steps 验证训练链路；
- 可以调试模型配置和推理接口；
- 100,000 steps 的正式训练速度和稳定性不如实验室 NVIDIA GPU。

因此推荐先在 Mac 上完成 smoke test，再将同一数据和配置交给实验室 GPU 正式训练。

## 18. 如何独立复现和学习

不要从头重写 MuJoCo、MediaPipe、LeRobot 或 ACT。科研中的“复现”通常是阅读论文和开源代码，独立完成环境配置、运行核心流程、验证结果、解释原理，并在此基础上做任务适配。

建议学习顺序：

1. **Python 基础**：变量、判断、循环、函数、类、NumPy、JSON；
2. **MuJoCo 基础**：模型、data、qpos、ctrl、mj_step、相机；
3. **机器人基础**：坐标系、正/逆运动学、Jacobian、关节约束；
4. **网页基础**：HTML、JavaScript、getUserMedia、async/await、fetch；
5. **MediaPipe**：视频推理、手部关键点、手势类别和置信度；
6. **数据采集**：observation、action、episode、同步与帧率；
7. **LeRobot Dataset**：data/meta/videos 和数据读取；
8. **ACT**：视觉骨干、Transformer、action chunk、训练和推理。

可在单独的练习目录中按最小系统重新实现：

```text
摄像头页面
→ 显示手势名称
→ 把手势映射为文字指令
→ Python 接收 HTTP 指令
→ 键盘控制 MuJoCo
→ 手势替换键盘
→ 保存一条简单 episode
→ 转成 LeRobot 数据
→ 小规模运行 ACT
```

当前完整项目应保留为参考和最终成果，不要为了学习而破坏它。

## 19. 常见问题与排查

### 19.1 网页打不开

- 检查 `npm start` 是否仍在运行；
- 确认访问 `http://localhost:8000`；
- 8000 端口被占用时先关闭旧网页服务。

### 19.2 网页显示 MuJoCo 未连接

- 确认第二个终端已经运行 `mjpython ...`；
- 后端应监听 `127.0.0.1:5001`；
- 一个端口只能有一个 MuJoCo 录制进程。

### 19.3 摄像头点不开

- 在 Safari 网站设置中允许 localhost 使用摄像头；
- 关闭会议、拍照等占用摄像头的软件；
- 刷新网页后重新点击；
- 确保通过 localhost 打开，而不是直接双击 HTML 文件。

### 19.4 机械臂卡顿或延迟后乱动

- 不要同时运行多个 recorder；
- 检查辅助视角刷新是否正常；
- 不要提高辅助相机分辨率或帧率；
- 旧动作应由前端合并和后端 250 ms 过期策略丢弃；
- 若只在第二次 episode 变慢，检查旧视频编码任务是否正常结束。

### 19.5 斜走时向下掉

检查 `/health` 中：

```text
z_locked
hand_z_m
locked_z_m
```

ILoveYou 搬运时 `z_locked` 应为真，且 `locked_z_m` 不应随命令不断下降。

### 19.6 保存后没有重置

保存是同步收尾过程。系统会等 LeRobot 和独立 MP4 写完后才重置，避免数据损坏。视频较长时可能需要短暂等待。

### 19.7 无效 demonstration

- 当前 episode 失败：使用“失败并重录”，不要保存；
- 已经保存：先用验证脚本和视频确认，再按 episode 索引删除；
- 不要仅凭时长判断有效性，应同时检查最终画面、动作连续性和夹爪开合。

## 20. 设计边界与注意事项

1. 网页侧视和正视只用于操作，不是训练相机；
2. MuJoCo 主窗口切换视角不会改变训练视频；
3. `episode_videos` 不参与 ACT 训练，但满足人工检查要求；
4. action 当前是末端目标 7 维，不是关节目标 8 维；
5. state 是实际关节状态 8 维；
6. 数据结构有效不代表策略一定泛化成功；
7. 模型评估必须在未见过的随机布局上进行；
8. 修改任何手势映射前应运行网页测试并逐项回归原有规则；
9. 修改相机时必须区分操作相机、辅助相机和训练相机；
10. 删除数据前先确认 episode 数、路径和是否已上传。

## 21. 当前项目状态

### 已完成

- Python 3.12 独立 LeRobot 环境；
- MuJoCo Panda 抓取放置场景；
- 摄像头与 MediaPipe 手势识别网页；
- 网页到 Python 的实时控制；
- 末端位置与姿态控制；
- Z 锁定和 top-down 任务状态机；
- 侧视、正视辅助定位；
- 三路固定训练相机；
- 随机物体与目标位置；
- LeRobot Dataset v3 同步录制；
- 每 episode 独立三视角 MP4；
- 50 条正式 demonstrations 的完整校验；
- Hugging Face 公开上传；
- ACT baseline 训练配置。

### 尚待完成

- 在本机用少量 steps 完成 ACT smoke test；
- 在合适 GPU 上完成正式训练；
- 保存并加载 ACT checkpoint；
- 编写/完善 MuJoCo 闭环推理接口；
- 在随机初始位置上重复评估；
- 统计成功率、失败类型和泛化结果；
- 整理最终实验报告与演示视频。

## 22. 一句话总结

本项目将基于 MediaPipe 的视觉手势交互 Demo 扩展成了一个完整的 MuJoCo Panda 遥操作与 LeRobot 数据采集系统：操作者通过手势完成具有随机初始位置的抓取放置任务，系统使用稳定的逆运动学和阶段性 Z 锁定控制机械臂，以三个固定视角同步记录 observation，以底座坐标系下的 7 维末端目标记录 action，并生成可用于 LeRobot ACT baseline 的标准数据集以及便于人工检查的逐 episode 视频。
