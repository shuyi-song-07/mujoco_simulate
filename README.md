# MuJoCo Robot Learning and Teleoperation

这是一个面向机器人操作研究的可扩展项目，覆盖从**仿真任务设计、视觉手势遥操作、demonstration 采集、数据校验，到策略训练与评估**的完整流程。

项目目前以 MuJoCo 中的 Franka Panda 单臂为基础，已经支持 pick-and-place 和 Push-T 数据采集，并提供 LeRobot ACT baseline 的训练入口。后续将逐步加入 dual-arm pushing、更多操作任务、不同机器人平台和多种 policy。

> 当前已经实现的功能与后续计划会在本文中分别标注。README 不绑定某一个训练数据集；数据集可根据实验需要从本地目录或 Hugging Face 单独获取。

## 项目目标

本项目希望提供一套模块化的机器人学习实验流程：

```text
任务与场景设计
      ↓
遥操作与 demonstrations 采集
      ↓
数据校验与格式转换
      ↓
Policy 训练
      ↓
MuJoCo 推理与量化评估
      ↓
真实机器人部署（后续）
```

每个任务、遥操作方式和训练策略保持独立，方便逐步扩展机器人、数据集和 policy。

## 当前进度

### 已实现

- MuJoCo Franka Panda 单臂仿真；
- 基于 MediaPipe 的网页手势遥操作；
- pick-and-place 任务；
- 固定末端工作高度的单臂 Push-T 任务；
- 任务初始位置随机化；
- 机器人状态、action 与多视角 RGB 图像同步采集；
- 成功 episode 与失败 episode 分开保存；
- LeRobot 兼容的数据组织与校验工具；
- ACT baseline 的 Apple Silicon（MPS）和 NVIDIA CUDA 训练脚本。

### 后续扩展

- dual-arm pushing；
- 更多 single-arm / dual-arm manipulation 任务；
- 更多机器人模型与遥操作设备；
- ACT 之外的 behavior cloning、Diffusion Policy、VLA 等策略；
- 统一的离线评估、MuJoCo rollout 和成功率统计；
- 仿真到真机迁移及真实机器人部署。

## 项目结构

```text
mujoco_simulate_sync/
├── simulation/
│   └── mujoco/                  # MuJoCo 场景、任务入口与渲染逻辑
├── teleoperation/
│   └── mediapipe/               # 网页端手势识别与遥操作界面
├── dataset/
│   └── demonstrations/          # 本地采集数据（默认不提交到 Git）
├── training/
│   └── ACT/                     # 当前 ACT baseline 训练入口
├── evaluation/                  # 数据校验与后续策略评估工具
├── PROJECT_COMPLETE_GUIDE_CN.md # 项目原理、规则与完整流程
├── requirements.txt
└── README.md
```

后续新增内容建议按模块放置：

```text
simulation/mujoco/<new_task>/     # 新任务或场景
training/<policy_name>/           # 新 policy 的配置与训练入口
evaluation/<task_or_policy>/      # 对应评估代码
```

如果暂时不调整现有目录，也应保证不同任务和 policy 使用独立、可识别的文件名与输出目录。

## 环境配置

建议使用 Python 3.12 的独立虚拟环境：

```bash
cd /path/to/mujoco_simulate_sync
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

训练代码依赖 LeRobot。可将本项目与 LeRobot 放在同一级目录，或通过环境变量指定 LeRobot 路径。

## 启动遥操作网页

```bash
cd teleoperation/mediapipe
npm install
npm start
```

浏览器打开：

```text
http://127.0.0.1:8000/
```

网页负责识别手势并发送控制信息；具体动作规则、锁高逻辑和任务状态机请以对应任务代码及项目完整说明为准。

## 运行当前任务

先进入项目根目录并激活虚拟环境：

```bash
cd /path/to/mujoco_simulate_sync
source .venv/bin/activate
```

### Pick-and-place

```bash
mjpython simulation/mujoco/record_mujoco_panda.py
```

单臂 Pick-and-place 每轮重采样红色目标方块、蓝/绿/黄干扰方块的尺寸、XY 位置与朝向，以及白色目标圆盘位置。采样会检查桌面间距、机械臂关节限位与抓取高度的可达性；操作时也限制末端工作空间和关节目标。Pointing_Up 以食指竖直向上为零角，使用手腕到食指尖的方向控制夹爪顺/逆时针旋转；机械臂关节锁定到识别出 ILoveYou。原有抓起后 ILoveYou 平移及俯视切换保持不变。

MuJoCo 的 Victory 左右映射已反转；ILoveYou 允许空夹爪斜向移动且不会误锁高度。辅助画面加入与窗口俯视切换使用相同相机参数的俯视图。指尖与方块的碰撞面使用较高摩擦，便于稳定夹取与搬运。

LeRobot 帧字段新增 `observation.gesture`、`observation.gesture_score`、`observation.hand_visible`。`observation.state` 包含七个机械臂关节、独立夹爪旋转角和夹爪开度，共 9 维。手势编码：`None=0`、`Closed_Fist=1`、`Open_Palm=2`、`Pointing_Up=3`、`Thumb_Down=4`、`Thumb_Up=5`、`Victory=6`、`ILoveYou=7`。摄像头断流超过 0.5 秒时写 `None`。每集 `episode_videos/initial_positions.jsonl` 同时记录随机 layout（每个方块的中心、半尺寸和 yaw）。

### Single-arm Push-T

```bash
mjpython simulation/mujoco/record_mujoco_push_t.py
```

运行前需要允许终端和浏览器访问摄像头。建议先确认网页显示“MuJoCo 已连接”，再开始遥操作或录制。

## Demonstration 采集

网页端提供以下录制操作：

- **开始录制**：开始当前 episode；
- **成功并保存**：将成功 episode 写入正式数据目录，并重置场景；
- **失败并保存**：将失败 episode 单独保存，便于排查与分析，并重置场景；
- **结束录制**：安全结束当前录制会话。

不同任务、不同机器人配置和不同 action 定义的数据，不应直接混入同一个训练数据集。建议每个数据集至少记录：

- 任务名称与版本；
- 机器人和关节配置；
- observation 字段、相机名称与图像尺寸；
- action 的语义、维度与坐标系；
- episode 边界、时间戳和成功标签；
- 场景随机化范围与任务成功条件。

数据集可保存在任意本地路径，或单独托管在 Hugging Face。大型数据、视频和模型权重不应直接提交到本代码仓库。

## 数据校验

将 `<dataset_path>` 替换为实际数据目录：

```bash
python evaluation/validate_multicam_dataset.py \
  --dataset-root <dataset_path> \
  --require-success
```

训练前至少需要检查：

- episode 数量和首尾索引是否正确；
- observation 与 action 的帧数是否同步；
- action 维度及数值范围是否符合当前机器人；
- 视频能否解码，且每个 episode 可单独查验；
- 是否存在 NaN、缺帧、冻结画面或异常关节跳变；
- 场景初始位置是否覆盖预先规定的随机范围。

## Policy 训练

项目不限定只能使用 ACT。不同 policy 应拥有独立的训练目录、配置、依赖说明和输出目录，并通过清晰的数据接口读取 observation、state 和 action。

### 当前提供：LeRobot ACT baseline

以下命令使用统一入口启动 ACT。数据集路径必须显式传入，可按机器选择 `cuda`、`mps` 或 `cpu`：

```bash
/home/dex/miniconda3/envs/mujoco/bin/python \
  training/ACT/train_act_variation_baseline.py \
  --dataset-root /absolute/path/to/dataset \
  --output-dir /absolute/path/to/outputs/act_run \
  --device cuda --steps 100000 --batch-size 8
```

加入手势监督时使用 `training/ACT/train_act_variation_gesture.py`；再传入 `--hand-delta-weight 0.1` 可同时启用手部位移监督。`--repo-id` 可选，省略时自动使用 `local/<数据集目录名>`。

### 新增其他 policy

建议每种 policy 使用如下结构：

```text
training/<policy_name>/
├── README.md            # 依赖、数据要求和运行命令
├── configs/             # 训练与评估配置
├── train.py             # 训练入口
└── evaluate.py          # 可选的离线评估入口
```

新增 policy 时应说明：

1. 支持的 observation 和 action schema；
2. 是否支持单臂、双臂及多相机输入；
3. 训练所需硬件与主要超参数；
4. checkpoint 保存与恢复方式；
5. 如何接入 MuJoCo 进行闭环推理。

## 新增任务规范

为了让 single-arm、dual-arm 和后续任务可以长期共存，新增任务时建议完成以下内容：

1. **场景定义**：机器人、物体、目标区域、碰撞和物理参数；
2. **控制接口**：明确 action 维度、坐标系、控制频率和安全限制；
3. **Observation schema**：相机、机器人状态和任务状态字段；
4. **任务状态机**：初始化、运行、成功、失败和重置逻辑；
5. **随机化规则**：物体和目标的合法生成范围；
6. **数据命名空间**：避免与其他任务的数据、视频和元数据混合；
7. **评估指标**：成功率、完成时间、轨迹长度、碰撞或任务特定指标；
8. **文档与测试**：提供启动命令、控制规则和最小可复现检查。

对于 dual-arm pushing，还需要额外明确双臂 observation/action 的排列顺序、同步控制方式、工作空间约束、机械臂之间的碰撞保护及任务协作逻辑。

## 文档

- [项目完整说明](PROJECT_COMPLETE_GUIDE_CN.md)
- [数据目录说明](dataset/demonstrations/README.md)
- [评估工具说明](evaluation/README.md)

## 仓库管理约定

以下内容默认不提交到 Git：

- demonstration 数据与录制视频；
- checkpoint、训练日志和评估输出；
- `.venv`、Conda 环境和缓存；
- `node_modules`；
- Hugging Face token、密钥及其他本机凭据。

GitHub 用于保存可复现的代码、配置和文档；数据集与模型产物由各实验单独管理。
