# Gesture-Controlled MuJoCo Panda + LeRobot ACT

这是一个使用 MediaPipe 手势遥操作 MuJoCo Panda、采集 LeRobot demonstrations，并训练 ACT baseline 的完整项目。

## 项目结构

```text
mujoco_simulate/
├── simulation/
│   └── mujoco/                  # Panda 场景、控制、逆运动学和录制主程序
├── teleoperation/
│   └── mediapipe/               # 摄像头网页、MediaPipe 手势识别和控制映射
├── dataset/
│   └── demonstrations/          # 数据集说明和 action 轨迹分析工具
├── training/
│   └── ACT/                     # LeRobot ACT baseline 训练脚本
├── evaluation/                  # 数据完整性与多相机验证工具
├── PROJECT_COMPLETE_GUIDE_CN.md # 完整中文项目说明
├── requirements.txt             # Python 依赖
└── README.md
```

训练数据不存入 GitHub。正式的 50 episodes 数据位于 Hugging Face：

<https://huggingface.co/datasets/shuyisong07/act_50eps>

## 1. 环境

推荐 Python 3.12。创建环境后安装依赖：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

网页依赖：

```bash
cd teleoperation/mediapipe
npm install
```

## 2. 启动网页

终端一：

```bash
cd teleoperation/mediapipe
npm start
```

浏览器打开 <http://localhost:8000>。

## 3. 启动 MuJoCo

终端二，在项目根目录运行：

```bash
source .venv/bin/activate
mjpython simulation/mujoco/record_mujoco_panda.py
```

网页显示“MuJoCo 已连接”后，允许摄像头权限即可操作。

录制控制：

- `S` / 开始录制：开始一条 demonstration；
- `N` / 成功并保存：保存 episode 并随机重置；
- `R` / 失败并重录：丢弃 episode 并随机重置；
- `Q` / 结束录制：完成并关闭数据集。

## 4. 下载数据

```bash
hf download shuyisong07/act_50eps \
  --repo-type dataset \
  --local-dir datasets/act_50eps
```

核心训练目录是 `data/`、`meta/` 和 `videos/`；`episode_videos/` 用于逐条人工检查。

## 5. 验证数据

```bash
python evaluation/validate_multicam_dataset.py datasets/act_50eps
```

## 6. 运行 ACT baseline

Mac MPS 示例：

```bash
DATASET_ROOT="$PWD/datasets/act_50eps" \
OUTPUT_DIR="$PWD/outputs/act_panda_baseline" \
zsh training/ACT/train_act_baseline_mps.sh
```

另一台 NVIDIA GPU 电脑应把训练设备改为 CUDA。训练前先用少量 steps 做 smoke test，再进行正式训练。

NVIDIA CUDA 电脑可直接使用：

```bash
DATASET_ROOT="$PWD/datasets/act_50eps" \
STEPS=1000 \
bash training/ACT/train_act_baseline_cuda.sh
```

确认 smoke test 正常后，将 `STEPS` 改为 `100000` 正式训练。

## 7. 详细说明

完整的手势规则、坐标系、Z 锁定、相机设计、数据字段、ACT 参数和故障排查见：

[PROJECT_COMPLETE_GUIDE_CN.md](PROJECT_COMPLETE_GUIDE_CN.md)

## 数据与大文件

以下内容不会上传到 GitHub：

- `.venv/`
- `node_modules/`
- `datasets/`
- `outputs/`
- checkpoints 与训练视频

它们分别通过本地安装、Hugging Face Dataset 或训练过程获得。
