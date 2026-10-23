# Demonstrations

训练数据不提交到 GitHub，避免仓库包含约 1.1 GB 的视频和 Parquet 文件。

正式数据集：

- Hugging Face：<https://huggingface.co/datasets/shuyisong07/act_50eps>
- Episodes：50
- Frames：19,682
- FPS：30
- State：8 维（Panda 7 个关节位置 + 夹爪宽度）
- Action：7 维（底座坐标系下 XYZ、RX/RY/RZ、夹爪目标）
- Visual observations：3 路 640×480 RGB 视频

下载：

```bash
hf download shuyisong07/act_50eps \
  --repo-type dataset \
  --local-dir datasets/act_50eps
```

数据结构：

```text
datasets/act_50eps/
├── data/              # ACT 训练必需
├── meta/              # ACT 训练必需
├── videos/            # ACT 训练必需
└── episode_videos/    # 每个 episode 的三视角独立视频，人工检查用
```

## Pick and place variation（LeRobot v3）

[源数据集](https://huggingface.co/datasets/shuyisong07/pickandplace_variation/tree/main)包含 6 个单独录制的 LeRobot v3 数据目录。运行以下命令可下载并合并为一个可训练的数据集：

```bash
env -u ALL_PROXY /home/dex/miniconda3/envs/mujoco/bin/python \
  dataset/demonstrations/merge_pickandplace_variation.py
```

如果已下载源数据，可加 `--skip-download`。默认输出到 `datasets/pickandplace_variation_lerobot/`，包含 `data/`、`meta/`、`videos/`，以及用于追溯每条示范布局的 `provenance/episodes.jsonl`。总计 50 条成功示范、42,561 帧、30 FPS，9 维状态、7 维动作、3 路 RGB 视频。原始示范编号与合并后编号的映射也在该 JSONL 文件中。脚本需要 LeRobot 0.6.1。

CUDA ACT baseline 训练：

```bash
/home/dex/miniconda3/envs/mujoco/bin/python \
  training/ACT/train_act_variation_baseline.py \
  --dataset-root "$PWD/datasets/pickandplace_variation_lerobot" \
  --output-dir "$PWD/outputs/act_pickandplace_variation_cuda"
```

训练入口不再绑定固定数据目录；更换 `--dataset-root` 即可选择其他兼容的 LeRobot 数据集，省略 `--repo-id` 时自动使用 `local/<数据集目录名>`。可用 `--steps`、`--batch-size` 和 `--output-dir` 调整训练。加入手势辅助监督时改用 `training/ACT/train_act_variation_gesture.py`；再传 `--hand-delta-weight 0.1` 即启用手部位移监督。此处只整理成功示范；布局元数据不是训练帧特征。
