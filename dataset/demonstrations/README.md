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
