# Diffusion Policy 对照实验

三个脚本分别保存到独立目录：

| 脚本 | 训练损失 | 默认输出目录 |
| --- | --- | --- |
| `train_diffusion_variation_baseline.py` | Diffusion 动作损失 | `outputs/diffusion_variation_baseline` |
| `train_diffusion_variation_gesture.py` | 动作损失 + 手势分类损失 | `outputs/diffusion_variation_gesture` |
| `train_diffusion_variation_gesture_hand_delta.py` | 动作损失 + 手势分类损失 + 手腕位移回归损失 | `outputs/diffusion_variation_gesture_hand_delta` |

运行示例（在项目根目录）：

```bash
/home/dex/miniconda3/envs/mujoco/bin/python training/Diffusion/train_diffusion_variation_baseline.py --dataset-root /path/to/dataset --print-command
/home/dex/miniconda3/envs/mujoco/bin/python training/Diffusion/train_diffusion_variation_gesture.py --dataset-root /path/to/dataset --print-command
/home/dex/miniconda3/envs/mujoco/bin/python training/Diffusion/train_diffusion_variation_gesture_hand_delta.py --dataset-root /path/to/newly_recorded_dataset --print-command
```

三个入口都必须显式传入 `--dataset-root /path/to/dataset`，不再绑定仓库里的旧数据目录。`--repo-id` 可选，省略时自动使用 `local/<数据集目录名>`。去掉 `--print-command` 即开始训练；先用 `--smoke-test --output-dir /tmp/<unique-name>` 检查一个训练步。手部位移实验要求数据集包含对应字段。三组比较时，应使用同一数据集、相同随机种子和相同训练参数。

策略图像输入固定为 `observation.images.overview`，并与 9 维机器人状态共同使用；数据集中的其他相机视频保留用于检查，不参与训练。

三个入口默认将训练图像缩放为宽 640、高 480。`--image-size` 按“高 宽”接收尺寸，例如 `--image-size 480 640`；单个整数仍表示正方形，例如 `--image-size 128`。`--smoke-test` 使用 96×96 图像。

三个入口共用以下完整训练配置，辅助实验只增加各自的监督头和损失：

| 项目 | 默认配置 |
| --- | --- |
| 观察历史 | overview 相机的上一帧和当前帧，`n_obs_steps=2` |
| 动作序列 | `horizon=30`；每次执行当前起的 8 个动作后重新规划 |
| 网络 | ResNet18 + GroupNorm + SpatialSoftmax + FiLM 条件 1D U-Net |
| U-Net 通道 | `[256, 512, 1024]` |
| 优化器 | AdamW，LR `1e-4`，betas `[0.95, 0.999]`，weight decay `1e-6` |
| 学习率调度 | warmup 500 步 + cosine，最终 LR 为峰值的 10%（默认 `1e-5`） |
| 扩散 | 100 步训练加噪、epsilon 预测，10 步 DDIM 推理 |
| 动作损失 | 按有效动作坐标数归一化的噪声 MSE，忽略 episode 补齐动作 |
| 状态/动作归一化 | 仅用训练集拟合 min-max，验证集沿用训练集统计量 |
| 训练与保存 | 40000 步，batch size 4，每 1000 步验证并保存，开启模型权重 EMA |

输入输出沿用当前 Panda 数据定义：S9/A7，动作爪控制量为 0–255；本次配置调整不转换为双臂 S16/A14 或米制爪动作。数据为 30 FPS 时，每次执行 8 步约为 0.267 秒。LeRobot 的 `horizon=30` 包含与观察历史对齐的动作索引 `[-1, 0, ..., 28]`。

GroupNorm 使用从头训练的 ResNet18，不加载 BatchNorm 预训练权重。`--no-pretrained-backbone` 作为兼容参数保留。

长度 30 的序列在 U-Net 内临时复制尾部补齐到 32，网络输出裁回 30；数据集序列、扩散采样和损失仍使用原始长度 30。训练入口及 `evaluation/diffusion_variation_rollout.py` 共用 `diffusion_runtime.install_temporal_padding()`。如果使用其他 Python 入口加载这些 checkpoint，应先调用这个适配器，再加载配置和模型。底层 LeRobot 包不需要修改。

Linux 环境安装训练依赖时，在项目根目录执行 `python -m pip install -r requirements.txt`；其中包含 LeRobot 的 `training` 和 `diffusion` 扩展。`--smoke-test` 仍使用缩小后的网络、图像及动作序列，只检查一个训练步。

## 合并 Pick-and-place 数据集

将 10-22 采集的四份兼容数据合并为独立训练集（原始目录不修改）。合并前会校验 feature schema、episode 元数据、每集三路检查视频和 `hand_trajectories`；输出同时保留重新编号后的手轨迹与布局清单。

```bash
/home/dex/miniconda3/envs/mujoco/bin/python dataset/demonstrations/merge_mujoco_panda_pick_for_diffusion.py --dry-run
/home/dex/miniconda3/envs/mujoco/bin/python dataset/demonstrations/merge_mujoco_panda_pick_for_diffusion.py
```

合并结果默认写到 `datasets/mujoco_panda_pick_diffusion_20261022`，包括 50 个 episode。训练时通过 `--dataset-root` 显式选择它；三种模式共用同一 schema：

```bash
# Diffusion Policy baseline
/home/dex/miniconda3/envs/mujoco/bin/python training/Diffusion/train_diffusion_variation_baseline.py --dataset-root datasets/mujoco_panda_pick_diffusion_20261022 --print-command

# 动作 + 手势分类辅助损失
/home/dex/miniconda3/envs/mujoco/bin/python training/Diffusion/train_diffusion_variation_gesture.py --dataset-root datasets/mujoco_panda_pick_diffusion_20261022 --print-command

# 动作 + 手势分类 + 手腕位移辅助损失
/home/dex/miniconda3/envs/mujoco/bin/python training/Diffusion/train_diffusion_variation_gesture_hand_delta.py --dataset-root datasets/mujoco_panda_pick_diffusion_20261022 --print-command
```

去掉 `--print-command` 即开始完整训练；首次训练建议先加 `--smoke-test --output-dir /tmp/panda-diffusion-smoke` 验证环境。可通过 `--device`, `--steps`, `--batch-size` 和 `--output-dir` 覆盖训练配置。不同采集轮次的手势控制增益有变化，因此基线动作策略可直接混训；若使用手腕位移辅助监督，需注意位移到机械臂动作的映射可能跨轮次略有不同。

新数据集每帧记录三类人手及控制信息：

1. `observation.raw_wrist_uv`、`observation.raw_wrist_timestamp_ms`、`observation.raw_wrist_valid`：MediaPipe 原始归一化手腕坐标、浏览器捕获时间（Unix 毫秒）、可见有效标记。无效帧用零坐标与 `-1` 时间占位。
2. `observation.filtered_wrist_uv`、`observation.filtered_wrist_timestamp_ms`、`observation.filtered_wrist_valid`：重新入镜稳定后开始的滤波手腕轨迹。`observation.hand_delta_uv`、`observation.hand_delta_valid`、`observation.hand_motion_segment` 是由这条轨迹得到的逐帧相对位移及分段信息。小于 0.0015 的变化忽略，大于 0.10 的追踪跳变重新设原始锚点并保持滤波位置；没有跨丢手或跨重入的位移。位移辅助监督只使用 `hand_delta_uv`。当前滤波为死区和跳变剔除，没有额外的平滑滤波。
3. `observation.mapped_control_delta_xyz_m`、`observation.mapped_control_valid`、`observation.mapped_control_commands`：该录制帧期间后端接收的网页运动命令，在机器人基坐标系下的位移请求之和及命令数，单位米。它供检查人手到机器人映射，不参与辅助损失。`action` 另存实际下发的机械臂目标位姿与夹爪目标。

每集另有 `hand_trajectories/episode-XXXXXX.jsonl`，逐行保留每个实际收到的浏览器手腕采样，而不是只保留录制帧上的最新采样。每行同时包含原始轨迹、滤波轨迹、各自时间戳、重新入镜段号及接收时的帧序号；丢手时用 `null` 明确表示轨迹中断。

辅助头只读取图像和机器人状态编码。分类损失忽略手势码 0 和填充帧；位移损失仅使用 `hand_delta_valid=1` 的非填充帧。总损失为 `diffusion_loss + gesture_weight * gesture_loss + hand_delta_weight * hand_delta_loss`。位移目标先除以 `--hand-delta-scale`（默认 0.03），再计算 Smooth L1。两个权重可通过 `--gesture-weight` 和 `--hand-delta-weight` 调整。推理时只加载普通 Diffusion Policy 权重；两个辅助头另存以供检查。
