# Diffusion Policy 对照实验

三个脚本分别保存到独立目录：

| 脚本 | 训练损失 | 默认输出目录 |
| --- | --- | --- |
| `train_diffusion_variation_baseline.py` | Diffusion 动作损失 | `outputs/diffusion_variation_baseline` |
| `train_diffusion_variation_gesture.py` | 动作损失 + 手势分类损失 | `outputs/diffusion_variation_gesture` |
| `train_diffusion_variation_gesture_hand_delta.py` | 动作损失 + 手势分类损失 + 手腕位移回归损失 | `outputs/diffusion_variation_gesture_hand_delta` |

运行示例（在项目根目录）：

```bash
/home/dex/miniconda3/envs/mujoco/bin/python training/Diffusion/train_diffusion_variation_baseline.py --print-command
/home/dex/miniconda3/envs/mujoco/bin/python training/Diffusion/train_diffusion_variation_gesture.py --print-command
/home/dex/miniconda3/envs/mujoco/bin/python training/Diffusion/train_diffusion_variation_gesture_hand_delta.py --dataset-root /path/to/newly_recorded_dataset --print-command
```

去掉 `--print-command` 即开始训练。先用 `--smoke-test --output-dir /tmp/<unique-name>` 检查一个训练步。现有的 `datasets/pickandplace_variation_lerobot` 没有手腕位移字段，可用于前两组；第三组需要用更新后的采集程序录制新数据。三组比较时，应使用同一份包含手腕字段的新数据集、相同随机种子和相同训练参数。

新数据集每帧记录三类人手及控制信息：

1. `observation.raw_wrist_uv`、`observation.raw_wrist_timestamp_ms`、`observation.raw_wrist_valid`：MediaPipe 原始归一化手腕坐标、浏览器捕获时间（Unix 毫秒）、可见有效标记。无效帧用零坐标与 `-1` 时间占位。
2. `observation.filtered_wrist_uv`、`observation.filtered_wrist_timestamp_ms`、`observation.filtered_wrist_valid`：重新入镜稳定后开始的滤波手腕轨迹。`observation.hand_delta_uv`、`observation.hand_delta_valid`、`observation.hand_motion_segment` 是由这条轨迹得到的逐帧相对位移及分段信息。小于 0.0015 的变化忽略，大于 0.10 的追踪跳变重新设原始锚点并保持滤波位置；没有跨丢手或跨重入的位移。位移辅助监督只使用 `hand_delta_uv`。当前滤波为死区和跳变剔除，没有额外的平滑滤波。
3. `observation.mapped_control_delta_xyz_m`、`observation.mapped_control_valid`、`observation.mapped_control_commands`：该录制帧期间后端接收的网页运动命令，在机器人基坐标系下的位移请求之和及命令数，单位米。它供检查人手到机器人映射，不参与辅助损失。`action` 另存实际下发的机械臂目标位姿与夹爪目标。

每集另有 `hand_trajectories/episode-XXXXXX.jsonl`，逐行保留每个实际收到的浏览器手腕采样，而不是只保留录制帧上的最新采样。每行同时包含原始轨迹、滤波轨迹、各自时间戳、重新入镜段号及接收时的帧序号；丢手时用 `null` 明确表示轨迹中断。

辅助头只读取图像和机器人状态编码。分类损失忽略手势码 0 和填充帧；位移损失仅使用 `hand_delta_valid=1` 的非填充帧。总损失为 `diffusion_loss + gesture_weight * gesture_loss + hand_delta_weight * hand_delta_loss`。位移目标先除以 `--hand-delta-scale`（默认 0.03），再计算 Smooth L1。两个权重可通过 `--gesture-weight` 和 `--hand-delta-weight` 调整。推理时只加载普通 Diffusion Policy 权重；两个辅助头另存以供检查。
