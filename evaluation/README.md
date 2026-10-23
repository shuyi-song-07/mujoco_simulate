# Evaluation

## Panda Diffusion Policy rollout

在项目根目录运行，默认读取 baseline 最后一个 checkpoint：

```bash
/home/dex/miniconda3/envs/mujoco/bin/python evaluation/diffusion_variation_rollout.py --viewer
```

默认运行 1 个 episode、最多 60 秒仿真时间。可用 `--checkpoint` 指定其他训练步数的 checkpoint，例如 `outputs/diffusion_variation_baseline/checkpoints/050000/pretrained_model`；用 `--episodes 50` 统计更多随机布局的成功率。默认按 checkpoint 的 `n_action_steps` 执行动作队列，可用 `--action-steps` 调整重规划频率。`--realtime` 让 viewer 尽量按 30 FPS 播放。机械臂每帧的位移和关节指令有限速，分别可通过 `--max-xyz-step-m`、`--max-joint-step-rad` 和 `--max-yaw-step-rad` 调整。

结果保存到时间戳目录中的 `config.json`、`rollouts.csv`、`rollouts.jsonl` 和 `summary.json`。成功判定沿用 ACT rollout：红色方块距目标中心不超过 5 cm、接触圆盘、处于圆盘表面高度、基本静止且夹爪打开并稳定保持设定时长。成功率是仿真任务指标，不等同于训练 loss。

`validate_multicam_dataset.py` 用于检查：

- episode 数与帧数；
- frame index 和 30 FPS 时间戳；
- action/state 维度和有限值；
- 三路视频分辨率和帧数同步；
- 夹爪开合是否完整；
- action 相邻帧跳变；
- 每条 episode 的最终画面。

运行：

```bash
python evaluation/validate_multicam_dataset.py datasets/act_50eps
```

## Pick and place variation ACT baseline rollout

在项目根目录运行，默认加载 `outputs/act_variation_baseline/checkpoints/last/pretrained_model`，按 50 个不同随机种子测试，每次最多运行 **120 秒仿真时间**（30 FPS、最多 3600 次策略动作）：

```bash
/home/dex/miniconda3/envs/mujoco/bin/python evaluation/act_variation_rollout.py
```

可用 `--checkpoint`、`--episodes`、`--max-seconds`、`--seed-base`、`--device`、`--output-dir` 修改设置；`--video-every 10` 每十次保存一段 overview 视频。默认不保存视频。每个 episode 结束时都会保存一张 overview 最终状态 PNG，默认目录为桌面上的 `<输出目录名>_final_states`，也可用 `--screenshot-dir` 指定；CSV/JSONL 会记录对应截图路径。结果写入时间戳目录中的 `config.json`、`rollouts.csv`、`rollouts.jsonl` 和 `summary.json`，每次 rollout 完成后立即刷新记录。成功条件为：红色刚体实际接触圆盘、由圆盘表面承托且速度足够低，并稳定保持 0.5 秒；不再限制刚体中心到圆心必须小于 5 cm，也不要求夹爪完全打开。到时仍未满足则记为 `timeout`。

Rollout 默认保留训练时的 100 帧 ACT 动作长度，并对每帧的末端目标位移、机械臂关节指令和独立夹爪旋转分别限速。可用 `--action-steps`、`--max-xyz-step-m`、`--max-joint-step-rad`、`--max-yaw-step-rad` 调整；`rollouts.csv` 同时记录每回合实际观测到的关节指令、关节位置和末端位移峰值。

要在 MuJoCo 窗口中看 50 次 rollout，运行 `python evaluation/act_variation_rollout.py --viewer`；加 `--realtime` 则尽量按真实时间的 30 FPS 播放，完整 50 次可能需要较长时间。关闭窗口后，脚本会保存已完成的回合和汇总。

脚本使用原始采集器的 MuJoCo 场景和布局采样；策略输入为 9 维机器人状态，并从 checkpoint 自动读取所需 RGB 相机（兼容仅 `overview` 和 `overview`/`camera_2`/`camera_3`），输出为底座坐标系下的 7 维末端目标与夹爪指令。
