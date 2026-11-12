# 单臂 ACT 与 GJB 对齐说明

更新：2026-11-12。参考 `papera_gjb` 的 `782e270`，入口仍是
`training/ACT/train_act_variation_baseline.py`。本次只对齐 ACT baseline；
没有修改采集、手势、物理参数、已有数据或历史模型。

## 已对齐的部分

| 项目 | 当前单臂 baseline |
| --- | --- |
| 模型实现 | 直接复用 GJB 固定的原作者 ACT/DETRVAE 核心；来源和许可证在 `official/` |
| Backbone | ResNet18 / FrozenBN，默认 ImageNet V1 初始化；可以指定本地权重 |
| Transformer | hidden512、8 heads、FF3200、encoder4、decoder7、posterior4、latent32、dropout0.1 |
| decoder 的特殊行为 | 保留上游取第一个 intermediate 输出的行为；配置7不代表七层依次共同决定最终动作 |
| 动作块 | K30，输入当前一帧 RGB 和当前机器人状态 |
| 训练损失 | 无效动作误差置零，再对完整 B×K×A 取均值；加 10×KL |
| 优化 | AdamW，主干/其他参数 LR 都为1e-5，WD1e-4；固定学习率、无 warmup、无梯度裁剪、无 EMA |
| 默认训练参数 | 60,000 次更新、batch16、seed20261016、FP32、workers0、CPU threads8 |
| 归一化 | 只拟合训练部分的 z-score；std 下限0.01；动作 RPY 按训练角度参考处理周期 |
| 验证与 best | z=0，不传未来动作给模型；按全部有效动作坐标的归一化 L1 选 best |
| 保存 | 默认每1000步保存 last，每5000步及训练终点额外留候选；保存模型、优化器、调度器、采样器、随机数和 loader generator |
| 恢复 | 记录已经消费的批次，避免把预取当作已训练；检查数据指纹、split、统计和源码；中断后只恢复完整 checkpoint |
| 来源 | `run.json` 保存配置、输入文件哈希、统计、split、代码哈希和运行环境；`source_snapshot/` 保存训练源码 |

训练损失的分母与以前的 LeRobot valid-mean 不同。特别是 episode 尾部
padding 比较多时，**新旧训练 loss 数值不能直接比较**。
`metrics.jsonl` 分别记录训练和验证；验证 L1 也不等于任务成功率。

## 必须保留的单臂差异

| 内容 | 单臂适配 |
| --- | --- |
| 状态 | S9：7关节、额外 gripper yaw、夹爪宽度；GJB 双臂是S16 |
| 动作 | A7：底座坐标 XYZ/RPY 与原有夹爪控制目标；GJB 是双臂A14、夹爪米制宽度 |
| 图像 | 只输入 `observation.images.overview`；已有另外两个录制视角不参与训练，视频读取也只解码 overview |
| 数据格式 | 继续读取现有 LeRobot 数据，不强制转换成 GJB 的双臂 Safe Stitch Pack |
| 时间对齐 | 保留现有“记录行对应的控制目标”，不擅自平移一帧；不声称等同于 GJB 的 next-control-tick 证书 |
| 验证划分 | 默认最后10%整 episode 留出；可传固定 split 文件。旧数据默认不具备 GJB 的整采集会话划分保证 |
| 序列边界 | 动作块不能跨 episode；没有生成不存在的 Safe Stitch/中途断点证书 |

这次统一的是模型和训练协议，不能消除两个任务在数据、动作语义、任务难度和
成功判据上的差异。需要严格按会话留出时，应先根据采集来源整理 split，
不能仅把文件中的 `unit` 改成 session 就当作证明。

## 命令

在项目根目录激活自己的训练环境后执行。下面的 `python` 应指向该环境。
替换数据集与输出路径；新训练使用不存在的输出目录，旧权重不会被覆盖。

```bash
python training/ACT/train_act_variation_baseline.py \
  --dataset-root /absolute/path/to/dataset \
  --output-dir /absolute/path/to/outputs/act_gjb_aligned_run \
  --device cuda
```

本地已有 ImageNet V1 权重时加 `--backbone-weights /absolute/path/resnet18-f37072fd.pth`，
否则首次训练由 torchvision 下载官方权重。`--no-pretrained-backbone` 明确关闭初始化，
这就与默认 GJB 配置不同，需作为另一实验记录。Mac 可用 `--device mps` 或
`--device cpu` 做检查，本次没有在本机启动正式长训练或验证 CUDA 吞吐。

先查看配置，不训练：

```bash
python training/ACT/train_act_variation_baseline.py \
  --dataset-root /absolute/path/to/dataset --print-command
```

工程短测：`--smoke-test --device cpu`。它缩小网络并只更新一次，产物标记
`engineering_smoke`，不能冒充正式 baseline 或真人任务成功证据。

恢复中断的原训练：

```bash
python training/ACT/train_act_variation_baseline.py \
  --dataset-root /absolute/path/to/dataset \
  --resume /absolute/path/to/run/checkpoints/last.pt --device cuda
```

恢复会使用 checkpoint 冻结的训练配置，不能用命令行新参数偷偷改优化器或预算。
需要延长已经完成的预算，使用 `--continue-from .../last.pt --steps 100000`
并给一个新的 `--output-dir`；沿用原数据、划分、统计、采样状态和固定学习率。
如果原来用了自定义 `--split-file`，恢复/续训也要传入同一文件。

自定义 split JSON 示例（episode 编号必须完整覆盖实际数据且不重叠）：

```json
{"train":[0,1,2,3],"val":[4],"unit":"explicit_episode_membership"}
```

## 权重与 rollout

```text
run/
├── run.json
├── source_snapshot/
├── metrics.jsonl
├── completion.json
└── checkpoints/
    ├── best.pt
    ├── last.pt
    └── step_005000.pt …
```

新模型标记为 `single-arm-gjb-act-v1`，不再是 LeRobot 的
`pretrained_model/model.safetensors`。不能用旧 LeRobot 权重或双臂 GJB
权重续训。`best.pt` 用来推理选型，恢复训练使用 `last.pt`。

已有 rollout 入口新增读取适配，控制执行器和成功判据不变：

```bash
python evaluation/act_variation_rollout.py \
  --checkpoint /absolute/path/to/run/checkpoints/best.pt \
  --device cuda --action-steps 30 --episodes 50
```

默认每30个动作重新观察并预测；可以显式改执行长度，但应视为另一个评估配置。
历史 LeRobot 目录仍可以显式传给 `--checkpoint`；未指定 `--action-steps` 时
读取各自 checkpoint 的 chunk 长度，不强迫旧模型变成K30。

## 与辅助实验的关系

`train_act_variation_gesture.py` 和 filtered 手腕辅助入口保留历史 LeRobot
协议，本次没有改其监督定义或损失。**当前不能把它们与新 baseline 的结果
当作只改变辅助损失的公平对照。** 如果后续迁移辅助实验，应共享本次模型、
split、归一化、预算、验证标准和执行方式，再单独增加手势/位移目标。
此次未自动启动这些实验，也未改动原始人手位移记录。

## 验证范围

使用人工构造的工程数据验证：单臂输入/输出、padding 数学、z=0验证不读取未来动作、
预取批次恢复、实际 LeRobot 读取、checkpoint保存/加载，以及两步连续训练与
一步中断后续训逐张量一致。该检查不是正式训练收敛或 MuJoCo 闭环成功率验证。
