# Paper A：双臂任务的采集、数据与学习工程


本工程将同一套 Panda 双臂控制器、数据处理和策略训练接入任务注册表。研究对照为 **ACT Baseline / ACT + Gesture** 和 **Diffusion Policy（DP）Baseline / DP + Gesture**。四组部署时都只输入单 oblique RGB 和机器人状态；手势类别仅在辅助模型训练时使用，不作为推理输入。设计、标签含义和公平对照约束见 [Gesture 辅助监督](docs/GESTURE_AUXILIARY_EXPERIMENT.md)。

```mermaid
flowchart LR
  T[Task Manifest / 场景工厂] --> C[Collection 手势采集]
  C --> R[按任务保存 Raw]
  R --> A[Audit → Curation / Safe Stitch]
  A --> P[人工审核或 auto → Prepared]
  P --> K[Training Pack / verify]
  K --> M[ACT Baseline / Gesture]
  K --> D[Diffusion Policy Baseline / Gesture]
  M --> W[best / last / 保留候选 Checkpoint]
  D --> DW[EMA best / last Checkpoint]
  T --> E[同场景 MuJoCo Rollout]
  W --> E
  DW --> E
```

## 当前任务

| task_id | 场景 | 状态 |
|---|---|---|
| `tray_transfer` | S1：双臂托盘搬运 | ready；完成本次工程全链路检查 |
| `tray_push_then_transfer` | S2：受阻托盘先推移再抓取搬运 | ready；场景加载/初始化回归通过；不是独立纯推移任务 |
| `tray_raised_placement` | S3：放到 25×33 cm、5 cm 高的平台 | ready；加载与既有物理回归通过；尚未在新工程进行真人验收 |
| `tray_to_platform` | 高侧壁空托盘，搬运到28×36 cm的实体高台，当前高度3 cm | ready；已有真人示范及模型闭环评估，结果与范围见第6.4节 |
| `lift`、`push` | 未来独立任务占位 | development；明确拒绝启动，尚未实现 |

ready 表示现有实现可加载，不表示新任务的真人采集质量或模型成功率已经验证。新增任务自动发现，不需要修改固定任务枚举。

## 目录职责

| 目录 | 职责 |
|---|---|
| `collection/` | 唯一采集菜单/启动入口；网页与 MediaPipe；调用共享 Recorder |
| `tasks/` | 动态注册、场景参数、任务钩子与来源绑定 |
| `simulation/mujoco/` | 共享 IK、双臂控制、接触保护、Recorder、XML/mesh |
| `dataset/` | 唯一数据管理入口；既有裁剪、Safe Stitch、审核、Prepared、Pack |
| `training/papera/` | 固定官方 ACT、正式单相机读取、训练/恢复与分析 |
| `training/diffusion/` | 单相机 Diffusion Policy、合法历史/未来窗口、训练与恢复 |
| `rollout/` | 唯一模型检查、预览、固定初态批量评估入口 |
| `config/` | 环境、采集基础、裁剪、训练和评估配置 |
| `assets/models/` | ResNet18 初始预训练资源；不是历史实验权重 |
| `datasets/<task_id>/` | 新采集原件；不自动覆盖 |
| `outputs/{dataset,training,rollout}/<task_id>/` | 按阶段、任务隔离的正式产物 |
| `logs/<task_id>/` | 采集服务与研究诊断 |
| `tests/` | 当前控制、Dataset、ACT/DP、Rollout回归；`run_regressions.py`为日常入口，上游ACT源码对照可选 |
| `.local/checks/` | 工程测试原件、短训模型、报告；不混入正式数据 |

本工程没有导入旧 datasets、outputs 或旧 Git 历史。迁移来源及逐文件对照见 [迁移验收](docs/MIGRATION.md)。

维护检查与分享：日常运行 `& $train tests/run_regressions.py`；正式数据接口用 `& $train -m tests.check_training_pack --task $task --pack $pack`，默认只读、不启动正式训练。可选上游源码对照及采集专项检查见[工程检查](docs/RUN_PIPELINE.md#7-可重复的工程检查不用于论文结果)。临时复现数据、下载权重及结果位于已忽略的`.local/`，不应上传GitHub；正式`datasets/outputs/logs`也已忽略。本文链接的本地报告、导航和权重随相应实验产物保存在这些目录中。队友先在`config/environments.json`修改两个Python和Node路径，再运行`scripts/configure_vscode.py`生成本机快捷配置。单臂和双臂可对齐模型方法与实验协议，但State/Action维度及任务成功条件需各自声明，不能互用Pack或Checkpoint。

## 环境与最短启动

复用本机 `mujoco_repro` 和 `paperA_train`，解释器及 Node 路径统一在 `config/environments.json`。不用安装另一套 Conda 环境。采集需要摄像头和浏览器授权；本次脚本检查没有代替真人摄像头验收。

```powershell
Set-Location 'D:\轻量化数据采集\paperA'
$envs = Get-Content config/environments.json -Raw | ConvertFrom-Json
$collect = $envs.collection_python
$train = $envs.training_python
& $collect -m collection list
& $collect -m collection start --task tray_transfer --operator-id operator01
```

也可双击 `START-PAPERA.cmd` 或 `collection/START-COLLECTION.cmd`，输入列表中的 task_id。默认网页 `http://127.0.0.1:8014/dual_arm.html`，后端 5014；点击“启动摄像头”后授权。终端保持打开，退出前保存或放弃本轮。端口占用会报错，不会替换未知服务。

采集策略保持迁移前最新的 `palm_fist_level_v2`：100 ms 确认；ILoveYou 控制 XY 并保持爪宽，张掌控制 Z 并匀速打开，握拳控制 Z 并匀速闭合；左右独立。绑定、暂停、丢手恢复和物理保护沿用原实现。

当前网页展示 top/front/oblique 三路（一主画面+两辅助，可切换主视图）；保留自动模式XY看top、Z看front，也可手动选择oblique。当前页面的斜视（side）统一命名为 oblique，oblique 是唯一录制与模型输入视角，top/front 只辅助观察、不录制。完整物理快照、状态、动作和手势日志继续保留。保存视角参数在 `config/recording.json`，新单视角和历史多视角均可审核及构建 Pack，详见全流程命令表第1～2节。

体验高台任务：双击启动器后输入 **`tray_to_platform`**，或运行 `& $collect -m collection start --task tray_to_platform --seed 0 --operator-id operator01`。当前 `tasks/tray_to_platform/scene.json` 的平台高度为0.03 m，加上0.35 m桌面，台顶为0.38 m；搬运时托盘底部需要高于台顶并留出间隙。几何、参数与物理验收见 [全流程命令第7.1节](docs/RUN_PIPELINE.md#71-高侧壁托盘到高台tray_to_platform)，学习模型结果见[第6.4节](docs/RUN_PIPELINE.md#64-四组模型的固定对照与结果查看)。脚本物理验收与学习模型自主成功率分别记录。

ACT与DP Baseline使用单 oblique RGB 640×480、30Hz模型动作、K30动作块、batch16和最大60,000次更新；当前DP已完成从头60k训练及独立闭环选型。ACT观察当前1帧，DP观察过去至当前2帧；相同更新次数不代表计算量或输入信息完全相同。物理仿真500Hz、控制器60Hz，每个模型动作目标执行2个控制tick。ACT Gesture配置为60k，DP Gesture仍保留历史40k，本轮未改未训；未来同模型Baseline/Gesture对照必须先同步数据、split、历史、主干、预算和选型协议。

ACT当前直接复用固定官方 DETRVAE、Post-LN/ReLU Transformer、位置编码、CVAE和Backbone，来源见 [ACT SOURCE.json](training/papera/official/SOURCE.json)。S16仅适配两个输入投影，输出A14和K30不移动标签。配置为hidden512、FF3200、encoder4/decoder7/posterior4、latent32，FP32、AdamW固定学习率1e-5。优化使用原版masked后按完整B×K×A均值的L1＋KL10，验证best仍依据独立的z=0有效动作L1；两种Loss数值不可混用。新ACT checkpoint为 `papera-act-official-checkpoint-v2`，旧本地模型权重不能加载或续训到新结构。使用说明见[ACT训练](docs/RUN_PIPELINE.md#4-act-公平训练两组)。

DP当前直接复用固定官方 U-Net、robomimic VisualCore/GN、DDPMScheduler和EMA，撤掉旧本地网络重写，来源见 [SOURCE.json](training/diffusion/official/SOURCE.json)。正式配置为 feature64、通道512/1024/2048、kernel5、FP32、DDPM100；RGB输入仍640×480，内部训练随机裁剪/评估中心裁剪472×632（高、宽）。保留S16/A14、train-only统计、Safe Stitch硬边界Mask，以及观察后下一合法tick为动作第0项。K30只在网络内部补到32再截回30。Baseline为fresh60k官方cosine训练，每5k保留完整EMA/优化器状态候选；不能将已到零学习率的40k last直接当作续训起点。配置与安装命令见[DP使用说明](docs/RUN_PIPELINE.md#41-diffusion-policy-baseline--gesture)。

新DP checkpoint为 `papera-diffusion-official-checkpoint-v2`。ACT和DP旧本地变体的权重、曲线和视频原样保留，但不能加载到新结构或接着续训；重新训练使用原来的统一入口，不维护第二套正式旧模型。Dataset裁剪、遥操作和任务成功条件保持不变。

当前官方 ACT Baseline（2026-11-06）：从40k继续至60k，保持同一Pack、split、归一化及官方网络/Loss/AdamW参数。独立20个验证初态选出 **50k保留权重 + chunk30**（17/20），随后冻结，在另外50个初态获得 **38/50（76%）**；7次超时、5次接触丢失，无基础设施错误，50路单oblique视频完整解码，CUDA推理与OpenGL均为RTX5090。47/50完成真实双抓和抬升。95% Wilson区间62.6%～85.7%，只反映这个训练seed的初态抽样不确定性；这是既有物理保护执行器下的成绩，未完成多训练seed或跨操作者质量对照。

本轮结果、曲线、逐例视频和代码复核见[ACT Baseline报告](outputs/campaigns/tray_to_platform/act_baseline_finalization_20261106_174942/REPORT.html)；准确权重与路径见[当前模型导航](outputs/ACT_BASELINE_CURRENT.json)，手动运行见[当前Baseline命令](docs/RUN_PIPELINE.md#63-当前官方-act-baseline明确权重与执行参数)。不要用本次run的`best.pt`或`last.pt`替代选中的`step_050000.pt`。60k是预算和最后恢复点，离线best也在60k，但闭环选型选择了50k。

同一20例验证中，40k→50k使chunk30成功数5→17，逐帧集成4→12；没有改数据、模型数学或成功判据。因此当前直接证据支持训练预算、权重选型及执行方式共同影响表现，不能把旧4%直接归因于裁剪。旧90%实际部署的是旧本地结构40k权重（后来训练至60k），其网络与优化配置不同，且使用另一批初态，不是本轮受控对照。可选1e-4诊断在16k停止并独立标记，没有完成同预算学习率比较，也未参与选型。

此前官方核心ACT40k按离线误差选择35k best得到2/50（4%）；DP40k的17k best得到8/50（16%），原结果保留在[此前实验清单](outputs/campaigns/tray_to_platform/official_baselines_20261105_235943/plan.json)。这些是历史成绩；本轮DP结果以独立选型后的新50例为准，不能与历史16%直接作单变量因果比较。仍共用64条示范及49/15整会话划分，数据人工质量审核提示保持不变。

当前官方 DP Baseline：从头60k官方余弦训练；独立20初态选中 **40000步EMA＋DDPM100/A30**（15/20），冻结后与ACT配对的50例为 **20/50（40%）**。全部失败保留，CUDA、60秒仿真上限及物理成功判据不变。[完整报告](outputs/campaigns/tray_to_platform/dp_baseline_finalization_20261106_205752/REPORT.html)、[准确模型导航](outputs/DP_BASELINE_CURRENT.json)、[手动命令](docs/RUN_PIPELINE.md#65-当前官方-dp-baseline配置与使用)。官方PushT参照各训练25,368步、各自last EMA评估50例，PushT `mean_score`（各回合最高归一化奖励的平均）为官方/本地0.8222 / 0.8222；它是同预算核心对照，不是完整论文训练预算或Panda训练主程序的替代验收。仅一个训练seed，结果包含既有物理保护执行器，不能据此单独证明数据优于其他采集方式。

历史优化对照曾将旧ACT两组从40,000步续训至60,000步，旧DP两组保持40,000步。训练预算与最终选用权重步数分开记录；`best.pt`的离线损失最优不等于物理任务成功率最高。历史结果与冻结执行配置见[四组模型的固定对照](docs/RUN_PIPELINE.md#64-四组模型的固定对照与结果查看)；运行历史权重需对应历史源码，当前官方核心不冒充旧实现。

历史结果（2026-11-04，ACT和DP均为旧本地变体）：四组各50次、同初态、60秒仿真上限、CUDA推理评估，ACT Baseline **45/50（90%）**，ACT + Gesture **40/50（80%）**，DP Baseline **17/50（34%）**，DP + Gesture **3/50（6%）**。该轮手势辅助设计未带来收益；只有一个训练seed，不作普遍优劣结论。实际部署权重分别为40k、40k、40k EMA、18k EMA；ACT的60k预算与选中的40k权重分别保留。这些不是新官方核心的成绩。历史模型、视频和曲线入口：[结果索引](outputs/rollout/tray_to_platform/comparison_20261104/manifest.json)。

2026-11-03 的托盘候选已按用户决定排除8条，冻结64条、60,035帧，以及49条训练/15条验证的整会话划分。`SELECTION_CURRENT.json` 导航选样；`PREPARED_CURRENT.json` 导航正式导出；`TRAINING_PACK_CURRENT.json` 导航模型实际输入。它们都位于 `outputs/dataset/tray_to_platform/`，选样清单本身不是Training Pack。多个采集来源由同一个 `prepare --run <多个Curation> --recorded-collection` 入口联合，保留每条原采集绑定；ACT/DP继续读取同一个正式Pack，不维护第二套Reader。排除范围、命令及边界见全流程命令第2～4节。

该联合Pack已实际完成：`outputs/dataset/tray_to_platform/training_packs/auto_20261103_213247_870817`，`complete=true`、`data_ready=true`。它采用自动裁剪，未冒充全量人工质量批准；`training_ready=false` 保留这项质量审核提示，现有探索性Baseline入口允许显式选用auto Pack。2026-11-04的旧本地模型四组实验与本轮官方核心Baseline分别保留；新的训练仍需显式启动。

同伴数据须由本工程的同任务、相同物理场景/模型相机和S16/A14契约采集，并提供完整会话（data、meta、episode_videos、research及物理快照）。不依赖操作者姓名或固定会话编号；已支持的手势置信度门槛差异会按冻结记录解释。只有MP4、缺日志的合并Parquet、不同机器人/任务/相机不能自动冒充同一数据契约。接收外部完整目录可用 `process --source <目录> --recorded-collection`，无需修改源码中的路径。

先采集至少两个独立会话，再按 [全流程命令](docs/RUN_PIPELINE.md) 构建 Pack、运行 ACT Baseline/Gesture 或 DP。DP 先用 `& $train -m training.diffusion --task $task --pack $pack --check-only` 只读检查，再去掉 `--check-only` 启动训练；默认参数见 `config/training_diffusion.json`，正式输出为 `outputs/training/<task>/diffusion/<run_id>`，支持 `--resume <run/checkpoints/last.pt>`。统一 `rollout` 已支持 ACT 和 DP：DP 使用EMA权重，每30Hz观察更新H2历史，按8步动作块滚动执行。两者离线误差都不是自主成功率；物理执行器与任务成功判据共用。`scripts/run_baseline_campaign.py`是初次训练后固定评估离线best的流程，不负责本轮闭环候选选型；复用当前官方ACT请用第6.3节的显式命令，历史四组仅按第6.4节的manifest查看。

- 新任务接入：[ADD_TASK.md](docs/ADD_TASK.md)
- 全流程命令、参数、VS Code、恢复训练：[RUN_PIPELINE.md](docs/RUN_PIPELINE.md)
- 字段、时间轴、任务隔离：[DATA_CONTRACT.md](docs/DATA_CONTRACT.md)
- 实际迁移清单、验证与限制：[MIGRATION.md](docs/MIGRATION.md)


## 采集保存使用独显（2026-11-03）

本机已将`config/environments.json`指定的`D:/Conda/envs/mujoco_repro/python.exe`设置为当前Windows用户的高性能图形应用。新启动的采集/快照保存进程已验证使用NVIDIA RTX5090 OpenGL；无需修改帧率、相机或数据格式。相同1107帧整段重放核显36.79/39.44秒，正式解释器切独显后4.49秒；两份视频均完整解码1107帧，状态/动作一致。此耗时不含人手会话日志导出及完整网页请求过程，下一次真人保存用于确认端到端体验。设置、备份、恢复和更换电脑说明见`docs/RUN_PIPELINE.md`末尾同名节；工程证据在`.local/checks/collection-nvidia-20261103/`。
