# 独立迁移交付与验收

交付日期：2026-10-23。新工程 `D:\轻量化数据采集\paperA`，正式数据目录尚无真人示范或模型；工程验证产物全部在 `.local/checks`。

## 来源保护、备份与版本边界

- 只读来源：`D:\轻量化数据采集\paperA_task3`。
- 来源 commit：`204c3c7bd8606ca7eb0dab14629e1280f5b957a0`。
- 外部备份：`D:\轻量化数据采集\paperA_版本备份\pipeline_migration_20261023_181639`。
- 四份旧主要说明和旧 AGENTS 已逐字节备份，`documents_manifest.json` 保存大小/SHA256；原文档没有改写。
- 新 Git 独立初始化，主分支 main，最终回溯标签 `migration-v1`；没有复制旧 `.git`，没有远程地址，没有推送。
- `source_inventory.json`、`dependency_imports.json`、`migration_plan.csv` 是复制前审计；`final_migration_all_files.csv` 是实际结果全清单。均位于外部备份目录。

复核结果：**431 个初始已计算哈希的源码、配置、文档和选定资源内容一致；来源 Git commit/status 不变；检查了16,731个初始可访问文件的大小及修改时间。**

必须单独说明：旧工程此前运行的50次评估在本次迁移期间于18:47:21完成。6个旧日志/评估结果的文件元数据发生变化，详细路径见 `.local/checks/source_protection.json`。本次迁移没有向这些文件写入，也没有回退或停止旧评估。因此不能宣称旧目录的所有历史输出在整个期间都完全静止。

另有一个原始子目录 `datasets/papera_task3_20261015_233028_764874_oblique` 继承ACL拒绝访问。它本就不在迁移范围，没有修改ACL、夺取所有权或跳过必要运行资源。对这个原始子树不作完整性认证；本次选择性迁移的运行资源均正常读取，新录制的工程视频均正常解码，无需管理员权限。

## 实际迁移前后对照

以下是主要接口；**完整逐文件记录含前后大小、SHA和理由，见 [MIGRATION_FILES.csv](MIGRATION_FILES.csv)**。所有历史生成文件的完整16,000余行清单保存在外部备份 `final_migration_all_files.csv`，没有塞入新运行目录。

| 迁移前 | 迁移后 | 实际处理及理由 |
|---|---|---|
| `START-TASK3.cmd` / 旧启动脚本 | `START-PAPERA.cmd`、`collection/START-COLLECTION.cmd`、`collection/cli.py` | 新统一任务入口；检查端口、启动自己的进程、打印当前输出；闲置后端用原退出命令正常关闭 |
| `teleoperation/mediapipe/` | `collection/teleoperation/mediapipe/` | 选择性复制当前网页、worker、手势模块、模型、所需MediaPipe JS/WASM及测试；调整根路径/代理端口 |
| `simulation/mujoco/dual_arm/` | 同相对目录 | 保留共享控制器、IK、双抓、保护、Recorder；只接入Task场景/初态/终态钩子和来源/输出路径 |
| `simulation/mujoco/assets/` | 同相对目录 | 按实际引用复制XML、Panda mesh、模型构造资源和Panda LICENSE |
| 散布在场景模块/配置的S1、S2、S3 | `tasks/tray_*/task.json`、`scene.json`、`scene.py`、`tasks/adapters/tray.py` | 三个实际场景注册；共享随机初态、场景工厂、成功语义、preflight、Dataset adapter |
| 无Task Registry | `tasks/registry.py`、`tasks/artifacts.py` | 动态扫描、配置/资产SHA、跨阶段身份检查；development拒绝启动 |
| `dataset/manage_dataset.py` | 同路径 | 增加必填task及工程范围；输入/输出/CURRENT/split按任务隔离；续建复用原split |
| `frame_evidence.py`、`curation_rules.py`、`sequence_stitching.py` | 同路径 | **三个核心文件字节一致**，不改裁剪或安全拼接算法 |
| `prepare_paper_dataset.py`、`rebuild_curation.py`、`training_pack.py` | 同路径 | 传播Task/工程来源，真实Reader兼容说明；保留原对齐/数组语义，移除重复CLI |
| `validate_task3_dataset.py` | `dataset/task3/source_paths.py` | 仅提取仍被引用的dataset_path安全函数；旧整套验证/导出入口不迁入最终工程 |
| `validate_recordings.py` | 同路径 | 保留实际调用的手势时间线/快照审计；删除未调用的旧物理删帧导出和CLI |
| 旧derive/render迁移、legacy、recording_recovery入口 | 不保留正式入口 | 旧工程留存；新采集原生四相机，不再维护旧数据派生/恢复链路 |
| `training/papera/act_model.py` | 同路径 | 归一化换行后文本及AST一致，网络、CVAE、L1/KL/辅助头未改变 |
| `training/papera/data.py` | 同路径 | 保留归一化/角度等共享工具，删除不再使用的旧Raw/三视角数据Reader |
| `pack_data.py`、`train.py` | 同路径 + `training/papera/__main__.py` | 复用正式Pack Reader；task/variant显式路由、来源冻结、Gesture必要接口、原优化器/恢复机制 |
| 旧结果分析脚本 | `analyze_checkpoint.py`、`plot_training_analysis.py` | 适配当前Pack；工程验证两批与全验证集分析分别标注，正式完整验证仍做数值复现检查 |
| `evaluation/papera/{rollout,policy,protocol,executor,action_chunks}.py` | `rollout/` 对应模块 | 同一套评估实现迁移，统一 `python -m rollout`；task/checkpoint匹配、共用场景、任务preflight/monitor |
| `config/` 旧冻结Pack/run路径 | `config/` 当前配置 | 清除旧绝对路径/历史排除Episode；保留当前训练数学参数、采集策略、裁剪阈值与60秒半开评估协议 |
| `.cache/torch/hub/checkpoints/resnet18-f37072fd.pth` | `assets/models/resnet18-f37072fd.pth` | 必须的初始预训练资源，不是旧实验checkpoint；已纳入新Git |
| 一次性pilot/debug/旧测试候选 | 不保留 | 检查运行导入、保留测试依赖闭包及JS夹具后清理；保留的回归有真实运行结果 |
| 旧README/采集/实验/Training/AGENTS | 外部备份；新README/AGENTS及docs | 旧文不变；新文档以接入和运行流程为主 |
| 旧`datasets/`、`outputs/`、logs、progress、.git | 不迁入 | 历史数据/结果只留原处；新生成工程数据独立标记 |

初始候选303个来源文件，最终210个有保留或部分提取的实际目标，93个候选在依赖审计后未留在最终新工程。该数字不包括另行新建的Registry、入口、文档和测试；逐文件表才是最终依据。部分“适配后迁入”只是换行格式变化，不能据此推断算法改变。`.gitattributes` 禁止Git换行转换，避免恢复时源码SHA因checkout换行发生变化。

完整交付文件树见 [PROJECT_TREE.txt](PROJECT_TREE.txt)。顶层职责：

```text
paperA/
  START-PAPERA.cmd  README.md  AGENTS.md
  .vscode/                # 生成的F5快捷入口
  collection/             # cli + 网页/MediaPipe
  tasks/                  # registry、adapters、3个ready及2个development
  simulation/mujoco/      # 共享控制、录制、物理资产
  dataset/task3/          # 既有裁剪、审核、Pack正式模块
  training/papera/        # 现有ACT数学实现与正式单相机训练
  rollout/                # 唯一评估实现
  config/  assets/  scripts/  tests/
  datasets/  outputs/     # 无旧数据，无正式真人产物
  logs/  .local/checks/   # 仅新工程本次生成的诊断/工程产物
  docs/                   # ADD_TASK、RUN_PIPELINE、DATA_CONTRACT、本报告及清单
```

## 环境及资源

实际检查：Windows11；mujoco_repro Python3.12.14、paperA_train Python3.12.15；MuJoCo3.12.0；PyTorch2.11.0+cu130 / torchvision0.26.0+cu130；RTX5090，CUDA13.0、BF16可用；PyAV15.1.0；PyArrow22.0.0；采集LeRobot0.6.1/Flask3.1.3；训练TensorBoard2.20.0；采集环境matplotlib3.11.2；Node24.21.0。

完整包检查见 `.local/checks/environment.json`。没有重装环境。必要MediaPipe模型、WASM/JS、ResNet18初始权重、XML/mesh都在新工程，不从旧工程查找。原Panda许可证已保留；MediaPipe包的license声明和bundle版权头保留；torchvision的LICENSE另存 `assets/licenses/torchvision-LICENSE`。没有为未复制的调试工具复制其依赖/许可证。

## 全流程实测结果

| 检查 | 真实结果/证据 |
|---|---|
| Registry | 3个ready场景创建与同seed初始化一致；动态加入临时CUSTOM manifest可发现并加载；未知/损坏/development/跨任务拒绝 |
| Python回归 | `accepted_regressions.log`：299项 + 采集环境17项，全部通过，0跳过；含Windows双Worker memmap读取、真实Safe Stitch窗口、Padding、归一化和BF16恢复 |
| 浏览器模块回归 | `final_browser_regressions.log`：211项通过；审核指标JS另行通过 |
| 真正启动器/HTTP | 本地网页、代理health、JS、手势模型均200；后端指向新根目录；正常请求退出后两个服务关闭，没有打开摄像头或生成真人录制 |
| 工程采集 | 两个新会话，真实MuJoCo/Recorder，每条211图像帧、420控制tick、140脚本输入；每条136接受，4次追踪恢复拒绝；明确scripted_test/failure |
| Dataset | 两条完整性/视频校验通过；422→256帧；2处安全拼接，2条虚拟序列；不修改原件 |
| ACT窗口 | train/val各1会话、128保留帧、127有效观察窗口；各98个全长K30窗口、29个跨安全拼接窗口；无效终端动作锚点排除 |
| Pack | build、verify、同目录resume通过；resume不改变manifest SHA；跨任务/工程来源校验保留 |
| 审核视频 | 通过真实HTTP生成三视图预览，各128帧完整解码；没有自动保存人工批准 |
| Baseline | 新Pack真实GPU前向/反向/AdamW更新；第1步保存，再恢复到第3步，checkpoint/日志完整 |
| Gesture | 同Pack/split/seed/主干完成3步；首步gesture_count=32，辅助损失实际非零，非空壳入口 |
| 分析与绘图 | 完整split离线分析及PNG/PDF/CSV导出实际完成；smoke日志仅两批，报告中明确不与全量均值混比 |
| 短Rollout | Baseline2秒、Gesture1秒；分别120/60控制步；零拒绝动作；两个MP4分别61/31帧可完整解码；均超时，**不是成功率实验** |
| 固定评估清单 | 50个静态初态检查、协议保存/复核、跨任务拒绝通过；没有执行50次模型策略 |
| 入口 | 14个正式入口/子命令帮助检查通过；实际数据、训练、恢复、模型检查和短Rollout命令记录在command_checks.json |

最终工程Pack：

```text
D:\轻量化数据采集\paperA\.local\checks\pipeline\outputs\dataset\tray_transfer\training_packs\auto_20261023_194522_323702
manifest SHA256 = ac4c8b22bf6484a9c112cfb2fe28a77920b4a13789470d3e91e07c00fec7a909
```

工程模型：`.local/checks/training/accepted_baseline` 和 `accepted_gesture`；工程Rollout在 `.local/checks/rollout/accepted_*`。这些只为接口检查，**不应拿来评估正式任务准确率，也不会作为默认正式训练数据**。失效中间工程模型和重复采集已清理，清理列表位于 `.local/checks/superseded_smoke_cleanup.json`；没有清理原工程。

## 发现的问题与仍需完成的事项

1. 当前正式输出为空；下一步是你用新入口进行真人摄像头操作验收并采集至少两个独立会话，再构建真人Pack。脚本没有验证真人识别体验。
2. `lift`、`push` 是development。S2是“推移后抓取搬运”，不是独立纯推移；不要把占位任务写成完成。
3. 50个静态初态中，25个在辅助视图存在裁切标记：top初态/抬高分别6个、front初态/抬高分别23个（有重叠）；**oblique及side的检查点全部在视野内**。本轮保留原相机，不擅自改变场景/训练输入；裁切标记没有过滤测试种子。几何可见不保证无遮挡。
4. 当前同类托盘任务共用成功语义；未来纯推移需要自己的Task success/monitor/evidence/preflight，不能只改名称。不同Schema仍需明确适配。
5. 原采集DONE逻辑与已采用半开规则的Rollout判据分别保留原行为；迁移没有把两者偷偷改成一套新判据。正式论文对照应使用同一冻结评估协议。
6. 本轮没有正式20,000步训练、没有大规模Rollout、没有GitHub推送；短训Loss和脚本通路通过不能证明模型任务成功。
7. 原工程仍有上述ACL子树及并行旧评估更新，故保护结论限定在实际检查范围，不声称全目录字节保持不变。

## 如何使用交付

- 逐项输入/输出/参数与PowerShell命令：[RUN_PIPELINE.md](RUN_PIPELINE.md)。
- 新任务建模、注册、场景函数与验证：[ADD_TASK.md](ADD_TASK.md)。
- 单相机S16/A14、因果动作、Safe Stitch和来源字段：[DATA_CONTRACT.md](DATA_CONTRACT.md)。
- 真实迁移前后逐文件表：[MIGRATION_FILES.csv](MIGRATION_FILES.csv)。

新工程的版本查看：`git log --oneline --decorate`、`git tag -n`、`git status`。回溯前先保存当前改动；`migration-v1` 是本次独立迁移基线，原工程的历史仍在原仓库。
