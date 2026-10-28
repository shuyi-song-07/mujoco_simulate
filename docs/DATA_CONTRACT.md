# 数据契约与来源链

```text
Task(task_id + task_version + task_binding)
  → Session(session_id, operator_id, seed)
  → Episode(episode_index / key, outcome, input_kind)
  → Audit / Curation(run_id，作为 dataset_version)
  → Prepared(review_mode, 原始帧映射)
  → Training Pack(路径/目录名作为 pack_id + manifest SHA + split)
  → Training Run(run_id + variant + signature + checkpoint)
  → Rollout(evaluation_id + protocol + checkpoint SHA)
```

内部既有 `run_id`、`session`、`episode_index` 不为命名整齐而重复改写：上图给出统一含义。Episode key 是数据会话目录和 episode_index 的稳定组合；读取 manifest，不用创建时间推断来源。

导航文件 `CURATION_CURRENT.json`、`PREPARED_CURRENT.json`、`TRAINING_PACK_CURRENT.json` 共用 `dataset.manage_dataset.write_navigation`：标准 JSON 序列化，UTF-8 无 BOM，中文采用 JSON Unicode 转义以兼容 Windows PowerShell 5.1 的默认读取；绝对路径使用 `/`，不手工拼接或替换 JSON 转义。写入同目录临时文件后原子替换，避免读取到半截 JSON。字段继续为 Curation 的 `selected_version/recent_versions`、Prepared 的 `path/curation`、Pack 的 `path/prepared`；候选与未完成 Pack 不更新 CURRENT。PowerShell 示例显式使用 `Get-Content -LiteralPath ... -Raw -Encoding UTF8 | ConvertFrom-Json`。导航修复不选择新版本、不改已有产物或其指纹，Unicode 转义只是文件表示，解析后仍得到中文路径。

## Task Registry

入口 `tasks.registry.get(task_id)` 返回经校验的 Task，`.config()` 合并共享采集配置和任务完整 scene 参数，`.binding()` 冻结身份。`tasks.artifacts` 负责跨 Raw/Audit/Prepared/Pack 的绑定检查。

Task Binding 含 task_id、task_version、scene_id、Manifest SHA、Schema、七个任务 hook、XML/mesh 路径和 source_files 哈希。共享采集/仿真配置也参与绑定。Source files 必须覆盖自定义任务代码的依赖；新增文件后要更新清单。Task 配置或资源变化时拒绝旧绑定，不自动取当前参数解释历史数据。

任务目录只允许本工程中的资源。模型创建、随机初态、reset、采集终态、Rollout monitor、评估前检查和 Dataset 事实语义通过任务钩子连接，签名见 [ADD_TASK.md](ADD_TASK.md)。执行器本身仍为共享 `dual_panda_v1`。

## Raw 契约

本工程当前采集 `task3-papera-v2`，保留既有 LeRobot 数据分片、视频、JSONL 和物理快照，不重新发明 Raw 格式。外层按任务分目录：`datasets/<task_id>/<session-directory>`；失败目录附 `_failures`。放弃记录是研究记录，不是成功示范。

每条来源可追踪 task_id/binding/version、scene_id、session_id/session、episode_index、operator_id、seed/session_seed/scene_index、outcome、input_kind、冻结 config/config_hash、相机及字段契约、控制策略版本和来源指纹。研究 JSONL 保留 accepted/rejected 输入、置信度、身份、measurement ID、event ID、控制 tick、时间映射、手势/骨架、暂停和执行结果。没有默认保存真人摄像头视频。

`input_kind=human_teleop` 且成功的正式示范才能进入普通 Dataset 流程。`scripted_test` 的失败工程保存只能走 `.local/checks/pipeline` 和显式 `--engineering`；任务成功仍为 false，不能伪造为正式候选。

新采集按 `config/recording.json` 默认只生成 oblique 原视频，完整保存物理快照与控制证据；实时辅助预览使用 top/front/oblique；后端和网页代理提供 `/oblique-preview`，替代旧 `/side-preview`。录制器、LeRobot features、相机清单及逐帧映射只声明实际生成的视角，不能把未生成的视频宣称为已保存。单视角审核直接使用 oblique；含 oblique 的三/四视角原件用 top/front/oblique 审核，历史派生数据继续严格解析其原始三视图来源。

录制输出视角与物理任务绑定分开：`collection.config.recording_config(task)` 在任务配置上应用输出选择，得到的完整配置及指纹写入 Raw；历史冻结配置不应用当前选项。任务、相机定义、State/Action Schema、裁剪和训练算法没有因此改变。允许原生 oblique 单视角、当前 top/front/oblique 三视角，兼容历史 top/front/side 三视角及 top/front/side/oblique 四视角；不接受意外缺失一两路的混合布局。Training Pack 仍只构建 oblique 图像输入。

## State / Action / 相机

`config/collection.json.training_camera` 的 `papera-oblique-config-v1` 允许调整 distance、azimuth、elevation、lookat；名称 oblique、透视投影、640×480 和 30 fps 保持当前契约。每次采集冻结完整相机参数和 spec_sha256，改变任一视角参数会改变指纹；回放使用来源配置，不套用当前视角。历史 `task3-oblique-v1` 仍严格核对原固定参数，不能修改旧记录以冒充新视角。Task Binding 和 Pack/Rollout 的相机来源一致性检查继续生效。

固定顺序 `left` 然后 `right`，具名字段见 `simulation/mujoco/dual_arm/action_contract.py`：

| 内容 | 契约 |
|---|---|
| State | 16D，每臂7个实测关节角(rad) + 实测双指总开度(m) |
| Action | 14D，每臂目标 xyz(m)、roll/pitch/yaw(rad)、目标总开度(m) |
| 坐标系 | Raw metadata 的 task_frame / transforms；不能默认用物体局部坐标 |
| 夹爪 | 连续米制宽度，0为闭；当前每爪最大约0.08m，以冻结配置为准；不是0/255 |
| Raw RGB / Pack | oblique RGB uint8，640×480×3；不透明机器人 |
| 模型 RGB | 单相机 `[B,1,3,240,320]`，全帧resize，float32/255，模型内部沿用原ImageNet归一化 |
| 物理/控制/图像 | 2ms物理步；60Hz控制；固定相位30Hz图像锚点，控制步离散化可造成小幅间隔差异 |

无效数值、Schema、相机或单位不匹配会拒绝；不同机器人的维度不能仅改 manifest 宣称兼容。

## 裁剪与 Safe Stitch 时间轴

正式流程：完整性检查 → 每帧事实证据 → 原有裁剪规则 → 原有 Safe Stitch → 人工覆盖校验 → Prepared → Pack。Task Adapter 提供任务事件语义；核心 `curation_rules.py`、`frame_evidence.py`、`sequence_stitching.py` 保持迁移前实现。

两个时钟始终分开：

- `raw_frame_index`、`original_time_s`、`source_control_tick` 和 `action_control_tick/action_time_s` 追踪原始物理记录。
- `train_sequence_id`、`training_time_s` 和虚拟索引描述压缩后的合法训练顺序。

每个保留观察的 Action **仍来自其原始观察之后下一合法60Hz控制 tick**。沿虚拟序列构造未来 K30，不把 Action 错移到剪辑后的下一原始帧，也不插值/缩放动作。

例如原帧100、101、200、201：只有 Dataset 明确认证101→200为安全拼接时，ACT动作块可以跨越。Episode、重置、人工排除或拒绝拼接形成硬边界；未来不足30步则 Padding。训练不重新判断距离阈值，不根据看起来流畅的视频重新拼接。

真实开合、接触建立/解除和物理收敛仍由原证据保护。未知手势不等于静止；是否压缩仍看原有机器人/物理规则。人工决定与自动证据分开保存。`data_ready` 表示结构可读，`training_ready/approved=false` 仍提醒质量审批没有被代码自动完成，不等于训练读取器未实现。

## 正式 Pack Reader

所有正式训练经同一接口读取，不解析 Raw 重做裁剪：

```python
from dataset.task3.training_pack import verify, read_episode

checked = verify(pack_path)
key = checked["manifest"]["episodes"][0]["key"]
episode = read_episode(pack_path, key, checked=checked)
arrays = episode["arrays"]
rgb = episode["images"]["oblique"]
sequence = episode["diagnostics"]["virtual_sequence"]
```

主要数组包含 state、action、action_mask、raw_frame_index、original_time_s、action_time_s、source_control_tick、action_control_tick、control_epoch、train_sequence_id、training_time_s、actual_done，以及 category_id/category_mask、hand_normalized/task_geometry_mask、stage_id。完整原始置信度及关联证据保留在 Raw 和 evidence 中，模型本轮不使用的字段不因此删除。

`verify` 校验 Manifest、文件哈希、Prepared来源、任务与工程身份、split、因果标签和拼接证明。轻量读取必须持有当前进程里由 verify 返回的有效 VerifiedPack；旧 JSON 报告不能绕过校验。RGB 是 memmap，Worker 不通过 pickle 复制整份缓存，Windows spawn 后重新打开本地映射。

Training 的 `training/papera/pack_data.py::SafeStitchDataset` 只负责训练集归一化、合法未来窗口和张量。训练/验证按会话固定划分；归一化只拟合训练部分，验证使用同一统计。输出 action `[B,30,14]`、is_pad `[B,30]`；Padding 不参加 Action Loss 或 CVAE 有效动作编码。Gesture 是观察锚点的左右分类监督，未知/缺失由 Mask 表达，不伪造标签。

## 训练与 Checkpoint

正式入口 `python -m training.papera --task ... --variant baseline|gesture --pack ...`。训练冻结 task_binding、Pack绝对路径/manifest SHA、split、来源指纹、模型配置、相机规格、归一化、variant、seed和源码/Git信息。

ACT 使用原有 ResNet18 + Transformer + CVAE；Baseline 保持 L1+KL；Gesture 复用原辅助头。模型类保留原有历史结构分支以避免改写核心，但新正式 CLI 只接受单 oblique 的两组实验；旧三视角完整复现留在只读原工程，没有第二套新正式流程。

完整 checkpoint 保留 model、optimizer、scheduler、sampler、RNG、当前步数、最佳验证值和 provenance。`best` 按验证 z=0 的动作误差选择，`last` 用于恢复；不根据测试成功率挑 best。模型在模拟中的闭环表现必须单独评估，离线 Loss 下降不等于成功。

## Rollout 与场景一致性

入口 `python -m rollout` 读取 run 冻结的 task与Pack；当前 Task不匹配直接拒绝。与采集共用 XML factory、initial_state/reset、物理资产和控制器，单 oblique 相机及图像预处理与训练一致。模型预测经现有双臂物理执行器，记录原预测、接受目标和保护干预。

托盘任务默认30Hz观察、60Hz控制、K30 temporal ensemble（现有系数0.01）、60秒预算；正式成功采用已存在的半开释放评估协议。具体任务成功及 preflight 可由 Task替换，不修改共享 ACT。静态几何检查不等于动态成功。

Evaluation manifest 保存task、checkpoint SHA、Pack/source SHA、初态、代码、执行配置和成功判据；每次结果记录实际物理证据、失败原因、轨迹和已完整封装视频。`--resume` 仅恢复同一协议中未完成项。Dataset CURRENT变化不能改变已冻结 run；评估协议也不会因文件夹时间改变。

## 高侧壁高台任务的接口与证据

`tray_to_platform`（`HIGHWALL_PLATFORM`，任务版本 1.0.0）与旧 5 cm 平台任务 `tray_raised_placement` 使用不同 task_id/Binding。它沿用同一单 oblique、S16/A14、60 Hz 控制、30 Hz 图像及 Safe Stitch 时间轴，不引入新的训练输入、Action 编码或 ACT 分支。

来源由 Task Binding 和录制时冻结的完整 config 保存：`scene.paper_object`、`scene.platform_task.geometry_version`、`scene.placement_platform`、随机初态/seed、相机、控制协议，以及 `scene.py`/`success.py`/评估配置的 SHA。平台顶面 Z=0.45 与目标 body Z=0.484 必须分别解释；底面偏移为 -0.034，物体位置字段不能被当作台顶或质心坐标。质量和惯量来自任务实际碰撞盒，具体尺寸与计算入口见 [ADD_TASK.md](ADD_TASK.md#6-已实现示例高侧壁托盘放置至-10-cm-高台)。正式原件位于 `datasets/tray_to_platform/`；处理、训练、评估的输出继续按该 task_id 隔离。

采集 `collection_success` 与 Rollout `rollout_monitor` 共同使用 `tasks/tray_to_platform/success.py::PlatformSuccessMonitor`，任务判据标识为 `papera-highwall-platform-contained-v2`（任务版本1.1.0）。目标朝向误差仅记录诊断，不作为成功条件。要求真实双指双臂抓持历史、稳定离桌抬升历史、最终 XY/高度/速度达标、台面容纳、双爪指令和实际开度至少 50%、无手指接触，并在真实平台支撑下持续稳定后 DONE。支撑证据同时核对 `support_surface="placement_platform"`、`support_ready` 和平台法向载荷；仅有目标位姿、桌面接触或支撑物名字均不算成功。评估配置保留通用半开 `success_criterion` 供原协议校验，另明确 `task_success_criterion`；实际监视器证据同时输出任务判据与 `opening_criterion`，不能只看通用半开名称判断任务要求。共享碰撞、失抓和 IK 保护保持原实现。

Dataset 复用现有 tray Adapter、事实证据和裁剪核心。**任务 DONE 与离线“释放完成”是两个不同判断**：现有 `curation_rules.py` 的释放尾段确认仍检查实际开度接近最大值（`protection.open_width_tolerance_m`），因此合法半开 DONE 的尾部可能保守保留、标记待复查，而没有立即裁到完成后 3 帧。原始成功标记不因此变成失败；本轮不改裁剪阈值，也不伪造全开事件。审核这类尾段后再按正式流程导出 Prepared。

seed 0、1、2、3、17 的完整物理脚本搬运已达 DONE；工程记录仍须标记 `scripted_test`，不能归入真人正式训练候选。另有两条正式Recorder保存的工程失败记录通过Audit→Safe Stitch→Prepared→Pack：422原始帧、256保留帧、2处安全拼接，验证按会话分开。这些结果证明有限初态的物理任务和数据通路，未验证真人操作质量、总体随机成功率或模型策略成功率。脚本释放请求为80 mm；40 mm半开阈值的边界验证来自单元测试，并非停在半开位置的专门动态试验。

## 版本与迁移边界

成功/失败保存使用 `hand-save-boundary-v1`：点击时冻结 `end_capture_epoch_ms`，等待已采帧的识别完成，再冻结日志 `last_sequence`；前端等待该固定前缀的持久化 ACK，后端核对实际写入序列，并在提交机器人 Episode 前导出人手侧文件。边界后的新测量继续进入会话日志，不扩大本条等待范围。上传或人手导出失败会拒绝本次保存并保留当前录制，重试复用边界；明确放弃仍可归档研究警告。未经该边界协议的旧网页不能保存新的真人示范，应刷新。

等待上传不修改测量的 `capture_epoch_ms`、识别时间、measurement ID、控制 event ID、`clock_pair` 或服务器 tick。收包/落盘时间不能代替测量时间。导出保留原字段；Dataset 的原始帧号、控制来源和 Safe Stitch 虚拟时间仍沿用原契约。成功判定点与采集终点分开：首次 DONE 记录 `episode_success` 事件；之后再记录 `config/recording.json::post_success_seconds`（默认0.5）仿真秒，在第一个达到收尾期限的控制tick截止，保持原30 Hz固定相位采样。`episode_end`（`episode-done-boundary-v2`）包含原成功点 `success`、配置延时、截止的session、epoch、tick、仿真时间、服务器墙钟和单调时钟。控制状态中的 `success_tail` 记录已接受开爪目标、来源、期限及中断原因；机械臂冻结，只允许原速度续开，不接受新运动/闭爪。手动暂停或物理故障取消续开。该行为只影响采集收尾，不改变半开成功判据；历史未配置延时的控制器仍立即暂停。该点之后停止本条视频快照和控制研究时间轴，但 `recording=true` 保留待保存语义；终态快照不再被稍后的保存时刻覆盖。人手导出取开始录制至收尾截止的采帧时间区间；保存时仍等待在途识别和持久化 ACK，晚写入的终点前测量保留，终点后测量只留会话日志。未达到 DONE 的失败保存使用点击边界。Dataset 原有裁剪规则不变，不能在原件终点后补造三帧。

本轮建立独立新 Git，没有复制旧 `.git`、历史数据或训练权重。旧来源 commit 及逐文件 SHA 在 [MIGRATION.md](MIGRATION.md) 和迁移清单。ResNet18预训练与MediaPipe/WASM属于运行资源，已选择性复制并单独登记。

目前严格绑定的是一套 Task代码/场景版本；升级场景后应保存旧 Git版本，或用新task_id开启新的数据链。跨任务、跨版本、跨工程/真人来源不允许静默混合。未来不同Schema需要新显式适配器及对应验证；本轮未实现任意机器人/任意物理任务的通用算法。

高台任务 `tray_to_platform` 的成功规则为 `papera-highwall-platform-contained-v2`（任务版本1.1.0）：取消目标朝向误差门槛，仍要求真实双抓/抬升历史、高台完整容纳及真实支撑、双爪实际松开和稳定。采集和 Rollout 共用同一监视器。朝向误差仅作诊断记录；其他任务判据不变。旧权重和旧记录的任务来源不重写，不能与新版规则的成功率不加区分地合并。
