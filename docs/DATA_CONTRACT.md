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

Task Binding 含 task_id、task_version、scene_id、Manifest SHA、Schema、七个任务 hook、XML/mesh 路径和 source_files 哈希。共享采集/仿真配置也参与绑定。Source files 必须覆盖自定义任务代码的依赖；新增文件后要更新清单。正式来源审计在 Task 配置或资源变化时拒绝旧绑定，不自动取当前参数解释历史数据。DP 默认训练检查 Pack 内冻结身份，完整文件和当前源码审计由显式 `--full-verify` 启用，见下文。

同任务的多来源联合使用明确的采集参数等价契约，不能直接重写旧绑定。`dataset/task3/merge_prepared.py` 先核各原Prepared与实际episode_metadata，只允许 `tasks.recorded_collection.ALLOWED_CONFIG_PATHS` 中已支持的采集门槛不同，场景、模型相机、资产、Schema和其余配置必须一致。联合Prepared/Pack的全局 `task_binding` 表示当前规范学习任务（learning domain）；`source_task_bindings`、`source_prepared_manifests` 和 `dataset_collection_compatibility` 完整冻结原采集版本与差异，每条episode保存 `source_task_binding`、`source_prepared`、`source_prepared_manifest_sha256`。原Raw和源Prepared绑定不变。

构建时实际读取原metadata，核对其冻结记录，不能仅靠手写proof取得资格；联合输入还需逐条经过原NPZ/index/sequence因果与拼接检查。Pack随后内嵌这些已核来源，小元数据及每条来源关系在轻量读取时仍验证，但不要求原电脑、原Prepared目录或Raw在线。排除名单取来源并集，重复episode、来源错配、不同相机/场景及split交叉均拒绝。提供的master split冻结到联合Prepared，Pack不得静默改用另一份划分。

任务目录只允许本工程中的资源。模型创建、随机初态、reset、采集终态、Rollout monitor、评估前检查和 Dataset 事实语义通过任务钩子连接，签名见 [ADD_TASK.md](ADD_TASK.md)。执行器本身仍为共享 `dual_panda_v1`。

## Raw 契约

本工程当前采集 `task3-papera-v2`，保留既有 LeRobot 数据分片、视频、JSONL 和物理快照，不重新发明 Raw 格式。外层按任务分目录：`datasets/<task_id>/<session-directory>`；失败目录附 `_failures`。放弃记录是研究记录，不是成功示范。

每条来源可追踪 task_id/binding/version、scene_id、session_id/session、episode_index、operator_id、seed/session_seed/scene_index、outcome、input_kind、冻结 config/config_hash、相机及字段契约、控制策略版本和来源指纹。研究 JSONL 保留 accepted/rejected 输入、置信度、身份、measurement ID、event ID、控制 tick、时间映射、手势/骨架、暂停和执行结果。没有默认保存真人摄像头视频。

`input_kind=human_teleop` 且成功的正式示范才能进入普通 Dataset 流程。`scripted_test` 的失败工程保存只能走 `.local/checks/pipeline` 和显式 `--engineering`；任务成功仍为 false，不能伪造为正式候选。

当前网页展示 top/front/oblique 三路（一主画面+两辅助，可切换主视图），保留自动模式XY看top、Z看front，也可手动选择oblique；当前页面的斜视（side）统一使用 oblique 命名。oblique 是唯一录制与模型输入视角，top/front 只辅助观察、不录制。新采集按 `config/recording.json` 只生成 oblique 原视频，完整物理快照与控制证据继续保存。后端和网页代理分别提供相应实时预览。录制器、LeRobot features、相机清单及逐帧映射只声明实际生成的视角，不能把未生成的视频宣称为已保存。单视角审核直接使用 oblique；历史含 oblique 的三/四视角原件用 top/front/oblique 审核，历史派生数据继续严格解析其原始三视图来源。

录制输出视角与物理任务绑定分开：`collection.config.recording_config(task)` 在任务配置上应用输出选择，得到的完整配置及指纹写入 Raw；历史冻结配置不应用当前选项。任务、相机定义、State/Action Schema和裁剪不因此改变。新采集为 oblique 单视角，历史 top/front/oblique、top/front/side 三视角及 top/front/side/oblique 四视角继续兼容；不接受意外缺失一两路的混合布局。Training Pack 仍只构建 oblique 图像输入。

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
| ACT 模型 RGB | 单帧单相机 `[B,1,3,480,640]`，全画面、float32/255，模型内部沿用原ImageNet归一化 |
| DP 模型观察 | 同一oblique的过去至当前2帧，RGB `[B,2,1,3,480,640]`、State `[B,2,16]`；内部RGB映射 `[-1,1]`，按高宽472×632训练随机裁剪、验证/推理中心裁剪，配置可调整，Pack仍为640×480 |
| 物理/控制/图像与动作 | 500Hz物理（2ms步）；60Hz控制；固定相位30Hz图像锚点和30Hz模型动作，每个模型目标执行2个控制tick；控制步离散化可造成小幅间隔差异 |

无效数值、Schema、相机或单位不匹配会拒绝；不同机器人的维度不能仅改 manifest 宣称兼容。

## 裁剪与 Safe Stitch 时间轴

正式流程：完整性检查 → 每帧事实证据 → 原有裁剪规则 → 原有 Safe Stitch → 人工覆盖校验 → Prepared → Pack。Task Adapter 提供任务事件语义；核心 `curation_rules.py`、`frame_evidence.py`、`sequence_stitching.py` 保持迁移前实现。

两个时钟始终分开：

- `raw_frame_index`、`original_time_s`、`source_control_tick` 和 `action_control_tick/action_time_s` 追踪原始物理记录。
- `train_sequence_id`、`training_time_s` 和虚拟索引描述压缩后的合法训练顺序。

每个保留观察的 Action **仍来自其原始观察之后下一合法60Hz控制 tick**。每个30Hz图像锚点绑定一条该因果动作，沿虚拟序列构造未来K30；这不是连续30个60Hz控制tick，也不把Action错移到剪辑后的下一原始帧，不插值/缩放动作。模型动作按30Hz执行，每个目标保持2个60Hz控制tick。

例如原帧100、101、200、201：只有 Dataset 明确认证101→200为安全拼接时，ACT/DP动作块可以跨越。Episode、重置、人工排除或拒绝拼接形成硬边界；未来不足30步则 Padding。DP历史观察也只在同一合法序列内取帧，首部不足时重复该序列首帧。训练不重新判断距离阈值，不根据看起来流畅的视频重新拼接。

真实开合、接触建立/解除和物理收敛仍由原证据保护。未知手势不等于静止；是否压缩仍看原有机器人/物理规则。人工决定与自动证据分开保存。`data_ready` 表示结构可读，`training_ready/approved=false` 仍提醒质量审批没有被代码自动完成，不等于训练读取器未实现。

推动任务通过冻结的 `action_contract.capabilities.independent_xy_hold=true` 声明单侧 XY 控制能力，由任务 `scene.push_alignment` 语义启用。双手身份、时效和关键点仍须有效：一侧 Unknown 时该侧发送 HOLD/零位移并保持爪宽，另一侧可发送 XY_HOLD；双方实际操作平面 XY/Z 冲突仍双侧保持。Unknown 侧恢复手势时按原100 ms确认并重建自己的手腕基准，不回放回中位移。真正丢手、无效追踪和手动暂停仍停止双臂。托盘任务及没有此能力声明的旧服务保留双手同时 XY 的前端规则。原有 S16/A14、控制频率、手势标签、物理保护和 Safe Stitch 未改变；单侧 Unknown 不意味着整帧无动作。

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

`verify` 默认校验 Manifest、文件哈希、Prepared来源、任务与工程身份、split、因果标签和拼接证明。轻量读取必须持有当前进程里由 verify 返回的有效 VerifiedPack；旧 JSON 报告不能绕过校验。RGB 是 memmap，Worker 不通过 pickle 复制整份缓存，Windows spawn 后重新打开本地映射。

单源和联合来源都使用上面的同一 `verify/read_episode` 接口。联合Prepared只链接/复制既有NPZ、索引和sequence，不解码RGB、不改变删留和动作；只有随后正式Pack阶段生成一次模型RGB缓存。小型来源账本不取代数据格式、Action Mask和Safe Stitch检查。

Training 的 `training/papera/pack_data.py::SafeStitchDataset` 只负责训练集归一化、合法未来窗口和张量。训练/验证按会话固定划分；归一化只拟合训练部分，验证使用同一统计。输出 action `[B,30,14]`、is_pad `[B,30]`；Padding 不参加 Action Loss 或 CVAE 有效动作编码。Gesture 是观察锚点的左右分类监督，未知/缺失由 Mask 表达，不伪造标签。

DP 的 `training/diffusion/data.py::prepare_data` 默认用 `verify(..., check_hashes=False)` 检查 Pack 文件存在与尺寸、Pack/内嵌Prepared冻结的 task/schema/工程身份、会话 split、逐帧证据、S16/A14有限值、next-control因果关系及 Safe Stitch 证明。不访问外部 Raw/Prepared，不重算大文件哈希或当前任务源码哈希；`--full-verify` 才启用原完整校验。默认检查只能确认格式与接口，不能替代文件内容审计。入口只接受正式 Pack，不能把缺少原始控制/来源日志的 HF merged 数据直接视为 Pack 或在训练时修复。

`DiffusionDataset` 根据运行配置取历史；当前正式配置返回2帧观察、30帧未来动作与有效Mask。历史只在同一合法虚拟序列内取值，开头不足2帧时重复该序列首帧。动作仍从当前原始观察后下一合法控制tick开始，与ACT使用同一因果标签；不能照搬作者含过去动作序列的 `To-1` 偏移。输出 action `[B,30,14]`、is_pad `[B,30]`，硬边界之后的目标 padding 并从 epsilon MSE 排除。状态和动作按 train split 拟合 min-max，将训练集范围映射到 `[-1,1]`；RPY先按训练集角度参考处理周期，验证使用同一统计，不重新拟合、不裁切，超出训练范围的值可以超出 `[-1,1]`。冻结绑定保存 Pack 路径、Manifest元数据摘要、task_binding、工程属性、split、episode/合法序列与窗口数量、2/30/8及640×480图像规格、归一化。该摘要只覆盖小型元数据，不冒充整包内容哈希。新DP网络替换不重建Pack或改变Safe Stitch；旧网络权重不兼容，按下文版本明确拒绝。

## 训练与 Checkpoint

正式入口 `python -m training.papera --task ... --variant baseline|gesture --pack ...`。训练冻结 task_binding、Pack绝对路径/manifest SHA、split、来源指纹、模型配置、相机规格、归一化、variant、seed和源码/Git信息。

ACT直接复用固定官方组件：DETRVAE、Post-LN/ReLU Transformer、原位置编码与CVAE；`training/papera/official/SOURCE.json`记录提交742c753及原源码和许可证。任务适配只改变两处S16状态输入投影、配置单oblique/K30与必要Padding消毒，A14不变。输入RGB仍为640×480、ImageNet归一化，State/Action仍按训练split拟合z-score和RPY周期。正式参数hidden512/FF3200、encoder4/decoder7/posterior4、latent32，FP32、batch16、40k；AdamW两组LR1e-5、WD1e-4、常量LR、无warmup/无梯度裁剪。固定上游DETRVAE使用第一个intermediate decoder输出的行为保持不变。

ACT优化的Action L1保持官方分母：先屏蔽无效误差，再对B×K×A全部位置取均值，KL按潜变量维求和后按有效样本均值，权重10。训练另记有效坐标L1供诊断。验证z=0，best仍按`prior_action_l1`（等同`prior_valid_action_l1`）的有效坐标误差选择，另报告`official_action_l1`，不按测试成功率挑best。当前checkpoint契约为`papera-act-official-checkpoint-v2`，保存model/optimizer/scheduler/sampler/RNG、归一化、来源及源码快照；包含全部官方小型源码与许可证。旧`papera-act-offline-checkpoint-v1`不可resume/continue或加载，历史权重原样保留。

DP入口仍为 `python -m training.diffusion --task ... --variant baseline|gesture --pack ...`，配置路径不变。当前直接使用固定官方组件，来源及许可证见 `training/diffusion/official/SOURCE.json`：robomimic0.2 VisualCore的ResNet18、官方GN规则、32关键点SpatialSoftmax、64维投影及ReLU；原ConditionalUnet1D、FiLM和skip图不重写，通道 `[512,1024,2048]`、卷积核5。100步cosine前向加噪和100步反向DDPM由diffusers执行。S16/A14、H2、K30/A8继续沿用Pack；网络内部K30补到32再截回30只解决长度整除，不增加训练目标。当前`FP32/batch16/40k`，AdamW1e-4、500步warmup后余弦降0、不裁剪梯度、官方幂次EMA。每个30Hz观察都更新H2，查询帧默认0、8、16……，首个动作不增加To-1偏移。

无效Action先清除非有限值再与完整高斯噪声合成；去噪输入不再把noisy后缀置零，Loss仍只按有效坐标计数。数据硬边界Mask和网络内部补齐的2个位置含义不同。旧本地变体的ImageNet归一化、特征128、核3、DDIM10/20和旧EMA属于历史实现，现有34%结果不能作为新官方核心的结果。

Gesture两组均采用权重0.1的左右三分类CE，只有有效手实例参与均值。标签是当前锚点关联的识别类别，Unknown不补类别、不删除机器人帧。ACT辅助头读原input_proj图像特征的空间/相机均值拼接当前S16，不读取latent、posterior或未来动作；旧本地观察Transformer已撤回。该辅助头是本工程研究扩展，不是官方ACT原有模块，Baseline不计其Loss；两组公共参数初始化一致。DP头仍只读H2观察条件。推理均无需手势，细节见`GESTURE_AUXILIARY_EXPERIMENT.md`。

ACT与DP正式默认均为单oblique、640×480、30Hz模型动作、K30、batch16和40,000次更新；ACT历史为1帧，DP历史为2帧。各自Baseline/Gesture固定同Pack、split、相机、历史及预算；跨ACT/DP比较必须披露历史差异，若要求相同输入信息，另做DP-H1对照。之后修改历史长度、分辨率、网络宽度或预算须记录为新实验。

DP run为 `outputs/training/<task>/diffusion/<run_id>/`，冻结配置、数据绑定和模型规格，记录指标、TensorBoard、完成状态及best/last。新Checkpoint契约为 `papera-diffusion-official-checkpoint-v2`：保存model、EMA权重与完整EMA状态（配置、optimization_step、decay）、optimizer、scheduler、sampler、RNG、归一化和数据绑定；run同时记录实际diffusers/robomimic/einops版本、官方commit、实现标识及源码快照。旧 `papera-diffusion-checkpoint-v1` 属于不同网络，当前训练/推理明确拒绝，须从同Pack新建训练；不修改历史权重，也不保留第二套正式旧模型入口。

`best` 继续按固定噪声和timestep的 **EMA验证noise MSE** 选择，不照搬官方按测试闭环score选TopK；动作误差仅供诊断。`last`恢复同配置、Pack、split和预算。默认不重复大文件内容哈希，`--full-verify`才做完整审计。离线误差不是自主成功率，短工程验证不能宣称新版已通过Panda50例，run/checkpoint仍记录 `closed_loop_verified=false`。

## Rollout 与场景一致性

入口 `python -m rollout` 支持ACT和DP各自的Baseline/Gesture，通过run格式选择适配器并读取冻结的task与Pack；当前 Task不匹配直接拒绝。与采集共用 XML factory、initial_state/reset、物理资产和控制器，单 oblique 相机及图像预处理与训练一致。模型预测经现有双臂物理执行器，记录原预测、接受目标和保护干预。DP通过 `rollout/diffusion_policy.py` 加载EMA权重及训练集min-max归一化，独立维护H2历史和逐episode随机种子，仍共用原物理执行器。DP显式使用评估配置；推理步数或动作执行长度的显式覆盖必须记录选择理由和实际值，不修改checkpoint冻结训练配置。不同执行配置使用独立协议，但50个seed与实际初态、成功判据保持一致。同算法Baseline/Gesture比较时冻结相同执行设置。

托盘任务默认30Hz观察与模型动作、60Hz控制、500Hz物理；每个模型目标执行2个控制tick，使用K30 temporal ensemble（现有系数0.01）、60秒预算。正式成功采用已存在的半开释放评估协议。具体任务成功及 preflight 可由 Task替换，不修改共享 ACT。静态几何检查不等于动态成功。

Evaluation manifest 保存task、checkpoint SHA、Pack/source SHA、初态、代码、执行配置和成功判据；每次结果记录实际物理证据、失败原因、轨迹和已完整封装视频。`--resume` 仅恢复同一协议中未完成项。Dataset CURRENT变化不能改变已冻结 run；评估协议也不会因文件夹时间改变。

## 高侧壁高台任务的接口与证据

`tray_to_platform`（`HIGHWALL_PLATFORM`）沿用同一单 oblique、S16/A14、60 Hz 控制、30 Hz 图像及 Safe Stitch 时间轴，不引入新的训练输入、Action 编码或 ACT 分支。

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
