# 全流程运行与命令表

所有命令在新工程根目录执行。下文 `<...>` 是需要替换的实际路径，不是文件夹字面名称。默认不依赖旧工程，也不自动选择“时间最新”的数据。CURRENT 只做导航，训练启动后冻结实际路径和 SHA。

采集点击“保存成功/保存失败”后，先显示“正在收齐本条人手日志”，随后才提交保存并重置场景。摄像头不用关闭，按钮等待期间不要重复点击；等待保存点击时已采帧的识别和日志写入，然后按固定的本条终点导出；新测量不扩大等待。识别等待上限5秒、日志确认等待上限15秒，失败会提示“本条未提交保存、场景未重置”；恢复连接/文件权限后再次点击保存，复用原边界。持久丢失的记录不能凭重试恢复，不应把该条当作完整的人手监督示范；可明确放弃本轮并保留研究原件。任务首次 DONE 立即保持机械臂位置，按原速度继续已接受的开爪目标，并再记录0.5仿真秒（约15个30 Hz采样间隔）。期间显示“放置成功 · 记录开爪收尾”，保存按钮暂不可用；收尾结束显示“已截止 · 待保存”，保存成功/失败按钮恢复。等待点击期间不再增加本条帧数；摄像头会话日志继续记录，导出只取开始录制至收尾截止的测量。参数位于 `config/recording.json` 的 `post_success_seconds`，改后重启采集服务生效；0表示立即截止。该延时不提高成功开爪门槛，不新增开爪授权，不保证实测完全张开；手动暂停和物理故障仍中止开爪。未录制的练习不能保存，需重置并先点击开始录制。升级后需重启采集服务并 Ctrl+F5 刷新网页。

## 0. 环境、编辑器与公共变量

模型输入斜视角在 `config/collection.json` 的 `training_camera` 中调整：`distance` 为相机距观察中心的距离（米，必须大于0），`azimuth` 为水平角（度），`elevation` 为俯仰角（度，-90至90；-90俯视），`lookat` 为观察中心世界坐标 `[X,Y,Z]`（米）。当前 `version=papera-oblique-config-v1`，只改这四项即可，不必修改 Python 常量；640×480、30 fps、名称和投影类型保持不变。重启采集服务生效，当前配置供所有任务共用。完整视角参数及指纹随新会话冻结，旧视频不会改变；更换视角会改变任务来源绑定，不能把新旧视角无标记混作相同训练条件。历史 `task3-oblique-v1` 保留原固定参数，仅用于历史兼容。

```powershell
Set-Location 'D:\轻量化数据采集\paperA'
$env:PYTHONUTF8 = '1'
$env:PYTHONNOUSERSITE = '1'
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$envs = Get-Content config/environments.json -Raw | ConvertFrom-Json
$collect = $envs.collection_python
$train = $envs.training_python
$task = 'tray_to_platform'
```

`$collect` 是已有 mujoco_repro：Flask、LeRobot、MuJoCo 和采集渲染。`$train` 是已有 paperA_train：PyTorch/CUDA、PyAV、PyArrow、TensorBoard 和训练/推理。不要在基础 Anaconda 环境里运行。Node 也从同一环境 JSON 读取。依赖以`requirements.txt`、`training/papera/requirements-training.txt`及下方DP依赖命令为准；`training/papera/environment.yml`提供Conda训练环境规格，DP依赖另行补装。不需要每次运行都安装。旧本机lock快照已移出当前工程，不能作为新版完整安装清单。

当前 DP 直接使用固定官方组件。首次配置或更新依赖时，在同一 `paperA_train` 中运行一次：

```powershell
& $train -m pip install -r training/diffusion/requirements.txt
& $train -m pip install --no-deps robomimic==0.2.0
```

第二条只安装用于视觉网络的 robomimic；本工程不用它的环境套件及 Linux EGL 探测包，所以在 Windows 上不安装这些无关依赖。requirements 中固定 diffusers 等实际运行版本，保留本机已有的 CUDA PyTorch。官方源码及许可证在 `training/diffusion/official/`，固定来源见其中 `SOURCE.json`。

VS Code：**打开 `paperA` 文件夹**，信任本地工程并启用 Python/Python Debugger 扩展；F5下拉仅保留ACT训练、DP训练、ACT手势辅助训练、DP手势辅助训练四项。按提示输入task_id和Pack绝对路径；恢复与Rollout使用下文终端命令。环境路径修改后重新生成快捷配置：

```powershell
& $train scripts/configure_vscode.py
```

它只更新 `.vscode`，不启动训练。ACT Baseline/Gesture和DP Baseline快捷项当前最大预算为60,000次更新，DP Gesture仍为40,000次，分别读取各自训练配置；未来DP手势公平对照须先同步预算及选型协议，本轮未改未训Gesture。正式条件统一为单oblique RGB 640×480、30Hz模型动作与K30；ACT观察1帧、DP观察2帧。物理仿真500Hz，控制器60Hz，每个模型动作目标执行2个控制tick。

## 1. 采集

```powershell
& $collect -m collection list
& $collect -m collection start --task $task --operator-id operator01
# 需要复现实验初态时，显式指定种子：
& $collect -m collection start --task $task --operator-id operator01 --seed 12345
```

启动器打开 MuJoCo 和 `http://127.0.0.1:8014/dual_arm.html`。摄像头在网页点击启动后授权，双掌绑定后使用现有 100 ms 手势确认策略。保存成功、保存失败和放弃本轮仍按原 Recorder 区分。保存视频需要等待完成；不要强关写视频的进程，否则 MP4 可能未封装完成。

保存视角统一在 `config/recording.json` 的 `cameras` 设置为 `["oblique"]`：先将完整物理快照保存，再在保存本轮时仅渲染模型输入视角。可直接播放 `datasets/<task_id>/<会话>/episode_videos/episode_000000/oblique.mp4`（后续示范编号递增）。State、Action、原始手势、控制日志、全部物理快照和逐帧映射仍完整保留。当前网页展示 top/front/oblique 三路（一主画面+两辅助，可切换主视图），自动模式XY看top、Z看front，也可手动选择oblique；当前页面的斜视（side）统一使用 oblique 命名。oblique 是唯一录制与模型输入视角，top/front 只辅助观察、不录制；oblique 与模型输入相机参数及不透明外观一致（网页为压缩预览）。保存仍需完成一遍渲染和编码，不是瞬间完成。

网页三个视图中仅 oblique 保存为模型视频。独立录制选择会写入每次会话的冻结配置、配置指纹和相机清单；新配置不覆盖、删除或改写旧视频。单视角原件不会在后台自动补生成辅助视角视频；目前没有一键补生成入口，后续补图需要使用原记录的快照和冻结场景核验后重放。

参数：`--task` 选择已注册场景；`--seed` 固定初态，省略则每次新启动随机；`--operator-id` 写入身份来源；`--headless` 不弹 MuJoCo；`--no-browser` 不自动打开网页；`--web-port/--backend-port` 改服务端口。服务冲突明确报错，不关闭其他工程服务。

```powershell
Get-ChildItem "datasets/$task"
Get-ChildItem "logs/$task"
```

Raw 保存到 `datasets/<task_id>/<会话目录>`；失败保存到同名 `_failures`；研究日志保留失败/放弃和事件。内部 session_id、episode_index、scene_id、seed 以 manifest/metadata 为准。终端 Ctrl+C 会请求闲置后端正常退出；录制未处理时先在网页保存或放弃。

## 2. Audit、自动裁剪和审核

双击 `dataset/PREVIEW-CURATION.cmd` 可进入裁剪预览菜单，先选择任务，再选择：1 打开 CURRENT 明确指定的结果；2 手动选择已有裁剪版本；3 调用现有 process 生成新版本后打开。选项1、2不会重新裁剪、修改 CURRENT 或保存人工审核。浏览器自动打开后，点击“生成／更新裁剪预览”再播放；默认展示原片，原片里的灰色帧仍然可见。只读预览直接使用已保存的 keep/原始帧映射，不按当前参数重新执行 Safe Stitch，临时预览视频不属于 Training Pack。若场景后来改变，仍可查看历史结果，但正式 prepare/pack 的来源一致性校验不放宽。

启动器从 `config/environments.json` 读取 `training_python` 和 `review_port`，默认端口8026；菜单发现占用后会尝试后续20个端口，显示实际地址，不关闭或替换旧服务。终端保持打开，Ctrl+C退出。只有遇到原视频ACL拒绝读取时，才需要关闭自己这次的服务、右键CMD选择管理员运行；管理员不会替换旧服务。所有旧原片、裁剪索引和历史结果保持不变。菜单选项3仍严格核验任务来源；旧数据场景与当前配置不同会明确报错，不会自动混用。

```powershell
# 固定任务打开菜单，也可指定预览端口
.\dataset\PREVIEW-CURATION.cmd -Task tray_to_platform -Port 8026
# 不经过菜单：读取该任务 CURRENT；--run 可显式指定另一个裁剪目录
& $train dataset/manage_dataset.py --task $task preview
& $train dataset/manage_dataset.py --task $task preview --run $curation --port 8027
```

```powershell
& $train dataset/manage_dataset.py --task $task locations
& $train dataset/manage_dataset.py --task $task process --workers 2
& $train dataset/manage_dataset.py --task $task list
$dataRoot = "outputs/dataset/$task"
$curation = (Get-Content -LiteralPath "$dataRoot/CURATION_CURRENT.json" -Raw -Encoding UTF8 | ConvertFrom-Json).selected_version
& $train dataset/manage_dataset.py --task $task review --run $curation
```

`process` 只读取该任务已保存成功的真人原件：完整性/视频解码 → 原始事实证据 → 既有自动裁剪与 Safe Stitch。输出分别在 `audits/` 和 `curation/`；源文件指纹不变时复用完整审计。`--fresh` 强制重查原件，`--config` 指向另一份合法裁剪参数文件，每次都产生独立结果。没有可处理数据会报错，不退回旧工程。

同一任务的历史采集如果只调整了握拳／张掌识别门槛，可显式使用下列兼容入口。它核对每条冻结配置，只允许 `gesture_control.close_score_min`、`open_score_min` 和可选的 `left_close_score_min` 不同；场景、相机、物理、动作契约和其余配置仍须一致。每组分别调用原有 Audit 和裁剪流程，保留原始配置、来源与兼容说明，不修改原件或裁剪规则。未加此参数时仍执行原有严格检查。

```powershell
& $train dataset/manage_dataset.py --task $task process --recorded-collection --workers 2
$versions = (Get-Content -LiteralPath "$dataRoot/CURATION_CURRENT.json" -Raw -Encoding UTF8 | ConvertFrom-Json).recent_versions
$versions # 每个路径是一组冻结采集配置的裁剪结果；不要只查看最后一组
& $train dataset/manage_dataset.py --task $task preview --run $versions[0] --port 8026
# 前一个命令持续运行服务；如需同时查看另一组，在另一个已设置 $train/$task 的终端运行：
# & $train dataset/manage_dataset.py --task $task preview --run '<另一组完整路径>' --port 8027
```

这个入口不依赖采集者姓名、日期或固定示范编号。它保留每组冻结来源；后续联合导出显式使用 `prepare --recorded-collection`，核对真实metadata后形成同一学习任务的Prepared，原来源绑定仍逐条保留。未提供兼容参数的旧单源流程仍严格检查。

同伴应发送整个原生会话目录：`data/`、`meta/`、`episode_videos/`、`research/`（含快照及episode_metadata），不能只发视频或缺研究日志的HF merged导出。相同任务、场景/模型相机、Schema与受支持的采集版本才能自动接入；操作者、seed、会话数量和原件所在盘符可不同。若不想先复制到本工程，可直接读取外部根目录：

```powershell
& $train dataset/manage_dataset.py --task $task process --source 'E:\同伴采集\完整会话根目录' --recorded-collection --workers 2
```

程序只读外部原件，仍将结果写到本工程 `outputs/dataset/<task>/`。路径、来源差异或缺文件错误会明确报告，不按采集者添加专用脚本，也不会自动补造丢失的控制记录。

完整会话应保留程序生成的原目录名。重复episode key或同名目录指向不同真实会话会报错，不覆盖、不自动改名。相同输入使用稳定选择目录，复用前检查别名目标，允许既有Audit正常命中；不重定向旧目录。若仅 `config/collection.json` 的排版/换行令文件指纹不同，须用完整冻结配置和实际metadata证明语义一致，不能凭空diff标记放行；场景、模型相机或字段变化仍拒绝。

审核网页默认 8026。新单视角记录显示 oblique，并可生成对应裁剪预览；历史含 oblique 的三/四视角记录显示 top/front/oblique；只有旧 top/front/side 原件仍显示旧三视角。单视角存在遮挡时，不要仅凭看不清就认定抓取正确，应结合物理/接触证据。正式模型始终只有 oblique；采集网页的其他视角仅辅助实时观察，不作为新录制视频或训练输入。原片、逐帧 a/s/手势、删留原因、保护区间、裁剪预览和人工覆盖保留。灰色帧在原片中仍然可见；只有“生成/更新裁剪预览”播放删留后的顺序。预览不自动批准数据，人工排除保护段仍需理由。

调参数后不必重新复制视频，可从明确的 Audit 重建：

```powershell
$audit = '<实际 outputs/dataset/任务/audits/evidence_... 目录>'
& $train dataset/manage_dataset.py --task $task reclip --audit $audit --config config/curation.json
```

同意图静止核心默认保留 3 帧、换意图核心保留 10 帧；事件及物理运动保护可能使总保留量更大。 当前从前向后分开识别稳定区间，恢复运动不会否定之前的静止。快速回中最低0.2秒（`waits.minimum_recenter_s`），普通同意图等待仍2秒，换意图也为0.2秒（`waits.minimum_transition_s`），仍保留10帧预算，不足10帧的区间不删帧。`physical_motion.position_m` 为0.0005米即0.5毫米，容忍微小持物漂移；速度、关节、开合和接触保护不变。若仅拼接两端速度超限，先补回有限边缘帧重新核验，通过才保留中间压缩，否则恢复原帧。旧结果不自动更新，需重新生成；新结果待人工审核。详细阈值的中文说明在 `config/curation.json`。迁移没有重写这些算法；仅移除旧数据特定 Episode 的排除名单，新任务名单为空。

## 3. Prepared 与 Training Pack

```powershell
# auto：使用自动建议，不冒充已人工审核
& $train dataset/manage_dataset.py --task $task prepare --run $curation --review-mode auto
# 如已逐条人工审核并按规则批准，可改用 --review-mode reviewed；两者分别导出新版本。
$prepared = (Get-Content -LiteralPath "$dataRoot/PREPARED_CURRENT.json" -Raw -Encoding UTF8 | ConvertFrom-Json).path
& $train dataset/manage_dataset.py --task $task pack --prepared $prepared
$pack = (Get-Content -LiteralPath "$dataRoot/TRAINING_PACK_CURRENT.json" -Raw -Encoding UTF8 | ConvertFrom-Json).path
& $train dataset/manage_dataset.py --task $task verify-pack --pack $pack
```

多个Curation来源用同一个prepare入口（单来源原命令继续适用）：

```powershell
$task = 'tray_to_platform'
$dataRoot = "outputs/dataset/$task"
$curations = (Get-Content "$dataRoot/CURATION_CURRENT.json" -Raw -Encoding UTF8 | ConvertFrom-Json).recent_versions
$choice = Get-Content "$dataRoot/SELECTION_CURRENT.json" -Raw -Encoding UTF8 | ConvertFrom-Json
# 本轮沿用已固定的49/15 split；新实验可提供另一份事先确定的完整会话split
& $train dataset/manage_dataset.py --task $task prepare --run $curations --recorded-collection --review-mode auto --split-file $choice.split --workers 2
$prepared = (Get-Content "$dataRoot/PREPARED_CURRENT.json" -Raw -Encoding UTF8 | ConvertFrom-Json).path
& $train dataset/manage_dataset.py --task $task pack --prepared $prepared --split-file $choice.split
$pack = (Get-Content "$dataRoot/TRAINING_PACK_CURRENT.json" -Raw -Encoding UTF8 | ConvertFrom-Json).path
```

`--run`可接一个或多个明确路径，不能把同一批数据的两个重复裁剪版本同时合并。prepare的`--workers 1..4`只控制独立来源导出的并行数，默认2；`--output`可指定一个不存在的新联合Prepared目录。每组原Prepared在`prepared_sources/`，给Pack使用的联合导出在`prepared/`；RGB只在`training_packs/`生成缓存，不复制原视频。联合来源声明在构建时核实，同伴姓名/会话数量不写死。固定split若不提供，普通Pack沿用既有整会话规则；本轮已给定split，因此不能静默替换。

Prepared 保存保留索引、原时间和标签；Pack 解码/cache oblique 并固化数组、虚拟序列、Mask、来源和 split。它不会再次删帧。正式输入来自 **Pack 目录**，不能直接给模型 Raw 或 Prepared。

默认按整个原始会话划分：种子 20261016，约 20% 会话作为验证，至少一个；因此至少需要两个独立采集会话。不是把同一示范的相邻窗口随机分到训练和验证。需要人工固定 split 时传 `--split-file <JSON>`，字段为 `train_sessions`、`val_sessions`（可参考自动生成的 `splits/*.json`）。

2026-11-03 本轮选样：明确排除 `20261103_182433_791007` 的 episode 2、3、4、8、11、12、13、15。理由保存在既有 Curation 的正式 `reviews.json`（`rejected`，不伪称完整观看），并加入 `config/curation.json → quality.excluded_episodes`，防止后续重裁剪时重新成为候选。原始文件和自动裁剪证据保留。仅修改当前配置不会追改旧 Curation，所以两处排除都已写入。

保留64条、60,035帧，按原有整会话规则固定49条训练/15条验证（验证会话 `20261103_144145_400130`）。`SELECTION_CURRENT.json` 导航到带逐条来源、排除理由、两组原始绑定及配置快照的选样清单和同一份 `split.json`：

```powershell
$task = 'tray_to_platform'
$dataRoot = "outputs/dataset/$task"
$chosen = Get-Content "$dataRoot/SELECTION_CURRENT.json" -Raw -Encoding UTF8 | ConvertFrom-Json
Get-Content $chosen.selection -Raw -Encoding UTF8 | ConvertFrom-Json | Select-Object included_episodes,excluded_episodes,retained_frames,is_training_pack
Get-Content $chosen.split -Raw -Encoding UTF8
```

这份清单不是Training Pack，不可传给 `--pack`。两组原绑定由联合Prepared/Pack完整保留；模型实际只接收 `TRAINING_PACK_CURRENT.json.path` 指向的一个正式Pack。ACT和DP在该Pack同一训练部分各拟合自己的归一化，不分别对两组拟合后拼接。各个有效单源Pack的命令继续适用。

本轮实际生成的联合Prepared为 `outputs/dataset/tray_to_platform/prepared/prepared_20261103_211127_436073`；联合Pack为 `outputs/dataset/tray_to_platform/training_packs/auto_20261103_213247_870817`。64条、60,035帧全部构建完成，`complete=true`、`data_ready=true`，CURRENT已指向该Pack。`review_mode=auto`、`human_reviewed=false`，所以`training_ready=false`仍提示未全量人工质量批准，不表示缓存构建失败；现有ACT的`allow_exploratory=true`与DP入口允许用它进行本轮数据可用性Baseline实验。结构检查通过不保证示范质量或任务成功率。

构建中断时，使用该次明确的 Prepared/output：

```powershell
$partialPack = '<该次 training_packs/... 的实际目录>'
& $train dataset/manage_dataset.py --task $task pack --prepared $prepared --output $partialPack --resume
```

会复用 `build.json` 中冻结的 split；源、代码、split 或图像规格改变将拒绝恢复。已有完整 Pack 不覆盖；修改裁剪后构建新 Pack 并明确选择。

## 4. ACT 公平训练两组

本节为当前官方核心版本的使用入口。ACT无需新增Python依赖，复用已有Torch/Torchvision和本地ResNet18权重；不依赖临时复现目录或IPython。旧本地版本权重不加载到新结构。

源码职责与复用边界：`training/papera/official/`是固定上游组件；`act_model.py`负责图像与State/Action维度适配并调用原模型，`engine.py`执行共用的前向、Action L1＋KL、反向和优化更新；`pack_data.py`消费Dataset已认证的虚拟序列，`data.py`负责训练集归一化，`train.py`组织训练、验证、完整保存和恢复。Baseline不使用手势辅助头。不为提高本任务成功率修改官方Transformer、CVAE、Loss或Dataset拼接判定。

“官方核心一致”不等于原始训练脚本完全相同：原脚本按示范选随机起点，本地按Pack中的合法观察窗口均匀采样；本地归一化仅拟合训练集，正式离线验证用`z=0`，完整保存优化器/采样器/RNG。这些是已存在的数据与运行协议适配，必须随实验披露，不能把本地结果冒称官方原任务复现。`--official-reference`专用入口另行恢复原始采样和后验验证，只有该对照可以直接与其官方脚本实验比较。本轮不添加事件权重、阶段采样或新Loss。

队友的单臂任务可以复用这些官方组件及更新循环，但**当前Panda入口不是只改一个维度JSON就能运行的通用单臂入口**。须先提供真实的State/Action字段、单位、相机和频率，再增加明确的任务数据适配：Pack契约、归一化角度列、物理诊断指标和Rollout执行器均需对应单臂。尤其上游DETRVAE的后验`encoder_action_proj`初始写为14维；当前A14无需替换，单臂A7则须在外层适配该投影和两个State投影，不能只改输出头，也不需要改vendor的计算结构。不要补零伪造另一条机械臂或复用双臂统计量。单臂适配完成后，先验收真实batch、动作窗口/Mask、前向反向、保存恢复及其自身执行器，再开始正式实验；本轮双臂结果不代表单臂已实现或验收。

先完成ACT和DP Baseline的工程与闭环验证，再做各自Gesture对照。以下命令会启动长训练；一般在同一GPU上顺序运行。2026-11-04已在本机实测两组Gesture并行训练的显存及内存余量后安排并行，未改变batch、精度或更新预算。实验固定辅助权重0.1，必须显式选择Gesture配置；只传 `--variant gesture` 而沿用ACT Baseline配置，会回落到历史默认权重1。

先确认 `$pack` 是本任务的正式真人数据，运行 verify-pack 后保留这个值。两组用 **同一个 Pack、split、seed、训练预算和主干配置**：

```powershell
& $train -m training.papera --task $task --variant baseline --pack $pack
& $train -m training.papera --task $task --variant gesture --config config/training_act_gesture.json --pack $pack
```

这是两条长训练命令，由你选择时间执行；本轮迁移没有执行它们。输出自动为 `outputs/training/<task_id>/<variant>/<run_id>`。

默认参数在 `config/training_baseline.json`：单oblique、640×480全画面、S16/A14、K30、batch16、最大60,000次更新、FP32。主干和其余参数均采用AdamW学习率1e-5、weight decay1e-4；`lr_scheduler="constant"`、`warmup_steps=0`，`gradient_clip_norm=null`表示不裁剪，只检查梯度有限。Transformer hidden512、FF3200、encoder4、decoder7、posterior4、heads8、latent32、dropout0.1；KL权重10。ResNet18从已有本地预训练文件加载，不自动下载。图像不裁剪，不改变Pack；模型动作30Hz、控制60Hz、物理500Hz。`--steps`改最大更新预算、`--seed`改种子、`--config`换参数文件；同一算法两组须同步。60k是训练上限，不表示必须部署60k或`best.pt`；保留的中间权重需要在独立闭环验证上选型，再冻结进行新测试。

当前ACT直接复用固定官方提交 `742c753c0d4a5d87076c8f69e5628c79a8cc5488` 的DETRVAE、Post-LN/ReLU Transformer、位置编码和CVAE，来源与两份许可证见 `training/papera/official/SOURCE.json`。只对两处状态输入投影适配S16，Action仍14维，K30与单相机由配置给出；不保留旧自写观察Transformer。该官方版本的DETRVAE取第一个intermediate decoder输出，本工程保留这一行为，没有改成最后层。

Baseline优化目标为官方Action L1 + 10×KL：无效Action误差置零后，L1按整个B×K×A坐标数取均值。`train/action`是该官方优化项，`train/action_valid_l1`另按有效坐标计。验证固定z=0，`prior_action_l1`与`prior_valid_action_l1`都表示有效坐标L1，作为best指标；`official_action_l1`另报告全分母值。两种分母不能混作同名曲线。`metrics.jsonl`、TensorBoard、best/last和完整恢复状态继续保留；离线Loss不是自主成功率。

2026-11-05工程验收：主回归617项通过、4项按既有条件跳过，另2项绘图兼容回归通过；独立加载固定官方源码后，训练前向、Loss及核心梯度数值一致。真实Pack在RTX5090上通过640×480、batch16、FP32的两次更新与保存恢复，峰值分配显存约4.04GiB；两条新建scripted_test记录通过Dataset、3步短训及CUDA短Rollout，视频完整解码61帧。这些只验证工程通路，未运行正式40,000步训练，也未产生新版模型成功率。证据集中在`.local/checks/act_official_restore_20261105/`。

```powershell
$run = '<实际 outputs/training/任务/baseline/run_id>'
Get-Content "$run/completion.json"
Get-Content "$run/metrics.jsonl" -Tail 5
& $train -m tensorboard.main --logdir "$run/tensorboard" --port 6006
```

更完整离线分析：

```powershell
& $train -m training.papera.analyze_checkpoint --run $run
& $collect -m training.papera.plot_training_analysis --run $run
```

分别输出 `run/analysis/evaluation` 和 `run/analysis/figures`。绘图使用已有安装 matplotlib 的采集环境；离线动作误差不是闭环成功率。重复分析要给新的 `--output`，不覆盖已有证据。

### 4.1 Diffusion Policy Baseline / Gesture

DP 复用同一任务的正式 Training Pack、会话 split、S16/A14 和 Safe Stitch 结果；训练入口不读取 Raw，不补日志，也不接收缺失控制/来源信息的 HF merged 导出。先只读检查张量和数据接口，再自行启动正式训练：

```powershell
& $train -m training.diffusion --task $task --pack $pack --check-only
& $train -m training.diffusion --task $task --pack $pack
# 独立的手势辅助对照；从相同初始化开始，不接着Baseline权重训练
& $train -m training.diffusion --task $task --variant gesture --config config/training_diffusion_gesture.json --pack $pack
```

默认配置为 `config/training_diffusion.json`：单 oblique RGB 640×480、过去至当前2帧观察（`model.observation_horizon=2`）与每帧S16，输出未来K30/A14，模型动作30Hz。模型内部先把RGB映射到 `[-1,1]`，按 `model.crop_shape=[472,632]`（高、宽）在训练时随机裁剪、验证与推理时中心裁剪；Raw和Pack仍保留640×480，不重新编码或裁剪数据原件。裁剪大小可通过配置调整，须保证托盘、抓柄和高台仍可见，设 `null` 表示不裁剪。

当前版本直接复用[固定官方实现](https://github.com/real-stanford/diffusion_policy/tree/5ba07ac6661db573af695b419a7947ecb704690f)的 U-Net、FiLM、skip连接、DDPM与EMA，并使用相同的 robomimic VisualCore：ResNet18，GN按 `channels//16` 分组，32个SpatialSoftmax关键点、64维视觉特征及ReLU。U-Net通道 `[512,1024,2048]`、卷积核5、时间嵌入128。默认100级cosine加噪，以epsilon MSE训练，预测用官方100步DDPM。`action_horizon=8` 表示每8个30Hz目标重新规划。内部将长度30补齐32后调用原U-Net，再截回30，只解决网络长度整除；不移动动作标签，也不产生两条额外训练目标。

Pack读取、train-only min-max、RPY周期处理、认证Safe Stitch和硬边界Mask不变；无效目标不计入Loss，未来动作仍从当前观察后的下一合法控制tick开始，执行索引0～7，不能照搬官方含过去动作序列的 `To-1` 偏移。无效目标先清除非有限值再整体加噪，去噪输入不再把无效后缀强制清零。Baseline不使用手势标签；Gesture只增加观察条件上的左右分类头和0.1倍CE，推理仍无手势输入。

ACT与DP Baseline使用相同640×480、K30、batch16和最大60,000次更新；DP本轮从头训练，结果待完成。ACT保持1帧观察，DP使用2帧，网络和计算量也不同，不能称输入信息或算力预算完全相同。DP Gesture配置仍保留40k，本轮未改未训；未来同一算法内Baseline/Gesture必须用相同数据、split、相机、历史、预算和对称选型协议。以后的历史、分辨率、宽度或预算变化都作为新实验记录。

Baseline默认 AdamW 学习率1e-4、betas(.95,.999)、weight decay1e-6、batch16、60,000次更新、warmup500、FP32。学习率使用diffusers官方余弦调度，最终降至0；首个warmup更新的LR为0。`gradient_clip_norm=null` 为官方默认不裁剪，只检查梯度有限性。EMA直接使用官方幂次预热，默认power0.75、inv_gamma1、上限0.9999。每25步记日志，每1000步验证/保存，最终或请求停止时也保存。`best.pt` 依据固定噪声与timestep的**完整验证集EMA noise MSE**选择；`sample_action_l1_normalized`和物理动作MAE只在前2批、共32个锚点生成DDPM动作后统计，是局部诊断，不是全验证集误差。离线best与采样误差均不直接决定自主成功率，正式权重需要独立闭环验证选型。

`candidate_save_every=5000`每5k及完整预算末步保留`checkpoints/step_NNNNNN.pt`，包含模型、EMA、优化器、scheduler、sampler与RNG；设`null`关闭，不额外触发验证。`checkpoints/candidates.json`绑定run/task/config/source、文件尺寸、真实保存步及同一步验证（未在该步验证则为null），不能从任意.pt文件名推断候选。训练先提交`last.pt`再写候选，恢复时撤回超出恢复步的索引、不删除原文件；遇到已有孤立同名候选会拒绝覆盖。历史best/last无需新索引，继续只读兼容。

DP 常用配置字段如下；ACT 的对应公共条件在 `config/training_baseline.json`，对照时一起核对。

| JSON 字段 | 当前默认 | 调整作用 |
|---|---:|---|
| `steps` / `batch_size` | 60000 / 16 | Baseline优化器更新总数 / 每次使用的样本窗口数；Gesture历史配置仍40000 |
| `candidate_save_every` | 5000 | 每N步及最终步保留完整候选；`null`关闭，不改变验证频率 |
| `image_size` | `[640,480]` | Pack/RGB输入宽、高，单位像素；模型内部另按crop_shape处理 |
| `model.crop_shape` | `[472,632]` | 裁剪高、宽；训练随机，验证/推理中心；`null`禁用，不改Raw/Pack |
| `model.observation_horizon` | 2 | 同一摄像头过去至当前的2个合法观察锚点；不增加相机，不移动当前动作的因果起点 |
| `model.down_dims` / `model.kernel_size` | `[512,1024,2048]` / 5 | 固定官方视觉配置的U-Net通道与卷积核 |
| `model.feature_dim` | 64 | 原视觉编码器投影维数；GN和FiLM计算保留官方实现 |
| `model.prediction_horizon` | 30 | 预测未来动作数量，本工程正式Pack接口使用K30 |
| `model.action_horizon` | 8 | 后续闭环滚动执行时计划取用的前8个目标；当前训练仍监督全部有效K30 |
| `model.diffusion_steps` | 100 | 从干净动作到噪声的100级扩散时间表 |
| `model.inference_steps` | 100 | 每次生成动作块使用100次官方DDPM去噪计算 |
| `ema` | 官方配置对象 | `update_after_step=0, inv_gamma=1, power=.75, min_value=0, max_value=.9999`；定义EMA预热及上下限 |
| `gradient_clip_norm` | `null` | 不裁剪；如单独实验设正数须记录，不能冒称官方默认 |
| `validation_batches` | `null` | 检查完整验证集；设正整数可限制为前N批工程快速检查 |
| `validation_sample_batches` | 2 | 验证时额外生成动作并统计误差的批数，设0可只统计噪声MSE |
| `num_workers` / `precision` | 2 / `fp32` | Windows异步数据读取进程数 / CUDA数值精度 |

2026-11-05的DP工程验收通过模型同权重对照、EMA/LR与恢复检查：完整回归608项通过、4项跳过；真实Pack上640×480/H2、batch16、FP32完成两次GPU工程更新及状态恢复，峰值显存约9.15GiB。两条工程记录也跑通Dataset→短训→CUDA Rollout；这些仅证明工程通路。证据在 `.local/checks/dp_official_restore_20261105/`。随后官方40k运行的17k best为8/50，历史本地旧DP为17/50，两者保留但不当作当前fresh60k成绩。当前实验的训练与评估状态见第6.5节。

| 参数 | 含义 |
|---|---|
| `--config <JSON>` | 替换默认 DP 配置；新运行时使用 |
| `--steps N` | 覆盖新运行的总更新预算，不是 epoch 数 |
| `--output <新目录>` | 指定新运行目录；已有目录拒绝覆盖 |
| `--check-only` | 只检查 Pack 和样本张量，不创建训练 run |
| `--full-verify` | 可选慢检查，额外重算文件哈希并核对当前任务源码来源 |
| `--stop-after N` | 本次额外运行N次更新后保存退出，冻结的总预算保持不变 |
| `--engineering` | 明确允许 scripted_test 工程 Pack；输出隔离在 `.local/checks` |

默认只做必要的 Pack 格式、文件尺寸、冻结身份、split、因果标签和合法序列接口检查，不重复重算全部大文件哈希，不访问外部 Raw/Prepared；这不等于完整内容审计。需要严格审计时显式使用 `--full-verify`。正常训练输出 `outputs/training/<task>/diffusion/<run_id>/`，含 `config.json`、`data_binding.json`、`run.json`、`metrics.jsonl`、`tensorboard/`、`checkpoints/{best,last}.pt`、已启用的候选及索引、`completion.json`。

```powershell
$dpRun = '<实际 outputs/training/任务/diffusion/run_id>'
Get-Content "$dpRun/completion.json"
Get-Content "$dpRun/metrics.jsonl" -Tail 5
& $train -m tensorboard.main --logdir "$dpRun/tensorboard" --port 6006
```

现有 `training.papera.analyze_checkpoint` 与ACT绘图不支持DP checkpoint。统一 `rollout` 已按run算法识别DP，加载EMA best/last及其冻结min-max归一化，并逐30Hz观察更新H2历史；DP必须明确使用 `config/evaluation_diffusion.json`，按chunked/8执行。不能将DP配置套到ACT，或把DP原始model权重当EMA best。DP物理执行器及任务成功判据与ACT共用。

### 4.2 历史记录：2026-11-03的64条选样与旧模型参数

本节保留2026-11-03旧ACT/DP本地变体的参数及当时依据，用于解释历史结果，**不是当前新训练默认值**。当前ACT官方核心/FP32/常量LR以第4节为准，DP以第4.1节为准；历史权重和冻结配置未改写。

2026-11-03准备阶段的调整发生在数据质量决定及正式JSON参数：排除指定8条；DP历史1→2，U-Net通道 `[128,256,512]`→`[256,512,1024]`。ACT配置数值不变，不修改两个模型的损失和训练数学实现，不启动正式训练。两份正式参数文件为 `config/training_baseline.json`、`config/training_diffusion.json`；选样目录同时保存它们的快照，不另造一套训练入口。旧checkpoint恢复使用其中冻结的配置，不能拿旧DP权重冒充本轮H2/新宽度模型。

| 条件 | ACT Baseline / ACT Gesture主干 | DP Baseline / DP Gesture主干 |
|---|---|---|
| 摄像头、图像 | 单oblique，640×480整幅 | 相同 |
| State / Action | 实测S16；末端XYZ+RPY+米制爪宽A14 | 相同 |
| RGB/State/模型动作 | 30Hz；控制60Hz；物理500Hz | 相同 |
| 观察历史 / 未来 | 当前1帧 / K30 | 过去至当前2帧 / 未来30；第0项仍是当前观察后下一合法tick |
| 训练预算 | 40,000次参数更新，batch16 | 相同；非40,000个epoch |
| seed / 精度 | 20261016 / BF16 autocast | 相同；关键归一化、loss和DDIM算术保留FP32路径 |
| Backbone | 本地ImageNet预训练ResNet18 | 本地同源ResNet18，BN换为GN，32关键点SpatialSoftmax，视觉投影128 |
| 主体 | hidden256、FFN1024、encoder/posterior/decoder各4层、heads8、latent32、dropout0.1 | FiLM条件1D U-Net，down256/512/1024、kernel3、GN最多8组、time_dim128 |
| 优化器 | AdamW；主模型1e-4 / backbone1e-5；WD1e-4；默认betas(.9,.999) | AdamW；LR1e-4；WD1e-6；betas(.95,.999) |
| LR / 梯度裁剪 | warmup500 + cosine，末值为峰值10%；clip1 | 相同 |
| 归一化 / loss | train-only z-score；有效动作L1 + KL10 | train-only min-max；有效动作坐标的epsilon-MSE；100步cosine加噪 |
| EMA / 推理 | 不使用EMA；z=0 | EMA预热、上限.999；DDIM10、eta0；规划执行前8步，约0.267秒后重规划（闭环入口待接） |
| 验证 / 保存 / 日志 | 每1000 / 1000 / 25次更新 | 相同；workers2，ACT当前workers0 |
| best选择 | 全验证集prior动作L1 | 全验证集固定噪声EMA noise-MSE；前2批DDIM动作指标只作诊断 |

选择依据和边界：

- [原DP论文v5](https://arxiv.org/html/2303.04137v5)附录B.1支持多数视觉任务使用2帧历史；附录A.4既有84×84仿真图，也有320×240真实图，不存在必须128×128的统一规则。本任务要同时看双臂抓柄、托盘和高台，先保持原4:3全画面和已有640×480契约；直接缩成128×128会改变比例。以后降分辨率应保持比例，并对所有同模型对照同步改变。
- `[256,512,1024]`、kernel3来自[作者ConditionalUnet1D类默认](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conditional_unet1d.py)，不是作者所有视觉实验的唯一默认：[视觉YAML](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/config/train_diffusion_unet_image_workspace.yaml)采用更宽的512/1024/2048、kernel5。本工程仍是适配实现；本轮从工程轻量版回到有源码依据的中等容量，不宣称已证明比小模型更好。
- 同一视觉YAML支持LR1e-4、AdamW betas(.95,.999)、WD1e-6、500步warmup与EMA+GroupNorm。我们的EMA预热曲线与作者幂律曲线不同，上限.999也非作者模板的.9999，作为本轮固定工程设置披露；不能仅因朋友关闭EMA就照关，也不能将clip1称为论文唯一规定。
- 作者论文正文§3.4描述100步训练/10步DDIM推理，附录又有16步真实推理和100步仿真推理。故本轮固定10作为实时推理起点，不代表它在此任务已最优；用同checkpoint比较更多去噪步属于后续执行参数实验。
- 朋友100,000×batch4约呈现40万窗口，本轮40,000×16约64万，不能只看steps断言谁训练更多。其log100×4与本轮log25×16都约每400窗口记录；其val5000×4与本轮val1000×16也接近。GPU耗时、有效非padding监督与更新次数仍不同，不能称总算力相同。
- seed1000与20261016没有质量高低。主运行固定20261016；正式论文建议预先声明多个训练seed（例如20261016/17/18），全部报告均值和波动，固定同一份split；DP训练seed在其JSON中设置。
- ACT-H1与DP-H2对应各自配置，历史信息不同。Paper A的辅助监督增益应先在同一算法内比较Baseline/Gesture；如直接比较ACT与DP，应公开差异或补DP-H1。不能强制两算法采用相同loss、归一化和EMA来冒充公平。

本机RTX5090实测本轮DP配置的合成B16/H2/640×480前向、反向、参数更新、padding及DDIM10输出均通过，模型约54.34M参数；单次更新峰值分配显存约5.16GiB。该数字不包含完整训练所有附加开销，也不能证明模型收敛或任务成功。检查结果在 `.local/checks/experiment_freeze_20261103/dp_h2_gpu_check.json`。40,000步是首轮固定预算，不是收敛保证；最终仍须独立闭环验证，而非只比较ACT的L1与DP的noise-MSE。

## 5. 中断恢复与延长预算

```powershell
# 中断后：同一次运行、同一冻结预算，恢复最后完整保存步
& $train -m training.papera --task $task --resume "$run/checkpoints/last.pt"
# 原预算已完成，明确要延长：派生新 run，累计目标50,000步
& $train -m training.papera --task $task --continue-from "$run/checkpoints/last.pt" --steps 50000
```

当前ACT Checkpoint版本是 `papera-act-official-checkpoint-v2`。`--resume`只恢复同一新v2 run的模型、优化器、scheduler、RNG、sampler、归一化和数据来源，不接受新steps或Pack；`--continue-from`只从已完成的同结构、同variant v2 last派生新run并保持其当前学习率。旧 `papera-act-offline-checkpoint-v1` 属于不同本地网络，明确拒绝resume和continue-from，须在同Pack从头新训。旧文件只读保留，历史复现使用相应历史源码。

历史2026-11-04的旧ACT两组采用相同两阶段预算：先40,000步，再用continue-from累计60,000步。该轮后20,000步保持当时40k的学习率，不重启warmup、不改为60k余弦；这些完成记录和选型不是新版官方ACT已训练的证明。新Gesture应从原始初始化独立训练，不能从Baseline权重开始累计比较预算。

ACT 候选权重保留参数在 `config/training_baseline.json` 的 `candidate_save_every`：新训练默认每 **5000** 步及最终步额外保存完整的 `checkpoints/step_005000.pt` 等文件，设为 **0** 关闭。`checkpoints/candidates.json` 记录各候选的保存步数、同一步验证指标及验证来源；该步没有验证时指标为空，不触发额外验证。`best.pt` 仍按整块验证 prior Action L1 选择，`last.pt` 仍是最后完整的恢复点；候选不会自动替代它们，离线最优也不等于闭环成功率最高。

同结构v2 run缺少候选保存字段时按0处理，`--resume` 保持原冻结配置，不能临时开启或改变保存间隔。需要延长已完成的 Baseline 并保留候选时，用新的派生续训：

```powershell
& $train -m training.papera --task $task --continue-from "$run/checkpoints/last.pt" --steps 50000 --candidate-save-every 1000
```

这里50000是累计训练步数，1000是额外候选保存间隔，不是验证频率；父run、Pack、模型、Loss及父checkpoint学习率保持不变，结果写入新run。新建训练也可用 `--candidate-save-every N` 覆盖JSON配置。ACT Rollout额外支持 `--checkpoint step_045000` 这类已登记候选；必须存在同run的 `candidates.json` 和完整候选权重，不能指定任意外部文件。正式批量评估采用此候选时，配置必须写 `checkpoint_selection="fixed_retained_training_step"`、匹配的 `expected_checkpoint_step` 与 `checkpoint_selection_reason`，并在测试前冻结。

任务、Pack、split、归一化或未经审阅的源码变化会拒绝恢复。历史候选保存功能的窄范围源码兼容记录，不授权从旧本地网络迁移到当前官方网络。恢复不读取CURRENT，也不按文件夹时间选择权重。

### 5.1 DP 中断恢复

```powershell
& $train -m training.diffusion --task $task --resume "$dpRun/checkpoints/last.pt"
```

DP 恢复同一 run 中的模型与 EMA、优化器、scheduler、sampler、RNG、归一化、Pack路径、split 和预算；输入必须是 DP 的 `last.pt`。不同时传 `--config`、`--steps` 或 `--output`，不更换 Pack。可加 `--stop-after N` 限制本次更新数用于工程验证。默认恢复仍做格式和接口检查；`--full-verify` 可额外核验内容哈希。DP 本轮没有 `--continue-from` 延长预算入口。

已完成40k的cosine学习率为0，不能照搬ACT保持末端LR的续训方式。当前60k使用相同官方余弦调度从头新建run；总预算改变也改变前40k的学习率轨迹，因此不是纯粹多训练20k的单变量实验。旧40k权重和结果保持原样，不能修改其冻结预算后覆盖恢复。

当前Checkpoint契约为 `papera-diffusion-official-checkpoint-v2`，同时保存EMA权重及完整EMA配置、optimization_step和decay。旧 `papera-diffusion-checkpoint-v1` 对应不同网络，明确拒绝恢复或加载到新网络；不能拿旧last接着训练，应复用同一Pack从头新建run。旧成果只保留历史查看，若确需运行历史模型须使用相应历史源码；当前新源码不维护第二套旧模型流程。

## 6. 模型检查、预览与50次评估

本节共用入口支持ACT和DP各自的Baseline/Gesture。未指定config时使用任务默认ACT评估配置；DP需额外指定DP配置。原 `best.pt` 选择指标保持不变；独立验证可以选择明确的last或ACT/DP保留候选，但必须先冻结理由、步数和执行配置，再评估50次测试，不能从测试结果反选checkpoint。

DP推理可以显式指定 `diffusion_inference_steps`（去噪次数）与 `diffusion_action_horizon`（本次预测执行几个30Hz目标），改变默认值必须提供 `inference_selection_reason`；`n_action_steps` 要与动作执行长度一致。实际值和原训练默认值同时记录，不改历史checkpoint。正式用last时配置为 `checkpoint_selection="fixed_last_training_step"`；用已登记`step_NNNNNN`时为`fixed_retained_training_step`。两者均填写真实`expected_checkpoint_step`及选择理由，例如新60k last为60000，保留20k候选为20000，不能沿用旧40000。实际始终加载该checkpoint的EMA参数。对同一算法的Gesture对照使用相同执行设置。

```powershell
& $train -m rollout --task $task --mode check --run $run
& $train -m rollout --task $task --mode preview --run $run --seed 910001 --max-seconds 60
```

check 不开始仿真；preview 单次仿真并保存视频。默认评估配置 `preview_headless=true`，所以默认不弹 MuJoCo 窗口，可在任务指定的评估配置中改为 false 后重新冻结评估协议。视频在实际输出目录 `episodes/seed_.../oblique.mp4`；运行完成前 MP4 可能无法播放。实时进度看运行终端及 `status.json`。

预览 seed 不能与示范或固定50次测试种子重复。`--max-seconds` 是仿真时间，不是墙钟运行时长；批量评估的60秒预算固定在配置和协议，不能用命令临时覆盖。

真正训练完、确认推理接口正常后，建立固定清单，再评估两组：

```powershell
$protocol = "outputs/rollout/$task/protocols/comparison_01/protocol.json"
& $train -m rollout --task $task --mode prepare --pack $pack --protocol $protocol
& $train -m rollout --task $task --mode batch --run $run --protocol $protocol --workers 1
# 第二组使用同一协议与Pack，替换 run
$gestureRun = '<实际 Gesture run 目录>'
& $train -m rollout --task $task --mode batch --run $gestureRun --protocol $protocol --workers 1
```

prepare 只固定50个初态、任务配置、几何检查和代码指纹，不执行策略。batch 使用验证选择的 best，默认 seed1910000～1910049，每次最多60秒。`--workers 1..4` 是并行数量，不是任务次数；本机先用1。输出在 `outputs/rollout/<task>/<variant>/<evaluation_id>`，有独立 run/status/summary、每次物理轨迹、预测和视频。

```powershell
$evaluation = '<实际 outputs/rollout/任务/variant/evaluation_id>'
& $train -m rollout --task $task --mode summarize --output $evaluation
# 条件未改时继续未完成条目
& $train -m rollout --task $task --mode batch --run $run --protocol $protocol --resume $evaluation
```

`summary.json` 的 task_success_rate 是任务成功率，timeout、基础设施错误、干预比例分别看；部分完成的比例不当作正式50次结果。托盘正式评估仍采用原来 **每爪至少半开**（0.5）加真实抓取、抬升、脱离接触、到位、支撑、停稳等条件，不要求全开；迁移没有放宽其余物理判据。采集 DONE 的原有逻辑也没有在本轮修改。

DP手动评估使用自己的执行协议，50个seed、初态和物理成功条件须与ACT相同：

```powershell
$dpProtocol = "outputs/rollout/$task/protocols/dp_comparison_01/protocol.json"
& $train -m rollout --task $task --config config/evaluation_diffusion.json --mode prepare --pack $pack --protocol $dpProtocol
& $train -m rollout --task $task --config config/evaluation_diffusion.json --mode check --run $dpRun --checkpoint best --device cuda
& $train -m rollout --task $task --config config/evaluation_diffusion.json --mode batch --run $dpRun --checkpoint best --protocol $dpProtocol --device cuda --workers 1
```

### 6.1 顺序训练与各50次评估

这是一条长任务命令，只有准备好开始正式实验时运行。程序依次训练ACT、训练DP，再加载各自验证集选出的best评估，不加入手势监督；不更换数据、不重选测试seed、不修改成功判据。两个评估协议执行方式不同，但实际50个初态必须完全相同。每个模型每轮上限为60秒仿真时间；实际墙钟时间取决于推理、物理和视频编码。

```powershell
Set-Location 'D:\轻量化数据采集\paperA'
$train = (Get-Content config/environments.json -Raw -Encoding UTF8 | ConvertFrom-Json).training_python
$task = 'tray_to_platform'
$pack = (Get-Content "outputs/dataset/$task/TRAINING_PACK_CURRENT.json" -Raw -Encoding UTF8 | ConvertFrom-Json).path
& $train -u -X utf8 scripts/run_baseline_campaign.py --task $task --pack $pack
```

程序创建独立 `outputs/campaigns/<task>/baselines_<时间>/`，`plan.json`冻结Pack和配置，`configs/`保存参数快照，`status.json`记录当前阶段、PID、命令、日志和异常，`results.json`在评估完成后保存成功数/失败数/成功率、区间与实际模型查询的推理均值/P95。模型与50次评估目录仍分别放在既有 `outputs/training/`、`outputs/rollout/`。任一子进程、设备或数据检查失败会停止并写明错误，不把未完成批次当成最终成功率。

已有ACT训练时不要重复启动，可显式附着其真实PID：`--act-run <ACT目录> --act-pid <PID> --act-log <现有日志>`；程序核对该进程的模块、输出目录及冻结配置，等待它完成后再启动DP。若ACT已经完整完成，只需`--act-run <ACT目录>`。此入口不自动接管中断训练；中断请先按第5节恢复原run，避免重复占用GPU。

正式评估均显式用CUDA，先对各best做独立2秒preview核对NVIDIA推理和OpenGL设备、实际耗时，再运行固定50次。短preview不计入成功率；其任务timeout不代表设备错误。当前 `paperA_train/python.exe` 的Windows图形偏好已从省电改为高性能，新进程实测CUDA和OpenGL均为RTX5090；备份及前后测量在 `.local/checks/baseline_campaign_20261103/`。这只影响显卡选择，不改变数据或模型。

### 6.2 历史ACT执行策略对照（只读结果）

2026-11-04旧本地ACT曾用独立验证初态选择chunk30执行与40,000步last，随后50个新初态成功38次。它使用旧v1网络，不是当前官方核心的成绩。当前任务仍支持动作块与逐帧集成；新模型应按第6节明确冻结自己的执行参数，不能继承这次历史选型结论。

这次绑定旧run、Pack与固定last的一次性`evaluation_act_optimized.json`已从当前配置目录归档移出，旧文档和配置保存在工程外`paperA_版本备份/source_cleanup_20261105_213715/`。历史视频与`summary.json`仍在`outputs/rollout/tray_to_platform/baseline/optimized_last_chunk30_50_20261104/`，只读查看即可；历史重跑需对应历史源码。当前新训练和新评估按第4～6节，不再提供可误用于新版权重的旧启动命令。

### 6.3 当前官方 ACT Baseline：明确权重与执行参数

本轮使用同一份64条真人示范的Pack，保持49/15整会话划分，从已完成40k的官方结构继续至60k；没有修改ACT数学、裁剪、IK或成功判据。最大预算与最终部署步数分别记录。独立20个初态验证比较40/45/50/55/60k与两种执行方式，选中 **50k + chunk30**（17/20）；60k虽为离线best，并未作为最终部署模型。完整选型表、原始曲线、最终50例和视频集中在 `outputs/campaigns/tray_to_platform/act_baseline_finalization_20261106_174942/REPORT.html`，机器可读入口为 `outputs/ACT_BASELINE_CURRENT.json`。

冻结后的新测试初态7217000～7217049：**38/50成功（76%）**，7次超时、5次接触丢失，无基础设施错误；47/50真实双抓并抬升，50个视频完整解码。单次上限60秒；推理和渲染均为RTX5090。95% Wilson区间为62.6%～85.7%，仅针对这份权重的初态抽样。成功率包含既有物理保护执行器作用，不是无约束原始预测；没有据此宣称数据优于其他采集方式。只完成一个训练seed，数据仍为auto审核候选。

下面命令复用本轮已选模型，不会训练，也不会误加载35k `best.pt`。先逐行设置变量，`--run`是训练目录，`--checkpoint`是该目录已登记的候选名称；`--config`固定chunk30、60秒与任务判据，`--device cuda`明确GPU推理：

```powershell
Set-Location 'D:\轻量化数据采集\paperA'
$train = (Get-Content config/environments.json -Raw -Encoding UTF8 | ConvertFrom-Json).training_python
$act = Get-Content 'outputs/ACT_BASELINE_CURRENT.json' -Raw -Encoding UTF8 | ConvertFrom-Json
$task = $act.task_id
$run = $act.training_run
$evalConfig = $act.evaluation_config
$checkpoint = $act.checkpoint
$protocol = $act.final_protocol
& $train -m rollout --task $task --config $evalConfig --mode check --run $run --checkpoint $checkpoint --device cuda
# 单次独立预览；不是50次最终测试的一部分
& $train -m rollout --task $task --config $evalConfig --mode preview --run $run --checkpoint $checkpoint --device cuda --seed 910001 --max-seconds 60
# 如需重放已冻结的50例，创建一个新结果目录；不会覆盖本轮结果
$replay = 'outputs/rollout/tray_to_platform/baseline/my_fixed_baseline_replay'
& $train -m rollout --task $task --config $evalConfig --mode batch --run $run --checkpoint $checkpoint --protocol $protocol --device cuda --workers 4 --output $replay
& $train -m rollout --task $task --mode summarize --output $replay
```

`workers`是并行进程数量，范围1～4，不是评估次数；本机本轮实测4进程的推理和OpenGL均使用RTX5090。重放是同一测试集的复查，不是新增独立样本；若条件变化，不能恢复或覆盖原批次。视频完成封装前可能无法播放，以`summary.json`的完整50例和逐例视频校验为准。

重新训练从第4节统一入口和 `config/training_baseline.json` 启动；本次60k上限保留每5k候选。之后新模型应另做独立闭环验证选型，再冻结新的测试初态，不自动沿用这次50k结论。`best.pt`仍按离线动作误差保存，`last.pt`仍作恢复点；二者不改名冒充本次选中权重。第6节不传config/checkpoint的通用示例默认使用任务best/逐帧配置，不能复现本节模型。`scripts/run_baseline_campaign.py`仍是离线best的通用流程，不负责本次闭环选型。

上述ACT实验未重训DP或Gesture；后续DP fresh60k固定实验见第6.5节，结果需分别记录。ACT Gesture仅同步60k预算，今后需要独立初始化及同等验证协议。ACT与历史40k DP不能声称等预算对照。旧90%是不同本地网络在另一批初态上的历史结果，不作为本轮官方模型的直接因果对照。

导航指向的实验归档`final_evaluation_config.json`只属于本次托盘实验及50k候选，不能直接用于单臂或另一份新run。候选范围是在观察验证结果后、最终测试前缩小为标准1e-5家族；未完成的1e-4诊断停在16k，未参与选型，也不当作40k对照。完整搜索并非全程预注册；50个新测试初态没有用于调参。核心16项定向检查通过，Pack接口检查通过；主回归609项中6项跳过、1项审核HTTP连接中止，原日志保留，该项单独重跑通过，不写成整套首跑全绿。独立审计未发现阻塞问题，但不保证不存在未知缺陷。

### 6.4 四组模型的固定对照与结果查看

本节记录2026-11-04的历史实验，ACT和DP均为旧本地变体。下面的ACT90%/80%、DP34%/6%不是当前官方核心成绩；已生成的summary、视频、曲线仍可只读查看。当前ACT/DP各自v2入口拒绝相应旧v1权重，历史模型重跑需对应历史源码，新模型须独立训练与评估。

2026-11-04本轮采用同一64条Pack、49/15整会话划分和训练seed20261016。ACT两组先40k再用相同低学习率续训到60k；DP两组40k。手势辅助仅在训练时增加0.1倍左右三类CE，Unknown只屏蔽分类项，不删掉动作样本；推理仍只有机器人RGB与State。

模型动作30Hz、物理控制60Hz和MuJoCo500Hz保持不变。ACT验证选定chunk30：依次执行30个目标后重规划；DP验证选定DDIM20、A8：去噪20步、执行8个目标后重规划。部署参数可区别于训练配置中的DDIM10，实际设置单独冻结；Baseline/Gesture同算法使用相同执行设置。

候选只用2811000～2811015的验证初态选择。四组最终测试使用另外的5811000～5811049，共同60秒仿真时间上限；没有根据这50次结果再调整同一轮权重或参数。四组物理初态相同，推理与渲染均使用NVIDIA GPU。单个训练seed的50次结果是探索性证据，不能替代多seed、跨操作者验证。

已完成的验证选型如下，表中的16次是选型用验证结果，**不是最终50次测试成功率**：

| 模型 | 完成的更新预算 | 选中权重 | 验证成功数 |
|---|---:|---|---:|
| ACT Baseline | 60,000 | 父run的40,000步 `last` | 15/16 |
| ACT + Gesture | 60,000 | 父run的40,000步 `last` | 13/16 |
| DP Baseline | 40,000 | 40,000步 `last`，EMA | 5/16 |
| DP + Gesture | 40,000 | 18,000步 `best`，EMA | 1/16 |

ACT两组都比较40/45/50/55/60k及各自的离线best，按验证任务成功数、验证动作L1、较早步数依次选择。DP先在Baseline上比较去噪次数与执行长度，再冻结20/A8给两组；Gesture只比较自身best/last，不继续用最终测试调参。该搜索范围未覆盖所有超参数，不能称全局最优。离线best仍按原验证loss保存，没有改名或覆盖历史checkpoint。

DP的搜索预算不完全对称：Baseline先比较4个“权重/执行参数”候选、前2名补齐16个验证初态；Gesture继承选定执行参数后比较2份权重。因此最终部署设置、数据和训练预算相同，但不能声称两组经历了完全相同的超参数搜索，也不能把一次成绩差异直接归因为辅助监督的普遍效果。

最终50次测试均已完成，未发生基础设施错误：

| 模型 | 成功 | 成功率 | 超时 | 接触丢失 |
|---|---:|---:|---:|---:|
| ACT Baseline | 45/50 | 90% | 3 | 2 |
| ACT + Gesture | 40/50 | 80% | 7 | 3 |
| DP Baseline | 17/50 | 34% | 28 | 5 |
| DP + Gesture | 3/50 | 6% | 46 | 1 |

两组ACT都实际完成60k更新，但验证闭环没有超过40k，所以最终部署父run的40k权重。继续训练没有稳定改善，不能把训练loss继续下降当作泛化改善。ACT Gesture的验证分类CE从早期约0.24上升到60k约0.82，而分类准确率仍约91%；这提示分类泛化差距，不能据此认定所有控制失败的原因。

当前这版0.1权重的三类辅助监督没有带来成功率收益。DP差距已出现在双抓和抬升阶段（Baseline分别43/50、43/50，Gesture为27/50、22/50），不只是最后未松爪。共享特征的梯度竞争、手势类别与连续动作不完全一致等属于待验证假设；本轮负结果不等于手势记录没有研究价值，也不能承诺下一次改权重就能改善。后续机制对照应重新预先规定训练seed、对称选型范围和测试协议，不在当前最终测试上反复选参。

本轮原始数据、自动裁剪/Safe Stitch和物理成功判据未改变。因此ACT改善有权重选择与执行方式的证据，不能归功于本轮重新剪裁。当前全部实验仍只覆盖一份Pack、一个训练seed和当前任务分布；物理初态可逐项相同，但OpenGL像素可存在±1差异，不宣称视频逐位相同或临界案例绝对可复现。第6.2节的38/50来自另一批初态，不能与此处45/50混在一起当作60k续训收益。

本轮冻结配置、协议、实际模型路径与结果索引集中在 `outputs/rollout/tray_to_platform/comparison_20261104/manifest.json`，四组训练曲线在 `outputs/training/tray_to_platform/comparison_20261104/`。后续运行应明确传 `--config`、`--protocol`、`--run`、`--checkpoint`，不要仅用默认best或按文件夹时间猜测。

`scripts/summarize_policy_comparison.py`核验四组真实算法身份、同Pack/split、相同初态、成功事实和同算法训练预算；ACT若选父run的40k权重，使用显式 `--act-baseline-completed-run` 或 `--act-gesture-completed-run`记录已完成的60k搜索预算。工具核验续训来源，不复制或重新签名历史权重，不把40k权重称为60k权重。`scripts/plot_policy_comparison.py`从真实日志绘制训练曲线，ACT训练总损失与z=0验证L1分别展示；DP总损失含辅助CE，不能用噪声MSE冒充Gesture总损失。

四组统一索引是下面的 `manifest.json`。它分别记录真实选中的模型路径和步数、完成的训练预算、评估配置、协议、视频目录及验证选择依据：

```powershell
Set-Location 'D:\轻量化数据采集\paperA'
$train = (Get-Content config/environments.json -Raw -Encoding UTF8 | ConvertFrom-Json).training_python
$comparison = Get-Content 'outputs/rollout/tray_to_platform/comparison_20261104/manifest.json' -Raw -Encoding UTF8 | ConvertFrom-Json
$groups = $comparison.groups
# 这行选择一组；可改成 act_gesture、dp_baseline 或 dp_gesture。
$model = $groups.act_baseline

# 只查看已经完成的50次结果。
Get-Content (Join-Path $model.evaluation 'summary.json') -Raw -Encoding UTF8
# 如需重建统计和HTML：会写回summary.json/episodes.csv/report.html，不重新仿真。
& $train -m rollout --task $comparison.task_id --mode summarize --output $model.evaluation
```

上述命令只查看或重建历史统计，不重新执行机器人策略；旧模型不能通过当前官方v2入口重跑。各案例视频位于 `$model.evaluation/episodes/seed_<编号>/oblique.mp4`，`result.json`记录该次的真实成功条件和失败原因。每组配置允许不同的checkpoint步数，但同算法Baseline/Gesture执行设置相同。

读取既有结果，重新生成统计和曲线；会写入比较JSON及图表，不训练、不重新仿真：

```powershell
& $train scripts/summarize_policy_comparison.py --act-baseline $groups.act_baseline.evaluation --act-gesture $groups.act_gesture.evaluation --dp-baseline $groups.dp_baseline.evaluation --dp-gesture $groups.dp_gesture.evaluation --act-baseline-completed-run $groups.act_baseline.completed_training_run --act-gesture-completed-run $groups.act_gesture.completed_training_run --output $comparison.comparison
$collect = (Get-Content config/environments.json -Raw -Encoding UTF8 | ConvertFrom-Json).collection_python
& $collect scripts/plot_policy_comparison.py --act-baseline $groups.act_baseline.completed_training_run --act-gesture $groups.act_gesture.completed_training_run --dp-baseline $groups.dp_baseline.completed_training_run --dp-gesture $groups.dp_gesture.completed_training_run --comparison $comparison.comparison --output $comparison.figures
```

`--decode-videos`可按需要附加到汇总命令，逐个完整解码视频；普通查看无需反复执行。图表输出包括 `training_curves.png/pdf`、`closed_loop_comparison.png/pdf` 和原始指标JSON；图中平滑仅用于展示，不参与权重选择。ACT训练posterior loss与推理z=0验证L1不是同一量，DP噪声MSE和ACT动作L1也不能横向比较大小。

在训练和其他评估进程退出后，单进程RTX5090/BF16、5次预热及30次计时，选中ACT查询均值12.6 ms、P95 13.4 ms；DP DDIM20查询均值156.2 ms、P95 167.3 ms。计时含图像预处理和推理，不含MuJoCo渲染/物理，不等于端到端实时延迟。DP每次查询输出8步执行目标，不需要每个30Hz目标重新去噪；60秒上限按仿真时间计，不是墙钟时间。首次启动的Pack结构检查可能耗时数分钟，这一阶段的CPU工作不代表策略退回CPU推理。

<!-- BEGIN CURRENT DP BASELINE -->
### 6.5 当前官方 DP Baseline：配置与使用

已完成：60k新余弦训练，独立20例选中第40000步EMA、DDPM100/A30（15/20），冻结后50例成功20次（40%）。使用`step_040000.pt`；不要擅自换成其他best/last。训练预算不是选中权重步数。失败、曲线、候选表和逐例视频在[报告](../outputs/campaigns/tray_to_platform/dp_baseline_finalization_20261106_205752/REPORT.html)，唯一准确导航是`outputs/DP_BASELINE_CURRENT.json`。

官方数据参照各25,368更新后独立last EMA、同50例，PushT `mean_score`（各回合最高归一化奖励的平均）官方/本地为0.8222 / 0.8222；不是二元任务成功率，也不是完整501/3050epoch论文复现。Panda数据接口、实际训练与Rollout另有检查。全回归618项（跳过6项）、采集39项通过，定向候选保存/恢复、数据与官方核心检查见报告。

先在工程根目录设置`$train`（第0节）。下列命令读取准确模型，不启动训练：

```powershell
$dp = Get-Content 'outputs/DP_BASELINE_CURRENT.json' -Raw -Encoding UTF8 | ConvertFrom-Json
Get-Content $dp.summary -Raw -Encoding UTF8
& $train -m rollout --task $dp.task_id --config $dp.evaluation_config --mode check --run $dp.training_run --checkpoint $dp.checkpoint
# 单次预览，仍用冻结DDPM100/A30和60秒上限
& $train -m rollout --task $dp.task_id --config $dp.evaluation_config --mode preview --run $dp.training_run --checkpoint $dp.checkpoint --seed 910001 --device cuda
# 重跑同50初态，创建新的输出目录，保留历史结果
$out = 'outputs/rollout/tray_to_platform/diffusion/manual_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& $train -m rollout --task $dp.task_id --config $dp.evaluation_config --mode batch --run $dp.training_run --checkpoint $dp.checkpoint --protocol $dp.final_protocol --output $out --workers 4 --device cuda
```

`--checkpoint`控制实际EMA权重，`--config`冻结DDPM去噪步数、执行长度和60秒上限；`--protocol`固定50初态；`--workers`是并行进程数，不是测试次数。`--device cuda`不会静默切CPU。推理墙钟耗时与仿真60秒分开统计。

新训练用第4.1节统一入口和`config/training_diffusion.json`（60k、候选5k）；同预算中断恢复用`--resume <run>/checkpoints/last.pt`。本次已完成的60k无需再resume；旧40k末尾LR为0，不能照搬ACT延长方式。完整验证noise MSE决定离线best，闭环20例另决定部署权重；sample Action MAE仅前2批32个anchor。网络、Loss、EMA、Safe Stitch、遥操作和成功条件没有改动。

Baseline与Gesture未来公平比较必须同步Pack、split、主干、预算和选型范围；本轮未训练Gesture，其配置仍保留历史40k。ACT为H1、DP为H2，不能称相同输入信息或相同算力。单臂可以复用官方模型核心，但需明确适配State/Action schema、归一化字段、Pack和执行器，不是只改task_id。
<!-- END CURRENT DP BASELINE -->

## 7. 可重复的工程检查（不用于论文结果）

日常维护只需选择相关入口；不需要逐个运行`tests`中的文件，也不需要下载官方基准数据集或预训练策略：

| 入口 | 用途 | 是否需要正式数据 |
|---|---|---|
| `& $train tests/run_regressions.py` | 当前任务、控制、Dataset、ACT/DP张量、恢复及Rollout回归；生成临时fixture，不运行正式长训练 | 否 |
| `& $train -m tests.check_training_pack --task $task --pack $pack` | 正式Pack的只读接口检查；仅显式加`--updates`才短更新 | 是 |
| 下方采集启动、浏览器、测量日志、完成保存及物理搬运专项检查 | 修改对应功能时选用，不能代替真人操作 | 按各入口说明 |

`tests/task3/`保留当前共享控制器和Dataset回归，目录名不表示使用旧工程；`tests/papera/`保留当前ACT/DP及Rollout回归。旧夹爪策略套件与绑定固定迁移产物的两个一次性smoke脚本已移除，仍使用的夹爪辅助函数集中在`tests/task3/rate_fixture.py`。

`test_act_official.py`是可选的独立上游源码数值对照，日常回归不再隐式依赖本机复现目录。已有固定版本ACT源码时，可设`$env:PAPERA_ACT_UPSTREAM='<ACT源码目录>'`再运行`& $train tests/run_regressions.py --with-upstream`；本机未设置时沿用已有缓存，缺少源码会明确跳过该组。入口不会自动下载源码、数据集或权重。工程内`training/*/official`及其许可证是正式依赖，仍必须随源码保留。

准备分享时先修改`config/environments.json`中的两个Python解释器和Node路径，再运行`& $train scripts/configure_vscode.py`。四个训练F5入口仍保留；训练Pack路径由运行者显式输入。`.local/`、`.cache/`、`datasets/`、`outputs/`、`logs/`均已被Git忽略。独立复现下载缓存继续留在`.local/`，不会随源码上传，也无需重新下载。单臂队友应对齐模型核心、归一化原则、频率、训练预算和选模协议，另行声明自己的State/Action Schema；当前双臂Pack或权重不能直接当作单臂输入。

对正式真人Pack可先做不写文件的接口检查，再按需要做至多每算法3次独立工程更新。工具读取两份实际JSON配置，当前ACT H1、DP H2，默认使用各自batch16；不是重新裁剪或正式长训练：

```powershell
$task = 'tray_to_platform'
$pack = (Get-Content "outputs/dataset/$task/TRAINING_PACK_CURRENT.json" -Raw -Encoding UTF8 | ConvertFrom-Json).path
& $train -m tests.check_training_pack --task $task --pack $pack
# 可选；输出目录需换成未使用的新名称
& $train -m tests.check_training_pack --task $task --pack $pack --updates 2 --device cuda --output '.local/checks/pack-check-01'
```

默认检查官方Pack Reader、ACT/DP历史形状、原观察后下一合法tick、所有安全拼接/硬边界两侧、Padding与train-only归一化。可选短更新会在第一步保存独立engineering checkpoint，重建模型并恢复optimizer/scheduler/sampler/RNG和DP EMA后继续；checkpoint明确不可部署，不等同完整正式训练或闭环成功。默认不重复全量大缓存哈希；确需完整字节核验可显式加 `--full-verify`。

2026-11-03 对本轮64条真人Pack的实际检查已完成：训练45,958个窗口、验证14,077个窗口，其中分别9,359/2,319个窗口跨Dataset已验证的安全拼接；非法动作锚点均为0。ACT H1、DP H2都以640×480、batch16、CUDA/BF16执行2次参数更新，第一步保存并恢复完整工程状态后继续，状态一致、损失和梯度有限。原始因果动作、全部拼接及硬边界两侧、Mask、训练集归一化检查通过。报告位于 `.local/checks/pipeline_readiness_20261103/real_pack_check/report.json`；全回归467项（3项跳过）及采集39项通过。这些结果仅证明数据接口与短更新/状态恢复可用，不证明模型已学会任务，也不替代正式trainer的长训练收敛或闭环成功率评估。本轮没有启动正式训练或执行Git提交。

```powershell
& $train tests/run_regressions.py
& $collect tests/collection_launch_smoke.py
& $collect tests/pipeline_smoke.py --task tray_to_platform --sessions 2
& $train dataset/manage_dataset.py --task tray_to_platform --engineering process
$engineeringRoot = '.local/checks/pipeline/outputs/dataset/tray_to_platform'
$curation = (Get-Content -LiteralPath "$engineeringRoot/CURATION_CURRENT.json" -Raw -Encoding UTF8 | ConvertFrom-Json).selected_version
& $train dataset/manage_dataset.py --task tray_to_platform --engineering prepare --run $curation --review-mode auto
$prepared = (Get-Content -LiteralPath "$engineeringRoot/PREPARED_CURRENT.json" -Raw -Encoding UTF8 | ConvertFrom-Json).path
& $train dataset/manage_dataset.py --task tray_to_platform --engineering pack --prepared $prepared
$testPack = (Get-Content -LiteralPath "$engineeringRoot/TRAINING_PACK_CURRENT.json" -Raw -Encoding UTF8 | ConvertFrom-Json).path
& $train dataset/manage_dataset.py --task tray_to_platform --engineering verify-pack --pack $testPack
$testRun = '.local/checks/training/my_smoke_baseline'
& $train -m training.papera --task tray_to_platform --variant baseline --pack $testPack --smoke-test --output $testRun
& $train -m rollout --task tray_to_platform --engineering --mode preview --run $testRun --max-seconds 2 --seed 910001 --headless
```

`--smoke-test` 固定3步、较小工程验证范围；`--engineering` 只允许显式工程数据/短预览，不允许正式50次评估。工程采集结果明确保存为 scripted_test / failure，证明记录通路，不伪称成功示范。重复生成前保留或归档旧工程目录；不同 task_binding 不会混合。无需为了迁移重新处理旧视频。

同一工程 Pack 可检查 DP，并验证短训练与恢复；使用新输出目录，冻结预算保持当前Baseline默认60,000步，以下只各运行2次更新：

```powershell
$testDpRun = '.local/checks/training/my_smoke_diffusion'
& $train -m training.diffusion --task tray_to_platform --pack $testPack --engineering --check-only
& $train -m training.diffusion --task tray_to_platform --pack $testPack --engineering --stop-after 2 --output $testDpRun
& $train -m training.diffusion --task tray_to_platform --resume "$testDpRun/checkpoints/last.pt" --stop-after 2
```

工程属性随 checkpoint 恢复，不会变成正式训练；`completion.json` 的停止状态与完成总预算分别记录。上述工程命令不验证 DP 自主成功率。

### 7.1 高侧壁托盘到高台：tray_to_platform

托盘任务 `tray_to_platform` 使用场景 ID `HIGHWALL_PLATFORM`。

```powershell
$task = 'tray_to_platform'
& $collect -m collection list
& $collect -m collection start --task $task --seed 0 --operator-id operator01
```

也可双击根目录 `START-PAPERA.cmd`，输入 `tray_to_platform`。首次先不录制：双掌绑定 → ILoveYou对准黄色抓耳上方 → 张掌下降至抓耳中部 → 握拳闭合、等实际双抓成立 → 握拳抬升 → ILoveYou搬到高台上方 → 握拳下降 → 等高台承重后双掌松开。左右独立、100 ms确认、XY/Z映射以及原抓持/IK保护均未修改。下放时保持握拳，张掌会打开爪；不要在尚未落台时提前张掌。

当前托盘主体24×32 cm，底板8 mm，侧壁高于底板内表面5 cm，两端壁6 cm；两个黄色抓耳各36×120×52 mm，总质量0.35 kg。当前高台28×36 cm、高3 cm，中心世界XY为 `[0.20, 0] m`，顶部Z=0.38 m。物体局部底面Z=-0.034 m，所以程序按实际台高计算目标参考点 `[0.20, 0, 0.414] m`，不会把场景JSON中历史target_position的Z直接作为有效目标。目标Yaw=0；初态XY中心 `[-0.19, 0] m`，随机半范围 `[0.04, 0.08] m`，Yaw ±0.04 rad。初态分布本次未调整。

当前紧凑布局参数均在 `tasks/tray_to_platform/scene.json` 的 `scene` 下：`table_half_size_xy=[0.40,0.45]` 是半尺寸，实际桌面80×90 cm；`table_top_z=0.35`；左右 `base_position=[-0.10,∓0.62,0.35]`。基座相对原布局各沿Y向内13 cm、沿X向负方向10 cm，中心距对应桌边17 cm。左右 `workspace` 的X范围改为 `[-0.30,0.30]`，Y仍分别 `[-0.45,-0.06]`、`[0.06,0.45]`，Z仍 `[0.36,0.80]`。这些是末端指令边界，不是保证处处无碰撞的机械工作空间；平台占据的低高度区域不可穿越。

布局版本为1.2.0，证据在 `.local/checks/reach_layout/`：100个随机初态预检通过；714个双臂对称位置/高度样本检查了朝下姿态IK及静态碰撞；seed 0、19原物理路线通过，seed 7原路线因未充分卸载到平台而中止，再真实下降1 mm的工程路线通过。未改变控制器、支撑门槛或成功判据。静态采样及脚本通过不等于整个连续空间可达或真人验收完成。旧Raw/Pack/Checkpoint冻结的是旧布局，不能冒充与新配置兼容；本次没有改写旧数据。重新启动采集进程才会加载新布局。

建议运输参考高度0.509 m（底面理论上高出台顶25 mm）。双侧grasp site局部位置为 `[0, ±0.22, 0] m`，向下接近，夹爪沿托盘X方向夹36 mm厚抓耳；仍是原80 mm最大开度。参数入口为 `tasks/tray_to_platform/scene.json`；修改尺寸后必须一起核对 `scene.py` 的几何一致性检查与可达性，不能只改目标Z。仅新任务辅助top的中心X从0.04移到0 m，垂直视野从0.62扩大到0.66 m，避免初态侧壁裁切；单oblique及其他任务相机保持原值。

任务DONE由 `tasks/tray_to_platform/success.py` 判定：真实稳定双抓历史、离开原桌面至少3 cm并稳定0.12秒、目标位置/姿态、真实高台承重与底面包含、两爪指令和实测开度至少40 mm且无托盘手指接触、物体停稳、无失败。原桌面承重、悬空保持、单侧假抓或未松爪均不能成功。MuJoCo软接触会有约毫米级沉降，使用现有位置/速度容差，不要求底面数值恰等于0.450000。

以下只运行工程检查，不打开摄像头，不生成正式真人示范：

```powershell
& $collect -m unittest tests.task3.test_highwall_platform tests.task3.test_platform_success -v
& $collect tests/collection_launch_smoke.py --task $task --browser-check
& $collect tests/platform_transport.py --seed 0 --media
& $collect tests/pipeline_smoke.py --task $task --sessions 2
& $train dataset/manage_dataset.py --task $task --engineering process
```

真实接触脚本通过正式场景工厂和Rollout的 `PolicyExecutor` 执行末端目标，初始化之后不写物体位姿、不添加焊接或外力。证据在 `.local/checks/tray_to_platform/`，每次新目录包含阶段状态、接触/支撑力与失败原因；`--media` 另存四视角阶段PNG和 `transport_oblique.mp4`。最终验收演示为 `final_acceptance/`（seed 0，484帧已完整解码）。它是脚本控制验收视频，不是ACT推理视频。`pipeline_smoke.py`另用正式Recorder保存标为 `scripted_test / failure` 的短记录，用于验证字段和Dataset接口，不能拿来训练正式模型。`--browser-check` 用独立无界面Chrome检查网页确实显示已连接，不请求摄像头。

正式采集保存于 `datasets/tray_to_platform/`，日志在 `logs/tray_to_platform/`；处理、Prepared、Pack使用第2～3节，把 `$task`设为本任务即可。模型输入仍为单oblique、S16，输出A14，Safe Stitch与训练算法没有改动。任务、几何、相机、seed、控制协议和源码SHA随绑定保存。Rollout使用同一Registry工厂与专属成功监视器；其他任务的checkpoint不可拿来作为本任务正式评估模型。

注意：现有Dataset的“释放完成后裁尾”使用接近全开证据；任务已半开DONE但录像未到全开时，Dataset可能保守保留尾帧供审核。本轮不更改裁剪规则，也不会把这种保留误称为自动完成裁尾。

工程验证：20个seed正式预检均无标记，5个初态（0、1、2、3、17）真实搬运到DONE；312项主回归（1项GPU选测跳过）、17项采集环境回归通过，最终相机调整另通过13项专属检查。两条新工程记录完成422→256帧裁剪和2处Safe Stitch，Prepared/Pack核验通过，正式读取器仍接收单oblique、S16和K30×A14。负例单测用受控状态隔离14种终态错误及抓持/抬升历史；真人摄像头操作、更多初态压力测试和训练后的自主成功率仍待验证。脚本末段请求全开，半开判据边界来自单元测试。先体验并确认操作顺畅，再正式采集。

版本回溯标签：`before-highwall-platform-20261025` 为修改前，`highwall-platform-v1-20261025` 为本次场景验收版本。使用 `git show <标签>` 查看说明、`git diff before-highwall-platform-20261025 highwall-platform-v1-20261025 -- tasks tests docs README.md` 查看改动；需要切回体验前先检查 `git status`，不要丢弃未提交工作。新工程已有独立Git，不推送远程。工程视频和检查包位于被Git忽略的 `.local/checks`，标签保存源码及说明，不保存这些体积较大的产物。

### 7.2 ACT官方A/B参照：先验证训练代码，再更换自己的数据

A是固定官方仓库原始`imitate_episodes.py`的完整训练和原任务评估；B通过本地`train.py --official-reference`入口，实际使用本地`ACTPolicy`、`compute_loss`和与正式Pack共用的`engine.run_loop/optimize_batch`。B没有另写临时参数更新循环，也没有将官方S14伪造成Panda S16。官方HDF5 Reader仅通过`reference.py`转换为本地batch字典；这不等于验证了Safe Stitch裁剪或Panda物理执行器。

本轮固定同一份50条官方脚本示范、40/10划分、全50条归一化统计、单top 640×480、S14/A14关节目标、K100、hidden512/FF3200、batch8、KL10、AdamW固定1e-5、seed0、2000 epochs（10000次更新）。B保留官方Episode等权随机时刻采样，以及后验L1+KL验证和best选择，评估保留K100动作块、seed1000的50个初态。不能把本项目按帧采样、train-only统计、z=0选型或K30默默混进这个比较。

当前机器的实际命令如下。数据和预训练权重复用本地缓存，不重新下载；输出在`.local/checks`，不是正式托盘训练结果。临时嵌入式Python不自动将当前目录加入import路径，因此`launch.py`只补项目路径并调用同一个`train.main()`，不包含训练逻辑。

```powershell
Set-Location 'D:\轻量化数据采集\paperA'
$refPy = '.local/checks/official_reproduction_20261105/act/physics_py311/python.exe'
$refData = '.local/checks/official_reproduction_20261105/act/data_original_physics'
$upstream = '.local/checks/act_reference_20261106/upstream'
$refB = '.local/checks/act_reference_B_20261106'
& $refPy "$refB/launch.py" --official-reference $refData --reference-upstream $upstream --reference-epochs 2000 --output "$refB/run"
# 本轮B完成后，用与A相同的原始评估入口及场景评估导出的policy_best.ckpt
& $refPy "$refB/evaluate.py"
```

这是本轮实际使用的命令；训练目录已有结果时会拒绝覆盖。需要重复训练时须显式指定新的`--output`，并让原始评估入口的`--ckpt_dir`指向该新目录；本轮`evaluate.py`固定读取`$refB/run`且拒绝覆盖已完成评估。`--smoke-test`只在独立新目录做两轮工程训练，不替代完整B。此profile只接受Baseline，拒绝Pack、Gesture、续训及任意步数覆盖，不改变`--task ... --pack ...`正式入口的S16/A14/K30约束。

适配类别：Pack读取、Safe Stitch窗口/Mask、S16的两个输入投影、单oblique与米制XYZ/RPY动作是任务数据接口；本地权重缓存、Windows路径和读取进程数是运行适配；按帧采样、z=0选best与学习率/预算是实验策略，不能统称适配必需。原ResNet/Transformer/CVAE核心不在本轮重新设计。旧Panda配置与默认计算保持不变，旧模型通过精确的源码兼容登记读取，不能用这个登记放行后续网络或数据变更。

2026-11-06已完成：A为48/50（96%），B为50/50（100%）；两者均完整2000轮/10000次更新，均选epoch1780，后验验证总损失分别0.043276/0.041681。共同成功48例，仅B成功2例。两组曲线接近，本次未见通用本地训练实现明显退化；单训练seed和50例不能证明性能提升或所有代码无误，更不是Panda托盘成功率。B的50段原视频全部解码400帧；全回归614项（6跳过）和39项采集通过，工程Collection→Dataset→Pack→3步训练→2秒CUDA预览也通过。结果与训练曲线入口：[A/B汇总](../.local/checks/act_reference_B_20261106/RESULTS.html)。B参考模式尚未提供CLI续训，不可拿其专用checkpoint直接走Panda恢复入口；正式Panda既有恢复能力已回归验证。本轮没有重训正式托盘模型或提交Git。

### 7.3 双臂 Push-R

在项目根目录使用采集环境运行，或双击 `START-PAPERA.cmd` 后输入 `push_r`。

```powershell
$task = 'push_r'
& $collect -m collection list
& $collect -m collection start --task $task --seed 10 --operator-id operator01
```

先双掌绑定，张掌打开夹爪并下降，随后用 ILoveYou 在 XY 平面推动深灰色 R 到绿色轮廓。双臂可分别接近后侧直边、上横梁和斜腿外侧，通过不同接触位置纠正方向。当前任务要求位置和朝向同时对齐并稳定0.5秒，达到 DONE 后按既有流程保存。正式原件写入 `datasets/push_r/`，Dataset、Training 和 Rollout 的通用命令均使用同一个 `$task`。

工程物理演示可用以下命令复核。输出目录必须是尚未存在的新目录。

```powershell
& $collect -m unittest tests.push_r_transport -v
& $collect -m tests.push_r_transport --media .local/checks/push_r/my_preview
```

本轮预览在 `.local/checks/push_r/preview/`，包含 top/oblique 初态、推动与归位图及 `contact_push.mp4`。物理脚本使用真实指尖接触完成平移和转向，未建立双抓。脚本验证及工程记录都标记为 `scripted_test`，真人试操和正式示范仍需另行完成。

## 输入输出速查表

| 程序/阶段 | 输入 | 输出 | 环境 |
|---|---|---|---|
| `-m collection list/start` | Task Manifest、seed、摄像头 | datasets/task、logs/task | collect + Node |
| `manage_dataset process` | datasets/task 的成功真人会话 | audits、curation、CURRENT | train |
| `manage_dataset review` | 指定 Curation | reviews.json、临时裁剪预览 | train + 浏览器 |
| `manage_dataset preview/preview-menu` | CURRENT或指定冻结Curation | 只读自动裁剪预览、临时视频；不改CURRENT | train + 浏览器 |
| `manage_dataset prepare` | Curation + auto/reviewed | prepared、PREPARED_CURRENT | train |
| `manage_dataset pack/verify-pack` | Prepared + 固定 split | training_packs、TRAINING_PACK_CURRENT | train |
| `-m training.papera` | task、variant、明确 Pack | outputs/training/task/variant/run | train/CUDA |
| `-m training.diffusion` | task、明确 Pack；可选check-only | outputs/training/task/diffusion/run（检查不建run） | train/CUDA |
| `--resume / --continue-from` | 指定 last.pt 和冻结来源 | 原 run / 新续训 run | train/CUDA |
| `analyze_checkpoint / plot_training_analysis` | run 与 checkpoint | run/analysis | train / collect |
| `-m rollout check` | task、run、best/last | 来源核验终端报告 | train |
| `-m rollout prepare` | task、Pack、评估配置 | 固定50次 protocol | train |
| `-m rollout preview/batch` | 同任务 run + 初态/协议 | outputs/rollout/task/variant/evaluation | train/MuJoCo |
| `-m rollout summarize` | 已有 evaluation | summary.json | train |

所有命令入口均经过帮助调用；本轮实际执行范围与限制见 [MIGRATION.md](MIGRATION.md)。正式长训练和50次策略评估由你手动启动。


## 8. 各启动程序说明（最后查这一节即可）

### 双击入口与后台关系

| 实际程序 | 如何使用 | 会启动什么 / 不会启动什么 | 输出与退出 |
|---|---|---|---|
| `START-PAPERA.cmd` | 在工程根目录双击，输入完整 task_id，例如 `h_frame_push_align` 后回车 | 调用 `scripts/launch.ps1`，进入采集任务菜单；启动 MuJoCo 后端、Node 网页服务、浏览器；不会自动启动摄像头、录制、训练 | 网页默认8014，后端5014；日志 `logs/<task_id>/<时间戳>/`；保留终端窗口 |
| `collection/START-COLLECTION.cmd` | 双击 | 转交根目录的 `START-PAPERA.cmd`；功能完全相同，不是另一套采集器 | 两个CMD任选一个，不要重复双击占用同一端口 |
| `scripts/launch.ps1` | CMD内部调用；日常无需单独双击 | 读取 `config/environments.json`，设置解释器/PATH/UTF-8、移除旧PYTHONPATH，再运行采集菜单 | 不安装环境，不自动关闭其他服务 |
| `python -m collection list` | 使用 `$collect` 运行 | 只列出注册任务及ready/development状态 | 不打开网页或仿真 |
| `python -m collection start --task ...` | 使用 `$collect`；可指定种子、操作者和端口 | 跳过交互菜单，启动指定任务的同一套采集服务 | 与CMD相同；`--no-browser`不打开浏览器，`--headless`不显示MuJoCo窗口 |
| `dataset/manage_dataset.py` | 使用 `$train`，指定 `--task` 和子命令；`preview-menu`可交互选择任务 | `process/reclip`处理，`review`可写审核，`preview`只读裁剪预览，`prepare/pack/verify-pack`导出和核验 | `outputs/dataset/<task_id>/`；审核默认8026，不控制机器人 |
| `dataset/PREVIEW-CURATION.cmd` | 双击，选任务与菜单1/2/3；可传 `-Task tray_to_platform -Port 8026` | 调用 `start-preview.ps1` 配置训练环境，再调用同一 `manage_dataset.py preview-menu`；选1/2仅预览，选3才处理数据；不训练 | 浏览器自动打开，只读预览；终端显示版本与实际端口，Ctrl+C退出 |
| `python -m training.papera` | 使用 `$train`；显式指定task、variant、Pack，或指定resume checkpoint | 启动训练；Baseline/Gesture共用入口，当前最大60,000步；不会自动打开采集网页 | `outputs/training/<task>/<variant>/<run_id>/`；按第5节恢复 |
| `python -m training.diffusion` | 使用 `$train`；指定task和Pack，或DP last.pt | Baseline默认60,000步fresh cosine；Gesture历史40,000步；`--check-only`只读检查，`--stop-after N`保存并暂停本次更新 | `outputs/training/<task>/diffusion/<run_id>/`；按第5.1节恢复 |
| `python -m rollout` | 使用 `$train`；显式指定task、mode、run/Pack | `check`只检查，`prepare`只固定初态清单，`preview`执行一次，`batch`批量执行，`summarize`汇总 | `outputs/rollout/<task>/<variant>/`；默认headless，视频须等完成后播放 |
| `python -m training.papera.analyze_checkpoint` | `& $train -m training.papera.analyze_checkpoint --run $run` | 读取checkpoint做离线指标分析，不继续训练 | `<run>/analysis/evaluation/` |
| `python -m training.papera.plot_training_analysis` | `& $collect -m training.papera.plot_training_analysis --run $run` | 读取已有训练/分析结果生成图表 | `<run>/analysis/figures/` |
| `scripts/configure_vscode.py` | `& $train scripts/configure_vscode.py` | 仅生成/更新`.vscode`快捷配置，不自动运行训练 | 解释器路径变更后执行一次即可 |

上表的Python命令均从 `D:\轻量化数据采集\paperA` 执行；`$collect`、`$train` 的赋值见第0节。不要在资源管理器里直接双击 `.py`，以免使用错误解释器。新工程不使用旧工程的 `START-TASK3.cmd`、`START-DATASET.cmd` 或旧训练入口。

### VS Code 的运行项

按F5之前，在“运行和调试”下拉选择目标项。**不要把运行当前Python文件的三角按钮与命名的F5配置混淆。**

| F5名称 | 实际含义 | 需要输入 |
|---|---|---|
| `ACT 训练` | `training_baseline.json`，最大60,000步 | task_id、正式Pack绝对路径 |
| `DP 训练` | `training_diffusion.json`，60,000步fresh cosine，每5k保留候选 | 相同task_id与Pack |
| `ACT + 手势辅助监督训练` | `training_act_gesture.json`，最大60,000步，CE权重0.1；本轮未训练 | 相同task_id与Pack |
| `DP + 手势辅助监督训练` | `training_diffusion_gesture.json`，历史40,000步，CE权重0.1；本轮未改未训，正式公平对照前须同步预算 | 相同task_id与Pack |

### 启动采集、检查连接、正常退出

1. 双击两个采集CMD中的任意一个；输入ready任务ID。等待终端打印 `Page:`、`Raw:`、`Logs:`。
2. 打开**本次终端打印的Page地址**；默认 `http://127.0.0.1:8014/dual_arm.html`。更新网页代码后按 **Ctrl+F5**。不要沿用旧工程8013页面。
3. 网页应先显示“仿真已连接”，此时摄像头尚未开启；再点击“启动摄像头”、授权、双掌绑定。MuJoCo窗口打开本身不保证网页状态渲染正常。
4. 退出前先保存或放弃正在录制的一轮，等待视频保存完成，再点击网页“结束采集并退出仿真”；也可在未录制/保存时于启动终端按Ctrl+C。不要在保存中强关进程。
5. 如果后端已经关闭，只刷新网页无法恢复仿真。关闭原采集终端，重新双击启动器。端口仍占用时，先用 `Get-NetTCPConnection -State Listen -LocalPort 8014,5014` 查监听进程，并用 `Get-Process -Id <OwningProcess>` 核对。若为旧采集服务，先保存或放弃录制，在原启动终端按 Ctrl+C。也可用 `& $collect -m collection start --task h_frame_push_align --web-port 8015 --backend-port 5015` 启动在另一组空闲端口，访问新终端打印的 Page 地址。

下面是只读诊断，不会启动摄像头、控制机器人或录制：

```powershell
$backendHealth = Invoke-RestMethod 'http://127.0.0.1:5014/health'
$webHealth = Invoke-RestMethod 'http://127.0.0.1:8014/api/dual/health'
$backendHealth | Select-Object ok, project_root, session, recording, saving
$webHealth | Select-Object ok, project_root, session, recording, saving
# 使用本次终端输出的日志目录替换下面的路径
Get-Content '<Logs目录>/backend.log' -Tail 30
Get-Content '<Logs目录>/web.log' -Tail 30
```

- 后端请求拒绝连接：仿真服务已停止或端口不对应，查看backend.log并重新启动。
- 后端正常、网页代理502：检查网页所属工程及后端端口，使用本次打印的URL。
- 两个请求都正常、网页仍未连接：检查浏览器控制台；不能直接认定MuJoCo断线。本次修复的原因就是操作帮助替换HTML后，继续写入已移除的`gripper-guide`元素，引发异常后被误报为未连接。

开发时可在**未录制、未保存**状态运行真实浏览器检查：

```powershell
& $train tests/collection_browser_smoke.py --url 'http://127.0.0.1:8014/dual_arm.html'
```

它用独立临时配置启动无界面Chrome/Edge，在连续状态刷新后检查真实DOM是否为“已连接”，不会请求摄像头或录制。它会短暂加载采集页面，因此不要与真人操作并行执行。报告、页面快照和浏览器日志保存到 `.local/checks/browser_connection/<时间戳>/`。这是界面连接检查，不替代真人手势验收。

高台任务现在不要求对齐目标朝向：完整落在高台支撑范围内且通过实际松爪、稳定及抓持历史等物理条件即可进入 DONE。部署 Rollout 使用相同规则。朝向差数字仍可供观察，但不再阻止成功；悬空、仍接触夹爪、越出台面或放回原桌面仍不成功。升级后重启采集服务并 Ctrl+F5，已运行进程不会自动加载新规则。


## 采集保存使用独显（2026-11-03）

本机对**当前Windows用户**的`D:\Conda\envs\mujoco_repro\python.exe`设置`GpuPreference=2;`（高性能）。注册表键为`HKCU\Software\Microsoft\DirectX\UserGpuPreferences`，值名是该Python的完整路径。相机仍是单oblique640×480/30Hz，保存流程、原始日志和物理快照不变；未启用裁剪或启动训练。

实际采集Python的新OpenGL上下文已核验：NVIDIA Corporation / NVIDIA GeForce RTX5090，驱动595.97。无需重新安装Conda或MuJoCo，下次正常双击`START-PAPERA.cmd`即可。GPU选择发生在新进程/图形上下文创建时；若以后改设置时已有采集服务运行，先保存或放弃本轮，再正常退出并重新启动，不能只刷新浏览器。

整段离线工程重放使用同一份1107帧快照和两个同视角视频输出：核显36.79、39.44秒；临时独显对照4.67秒，正式采集解释器独显验证4.49秒。视频、Parquet均1107帧/行，状态与动作逐项相同；不声称不同显卡像素逐项完全相同。人手会话日志导出另测435MB完整会话约3.74秒，因此4.49秒不是“点击保存至下一轮”的保证时间。

换电脑、Windows用户或Python安装路径时，需要重新配置：Windows“设置→系统→屏幕→显示卡/图形”，添加`config/environments.json`中的采集Python作为桌面应用，选择“高性能”，确认显示NVIDIA GPU并保存。训练使用CUDA并不等于采集OpenGL自动使用同一显卡。

旧设置备份`.local/checks/collection-nvidia-20261103/gpu-preference-before.json`，本次此前没有该Python专用条目；如要恢复，可在Windows图形设置删除该应用的专用偏好。不要删除整个UserGpuPreferences键。Git记录本节说明，Windows显卡偏好属于本机设置，不会随源码回退或Git推送自动迁移。实际上下文、完整重放和输出检查证据均在同一检查目录。
