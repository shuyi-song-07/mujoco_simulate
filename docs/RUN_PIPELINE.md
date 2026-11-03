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
$task = 'tray_transfer'
```

`$collect` 是已有 mujoco_repro：Flask、LeRobot、MuJoCo 和采集渲染。`$train` 是已有 paperA_train：PyTorch/CUDA、PyAV、PyArrow、TensorBoard 和训练/推理。不要在基础 Anaconda 环境里运行。Node 也从同一环境 JSON 读取。实际版本见迁移报告及 `.local/checks/environment.json`；`requirements.txt`、`training/papera/environment.yml`、`requirements-lock.txt` 是依赖参考，不需要每次运行都安装。

VS Code：**打开 `paperA` 文件夹**，信任本地工程并启用 Python/Python Debugger 扩展；F5 下拉提供 Collection、ACT Baseline/Gesture/Resume、DP Train/check/Resume、Rollout check/preview。按提示输入 task_id 和 Pack/run 绝对路径。环境路径修改后重新生成快捷配置：

```powershell
& $train scripts/configure_vscode.py
```

它只更新 `.vscode`，不启动训练。ACT Baseline/Gesture 和 DP Train 快捷项默认预算均为40,000次参数更新，分别读取各自训练配置；使用前必须准备正式真人 Pack。正式条件统一为单 oblique RGB 640×480、30Hz模型动作与K30；ACT观察1帧、DP观察2帧。物理仿真500Hz，控制器60Hz，每个模型动作目标执行2个控制tick。

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

本轮先只运行ACT Baseline和DP Baseline，用于验证这批数据能否学到任务，不增加新模型或新loss，不做大范围调参。以下只是正式训练命令，本轮数据准备不会自动执行；同一GPU先运行一个模型，结束后再运行另一个。

先确认 `$pack` 是本任务的正式真人数据，运行 verify-pack 后保留这个值。两组用 **同一个 Pack、split、seed、训练预算和主干配置**：

```powershell
& $train -m training.papera --task $task --variant baseline --pack $pack
& $train -m training.papera --task $task --variant gesture --pack $pack
```

这是两条长训练命令，由你选择时间执行；本轮迁移没有执行它们。输出自动为 `outputs/training/<task_id>/<variant>/<run_id>`。

默认参数在 `config/training_baseline.json`：K30、batch16、40,000次参数更新、AdamW、主干学习率1e-5/其余1e-4、weight decay 1e-4、warmup500、KL10、BF16。不是40,000个epoch；终端会打印样本窗口与约当epoch。模型使用640×480全画面，不裁剪、不降低Pack原图分辨率；每次输入单帧观察。模型动作频率30Hz，每个目标由60Hz控制器执行2个tick；物理仿真500Hz。`--steps` 覆盖训练更新预算，`--seed` 改训练随机种子，`--config` 换参数文件；对照实验应同步变化。

Baseline：Action L1 + 10×KL。Gesture：同一主干和动作损失，额外使用原有 gesture head 与分类辅助损失（默认权重1）；缺失/未知/不可信标签按已有 Mask 处理，不以 Mask 删除机器人训练帧。两组推理均为 z=0、无真人输入。现有 ACT 数学实现未重写。

参数更新、验证日志在 `metrics.jsonl`，TensorBoard 在 run 的 `tensorboard/`，`best.pt` 按验证 prior action L1 选择，`last.pt` 用于恢复。质量批准与训练接口是两件事：auto Pack 允许探索性训练，但不能据此称全部示范高质量。

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

### 4.1 Diffusion Policy Baseline

DP 复用同一任务的正式 Training Pack、会话 split、S16/A14 和 Safe Stitch 结果；训练入口不读取 Raw，不补日志，也不接收缺失控制/来源信息的 HF merged 导出。先只读检查张量和数据接口，再自行启动正式训练：

```powershell
& $train -m training.diffusion --task $task --pack $pack --check-only
& $train -m training.diffusion --task $task --pack $pack
```

默认配置为 `config/training_diffusion.json`：单 oblique RGB 640×480全画面、过去至当前2帧观察（`model.observation_horizon=2`）与每帧16D state 条件，预测从当前锚点起的30帧14D action，模型动作频率30Hz。单视角与多帧历史是两个概念，不增加摄像头。动作尾部 padding 不参加损失；历史观察不跨硬边界，首部不足时重复同一合法序列首帧。State/Action 的 min-max 归一化仅拟合训练 split，再供验证和恢复使用。模型为 ResNet18（GN）+ SpatialSoftmax（32个空间关键点）+ 128维视觉投影 + 带 FiLM 条件的1D U-Net，通道 `[256,512,1024]`、卷积核3、时间嵌入128。cosine 100步加噪，以 epsilon MSE 训练，预测用10步 DDIM；`action_horizon=8` 保存为后续滚动执行配置，本轮未接入闭环执行。DP 不使用手势标签或 Gesture head。这是依据[原作者实现](https://github.com/real-stanford/diffusion_policy)适配本工程数据接口的基线，未声称复现论文全部超参数或结果。

ACT 与 DP 使用相同640×480、K30、batch16和40,000次更新。ACT保持1帧观察，DP使用2帧；同一算法内Baseline/Gesture必须用相同数据、split、相机、历史和预算。跨算法比较需要披露历史不同，不能称输入信息完全相同；严格控制历史的比较可另跑DP-H1。以后的历史、分辨率、宽度或预算变化都作为新实验记录。

默认 AdamW 学习率1e-4、batch16、40,000次更新、warmup500、BF16，EMA 衰减上限0.999且开始阶段预热。每25步记录训练日志，每1000步验证和保存，最终/请求停止时也验证保存。`best.pt` 依据固定噪声与 timestep 的 **EMA 验证 noise MSE** 选择；默认检查完整验证集（`validation_batches=null`），其中前2个 batch 另记录 DDIM 采样动作误差作诊断。采样误差与训练损失都不是任务成功率。

DP 常用配置字段如下；ACT 的对应公共条件在 `config/training_baseline.json`，对照时一起核对。

| JSON 字段 | 当前默认 | 调整作用 |
|---|---:|---|
| `steps` / `batch_size` | 40000 / 16 | 优化器更新总数 / 每次使用的样本窗口数 |
| `image_size` | `[640,480]` | 模型全画面输入的宽、高，单位像素 |
| `model.observation_horizon` | 2 | 同一摄像头过去至当前的2个合法观察锚点；不增加相机，不移动当前动作的因果起点 |
| `model.down_dims` | `[256,512,1024]` | U-Net三层通道；从工程轻量版扩大，选作者类构造器默认值，不冒称视觉YAML唯一默认 |
| `model.prediction_horizon` | 30 | 预测未来动作数量，本工程正式Pack接口使用K30 |
| `model.action_horizon` | 8 | 后续闭环滚动执行时计划取用的前8个目标；当前训练仍监督全部有效K30 |
| `model.diffusion_steps` | 100 | 从干净动作到噪声的100级扩散时间表 |
| `model.inference_steps` | 10 | 每次生成一段动作使用10次DDIM去噪计算 |
| `validation_batches` | `null` | 检查完整验证集；设正整数可限制为前N批工程快速检查 |
| `validation_sample_batches` | 2 | 验证时额外生成动作并统计误差的批数，设0可只统计噪声MSE |
| `num_workers` / `precision` | 2 / `bf16` | Windows异步数据读取进程数 / CUDA数值精度 |

已完成640×480、batch16的ACT参数更新和DP真实物理工程Pack短训、保存恢复检查；这些是接口验收，不是正式模型训练或任务成功率结果。

| 参数 | 含义 |
|---|---|
| `--config <JSON>` | 替换默认 DP 配置；新运行时使用 |
| `--steps N` | 覆盖新运行的总更新预算，不是 epoch 数 |
| `--output <新目录>` | 指定新运行目录；已有目录拒绝覆盖 |
| `--check-only` | 只检查 Pack 和样本张量，不创建训练 run |
| `--full-verify` | 可选慢检查，额外重算文件哈希并核对当前任务源码来源 |
| `--stop-after N` | 本次额外运行N次更新后保存退出，冻结的总预算保持不变 |
| `--engineering` | 明确允许 scripted_test 工程 Pack；输出隔离在 `.local/checks` |

默认只做必要的 Pack 格式、文件尺寸、冻结身份、split、因果标签和合法序列接口检查，不重复重算全部大文件哈希，不访问外部 Raw/Prepared；这不等于完整内容审计。需要严格审计时显式使用 `--full-verify`。正常训练输出 `outputs/training/<task>/diffusion/<run_id>/`，含 `config.json`、`data_binding.json`、`run.json`、`metrics.jsonl`、`tensorboard/`、`checkpoints/{best,last}.pt`、`completion.json`。

```powershell
$dpRun = '<实际 outputs/training/任务/diffusion/run_id>'
Get-Content "$dpRun/completion.json"
Get-Content "$dpRun/metrics.jsonl" -Tail 5
& $train -m tensorboard.main --logdir "$dpRun/tensorboard" --port 6006
```

现有 `training.papera.analyze_checkpoint`、ACT 绘图和 `rollout` 不支持 DP checkpoint。DP 的独立闭环执行与成功率评估需后续接入验证，不能把 DP 权重交给现有 ACT Rollout。

### 4.2 64条选样的固定实验参数与队友对齐

本轮调整只发生在数据质量决定及正式JSON参数：排除指定8条；DP历史1→2，U-Net通道 `[128,256,512]`→`[256,512,1024]`。ACT配置数值不变，不修改两个模型的损失和训练数学实现，不启动正式训练。两份正式参数文件为 `config/training_baseline.json`、`config/training_diffusion.json`；选样目录同时保存它们的快照，不另造一套训练入口。旧checkpoint恢复使用其中冻结的配置，不能拿旧DP权重冒充本轮H2/新宽度模型。

| 条件 | ACT Baseline / ACT Gesture主干 | DP Baseline（DP Gesture尚未实现） |
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

`--resume` 适用于 Baseline/Gesture，恢复优化器、scheduler、RNG、sampler、归一化和数据来源，不接受新的 steps 或 Pack。`--continue-from` 保留历史 Baseline 续训行为，目前仅支持 Baseline，保持父 checkpoint 学习率并派生新 run；不要把只延长一组的结果与另一组当公平预算。未完成的半个梯度步不保存，应恢复最后完整 checkpoint。

源码、任务、Pack、split 或归一化变化都会拒绝恢复。恢复训练不是重新读取 CURRENT，也不是自动加载“最新文件夹”。

### 5.1 DP 中断恢复

```powershell
& $train -m training.diffusion --task $task --resume "$dpRun/checkpoints/last.pt"
```

DP 恢复同一 run 中的模型与 EMA、优化器、scheduler、sampler、RNG、归一化、Pack路径、split 和预算；输入必须是 DP 的 `last.pt`。不同时传 `--config`、`--steps` 或 `--output`，不更换 Pack。可加 `--stop-after N` 限制本次更新数用于工程验证。默认恢复仍做格式和接口检查；`--full-verify` 可额外核验内容哈希。DP 本轮没有 `--continue-from` 延长预算入口。

## 6. 模型检查、预览与50次评估

本节仅适用于 ACT Baseline/Gesture。DP 训练入口可做离线验证，当前 `rollout` 未接入 DP 的历史缓存和滚动执行。

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

## 7. 可重复的工程检查（不用于论文结果）

对正式真人Pack可先做不写文件的接口检查，再按需要做至多每算法3次独立工程更新。工具读取两份实际JSON配置，当前ACT H1、DP H2，默认使用各自batch16；不是重新裁剪或正式40,000步训练：

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
& $collect tests/pipeline_smoke.py --task tray_transfer --sessions 2
& $train dataset/manage_dataset.py --task tray_transfer --engineering process
$engineeringRoot = '.local/checks/pipeline/outputs/dataset/tray_transfer'
$curation = (Get-Content -LiteralPath "$engineeringRoot/CURATION_CURRENT.json" -Raw -Encoding UTF8 | ConvertFrom-Json).selected_version
& $train dataset/manage_dataset.py --task tray_transfer --engineering prepare --run $curation --review-mode auto
$prepared = (Get-Content -LiteralPath "$engineeringRoot/PREPARED_CURRENT.json" -Raw -Encoding UTF8 | ConvertFrom-Json).path
& $train dataset/manage_dataset.py --task tray_transfer --engineering pack --prepared $prepared
$testPack = (Get-Content -LiteralPath "$engineeringRoot/TRAINING_PACK_CURRENT.json" -Raw -Encoding UTF8 | ConvertFrom-Json).path
& $train dataset/manage_dataset.py --task tray_transfer --engineering verify-pack --pack $testPack
$testRun = '.local/checks/training/my_smoke_baseline'
& $train -m training.papera --task tray_transfer --variant baseline --pack $testPack --smoke-test --output $testRun
& $train -m rollout --task tray_transfer --engineering --mode preview --run $testRun --max-seconds 2 --seed 910001 --headless
```

`--smoke-test` 固定3步、较小工程验证范围；`--engineering` 只允许显式工程数据/短预览，不允许正式50次评估。工程采集结果明确保存为 scripted_test / failure，证明记录通路，不伪称成功示范。重复生成前保留或归档旧工程目录；不同 task_binding 不会混合。无需为了迁移重新处理旧视频。

同一工程 Pack 可检查 DP，并验证短训练与恢复；使用新输出目录，预算仍保持默认40,000步，以下只各运行2次更新：

```powershell
$testDpRun = '.local/checks/training/my_smoke_diffusion'
& $train -m training.diffusion --task tray_transfer --pack $testPack --engineering --check-only
& $train -m training.diffusion --task tray_transfer --pack $testPack --engineering --stop-after 2 --output $testDpRun
& $train -m training.diffusion --task tray_transfer --resume "$testDpRun/checkpoints/last.pt" --stop-after 2
```

工程属性随 checkpoint 恢复，不会变成正式训练；`completion.json` 的停止状态与完成总预算分别记录。上述工程命令不验证 DP 自主成功率。

### 7.1 高侧壁托盘到高台：tray_to_platform

这是独立的新任务，场景ID为 `HIGHWALL_PLATFORM`。原 `tray_raised_placement` 仍是薄托盘和5 cm高台，`lift/push`仍为development占位；不要把三者当作同一个场景。

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
| `START-PAPERA.cmd` | 在工程根目录双击，输入完整 task_id，例如 `tray_raised_placement` 后回车 | 调用 `scripts/launch.ps1`，进入采集任务菜单；启动 MuJoCo 后端、Node 网页服务、浏览器；不会自动启动摄像头、录制、训练 | 网页默认8014，后端5014；日志 `logs/<task_id>/<时间戳>/`；保留终端窗口 |
| `collection/START-COLLECTION.cmd` | 双击 | 转交根目录的 `START-PAPERA.cmd`；功能完全相同，不是另一套采集器 | 两个CMD任选一个，不要重复双击占用同一端口 |
| `scripts/launch.ps1` | CMD内部调用；日常无需单独双击 | 读取 `config/environments.json`，设置解释器/PATH/UTF-8、移除旧PYTHONPATH，再运行采集菜单 | 不安装环境，不自动关闭其他服务 |
| `python -m collection list` | 使用 `$collect` 运行 | 只列出注册任务及ready/development状态 | 不打开网页或仿真 |
| `python -m collection start --task ...` | 使用 `$collect`；可指定种子、操作者和端口 | 跳过交互菜单，启动指定任务的同一套采集服务 | 与CMD相同；`--no-browser`不打开浏览器，`--headless`不显示MuJoCo窗口 |
| `dataset/manage_dataset.py` | 使用 `$train`，指定 `--task` 和子命令；`preview-menu`可交互选择任务 | `process/reclip`处理，`review`可写审核，`preview`只读裁剪预览，`prepare/pack/verify-pack`导出和核验 | `outputs/dataset/<task_id>/`；审核默认8026，不控制机器人 |
| `dataset/PREVIEW-CURATION.cmd` | 双击，选任务与菜单1/2/3；可传 `-Task tray_to_platform -Port 8026` | 调用 `start-preview.ps1` 配置训练环境，再调用同一 `manage_dataset.py preview-menu`；选1/2仅预览，选3才处理数据；不训练 | 浏览器自动打开，只读预览；终端显示版本与实际端口，Ctrl+C退出 |
| `python -m training.papera` | 使用 `$train`；显式指定task、variant、Pack，或指定resume checkpoint | 启动训练；Baseline/Gesture共用入口，默认40,000步；不会自动打开采集网页 | `outputs/training/<task>/<variant>/<run_id>/`；按第5节恢复 |
| `python -m training.diffusion` | 使用 `$train`；指定task和Pack，或DP last.pt | DP默认40,000步；`--check-only`只读检查；`--stop-after N`保存并暂停本次更新 | `outputs/training/<task>/diffusion/<run_id>/`；按第5.1节恢复 |
| `python -m rollout` | 使用 `$train`；显式指定task、mode、run/Pack | `check`只检查，`prepare`只固定初态清单，`preview`执行一次，`batch`批量执行，`summarize`汇总 | `outputs/rollout/<task>/<variant>/`；默认headless，视频须等完成后播放 |
| `python -m training.papera.analyze_checkpoint` | `& $train -m training.papera.analyze_checkpoint --run $run` | 读取checkpoint做离线指标分析，不继续训练 | `<run>/analysis/evaluation/` |
| `python -m training.papera.plot_training_analysis` | `& $collect -m training.papera.plot_training_analysis --run $run` | 读取已有训练/分析结果生成图表 | `<run>/analysis/figures/` |
| `scripts/configure_vscode.py` | `& $train scripts/configure_vscode.py` | 仅生成/更新`.vscode`快捷配置，不自动运行训练 | 解释器路径变更后执行一次即可 |

上表的Python命令均从 `D:\轻量化数据采集\paperA` 执行；`$collect`、`$train` 的赋值见第0节。不要在资源管理器里直接双击 `.py`，以免使用错误解释器。新工程不使用旧工程的 `START-TASK3.cmd`、`START-DATASET.cmd` 或旧训练入口。

### VS Code 的运行项

按F5之前，在“运行和调试”下拉选择目标项。**不要把运行当前Python文件的三角按钮与命名的F5配置混淆。**

| F5名称 | 实际含义 | 需要输入 |
|---|---|---|
| `Paper A · Collection` | 启动采集，使用mujoco_repro | task_id |
| `Paper A · Baseline 40000 steps` | 开始正式Baseline训练 | task_id、已验证的正式Pack绝对路径 |
| `Paper A · Gesture 40000 steps` | 开始Gesture辅助监督训练 | 与Baseline相同的task_id及Pack |
| `Paper A · Resume interrupted run` | 恢复同一次中断训练，不改变预算 | task_id、完整`checkpoints/last.pt`路径 |
| `Paper A · DP Train 40000 steps` | 按DP配置开始训练，默认40,000次更新 | task_id、正式Pack绝对路径 |
| `Paper A · DP check` | 只检查DP数据接口与张量，不启动训练 | task_id、正式Pack绝对路径 |
| `Paper A · DP Resume` | 恢复同一次DP训练，不改变预算 | task_id、DP的完整`checkpoints/last.pt`路径 |
| `Paper A · Rollout check` | 核对模型、数据和场景，不执行仿真 | task_id、训练run目录 |
| `Paper A · Rollout preview 60 s` | 执行一次最多60秒仿真 | task_id、训练run目录 |

### 启动采集、检查连接、正常退出

1. 双击两个采集CMD中的任意一个；输入ready任务ID。等待终端打印 `Page:`、`Raw:`、`Logs:`。
2. 打开**本次终端打印的Page地址**；默认 `http://127.0.0.1:8014/dual_arm.html`。更新网页代码后按 **Ctrl+F5**。不要沿用旧工程8013页面。
3. 网页应先显示“仿真已连接”，此时摄像头尚未开启；再点击“启动摄像头”、授权、双掌绑定。MuJoCo窗口打开本身不保证网页状态渲染正常。
4. 退出前先保存或放弃正在录制的一轮，等待视频保存完成，再点击网页“结束采集并退出仿真”；也可在未录制/保存时于启动终端按Ctrl+C。不要在保存中强关进程。
5. 如果后端已经关闭，只刷新网页无法恢复仿真。关闭原采集终端，重新双击启动器。端口仍占用时先核对所属服务，不要无差别结束所有Python/Node进程。

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
