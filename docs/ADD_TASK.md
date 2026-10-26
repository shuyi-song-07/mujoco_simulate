# 接入一个新任务

当前支持共享 Panda 双臂、S16/A14 和单 oblique 模型输入。新物体/任务可以复用执行器；不同机器人自由度、Action Schema 或传感器不能静默套用这个接口。

## 1. 建立目录并注册

先复制一个已有 **任务定义**，不是整个工程或控制器。例如从桌面托盘任务创建新实验：

```powershell
Set-Location 'D:\轻量化数据采集\paperA'
$envs = Get-Content config/environments.json -Raw | ConvertFrom-Json
$collect = $envs.collection_python
New-Item -ItemType Directory tasks/new_task
Copy-Item tasks/tray_transfer/task.json tasks/new_task/task.json
Copy-Item tasks/tray_transfer/scene.json tasks/new_task/scene.json
Copy-Item tasks/tray_transfer/scene.py tasks/new_task/scene.py
```

编辑 `tasks/new_task/task.json`：

- `task_id` 改为 `new_task`，必须与目录名一致；`display_name` 自定；`version` 为场景版本。
- 开发期间 `status="development"` 并加 `reason`；只有完成加载与物理检查后才改为 `ready`。
- `scene_id` 是场景标签，可用 `NEW_TASK`，不要求 S1/S2 枚举。
- `scene_config="tasks/new_task/scene.json"`。
- 七个 hooks 均指向 `tasks.new_task.scene:函数名`。
- `source_files` 列出 `tasks/new_task/scene.py` 及其自定义任务代码；共享逻辑也须列入，确保变动能被指纹发现。
- 共享 Panda 模型可保留当前 `model_xml`、`mesh_directory`。新模型放 `tasks/new_task/assets/`，更改这两个路径；mesh 目录中的实际资源参与绑定。
- `controller="dual_panda_v1"` 与 `schema` 保留。`evaluation_config` 可引用共享配置，或指向该任务自己的评估 JSON。

Registry 实现是 `tasks/registry.py`，扫描 `tasks/*/task.json`。不会根据 XML 文件名自动推断任务；未知 ID、缺字段、缺资源、错误 hook、development 状态均明确失败。

## 2. 场景参数与工厂

`scene.json` 保存完整 `scene` 对象。已有字段包括桌面、基座、物体、抓柄、目标、随机范围和可选平台。长度使用米，旋转使用弧度；例如 5 cm 写 `0.05`。`scene.randomization` 控制物体/目标 XY 范围及 yaw 范围。目标是否随机由该任务参数决定，不由 Dataset 再随机化。

共享工厂入口是 `simulation/mujoco/dual_arm/task_controller.py::create_model`：

1. 读取 Task 指定 XML；
2. 解析 XML，调用任务 `configure_xml(root, config)`；
3. 从指定 mesh 目录构建 MuJoCo VFS；
4. 编译后返回 `mujoco.MjModel`。

任务的 `configure_xml` **原地修改 XML root，不返回另一套控制器**。目前托盘实现调用 `task_scenes.configure_paper_scene`。同类托盘只改 `scene.json` 即可；不同物体则在自己的 `scene.py` 实现几何和命名适配。共享执行器依赖具名左右 Panda 关节、末端 site、手指、物体 free joint、抓柄和目标；可从现有 XML、`DualArmTaskController.__init__` 以及 `contact_control.py` 核对必需名称。不能只替换任意 XML 就假定兼容。

## 3. 必须实现的任务 hooks

参考 `tasks/adapters/tray.py` 和 `tasks/tray_transfer/scene.py`：

| hook | 实际签名/职责 |
|---|---|
| `initial_state(config, seed)` | 返回 seed、object_position、object_yaw_rad、object_quaternion_wxyz、target_position、target_yaw_rad；Collection 和 Rollout 共用 |
| `configure_xml(root, config)` | 设置任务几何、目标区域和相机等；保留共享执行器需要的命名契约 |
| `reset(controller, object_position, yaw)` | 设置任务附加障碍等；基础物体和目标初始化已由共享控制器处理 |
| `collection_success(contact_control, controller, now, object_pos, object_rot, any_contact)` | 任务终态检查，更新 DONE/完成计时；共享碰撞、失抓等物理失败保护仍有效 |
| `rollout_monitor(config, initial_state, **options)` | 返回评估 monitor；需提供既有 `SuccessMonitor` 的 `update(controller)` 返回终态与物理证据字典的契约，具体调用见 `rollout/rollout.py` |
| `preflight(config, seeds, demonstration_seeds=(), engineering_seeds=())` | 任务专属初态、可达性与相机检查，返回 cases/summary；保留全部种子及异常标记 |
| `dataset_adapter()` | 返回含 `validate(metadata)` 与 `build_evidence(episode, curation_config)` 的对象；把任务事件转换成统一事实证据，交给未改动的裁剪和 Safe Stitch 核心 |

几何和成功语义相同的托盘可以直接复用这七个函数；目标支撑物或完成条件改变时，应覆盖对应任务 hook。**纯推移**的完成条件、接触含义不等于“曾双抓后释放”，必须编写对应 success/monitor/evidence adapter；因此占位 `push` 目前不能采集。不要在 `curation_rules.py` 添加 task_id 分支。

需要新增任务特有失败/终止语义时，由任务 monitor/success hook 明确判定并记录；不要绕过共享碰撞、IK、限位保护。若新任务必须改变控制自由度或物理执行方式，已经超出现有 controller/schema，需要显式注册新控制器和适配器，不属于“仅换场景”。

## 4. 复用与不可复用的边界

- 复用：网页双手识别、100 ms 掌拳/ILY 控制、身份绑定、动作协议、Panda IK/限速、Recorder、快照、相机、S16/A14。
- 任务负责：物体/障碍/目标几何、随机初态、附加 reset、成功/任务终止、离线事件含义。
- ACT 主体不变：只要仍为单 RGB、S16、A14 和相同单位。Task Pack 必须有完整辅助标签和 Mask 才可做 Gesture 对照；缺标签不能伪造。
- 不同 Schema：Registry 当前会拒绝。需新增具名 Schema 与 Raw/Pack/训练/执行适配并测试，而不是修改 manifest 字符串绕过检查。
- 当前 Dataset 适配点在事实证据层；自动裁剪、合法缺口和硬边界仍由同一核心计算。改变任务成功语义时必须检查事件保护输出，不能声称所有任务无需验证即可复用。

## 5. 最小验收顺序

将新任务开发状态改为 ready 后，在 **新工程** 运行：

```powershell
& $collect -m collection list
& $collect -c "from tasks.registry import get; from simulation.mujoco.dual_arm.task_controller import DualArmTaskController; c=DualArmTaskController(get('new_task').config()); c.reset(123); c.step(1/60); print(c.get_status()['task_state'])"
& $collect -m collection start --task new_task --seed 123 --operator-id operator01
```

先不录制，验证物理接触、相机遮挡、绑定、移动、开合、暂停和成功/失败逻辑。再保存一条明确标注的工程示范及正式真人示范；不要把物理脚本标为真人成功。

正式记录输出到 `datasets/new_task/<session>`，失败是相应 `_failures` 会话，放弃记录按原 Recorder 的研究日志归档而不成为成功 Episode。后续直接将 [RUN_PIPELINE.md](RUN_PIPELINE.md) 的 `$task` 改成 `new_task`；输出、CURRENT 和 split 自动按任务隔离。

工程全链路可参考 `tests/pipeline_smoke.py`，改为调用 `capture('new_task', seed)`，明确 `scripted_test` 且放 `.local/checks/pipeline`。只有保存过的工程记录通过完整验证后才可使用 `--engineering` 构建测试 Pack。新增任务至少测试双相机/手身份异常、实际动作、末端限位、预期任务阶段、Prepared 对齐、Safe Stitch、两组短训练和同任务短 Rollout。

新任务注册动态发现已由 `tests/test_task_registry.py` 用临时 CUSTOM 场景验证。修改任务代码、场景 JSON 或资产后旧 Task Binding 会失配并拒绝混用；保留旧代码版本，或注册新的 task_id 后重新采集，不能用新场景解释旧 Pack。

## 6. 已实现示例：高侧壁托盘放置至 10 cm 高台

`tray_to_platform` 是独立任务；旧 `tray_raised_placement` 仍为原托盘放到 5 cm 高、25×33 cm 的平台，不被新定义覆盖。新任务的 `task.json` 注册 `HIGHWALL_PLATFORM`；几何、初态和成功参数放在 `tasks/tray_to_platform/scene.json`，建模/可达性入口是同目录 `scene.py`，成功判据是 `success.py`。`evaluation.json` 给出该任务评估协议。

所有长度是米、角度是弧度、质量是 kg、惯量是 kg·m²。以下为当前默认值；修改尺寸后需同步检查抓取点、目标高度和可达性：

| 对象/参数入口（相对 `scene`） | 当前值和含义 |
|---|---|
| `table_top_z` / `table_half_size_xy` | 桌面 Z=0.35，桌面完整尺寸 1.00×1.10；双臂基座保持原位 |
| `paper_object.pan_width_m` / `pan_length_m` | 托盘主体 X×Y=0.24×0.32，不含外侧抓耳 |
| `paper_object.bottom_thickness_m` | 实体底板 0.008 |
| `platform_task.side_wall_height_above_floor_m` / `end_wall_height_above_floor_m` | 底板上表面以上侧壁 0.05、两端加强壁 0.06；托盘最高总厚度 0.068 |
| `paper_object.wall_thickness_m` / `platform_task.end_wall_thickness_m` | 普通侧壁厚 0.008、端壁厚 0.010 |
| `paper_object.handle_width_m` / `handle_length_m` / `platform_task.grip_height_m` | 两个实际碰撞抓耳各为 X×Y×Z=0.036×0.12×0.052；Panda 总开度仍为 0～0.08 |
| `object_mass_kg` | 0.35；7 个不重叠实体盒按体积分配质量，MuJoCo 自动合并质心和惯量 |
| `placement_platform.height_m` / `extra_size_xy_m` | 平台高 0.10，主体长宽各加 0.04；实体台面 0.28×0.36 |
| `platform_task.minimum_lift_m` / `lift_stable_seconds` | 有效双抓后离桌至少 0.03，持续 0.12 秒；用于任务历史证明 |
| `platform_task.transport_clearance_m` | 工程路径预留底面高于台顶 0.025；不自动代替操作者抬升 |
| `platform_task.minimum_open_fraction` | 两爪指令及实际总开度至少最大开度的一半；还须无手指接触、真实平台支撑和停稳 |

托盘 body 原点是最高外形的几何中点，**不是质心**。当前局部质心约 `[0, 0, -0.011088]`，主惯量约 `[0.0096374, 0.0081023, 0.0017998]`；这些数值由实际几何和质量计算，不是另外写入的经验惯量。底板、四壁、两抓耳都有碰撞几何，物体使用自由关节，没有 weld 或外力跟随。

左右 grasp site 在托盘局部坐标分别为 `[0,-0.22,0]`、`[0,+0.22,0]`；方向四元数为 `[0,√0.5,√0.5,0]`（wxyz）。Panda 手指沿末端局部 Y 开合，此方向对应托盘局部 X，夹住 36 mm 厚的抓耳；末端局部 Z 向下。托盘 Yaw 改变时，抓取方向随托盘旋转。进入和退出区域在托盘两端外侧，不能把侧壁顶面当成抓取点。

平台中心为世界 `[0.16,0,0.40]`、Yaw=0.08；**台顶 Z=0.45，目标托盘 body Z=0.484**。因为托盘底面在 body 局部 Z=-0.034，目标高度由 `target_surface_z + object_half_size[2]` 得到，并由任务初始化核验，不能把 0.45 直接当成 body 高度。工程搬运路径的 body Z=0.509，给底部留出 25 mm 越台间隙。

初始 body 中心为 `[-0.16,0,0.3842]`；`randomization` 仅扰动 X±0.008、Y±0.02、Yaw±0.04，目标固定。相同 seed 可复现，不沿用大范围随机化。扩大范围或改变台面后，必须重跑初始穿模、IK 和真实搬运检查。当前三视图为 top/front/oblique；oblique 和 front 保持原参数，旧 side 仅保留历史重放兼容。仅本任务辅助top中心X从0.04改为0，正交垂直范围从0.62增至0.66 m，避免初态侧壁裁切；其他任务的相机不受影响。

```powershell
& $collect -m collection list
& $collect -m unittest tests.task3.test_highwall_platform -v
& $collect -m collection start --task tray_to_platform --seed 0 --operator-id operator01
```

正式示范写入 `datasets/tray_to_platform/<session>`。当前已通过几何/质量、20 个 seed 的初始化/正式预检（IK、工作区、四视角范围均无标记）；seed 0、1、2、3、17 已以真实接触完成抓取、抬升、搬运、落台、释放到 DONE，未发生碰撞。最终参数的演示在 `.local/checks/tray_to_platform/final_acceptance/`。**这不等于真人遥操作已验收，也不等于所有随机初态都已动态通过**。先亲自试操，再开始正式采集。
