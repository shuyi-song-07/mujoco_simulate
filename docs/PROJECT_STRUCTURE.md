# 工程目录与任务扩展

本次整理只调整文件归属、导入路径和任务分派。场景参数、手势策略、物理控制、裁剪规则、网络数学和成功判据沿用重构前版本。正式 `datasets`、`outputs` 和历史权重不搬动、不重写。

## 1. 按环节找代码，再按任务找差异

```text
paperA/
├─ tasks/                         场景、参数、成功判据和任务注册
│  ├─ push_t/                     单臂推 T
│  ├─ tray_to_platform/           双臂抓取托盘并放到高台
│  ├─ h_frame_push_align/         双臂推 H
│  ├─ push_r/                     双臂推 R
│  ├─ registry.py                发现任务、检查来源
│  ├─ stages.py                  发现各环节的任务适配入口
│  └─ runtime.py                 调用选定的控制器和场景工厂
├─ collection/
│  ├─ cli.py                     采集总入口
│  ├─ common/                    共用采集适配器、双臂网页和识别组件
│  ├─ push_t/                    单臂采集适配器及 web/ 页面
│  ├─ tray_to_platform/          托盘采集入口
│  ├─ h_frame_push_align/        H 形任务采集入口
│  └─ push_r/                    R 形任务采集入口
├─ dataset/
│  ├─ manage_dataset.py          数据处理总入口
│  ├─ common/                    Audit、裁剪、Safe Stitch、审核、Prepared、Pack
│  ├─ push_t/                    单臂推移证据和处理入口
│  ├─ tray_to_platform/          双臂抓取证据和处理入口
│  ├─ h_frame_push_align/        H 形双臂推移证据和处理入口
│  └─ push_r/                    R 形双臂推移证据和处理入口
├─ training/
│  ├─ __main__.py                按任务和模型启动训练
│  ├─ common/                    共用训练任务配置接口
│  ├─ act/                       ACT 网络、官方组件、读取器和训练循环
│  ├─ diffusion/                 DP 网络、官方组件、读取器和训练循环
│  ├─ push_t/                    本任务支持的模型与维度配置
│  ├─ tray_to_platform/
│  ├─ h_frame_push_align/
│  └─ push_r/
├─ rollout/
│  ├─ common/                    权重检查、动作块、共享双臂执行器和评估流程
│  ├─ push_t/                    单臂执行器、覆盖率验收与阶段说明
│  ├─ tray_to_platform/          托盘回放适配入口
│  ├─ h_frame_push_align/        H 形回放适配入口
│  └─ push_r/                    R 形回放适配入口
├─ simulation/mujoco/
│  ├─ common/                    录制、快照渲染、手部测量、服务与日志
│  ├─ single_arm/                现有单臂物理控制
│  ├─ dual_arm/                  现有双臂物理控制
│  └─ assets/                    共享模型资源
├─ common/                       旧路径兼容与本次精确迁移记录
├─ config/                       共享默认配置和本机环境路径
├─ datasets/<task_id>/           原始示范，注意是复数 datasets
├─ outputs/{dataset,training,rollout}/<task_id>/
├─ logs/<task_id>/
├─ tests/                        回归检查
└─ .local/checks/                工程验证产物，不作为正式示范
```

`dataset` 是处理代码，`datasets` 是采集的数据。不要把原始示范放到代码子目录中。

每个环节的 `adapter.py` 是该任务的接入点。小文件直接调用公共实现是有意设计：例如三种双臂任务复用同一控制器，但抓取与推移分别解释接触、释放和完成证据。无需给每个任务复制整套裁剪算法、ACT 或 DP。

训练同时有“任务”和“模型”两个维度。任务目录决定可用模型、状态/动作维度及默认配置；`act/`、`diffusion/` 各自只保留一份网络实现。当前 Push-T 是 S9/A7，仅接入 ACT；三个双臂任务是 S16/A14，可用 ACT/DP。这次没有增加单臂 DP 功能。

`collection/teleoperation/mediapipe/` 暂时保留 Node 依赖、MediaPipe 模型和一个旧服务器入口；实际网页源码已经移到 `common/web/` 或任务的 `web/`。这避免无必要地重装依赖和复制模型资源。

## 2. 新增任务怎么做

1. 新建 `tasks/<新任务名>/`，定义 `task.json`、场景、随机初态与真实成功条件。任务名必须在各环节一致。
2. 新建 `collection/<新任务名>/adapter.py`，声明 schema、控制器、页面和服务入口，提供 `create_model()`、`create_controller()`。复用已有双臂控制时可导入 `collection.common.adapters.dual_panda_v1`；专用页面放在本任务 `web/`。
3. 新建 `dataset/<新任务名>/adapter.py`。通常从 `dataset.common.pipeline` 导入 `audit/reclip/prepare/pack/verify_pack`，把自己的 `evidence.py` 工厂暴露为 `adapter_factory`。任务差异优先放在事实证据解释中，不在公共裁剪规则里堆任务名判断。
4. 新建 `training/<新任务名>/adapter.py`，声明支持哪些模型，复用共用配置函数；只有确实改变模型输入契约时才新增相应读取/维度适配。
5. 新建 `rollout/<新任务名>/adapter.py`，指定执行器、验收配置检查与阶段说明。实际完成判据仍由 `tasks/<新任务名>/` 的 monitor 提供。
6. 运行任务加载、真实物理接触、裁剪/Pack、短训练和短回放检查，最后由真人验收。工程样本必须标明 `scripted_test` 并保存在 `.local/checks/`。

调度器自动寻找 `<环节>.<task_id>.adapter`，不需要在总入口添加任务枚举。已有同构双臂任务在未提供专属入口时可继承公共适配器，原 manifest hook 保持可用；建议新任务显式建这四个目录，便于查看。

**新增目录不是任意机器人的万能适配。** 当前单臂控制器和网页专用于 Push-T，其他单臂任务必须提供自己的采集和回放入口，否则明确拒绝。新增关节、动作维度、机器人类型或不同控制方式，需要实现真实接口并验证，不能仅改任务名冒充支持。

详细场景接口仍见 [ADD_TASK.md](ADD_TASK.md)。

## 3. 运行方式

既有采集、裁剪和回放命令继续有效。新代码阅读与新增调用使用新路径：

```powershell
# 在工程根目录，使用 config/environments.json 中对应解释器
& $collect -m collection start --task push_t
& $train dataset/manage_dataset.py --task push_t locations
# 只检查 Pack；以下命令不启动训练
& $train -m training --task push_t --model act --pack $pack --check-only
& $train -m training --task tray_to_platform --model diffusion --pack $pack --check-only
& $train -m rollout --help
```

也可直接使用 `python -m training.act ...` 和原有 `python -m training.diffusion ...`。具体采集、训练、恢复、评估参数见 [RUN_PIPELINE.md](RUN_PIPELINE.md)，不要把示例中的 `$pack` 当作自动选择最新数据。

## 4. 旧数据、旧命令和来源检查

旧 `dataset.task3`、`training.papera`、顶层 `rollout.*` 以及已迁出的 `simulation.mujoco.dual_arm.*` 导入由 `common/layout.py` 映射到新模块。新旧名称使用同一个模块对象，没有维护两份算法；旧 `python -m` 入口也保留。新开发请使用实际新目录。

旧记录中的逻辑源码路径仍可解析，但读取和计算指纹的是实际新文件。`common/layout_migration.json` 冻结本次搬迁前后哈希和完整来源清单；只有这一次已验证的结构调整可以通过兼容检查。它不是忽略来源检查的开关，缺项、额外来源或未审核的代码变更不会被当成搬目录。原来就与当前版本不兼容的场景或旧模型，不会因此变得兼容。

新录制的来源指纹还包含任务适配器和调度依赖，避免以后改了任务入口却继续声称旧数据来自相同实现。不要手工改历史 metadata、Pack、checkpoint 的哈希，也不要随意重新生成迁移证明来绕过错误。

## 5. 本次验证与回溯

重构前版本备份标签：`backup-before-task-layout-20261113-225923`；该版本包含用户刚提交的接触受阻退让修复。工作继续保存在现有 `feat/single-arm-push-t` 分支，不上传远程。

已逐项对照：四任务固定初态下的物理状态与三路画面；五份输入的完整裁剪输出；ACT 单臂/双臂预测及 loss；三种策略执行器；官方 ACT/DP 核心文件。临时网页服务、真实 HTTP 资源、Windows 多进程录制保存、历史 Pack 和权重入口另外检查。验证报告集中在 `.local/checks/task_layout/` 与 `.local/checks/layout_refactor/`，最终汇总见 `task_layout/VALIDATION.json`。

最终主回归 685 项：682 通过、3 跳过。采集专项 94 项：92 通过，2 项为重构前已存在的 H 形任务问题（跨 ±π 角度成功检查、脚本真实接触归位）；在备份源码上复现了相同失败和数值，本轮没有改动其物理或判据。完整执行命令仍会因这两项返回失败，不能宣称全套回归全绿。

新增完整工程链路：两个独立 `scripted_test` 会话共 182 帧，经 Audit、裁剪、Prepared、完整 Pack 校验；ACT Baseline / Gesture 各 3 次更新、各 1 秒后台回放，六个视频全部解码并核对索引。这只验证接口连通，短回放以超时结束，不表示模型会完成任务。结束时保留原文件换行，减少无关 diff；AST 对照与最终来源检查另有记录。

旧 `dataset/task3` 目录如仍可见，只剩已忽略的 `__pycache__`；本次自动审批拒绝删除该缓存，没有遗留第二套活动源码。测试用基线 worktree 已移除，备份标签和对照报告保留。

真人摄像头操作和长时间正式训练不由这些工程测试替代。正式采集前，重启原服务并刷新网页，做一次短的绑定、移动、保存试用。
