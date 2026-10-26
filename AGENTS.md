# Paper A 工程维护

先读 `README.md`。新增任务读 `docs/ADD_TASK.md`；修改模块接口读 `docs/DATA_CONTRACT.md`；执行与验证按 `docs/RUN_PIPELINE.md`。实际迁移证据在 `docs/MIGRATION.md` 和 `.local/checks/`。

本工程独立于只读来源 `../paperA_task3`。所有运行代码、资源、测试从当前工程解析；来源工程与外部文档备份只用于审计。历史数据和实验仍留在原工程。

任务通过 `tasks/*/task.json` 注册。Collection、Dataset、Training、Rollout 必须验证同一 task_binding；新增任务语义放在任务 adapter，State/Action Schema 不兼容时明确拒绝。

本轮迁移保持遥操作、物理保护、成功判据、Safe Stitch 核心与 ACT 数学实现不变。修改这些行为属于独立实验，先说明影响并获得用户授权。

正式数据位于 `datasets/<task_id>`，正式产物位于 `outputs/<stage>/<task_id>`。工程输入只允许标记 scripted_test，并放在 `.local/checks`；失败工程通路验证不代表真人任务成功。测试不得覆盖原始记录或历史权重。

环境通过 `config/environments.json` 管理。采集与物理测试使用 mujoco_repro；数据、训练和推理使用 paperA_train。代码修改后运行相关回归和 `tests/run_regressions.py`；接口变更需验证 `tests/pipeline_smoke.py` 到 Pack、短训练及短 Rollout 链路。

保持同一套正式入口。不要把来源工程中的一次性脚本、旧缓存或兼容启动器重新加入。长训练、批量评估、外部发布与 GitHub 推送由用户明确启动。
