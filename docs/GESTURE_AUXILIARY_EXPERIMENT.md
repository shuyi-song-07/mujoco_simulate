# ACT / DP 的 Gesture 辅助监督对照

本实验只增加训练期的左右三分类监督。策略观察仍为单 oblique RGB 640×480 与 S16 机器人状态，输出仍为 K30、A14；机器人动作30 Hz、控制60 Hz、物理500 Hz。不得修改 Raw、裁剪、Safe Stitch、Action 时间对齐、物理保护或成功标准。

## 标签究竟表示什么

固定 Pack 为 `outputs/dataset/tray_to_platform/training_packs/auto_20261103_213247_870817`，64条、49 train / 15 val，按会话隔离。训练和验证都只消费官方 Pack Reader 已有的 `category_id/category_mask`，不从机器人动作、未来阶段或手骨架反推标签。

类别顺序固定为 `ILoveYou / Open_Palm / Closed_Fist`。这些字段来自**当前观察锚点关联的人手测量记录中的识别类别**，不是100 ms确认后控制器已接受的模式。具体证据为 `collection/teleoperation/mediapipe/hand-measurement.js` 记录 `gesture.categoryName`，`dataset/task3/frame_evidence.py::_auxiliary` 使用 `recorded_category_no_score_v1`，`prepare_paper_dataset.py` 原样映射类别。该冻结策略不再额外施加分类分数门槛；它保留原有时间因果性、测量年龄和日志先后检查。`accepted_intent` 是另一条控制事件来源链，不能把两者混称。`raw_label` 指手的原始左右身份标签，也不是三分类控制手势。

分类与动作不会完全一一对应：手势还需确认、命令接受、状态累积与控制器保护；一类手势可以出现在多个位置和阶段，也可以对应保持、移动或开合中的不同连续目标。`ILoveYou` 对应 XY 控制意图且保持爪宽，张掌与握拳分别对应 Z 控制并开爪/闭爪，但绝对 A14 目标还取决于历史状态、手的连续移动与实际执行。分类在过渡、低置信度或模式切换期间可能与已接受控制模式不同。

Unknown、未分配类别及不可用记录不新增第四类，不转换成静止，不沿用前一个类别，不用“有效骨架”恢复类别。分类 mask=false 时不贡献 CE 或分母。骨架有效性和分类有效性是不同事实；Unknown 的有效骨架可以保留研究用途，本对照不训练骨架或阶段头，也不改变数据删留。

## 网络与损失

ACT 沿用现有实现：ResNet18 → observation Transformer → 共享观察表征。左右三类头是同一个6输出线性层中两组独立参数，每组单独 softmax。它读 observation tokens 的均值，位于 CVAE posterior/latent 进入动作 decoder 之前。CVAE 只供主任务训练；辅助头无法看到未来动作、posterior tokens 或 z。推理固定 z=0，只接受 RGB 与 S16。现有 padding 在投影前清零并屏蔽 attention，动作 L1 与 KL 排除无效目标。无需改变 ACT 网络和 Pack 接口。

DP 沿用 ResNet18(GN)+spatial softmax、H2观察条件、Conditional U-Net、K30/A8；训练配置的采样默认DDIM10，本轮冻结部署为DDIM20、每次执行A8，两组一致。当前及上一帧的观察条件特征同时送给原去噪网络与新增左右三分类线性头；监督目标仅为当前锚点。分类分支看不到 noisy action、epsilon、扩散 timestep 或未来图像。观察编码只计算一次，梯度回传共享视觉编码器；推理不执行分类头、不需要标签。DP数据接口本来就继承当前锚点手势字段，增加H2时未改变其时间索引。

固定第一轮损失：

```
ACT baseline: L_action_L1 + 10 L_KL
ACT gesture:  L_action_L1 + 10 L_KL + 0.1 L_gesture_CE
DP baseline:  L_epsilon_MSE
DP gesture:   L_epsilon_MSE + 0.1 L_gesture_CE
```

`L_gesture_CE` 对有效左右手实例求和，再除以有效实例数；全mask时为可微零。0.1是事先固定的保守辅助系数，并非已经证明最优，也不是两个算法之间严格相等的梯度强度。第一轮不引入类权重、过采样、事件采样、focal loss、骨架或阶段loss。是否存在类别不平衡通过训练集计数及验证集每手每类混淆矩阵披露，同时报告macro recall/F1，避免整体accuracy掩盖少数类。如果后续确需重加权，应单独登记实验，只从训练集估计权重、只在val选择，不能追看闭环测试后修改本轮设定。

## 公平性与选择规则

同算法的 baseline / gesture 使用同 Pack、完整固定split、训练集归一化、seed、batch16、图像尺寸、观察历史、动作长度、优化器、精度、更新次数、验证频率和训练内best指标定义；闭环选型的搜索范围差异另行披露。ACT仍为训练集标准化，DP仍为训练集min-max；不要求跨算法强行统一归一化或观察历史。ACT H1、DP H2须在跨算法表格中明确披露。

ACT的全部variant本来就有同一套头，同seed初始化逐位一致。DP baseline不增加参数；gesture在全部公共参数初始化后创建新增头，并恢复全局CPU RNG，确保公共参数与后续训练噪声序列不因额外头初始化变化。两者均从原始初始化开始gesture训练，不能从已经学过40k步的baseline开始计gesture预算。

截至2026-11-04，四组训练均已完成：ACT baseline / gesture各60k，DP baseline / gesture各40k。ACT两组前40k均执行原500步warmup与40k cosine，后20k保持40k checkpoint已有最低学习率，完整继承优化器/采样器/RNG；gesture从原始初始化训练，未改成60k cosine。DP两组均使用原40k调度。完成记录见 [ACT baseline](../outputs/training/tray_to_platform/baseline/baseline_continued40000_to60000_20261104/completion.json)、[ACT gesture](../outputs/training/tray_to_platform/gesture/gesture_continued40000_to60000_20261104/completion.json)、[DP baseline](../outputs/training/tray_to_platform/diffusion/baselines_20261103_232447_579530/completion.json)、[DP gesture](../outputs/training/tray_to_platform/diffusion_gesture/gesture_seed20261016_40000_20261104/completion.json)。验证选型及四组最终50初态评估也已完成；最终测试未用于重新选择checkpoint或调整参数。

首段配置分别为 `config/training_act_gesture.json` 与 `config/training_diffusion_gesture.json`；ACT使用 `--variant gesture`，DP配置内外variant均明确为gesture。两份配置都从对应baseline配置复制，只增加辅助监督选择/权重及候选保存设置，训练由主任务统一调度GPU。ACT两组已通过同variant `continue-from` 完成后20k，不允许从baseline切换variant；正式续训前已显式核验父run的完整40k完成状态、非smoke/overfit和全量heldout验证。

ACT启动必须同时显式传 `--config config/training_act_gesture.json`。仅传 `--variant gesture` 仍会读baseline默认配置，未声明的gesture权重会回落到模型历史默认1.0，不属于本轮λ=0.1实验。运行后以冻结的 `run.json/config.json` 再确认实际权重。

ACT best按val的z=0 action L1选择；DP best保持按固定噪声/timestep的EMA val noise MSE选择，采样动作误差另作诊断。不能因为gesture分类好而改选best。候选checkpoint和离线指标可用val诊断；正式闭环测试初态、预算、物理判据与协议固定，不能用测试成功率调λ、选种子或挑checkpoint。本轮ACT实际部署为chunked/30，DP为DDIM20、chunked/8；两种执行协议不同，须披露，均沿用同一物理执行器。

探索性追加协议（exploratory addendum）：在观察训练/验证诊断后、尚未使用新最终测试初态前，ACT两组原候选40/45/50/55/60k各追加自己的 `best.pt`。best仅由整个60k训练轨迹的全量heldout、z=0 action L1选出；同一步候选去重。两组均用同一16个验证初态选择部署候选，不调整λ、训练数据或调度。保留原计划和此次追加记录，不能称追加规则在训练前已经预注册；手势CE上升本身也不证明动作策略过拟合。

DP选型搜索并不完全对称：baseline对checkpoint、DDIM步数及A4/A8执行长度做了有限搜索，两阶段共48条验证执行；gesture沿用baseline选出的DDIM20/A8，仅比较自身best/last各16个初态，共32条验证执行。最终执行设置相同，但两组搜索空间和验证执行预算不同。因此DP结果应解释为这两套已冻结训练与选型流程的比较，不能称为所有选型因素均严格对称的纯辅助loss消融。

实际部署权重步数与完成的候选搜索预算分别报告。例如选父run的40k `last.pt` 时，仍称40k权重；其60k子run仅证明该组完成了60k预算。汇总可显式传 `--act-baseline-completed-run` 或 `--act-gesture-completed-run`，核验子run记录的父路径、签名、last路径/步数/指纹、同模型与完整数据绑定、完成状态及最低学习率延续。没有显式参数就不推测目录关系，不复制或重签历史权重。

## 能支持的结论与局限

该监督没有把人手标签输入策略，也没有从未来动作路径回传到辅助头，因此不是显式的推理标签泄漏。它可能使机器人视觉表征携带“移动/抓取/松开”的可预测关联；也可能只学习爪宽或操作者习惯的捷径，并可能被识别噪声干扰。因而“可推广阶段先验”是待验证假设，不能由高分类准确率直接宣称。

本轮ACT/DP手势分支均出现训练CE下降、验证CE在后期上升，而分类准确率仍约90%的现象。这支持“辅助分类在对数损失上出现过拟合、类别判别准确率相对稳定”的表述。准确率只检查最大概率类别，CE还惩罚真实类别概率过低；少数错误预测变得更自信可以产生这一现象，但仅凭汇总指标不能确定原因或断言校准恶化。约90%的准确率说明标签在当前验证分布中具有一定可预测性，仍须结合左右手各类recall/F1和类别占比，不能据此认定模型理解了控制意图。

辅助分类的过拟合不等同于动作策略泛化变差；离线动作损失、分类CE和闭环成功率也不必同步。60k是ACT完成的训练预算，并不保证最后权重最适合部署。两组完成同预算后，在独立验证初态上按统一规则比较预先列出的候选，更贴近控制任务目标；选中早期权重仍须分别披露实际checkpoint步数与完整搜索预算。验证选型成绩不能充当无偏的最终性能，最终测试只评价已冻结的选择。

需要同预算baseline对照、held-out会话上的每手每类指标、同一固定闭环协议的结果共同支持效果。49/15划分只有4个训练会话和1个验证会话；相邻帧高度相关，60,035帧不等于60,035个独立样本。单seed只能作为探索性结果，不能声称跨seed稳定收益或跨操作者泛化。后续若做打乱标签负对照、移除机器人state的诊断或多seed，应作为额外实验，不混入本轮最小对照。

## 最终结果与只读复审

2026-11-04复审实际历史baseline冻结配置与两份gesture配置：除实验身份、辅助权重/头及每5000步候选保存外，同算法优化超参数一致；候选保存只改变磁盘产物。官方reader的窗口生成完全不读取variant或gesture mask，因此缺少类别的观察仍有同一动作学习资格。train/val归一化统计来自同一训练集合且不因gesture重新选样。DP头在CPU初始化后恢复CPU RNG，不改变同seed公共参数、采样器次序及随后扩散噪声draw；ACT也保留同seed公共初始化。测试证明小型CPU初始输出逐位一致，不承诺CUDA上全训练轨迹逐位复现；DP未强制deterministic algorithms，需保留环境版本。

四组最终评估使用相同的5811000～5811049初态、60秒仿真上限和既有物理成功判据，每组50条均完成，基础设施错误均为0。统一证据入口为[四组审计比较](../outputs/rollout/tray_to_platform/comparison_20261104/comparison.json)与[模型、配置、协议及结果索引](../outputs/rollout/tray_to_platform/comparison_20261104/manifest.json)。

| 组别 | 完成训练预算 | 所选权重步数 | 观察 / 执行 | 最终成功 | 相对同算法baseline |
|---|---:|---:|---|---:|---:|
| ACT baseline | 60k | 40k last | H1 / chunk30 | 45/50（90%） | — |
| ACT gesture | 60k | 40k last | H1 / chunk30 | 40/50（80%） | −10个百分点 |
| DP baseline | 40k | 40k EMA last | H2 / DDIM20、A8 | 17/50（34%） | — |
| DP gesture | 40k | 18k EMA best | H2 / DDIM20、A8 | 3/50（6%） | −28个百分点 |

本轮两种模型的手势辅助均未提高最终成功率。这是单个训练seed20261016、固定Pack、λ=0.1及上述选型流程下的负结果，不证明手势数据本身没有价值，也不支持把约90%的辅助分类准确率等同于有效动作先验。DP退化已出现在双抓和有效抬升阶段，不能只归因于最后松爪；共享编码器的梯度竞争、粗粒度识别标签与连续动作目标不完全对齐、有限会话中的捷径学习都是待验证假设，尚未由本轮对照区分。实际部署的DP gesture是18k（验证CE约0.340、分类准确率89.92%），不能用未部署的40k分类CE约0.681直接解释全部失败。

各初态的成功事实、失败原因、完成时间和干预情况保留在统一索引指向的原始结果中。以50个固定初态计算的区间仅针对所选checkpoint，不包含训练seed变化；相同初态下的配对比较也不能代替多训练seed验证。本轮保留负结果，不据最终测试重选权重或增加实验。

离线指标保持明确口径：ACT全val的z=0 L1与物理误差，DP全val的EMA noise MSE；DP现有采样动作误差只覆盖前2个batch，即32个固定窗口，不是全val。ACT `xyz_error_mm` 是每臂三维欧氏距离，DP `position_abs_m` 是逐坐标绝对误差平均，两者不能直接作为同一个位置指标比较。需跨算法横比时应另用完全同一批窗口、单位和归约重新计算，且不据测试成绩挑权重。

辅助部分报告左右手三类support、混淆矩阵、macro recall/F1、有效类别覆盖与原始CE；训练的0.1CE分量和主任务分量分别记录。baseline的未训练分类头accuracy不代表一个有效分类基线，无需拿其随机结果宣称监督改进。现有 `summarize_gesture_confusion` 可从保存的混淆矩阵离线计算宏指标。

若DP滚动步数或DDIM步数经val/工程诊断调整，应在正式测试前冻结同一推理设置给DP两组，保留训练默认与实际推理设置。best与last不可混称：训练best仍是原离线指标，固定最终步数只能作为另一个事先明确、两组一致的选择规则。DP部署身份必须为 `diffusion_gesture`，同时保留algorithm=`diffusion_policy`；历史缺失variant字段的baseline按默认baseline解释，不重写其原冻结配置或checkpoint。

## 证据与验证

覆盖审计入口 `scripts/audit_gesture_supervision.py` 通过 `verify/read_episode` 只读统计合法action anchors，按train/val、左右手、三类及Unknown骨架覆盖输出；见[完整覆盖报告](../.local/checks/gesture_auxiliary_design/coverage.json)。该检查使用轻量verify并冻结manifest SHA，不冒充全包大文件哈希审计。

2026-11-04已实际完成全64条官方读取：

| split / 手 | 合法观察数 | 有效类别覆盖 | ILoveYou | 张掌 | 握拳 | Unknown但骨架有效 |
|---|---:|---:|---:|---:|---:|---:|
| train 左 | 45,958 | 42,092 (91.59%) | 26,190 | 9,764 | 6,138 | 3,656 |
| train 右 | 45,958 | 41,476 (90.25%) | 25,687 | 9,716 | 6,073 | 4,317 |
| val 左 | 14,077 | 13,511 (95.98%) | 7,916 | 3,537 | 2,058 | 562 |
| val 右 | 14,077 | 13,156 (93.46%) | 7,588 | 3,487 | 2,081 | 904 |

每条训练和验证episode的左右手都包含全部三类。训练有效类别比例约62% / 23% / 15%，存在中等不平衡而非缺失类，最小对照采用未加权CE并报告每类指标。有效Unknown骨架共9,439个手实例，但其 `task_geometry_mask` 全为false；这些实例保留在Pack中，不进入本轮分类loss。train有3,355个锚点左右类别均无效，val为500个，仍保留原动作训练资格。

`tests/papera/test_gesture_auxiliary.py` 核验公共初始化/RNG、初始DP动作与epsilon loss不变、Unknown零梯度、无动作/噪声泄漏、辅助梯度只到观察编码器和head。既有ACT测试覆盖posterior泄漏、z=0推理、padding与初始化。工程测试只证明实现边界，不能代替训练与物理成功率。

最新统一回归共562项：558项通过、4项跳过；采集环境39项全部通过。4项跳过包括3项既有跳过，以及训练环境未安装matplotlib时的真实绘图测试；该绘图测试已在采集环境实际通过。见[最终统一回归日志](../.local/checks/guard_and_html_20261104/full_regressions.log)、[检查与源码冻结摘要](../.local/checks/guard_and_html_20261104/completion.json)及[采集环境绘图与统计回归](../.local/checks/policy_comparison_audit/report_edge_regressions_collection.log)。正式回归覆盖四组身份/共同Pack、初态、Wilson与配对统计、续训预算链、DP总loss、零查询失败和0/50绘图边界；同一个结果重复传四组以及明确标记的工程自检报告都会被拒绝。

本轮长训结束后补齐了ACT续训来源守卫：runtime、completion或checkpoint任一标记为smoke即拒绝将其延长为正式训练；原4项预期失败已转为正式通过测试。兼容性仅登记已审计的精确trainer源码指纹组合，保留历史run/checkpoint原签名，并实测正式baseline 40k与gesture 60k模型仍能只读加载。另将HTML中错误硬编码的“2厘米”改为实际settings或物理证据中的阈值，本轮为3厘米。两项修复不改变训练计算、物理成功判据或历史结果；见[真实模型读取核验](../.local/checks/guard_and_html_20261104/real_run_readability.json)、[修改范围审计](../.local/checks/guard_and_html_20261104/ast_scope_audit.json)与[精确源码指纹](../.local/checks/guard_and_html_20261104/frozen_source_hashes.json)。

旧pipeline工程Pack仍可训练和加载张量推理，但其旧场景绑定无法通过当前部署验证，因此没有伪造绑定或改写旧Pack。当前场景的短链路验收使用以下独立工程记录。

通过原 `tests.pipeline_smoke.capture` 生成当前 `tray_to_platform` 的两个约7秒无头scripted_test会话，seed731020/731021，均明确记录failure，不访问真人摄像头。使用正式DatasetWorkflow完成audit/curation、prepare联合与pack，全部产物在 `.local/checks`。新工程Pack为 `.local/checks/gesture_auxiliary_design/current_pack`：2条、254保留帧、126个train窗口/126个val窗口。用CPU、工程缩小网络/64×48图像完成DP gesture两步训练，真实读取其中脚本识别标签（无mock/注入/重写），分类头更新、250个val有效手实例，EMA严格加载及有限 `[1,30,14]` 输出通过。[CPU短训报告](../.local/checks/gesture_auxiliary_design/cpu2_current_pack/cpu_smoke_report.json)保留配置和指标，不能将工程尺寸/2步模型计入正式对照。

正式部署adapter的当前task binding、`diffusion_gesture`身份、EMA严格加载与 `human_inference_input=false` 已通过，见[加载核验](../.local/checks/gesture_auxiliary_design/current_adapter_check.json)。CPU两秒物理preview已完成：1个工程case、预期timeout、0项基础设施错误，未访问真人摄像头，见[短物理执行摘要](../.local/checks/gesture_auxiliary_design/cpu2_current_rollout/summary.json)。它证明两步工程模型可以完成训练→checkpoint→当前场景加载→短物理执行通路，不代表自主任务成功。

参考作者来源：

- [ACT项目与方法](https://tonyzhaozh.github.io/aloha/)：动作块、CVAE及推理z=0；[作者实现](https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py)。本工程独立observation encoder为本地实现，不能称逐层精确复现作者网络。
- [Diffusion Policy项目](https://diffusion-policy.cs.columbia.edu/)与[作者图像策略实现](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_image_policy.py)：视觉条件扩散与滚动动作预测。本轮gesture支路是本工程提出的辅助监督，不是原论文已有的结论。
