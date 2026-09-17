# Evaluation

`validate_multicam_dataset.py` 用于检查：

- episode 数与帧数；
- frame index 和 30 FPS 时间戳；
- action/state 维度和有限值；
- 三路视频分辨率和帧数同步；
- 夹爪开合是否完整；
- action 相邻帧跳变；
- 每条 episode 的最终画面。

运行：

```bash
python evaluation/validate_multicam_dataset.py datasets/act_50eps
```
