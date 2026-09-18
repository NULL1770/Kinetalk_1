# 联合运动先验迁移记录（2026-09-18）

用户授权将当前实验迁至端口 11473 的实例，并清理目标实例中的无用旧副本。旧端口 24539 仅用于读取需要迁移的源文件，不启动或停止其任务。

目标目录为 `/root/kinetalk_joint_20260918`：

- `code/`：本地提交 `7ae25a5` 的完整源码，另加后台启动器。
- `run12/audio/final.pt`：来自本地已核验备份，SHA256 `73a8f17137772f882bf6d8c65dbf2f6c36b6c42f5724091f7a2e7a42870df42c`，同时保留原 `complete.json` 和 `run12/provenance.json`。
- `run12/dynamics/final.pt`：完整系统及原动态阶段终点，用于后续全脸对照；SHA256 `9f7c40d201ef782dce46ed337e32eda9f819cd27686d7de3de058b38db152501`。
- `full_native_audio_delta/`：从旧实例复制原始文件，逐文件核对 SHA256，保留原 manifest/complete 字节，不重新提取。
- `joint_prior_formal30/`、同级 `.run.log`、`launches/`：新实验及脱离 SSH 的后台进程记录。

直接复用目标实例已有数据，未重复上传：

| 输入 | 目标路径 | SHA256 |
|---|---|---|
| Audio cache | `/root/kinetalk_runs/audio_text_v1/data/audio.pt` | `f1f03c58e3a2175a72ade9d32bb97831d534919755db5c7d17e6a1a3d49ac7ff` |
| Targets | `/root/kinetalk_runs/audio_text_v1/data/intensity_targets/targets.pt` | `a68cb18954f8705d1c2ca124df506c1eb4f9c2e36dea6bbe50bcd3452e5199e0` |
| Renderer cache | `/root/kinetalk_runs/teacher_schedule_v1/data_locked/renderer_cache.pt` | `89537672239b805876ed15ab24aa1e5abd0fe231cb75cfa7c1e07bf7548efcae` |
| Native manifest | `/root/autodl-tmp/kinetalk_data/processed/native_affect_style_v4_refmask/train.jsonl` | `f88754ad012a6c42382d1e46a69afd976e861ae53aa2ae5331f82d86f71dbc18` |

清理目标实例以下旧目录，数据盘可用空间从约 407 MiB 增至约 4.2 GiB：

- `/root/autodl-tmp/kinetalk_b0_residual_train_backup_v1_20260913_215357`
- `/root/autodl-tmp/kinetalk_neutral_affect_pilot_20260916`
- `/root/autodl-tmp/kinetalk_v4_staging`
- `/root/autodl-tmp/kinetalk_semantic_v9_20260915`

清理后发现 pilot 目录中的 `run09_emotion2vec_probe/audio.pt` 与 `effective_config.yaml` 仍是后续全脸加载依赖。这两份已从旧实例恢复到原路径，并核对 SHA256 分别为 `6e475ad84460a32e0810fcc6455526a2913f046d4d4494e3c6fb8a78a251c6cb` 与 `a9f78b954d587f60797ac3341c68f1a50597e6bb1c317d6efd93d091aebc4ef5`；本地另存副本。没有删除 native 数据、waveform、`teacher_schedule_v1/data_locked`、提取模型或当前 b0 代码。

尽管用户说明以无卡方式开机，实际预检检测到 RTX 4090，`torch.cuda.is_available()` 为真且 CUDA tensor 分配成功。正式启动仍要求当时 CUDA 预检通过；不在 GPU 不可用时伪报训练开始。

新机已运行联合模块 95 项测试，全部通过。run12 加载、冻结来源绑定、现有输入哈希核验均通过。迁移、试跑和正式运行状态以目标实例 `migration_*.json`、`launches/*/launch.json`、实验 `status.json` 为准；协议成功与进程完成分别报告。
