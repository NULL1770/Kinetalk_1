# 代码整理完成记录（2026-09-21）

本轮整理代码入口和历史实验文件，不改变模型计算逻辑，不启动新训练，也不将多个实验合并为一个已验收系统。架构边界见[当前系统说明](CURRENT_SYSTEM_20260921.md)。

## 本地

- 工作区：`D:\实验室项目\新实验\kinetalk_b0_residual_train`。
- `scripts/*.py`：326 → 74；`tests/test_*.py`：228 → 84。
- 活动路径移出 1571 个文件：252 个 scripts、144 个 tests、1127 个 artifacts/tmp 历史 Python 文件、7 个根目录旧入口、41 个一次性 helper/下载页。
- 保留训练/评估/渲染入口的依赖闭包；核心模型源码暂不裁剪，部分旧文件承载兼容和公共导入。
- 数据、checkpoint、曲线、视频及实验文档未被移走；第三方依赖保留。
- 私有归档：`archive/cleanup_20260921/`。清理前快照、SHA256、原路径和计划可用于恢复，已被 `.gitignore` 排除；pytest 排除 archive。

本地批量永久删除被工具自动安全审查拒绝，返回原因只有 `blocked by policy`。实际采用可恢复迁移，不将这次整理描述为永久删除或显著释放磁盘空间。

## SSH

当前活动源码目录：`/root/autodl-tmp/kinetalk_active_20260921/code`。上传本地清理后源码、配置、测试、文档和第三方依赖；逐文件核对哈希。

以下四份旧 `code` 已移到 `/root/autodl-tmp/cleanup_20260921_remote/archive/relocated/`，同时保留完整压缩快照；共 3034 个文件逐一核对了迁移副本及压缩包内容：

| 原路径 | 已验证文件数 |
| --- | ---: |
| `/root/autodl-tmp/kinetalk_relative_20260921/code` | 808 |
| `/root/autodl-tmp/kinetalk_repair_20260920/code` | 740 |
| `/root/autodl-tmp/kinetalk_calibrated_20260920/code` | 763 |
| `/root/kinetalk_paper_20260920/code` | 723 |

另外归档了这些实验根目录及两处早期实验目录的 18 个一次性 Python/shell helper 和 3 个旧 `code.tar`。位于结果内部的源码快照、数据准备工具、模型缓存及数据环境保留；它们可能是复现依据，不按文件扩展名批量删除。

本轮检查时无训练任务运行。根盘可用约 26 GB，数据盘约 15 GB；上述归档保留了可恢复副本，不作为腾出大空间的操作。

## 验证

- 本地：`python -m pytest tests -q`，**764 passed**，28.76 s。
- SSH：在活动代码目录执行 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/miniconda3/bin/python -m pytest tests -q`，**764 passed**，19.64 s。
- 远端核心模型导入成功；归档、迁移和活动副本有独立 SHA256 记录。
- 测试证明整理后所保留代码的现有检查通过，不证明科研动态指标已成功。

## 历史恢复与后续运行

旧协议的 `source_inventory` 绑定当时的 scripts 和模型源码。恢复或 `--resume` 必须使用对应的完整源码快照和原协议；不能拿清理后代码冒充旧源码，也不能改旧哈希绕过验证。原 code 目录现已迁出，不应继续使用历史启动器中的旧路径。

新实验从上述活动目录创建独立运行及新的源码清单。没有自动提交或推送 Git；私有归档不得加入公开提交。
