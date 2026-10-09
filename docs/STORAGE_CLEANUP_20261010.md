# 本地存储清理记录（2026-10-10）

D 盘空间在本次工作前约剩 2.97 GB。检查发现 `final_experiment/remote_archives/` 中保存了已经拒绝的旧远端实验完整压缩包：20 个 `.tar/.tar.zst/.zip` 文件共 25,617,789,352 字节，另有 8 个分片文件共 1,269,462,251 字节。

这些归档对应 Phase19–25 和 Phase41 的已结束实验，当前正式模型、数据、论文图、Phase53 结果以及各归档的 manifest、verified、reclaimed 和 README 均已保留。删除前将文件路径、大小和修改时间写入 `.codex-finalizer/storage_cleanup_20261010.json`；该文件未进入 Git，因为它包含本机路径。删除后 D 盘可用空间约 27.99 GB。

本次没有删除当前数据、Phase53 `final.pt`、论文图、Phase65 失败验收报告或 `third_party/voca_reference`。如果未来需要恢复旧实验，应先依据保留的 manifest 核对来源；当前论文和模型工作不依赖这些压缩包。
