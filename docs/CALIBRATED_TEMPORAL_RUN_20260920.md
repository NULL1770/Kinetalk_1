# 参考口型校准与时序眉眼实验，2026-09-20

## 完整终点核验（约17:19北京时间）

自动队列已完成，耗时2486.08秒（41.43分钟），smoke/protected/audio/static全部exit0。身份200参考对轮次、teacher12、audio12以及两臂dynamics各12均执行完成，后四类query训练每轮4098条。GPU现已空闲。全部446 validation×3种子与原门控执行完成，**mouth_preserved=true，quantitative_timing_passed=false**。未启动额外训练、未替换默认、未读取sealed test。

| 对比条件 | ARKit-MBE | ARKit-LBE | 绝对FDD | Upper9 centered ES | Variogram |
|---|---:|---:|---:|---:|---:|
| 本轮Stage4基座 | 0.761587 | 0.314745 | 0.123837 | 0.122573 | 0.093371 |
| 时序分支+真实音频 | 0.740825 | 0.314745 | 0.109981 | 0.115918 | 0.033334 |
| 独立训练静态条件 | 0.743561 | 0.314745 | 0.110379 | 0.116007 | 0.031794 |
| 同模型静态条件 | 0.741150 | 0.314745 | 0.110018 | 0.115945 | 0.032173 |
| 同模型反序条件 | 0.741601 | 0.314745 | 0.110865 | 0.123011 | 0.033588 |

这些是开发集系数/分布指标，不是最终论文测试集或SyncNet指标。相较先前full_v1，MBE 0.805965→0.740825、LBE 0.433960→0.314745；同时改变了B0恢复、校准、嘴部保护和时序结构，不能归因于某一个部件。

### 明确改善与仍失败的部分

- 口型保护30个seed/mode组合全部通过，seed42 mouth raw MSE0.01078375、correlation0.4453056，与校准值仅数值舍入差异。
- Upper9归一化邻帧速度RMS由本轮Stage4的0.322131降到0.081123，GT0.058044，约5.55倍GT→1.40倍GT。它支持相对基座的过快运动改善，尚无本轮视频视觉验收，不能直接叫自然动态成功。
- Seed42眉部/眼表情幅度比0.9216/1.0067，时序相关仅0.0571/0.0869。幅度接近GT仍不代表动作发生时刻对。
- 对独立static，centered ES差−0.00008849，85句簇95%CI[−0.00116379,+0.00084219]跨零；variogram差+0.00153980，CI[+0.00101607,+0.00199110]，真实音频反而较差。对同模型static也不通过。反序centered ES差−0.00709296、区间全负，但反序变差单独不足以证明有用的音频时序。
- Audio慢状态centered correlation：固定64训练query−0.28508、完整validation−0.02502；训练探针也没正确拟合时间变化。因此不能把失败仅归结于未见身份泛化。该诊断尚不能区分监督/表示/优化竞争的具体原因。
- 身份参考检索fit22/22、development3/3；只说明这项参考表征诊断，不能宣称完整身份视觉质量已解决。
- Audio情感分类96.41%，motion teacher对GT分类仅53.81%；teacher读取生成动作的seed42准确率由Stage4 65.25%降到时序分支50.67%。这是非独立诊断且teacher本身有限，但提示动态替换可能损伤情感表达，不能声称情感已保证。
- AV offset/confidence、FD/WInD、文献特征空间Multimodality仍pending，未虚构数值。

完整小型报告已下载到`artifacts/calibrated_temporal_20260920/run12_reports`，归档SHA256 `b8e65482e31a9031cdd5b446c6ae432f30384adec8a16bc25495674641cc7514`与远端一致；提取摘要`endpoint_summary.json`。三份完整native曲线（约50.6/170.8/170.8MB）已在远端持久盘重载逐值验证保存；本次未下载这些曲线，避免混淆本地备份范围。剩余磁盘约866MiB。

## 已验证的口型修复

此前4098条完整训练query、30轮B0恢复已改善口型时序，但neutral原始误差比旧B0高5.96%，自动流程因此停止。保存曲线的独立分解指出问题主要是静态均值偏差。

本轮保持恢复后的B0权重不变，使用TRAIN的715条neutral query残差均值与独立neutral enrollment残差均值，按身份等权拟合27通道的固定gain和bias。gain限制在[0,1]，固定ridge=1e-3、bias ridge=1e-4。部署只从独立参考动作及其B0输出计算一个固定口部偏移；不读取query目标、query均值或情感标签。参考clip及同身份句子均与query分离。

完整446条validation，未裁剪的口部数值：

| 指标 | 恢复B0 | 加参考校准 |
|---|---:|---:|
| Overall raw MSE | 0.01499158 | 0.01078375 |
| Neutral raw MSE | 0.01470305 | 0.00652539 |
| Nonneutral raw MSE | 0.01505104 | 0.01166126 |
| Centered correlation | 0.44530670 | 0.44530670 |
| Centered MSE | 0.00528701 | 0.00528701 |
| 邻帧位移MSE | 0.00104136 | 0.00104136 |

原有口型保护门槛、nonneutral额外保护和真实输入优于static/reverse的对照均通过。校准运行38.41秒。固定偏移保留时序轨迹，不能解释为新增音频时间信息。

限制：未裁剪越界比例从19.96%升到27.14%（旧B0为33.50%），仍需视觉和独立AV检查。不能仅凭MSE宣称口型、身份、情感全部达标。当前训练划分是MEAD四类4098条query/454566有效帧，不是全部原始MEAD或八类数据。

## 眉眼结构与检验

当前raw upper9 DiT在静态条件下缺少显式帧位置和相邻状态处理。新实验引入固定native-frame位置编码、零初始化的有界rank4相邻状态卷积、rho=0.9的相关初始噪声；缺帧处重置，不跨缺帧卷积。噪声变换同时用于flow训练与推理，没有输出后处理或指标裁剪。这是待验证的时序先验机制，不能先验保证音频可预测性。

保留motion teacher→audio global蒸馏和独立参考身份。Stage5只替换upper9，其他43通道逐值保留Stage4；所有随机残差均排除27个口部通道。真实音频与独立静态臂共享Stage4、初始化、数据和随机预算，各12轮覆盖全部4098条训练query。

静态干预将编码后的local/state/h0一起时间池化；反序干预将它们一起在有效native位置反序，global、identity、noise保持一致。静态编码器仍见过整段有序音频，因此准确解释是“没有随帧变化的条件”；反序是“编码条件时间线反序”，不是原始波形反放。

固定终点评估完整446 validation×3噪声种子，并保留固定64训练query诊断用于区分欠拟合与泛化失败。对独立static、同模型static、反序条件分别检验centered fair ES和variogram，句簇bootstrap上界均需<0，同时必须通过口型保护。缺样本、缺种子、NaN或空门控结果会报错。通过只是量化时序证据，视觉、独立情感与AV验收仍另算；sealed test不读取，不自动替换默认模型。

## 自动流程与持久化

远端根目录：`/root/autodl-tmp/kinetalk_calibrated_20260920`。

1. 校准门控通过后，真实CUDA smoke依次走identity/teacher/audio/temporal dynamics。
2. 正式identity200轮（22组独立参考对，不是200轮query）→teacher12→audio12。
3. Audio temporal dynamics12→独立static temporal dynamics12→完整配对验收。

2026-09-20 16:35北京时间启动后台队列PID3559。真实CUDA smoke四阶段已完成，8步训练、18.68秒、exit0；16:36自动启动正式protected子进程3784。身份200参考对轮次已完成并通过口型保护，已核验teacher2/12，每轮4098条、257批，最近26.93秒/轮。包含后续两臂动态和完整验证估计还需40–70分钟，属于运行估计而非完成保证。最终实际运行状态见`run12/queue_status.json`。任何训练失败或口型保护失败会停止下游。SSH断开不影响后台进程；实例关机/重启会中断，自动队列暂不支持跨重启续跑，需要按保存的last.pt恢复。

原始padding曲线及smoke放在`/dev/shm/kinetalk_calibrated_temporal_run12`。每个正式臂完成后，把全部native帧（包括native无效边缘）原dtype无损打包至持久盘`run12/native_artifacts`，重载逐值验证并记录独立hash；只省去超出native长度的batch padding，不量化。持久checkpoint、日志和小型报告保留在run12。compact不是原始曲线文件，二者hash分别记录。

已核对历史neutral/style两对12000/final权重的所有ZIP成员，内容完全一致后把两个12000重复文件改为指向final的链接，保留旧路径，回收887054032字节，剩余约1.9GiB。数据集和实际final权重保留。逐项证据：`artifacts/calibrated_temporal_20260920/storage_duplicate_audit.json`。

本地和远端各76项相关测试通过。校准JSON与final.pt已下载，checkpoint SHA256本地/远端一致：`8415b6e692479cf6f6723e649fe7f867a42fc11896b9631e8940ebdc4e364dfe`。

数据manifest SHA256：`6bdc33c3ee832e968b421781267dc70654e416b8e76395f2c6ce83c859e8603e`；prepared index SHA256：`27d6a4dce0d76020a48d3f8445083bc1178f760f68ed071f61f25c4aba39d674`。
