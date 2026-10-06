# 当前final12连续情感条件的泛化核查

Phase28 v2已完成三个固定standardized12模型的全部12536 TRAIN段/1372104帧、22位说话人核查。仅前向读取，使用独立TRAIN中性参考；所有模型state前后逐位不变，sample/GT/mask流三模型一致。验证集仅使用Phase27已有并绑定SHA的codes做描述性对比，没有拟合验证集偏移或系数。20个远端文件及三个codes已下载核SHA。

| 连续global指标 | seed47 | seed48 | seed49 |
|---|---:|---:|---:|
| TRAIN MSE | .148714 | .149859 | .149300 |
| validation MSE | 2.304595 | 2.323142 | 2.354100 |
| validation/TRAIN | 15.50 | 15.50 | 15.77 |
| TRAIN cosine | .950 | .948 | .949 |
| validation cosine | .645 | .626 | .628 |
| TRAIN平均偏差MSE | .002134 | .001819 | .001822 |
| validation平均偏差MSE | .666969 | .735364 | .765154 |

TRAIN内留一说话人估计常量offset仅把MSE降到.146909/.148400/.147807，约1%改善，不足以支撑用固定整体偏移解决验证问题。原编码器看过全部TRAIN；这里的LOSO只检查offset估计，不是整模型留一说话人训练。

audio自己的分类头TRAIN F1=.9993/.9992/.9983，validation=.8444/.8632/.8408；这些不是生成动作F1。teacher自己的分类头TRAIN约.76/.77，validation约.60/.62，亦不能混用。总体类别信息能泛化，但连续姿态条件存在明显TRAIN/validation差距。

以每个split的情感×强度均值去掉组间差异后，audio/teacher global的pooled相关性：TRAIN=.931/.925/.928，validation=.204/.192/.221。它是描述性相关，不是因果身份归因，也不是新的情感指标。表示同一情感和强度内部的细节在TRAIN对齐，在新说话人上对齐明显不足。

仅用TRAIN计算的情感×强度centroid描述也表明问题不能完全简化为音频类中心漂移：teacher TRAIN组内MSE≈.93–1.01、validation到同TRAIN中心≈1.39–1.42；audio TRAIN组内≈.74–.81、validation到同TRAIN中心≈.94–1.01。连续目标还承载了样本表达差异和说话人风格。不能据此断言全部是身份泄漏、不能直接用class prototype替代真实幅度；旧prototype/线性/neutral-reference映射失败证据仍需保留。

## 后续单变量方向

当前瓶颈从“生成器学不出情感”转为“音频连续情感条件对未见说话人的泛化不足”。仅延长预算、增大global MSE权重或统一嘴gain缺少依据。下一候选应优先限定在audio-global分支的泛化/正则化，保留既有u_a内容时序、身份输入、全嘴残差和全部几何门槛。先核对旧失败方案，执行默认不变、独立随机流和冻结/输入smoke，之后才登记固定三seed matched训练；未证明有效前不能推广。

中性jaw时序是另一个未解决项，不能假定global改进自动修复。Phase27替换global后jaw相关性仍未改善。应单独定位B0与残差的时序贡献，再决定修改；不并行改变两个变量或叠加loss。

旧Phase28第一次在提取结束、生成报告时因`sha(__file__)`误把字符串当Path失败，原日志保留；修复Path/str兼容，并让codes/frozen/stream先持久保存，再生成报告。没有重复模型训练。v2源代码、完整性结果和记录绑定在证据目录。

证据：`final_experiment/evaluation/diagnostics/phase28_train_global_v2_20261006/`。总目标尚未完成；F1的clip协议目标已在三seed实现，MBE约.7、联合口型/表情/身份质量和论文公平对照仍未达成。
