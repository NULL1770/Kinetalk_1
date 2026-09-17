# 完整五阶段续训：有符号慢状态与投影随机残差

2026-09-17，正式启动前锁定。用户要求每阶段10–15 epoch、完整训练身份/情感/口型/动态，启动后交付估时，用户稍后返回查看。固定12 epoch/阶段，不根据开发集选最佳轮次，不替换默认权重。

这是已有预训练基座的完整分阶段续训；声学/内容特征提取器保持冻结。它不是从随机权重重训全部网络，也不承诺效果或论文新颖性。当前只有4种情感实际数据（neutral/angry/happy/sad），即使分类头有8个输出也不能声称八类训练。

## 数据与初始化

沿用当前2315 fit / 405 internal development、19/3身份、96帧原生25Hz、已有独立中性参考。内部dev与fit共享句子，历史继承基座可能见过dev身份；只支持这一内部验证协议。封存test不读取，身份dev参考只用于推理、不反传。

从 run09 的原始完整 NeutralAffectSystem checkpoint 恢复，包含其匹配的 learned-control projection；不载入后续RRR投影。数据谱系由 uniform experiment 的输入hash与锁验证。旧cache仅提供query目标与metadata；旧B0/h0/identity/affect均不作为本轮动态更新后的条件。原生FP32 content来自已验证1540维缓存前768维，音频其余为中间层emotion2vec+韵律。

新音频encoder从1540维输入重新学习global，接受更新后的motion-global教师监督；不把中间层768维冒充旧final-layer global输入，不使用旧全局归一化。

## 五阶段

| 阶段 | 可训练部分 | 一个epoch | 预算 |
|---|---|---|---|
| articulation | 原B0网络 | 422条neutral fit query遍历 | 12 epoch |
| identity | neutral identity encoder＋bias | fit身份独立互补参考对遍历（2–4参考/人） | 12 epoch |
| teacher | motion teacher＋原local projection＋全脸renderer | 全2315 fit遍历 | 12 epoch |
| audio | 新1540音频encoder＋全脸renderer；motion teacher冻结 | 全2315 fit遍历 | 12 epoch |
| dynamics | 新上脸9维flow＋可训练local audio副本；主系统和audio-global冻结 | 全2315 fit遍历 | 12 epoch |

batch16。B0 lr1e−5；identity1e−4；teacher/local1e−4、renderer3e−5；audio1e−4、renderer1e−5；dynamic local与upper flow各1e−4。AdamW、weight decay1e−5、gradient clip1。12步Euler，固定seed47。真实epoch无放回遍历，身份epoch与query epoch规模不同，不能相加当成相同训练量。

B0只用neutral唇/颌14–40观测重建＋位移约束；tongueOut51在数据中不可观测，不将零占位当GT。B0训练显式启用梯度，绕过旧system.base的no_grad；完成后重新计算所有query/参考B0/h0。

身份使用两组不相交neutral参考，公共观测通道baseline重建与身份contrast；仅fit身份参与训练。身份含义为动作系数偏置与风格，不等价于mesh人脸几何身份。

teacher使用全脸flow＋情感/等级语义；audio同时使用motion→audio global MSE蒸馏、情感/等级监督、4维慢状态监督与真实音频条件flow。teacher/audio每4个batch进行一次真实noise rollout的区域raw重建及域保护，不将插值一步估计伪装成完整生成。该保护不保证第四阶段嘴部优于历史模型，必须实测。

## 与前版及SubtleTalk的实际区别

前版是逐帧无符号标量强度调制global，再与声学hidden融合。新版本不使用文本、VA、窗口std输入或前序motion history。定义相对独立neutral anchor和训练集通道尺度的4个有符号运动代理：raise(43/44/45)、down(41/42)、squint(5/12)、wide(6/13)。各组保留符号而不是上抬减下压；不是心理情绪VA。

固定时间结点0/16/32/...帧，线性hat basis按有效观测做最小二乘投影P，得到连续的粗尺度样条状态。正式推理状态始终由音频预测，GT只作为训练监督与明确标识的oracle评估。不是把逐帧9维目标直接提供给生成器。

对规范化上脸9维，P还要求同组通道共享慢状态，Q=I−P。flow目标、初始噪声、velocity、ODE状态均在Q空间。最终上脸为 independent neutral anchor＋lift(audio state)＋scale×Q residual。这样随机残差不能抵消四组共享的样条状态，同时允许左右差异及其他时间细节。低频状态预测错误无法由Q修正，这是明确的模型限制，需用oracle-state对照衡量。

最终其余43通道逐位复制冻结第四阶段输出，保证第五阶段不修改嘴/颌；这不表示本轮前四阶段与历史模型完全相同。上脸中性基线来自独立neutral anchor，identity code继续条件化创新分支；身份检索分数不是最终生成身份感知认证。

“有符号个体相对慢状态＋不能抵消其状态的随机残差”是本轮可检验的设计区别。它不是通过从9维降到4维就形成创新，也不是已经证明音频可预测与不可预测因素被完全分离。若效果有益，再进行同预算direct audio上脸flow、取消Q等消融；本次单臂完整训练只做整体可行性评估。

## 保存与评价

每epoch原子保存last.pt，包含全部模型、当前optimizer、stage/epoch/step与训练RNG；可在完成epoch边界恢复。每stage保存固定final.pt、evaluation.json、3seed全405曲线（42/123/2026），不保留每epoch大型权重以控制空间。

每4epoch固定dev64诊断。最终dynamic seed42另存base/static_state/oracle_state/reverse_audio。报告身份参考检索、global情感分类、非独立teacher生成情感读出、眉/眼/唇颌raw与去均值动态MSE/相关/R²/幅度、域越界、慢状态raw及centered。teacher/identity使用target-motion oracle，单独标识；audio/dynamics才是audio-only结果。

当前已实现的自动评估以系数和固定多seed曲线为主；尚不等于独立唇音同步、几何身份或感知情感认证。训练后可从保存曲线渲染连续rig视频，后续用户回来时进行实际视觉复核。

远程隔离目录 `/root/kinetalk_full_staged_20260917/`。不删除或覆盖历史数据。数据盘余量小，新输出位于系统盘。正式启动前完成5阶段真实smoke、梯度与冻结边界测试、恢复测试，再根据实测速度提供估时。
