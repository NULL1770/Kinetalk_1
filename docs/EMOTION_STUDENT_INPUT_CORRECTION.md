# 情感student输入纠偏（2026-10-06）

用户明确纠正：student是情感通路，应仅输入情感特征和韵律，学习情感、强度和表达动态；音素内容由HuBERT→B0/h0提供。不能把u_a解释成任意音频时序上下文，更不能用口型响应当作纯情感动态的证据。情感/风格仍能改变嘴部幅度，不恢复硬嘴mask。

## 已确认的旧实现问题

实际audio_features1540拼HuBERT768、emotion2vec768、prosody4；global和u_a共享TCN，所以二者都直接可读取内容。teacher阶段motion teacher输出global/intensity，音频学生提供u_a，三者和renderer联合训练；audio阶段冻结teacher，仅蒸馏global。旧motion controls没有作为当前u_a的逐帧教师目标。u_a通过全残差flow和生成语义项间接学习，因此还可能拟合B0发音误差；不能称已经解耦或已保证动态情感正确。

## 已完成的输入修复

- 后续train_full_staged默认`--audio-feature-layout affect-prosody`：从已拟合TRAIN mean/std选出最后772维，不重新拟合，不动packed缓存或B0内容输入。
- 772D SlowStateAffect从完整1540D缓存读取时，在归一化/投影之前仅选择emotion2vec+prosody；global与u_a均不能访问前768维HuBERT。
- 保存772维mean/std及recipe里的输入layout，旧推理构造方法可据统计宽度识别正确路由。没有增加loss/层/参数分支，也没有关闭嘴部通道。
- 历史1540D checkpoint仍严格兼容；仅为历史重放保留显式legacy layout。旧warm迁移到772必须显式`--allow-audio-input-migration`，核对相同TRAIN标准化，精确删除input.weight的HuBERT列，其他已有权重不动，不能静默加载/继续旧结果。
- 原训练source保存在.codex-finalizer/*_before_affect_only_input.py，远端既有冻结source/权重均未修改。

## 真实验证

24项本地检查通过：新增输入独立性、零HuBERT梯度、正确声学梯度、TRAIN统计选择、旧state兼容、迁移边界及默认CLI；已有global dropout、staged runner、frozen renderer检查通过。首次新fixture漏`--output`导致parser报错，模型23项通过；补齐fixture后24全部通过，没有改变实现规避错误。

远端只读验证使用真实Phase26 seed47 checkpoint和16段validation音频，CPU/GPU均通过：旧模型正常输出逐位不变；772模型完整缓存输入与显式772输入逐位相同；把HuBERT列改为NaN仍所有global/u_a输出逐位相同；HuBERT梯度max=0，剩余输入梯度非零且finite；所有模型state不变，无optimizer/训练/sealed数据。report SHA77ceca4820fd75d7edaa676312863d2b32551f171a75d4f93c455150ff59f1a5。完整验证代码/report/state已下载。

## 尚未完成，不能混用成绩

这只是去掉显式内容入口。emotion2vec本身仍可能含音素信息；teacher残差代码也可能带B0误差。无逐帧情感标签/teacher控制蒸馏，输入修复不等于u_a监督已改成纯情感动态，也不代表指标已经提升。旧Phase26/29成绩仍来自1540混合输入，不能写成772成果。

Phase30旧模型u_a六条件审计在用户纠偏前仅准备本地helper，未创建远端launch、未启动GPU诊断或训练；现不按其混合输入假设继续。后续先核对情感动态监督/teacher与renderer的职责，再真实teacher/audio迁移smoke和固定预算比较；不可拿截列后的旧权重直接宣称已重训或SOTA。用户目标仍是正确内容/表情/身份和全部指标提升。
