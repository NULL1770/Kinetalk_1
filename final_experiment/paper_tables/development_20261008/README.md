# 开发集实验表：同协议基线与训练消融

Material Passport: MODE=validate; STATUS=ANALYZED; sources为已完成、SHA核验的原报告；本次汇总未重新训练/重新推理。
全部1,367条开发集、相同原生有效帧、51观察通道、固定rig和四冻结probe。主表统一clip[0,1]；raw完整表同时导出。未读取sealed。
基线是ARKit共享音频改编，非原论文官方分数。训练数据/预算/seed不同，见table8；不是严格等预算SOTA结论。FaceDiffuser原预测无checkpoint哈希绑定，原生时钟/GT一致，历史报告有此限制；其余三基线原首batch权重重放通过。本次远端checkpoint/预测SHA重新核验。
Lip/Expression mean为顶点欧氏距离平均(mm)；LVE-max/EVE-max另列，不能混用。这里mesh FDD为上区平方位移能量时间std差的绝对值(mm²)，是沿用实现，不能将所有论文同名FDD当同公式。MBE/LBE为现有系数区域L2误差定义，不能拿其他rig的裸数值比较。
四F1依次为原128/原64/辅助128/辅助64整段统计probe；不是逐帧情感GT。jaw范围GT平均0.175279，范围越大不必越好；相关与位移误差联合判断，不能以平滑低误差替代正确动态。
table4只列实际重训练消融；table5是冻结推理干预，不混为训练模块消融。oracle不进入方法排名。仅单seed结果，不报伪造多seed误差条。
论文实践：EmoTalk Table5分别检查emotion disentangling encoder、emotion-guided attention、Lvel/Lcls、HDTF数据及encoder替换；MEDTalk §4.5/Table3检查overlap exchange、cycle exchange、disentangle、intensity和text。本项目应围绕自己的g/u职责、style/reference与teacher/student提出并重训练消融；不能直接照搬其模块名。
统计核验11/11已检查：分组结果另列以检查聚合反转；不由3人推断总体个体；MEAD演员/伪GT选择偏差保留；无协变量调整/collider推断；报告8类而非只happy；不选极端片段宣称回归改善；所有1367无删坏样本；所有probe/raw保留；明确多轮开发探索；冻结交换非因果独立证明；教师看到GT非部署预测/因果反向结论。没有开展显著性检验，三开发身份及单训练seed不足以证明统计稳定。

## table1_geometry

| method | lve_mean_mm_mean | eve_mean_mm_mean | vertex_lve_sqrt_mean | eye_forehead_eve_sqrt_mean | vertex_fdd_absolute_mm2_mean |
|---|---|---|---|---|---|
| VOCA-core (shared audio/ARKit) | 3.01442 | 0.703497 | 5.61668 | 2.41175 | 144.238 |
| FaceFormer (shared audio/ARKit) | 4.53784 | 1.1383 | 8.28442 | 3.78867 | 415.804 |
| EmoTalk-core (shared audio/ARKit) | 2.97323 | 0.691868 | 5.56334 | 2.41675 | 152.495 |
| FaceDiffuser (frozen KineTalk conditions/ARKit) | 3.07168 | 0.754096 | 5.86398 | 2.58438 | 137.962 |
| Phase41 | 3.19337 | 0.732675 | 6.12421 | 2.57124 | 149.172 |
| Phase42 | 3.11847 | 0.708939 | 6.00572 | 2.48554 | 147.264 |

## table2_coefficients

| method | arkit_mbe | arkit_lbe | arkit_fdd_absolute | arkit_fdd_signed |
|---|---|---|---|---|
| VOCA-core (shared audio/ARKit) | 0.746703 | 0.332384 | 0.102816 | 0.0951416 |
| FaceFormer (shared audio/ARKit) | 1.2968 | 0.544334 | 0.350484 | -0.331659 |
| EmoTalk-core (shared audio/ARKit) | 0.745003 | 0.331531 | 0.0945103 | 0.0652765 |
| FaceDiffuser (frozen KineTalk conditions/ARKit) | 0.792727 | 0.32311 | 0.0997374 | 0.0751161 |
| Phase41 | 0.810014 | 0.372413 | 0.112121 | 0.110557 |
| Phase42 | 0.784298 | 0.352371 | 0.112287 | 0.110094 |

## table3_dynamics_semantics

| method | mouth_displacement_mse | jaw_centered_correlation | jaw_q90_q10 | F1_1 | F1_2 | F1_3 | F1_4 |
|---|---|---|---|---|---|---|---|
| VOCA-core (shared audio/ARKit) | 0.000795734 | 0.554159 | 0.136455 | 0.497716 | 0.484458 | 0.469954 | 0.475981 |
| FaceFormer (shared audio/ARKit) | 0.00085878 | 0.433926 | 0.153497 | 0.123846 | 0.17603 | 0.188483 | 0.188549 |
| EmoTalk-core (shared audio/ARKit) | 0.00142921 | 0.565796 | 0.154294 | 0.591884 | 0.686229 | 0.739398 | 0.730805 |
| FaceDiffuser (frozen KineTalk conditions/ARKit) | 0.000935414 | 0.452377 | 0.135109 | 0.154482 | 0.270141 | 0.600528 | 0.568948 |
| Phase41 | 0.000815956 | 0.468447 | 0.125103 | 0.636091 | 0.584336 | 0.673703 | 0.591357 |
| Phase42 | 0.000794127 | 0.464103 | 0.109285 | 0.60631 | 0.559938 | 0.637381 | 0.566032 |

## table4_trained_ablation

| method | prior_variance | center_local | reference_training | fit_clips | updates | arkit_mbe | arkit_lbe | lve_mean_mm_mean | jaw_centered_correlation | jaw_q90_q10 | F1_1 | F1_2 | F1_3 | F1_4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Phase41 | learned | False | single | 10903 | 16368 | 0.810014 | 0.372413 | 3.19337 | 0.468447 | 0.125103 | 0.636091 | 0.584336 | 0.673703 | 0.591357 |
| Phase42 | unit | False | single | 10903 | 16368 | 0.784298 | 0.352371 | 3.11847 | 0.464103 | 0.109285 | 0.60631 | 0.559938 | 0.637381 | 0.566032 |

## table5_inference_interventions

| method | intervention | n | arkit_mbe | arkit_lbe | jaw_centered_correlation | jaw_q90_q10 | brows/centered_correlation |
|---|---|---|---|---|---|---|---|
| Phase41 | normal | 96 | 0.83247 | 0.386539 | 0.441469 | 0.124997 | 0.0262048 |
| Phase41 | static_u | 96 | 0.832998 | 0.386835 | 0.423042 | 0.109432 | 0.0249089 |
| Phase41 | reverse_u | 96 | 0.837718 | 0.389266 | 0.396151 | 0.116135 | 0.000777258 |
| Phase41 | shuffle_u | 96 | 0.83453 | 0.387867 | 0.40831 | 0.113559 | 0.0214018 |
| Phase41 | wrong_reference | 96 | 0.908512 | 0.424659 | 0.414213 | 0.117229 | 0.0178094 |
| Phase41 | reference_A | 96 | 0.823572 | 0.382707 | 0.423544 | 0.108758 | 0.0364589 |
| Phase41 | reference_B | 96 | 0.856854 | 0.393018 | 0.442175 | 0.129523 | 0.0422059 |
| Phase41 | wrong_matched_audio | 96 | 0.860952 | 0.385498 | 0.396471 | 0.114433 | 0.023186 |
| Phase41 | gt | 96 | 0 | 0 | 1 | 0.199388 | 1 |
| Phase41 | gt_static | 96 | 0.395301 | 0.250083 | 5.19735e-09 | 0 | -1.78116e-07 |
| Phase41 | gt_reverse | 96 | 0.557077 | 0.335443 | -8.869e-06 | 0.199388 | -0.135361 |
| Phase41 | gt_shift | 96 | 0.427103 | 0.265775 | 0.0838484 | 0.199388 | 0.409335 |
| Phase41 | gt_gain | 96 | 0.352843 | 0.144018 | 1 | 0.249235 | 1 |
| Phase42 | normal | 96 | 0.812196 | 0.367404 | 0.450464 | 0.11207 | 0.0308557 |
| Phase42 | static_u | 96 | 0.812736 | 0.370411 | 0.411601 | 0.0924438 | 0.0349191 |
| Phase42 | reverse_u | 96 | 0.821691 | 0.376252 | 0.374107 | 0.100445 | 0.0354098 |
| Phase42 | shuffle_u | 96 | 0.81668 | 0.374071 | 0.394256 | 0.0982357 | 0.027988 |
| Phase42 | wrong_reference | 96 | 0.922427 | 0.43227 | 0.424524 | 0.107928 | 0.0342763 |
| Phase42 | reference_A | 96 | 0.809335 | 0.368828 | 0.421311 | 0.100386 | 0.0240719 |
| Phase42 | reference_B | 96 | 0.827492 | 0.372145 | 0.463563 | 0.119775 | 0.0348043 |
| Phase42 | wrong_matched_audio | 96 | 0.836678 | 0.362429 | 0.364246 | 0.0928888 | 0.00611882 |
| Phase42 | gt | 96 | 0 | 0 | 1 | 0.199388 | 1 |
| Phase42 | gt_static | 96 | 0.395301 | 0.250083 | 5.19735e-09 | 0 | -1.78116e-07 |
| Phase42 | gt_reverse | 96 | 0.557077 | 0.335443 | -8.869e-06 | 0.199388 | -0.135361 |
| Phase42 | gt_shift | 96 | 0.427103 | 0.265775 | 0.0838484 | 0.199388 | 0.409335 |
| Phase42 | gt_gain | 96 | 0.352843 | 0.144018 | 1 | 0.249235 | 1 |

## table6_emotion_breakdown

| method | emotion | F1_1 | F1_2 | F1_3 | F1_4 |
|---|---|---|---|---|---|
| VOCA-core (shared audio/ARKit) | neutral | 0 | 0.116129 | 0 | 0.037037 |
| VOCA-core (shared audio/ARKit) | angry | 0.333333 | 0.238095 | 0.27907 | 0.203883 |
| VOCA-core (shared audio/ARKit) | contempt | 0.569052 | 0.674374 | 0.496392 | 0.637523 |
| VOCA-core (shared audio/ARKit) | disgust | 0.722611 | 0.577273 | 0.652452 | 0.65721 |
| VOCA-core (shared audio/ARKit) | fear | 0.191388 | 0.222222 | 0.345865 | 0.118812 |
| VOCA-core (shared audio/ARKit) | happy | 0.78187 | 0.785714 | 0.788571 | 0.765363 |
| VOCA-core (shared audio/ARKit) | sad | 0.731302 | 0.630769 | 0.617647 | 0.751843 |
| VOCA-core (shared audio/ARKit) | surprise | 0.652174 | 0.63109 | 0.579634 | 0.636175 |
| FaceFormer (shared audio/ARKit) | neutral | 0 | 0 | 0 | 0 |
| FaceFormer (shared audio/ARKit) | angry | 0 | 0 | 0 | 0 |
| FaceFormer (shared audio/ARKit) | contempt | 0.289679 | 0.334947 | 0.345528 | 0.36182 |
| FaceFormer (shared audio/ARKit) | disgust | 0.0379147 | 0.315152 | 0.320802 | 0.268293 |
| FaceFormer (shared audio/ARKit) | fear | 0 | 0 | 0 | 0.0106383 |
| FaceFormer (shared audio/ARKit) | happy | 0.5311 | 0.55814 | 0.545 | 0.52968 |
| FaceFormer (shared audio/ARKit) | sad | 0.132075 | 0.2 | 0.29653 | 0.337963 |
| FaceFormer (shared audio/ARKit) | surprise | 0 | 0 | 0 | 0 |
| EmoTalk-core (shared audio/ARKit) | neutral | 0.0930233 | 0.294737 | 0.495575 | 0.460177 |
| EmoTalk-core (shared audio/ARKit) | angry | 0.609023 | 0.757282 | 0.863636 | 0.859599 |
| EmoTalk-core (shared audio/ARKit) | contempt | 0.665392 | 0.82963 | 0.847545 | 0.845745 |
| EmoTalk-core (shared audio/ARKit) | disgust | 0.725212 | 0.730853 | 0.77619 | 0.796069 |
| EmoTalk-core (shared audio/ARKit) | fear | 0.653153 | 0.649165 | 0.591195 | 0.5 |
| EmoTalk-core (shared audio/ARKit) | happy | 0.808743 | 0.883598 | 0.885496 | 0.873096 |
| EmoTalk-core (shared audio/ARKit) | sad | 0.725624 | 0.794737 | 0.778378 | 0.823232 |
| EmoTalk-core (shared audio/ARKit) | surprise | 0.454902 | 0.549828 | 0.677165 | 0.688525 |
| FaceDiffuser (frozen KineTalk conditions/ARKit) | neutral | 0.0243902 | 0.0243902 | 0.222222 | 0.196078 |
| FaceDiffuser (frozen KineTalk conditions/ARKit) | angry | 0 | 0 | 0.596364 | 0.482759 |
| FaceDiffuser (frozen KineTalk conditions/ARKit) | contempt | 0.27542 | 0.468553 | 0.707447 | 0.659574 |
| FaceDiffuser (frozen KineTalk conditions/ARKit) | disgust | 0.173077 | 0.236364 | 0.710706 | 0.70437 |
| FaceDiffuser (frozen KineTalk conditions/ARKit) | fear | 0.189189 | 0.320487 | 0.544872 | 0.409091 |
| FaceDiffuser (frozen KineTalk conditions/ARKit) | happy | 0.142857 | 0.544218 | 0.812933 | 0.803783 |
| FaceDiffuser (frozen KineTalk conditions/ARKit) | sad | 0.292308 | 0.524333 | 0.741758 | 0.776942 |
| FaceDiffuser (frozen KineTalk conditions/ARKit) | surprise | 0.138614 | 0.0427807 | 0.467925 | 0.518987 |
| Phase41 | neutral | 0.429752 | 0.371747 | 0.487805 | 0.420601 |
| Phase41 | angry | 0.290909 | 0.176471 | 0.401674 | 0.26484 |
| Phase41 | contempt | 0.792176 | 0.746606 | 0.81203 | 0.771429 |
| Phase41 | disgust | 0.808612 | 0.761682 | 0.774775 | 0.740385 |
| Phase41 | fear | 0.331915 | 0.27193 | 0.425197 | 0.154589 |
| Phase41 | happy | 0.949153 | 0.915942 | 0.949153 | 0.925287 |
| Phase41 | sad | 0.838384 | 0.808864 | 0.875318 | 0.826196 |
| Phase41 | surprise | 0.647826 | 0.621444 | 0.663677 | 0.62753 |
| Phase42 | neutral | 0.390805 | 0.342466 | 0.455814 | 0.395161 |
| Phase42 | angry | 0.293578 | 0.176471 | 0.376068 | 0.203883 |
| Phase42 | contempt | 0.790588 | 0.743875 | 0.800959 | 0.76082 |
| Phase42 | disgust | 0.796253 | 0.751131 | 0.761905 | 0.735632 |
| Phase42 | fear | 0.192308 | 0.182692 | 0.258065 | 0.10101 |
| Phase42 | happy | 0.911243 | 0.894895 | 0.917647 | 0.89759 |
| Phase42 | sad | 0.846154 | 0.787535 | 0.88946 | 0.819095 |
| Phase42 | surprise | 0.62955 | 0.600442 | 0.63913 | 0.615063 |

## table7_geometry_groups

| method | group_type | group | n | arkit_mbe | arkit_lbe | lve_mean_mm_mean | eve_mean_mm_mean | jawOpen/centered_correlation | jawOpen/pred_q90_q10 |
|---|---|---|---|---|---|---|---|---|---|
| Phase41 | emotion | neutral | 80 | 0.608191 | 0.402287 | 2.47547 | 0.369928 | 0.499701 | 0.0838604 |
| Phase41 | emotion | angry | 185 | 0.844874 | 0.425994 | 3.83337 | 0.709885 | 0.468085 | 0.147779 |
| Phase41 | emotion | contempt | 177 | 1.00393 | 0.384497 | 4.19485 | 1.06287 | 0.464792 | 0.0671473 |
| Phase41 | emotion | disgust | 187 | 0.819063 | 0.389271 | 2.56023 | 0.77305 | 0.400659 | 0.110091 |
| Phase41 | emotion | fear | 187 | 0.849679 | 0.393718 | 3.65628 | 0.716906 | 0.463824 | 0.195059 |
| Phase41 | emotion | happy | 181 | 0.701873 | 0.303794 | 3.25031 | 0.523028 | 0.49623 | 0.0811743 |
| Phase41 | emotion | sad | 188 | 0.781086 | 0.265836 | 2.04411 | 0.93031 | 0.520373 | 0.12524 |
| Phase41 | emotion | surprise | 182 | 0.762075 | 0.432187 | 3.18988 | 0.57323 | 0.451763 | 0.163639 |
| Phase41 | speaker | M025 | 606 | 0.840763 | 0.424926 | 3.61229 | 0.598439 | 0.481388 | 0.124879 |
| Phase41 | speaker | M037 | 371 | 0.80614 | 0.413202 | 3.386 | 0.738048 | 0.449571 | 0.122107 |
| Phase41 | speaker | M039 | 390 | 0.765918 | 0.252014 | 2.35921 | 0.936146 | 0.466296 | 0.128302 |
| Phase42 | emotion | neutral | 80 | 0.594628 | 0.401557 | 2.48268 | 0.339871 | 0.479741 | 0.0673897 |
| Phase42 | emotion | angry | 185 | 0.829343 | 0.40545 | 3.66124 | 0.701645 | 0.458773 | 0.137218 |
| Phase42 | emotion | contempt | 177 | 1.01347 | 0.376778 | 4.30576 | 1.06283 | 0.367752 | 0.0473684 |
| Phase42 | emotion | disgust | 187 | 0.781634 | 0.336935 | 2.5309 | 0.752427 | 0.406206 | 0.103079 |
| Phase42 | emotion | fear | 187 | 0.765527 | 0.358133 | 3.28627 | 0.628551 | 0.488936 | 0.178577 |
| Phase42 | emotion | happy | 181 | 0.679443 | 0.292207 | 3.07059 | 0.498719 | 0.525099 | 0.065672 |
| Phase42 | emotion | sad | 188 | 0.789252 | 0.270409 | 2.27871 | 0.95894 | 0.53369 | 0.108868 |
| Phase42 | emotion | surprise | 182 | 0.720196 | 0.4075 | 3.03794 | 0.523149 | 0.457782 | 0.138505 |
| Phase42 | speaker | M025 | 606 | 0.793417 | 0.383399 | 3.40335 | 0.558832 | 0.462536 | 0.0972146 |
| Phase42 | speaker | M037 | 371 | 0.793565 | 0.41006 | 3.41839 | 0.718582 | 0.448465 | 0.117592 |
| Phase42 | speaker | M039 | 390 | 0.761315 | 0.249281 | 2.39051 | 0.933009 | 0.481414 | 0.120137 |

## table8_training_protocol

| method | scope | fit_clips | epochs | updates | train_seed | draws |
|---|---|---|---|---|---|---|
| VOCA-core (shared audio/ARKit) | shared-audio ARKit adaptation | 12536 | 80 | — | 42 | 1 |
| FaceFormer (shared audio/ARKit) | shared-audio ARKit adaptation | 12536 | 100 | — | 42 | 1 |
| EmoTalk-core (shared audio/ARKit) | shared-audio ARKit adaptation | 12536 | 80 | — | 42 | 1 |
| FaceDiffuser (frozen KineTalk conditions/ARKit) | shared-audio ARKit adaptation | 12536 | 100 | — | 42 | 3 |
| Phase41 | neutral B0 + audio expression + independent reference | 10903 | 24 | 16368 | 47 | prior mean |
| Phase42 | neutral B0 + audio expression + independent reference | 10903 | 24 | 16368 | 47 | prior mean |
