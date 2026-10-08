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
| Phase43-a | 3.31945 | 0.736281 | 6.30159 | 2.52246 | 146.237 |
| Phase43-b | 3.03315 | 0.718263 | 5.81135 | 2.49793 | 145.763 |
| Phase43-ab | 3.20535 | 0.750063 | 6.16146 | 2.62067 | 147.988 |

## table2_coefficients

| method | arkit_mbe | arkit_lbe | arkit_fdd_absolute | arkit_fdd_signed |
|---|---|---|---|---|
| VOCA-core (shared audio/ARKit) | 0.746703 | 0.332384 | 0.102816 | 0.0951416 |
| FaceFormer (shared audio/ARKit) | 1.2968 | 0.544334 | 0.350484 | -0.331659 |
| EmoTalk-core (shared audio/ARKit) | 0.745003 | 0.331531 | 0.0945103 | 0.0652765 |
| FaceDiffuser (frozen KineTalk conditions/ARKit) | 0.792727 | 0.32311 | 0.0997374 | 0.0751161 |
| Phase41 | 0.810014 | 0.372413 | 0.112121 | 0.110557 |
| Phase42 | 0.784298 | 0.352371 | 0.112287 | 0.110094 |
| Phase43-a | 0.827621 | 0.388066 | 0.110595 | 0.106705 |
| Phase43-b | 0.787836 | 0.358676 | 0.110298 | 0.10748 |
| Phase43-ab | 0.819662 | 0.37271 | 0.110897 | 0.108724 |

## table3_dynamics_semantics

| method | mouth_displacement_mse | jaw_centered_correlation | jaw_q90_q10 | F1_1 | F1_2 | F1_3 | F1_4 |
|---|---|---|---|---|---|---|---|
| VOCA-core (shared audio/ARKit) | 0.000795734 | 0.554159 | 0.136455 | 0.497716 | 0.484458 | 0.469954 | 0.475981 |
| FaceFormer (shared audio/ARKit) | 0.00085878 | 0.433926 | 0.153497 | 0.123846 | 0.17603 | 0.188483 | 0.188549 |
| EmoTalk-core (shared audio/ARKit) | 0.00142921 | 0.565796 | 0.154294 | 0.591884 | 0.686229 | 0.739398 | 0.730805 |
| FaceDiffuser (frozen KineTalk conditions/ARKit) | 0.000935414 | 0.452377 | 0.135109 | 0.154482 | 0.270141 | 0.600528 | 0.568948 |
| Phase41 | 0.000815956 | 0.468447 | 0.125103 | 0.636091 | 0.584336 | 0.673703 | 0.591357 |
| Phase42 | 0.000794127 | 0.464103 | 0.109285 | 0.60631 | 0.559938 | 0.637381 | 0.566032 |
| Phase43-a | 0.00081735 | 0.473212 | 0.142617 | 0.695182 | 0.645552 | 0.722729 | 0.647927 |
| Phase43-b | 0.000816235 | 0.439372 | 0.111435 | 0.638624 | 0.579066 | 0.67592 | 0.600079 |
| Phase43-ab | 0.000813242 | 0.471168 | 0.126986 | 0.672238 | 0.614962 | 0.7031 | 0.616208 |

## table4_trained_ablation

| method | prior_variance | center_local | reference_training | fit_clips | updates | arkit_mbe | arkit_lbe | lve_mean_mm_mean | jaw_centered_correlation | jaw_q90_q10 | F1_1 | F1_2 | F1_3 | F1_4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Phase41 | learned | False | single | 10903 | 16368 | 0.810014 | 0.372413 | 3.19337 | 0.468447 | 0.125103 | 0.636091 | 0.584336 | 0.673703 | 0.591357 |
| Phase42 | unit | False | single | 10903 | 16368 | 0.784298 | 0.352371 | 3.11847 | 0.464103 | 0.109285 | 0.60631 | 0.559938 | 0.637381 | 0.566032 |
| Phase43-a | learned | True | single | 10903 | 16368 | 0.827621 | 0.388066 | 3.31945 | 0.473212 | 0.142617 | 0.695182 | 0.645552 | 0.722729 | 0.647927 |
| Phase43-b | learned | False | mixed | 10903 | 16368 | 0.787836 | 0.358676 | 3.03315 | 0.439372 | 0.111435 | 0.638624 | 0.579066 | 0.67592 | 0.600079 |
| Phase43-ab | learned | True | mixed | 10903 | 16368 | 0.819662 | 0.37271 | 3.20535 | 0.471168 | 0.126986 | 0.672238 | 0.614962 | 0.7031 | 0.616208 |

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
| Phase43-a | normal | 96 | 0.858782 | 0.397201 | 0.449523 | 0.139766 | 0.012053 |
| Phase43-a | static_u | 96 | 0.862059 | 0.39891 | 0.423373 | 0.126117 | 0.00236475 |
| Phase43-a | reverse_u | 96 | 0.864495 | 0.399717 | 0.405746 | 0.131222 | 0.00994021 |
| Phase43-a | shuffle_u | 96 | 0.862615 | 0.399851 | 0.411402 | 0.129964 | -0.00210876 |
| Phase43-a | wrong_reference | 96 | 0.924149 | 0.421381 | 0.435008 | 0.130123 | 0.017217 |
| Phase43-a | reference_A | 96 | 0.83518 | 0.377306 | 0.454978 | 0.115284 | 0.0180453 |
| Phase43-a | reference_B | 96 | 0.873248 | 0.4024 | 0.443447 | 0.145528 | 0.0165126 |
| Phase43-a | wrong_matched_audio | 96 | 0.865375 | 0.388642 | 0.403583 | 0.127883 | 0.0283098 |
| Phase43-a | gt | 96 | 0 | 0 | 1 | 0.199388 | 1 |
| Phase43-a | gt_static | 96 | 0.395301 | 0.250083 | 5.19735e-09 | 0 | -1.78116e-07 |
| Phase43-a | gt_reverse | 96 | 0.557077 | 0.335443 | -8.869e-06 | 0.199388 | -0.135361 |
| Phase43-a | gt_shift | 96 | 0.427103 | 0.265775 | 0.0838484 | 0.199388 | 0.409335 |
| Phase43-a | gt_gain | 96 | 0.352843 | 0.144018 | 1 | 0.249235 | 1 |
| Phase43-b | normal | 96 | 0.813298 | 0.37353 | 0.429339 | 0.113382 | 0.0353288 |
| Phase43-b | static_u | 96 | 0.813701 | 0.374848 | 0.414769 | 0.0987016 | 0.0366306 |
| Phase43-b | reverse_u | 96 | 0.81884 | 0.377367 | 0.393041 | 0.103085 | 0.0417822 |
| Phase43-b | shuffle_u | 96 | 0.815476 | 0.375937 | 0.408419 | 0.102844 | 0.028026 |
| Phase43-b | wrong_reference | 96 | 0.928524 | 0.445339 | 0.405742 | 0.11196 | 0.0207336 |
| Phase43-b | reference_A | 96 | 0.828645 | 0.378076 | 0.400616 | 0.0955186 | 0.0407645 |
| Phase43-b | reference_B | 96 | 0.830112 | 0.373618 | 0.438687 | 0.121553 | 0.0274818 |
| Phase43-b | wrong_matched_audio | 96 | 0.843485 | 0.367717 | 0.379294 | 0.102209 | 0.0374274 |
| Phase43-b | gt | 96 | 0 | 0 | 1 | 0.199388 | 1 |
| Phase43-b | gt_static | 96 | 0.395301 | 0.250083 | 5.19735e-09 | 0 | -1.78116e-07 |
| Phase43-b | gt_reverse | 96 | 0.557077 | 0.335443 | -8.869e-06 | 0.199388 | -0.135361 |
| Phase43-b | gt_shift | 96 | 0.427103 | 0.265775 | 0.0838484 | 0.199388 | 0.409335 |
| Phase43-b | gt_gain | 96 | 0.352843 | 0.144018 | 1 | 0.249235 | 1 |
| Phase43-ab | normal | 96 | 0.835692 | 0.381579 | 0.442638 | 0.128059 | 0.0276687 |
| Phase43-ab | static_u | 96 | 0.837432 | 0.382057 | 0.420901 | 0.113876 | 0.0333456 |
| Phase43-ab | reverse_u | 96 | 0.841181 | 0.384032 | 0.403578 | 0.121183 | 0.0318294 |
| Phase43-ab | shuffle_u | 96 | 0.838443 | 0.383177 | 0.4107 | 0.117448 | 0.0307643 |
| Phase43-ab | wrong_reference | 96 | 0.910419 | 0.426796 | 0.435166 | 0.121377 | 0.0141144 |
| Phase43-ab | reference_A | 96 | 0.828945 | 0.370985 | 0.43368 | 0.0989329 | 0.0287595 |
| Phase43-ab | reference_B | 96 | 0.859299 | 0.394351 | 0.438917 | 0.144427 | 0.0251381 |
| Phase43-ab | wrong_matched_audio | 96 | 0.866741 | 0.380272 | 0.423668 | 0.118242 | 0.0370477 |
| Phase43-ab | gt | 96 | 0 | 0 | 1 | 0.199388 | 1 |
| Phase43-ab | gt_static | 96 | 0.395301 | 0.250083 | 5.19735e-09 | 0 | -1.78116e-07 |
| Phase43-ab | gt_reverse | 96 | 0.557077 | 0.335443 | -8.869e-06 | 0.199388 | -0.135361 |
| Phase43-ab | gt_shift | 96 | 0.427103 | 0.265775 | 0.0838484 | 0.199388 | 0.409335 |
| Phase43-ab | gt_gain | 96 | 0.352843 | 0.144018 | 1 | 0.249235 | 1 |

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
| Phase43-a | neutral | 0.437768 | 0.387597 | 0.486772 | 0.449541 |
| Phase43-a | angry | 0.638596 | 0.589091 | 0.751592 | 0.631579 |
| Phase43-a | contempt | 0.75814 | 0.725275 | 0.778032 | 0.736607 |
| Phase43-a | disgust | 0.823821 | 0.796954 | 0.833333 | 0.805195 |
| Phase43-a | fear | 0.485075 | 0.32 | 0.525253 | 0.214953 |
| Phase43-a | happy | 0.927536 | 0.895522 | 0.915942 | 0.896142 |
| Phase43-a | sad | 0.825485 | 0.790831 | 0.857143 | 0.805333 |
| Phase43-a | surprise | 0.665037 | 0.659142 | 0.633766 | 0.644068 |
| Phase43-b | neutral | 0.424242 | 0.357447 | 0.474576 | 0.426396 |
| Phase43-b | angry | 0.431535 | 0.350877 | 0.501961 | 0.349345 |
| Phase43-b | contempt | 0.765376 | 0.719828 | 0.758929 | 0.781395 |
| Phase43-b | disgust | 0.804762 | 0.763033 | 0.770302 | 0.747045 |
| Phase43-b | fear | 0.294643 | 0.187793 | 0.482759 | 0.165049 |
| Phase43-b | happy | 0.910145 | 0.90379 | 0.915452 | 0.907514 |
| Phase43-b | sad | 0.857143 | 0.774536 | 0.860697 | 0.820513 |
| Phase43-b | surprise | 0.621145 | 0.575221 | 0.642686 | 0.603376 |
| Phase43-ab | neutral | 0.529101 | 0.464455 | 0.534884 | 0.52514 |
| Phase43-ab | angry | 0.631579 | 0.501961 | 0.671141 | 0.607143 |
| Phase43-ab | contempt | 0.700855 | 0.679089 | 0.715517 | 0.68323 |
| Phase43-ab | disgust | 0.812183 | 0.740566 | 0.820388 | 0.774359 |
| Phase43-ab | fear | 0.475806 | 0.350877 | 0.565056 | 0.182692 |
| Phase43-ab | happy | 0.850932 | 0.855385 | 0.850932 | 0.840125 |
| Phase43-ab | sad | 0.710784 | 0.698925 | 0.783715 | 0.688279 |
| Phase43-ab | surprise | 0.666667 | 0.62844 | 0.683168 | 0.628692 |

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
| Phase43-a | emotion | neutral | 80 | 0.636611 | 0.433382 | 2.59521 | 0.359812 | 0.509575 | 0.088047 |
| Phase43-a | emotion | angry | 185 | 0.867818 | 0.447712 | 4.06934 | 0.688439 | 0.464298 | 0.170685 |
| Phase43-a | emotion | contempt | 177 | 1.00784 | 0.406352 | 3.8457 | 1.09887 | 0.471906 | 0.0633566 |
| Phase43-a | emotion | disgust | 187 | 0.826458 | 0.392548 | 2.69148 | 0.76154 | 0.409367 | 0.12672 |
| Phase43-a | emotion | fear | 187 | 0.880298 | 0.412564 | 4.06466 | 0.726266 | 0.457247 | 0.24185 |
| Phase43-a | emotion | happy | 181 | 0.74135 | 0.352 | 3.3984 | 0.537601 | 0.517548 | 0.07263 |
| Phase43-a | emotion | sad | 188 | 0.786972 | 0.247748 | 2.06168 | 0.956842 | 0.526575 | 0.142544 |
| Phase43-a | emotion | surprise | 182 | 0.770314 | 0.440766 | 3.46398 | 0.551856 | 0.450347 | 0.199212 |
| Phase43-a | speaker | M025 | 606 | 0.867912 | 0.470193 | 3.89765 | 0.593043 | 0.479246 | 0.156189 |
| Phase43-a | speaker | M037 | 371 | 0.815291 | 0.413048 | 3.36523 | 0.742698 | 0.470811 | 0.145976 |
| Phase43-a | speaker | M039 | 390 | 0.776746 | 0.236687 | 2.37746 | 0.952747 | 0.466121 | 0.118333 |
| Phase43-b | emotion | neutral | 80 | 0.662898 | 0.418036 | 2.50773 | 0.444775 | 0.47516 | 0.0707865 |
| Phase43-b | emotion | angry | 185 | 0.831338 | 0.396395 | 3.68645 | 0.710581 | 0.455653 | 0.134099 |
| Phase43-b | emotion | contempt | 177 | 1.02535 | 0.381776 | 4.18839 | 1.08624 | 0.396352 | 0.061738 |
| Phase43-b | emotion | disgust | 187 | 0.796134 | 0.378975 | 2.50396 | 0.74939 | 0.367008 | 0.104631 |
| Phase43-b | emotion | fear | 187 | 0.774779 | 0.358344 | 3.19707 | 0.675021 | 0.453305 | 0.177018 |
| Phase43-b | emotion | happy | 181 | 0.664706 | 0.303684 | 3.0399 | 0.470204 | 0.45885 | 0.0621151 |
| Phase43-b | emotion | sad | 188 | 0.744805 | 0.276807 | 2.03882 | 0.830956 | 0.486847 | 0.113738 |
| Phase43-b | emotion | surprise | 182 | 0.73933 | 0.390523 | 2.87225 | 0.631156 | 0.440553 | 0.140875 |
| Phase43-b | speaker | M025 | 606 | 0.793349 | 0.388494 | 3.28804 | 0.580701 | 0.433448 | 0.0999021 |
| Phase43-b | speaker | M037 | 371 | 0.8033 | 0.405398 | 3.35233 | 0.727485 | 0.436134 | 0.119641 |
| Phase43-b | speaker | M039 | 390 | 0.764556 | 0.267899 | 2.33347 | 0.923243 | 0.451656 | 0.12155 |
| Phase43-ab | emotion | neutral | 80 | 0.625097 | 0.412832 | 2.47049 | 0.388497 | 0.508722 | 0.0843788 |
| Phase43-ab | emotion | angry | 185 | 0.828339 | 0.404051 | 3.76956 | 0.677928 | 0.453111 | 0.146353 |
| Phase43-ab | emotion | contempt | 177 | 1.05872 | 0.398423 | 4.20647 | 1.15861 | 0.463792 | 0.0632021 |
| Phase43-ab | emotion | disgust | 187 | 0.812584 | 0.402672 | 2.50053 | 0.754912 | 0.418382 | 0.106689 |
| Phase43-ab | emotion | fear | 187 | 0.862423 | 0.379709 | 3.63863 | 0.760371 | 0.462766 | 0.226183 |
| Phase43-ab | emotion | happy | 181 | 0.727015 | 0.320539 | 3.45847 | 0.541547 | 0.511227 | 0.0604625 |
| Phase43-ab | emotion | sad | 188 | 0.802615 | 0.270409 | 2.07483 | 0.963016 | 0.515831 | 0.123224 |
| Phase43-ab | emotion | surprise | 182 | 0.736955 | 0.41779 | 3.17633 | 0.556816 | 0.457084 | 0.177033 |
| Phase43-ab | speaker | M025 | 606 | 0.847666 | 0.430805 | 3.6711 | 0.614752 | 0.469946 | 0.133581 |
| Phase43-ab | speaker | M037 | 371 | 0.804394 | 0.414531 | 3.36288 | 0.725614 | 0.459467 | 0.129713 |
| Phase43-ab | speaker | M039 | 390 | 0.790672 | 0.242655 | 2.3318 | 0.983574 | 0.484199 | 0.114143 |

## table8_training_protocol

| method | scope | fit_clips | epochs | updates | train_seed | draws |
|---|---|---|---|---|---|---|
| VOCA-core (shared audio/ARKit) | shared-audio ARKit adaptation | 12536 | 80 | — | 42 | 1 |
| FaceFormer (shared audio/ARKit) | shared-audio ARKit adaptation | 12536 | 100 | — | 42 | 1 |
| EmoTalk-core (shared audio/ARKit) | shared-audio ARKit adaptation | 12536 | 80 | — | 42 | 1 |
| FaceDiffuser (frozen KineTalk conditions/ARKit) | shared-audio ARKit adaptation | 12536 | 100 | — | 42 | 3 |
| Phase41 | neutral B0 + audio expression + independent reference | 10903 | 24 | 16368 | 47 | prior mean |
| Phase42 | neutral B0 + audio expression + independent reference | 10903 | 24 | 16368 | 47 | prior mean |
| Phase43-a | neutral B0 + audio expression + independent reference | 10903 | 24 | 16368 | 47 | prior mean |
| Phase43-b | neutral B0 + audio expression + independent reference | 10903 | 24 | 16368 | 47 | prior mean |
| Phase43-ab | neutral B0 + audio expression + independent reference | 10903 | 24 | 16368 | 47 | prior mean |
