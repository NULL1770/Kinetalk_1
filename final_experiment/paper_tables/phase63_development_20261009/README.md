# 开发集实验表：同协议基线与训练消融

Material Passport: MODE=validate; STATUS=ANALYZED; sources为已完成、SHA核验的原报告；本次汇总未重新训练/重新推理。
全部1,367条开发集、相同原生有效帧、51观察通道、固定rig和四冻结probe。主表统一clip[0,1]；raw完整表同时导出。未读取sealed。
基线是ARKit共享音频改编，非原论文官方分数。训练数据/预算/seed不同，见table8；不是严格等预算SOTA结论。FaceDiffuser原预测无checkpoint哈希绑定，原生时钟/GT一致，历史报告有此限制；其余三基线原首batch权重重放通过。本次远端checkpoint/预测SHA重新核验。
Lip/Expression mean为顶点欧氏距离平均(mm)；LVE-max/EVE-max另列，不能混用。这里mesh FDD为上区平方位移能量时间std差的绝对值(mm²)，是沿用实现，不能将所有论文同名FDD当同公式。MBE/LBE为现有系数区域L2误差定义，不能拿其他rig的裸数值比较。
四F1依次为原128/原64/辅助128/辅助64整段统计probe；不是逐帧情感GT。jaw范围GT平均0.175279，范围越大不必越好；相关与位移误差联合判断，不能以平滑低误差替代正确动态。
table4只列实际重训练消融；table5是冻结推理干预，不混为训练模块消融。oracle不进入方法排名。仅单seed结果，不报伪造多seed误差条。
若parent_epochs/parent_updates非零，epochs/updates为追加训练预算，总预算必须加父模型；只在相同父模型与追加预算内比较匹配方式，不能将不同父模型候选当单变量消融。
analytic_fit_passes/analytic_fit_clips非零的候选实际进行了TRAIN监督解析拟合，updates=0只代表追加SGD步数为零，不能称未训练或与SGD同预算。g/u/gu复用同一次固定ridge求解，fitted_coefficients列明替换的现有mean参数数目；没有新增网络，方差未重新拟合。
inherited_analytic_fit_*记录父模型已有的解析拟合，不是本轮重新拟合。Phase46两组均只更新decoder、使用双参考聚合，总重建系数1.5且步数相同；mixed每步两次重建并运行posterior，deploy一次，因此计算量不相同。与父模型对比同时改变了receiver训练和参考聚合，不能当成单变量参考消融。
Phase47为冻结Phase45-u加TRAIN监督拟合的clip常量接收器；parent checkpoint和correction分别SHA绑定。latent/reference分别5044/29380个仿射系数，容量不同；原始centered动态不变，clip后幅度与闭嘴仍需实测。不修改主评分器或训练情感probe，raw和四probe一并保留；不能由F1或小幅均值改善宣称全部动态正确。
论文实践：EmoTalk Table5分别检查emotion disentangling encoder、emotion-guided attention、Lvel/Lcls、HDTF数据及encoder替换；MEDTalk §4.5/Table3检查overlap exchange、cycle exchange、disentangle、intensity和text。本项目应围绕自己的g/u职责、style/reference与teacher/student提出并重训练消融；不能直接照搬其模块名。
统计核验11/11已检查：分组结果另列以检查聚合反转；不由3人推断总体个体；MEAD演员/伪GT选择偏差保留；无协变量调整/collider推断；报告8类而非只happy；不选极端片段宣称回归改善；所有1367无删坏样本；所有probe/raw保留；明确多轮开发探索；冻结交换非因果独立证明；教师看到GT非部署预测/因果反向结论。没有开展显著性检验，三开发身份及单训练seed不足以证明统计稳定。

## table1_geometry

| method | lve_mean_mm_mean | eve_mean_mm_mean | vertex_lve_sqrt_mean | eye_forehead_eve_sqrt_mean | vertex_fdd_absolute_mm2_mean |
|---|---|---|---|---|---|
| VOCA-core (shared audio/ARKit) | 3.01442 | 0.703497 | 5.61668 | 2.41175 | 144.238 |
| FaceFormer (shared audio/ARKit) | 4.53784 | 1.1383 | 8.28442 | 3.78867 | 415.804 |
| EmoTalk-core (shared audio/ARKit) | 2.97323 | 0.691868 | 5.56334 | 2.41675 | 152.495 |
| FaceDiffuser (frozen KineTalk conditions/ARKit) | 3.07168 | 0.754096 | 5.86398 | 2.58438 | 137.962 |
| Phase52 | 3.08177 | 0.692554 | 5.92277 | 2.351 | 139.981 |
| Phase53-style-only | 3.26256 | 0.660521 | 6.13245 | 2.23756 | 143.872 |
| Phase63-native-affine | 3.28904 | 0.71611 | 6.16055 | 2.33049 | 161.885 |

## table2_coefficients

| method | arkit_mbe | arkit_lbe | arkit_fdd_absolute | arkit_fdd_signed |
|---|---|---|---|---|
| VOCA-core (shared audio/ARKit) | 0.746703 | 0.332384 | 0.102816 | 0.0951416 |
| FaceFormer (shared audio/ARKit) | 1.2968 | 0.544334 | 0.350484 | -0.331659 |
| EmoTalk-core (shared audio/ARKit) | 0.745003 | 0.331531 | 0.0945103 | 0.0652765 |
| FaceDiffuser (frozen KineTalk conditions/ARKit) | 0.792727 | 0.32311 | 0.0997374 | 0.0751161 |
| Phase52 | 0.762598 | 0.355341 | 0.103766 | 0.0981366 |
| Phase53-style-only | 0.763407 | 0.369546 | 0.108176 | 0.104091 |
| Phase63-native-affine | 0.798408 | 0.389571 | 0.118001 | 0.117195 |

## table3_dynamics_semantics

| method | mouth_displacement_mse | jaw_centered_correlation | jaw_q90_q10 | F1_1 | F1_2 | F1_3 | F1_4 |
|---|---|---|---|---|---|---|---|
| VOCA-core (shared audio/ARKit) | 0.000795734 | 0.554159 | 0.136455 | 0.497716 | 0.484458 | 0.469954 | 0.475981 |
| FaceFormer (shared audio/ARKit) | 0.00085878 | 0.433926 | 0.153497 | 0.123846 | 0.17603 | 0.188483 | 0.188549 |
| EmoTalk-core (shared audio/ARKit) | 0.00142921 | 0.565796 | 0.154294 | 0.591884 | 0.686229 | 0.739398 | 0.730805 |
| FaceDiffuser (frozen KineTalk conditions/ARKit) | 0.000935414 | 0.452377 | 0.135109 | 0.154482 | 0.270141 | 0.600528 | 0.568948 |
| Phase52 | 0.00081486 | 0.495885 | 0.121521 | 0.620855 | 0.626592 | 0.690247 | 0.629989 |
| Phase53-style-only | 0.000820078 | 0.477797 | 0.124295 | 0.676805 | 0.650936 | 0.734088 | 0.6813 |
| Phase63-native-affine | 0.00092873 | 0.446388 | 0.0938419 | 0.464785 | 0.475715 | 0.481048 | 0.494836 |

## table4_trained_ablation

| method | response_head | prior_variance | center_local | reference_training | matching | style_modulation | posture_supervision | fit_clips | parent_epochs | parent_updates | updates | analytic_fit_passes | analytic_fit_clips | fitted_coefficients | inherited_analytic_fit_passes | inherited_analytic_fit_clips | trainable_module | reconstruction_passes_per_update | arkit_mbe | arkit_lbe | lve_mean_mm_mean | jaw_centered_correlation | jaw_q90_q10 | F1_1 | F1_2 | F1_3 | F1_4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Phase52 | residual | learned | True | diverse_neutral_two | reference_style_only | joint | all | 10903 | 24 | 16368 | 5456 | 0 | 0 | 0 | 1 | 10903 | style_encoder+posture_bias+style_modulation_columns | 1 | 0.762598 | 0.355341 | 3.08177 | 0.495885 | 0.121521 | 0.620855 | 0.626592 | 0.690247 | 0.629989 |
| Phase53-style-only | residual | learned | True | diverse_neutral_two | reference_style_only | factorized | neutral_only | 10903 | 24 | 16368 | 5456 | 0 | 0 | 0 | 1 | 10903 | style_encoder+posture_bias+style_modulation_columns | 1 | 0.763407 | 0.369546 | 3.26256 | 0.477797 | 0.124295 | 0.676805 | 0.650936 | 0.734088 | 0.6813 |
| Phase63-native-affine | native_affine | learned | True | diverse_neutral_two | native_affine_response | factorized | all | 10903 | 32 | 21824 | 5456 | 0 | 0 | 0 | 1 | 10903 | native_affine_decoder_except_posture_bias | 1 | 0.798408 | 0.389571 | 3.28904 | 0.446388 | 0.0938419 | 0.464785 | 0.475715 | 0.481048 | 0.494836 |

## table5_inference_interventions

| method | intervention | n | arkit_mbe | arkit_lbe | jaw_centered_correlation | jaw_q90_q10 | brows/centered_correlation |
|---|---|---|---|---|---|---|---|
| Phase52 | normal | 96 | 0.788806 | 0.363703 | 0.469676 | 0.119049 | 0.023846 |
| Phase52 | static_u | 96 | 0.791329 | 0.36568 | 0.425489 | 0.0991608 | 0.0104758 |
| Phase52 | reverse_u | 96 | 0.797603 | 0.369934 | 0.397767 | 0.108412 | 0.022868 |
| Phase52 | shuffle_u | 96 | 0.794078 | 0.368158 | 0.400556 | 0.103316 | 0.0157587 |
| Phase52 | wrong_reference | 96 | 0.899439 | 0.431022 | 0.459437 | 0.124902 | 0.0189436 |
| Phase52 | reference_A | 96 | 0.798745 | 0.362712 | 0.456346 | 0.111901 | 0.0197009 |
| Phase52 | reference_B | 96 | 0.789214 | 0.36644 | 0.476243 | 0.12577 | 0.0303043 |
| Phase52 | wrong_matched_audio | 96 | 0.796798 | 0.359532 | 0.40332 | 0.104686 | 0.0285807 |
| Phase52 | gt | 96 | 0 | 0 | 1 | 0.199388 | 1 |
| Phase52 | gt_static | 96 | 0.395301 | 0.250083 | 5.19735e-09 | 0 | -1.78116e-07 |
| Phase52 | gt_reverse | 96 | 0.557077 | 0.335443 | -8.869e-06 | 0.199388 | -0.135361 |
| Phase52 | gt_shift | 96 | 0.427103 | 0.265775 | 0.0838484 | 0.199388 | 0.409335 |
| Phase52 | gt_gain | 96 | 0.352843 | 0.144018 | 1 | 0.249235 | 1 |
| Phase53-style-only | normal | 96 | 0.788985 | 0.377382 | 0.457556 | 0.121991 | 0.0101243 |
| Phase53-style-only | static_u | 96 | 0.790791 | 0.37802 | 0.405201 | 0.103901 | 0.000160323 |
| Phase53-style-only | reverse_u | 96 | 0.797464 | 0.384253 | 0.381089 | 0.112486 | 0.00948495 |
| Phase53-style-only | shuffle_u | 96 | 0.793885 | 0.381658 | 0.384186 | 0.106705 | 0.00448703 |
| Phase53-style-only | wrong_reference | 96 | 0.914223 | 0.46232 | 0.448263 | 0.132289 | 0.00122704 |
| Phase53-style-only | reference_A | 96 | 0.791095 | 0.378618 | 0.456057 | 0.119668 | 0.0104604 |
| Phase53-style-only | reference_B | 96 | 0.791517 | 0.378707 | 0.462124 | 0.127128 | 0.0108122 |
| Phase53-style-only | wrong_matched_audio | 96 | 0.795042 | 0.373614 | 0.378632 | 0.108648 | 0.0170522 |
| Phase53-style-only | gt | 96 | 0 | 0 | 1 | 0.199388 | 1 |
| Phase53-style-only | gt_static | 96 | 0.395301 | 0.250083 | 5.19735e-09 | 0 | -1.78116e-07 |
| Phase53-style-only | gt_reverse | 96 | 0.557077 | 0.335443 | -8.869e-06 | 0.199388 | -0.135361 |
| Phase53-style-only | gt_shift | 96 | 0.427103 | 0.265775 | 0.0838484 | 0.199388 | 0.409335 |
| Phase53-style-only | gt_gain | 96 | 0.352843 | 0.144018 | 1 | 0.249235 | 1 |
| Phase63-native-affine | normal | 96 | 0.809968 | 0.401827 | 0.410687 | 0.0943008 | -0.00542848 |
| Phase63-native-affine | static_u | 96 | 0.814509 | 0.405905 | 0.356349 | 0.0760942 | 0.0120533 |
| Phase63-native-affine | reverse_u | 96 | 0.820337 | 0.410394 | 0.33061 | 0.085935 | 0.00188539 |
| Phase63-native-affine | shuffle_u | 96 | 0.817295 | 0.408376 | 0.326208 | 0.0847436 | 0.00161154 |
| Phase63-native-affine | wrong_reference | 96 | 0.879622 | 0.427114 | 0.418032 | 0.0947527 | 0.0202148 |
| Phase63-native-affine | reference_A | 96 | 0.815573 | 0.407279 | 0.411707 | 0.0933892 | -0.00310057 |
| Phase63-native-affine | reference_B | 96 | 0.809105 | 0.399912 | 0.410923 | 0.0956289 | -0.00700161 |
| Phase63-native-affine | wrong_matched_audio | 96 | 0.827469 | 0.39697 | 0.325527 | 0.0850129 | 0.0229538 |
| Phase63-native-affine | gt | 96 | 0 | 0 | 1 | 0.199388 | 1 |
| Phase63-native-affine | gt_static | 96 | 0.395301 | 0.250083 | 5.19735e-09 | 0 | -1.78116e-07 |
| Phase63-native-affine | gt_reverse | 96 | 0.557077 | 0.335443 | -8.869e-06 | 0.199388 | -0.135361 |
| Phase63-native-affine | gt_shift | 96 | 0.427103 | 0.265775 | 0.0838484 | 0.199388 | 0.409335 |
| Phase63-native-affine | gt_gain | 96 | 0.352843 | 0.144018 | 1 | 0.249235 | 1 |

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
| Phase52 | neutral | 0.520231 | 0.486772 | 0.522293 | 0.52381 |
| Phase52 | angry | 0.587814 | 0.498054 | 0.651163 | 0.547445 |
| Phase52 | contempt | 0.665339 | 0.642023 | 0.722944 | 0.662626 |
| Phase52 | disgust | 0.786364 | 0.764835 | 0.785553 | 0.767494 |
| Phase52 | fear | 0.170616 | 0.416667 | 0.445255 | 0.283186 |
| Phase52 | happy | 0.9 | 0.905882 | 0.889552 | 0.876877 |
| Phase52 | sad | 0.699708 | 0.652038 | 0.868852 | 0.723529 |
| Phase52 | surprise | 0.636771 | 0.646465 | 0.636364 | 0.654945 |
| Phase53-style-only | neutral | 0.546584 | 0.474227 | 0.554839 | 0.533333 |
| Phase53-style-only | angry | 0.648276 | 0.472 | 0.732484 | 0.636986 |
| Phase53-style-only | contempt | 0.714588 | 0.717622 | 0.76298 | 0.739606 |
| Phase53-style-only | disgust | 0.763158 | 0.717213 | 0.782998 | 0.753247 |
| Phase53-style-only | fear | 0.34188 | 0.439024 | 0.619883 | 0.369099 |
| Phase53-style-only | happy | 0.868502 | 0.888889 | 0.868502 | 0.854489 |
| Phase53-style-only | sad | 0.857143 | 0.803681 | 0.90027 | 0.882022 |
| Phase53-style-only | surprise | 0.674312 | 0.694836 | 0.650746 | 0.681614 |
| Phase63-native-affine | neutral | 0.529801 | 0.468085 | 0.577181 | 0.533333 |
| Phase63-native-affine | angry | 0.0319149 | 0 | 0.0421053 | 0.149254 |
| Phase63-native-affine | contempt | 0.435 | 0.466576 | 0.5 | 0.460509 |
| Phase63-native-affine | disgust | 0.469565 | 0.596059 | 0.540682 | 0.387692 |
| Phase63-native-affine | fear | 0.214876 | 0.290541 | 0.282686 | 0.319066 |
| Phase63-native-affine | happy | 0.821656 | 0.858025 | 0.821656 | 0.814103 |
| Phase63-native-affine | sad | 0.741573 | 0.825065 | 0.691057 | 0.741228 |
| Phase63-native-affine | surprise | 0.473896 | 0.30137 | 0.393013 | 0.553506 |

## table7_geometry_groups

| method | group_type | group | n | arkit_mbe | arkit_lbe | lve_mean_mm_mean | eve_mean_mm_mean | jawOpen/centered_correlation | jawOpen/pred_q90_q10 |
|---|---|---|---|---|---|---|---|---|---|
| Phase52 | emotion | neutral | 80 | 0.583978 | 0.387618 | 2.39404 | 0.409329 | 0.509641 | 0.0714103 |
| Phase52 | emotion | angry | 185 | 0.795835 | 0.400861 | 3.63499 | 0.634963 | 0.487344 | 0.151575 |
| Phase52 | emotion | contempt | 177 | 0.975177 | 0.388483 | 4.39435 | 0.98587 | 0.475788 | 0.0568914 |
| Phase52 | emotion | disgust | 187 | 0.773747 | 0.35461 | 2.43841 | 0.706066 | 0.421969 | 0.111438 |
| Phase52 | emotion | fear | 187 | 0.734184 | 0.355886 | 2.99437 | 0.684819 | 0.50795 | 0.203979 |
| Phase52 | emotion | happy | 181 | 0.694329 | 0.335518 | 3.43296 | 0.507529 | 0.51201 | 0.0662454 |
| Phase52 | emotion | sad | 188 | 0.742969 | 0.25526 | 2.2773 | 0.838679 | 0.550135 | 0.119554 |
| Phase52 | emotion | surprise | 182 | 0.706497 | 0.38594 | 2.77778 | 0.61746 | 0.509538 | 0.158494 |
| Phase52 | speaker | M025 | 606 | 0.805663 | 0.406773 | 3.41528 | 0.600946 | 0.494002 | 0.101817 |
| Phase52 | speaker | M037 | 371 | 0.764321 | 0.398503 | 3.35825 | 0.666387 | 0.471389 | 0.147233 |
| Phase52 | speaker | M039 | 390 | 0.694042 | 0.234365 | 2.30054 | 0.85979 | 0.522112 | 0.127679 |
| Phase53-style-only | emotion | neutral | 80 | 0.597018 | 0.418304 | 2.55806 | 0.360612 | 0.498672 | 0.0759557 |
| Phase53-style-only | emotion | angry | 185 | 0.784589 | 0.422958 | 3.79906 | 0.600194 | 0.477528 | 0.149386 |
| Phase53-style-only | emotion | contempt | 177 | 0.95426 | 0.399562 | 4.66232 | 0.902675 | 0.406757 | 0.05635 |
| Phase53-style-only | emotion | disgust | 187 | 0.758168 | 0.349996 | 2.715 | 0.684067 | 0.408617 | 0.107588 |
| Phase53-style-only | emotion | fear | 187 | 0.769629 | 0.366845 | 3.13798 | 0.69496 | 0.501924 | 0.203421 |
| Phase53-style-only | emotion | happy | 181 | 0.711275 | 0.364913 | 3.74165 | 0.475012 | 0.501023 | 0.0731581 |
| Phase53-style-only | emotion | sad | 188 | 0.72905 | 0.279401 | 2.33752 | 0.796428 | 0.542877 | 0.131805 |
| Phase53-style-only | emotion | surprise | 182 | 0.715726 | 0.385218 | 2.8353 | 0.602693 | 0.493954 | 0.165081 |
| Phase53-style-only | speaker | M025 | 606 | 0.789969 | 0.41561 | 3.72145 | 0.547233 | 0.457586 | 0.0988268 |
| Phase53-style-only | speaker | M037 | 371 | 0.77569 | 0.417079 | 3.44374 | 0.635026 | 0.473354 | 0.179655 |
| Phase53-style-only | speaker | M039 | 390 | 0.710449 | 0.252753 | 2.37718 | 0.860807 | 0.51343 | 0.111206 |
| Phase63-native-affine | emotion | neutral | 80 | 0.607369 | 0.422644 | 2.72537 | 0.386561 | 0.461109 | 0.083488 |
| Phase63-native-affine | emotion | angry | 185 | 0.840038 | 0.423466 | 3.5956 | 0.674661 | 0.43413 | 0.104337 |
| Phase63-native-affine | emotion | contempt | 177 | 1.00487 | 0.452028 | 4.98616 | 0.904753 | 0.46873 | 0.0752023 |
| Phase63-native-affine | emotion | disgust | 187 | 0.812662 | 0.411775 | 2.59683 | 0.762363 | 0.385241 | 0.0825418 |
| Phase63-native-affine | emotion | fear | 187 | 0.783524 | 0.356806 | 3.18374 | 0.841825 | 0.442861 | 0.120909 |
| Phase63-native-affine | emotion | happy | 181 | 0.701464 | 0.372272 | 3.72651 | 0.464375 | 0.472181 | 0.0771366 |
| Phase63-native-affine | emotion | sad | 188 | 0.73249 | 0.267667 | 2.10064 | 0.728532 | 0.489693 | 0.0898368 |
| Phase63-native-affine | emotion | surprise | 182 | 0.804418 | 0.433816 | 3.18659 | 0.78047 | 0.426715 | 0.110403 |
| Phase63-native-affine | speaker | M025 | 606 | 0.881254 | 0.471036 | 3.77426 | 0.677214 | 0.443093 | 0.0880934 |
| Phase63-native-affine | speaker | M037 | 371 | 0.778589 | 0.421572 | 3.5063 | 0.679351 | 0.405234 | 0.116154 |
| Phase63-native-affine | speaker | M039 | 390 | 0.688531 | 0.232544 | 2.32839 | 0.811517 | 0.490655 | 0.0815488 |

## table8_training_protocol

| method | scope | fit_clips | parent_epochs | parent_updates | epochs | updates | analytic_fit_passes | analytic_fit_clips | fitted_coefficients | inherited_analytic_fit_passes | inherited_analytic_fit_clips | trainable_module | reconstruction_passes_per_update | matching | train_seed | draws |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VOCA-core (shared audio/ARKit) | shared-audio ARKit adaptation | 12536 | — | — | 80 | — | — | — | — | — | — | — | — | — | 42 | 1 |
| FaceFormer (shared audio/ARKit) | shared-audio ARKit adaptation | 12536 | — | — | 100 | — | — | — | — | — | — | — | — | — | 42 | 1 |
| EmoTalk-core (shared audio/ARKit) | shared-audio ARKit adaptation | 12536 | — | — | 80 | — | — | — | — | — | — | — | — | — | 42 | 1 |
| FaceDiffuser (frozen KineTalk conditions/ARKit) | shared-audio ARKit adaptation | 12536 | — | — | 100 | — | — | — | — | — | — | — | — | — | 42 | 3 |
| Phase52 | neutral B0 + audio expression + independent reference | 10903 | 24 | 16368 | 8 | 5456 | 0 | 0 | 0 | 1 | 10903 | style_encoder+posture_bias+style_modulation_columns | 1 | reference_style_only | 47 | prior mean |
| Phase53-style-only | neutral B0 + audio expression + independent reference | 10903 | 24 | 16368 | 8 | 5456 | 0 | 0 | 0 | 1 | 10903 | style_encoder+posture_bias+style_modulation_columns | 1 | reference_style_only | 47 | prior mean |
| Phase63-native-affine | neutral B0 + audio expression + independent reference | 10903 | 32 | 21824 | 8 | 5456 | 0 | 0 | 0 | 1 | 10903 | native_affine_decoder_except_posture_bias | 1 | native_affine_response | 47 | prior mean |
