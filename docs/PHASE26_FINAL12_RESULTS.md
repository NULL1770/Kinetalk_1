# Phase26 fixed12结果（2026-10-06）

全部三种子47/48/49、两个坐标臂均完成12轮/9408更新。1367个完整native验证段×三个固定噪声draw42/123/2026，四个冻结TRAIN动作probe，raw/clip_all分别评分。前1568更新权重及真实输入/GT/mask/noise均逐位等各自stable2基线；B0、身份、motion teacher保持冻结。未挑epoch/seed/draw，未打开sealed test，未推广默认。

## 固定输出协议的生成指标

以下是三seed×三draw等权的`clip_all`，将最终ARKit系数裁到合法[0,1]，与历史同协议比较。

| 指标 | 两轮standardized | 十二轮centered对照 | 十二轮standardized |
|---|---:|---:|---:|
| MBE↓ | .855815 | .887183 | .870982 |
| LBE↓ | .416628 | .434720 | .421170 |
| 原128生成macro-F1↑ | .778163 | .292569 | .796875 |
| 原64生成macro-F1↑ | .665453 | .353879 | .721358 |
| 辅助128生成macro-F1↑ | .792092 | .458828 | .800471 |
| 辅助64生成macro-F1↑ | .748873 | .459079 | .783213 |
| 口型位移MSE↓ | .002201024 | .002057878 | .001816626 |

三个种子各自的两套原生成F1分别为47=.789976/.727179、48=.802586/.717607、49=.798063/.719287，均>.7；MBE分别.867888/.861614/.883445，均未到.7。这里是最终动作的独立F1，不是音频分类头或teacher头。

raw未经裁剪时原F1=.619331/.591701，MBE=.884162，越界系数占20.40%。裁剪会显著影响统计probe结果，不能隐去raw，更不能仅用clip F1证明表达逼真。GT原probe宏F1本身约.66/.64，识别更容易不代表几何更准确。

## 剩余误差

十二轮嘴部范围.090651，GT .091991；jaw范围.181152，GT .175279。口型位移MSE比两轮下降17.46%，但mouth centered RMS .064861仍低于GT .071162，jaw centered RMS .075132接近GT .074501。各通道情况不同，不能全嘴统一放大。

眉MSE=.054552，其中每clip平均姿态偏差MSE=.052693，占96.59%；嘴MSE=.014781，平均偏差=.007964，占53.88%。延长训练改善动态误差，但平均姿态偏差变大，导致MBE/LBE略差，不支持继续单纯延长预算。

中性jaw时序相关性centered对照→standardized：47=.283986→.255299、48=.293686→.247184、49=.285504→.257602。三个联合质量gate均false，默认timing000保持；F1目标部分达成，优化总体目标未完成。

各类原128/64 F1（三seed×三draw等权）：

| 类别 | 原128 | 原64 |
|---|---:|---:|
| neutral | .438 | .377 |
| angry | .849 | .798 |
| contempt | .812 | .782 |
| disgust | .870 | .798 |
| fear | .798 | .610 |
| happy | .914 | .893 |
| sad | .885 | .807 |
| surprise | .808 | .705 |

宏F1仍以neutral/fear为弱项，happy已经容易识别。下一步优先定位连续global条件与静态姿态误差，以及B0和残差对jaw时序的影响，不能继续盲目强化happy或增加loss。

## 下一步冻结诊断

Phase27对当前三个固定standardized12生成器运行global-only干预：首先对全部1367×3draw正常audio输出逐位重放；通过后只换真实query motion-teacher global，保留u_a、intensity、身份、B0、noise和native时钟。评估全四probe/raw/clip/几何/时序/分组/paired CI，并记录每clip B0、identity baseline、真实/生成残差平均值，定位静态偏差来源。该oracle读取query GT，仅用于定位，不能计入部署成绩。

完整结果根目录：`final_experiment/evaluation/diagnostics/phase26_channel_coordinates_stable_budget_20261006/`。逐SHA增量下载完成凭`download_verified.json`和对应manifest判断；归档保留optimizer和原curves，可核SHA恢复，不重训。

论文SOTA还需要同split、同参考、同音频预算、同rig、同指标的对比方法与一次sealed最终评估，现在不能宣称完成。
