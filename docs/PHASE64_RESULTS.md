# Phase64 results

Fixed eight epochs; no checkpoint selection. Strict gate: False

|Metric|Phase53|Phase64|Delta|
|---|---:|---:|---:|
|arkit_mbe|0.763407|0.762427|-0.000979|
|arkit_lbe|0.369546|0.376756|+0.007210|
|lve_mean_mm_mean|3.262565|3.254002|-0.008563|
|jawOpen/pred_q90_q10|0.124295|0.122752|-0.001543|
|jawOpen/gt_q90_q10|0.175279|0.175279|+0.000000|
|jawOpen/centered_correlation|0.477797|0.464588|-0.013209|
|jaw_closure_frame_f1|0.401021|0.435494|+0.034473|
|probe_f1_1|0.676805|0.671319|-0.005486|
|probe_f1_2|0.650936|0.672165|+0.021229|
|probe_f1_3|0.734088|0.722229|-0.011858|
|probe_f1_4|0.681300|0.675266|-0.006033|

Failed criteria: arkit_lbe, jawOpen/centered_correlation, jaw_range_error_reduced, main_probe_not_worse, weak_emotions_improve, own_A_B/mouth/mae, cross_AB/jaw_closure_disagreement

No automatic promotion. Four F1s are frozen whole-clip probes. Full 25-person breakdown and native videos must also be reviewed; 20 people are seen speakers, five are unseen. No sealed results or SOTA claim.
