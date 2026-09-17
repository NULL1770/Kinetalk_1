# 前序动作上下文：固定配对实验

现有2315fit/405内部开发，25Hz中央96帧、身份参考、全部输入来源保持；不读封存test。原始视频审计只取元数据预选的8条fit，不据动作质量删样本。本轮不混入新类别或改变裁剪。

两臂no_history/scheduled_history，同shape/初值、batch、训练噪声、自生成历史噪声、flow time和选择随机数，各12epoch、batch16、seed67、AdamW1e-4/wd1e-5/gradclip1。两臂只差使用历史；所有随机选择即使no_history忽略也照常生成。固定最后轮，不挑最好epoch。

当前chunk16帧，history8帧。小MLP+masked GRU读取按时间排列的归一化动作与相对age，投影到local条件，送入每层flow条件路径；空历史为零。无chunk去均值，避免每16帧强制回到中线。监督为真实九维上脸减去冻结音频预测静态均值，除以fit-only RMS（floor.02）。不减GT全段均值，避免前序GT通过中心化携带未来目标信息。静态均值由原已训audio4Dstate窗口均值+独立neutral参考得到，非心理学真值。

训练每batch先no_grad完整自回归生成一次，使用相同12步Euler和generated history，无teacher干预。训练loss各chunk历史按预定概率选择GT前序或停止梯度的完整自生成前序；teacher概率.75线性降至0，第8轮开始为0，最后五轮全generated。每个batch聚合所有有效chunk的flow误差，按有效帧加权，一次optimizer update。其他loss不增加。音频上下文非因果，不能称流式实时模型。

全局audio分类器、B0、identity、Stage3完整renderer保持冻结。独立local trunk从已对齐adapter起步，新local head0；与flow一起训练。全部43非上脸输出逐值复制上一轮冻结原local Stage3基座。history分支可修改九维均值/形态，它并非上一轮mean-preserving分支，需独立报告raw/centered与情感变化。只比较新两臂才能隔离history作用。

推理完全使用此前已生成chunks。GT历史只在oracle_history诊断出现，单独标示，不能作为效果结果。seed42同时评empty/reverse_history/static/reverse/oracle_history；全405三seed正式输出。static/reverse只干预audio local+h0，保留global与基座，为条件依赖检查而非真实音频事件因果证明。

验收：raw/centered MSE与corr/RMS、ES/VS、速度、越界、情感teacher非独立读出；额外报告每16帧边界跳变与chunk内跳变，必须查完整rollout漂移。无GT推理泄漏、冻结hash、随机流匹配与43copy真实断言。固定九片/三视频与旧候选对照，不调gain/时移。12轮预算中失败也完整记录，不宣称论文创新或联合成功。
