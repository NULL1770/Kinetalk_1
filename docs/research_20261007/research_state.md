# 研究恢复入口：2026-10-07

当前任务：全面调研2023–2026音频驱动3D面部动态、内容/情感/身份（几何与说话风格分开）、teacher/student、训练解耦和评价，结合本项目Phase34–39证据给出新的最优可验证方案。Phase40旧提案暂停，不改模型/启动训练；新设计须用户确认。

**本轮已完成（2026-10-07）：首先读LITERATURE_SYNTHESIS.md和ARCHITECTURE_PROPOSAL.md。** 21新PDF已逐个SHA核验，另复用MEDTalk/EmoTalk；30条claim ledger定位原文；阅读范围P/S/M见综述。新提案为内容条件表达响应，q/p联合建立表达空间、772D学生不接内容、跨参考query训练style、共享decoder；并加入h0捷径与两neutral参考不可辨识多情感风格的检查。所有新架构只在文档中，尚未实施。下一步是用户审阅/讨论，不运行旧Phase40或重新调研相同资料。

文件：evidence/source_manifest.json、claim_ledger.json；papers/包含来源/原SHA，fulltext/可按[PDF PAGE n]查找。范围21篇不是全部逐页读完，见各论文级别。ProsodyTalker下载长期未完成，已只终止fetch_export.py进程98388，其他成功逐论文manifest完整；不重启整个批处理。错误PESTalk DOI排除，正确PDF DOI3755190独立查询SSL失败；ECHO PDF会议信息未独立核实，按预印本引用。

具体问题：1）哪些动态确实由音频决定，哪些属于一对多？2）论文是否比较GT和生成的逐帧动态，或只评估范围/分布/人感？3）身份实际是one-hot训练人、参考风格还是几何身份，是否跨人泛化？4）解耦有没有可检查目标和干预证据？5）teacher/student如何避免蒸馏不可预测/含内容的motion latent？6）适合当前52D/MEAD/native-clock的简洁架构、训练阶段和公平评价协议是什么？

已核关键发现：DEEPTalk主文p3–7的DEE是GPO池化后跨模态概率全局embedding，不等于逐帧动态GT；其明确不以LVE衡量多样情感生成，主要FFD/Emo-FID/SyncNet/多样性/人评。Mimic主文p3–5采用motion内容/风格encoder、GRL身份抑制、音频内容对比及latent cycle；不是一项cycle就保证因果解耦。DiffPoseTalk p4–5用两临近窗口对比学短期风格，并交叉参考条件重建；短期风格与长期身份需区分。PESTalk p4–6以voiceprint×emotion库检索风格；p6 Eq11的符号与文本拉近content/拉远emotion目标表面矛盾，不可直接照抄。EditEmoTalk p5–6主动不报分类accuracy，但ΔCH也不能证明逐帧情感动态。ECHO 2026已有deterministic anchor+stochastic residual，不能把这个组合本身作为原创。

传输记录：requests主arxiv PDF/CVF目录、curl批量部分SSL失败；export.arxiv.org成功获得多篇原文（含EMOTE/Mimic/UniTalker/Media2Face/SAiD/Wav2Sem）；DEEPTalk主URL单curl成功。原失败manifest均保留。wildwest大学仓储PDF已取得。仅网页/下载辅助脚本改动，模型代码/训练未变。

范围：rig/3D mesh参数动画为主；3DGS/2D像素方法仅补充身份/风格概念，不把它们的指标当本项目直接对比。优先2025/2026，其次2023/2024关键基线。全文Methods/Experiments支持的结论与摘要/元数据分开；不凭标题或搜索摘要推断方法。

进度：OpenAlex已完成16组2023–2026检索（每组至多15项，原响应在searches/，汇总harvest_summary.json）。找到2025 ExpTalk/Wav2Sem/ProsodyTalker/DEEPTalk及评价benchmark，2026 EditEmoTalk/PESTalk/ECHO/FMReward/Shape of Speech等候选；尚待官方原文确认，不能由元数据推断方法。CVF2025/2026目录请求SSL EOF失败，改为候选论文官方单页/原文获取。已有MEDTalk v4与EmoTalk ICCV2023原文可复用，DESTalker全文尚未获取。研究材料存本目录，最终综述与架构提案另写文档。研究角色以内联执行，不创建子agent。
