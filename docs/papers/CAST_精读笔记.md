# CAST (StyleVector) 精读笔记 v2 — 全文已获取（ACL官方PDF, 14页）

## 实验配置（确认）
- 模型: Llama-2-7B-chat, **8-bit量化, 贪心解码**（对我们极友好：贪心=可复现性强）
- 中性回复生成器: gpt-3.5-turbo（我们无API预算用它→用本地模型替代,是复现差异点）
- 提取: 均值差(最简策略); 层位/α按任务在验证集选; 层位结论"15+(约0.4-0.5深度)"
- 任务(排除email): LongLaMP{AbstractGen, TopicWriting, ReviewGen} + LaMP{NewsHeadline, ScholarlyTitle, TweetParaphrasing}
- 指标: ROUGE-L, METEOR（vs 参考答案=用户真实测试输出）

## 主表完整数字（相对提升, Ours vs Non-personalized）
- AbstractGen: R-L +0.2% / METEOR +0.8%（几乎为零!）
- TopicWriting: +4.7% / +4.0%
- ReviewGen: +5.0% / **+11.8%**
- NewsHeadline: +3.2% / +2.5%
- ScholarlyTitle: **+25.8%** / +10.2%
- TweetParaphrasing: **+12.8% / +17.5%**
- "8%平均"主要由 ScholarlyTitle/TweetPara/ReviewGen 三个任务扛起
- RAG基线(k=2检索)几乎处处不敌向量法, 常常低于非个性化基线

## 杀手实验设计 v3（全信息版）
选 **Tweet Paraphrasing**（第二大效应+输出最短17.74 token+1496用户→笔记本可行）
+ **ScholarlyTitle**（最大效应25.8%）

核心洞察（指标循环性）: ROUGE-L/METEOR 测与参考(用户真实输出)的词面重合；
而风格向量恰是从同一用户的**历史**提取的——历史与参考共享该用户的表层统计
（句长/功能词/标点/习语）。向量把生成推向用户历史分布 ⇒ 参考重合度机械上升。
**指标提升部分是构造使然（by construction），与"风格语义质量"混同。**

四臂设计:
  A. CAST复现（向量, 按其协议: 均值差+贪心+生成位置注入）
  B. 全档案上下文对照（把提取用的全部历史条目放进prompt, RAG只给k=2条——
     分离"机制"与"信息量": 若全档案prompt≈向量, 机制无优势）
  C. 随机用户向量（错配用户→应不提升; 验证向量确实用户特异, 公平性检查）
  D. 表面统计中介分析（不生成, 纯分析）:
     - 风格计量距离: 生成 vs 用户历史的句长/功能词/标点距离（A vs B vs C）
     - METEOR提升的词面分解: 功能词重合增量 vs 实义词重合增量
     预测: A的提升大部分可由表面统计收敛解释

预注册判据:
  - 复现验收: A任务上同向提升(量级可异, 环境/8bit差异预期)
  - 分解结论: 若B≈A 且/或 表面统计中介占比>50% → "指标提升主要为表层收敛"
  - 任一结果可发表: 确认其效应=诚实修正; 部分分解=定量刻画

## 复现差异声明（将写入论文）
- gpt-3.5-turbo中性回复 → 本地开源模型替代（提取负臂的来源变化）
- 4-bit(我们) vs 8-bit(它们) 量化
- 验证集层位/α全扫描 → 采用其公开结论(15+)的代表性层位+小扫描
