# steer_lab

**示例衍生转向向量的失效边界与突破口** —— 一个在消费级笔记本（RTX 4060 / 8GB）上完成的激活转向研究项目。

English TL;DR: We show that naively-extracted steering vectors from character example sentences **fail** to transfer idiolect style (20 configs, positive control validated), while **same-content contrastive extraction** at the right layer **works** (replicated on 2 models × 2 characters, reaching ~72% of the in-context ceiling with zero content smuggling), alongside a characterization of context pollution as **basin-entry dynamics** with a real-time probability sentinel.

## 核心发现（四条，全部可复现）

| # | 发现 | 关键数字 | 脚本 |
|---|---|---|---|
| 1 | **朴素提取失效 + 消元提取突破**：朴素转向向量跨 20 配置无法转移习语风格；同内容配对消元提取在正确层位可达 in-context 天花板的 72%，零实体走私 | Δ=+1.17 (p=1.5e-12)；格式对照后习语增量 +0.34 (p=0.005) | step6/7/14/16/17 |
| 2 | **阳性对照证明失效非管线问题**：正式度维度同管线双向可移 | 书面标记 1.88→0~0.5 | step8 |
| 3 | **上下文污染 = 盆地进入动力学**：泄漏是双峰事件（中位潜伏 8e-5 vs 盆地内>0.6）；温度只是边缘调制器；哨兵可提前预警（显影事件=潜伏峰值前三） | step9/10 |
| 4 | **距离定律**：话题-金丝雀的模型内部距离预测错境泄漏率（净化后 Spearman −0.73）；高维表征空间高度拥挤（句-句距离 0.02–0.12），信号在带内排名 | step11 |

配套方法学：E 似然比评分器（双人设+第三控制）、金丝雀协议、回弹协议、多家族裁判（DS-V4-Pro + GLM 子代理 + 人类盲评 86% 一致）。

## 快速开始

```bash
pip install -r requirements.txt
# 国内自动走 hf-mirror（见 env_setup.py），首次运行自动下载模型
python scripts/step1_extract.py     # 向量提取 + 余弦矩阵（10 分钟内）
python scripts/step2_generate.py    # 四条件生成（可断点续跑）
python scripts/step3_report.py      # 报告
```

核心确认实验：`step14_confirm.py`（预注册双终点）；复现：`step17_replication.py`。
8GB 显存即可（Qwen2.5-3B / Qwen3-1.7B FP16）。实验数据全部在 `results/`。

## 目录

```
configs/       实验配置
data/          角色卡（含游戏台词提取的示例，仅作风格研究，版权归鹰角网络）
scripts/       step1-18 按实验顺序编号
src/           模型hook/提取/注入/度量 库
results/       全部实验数据（jsonl + 向量 + 报告）
docs/          机理手册 / 想法积压 / 论文骨架 / 人类盲评对账
```

## 科研过程记录

全程预注册（判据写于运行前）、盲测、阴性结果如实报告、三次仪器故障被自检当场捕获
（cos 归一化 bug / DS 推理模型空判 / 零基线类型混杂），完整历史见 git log。
人类盲评者发现的关键混杂（格式 vs 风格）触发了 step16 判别实验并改写了主张。

## 引用的游戏素材

乌啾（Ukusik）为《明日方舟》角色，本文引用其 12 句游戏台词（约 300 字）仅作
风格分析研究用途，全部版权归上海鹰角网络所有。

## License

MIT（见 LICENSE）。实验数据与代码可自由用于研究，引用请注明本仓库。
