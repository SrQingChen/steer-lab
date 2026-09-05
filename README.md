# steer_lab —— 示例语句内容走私：复现与防治

研究问题：从角色示例语句提取的风格转向向量，会把示例的**内容**（实体、事件、话题）
走私进无关话题的生成里（"出戏"）。本项目在本地小模型上复现该现象，并逐步验证防治方案。

对应文献空位：arXiv 2502.18862 记录了单示例转向向量的过拟合/方差现象并呼吁防治方法；
本仓库的目标是给出「提取侧净化 + 生成侧拦截 + 过拟合预测」的系统性答案。

## 路线图

- **Phase 1（当前）**：复现现象 —— 单示例向量方差 + 平均向量内容走私，held-out 话题度量。
- **Phase 2**：对照消元提取（同句自动改写成中性语体做配对差分）。
- **Phase 3**：解码端 n-gram 拦截 + 多样性校准。
- **Phase 4**：过拟合预测器（向量稳定性、内容子空间投影占比）。

## 快速开始

```bash
pip install -r requirements.txt
python scripts/step1_extract.py    # 提取向量 + 单示例余弦矩阵（方差证据）
python scripts/step2_generate.py   # 三条件生成：baseline / mean / oneshot_i
python scripts/step3_report.py     # 度量汇总 + 报告图
```

需要约 8GB 显存（7B 4-bit）。HF 下载自动走 hf-mirror（见 env_setup.py）。

## 结构

```
configs/phase1.py               实验配置（层位、强度、规模）
data/character_old_fisherman.json  测试角色：强风格 + 强内容锚
data/heldout_topics.json        与示例零重叠的 6 个话题
src/model.py                    模型加载 + 残差流 hook（捕获/注入）
src/extract.py                  向量提取（单示例 / 平均）
src/steer.py                    生成期注入
src/metrics.py                  泄漏 / 复读 / 风格 三类度量
scripts/step1~3                 提取 → 生成 → 报告
results/                        向量、生成记录、报告
```

## 度量定义

| 指标 | 含义 | 预期（现象成立时） |
|---|---|---|
| leak | 生成中锚实体（海/船/阿秀/潮水…）出现数 | mean 与 oneshot 显著高于 baseline |
| max_lcs / gram6 | 与示例的逐字重叠 | 复读的逐字证据 |
| tics | 口头禅（晓得伐/阿拉/唉）频率 | 风格迁移是否生效 |
| sent_len | 平均句长 | 向示例短句风格靠拢 = 生效 |

判读：**mean/oneshot 高 tics 且高 leak = 风格与内容一起被搬运**，即走私现象；
oneshot 各向量间指标方差大 = 18862 的过拟合现象。
