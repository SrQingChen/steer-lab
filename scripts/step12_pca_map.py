"""Step 12：把表征空间画成地图 —— 89 个文本的第23层表征 PCA 投影 + 有效维数。

四组文本：7 个话题(带标签) / 2 个金丝雀(星标) / 40 个无关名词 / 40 个无关句子。
产出：results/pca_map.png + 有效维数(参与率) + 各主成分方差解释率。

读图须知（诚实）：PCA 是影子——影子里的距离不等于真实距离，
但簇结构（谁和谁扎堆）通常存活。这张图主要用来"看见"分层结构。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import env_setup  # noqa: F401
import torch
from configs.phase1 import CONFIG
from src.model import load_model
from scripts.step10_chain_probe import TOPICS, CANARIES
from scripts.step11_internal_distance import NULL_NOUNS

L = 23

NULL_SENTS = [
    "今天天气晴朗，适合出门散步。", "超市里的蔬菜价格最近有所下降。", "这份报告需要在周五之前完成。",
    "公交车每隔十五分钟一班。", "图书馆下午五点闭馆。", "小明每天早上七点起床吃早餐。",
    "公司下季度要开三个新项目。", "小区门口新开了一家早餐店。", "火车晚点了二十分钟。",
    "这份合同需要双方签字盖章。", "明天的会议改到下午三点。", "快递放在了前台代收点。",
    "这条路的限速是每小时六十公里。", "银行的营业时间是九点到五点。", "这台洗衣机是去年买的。",
    "学校的运动会定在下个月举行。", "他每天骑自行车上班。", "菜市场下午人比较少。",
    "这部电影上映两周了。", "办公室的空调坏了。", "地铁末班车是十一点半。",
    "这个软件需要更新到最新版本。", "门口的鞋柜放不下了。", "她喜欢在阳台种花。",
    "晚饭做了三个菜一个汤。", "楼下的超市二十四小时营业。", "打印机缺纸了。",
    "下周要出差去南方。", "小区里新装了充电桩。", "这本书已经借了两周了。",
    "会议室被预订到中午。", "她的钢琴课在每周六上午。", "这个路口经常堵车。",
    "冰箱里没有鸡蛋了。", "车站对面有一家面馆。", "公司年会定在一月十五号。",
    "他的驾照这个月到期。", "阳台的晾衣架坏了。", "这个月的电费比上月高。",
]


@torch.no_grad()
def rep(model, tok, text):
    ids = tok(text, return_tensors="pt").to(model.device)
    hs = model(**ids, output_hidden_states=True).hidden_states
    return hs[L][0].float().mean(dim=0)


def main():
    model, tok = load_model(CONFIG["model_name"], CONFIG["load_4bit"])

    groups = []
    for tid_, q in TOPICS:
        groups.append(("topic", tid_, rep(model, tok, q)))
    for cname, c in CANARIES.items():
        groups.append(("canary", cname, rep(model, tok, c["line"])))
    for i, n in enumerate(NULL_NOUNS):
        groups.append(("noun", str(i), rep(model, tok, n)))
    for i, s in enumerate(NULL_SENTS):
        groups.append(("sentence", str(i), rep(model, tok, s)))

    X = torch.stack([g[2] for g in groups])          # [N, d]
    # 逐维标准化：残差流的离群维度方差极大，不标准化则 PCA 被单轴吞掉（实测 PC1=99.9%）
    Xc = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-6)
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    lam = (S ** 2)
    var_explained = (lam / lam.sum())

    pr = (lam.sum() ** 2 / (lam ** 2).sum()).item()   # 参与率 = 有效维数
    print(f"[有效维数] 参与率(PR) = {pr:.1f}   （名义维度 = {X.shape[1]}）")
    print(f"[方差] PC1={var_explained[0]:.1%}  PC2={var_explained[1]:.1%}  "
          f"PC3={var_explained[2]:.1%}  前10累计={var_explained[:10].sum():.1%}")

    P = Xc @ Vh[:3].T                                  # [N,3]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"topic": "#c44", "canary": "#c90", "noun": "#88a", "sentence": "#8a8"}
    for gname in ["noun", "sentence", "topic", "canary"]:
        idx = [i for i, g in enumerate(groups) if g[0] == gname]
        for ax, (a, b) in zip(axes, [(0, 1), (0, 2)]):
            ax.scatter(P[idx, a], P[idx, b], s=18 if gname in ("noun", "sentence") else 60,
                       c=colors[gname], label=gname, alpha=0.7,
                       marker="*" if gname == "canary" else "o")
    for i, g in enumerate(groups):
        if g[0] == "topic":
            for ax, (a, b) in zip(axes, [(0, 1), (0, 2)]):
                ax.annotate(g[1], (P[i, a].item(), P[i, b].item()),
                            fontsize=8, xytext=(3, 3), textcoords="offset points")
        if g[0] == "canary":
            for ax, (a, b) in zip(axes, [(0, 1), (0, 2)]):
                ax.annotate(g[1], (P[i, a].item(), P[i, b].item()),
                            fontsize=9, fontweight="bold", xytext=(3, 3),
                            textcoords="offset points")
    for ax, (a, b), ttl in zip(axes, [(0, 1), (0, 2)],
                               ["PC1-PC2", "PC1-PC3"]):
        ax.set_xlabel(f"PC{a+1} ({var_explained[a]:.0%})")
        ax.set_ylabel(f"PC{b+1} ({var_explained[b]:.0%})")
        ax.set_title(ttl)
        ax.legend()
    plt.suptitle(f"表征空间地图（L23 均值池化，89 个文本的影子） 有效维数≈{pr:.0f}", fontsize=13)
    plt.tight_layout()
    out = os.path.join(ROOT, "results", "pca_map.png")
    fig.savefig(out, dpi=150)
    print(f"[ok] 地图已保存: {out}")


if __name__ == "__main__":
    main()
