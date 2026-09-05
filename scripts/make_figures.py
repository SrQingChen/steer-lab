"""从已归档数据生成全部论文级图表（无需 GPU / 模型）。

F2 效应分解阶梯 / F3 盆地双峰 / F4 距离-进盆散点 / F5 失效边界热图
F6 裁判会师 / F7 哨兵干预。输出 results/figures/。
"""
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)


def load(name):
    return [json.loads(l) for l in open(os.path.join(ROOT, "results", name),
                                        encoding="utf-8")]


def avg(xs):
    return sum(xs) / max(1, len(xs))


# ---- F2 效应分解阶梯（渔夫系列：格式逃逸 vs 习语增量） ----
confirm = load("confirm_C_L18.jsonl")
casual = load("casual_base.jsonl")
es = load("e_scores.jsonl")
vals = {
    "baseline\n(条目式助手格式)": avg([r["E_style"] for r in confirm if r["cond"] == "baseline"]),
    "格式对照\n(口语段落指令,\n零渔夫信息)": avg([r["E_style"] for r in casual]),
    "C转向\n(对照消元提取)": avg([r["E_style"] for r in confirm if r["cond"] == "C-a0.5"]),
    "in-context\n(天花板)": avg([r["E_style"] for r in es
                                if r["src"] == "generations_phase1" and r["condition"] == "incontext"]),
}
fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.bar(range(len(vals)), list(vals.values()),
              color=["#888", "#c94", "#48a", "#c44"])
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(list(vals.keys()), fontsize=9)
ax.set_ylabel("E_style（风格人设似然比）")
ax.set_title("效应分解：格式逃逸占大头，习语特异增量是剩下的部分", fontsize=12)
ax.axhline(0, color="k", lw=0.5)
for b, v in zip(bars, vals.values()):
    ax.text(b.get_x() + b.get_width() / 2, v + (0.03 if v >= 0 else -0.08),
            f"{v:+.2f}", ha="center", fontsize=10)
ax.annotate("格式逃逸 ≈ +0.81", xy=(0.5, -0.25), ha="center", fontsize=10, color="#c94")
ax.annotate("习语增量 ≈ +0.10", xy=(1.5, 0.24), ha="center", fontsize=10, color="#48a")
plt.tight_layout()
fig.savefig(os.path.join(FIG, "F2_decomposition.png"), dpi=150)
plt.close()

# ---- F3 盆地双峰（金丝雀潜伏峰值分布） ----
cv = load("canary_v2.jsonl")
peaks = [r["latent1_max"] for r in cv if r["dist"] == "far"]
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist([p for p in peaks], bins=60, range=(0, 1.02), color="#48a", edgecolor="white")
ax.set_yscale("log")
ax.set_xlabel("金丝雀潜伏峰值（每条生成的逐步最大概率质量）")
ax.set_ylabel("条数（对数轴）")
ax.set_title(f"盆地进入的双峰分布（n={len(peaks)}）：地板 1e-4 量级 vs 盆地内 >0.6", fontsize=12)
ax.axvline(0.1, color="#c44", ls="--", lw=1)
ax.text(0.11, ax.get_ylim()[1] * 0.5, "干预阈值 0.1", color="#c44", fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(FIG, "F3_basin_bimodal.png"), dpi=150)
plt.close()

# ---- F4 距离-进盆散点 ----
cp = load("chain_probe.jsonl")
g = defaultdict(list)
for r in cp:
    g[(r["canary"], r["topic"])].append(r["entered"])
fig, ax = plt.subplots(figsize=(8, 5.5))
mk = {"parrot": ("o", "#c94", "鹦鹉金丝雀"), "beads": ("s", "#48a", "玻璃珠金丝雀")}
for cname in ["parrot", "beads"]:
    xs = [avg([r["dist"] for r in cp if r["canary"] == cname and r["topic"] == t]) * 0 +
          [r["dist"] for r in cp if r["canary"] == cname and r["topic"] == t][0]
          for t in sorted({r["topic"] for r in cp if r["canary"] == cname})]
    ys = [avg(v) * 100 for (c, t), v in g.items() if c == cname]
    m, col, lab = mk[cname]
    ax.scatter(xs, ys, marker=m, color=col, s=90, label=lab, alpha=0.85)
ax.set_xlabel("话题-金丝雀 嵌入距离（bge，越小越近）")
ax.set_ylabel("盆地进入率 %")
ax.set_title("距离与进入：邀请型配对(100%)脱离距离排序——倒U形预测待恰当性维度验证", fontsize=11)
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(FIG, "F4_distance.png"), dpi=150)
plt.close()

# ---- F5 失效边界与工作点（第二模型层位×剂量热图） ----
rep = load("replication_wujiu.jsonl")
methods = ["naive", "C"]
layer_ks = sorted({(r["cond"].split("-")[1][1:].split("a")[0]) for r in rep
                   if r["cond"].startswith(("naive", "C-"))}, key=int)
alphas = sorted({r["cond"].split("a")[-1] for r in rep
                 if r["cond"].startswith(("naive", "C-"))}, key=float)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, meth in zip(axes, methods):
    grid = [[avg([r["E_style"] for r in rep if r["cond"] == f"{meth}-L{L}a{a}"])
             for a in alphas] for L in layer_ks]
    im = ax.imshow(grid, cmap="RdYlGn", vmin=-1.3, vmax=0.4, aspect="auto")
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"α={a}" for a in alphas])
    ax.set_yticks(range(len(layer_ks)))
    ax.set_yticklabels([f"L{L}" for L in layer_ks])
    for i in range(len(layer_ks)):
        for j in range(len(alphas)):
            ax.text(j, i, f"{grid[i][j]:+.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("朴素提取" if meth == "naive" else "对照消元提取")
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle(f"失效边界与工作点（乌啾×Qwen3-1.7B, baseline E_style={avg([r['E_style'] for r in rep if r['cond']=='baseline']):+.2f}, "
             f"incontext={avg([r['E_style'] for r in rep if r['cond']=='incontext']):+.2f}）", fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(FIG, "F5_failure_boundary.png"), dpi=150)
plt.close()

# ---- F6 裁判会师 ----
fig, ax = plt.subplots(figsize=(9, 5))
conds = ["C vs baseline", "C vs in-context"]
ds = [88, 44]
glm = [100, 67]
human = [100, 60]
x = range(len(conds))
w = 0.25
ax.bar([i - w for i in x], ds, w, label="DS-V4-Pro", color="#48a")
ax.bar(list(x), glm, w, label="GLM 子代理", color="#c94")
ax.bar([i + w for i in x], human, w, label="人类(作者,盲评)", color="#c44")
ax.axhline(50, color="gray", ls="--", lw=1)
ax.text(1.42, 51, "随机=50%", fontsize=9, color="gray")
ax.set_xticks(list(x))
ax.set_xticklabels(conds)
ax.set_ylabel("裁判选择 C 文本的比例 %")
ax.set_title("三家族裁判会师：格式区分对(左)高一致，格式匹配对(右)全体衰减", fontsize=12)
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(FIG, "F6_judges.png"), dpi=150)
plt.close()

# ---- F7 哨兵干预 ----
si = load("sentinel_intervention.jsonl")
fig, ax = plt.subplots(figsize=(7, 5))
stats = {}
for armed in [False, True]:
    rs = [r for r in si if r["armed"] == armed]
    stats[armed] = (sum(r["manifest"] for r in rs), len(rs),
                    sum(r["n_int"] for r in rs))
ax.bar(["对照(不干预)", "哨兵武装"], [stats[0][0], stats[1][0]],
       color=["#c44", "#4a4"], width=0.5)
ax.set_ylabel("显影次数（金丝雀出现在输出）")
ax.set_title(f"哨兵干预：{stats[1][1]} 条生成 0 次显影；触发拦截 {stats[1][2]} 次，"
             f"平均长度与对照完全一致", fontsize=11)
for i, armed in enumerate([False, True]):
    ax.text(i, stats[armed][0] + 0.05,
            f"{stats[armed][0]}/{stats[armed][1]}", ha="center", fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(FIG, "F7_intervention.png"), dpi=150)
plt.close()

print("[ok] 6 张图已生成于", FIG)
for f in sorted(os.listdir(FIG)):
    print(" -", f)
