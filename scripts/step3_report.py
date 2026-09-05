"""Step 3：度量汇总 + 出图（现象报告）。"""
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.metrics import score_record

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


def main():
    with open(os.path.join(ROOT, "data", "character_old_fisherman.json"), encoding="utf-8") as f:
        char = json.load(f)
    ex, anchors, tics = char["examples"], char["anchor_entities"], char["style_tics"]

    recs = []
    with open(os.path.join(ROOT, "results", "generations_phase1.jsonl"), encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    for r in recs:
        r["m"] = score_record(r["text"], ex, anchors, tics)

    # ---- 按条件汇总 ----
    by_cond = defaultdict(list)
    for r in recs:
        key = r["condition"] + (f":{r['tag']}" if r["tag"] else "")
        by_cond[key].append(r["m"])

    def avg(ms, k):
        return sum(m[k] for m in ms) / max(1, len(ms))

    ex_sent = sum(len(s) for s in ex) / len(ex)  # 示例平均句长（风格参照）

    lines = ["# Phase 1 现象报告：示例内容走私", "",
             f"- 示例平均句长（风格参照）: {ex_sent:.1f} 字/句", "",
             "| 条件 | 泄漏实体数/条 | 最长公共子串 | 6-gram命中 | 口头禅/条 | 平均句长 | 条数 |",
             "|---|---|---|---|---|---|---|"]
    for key in sorted(by_cond):
        ms = by_cond[key]
        lines.append(f"| {key} | {avg(ms,'leak'):.1f} | {avg(ms,'max_lcs'):.1f} | "
                     f"{avg(ms,'gram6'):.1f} | {avg(ms,'tics'):.1f} | {avg(ms,'sent_len'):.1f} | {len(ms)} |")
    report = "\n".join(lines)
    print(report)

    # ---- 图 1：各条件泄漏与风格 ----
    conds = ["incontext", "baseline", "mean"]
    alpha_keys = sorted(k for k in by_cond if "@" in k)
    ones = [k for k in by_cond if k.startswith("oneshot")]
    keys = conds + alpha_keys + ones
    leak = [avg(by_cond[k], "leak") for k in keys]
    tic = [avg(by_cond[k], "tics") for k in keys]
    colors = (["#c94", "#888", "#c44"] + ["#48a"] * len(alpha_keys) + ["#aaa"] * len(ones))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(len(keys)), leak, color=colors)
    axes[0].set_xticks(range(len(keys)))
    axes[0].set_xticklabels([k.replace("oneshot:", "单句") for k in keys], rotation=60, fontsize=8)
    axes[0].set_ylabel("内容泄漏（锚实体数/条）")
    axes[0].set_title("内容走私：held-out 话题上出现渔事实体")
    axes[1].bar(range(len(keys)), tic, color=colors)
    axes[1].set_xticks(range(len(keys)))
    axes[1].set_xticklabels([k.replace("oneshot:", "单句") for k in keys], rotation=60, fontsize=8)
    axes[1].set_ylabel("口头禅次数/条")
    axes[1].set_title("风格迁移：口头禅出现频率")
    plt.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "phase1_report.png"), dpi=150)

    with open(os.path.join(ROOT, "results", "phase1_report.md"), "w", encoding="utf-8") as f:
        f.write(report + "\n\n![chart](phase1_report.png)\n")
    print(f"\n[ok] 报告已保存: results/phase1_report.md + phase1_report.png")


if __name__ == "__main__":
    main()
