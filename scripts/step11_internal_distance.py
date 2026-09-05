"""Step 11：内部距离 vs 代理距离的预测力对照（预注册配置）。

预注册声明（跑之前固定，防挑拣）：
  主终点   = 第 23 层（36层模型的 2/3 深度）均值池化表征的
             Spearman(距离, 盆地进入率)
  全报告   = 第 18/23/29 层 + 末token变体 + bge 代理，全部如实打印
  敏感性   = 剔除 2 个"合理提及"配对（animal→parrot, sleep2→parrot）
             后重算主终点，仅作敏感性分析
  零基线   = 40 个无关日常名词到金丝雀的距离分布（回答"近距离
             配对是否密密麻麻"的高维拥挤问题）
表征协议：裸文本（不加对话模板），output_hidden_states 取各层，
          对序列维均值池化；末token变体取最后一个位置。
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import env_setup  # noqa: F401
import torch
from configs.phase1 import CONFIG
from src.model import load_model
from scripts.step10_chain_probe import TOPICS, CANARIES

LAYERS = [18, 23, 29]          # 0.5 / 0.65 / 0.8 深度（36层模型）
PRIMARY_LAYER = 23

NULL_NOUNS = [
    "铅笔", "发票", "高压锅", "地铁站", "螺丝刀", "天气预报", "公交卡", "牙膏",
    "台历", "雨伞", "拖鞋", "快递", "菜谱", "空调遥控器", "钥匙", "毛巾",
    "书包", "红绿灯", "洗衣液", "筷子", "地毯", "闹钟", "剪刀", "插座",
    "水壶", "眼镜", "床单", "灯泡", "脸盆", "口罩", "拖把", "衣架",
    "酱油", "冰箱", "鞋柜", "枕头", "风扇", "蚊帐", "门铃", "公交站牌",
]

APPROPRIATE_PAIRS = {("animal", "parrot"), ("sleep2", "parrot")}


@torch.no_grad()
def reps(model, tok, texts):
    """返回 {layer: {"mean": Tensor(N,d), "last": Tensor(N,d)}}"""
    out = {L: {"mean": [], "last": []} for L in LAYERS}
    for t in texts:
        ids = tok(t, return_tensors="pt").to(model.device)
        hs = model(**ids, output_hidden_states=True).hidden_states
        for L in LAYERS:
            h = hs[L][0].float()
            out[L]["mean"].append(h.mean(dim=0))
            out[L]["last"].append(h[-1])
    for L in LAYERS:
        out[L]["mean"] = torch.stack(out[L]["mean"])
        out[L]["last"] = torch.stack(out[L]["last"])
    return out


def cos_dist(a, b):
    return 1 - (torch.dot(a, b) / (a.norm() * b.norm())).clamp(-1, 1).item()


def main():
    # 结果变量：从 chain_probe.jsonl 读（不转录）
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "results", "chain_probe.jsonl"),
                                        encoding="utf-8")]
    entry = {}
    for cname in CANARIES:
        for tid_, _ in TOPICS:
            rs = [r for r in rows if r["canary"] == cname and r["topic"] == tid_]
            entry[(tid_, cname)] = sum(r["entered"] for r in rs) / len(rs)

    model, tok = load_model(CONFIG["model_name"], CONFIG["load_4bit"])

    topic_qs = [q for _, q in TOPICS]
    canary_lines = [c["line"] for c in CANARIES.values()]
    cnames = list(CANARIES.keys())

    reps_tc = reps(model, tok, topic_qs)                 # 7 个话题
    reps_canary = reps(model, tok, canary_lines)         # 2 个金丝雀
    reps_null = reps(model, tok, NULL_NOUNS)             # 40 个零基线名词

    from scipy.stats import spearmanr

    # bge 代理（复用 step10 的模型与协议）
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    E_t = emb.encode(topic_qs, normalize_embeddings=True)
    E_c = emb.encode(canary_lines, normalize_embeddings=True)

    def corr_for(dist_fn, exclude=()):
        ds, rs_ = [], []
        for ti, (tid_, _) in enumerate(TOPICS):
            for ci, cname in enumerate(cnames):
                if (tid_, cname) in exclude:
                    continue
                ds.append(dist_fn(ti, ci))
                rs_.append(entry[(tid_, cname)])
        rho, p = spearmanr(ds, rs_)
        return rho, p, ds

    print("== 主终点与全层报告 ==", flush=True)
    print(f"{'表征':16s} {'Spearman':>9s} {'p':>8s} {'距离动态范围':>12s}")
    results = {}
    for L in LAYERS:
        for mode in ["mean", "last"]:
            f = lambda ti, ci, L=L, mode=mode: cos_dist(
                reps_tc[L][mode][ti], reps_canary[L][mode][ci])
            rho, p, ds = corr_for(f)
            tag = f"L{L}-{mode}" + ("  [主终点]" if (L == PRIMARY_LAYER and mode == "mean") else "")
            results[(L, mode)] = (rho, p)
            print(f"L{L}-{mode:4s}{tag.replace(f'L{L}-{mode}', ''):8s} "
                  f"{rho:>9.3f} {p:>8.4f} [{min(ds):.3f}, {max(ds):.3f}]", flush=True)
    f_bge = lambda ti, ci: float(1 - (E_t[ti] @ E_c[ci]))
    rho, p, ds = corr_for(f_bge)
    print(f"{'bge代理':12s} {rho:>9.3f} {p:>8.4f} [{min(ds):.3f}, {max(ds):.3f}]", flush=True)

    print("\n== 敏感性分析：剔除合理提及配对 ==", flush=True)
    for L in LAYERS:
        f = lambda ti, ci, L=L: cos_dist(reps_tc[L]["mean"][ti], reps_canary[L]["mean"][ci])
        rho, p, _ = corr_for(f, exclude=APPROPRIATE_PAIRS)
        print(f"L{L}-mean: rho={rho:.3f} p={p:.4f}", flush=True)
    rho, p, _ = corr_for(f_bge, exclude=APPROPRIATE_PAIRS)
    print(f"bge    : rho={rho:.3f} p={p:.4f}", flush=True)

    print("\n== 零基线：高维拥挤度（回答'近距离是否密密麻麻'）==", flush=True)
    for L in LAYERS:
        ds_null = [cos_dist(reps_null[L]["mean"][i], reps_canary[L]["mean"][ci])
                   for i in range(len(NULL_NOUNS)) for ci in range(2)]
        ds_obs = [cos_dist(reps_tc[L]["mean"][ti], reps_canary[L]["mean"][ci])
                  for ti in range(len(TOPICS)) for ci in range(2)]
        import statistics as st
        m, s = st.mean(ds_null), st.pstdev(ds_null)
        obs_m = st.mean(ds_obs)
        z = (obs_m - m) / s if s > 0 else 0
        print(f"L{L}: 零基线距离 {m:.3f}±{s:.3f}  话题-金丝雀距离均值 {obs_m:.3f} "
              f"(z={z:+.2f})  零基线中 <0.6 的比例 "
              f"{sum(1 for d in ds_null if d < 0.6)/len(ds_null):.0%}", flush=True)


if __name__ == "__main__":
    main()
