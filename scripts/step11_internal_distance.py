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


    print("\n== 修正版零基线：40 个无关完整句子（名词基线混入了文本类型差异）==", flush=True)
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
    reps_nullsent = reps(model, tok, NULL_SENTS)
    for L in LAYERS:
        ds_null = [cos_dist(reps_nullsent[L]["mean"][i], reps_canary[L]["mean"][ci])
                   for i in range(len(NULL_SENTS)) for ci in range(2)]
        ds_obs = [cos_dist(reps_tc[L]["mean"][ti], reps_canary[L]["mean"][ci])
                  for ti in range(len(TOPICS)) for ci in range(2)]
        m, s = st.mean(ds_null), st.pstdev(ds_null)
        z = (st.mean(ds_obs) - m) / s if s > 0 else 0
        print(f"L{L}: 句子级零基线 {m:.4f}±{s:.4f}  话题-金丝雀 {st.mean(ds_obs):.4f} "
              f"(z={z:+.2f})", flush=True)

    print("\n== 尺度修正：去中心化后再算距离（用户假设：窄带是公共分量所致）==", flush=True)
    for L in LAYERS:
        # 用全部背景文本估计公共均值（40名词 + 7话题），减去后重归一化
        bg = torch.cat([reps_null[L]["mean"], reps_tc[L]["mean"]], dim=0)
        center = bg.mean(dim=0)

        def cen(t):
            v = t - center
            return v / v.norm().clamp_min(1e-8)

        tc_c = [cen(v) for v in reps_tc[L]["mean"]]
        ca_c = [cen(v) for v in reps_canary[L]["mean"]]
        nu_c = [cen(v) for v in reps_null[L]["mean"]]

        ds_null = [cos_dist(nu_c[i], ca_c[ci]) for i in range(len(NULL_NOUNS))
                   for ci in range(2)]
        import statistics as st
        m, s = st.mean(ds_null), st.pstdev(ds_null)

        def dist_c(ti, ci):
            return cos_dist(tc_c[ti], ca_c[ci])

        rho_all, p_all, ds = corr_for(dist_c)
        rho_ex, p_ex, _ = corr_for(dist_c, exclude=APPROPRIATE_PAIRS)
        tag = " [主终点]" if L == PRIMARY_LAYER else ""
        print(f"L{L}-centered{tag}: 全配对 rho={rho_all:.3f} (p={p_all:.4f}) | "
              f"净化后 rho={rho_ex:.3f} (p={p_ex:.4f}) | "
              f"带宽 [{min(ds):.3f},{max(ds):.3f}] vs 原始 "
              f"[{min(cos_dist(reps_tc[L]['mean'][t], reps_canary[L]['mean'][c]) for t in range(7) for c in range(2)):.3f},"
              f"{max(cos_dist(reps_tc[L]['mean'][t], reps_canary[L]['mean'][c]) for t in range(7) for c in range(2)):.3f}] | "
              f"零基线 {m:.3f}±{s:.3f}", flush=True)


if __name__ == "__main__":
    main()
