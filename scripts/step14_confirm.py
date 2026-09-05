"""Step 14：C@L18 双终点确认实验（预注册）。

背景：step13 重评发现对照消元提取(C变体)@L18 候选阳性（E_style +0.22,
口语率最高），n=8 存在多重比较风险。本实验用新鲜独立样本确认或击杀。

预注册（跑前固定）：
  条件     C@L18α0.5(n=24) / C@L18α1.0(n=24) / baseline(n=24) 新鲜生成；
           incontext 不新生成，沿用已有 18 条做天花板锚。
  主终点   C@L18α0.5 的 E_style − baseline E_style > +0.15 nat/词元
           （Welch t 检验，p<0.05）
  副终点   C@L18α1.0 同上标准
  走私终点 C 条件的 E_content 与 baseline 差 |Δ|<0.10 且锚实体泄漏率≈0
  判读     两类终点都过 = 风格转移+零走私；仅主终点过 = 风格在但走私存疑；
           主终点不过 = 线索击杀（多重比较结案）
提取协议与 step7 完全一致（末token、复述模板、示例−同内容中性改写），
改写句复用 results/paraphrases.json。
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import env_setup  # noqa: F401
import torch
from configs.phase1 import CONFIG
from src.model import load_model, HookManager, chat_template
from src.steer import generate as steer_generate
from scripts.step13_e_score import (STYLE_PERSONA, CONTENT_PERSONA, TOPIC_Q,
                                    mean_lp, stylometry)

L = 18
CONDS = {"C-a0.5": 0.5, "C-a1.0": 1.0}
N_PER = 24  # 6 话题 x 4 采样
SEED_BASE = 77000


@torch.no_grad()
def last_acts(model, tok, hook, texts):
    hook.clear_captures()
    hook.mode = "capture"
    for t in texts:
        msgs = [{"role": "user", "content": "请一字不差地复述下面这句话。"},
                {"role": "assistant", "content": t}]
        prompt = chat_template(tok, msgs, add_generation_prompt=False)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        model(**ids)
    hook.mode = "idle"
    acts = torch.cat(hook.captures[L], dim=0)
    norm = sum(hook.norm_stats[L]) / len(hook.norm_stats[L])
    hook.clear_captures()
    return acts, norm


def main():
    with open(os.path.join(ROOT, "data", "character_old_fisherman.json"), encoding="utf-8") as f:
        char = json.load(f)
    with open(os.path.join(ROOT, "data", "heldout_topics.json"), encoding="utf-8") as f:
        topics = json.load(f)["topics"]
    paras = json.load(open(os.path.join(ROOT, "results", "paraphrases.json"), encoding="utf-8"))

    model, tok = load_model(CONFIG["model_name"], CONFIG["load_4bit"])
    hook = HookManager(model)
    hook.install([L])

    # C 向量提取（与 step7 相同协议）
    acts_ex, norm = last_acts(model, tok, hook, char["examples"])
    acts_pa, _ = last_acts(model, tok, hook, paras)
    raw = acts_ex - acts_pa
    v = raw.mean(dim=0)
    v = v / v.norm().clamp_min(1e-8)
    print(f"[extract] C@L{L} 完成, hidden_norm={norm:.1f}", flush=True)

    # 生成（新鲜种子）
    out_path = os.path.join(ROOT, "results", "confirm_C_L18.jsonl")
    t0 = time.time()
    rows = []
    with open(out_path, "w", encoding="utf-8") as f:
        for cond, alpha in list(CONDS.items()) + [("baseline", 0)]:
            vecs = {L: v} if cond != "baseline" else {}
            hn = {L: norm}
            for tp in topics:
                for s in range(4):
                    seed = SEED_BASE + hash((cond, tp["id"], s)) % 10000
                    text = steer_generate(model, tok, hook, tp["question"], vecs, hn,
                                           steer_scale=alpha, max_new_tokens=120,
                                           temperature=CONFIG["temperature"],
                                           top_p=CONFIG["top_p"], seed=seed)
                    r = {"cond": cond, "topic": tp["id"], "sample": s, "text": text}
                    rows.append(r)
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
            print(f"[{time.time()-t0:6.1f}s] {cond} 完成", flush=True)
    hook.remove()

    # 评分（同模型直接评）
    for r in rows:
        q = TOPIC_Q[r["topic"]]
        lp_base, _ = mean_lp(model, tok, None, q, r["text"])
        lp_style, _ = mean_lp(model, tok, STYLE_PERSONA, q, r["text"])
        lp_cont, _ = mean_lp(model, tok, CONTENT_PERSONA, q, r["text"])
        leak = sum(r["text"].count(a) for a in char["anchor_entities"])
        r["E_style"] = lp_style - lp_base
        r["E_content"] = lp_cont - lp_base
        r["leak"] = leak
        r.update({f"sty_{k}": val for k, val in stylometry(r["text"]).items()})

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # incontext 锚（沿用已有评分）
    ic = []
    for line in open(os.path.join(ROOT, "results", "e_scores.jsonl"), encoding="utf-8"):
        d = json.loads(line)
        if d["src"] == "generations_phase1" and d["condition"] == "incontext":
            ic.append(d)

    # ---- 统计 ----
    from scipy.stats import ttest_ind
    import statistics as st
    print("\n== 双终点确认（每条件 n=24, incontext 锚 n=18）==", flush=True)
    base = [r for r in rows if r["cond"] == "baseline"]
    print(f"{'条件':10s} {'E_style±sd':>16s} {'E_content±sd':>16s} {'泄漏/条':>7s} {'句长':>5s} {'口语/100':>7s}")

    def summarize(name, rs):
        es = [r["E_style"] for r in rs]
        ec = [r["E_content"] for r in rs]
        print(f"{name:10s} {st.mean(es):>+8.4f}±{st.pstdev(es):.4f} "
              f"{st.mean(ec):>+8.4f}±{st.pstdev(ec):.4f} "
              f"{sum(r['leak'] for r in rs)/len(rs):>7.2f} "
              f"{st.mean([r['sty_sent_len'] for r in rs]):>5.1f} "
              f"{st.mean([r['sty_colloq'] for r in rs]):>7.2f}", flush=True)
        return es

    es_base = summarize("baseline", base)
    es_ic = [x["E_style"] for x in ic]
    print(f"{'incontext':10s} {st.mean(es_ic):>+8.4f}±{st.pstdev(es_ic):.4f}"
          f"   (沿用已有评分)", flush=True)
    for cond in CONDS:
        rs = [r for r in rows if r["cond"] == cond]
        es = summarize(cond, rs)
        t, p = ttest_ind(es, es_base, equal_var=False)
        d_diff = st.mean(es) - st.mean(es_base)
        verdict = "阳性" if (d_diff > 0.15 and p < 0.05) else "未过"
        print(f"  -> {cond}: ΔE_style={d_diff:+.4f}, Welch p={p:.4f} => 主终点[{verdict}]",
              flush=True)
    for cond in CONDS:
        rs = [r for r in rows if r["cond"] == cond]
        d_c = st.mean([r["E_content"] for r in rs]) - st.mean([r["E_content"] for r in base])
        lk = sum(r["leak"] for r in rs) / len(rs)
        verdict = "通过" if (abs(d_c) < 0.10 and lk < 0.5) else "存疑"
        print(f"  -> {cond} 走私终点: ΔE_content={d_c:+.4f}, 泄漏/条={lk:.2f} => [{verdict}]",
              flush=True)


if __name__ == "__main__":
    main()
