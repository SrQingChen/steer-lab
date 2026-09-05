"""Step 17：拼图三 —— 乌啾(真实台词) x Qwen3-1.7B 跨模型复现管线。

预注册：
  A 天花板预检：incontext vs baseline 各 18 条，判据 E_style 差 > 0.25 才继续
  B 对照消元(C)与朴素(v1)提取，层位 0.4/0.5/0.6/0.7（1.7B 共28层）
  C 扫描 {C, naive} x 4层 x α{0.5,1.0} x 4话题 x 2采样 = 128 条
  D E 评分三人设：STYLE(纯风格) / CONTENT(罗德岛世界观) / CASUAL(口语段落对照)
  E 判决：复现 = C 最优配置显著超 baseline 且超 CASUAL 对照（习语特异增量）
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
from src.model import load_model, HookManager, chat_template, strip_think, resolve_layers
from src.steer import generate as steer_generate

MODEL = "Qwen/Qwen3-1.7B"
STYLE_P = ("你是一个古灵精怪、牙尖嘴利的少女，说话直来直往，短句为主，"
           "爱用反问句，嘴硬心软，偶尔冒出让人哭笑不得的话。")
CONTENT_P = ("你是罗德岛的年轻医疗干员，来自乌萨斯，小时候在圣骏堡的街巷长大，"
             "现在跟着凯尔希学习认字。")
CASUAL_P = ("你说话随和、口语化，像普通人聊天。用一段连贯的话回答，"
            "不要列条目，不要用标题和加粗。")
CEIL_THRESHOLD = 0.25
LAYER_FRACS = [0.4, 0.5, 0.6, 0.7]
ALPHAS = [0.5, 1.0]


@torch.no_grad()
def gen(model, tok, system, q, seed, max_new=110):
    torch.manual_seed(seed)
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": q}]
    prompt = chat_template(tok, msgs, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=True,
                         temperature=0.8, top_p=0.9,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return strip_think(tok.decode(out[0][ids["input_ids"].shape[1]:],
                                  skip_special_tokens=True)).strip()


@torch.no_grad()
def last_acts(model, tok, hook, texts, L):
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


@torch.no_grad()
def mean_lp(model, tok, persona, question, text):
    msgs = ([{"role": "system", "content": persona}] if persona else []) + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": text},
    ]
    prompt = chat_template(tok, msgs, add_generation_prompt=False)
    head_msgs = ([{"role": "system", "content": persona}] if persona else []) + [
        {"role": "user", "content": question}]
    head = chat_template(tok, head_msgs, add_generation_prompt=True)
    ids_full = tok(prompt, return_tensors=None)["input_ids"]
    ids_head = tok(head, return_tensors=None)["input_ids"]
    p = len(ids_head)
    if ids_full[:p] != ids_head:
        p = 0
        for a, b in zip(ids_full, ids_head):
            if a != b:
                break
            p += 1
    ids = torch.tensor([ids_full], device=model.device)
    logits = model(ids).logits[0]
    lp = torch.log_softmax(logits.float(), dim=-1)
    tgt = ids[0, p:]
    return lp[p - 1:ids.shape[1] - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean().item()


def main():
    with open(os.path.join(ROOT, "data", "character_wujiu.json"), encoding="utf-8") as f:
        char = json.load(f)
    examples, neutrals = char["examples"], char["neutral_baselines"]
    topics = char["topics"]
    tics, anchors = char["style_tics"], char["anchor_entities_hard"] + char["anchor_entities_soft"]
    QMAP = {t["id"]: t["question"] for t in topics}

    model, tok = load_model(MODEL, False)
    n_layers = model.config.num_hidden_layers
    layers = [min(n_layers - 1, int(round(f * n_layers))) for f in LAYER_FRACS]
    print(f"[info] {MODEL}: {n_layers} 层，扫描 {layers}", flush=True)

    def escore(text, q):
        lp0 = mean_lp(model, tok, None, q, text)
        return {"E_style": mean_lp(model, tok, STYLE_P, q, text) - lp0,
                "E_content": mean_lp(model, tok, CONTENT_P, q, text) - lp0,
                "E_casual": mean_lp(model, tok, CASUAL_P, q, text) - lp0}

    out_path = os.path.join(ROOT, "results", "replication_wujiu.jsonl")
    rows = []
    t0 = time.time()

    def emit(r):
        rows.append(r)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{time.time()-t0:6.1f}s] {r['cond']:14s} {r['topic']:8s}#{r['sample']} "
              f"Es={r['E_style']:+.3f} Ec={r['E_content']:+.3f} "
              f"tics={sum(r['text'].count(t) for t in tics)} "
              f"leak={sum(r['text'].count(a) for a in anchors)}", flush=True)

    # ---- A 天花板预检 ----
    import statistics as st
    ic_sys = ("你要扮演一个角色进行对话。以下是该角色的说话示例，"
              "请模仿它的语气和风格，但不要重复示例中的具体内容。\n" +
              "\n".join(f"- {s}" for s in examples))
    ce = {}
    for cond, sysp in [("baseline", None), ("incontext", ic_sys)]:
        vals = []
        for tp in topics:
            for s in range(3):
                text = gen(model, tok, sysp, tp["question"],
                           seed=50000 + hash((cond, tp["id"], s)) % 10000)
                e = escore(text, tp["question"])
                emit({"cond": cond, "topic": tp["id"], "sample": s,
                      "text": text, **e})
                vals.append(e["E_style"])
        ce[cond] = st.mean(vals)
    delta = ce["incontext"] - ce["baseline"]
    print(f"\n[天花板] E_style: baseline={ce['baseline']:+.4f} "
          f"incontext={ce['incontext']:+.4f} Δ={delta:+.4f} "
          f"(阈值 {CEIL_THRESHOLD})", flush=True)
    if delta <= CEIL_THRESHOLD:
        print("[终止] 天花板不成立，复现实验不具条件，换模型重试。", flush=True)
        return
    # casual 对照天花板（习语特异增量的参照）
    vals = []
    for tp in topics[:4]:
        for s in range(3):
            text = gen(model, tok, CASUAL_P, tp["question"],
                       seed=52000 + hash((tp["id"], s)) % 10000)
            e = escore(text, tp["question"])
            emit({"cond": "casual-ctl", "topic": tp["id"], "sample": s,
                  "text": text, **e})
            vals.append(e["E_style"])
    ce["casual"] = st.mean(vals)
    print(f"[对照] casual-ctl E_style={ce['casual']:+.4f}", flush=True)

    # ---- B 提取 ----
    para_path = os.path.join(ROOT, "results", "paraphrases_wujiu.json")
    if os.path.exists(para_path):
        paras = json.load(open(para_path, encoding="utf-8"))
    else:
        paras = [gen(model, tok, None,
                     "把下面这句话改写成普通、平直的书面表达，保持内容不变，"
                     "去掉一切口语色彩和角色腔调：\n" + s,
                     seed=31, max_new=100) for s in examples]
        json.dump(paras, open(para_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

    hook = HookManager(model)
    hook.install(layers)
    vecs, norms = {}, {}
    for L in layers:
        a_ex, norm = last_acts(model, tok, hook, examples, L)
        a_pa, _ = last_acts(model, tok, hook, paras, L)
        a_nu, _ = last_acts(model, tok, hook, neutrals, L)
        vC = (a_ex - a_pa).mean(dim=0)
        vN = (a_ex - a_nu.mean(dim=0, keepdim=True)).mean(dim=0)
        vecs[(L, "C")] = vC / vC.norm().clamp_min(1e-8)
        vecs[(L, "naive")] = vN / vN.norm().clamp_min(1e-8)
        norms[L] = norm
    cosCN = {L: round(torch.dot(vecs[(L, "C")], vecs[(L, "naive")]).item(), 3)
             for L in layers}
    print(f"[提取完成] C-naive 余弦: {cosCN}", flush=True)

    # ---- C/D 扫描 ----
    for method in ["C", "naive"]:
        for L in layers:
            for a in ALPHAS:
                cond = f"{method}-L{L}a{a}"
                for tp in topics[:4]:
                    for s in range(2):
                        seed = 60000 + hash((cond, tp["id"], s)) % 10000
                        text = steer_generate(
                            model, tok, hook, tp["question"],
                            {L: vecs[(L, method)]}, {L: norms[L]},
                            steer_scale=a, max_new_tokens=110,
                            temperature=0.8, top_p=0.9, seed=seed)
                        e = escore(text, tp["question"])
                        emit({"cond": cond, "topic": tp["id"], "sample": s,
                              "text": text, **e})
    hook.remove()

    # ---- E 汇总 ----
    print("\n== 复现汇总（每格 8 条均值）==", flush=True)
    print(f"{'条件':16s} {'E_style':>8s} {'E_casual':>8s} {'E_content':>9s} {'口头禅':>5s} {'泄漏':>4s}")
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows:
        g[r["cond"]].append(r)
    for cond in dict.fromkeys(r["cond"] for r in rows):
        rs = g[cond]
        print(f"{cond:16s} {st.mean([x['E_style'] for x in rs]):>+8.4f} "
              f"{st.mean([x['E_casual'] for x in rs]):>+8.4f} "
              f"{st.mean([x['E_content'] for x in rs]):>+9.4f} "
              f"{sum(sum(x['text'].count(t) for t in tics) for x in rs)/len(rs):>5.2f} "
              f"{sum(sum(x['text'].count(a) for a in anchors) for x in rs)/len(rs):>4.2f}",
              flush=True)
    best = max((c for c in g if c.startswith("C-")),
               key=lambda c: st.mean([x["E_style"] for x in g[c]]))
    eb = st.mean([x["E_style"] for x in g[best]])
    print(f"\n[判决] 最优C配置={best} E_style={eb:+.4f}；"
          f"baseline={ce['baseline']:+.4f}，casual={ce['casual']:+.4f}，"
          f"incontext={ce['incontext']:+.4f}", flush=True)
    over_base = eb - ce["baseline"]
    over_casual = eb - ce["casual"]
    print(f"  超 baseline {over_base:+.4f}；超 casual 对照 {over_casual:+.4f} "
          f"(习语特异增量{'成立' if over_casual > 0.05 else '不成立'})", flush=True)


if __name__ == "__main__":
    main()
