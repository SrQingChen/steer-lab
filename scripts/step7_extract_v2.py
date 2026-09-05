"""Step 7：决定性实验 —— 三种修复假设同时检验。

层位扫描全零后的三个嫌疑，各对应一个变体：
  A: K-token 池化提取（末8token平均，替代单末token）
  B: 只在解码步注入（v1向量 + steer_only_decode，保护 prefill 理解）
  C: 对照消元提取（示例 vs 同内容中性改写 的配对差分——Phase 2 主菜提前）
矩阵：3变体 x 2层(18,22) x 2剂量(0.5,1.0) x 4话题 x 2采样。
"""
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import env_setup  # noqa: F401
import torch
from configs.phase1 import CONFIG
from src.model import load_model, HookManager, chat_template, strip_think
from src.steer import generate as steer_generate

LAYERS = [18, 22]
ALPHAS = [0.5, 1.0]
N_TOPICS = 4
SAMPLES = 2
K_POOL = 8


def gen_plain(model, tok, user_msg, seed=0, max_new=120):
    torch.manual_seed(seed)
    msgs = [{"role": "user", "content": user_msg}]
    prompt = chat_template(tok, msgs, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=True,
                             temperature=0.3, top_p=0.9,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return strip_think(tok.decode(out[0][ids["input_ids"].shape[1]:],
                                  skip_special_tokens=True)).strip()


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
    acts = {L: torch.cat(hook.captures[L], dim=0) for L in hook.captures}
    norms = {L: sum(hook.norm_stats[L]) / len(hook.norm_stats[L]) for L in hook.norm_stats}
    hook.clear_captures()
    return acts, norms


def main():
    with open(os.path.join(ROOT, "data", "character_old_fisherman.json"), encoding="utf-8") as f:
        char = json.load(f)
    with open(os.path.join(ROOT, "data", "heldout_topics.json"), encoding="utf-8") as f:
        topics = json.load(f)["topics"][:N_TOPICS]
    anchors, tics = char["anchor_entities"], char["style_tics"]
    ex, neutrals = char["examples"], char["neutral_baselines"]

    model, tok = load_model(CONFIG["model_name"], CONFIG["load_4bit"])
    hook = HookManager(model)
    hook.install(LAYERS)

    # ---- C 变体准备：同内容中性改写 ----
    para_path = os.path.join(ROOT, "results", "paraphrases.json")
    if os.path.exists(para_path):
        paras = json.load(open(para_path, encoding="utf-8"))
    else:
        print("[prep] 生成 12 句中性改写...", flush=True)
        paras = []
        for s in ex:
            p = gen_plain(model, tok,
                          "把下面这句话改写成普通、平直的书面表达，保持内容不变，"
                          "去掉一切方言、口头禅和口语色彩：\n" + s, seed=7)
            paras.append(p)
        json.dump(paras, open(para_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for s, p in zip(ex[:3], paras[:3]):
        print(f"  原: {s}\n  改: {p}", flush=True)

    # ---- 提取向量 ----
    # A: K池化
    hook.capture_k = K_POOL
    acts_A, norms = last_acts(model, tok, hook, list(ex) + list(neutrals))
    # C: 单末token（配对差分）
    hook.capture_k = 1
    acts_C, _ = last_acts(model, tok, hook, list(ex) + list(paras) + list(neutrals))

    n_ex = len(ex)
    vecs = {}  # (variant, layer) -> unit vector
    for L in LAYERS:
        aA = acts_A[L]
        rawA = aA[:n_ex] - aA[n_ex:].mean(dim=0, keepdim=True)
        vA = rawA.mean(dim=0); vA = vA / vA.norm().clamp_min(1e-8)
        vecs[("A", L)] = vA

        aC = acts_C[L]
        rawC = aC[:n_ex] - aC[n_ex:2 * n_ex]          # 示例 - 同内容中性改写
        vC = rawC.mean(dim=0); vC = vC / vC.norm().clamp_min(1e-8)
        vecs[("C", L)] = vC

        # B 用 v1 单末token提取的向量（重提一遍，等价于 step1 协议）
        vB = aC[:n_ex] - aC[2 * n_ex:].mean(dim=0, keepdim=True)  # v1协议
        vB = vB.mean(dim=0); vB = vB / vB.norm().clamp_min(1e-8)
        vecs[("B", L)] = vB

    # A与C的向量相似度（理论诊断：若高度相似，池化/差分不改变方向）
    for L in LAYERS:
        cos_AC = torch.dot(vecs[("A", L)], vecs[("C", L)]).item()
        cos_AB = torch.dot(vecs[("A", L)], vecs[("B", L)]).item()
        cos_BC = torch.dot(vecs[("B", L)], vecs[("C", L)]).item()
        print(f"[diag] L{L}: cos(A,C)={cos_AC:.3f} cos(A,B)={cos_AB:.3f} cos(B,C)={cos_BC:.3f}",
              flush=True)

    # ---- 生成矩阵 ----
    out_path = os.path.join(ROOT, "results", "extract_v2.jsonl")
    t0 = time.time()
    rows = []
    with open(out_path, "w", encoding="utf-8") as f:
        for variant in ["A", "B", "C"]:
            hook.steer_only_decode = (variant == "B")
            for L in LAYERS:
                for a in ALPHAS:
                    cond = f"{variant}@L{L}a{a}"
                    for tp in topics:
                        for s in range(SAMPLES):
                            seed = CONFIG["seed"] + hash((cond, tp["id"], s)) % 10000
                            text = steer_generate(
                                model, tok, hook, tp["question"],
                                {L: vecs[(variant, L)]}, {L: norms[L]},
                                steer_scale=a, max_new_tokens=100,
                                temperature=CONFIG["temperature"],
                                top_p=CONFIG["top_p"], seed=seed)
                            tic = sum(text.count(x) for x in tics)
                            leak = sum(text.count(x) for x in anchors)
                            row = {"cond": cond, "variant": variant, "layer": L,
                                   "alpha": a, "topic": tp["id"], "sample": s,
                                   "tics": tic, "leak": leak, "chars": len(text),
                                   "text": text}
                            rows.append(row)
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            f.flush()
                    rs = [r for r in rows if r["cond"] == cond]
                    print(f"[{time.time()-t0:6.1f}s] {cond:12s} "
                          f"tics均值={sum(r['tics'] for r in rs)/len(rs):.2f} "
                          f"leak均值={sum(r['leak'] for r in rs)/len(rs):.2f}", flush=True)
    hook.steer_only_decode = False
    hook.remove()

    print("\n== 决定性实验汇总（每格 8 条）==", flush=True)
    print(f"{'条件':14s} {'口头禅/条':>8s} {'泄漏/条':>8s} {'字数':>6s}")
    for r0 in rows:
        pass
    seen = []
    for r in rows:
        if r["cond"] not in seen:
            seen.append(r["cond"])
    for cond in seen:
        rs = [r for r in rows if r["cond"] == cond]
        print(f"{cond:14s} {sum(r['tics'] for r in rs)/len(rs):>8.2f} "
              f"{sum(r['leak'] for r in rs)/len(rs):>8.2f} "
              f"{sum(r['chars'] for r in rs)/len(rs):>6.0f}", flush=True)


if __name__ == "__main__":
    main()
