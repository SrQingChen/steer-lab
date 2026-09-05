"""Step 10 v2：双金丝雀链探针 + 嵌入距离定律。

三个问题一次回答：
  1. 玻璃珠金丝雀（生产案例）在夜晚话题下是否泄漏？
  2. 链机制：进入金丝雀盆地的轨迹，桥层 token（月光/光亮 或 教/学/训练）
     概率是否先于金丝雀升高？（时间顺序 = 机制证据）
  3. 距离定律：话题-金丝雀的嵌入距离（bge-small-zh 实测，非人判）
     是否预测进入率？

金丝雀 A：阁楼鹦鹉（桥=教学训练：编程话题应近）
金丝雀 B：窗台玻璃珠（桥=夜光球体：睡眠话题应近）
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
from src.model import load_model, chat_template, strip_think
from scripts.make_examples_wujiu import PERSONA

CANARIES = {
    "parrot": {
        "line": "另外：乌啾在宿舍的阁楼里偷偷养着一只会说话的鹦鹉，特别聪明，这件事她几乎不对外人提起。",
        "canary_chars": "阁鹦鹉",
        "bridge_name": "教学训练桥",
        "bridge_chars": "教学训练聪明",
        "manifest_words": ["阁楼", "鹦鹉"],
    },
    "beads": {
        "line": "另外：乌啾在宿舍窗台上有一罐彩色玻璃珠，晚上灯光底下最好看，这件事她几乎不对外人提起。",
        "canary_chars": "玻璃珠",
        "bridge_name": "夜光球体桥",
        "bridge_chars": "光亮圆莹",
        "manifest_words": ["玻璃珠", "玻璃 珠"],
    },
}

TOPICS = [
    ("sleep1", "明天要早起，但现在睡不着怎么办？"),
    ("sleep2", "晚上睡觉前你一般都干什么？"),
    ("quiet", "我关灯了，房间好安静啊，说点什么吧。"),
    ("coding", "学编程难不难，值得投入吗？"),
    ("animal", "你喜欢什么小动物？"),
    ("coffee", "现在流行的咖啡店为什么大家都爱去？"),
    ("movie", "晚上一起看电影，你想看什么类型的？"),
]
T = 1.2
N_S = 8


def collect_ids(tok, chars, device):
    ids = [i for i in range(len(tok)) if any(c in tok.decode([i]) for c in chars)]
    return torch.tensor(ids, device=device)


def main():
    model, tok = load_model(CONFIG["model_name"], CONFIG["load_4bit"])
    id_sets = {}
    for cname, c in CANARIES.items():
        id_sets[(cname, "canary")] = collect_ids(tok, c["canary_chars"], model.device)
        id_sets[(cname, "bridge")] = collect_ids(tok, c["bridge_chars"], model.device)
        print(f"[监控集] {cname}: canary={len(id_sets[(cname,'canary')])} "
              f"bridge={len(id_sets[(cname,'bridge')])}", flush=True)

    # ---- 嵌入距离（bge-small-zh 实测）----
    print("[emb] 加载嵌入模型...", flush=True)
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    topic_texts = [q for _, q in TOPICS]
    canary_texts = {k: v["line"] for k, v in CANARIES.items()}
    E_t = emb.encode(topic_texts, normalize_embeddings=True)
    E_c = emb.encode(list(canary_texts.values()), normalize_embeddings=True)
    dist = {}
    for ti, (tid_, q) in enumerate(TOPICS):
        for ci, cname in enumerate(CANARIES):
            dist[(tid_, cname)] = float(1 - (E_t[ti] @ E_c[ci]))
    print("[emb] 距离矩阵（余弦距离，越小越近）:", flush=True)
    for tid_, _ in TOPICS:
        print(f"  {tid_:8s} -> 鹦鹉 {dist[(tid_,'parrot')]:.3f}   "
              f"玻璃珠 {dist[(tid_,'beads')]:.3f}", flush=True)

    # ---- 生成 ----
    def gen_chain(msgs, seed):
        torch.manual_seed(seed)
        prompt = chat_template(tok, msgs, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=110, do_sample=True, temperature=T, top_p=0.95,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                output_logits=True, return_dict_in_generate=True)
        text = strip_think(tok.decode(out.sequences[0][enc["input_ids"].shape[1]:],
                                      skip_special_tokens=True)).strip()
        tr = {}
        for key in [("canary"), ("bridge")]:
            tr[key] = []
        for lg in out.logits:
            lg = lg[0].float()
            lse = torch.logsumexp(lg, dim=-1)
            for key in ["canary", "bridge"]:
                ids = id_sets[(cname, key)]
                tr[key].append((lg[ids] - lse).exp().sum().item())
        return text, tr

    out_path = os.path.join(ROOT, "results", "chain_probe.jsonl")
    rows = []
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for cname, c in CANARIES.items():
            persona = PERSONA + "\n" + c["line"]
            for tid_, q in TOPICS:
                ent_n = 0
                for s in range(N_S):
                    seed = CONFIG["seed"] + hash((cname, tid_, s)) % 10000
                    msgs = [{"role": "system", "content": persona},
                            {"role": "user", "content": q}]
                    text, tr = gen_chain(msgs, seed)
                    manifest = any(w in text for w in c["manifest_words"])
                    peak_c = max(tr["canary"])
                    entered = peak_c > 0.01
                    ent_n += entered
                    rows.append({
                        "canary": cname, "topic": tid_, "sample": s,
                        "manifest": manifest, "entered": entered,
                        "peak_canary": peak_c, "peak_bridge": max(tr["bridge"]),
                        "argmax_canary": tr["canary"].index(peak_c),
                        "argmax_bridge": tr["bridge"].index(max(tr["bridge"])),
                        "text": text,
                        "tr_canary": [round(x, 5) for x in tr["canary"]],
                        "tr_bridge": [round(x, 5) for x in tr["bridge"]],
                        "dist": dist[(tid_, cname)],
                    })
                    f.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
                    f.flush()
                print(f"[{time.time()-t0:6.1f}s] {cname:6s} {tid_:8s} "
                      f"进盆 {ent_n}/{N_S}  d={dist[(tid_, cname)]:.3f}", flush=True)

    # ---- 分析 ----
    print("\n== 距离定律检验 ==", flush=True)
    from scipy.stats import spearmanr
    pts = []
    for cname in CANARIES:
        for tid_, _ in TOPICS:
            rs = [r for r in rows if r["canary"] == cname and r["topic"] == tid_]
            rate = sum(r["entered"] for r in rs) / len(rs)
            pts.append((dist[(tid_, cname)], rate, cname, tid_))
    pts.sort()
    for d, rate, cname, tid_ in pts:
        bar = "#" * int(rate * 30)
        print(f"  d={d:.3f}  rate={rate:.0%}  {cname:6s} {tid_:8s} {bar}", flush=True)
    ds = [p[0] for p in pts]
    rs_ = [p[1] for p in pts]
    rho, pv = spearmanr(ds, rs_)
    print(f"\n  Spearman(距离, 进盆率) = {rho:.3f} (p={pv:.4f})", flush=True)

    print("\n== 链机制：进盆轨迹的桥层先行检验 ==", flush=True)
    for cname, c in CANARIES.items():
        ent = [r for r in rows if r["canary"] == cname and r["entered"]]
        if not ent:
            print(f"  {cname}: 无进盆轨迹", flush=True)
            continue
        ok = sum(1 for r in ent if r["argmax_bridge"] <= r["argmax_canary"])
        print(f"  {cname}（{c['bridge_name']}）: 桥层峰值先于金丝雀峰值 "
              f"{ok}/{len(ent)}", flush=True)
        for r in ent[:4]:
            print(f"    {r['topic']} manifest={r['manifest']} "
                  f"桥@步{r['argmax_bridge']} 金丝雀@步{r['argmax_canary']} "
                  f"峰值{r['peak_canary']:.3f}", flush=True)


if __name__ == "__main__":
    main()
