"""Step 2b：剂量扫描 —— 找到风格生效的 α，检验生效剂量下内容泄漏是否共现。

向量：mean + 单例0（晓得伐句）+ 单例4（晓得伐句）；α ∈ {0.2, 0.5, 1.0, 2.0}。
条件名形如 "mean@a0.5"，追加写入同一个 jsonl（含断点续跑）。
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
from src.model import load_model, HookManager, resolve_layers
from src.steer import generate as steer_generate

C = CONFIG
ALPHAS = [0.2, 0.5, 1.0, 2.0]
N_TOPICS = 4


def main():
    with open(os.path.join(ROOT, "data", "character_old_fisherman.json"), encoding="utf-8") as f:
        char = json.load(f)
    with open(os.path.join(ROOT, "data", "heldout_topics.json"), encoding="utf-8") as f:
        topics = json.load(f)["topics"][:N_TOPICS]
    vecs = torch.load(os.path.join(ROOT, "results", "vectors_phase1.pt"))

    model, tok = load_model(C["model_name"], C["load_4bit"])
    L = resolve_layers(model, C["layer_frac"])[0]
    if L not in vecs:
        raise RuntimeError(f"向量文件里没有 layer {L}（有 {list(vecs.keys())}）")
    d = vecs[L]
    hnorm = {L: d["hidden_norm"]}
    hook = HookManager(model)
    hook.install([L])

    targets = {"mean": d["mean"], "os0": d["oneshot"][0], "os4": d["oneshot"][4]}

    out_path = os.path.join(ROOT, "results", "generations_phase1.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["condition"], r.get("tag", ""), r["topic"], r["sample"]))
                except json.JSONDecodeError:
                    pass
    outf = open(out_path, "a", encoding="utf-8")
    t0, n = time.time(), 0

    for name, v in targets.items():
        for a in ALPHAS:
            cond = f"{name}@a{a}"
            for tp in topics:
                if (cond, "", tp["id"], 0) in done:
                    continue
                seed = C["seed"] + hash((name, a, tp["id"])) % 10000
                text = steer_generate(
                    model, tok, hook, tp["question"], {L: v}, hnorm,
                    steer_scale=a, max_new_tokens=C["max_new_tokens"],
                    temperature=C["temperature"], top_p=C["top_p"], seed=seed)
                rec = {"condition": cond, "tag": "", "topic": tp["id"], "sample": 0, "text": text}
                outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                outf.flush()
                n += 1
                leak = sum(text.count(x) for x in char["anchor_entities"])
                tic = sum(text.count(x) for x in char["style_tics"])
                print(f"[{time.time()-t0:6.1f}s #{n:2d}] {cond:12s} {tp['id']:8s} "
                      f"leak={leak} tics={tic} | {text[:30]}...", flush=True)

    hook.remove()
    outf.close()
    print(f"\n[ok] 剂量扫描完成，新增 {n} 条", flush=True)


if __name__ == "__main__":
    main()
