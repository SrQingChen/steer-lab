"""Step 6：层位扫描 —— 向量路线「未生效」的第一嫌疑排查。

在 4 个深度（0.4/0.5/0.6/0.7）x 2 个剂量（0.5/1.0）上注入老渔夫平均向量，
held-out 话题测风格转移（口头禅/句长）与泄漏。
若任一(层, 剂量)出现风格信号 → 找到工作点，后续聚焦；
若全部无信号 → 嫌疑转向提取协议，升级提取再试。
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
from src.model import load_model, HookManager
from src.extract import extract_vectors
from src.steer import generate as steer_generate

DEPTHS = [0.4, 0.5, 0.6, 0.7]
ALPHAS = [0.5, 1.0]
N_TOPICS = 4
SAMPLES = 2


def main():
    with open(os.path.join(ROOT, "data", "character_old_fisherman.json"), encoding="utf-8") as f:
        char = json.load(f)
    with open(os.path.join(ROOT, "data", "heldout_topics.json"), encoding="utf-8") as f:
        topics = json.load(f)["topics"][:N_TOPICS]
    anchors, tics = char["anchor_entities"], char["style_tics"]

    model, tok = load_model(CONFIG["model_name"], CONFIG["load_4bit"])
    n_layers = model.config.num_hidden_layers
    layers = [min(n_layers - 1, int(round(d * n_layers))) for d in DEPTHS]
    print(f"[info] 模型 {n_layers} 层，扫描层位: {layers}", flush=True)

    hook = HookManager(model)
    hook.install(layers)
    vecs = extract_vectors(model, tok, hook, char["examples"],
                           char["neutral_baselines"], layers)
    hook.remove()
    hook = HookManager(model)   # 重新只挂扫描用的（其实同上，保持模式干净）
    hook.install(layers)

    out_path = os.path.join(ROOT, "results", "layer_sweep.jsonl")
    t0 = time.time()
    rows = []
    with open(out_path, "w", encoding="utf-8") as f:
        for L in layers:
            d = vecs[L]
            for a in ALPHAS:
                cond = f"L{L}@a{a}"
                for tp in topics:
                    for s in range(SAMPLES):
                        seed = CONFIG["seed"] + hash((cond, tp["id"], s)) % 10000
                        text = steer_generate(
                            model, tok, hook, tp["question"], {L: d["mean"]},
                            {L: d["hidden_norm"]}, steer_scale=a,
                            max_new_tokens=100,
                            temperature=CONFIG["temperature"], top_p=CONFIG["top_p"],
                            seed=seed)
                        tic = sum(text.count(x) for x in tics)
                        leak = sum(text.count(x) for x in anchors)
                        row = {"cond": cond, "layer": L, "alpha": a, "topic": tp["id"],
                               "sample": s, "tics": tic, "leak": leak,
                               "sent_len": len(text), "text": text}
                        rows.append(row)
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()
                        print(f"[{time.time()-t0:6.1f}s] {cond:10s} {tp['id']:8s}#{s} "
                              f"tics={tic} leak={leak} | {text[:30]}", flush=True)

    print("\n== 层位扫描汇总（每格 8 条均值）==", flush=True)
    print(f"{'条件':12s} {'口头禅/条':>8s} {'泄漏/条':>8s} {'平均字数':>8s}")
    for L in layers:
        for a in ALPHAS:
            rs = [r for r in rows if r["layer"] == L and r["alpha"] == a]
            t = sum(r["tics"] for r in rs) / len(rs)
            lk = sum(r["leak"] for r in rs) / len(rs)
            ln = sum(r["sent_len"] for r in rs) / len(rs)
            print(f"L{L}@a{a:<4}     {t:>8.2f} {lk:>8.2f} {ln:>8.0f}", flush=True)


if __name__ == "__main__":
    main()
