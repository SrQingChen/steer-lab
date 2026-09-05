"""Step 1：提取向量并保存。产出单示例向量间的余弦矩阵（方差现象的第一个证据）。"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import env_setup  # noqa: F401
import torch
from configs.phase1 import CONFIG
from src.model import load_model, HookManager, resolve_layers
from src.extract import extract_vectors


def main():
    with open(os.path.join(ROOT, "data", "character_old_fisherman.json"), encoding="utf-8") as f:
        char = json.load(f)

    try:
        model, tok = load_model(CONFIG["model_name"], CONFIG["load_4bit"])
    except Exception as e:
        print(f"[warn] 主模型加载失败（{e}），退回 {CONFIG['fallback_model']}")
        model, tok = load_model(CONFIG["fallback_model"], False)

    layers = resolve_layers(model, CONFIG["layer_frac"])
    print(f"[info] 模型共 {model.config.num_hidden_layers} 层，提取/注入层: {layers}")

    hook = HookManager(model)
    hook.install(layers)

    vecs = extract_vectors(model, tok, hook, char["examples"], char["neutral_baselines"], layers)

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out_path = os.path.join(ROOT, "results", "vectors_phase1.pt")
    torch.save(vecs, out_path)
    print(f"[ok] 向量已保存: {out_path}")

    for L, d in vecs.items():
        cos = d["cos_matrix"]
        n = cos.shape[0]
        off = cos[~torch.eye(n, dtype=torch.bool)]
        print(f"\n== layer {L} ==")
        print(f"hidden_norm(末token均值): {d['hidden_norm']:.2f}")
        print(f"单示例向量两两余弦: mean={off.mean():.3f}  min={off.min():.3f}  max={off.max():.3f}")
        print("余弦矩阵:")
        for i in range(n):
            print("  " + " ".join(f"{cos[i, j]: .2f}" for j in range(n)))
    hook.remove()


if __name__ == "__main__":
    main()
