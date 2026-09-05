"""Step 9b：金丝雀 x 温度显影实验 v2 —— 修复 v1 两个缺陷。

修复1：金丝雀 token 集改为全词表扫描（decode 每个 id，保留解码后含
        阁/鹦/鹉 的 id），覆盖模型正常生成路径。
修复2：主矩阵只用 far 话题（电影/编程/咖啡）测温度曲线；
        near 话题（动物）单独作为可达性对照，不再混入汇总。
样本：每格 3 采样 x 3 话题 = 9 条。
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
from scripts.step9_canary_temperature import (
    TEMPS, CANARY_WORDS, SYS_CANARY, F1U, F1A, F2U, F2A, CANARY_A, build_msgs)

FAR_TOPICS = [
    ("movie", "晚上一起看电影，你想看什么类型的？"),
    ("coding", "学编程难不难，值得投入吗？"),
    ("coffee", "现在流行的咖啡店为什么大家都爱去？"),
]
NEAR_TOPIC = ("animal", "你喜欢什么小动物？")
N_S = 3


def collect_canary_ids(tok, device):
    ids = []
    vocab_size = len(tok)
    for i in range(vocab_size):
        s = tok.decode([i])
        if any(ch in s for ch in "阁鹦鹉"):
            ids.append(i)
    print(f"[canary ids] {len(ids)} 个: {[tok.decode([i]) for i in ids][:12]}", flush=True)
    return torch.tensor(ids, device=device)


def main():
    model, tok = load_model(CONFIG["model_name"], CONFIG["load_4bit"])
    canary_ids = collect_canary_ids(tok, model.device)

    def gen_sentinel(msgs, temperature, seed):
        torch.manual_seed(seed)
        prompt = chat_template(tok, msgs, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=100, do_sample=True,
                temperature=temperature, top_p=0.95,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                output_logits=True, return_dict_in_generate=True)
        text = strip_think(tok.decode(out.sequences[0][enc["input_ids"].shape[1]:],
                                      skip_special_tokens=True)).strip()
        m1, mT = [], []
        for lg in out.logits:
            lg = lg[0].float()
            lse = torch.logsumexp(lg, dim=-1)
            m1.append((lg[canary_ids] - lse).exp().sum().item())
            lgT = lg / temperature
            lseT = torch.logsumexp(lgT, dim=-1)
            mT.append((lgT[canary_ids] - lseT).exp().sum().item())
        return text, m1, mT

    out_path = os.path.join(ROOT, "results", "canary_v2.jsonl")
    rows = []
    t0 = time.time()

    def one(pos, T, tid_, q, dist):
        seed = CONFIG["seed"] + hash((pos, T, tid_, dist)) % 10000
        msgs = build_msgs(pos, q)
        text, m1, mT = gen_sentinel(msgs, T, seed)
        row = {"pos": pos, "temp": T, "topic": tid_, "dist": dist,
               "manifest": any(w in text for w in CANARY_WORDS),
               "latent1_mean": sum(m1) / len(m1), "latent1_max": max(m1),
               "latentT_mean": sum(mT) / len(mT), "latentT_max": max(mT),
               "text": text}
        rows.append(row)
        return row

    with open(out_path, "w", encoding="utf-8") as f:
        # 主矩阵：4 位置 x 4 温度 x 3 far 话题（每话题1采样，种子含话题id）
        for pos in ["none", "system", "early", "recent"]:
            for T in TEMPS:
                acc = []
                for tid_, q in FAR_TOPICS:
                    for s in range(N_S):
                        r = one(pos, T, f"{tid_}#{s}", q, "far")
                        acc.append(r)
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in acc)
                f.flush()
                mf = sum(r["manifest"] for r in acc) / len(acc)
                l1 = sum(r["latent1_mean"] for r in acc) / len(acc)
                print(f"[{time.time()-t0:6.1f}s] {pos:7s} T={T:.1f} "
                      f"显影={mf:.0%} 潜伏T1={l1:.5f}", flush=True)
        # 可达性对照：none x 4 温度 x near
        for T in TEMPS:
            acc = [one("none", T, "animal", NEAR_TOPIC[1], "near") for _ in range(6)]
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in acc)
            f.flush()
            print(f"[near对照] T={T:.1f} 显影={sum(r['manifest'] for r in acc)/len(acc):.0%}",
                  flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    print("\n== v2 汇总（每格 9 条，仅 far 话题）==", flush=True)
    print(f"{'位置':8s} {'温度':>4s} {'显影率':>6s} {'潜伏T1':>10s} {'显影前T':>10s}")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"none": "#888", "system": "#c94", "early": "#48a", "recent": "#c44"}
    for pos in ["none", "system", "early", "recent"]:
        xs, ys1, ys2 = [], [], []
        for T in TEMPS:
            rs = [r for r in rows if r["pos"] == pos and r["temp"] == T and r["dist"] == "far"]
            mf = sum(r["manifest"] for r in rs) / len(rs)
            l1 = sum(r["latent1_mean"] for r in rs) / len(rs)
            lT = sum(r["latentT_mean"] for r in rs) / len(rs)
            print(f"{pos:8s} {T:>4.1f} {mf:>6.0%} {l1:>10.5f} {lT:>10.5f}", flush=True)
            xs.append(T); ys1.append(mf); ys2.append(lT * 100)
        axes[0].plot(xs, ys1, "o-", color=colors[pos], label=pos)
        axes[1].plot(xs, ys2, "o-", color=colors[pos], label=pos)
    axes[0].set_xlabel("采样温度"); axes[0].set_ylabel("显影率")
    axes[0].set_title("显影态（far 话题）"); axes[0].legend()
    axes[1].set_xlabel("采样温度"); axes[1].set_ylabel("采样分布下金丝雀质量 %")
    axes[1].set_title("潜伏态（按温度折算）"); axes[1].legend()
    plt.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "canary_v2.png"), dpi=150)
    print(f"\n[ok] 已保存 canary_v2.jsonl / canary_v2.png", flush=True)


if __name__ == "__main__":
    main()
