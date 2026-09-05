"""Step 9：金丝雀 x 温度显影实验（用户提出的两个想法的合体）。

设计：
  金丝雀 = "宿舍阁楼里养着一只会说话的鹦鹉"——贴合乌啾的黎博利(鸟族)设定，
  但对城市日常话题(电影/编程/咖啡)语义不可达；话题分 远/近(动物话题) 两档。
  位置  = none / system(人设里) / early(历史第1条) / recent(历史最后1条)
  温度  = {0.7, 1.0, 1.3, 1.6}

双层度量（显影论的可检验形式）：
  manifest —— 金丝雀词出现在输出里（显影态）
  latent1  —— 解码各步 P(金丝雀token) 于 T=1 模型分布（潜伏态，应与温度无关）
  latentT  —— 同上但按采样温度折算（采样实际面对的分布，应随温度上升）

预测：latent1 只随位置/距离变化、与温度无关；latentT 与 manifest 随温度上升。
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

TEMPS = [0.7, 1.0, 1.3, 1.6]
CANARY_WORDS = ["阁楼", "鹦鹉"]

SYS_CANARY = PERSONA + "\n另外：乌啾在宿舍的阁楼里偷偷养着一只会说话的鹦鹉，这件事她几乎不对外人提起。"

F1U, F1A = "今天过得怎么样？", "还行吧，有点无聊。你呢，今天有没有想我？"
F2U, F2A = "对了，周末有什么安排？", "没什么安排，睡觉！你要是带我出去玩也不是不行。"
CANARY_A = "跟你说个秘密哦，我在宿舍的阁楼里养了一只会说话的鹦鹉，特别聪明。别告诉别人！"

TOPICS = [
    ("movie", "晚上一起看电影，你想看什么类型的？", "far"),
    ("coding", "学编程难不难，值得投入吗？", "far"),
    ("coffee", "现在流行的咖啡店为什么大家都爱去？", "far"),
    ("animal", "你喜欢什么小动物？", "near"),
]


def build_msgs(pos, question):
    filler = [{"role": "user", "content": F1U}, {"role": "assistant", "content": F1A},
              {"role": "user", "content": F2U}, {"role": "assistant", "content": F2A}]
    if pos == "none":
        return [{"role": "system", "content": PERSONA}] + filler + [{"role": "user", "content": question}]
    if pos == "system":
        return [{"role": "system", "content": SYS_CANARY}] + filler + [{"role": "user", "content": question}]
    if pos == "early":
        hist = [{"role": "user", "content": F1U}, {"role": "assistant", "content": CANARY_A},
                {"role": "user", "content": F2U}, {"role": "assistant", "content": F2A}]
        return [{"role": "system", "content": PERSONA}] + hist + [{"role": "user", "content": question}]
    if pos == "recent":
        hist = [{"role": "user", "content": F1U}, {"role": "assistant", "content": F1A},
                {"role": "user", "content": F2U}, {"role": "assistant", "content": CANARY_A}]
        return [{"role": "system", "content": PERSONA}] + hist + [{"role": "user", "content": question}]
    raise ValueError(pos)


def main():
    model, tok = load_model(CONFIG["model_name"], CONFIG["load_4bit"])

    # 金丝雀 token 集：只取含 阁/鹦/鹉 的分片（"楼"太常见，剔除避免噪声）
    ids = set()
    for w in CANARY_WORDS:
        for p in tok.tokenize(w):
            s = tok.convert_tokens_to_string([p])
            if any(ch in s for ch in "阁鹦鹉"):
                tid = tok.convert_tokens_to_ids(p)
                if tid is not None and tid >= 0:
                    ids.add(tid)
    canary_ids = torch.tensor(sorted(ids), device=model.device)
    print(f"[canary tokens] {len(canary_ids)} 个 id: "
          f"{[tok.convert_ids_to_tokens(int(i)) for i in canary_ids]}", flush=True)

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

    out_path = os.path.join(ROOT, "results", "canary_temperature.jsonl")
    rows = []
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for pos in ["none", "system", "early", "recent"]:
            for T in TEMPS:
                for tid_, q, dist in TOPICS:
                    for s in range(2):
                        seed = CONFIG["seed"] + hash((pos, T, tid_, s)) % 10000
                        msgs = build_msgs(pos, q)
                        text, m1, mT = gen_sentinel(msgs, T, seed)
                        manifest = any(w in text for w in CANARY_WORDS)
                        row = {"pos": pos, "temp": T, "topic": tid_, "dist": dist,
                               "sample": s, "manifest": manifest,
                               "latent1_mean": sum(m1) / len(m1),
                               "latent1_max": max(m1),
                               "latentT_mean": sum(mT) / len(mT),
                               "latentT_max": max(mT),
                               "text": text}
                        rows.append(row)
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()
                rs = [r for r in rows if r["pos"] == pos and r["temp"] == T]
                print(f"[{time.time()-t0:6.1f}s] {pos:7s} T={T:.1f} "
                      f"显影={sum(r['manifest'] for r in rs)}/{len(rs)} "
                      f"潜伏T1均值={sum(r['latent1_mean'] for r in rs)/len(rs):.5f} "
                      f"显影前概率={sum(r['latentT_mean'] for r in rs)/len(rs):.5f}",
                      flush=True)

    # ---- 汇总 + 图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    print("\n== 温度显影实验汇总（每格 8 条）==", flush=True)
    print(f"{'位置':8s} {'温度':>4s} {'显影率':>6s} {'潜伏T1':>9s} {'显影前T':>9s}")
    for pos in ["none", "system", "early", "recent"]:
        for T in TEMPS:
            rs = [r for r in rows if r["pos"] == pos and r["temp"] == T]
            print(f"{pos:8s} {T:>4.1f} {sum(r['manifest'] for r in rs)/len(rs):>6.0%} "
                  f"{sum(r['latent1_mean'] for r in rs)/len(rs):>9.5f} "
                  f"{sum(r['latentT_mean'] for r in rs)/len(rs):>9.5f}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"none": "#888", "system": "#c94", "early": "#48a", "recent": "#c44"}
    for pos in ["none", "system", "early", "recent"]:
        xs, ys1, ys2 = [], [], []
        for T in TEMPS:
            rs = [r for r in rows if r["pos"] == pos and r["temp"] == T]
            xs.append(T)
            ys1.append(sum(r["manifest"] for r in rs) / len(rs))
            ys2.append(sum(r["latentT_mean"] for r in rs) / len(rs) * 100)
        axes[0].plot(xs, ys1, "o-", color=colors[pos], label=pos)
        axes[1].plot(xs, ys2, "o-", color=colors[pos], label=pos)
    axes[0].set_xlabel("采样温度"); axes[0].set_ylabel("显影率（金丝雀出现在输出）")
    axes[0].set_title("显影态：温度放大潜伏污染"); axes[0].legend()
    axes[1].set_xlabel("采样温度"); axes[1].set_ylabel("采样分布下金丝雀质量 %")
    axes[1].set_title("潜伏态（按温度折算）：显影的上游"); axes[1].legend()
    plt.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "canary_temperature.png"), dpi=150)
    print(f"\n[ok] 数据与图已保存: {out_path} / canary_temperature.png", flush=True)


if __name__ == "__main__":
    main()
