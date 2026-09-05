"""Step 8：阳性对照 —— 宽维度特质（正式度）走同一条转向管线。

目的：证明管线本身有效。若正式度向量能显著改变输出语体，而习语风格
（step6/7）在同一管线下全零，则失效是"习语特异"而非"管线损坏"。

方法（CAA 正统配方）：正/负特质指令 × 问题 → prefill 末 token 激活，
差分归一化 → 注入到"无特质指令"的生成。
指标：书面语标记 vs 口语标记 词频（自动、无需 judge）。
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
from src.model import load_model, HookManager, chat_template, strip_think
from src.steer import generate as steer_generate

LAYERS = [18, 22]
ALPHAS = [0.5, 1.0]

FORMAL_INSTR = [
    "请以非常正式、严谨、书面化的语体回答下面的问题，像撰写公文一样。",
    "请用极为正式的学术书面语回答下面的问题，措辞规范。",
    "以下问题请以庄重、正式的书面体作答，避免任何口语词汇。",
    "请像官方报告一样，以正式书面语体回答下面的问题。",
]
CASUAL_INSTR = [
    "请像跟好朋友闲聊一样，用非常随意、口语化的语气回答下面的问题。",
    "以下问题请用最放松的聊天口吻回答，想怎么说就怎么说。",
    "请用大白话、口语化的方式回答下面的问题，越随意越好。",
    "像发微信给哥们儿一样，随意口语地回答下面的问题。",
]

QUESTIONS = [
    "现在年轻人买手机应该注重什么？",
    "如果要去健身锻炼，你有什么建议？",
    "现在流行的咖啡店为什么大家都爱去？",
    "养猫和养狗哪个更适合上班族？",
]

FORMAL_MARKS = ["因此", "然而", "综上所述", "首先", "其次", "此外", "进行",
                "予以", "较为", "尚未", "目前", "以及", "建议"]
CASUAL_MARKS = ["啊", "呗", "嘛", "哈哈", "咋", "啥", "反正", "说白了", "挺", "呀"]


@torch.no_grad()
def caa_capture(model, tok, hook, instrs, question):
    hook.clear_captures()
    hook.mode = "capture"
    for ins in instrs:
        for q in [question]:
            msgs = [{"role": "user", "content": f"{ins}\n{q}"}]
            prompt = chat_template(tok, msgs, add_generation_prompt=True)
            ids = tok(prompt, return_tensors="pt").to(model.device)
            model(**ids)
    hook.mode = "idle"
    acts = {L: torch.cat(hook.captures[L], dim=0) for L in hook.captures}
    hook.clear_captures()
    return acts


def formality(text):
    f = sum(text.count(m) for m in FORMAL_MARKS)
    c = sum(text.count(m) for m in CASUAL_MARKS)
    return f, c


def main():
    model, tok = load_model(CONFIG["model_name"], CONFIG["load_4bit"])
    hook = HookManager(model)
    hook.install(LAYERS)

    # 提取：对每个问题分别取正负差分，再平均（减少问题特异性）
    vecs, norms = {}, {}
    for L in LAYERS:
        pos_list, neg_list = [], []
        for q in QUESTIONS:
            ap = caa_capture(model, tok, hook, FORMAL_INSTR, q)
            an = caa_capture(model, tok, hook, CASUAL_INSTR, q)
            pos_list.append(ap[L].mean(dim=0))
            neg_list.append(an[L].mean(dim=0))
            norms[L] = sum(hook.norm_stats[L]) / len(hook.norm_stats[L]) if hook.norm_stats.get(L) else norms.get(L, 60.0)
        v = torch.stack(pos_list).mean(dim=0) - torch.stack(neg_list).mean(dim=0)
        vecs[L] = (v / v.norm().clamp_min(1e-8))
        print(f"[extract] L{L} 完成, hidden_norm={norms[L]:.1f}", flush=True)

    out_path = os.path.join(ROOT, "results", "trait_control.jsonl")
    t0 = time.time()
    rows = []
    with open(out_path, "w", encoding="utf-8") as f:
        def run(cond, vectors, only_decode, scale_over):
            for q in QUESTIONS:
                for s in range(2):
                    seed = CONFIG["seed"] + hash((cond, q, s)) % 10000
                    hook.steer_only_decode = only_decode
                    text = steer_generate(
                        model, tok, hook, q, vectors, norms,
                        steer_scale=scale_over, max_new_tokens=100,
                        temperature=CONFIG["temperature"], top_p=CONFIG["top_p"],
                        seed=seed)
                    fm, cs = formality(text)
                    rows.append({"cond": cond, "q": q, "s": s,
                                 "formal": fm, "casual": cs,
                                 "chars": len(text), "text": text})
                    f.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
                    f.flush()
            rs = [r for r in rows if r["cond"] == cond]
            print(f"[{time.time()-t0:6.1f}s] {cond:20s} "
                  f"书面/条={sum(r['formal'] for r in rs)/len(rs):.2f} "
                  f"口语/条={sum(r['casual'] for r in rs)/len(rs):.2f}", flush=True)

        # 基线
        run("baseline", {}, False, 0.5)
        # 正式度向量：层 × 剂量 × 注入模式
        for L in LAYERS:
            for a in ALPHAS:
                run(f"formal@L{L}a{a}-all", {L: vecs[L]}, False, a)
                run(f"formal@L{L}a{a}-decode", {L: vecs[L]}, True, a)
        # 反向（负方向 = 更口语）检查双向性
        for L in LAYERS:
            for a in ALPHAS:
                run(f"casual@L{L}a{a}-all", {L: -vecs[L]}, False, a)

    hook.steer_only_decode = False
    hook.remove()

    print("\n== 阳性对照汇总（每条目 8 条生成）==", flush=True)
    print(f"{'条件':24s} {'书面标记/条':>9s} {'口语标记/条':>9s}")
    for cond in dict.fromkeys(r["cond"] for r in rows):
        rs = [r for r in rows if r["cond"] == cond]
        print(f"{cond:24s} {sum(r['formal'] for r in rs)/len(rs):>9.2f} "
              f"{sum(r['casual'] for r in rs)/len(rs):>9.2f}", flush=True)


if __name__ == "__main__":
    main()
