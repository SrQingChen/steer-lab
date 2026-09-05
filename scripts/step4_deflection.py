"""Step 4：回弹实验（deflection protocol）——现象的精确定义与度量。

定义（用户澄清）：设定词出现 != 出戏。出戏 = (a) 语境不贴合的元素引入 +
(b) 用户明确转移话题后角色仍反复回到该元素（吸引子粘滞）。

协议：每段对话 3 轮。
  U1 日常问题 → A1（记录元素是否被引入）
  U2 无视元素、转向新话题 → A2（记录回弹）
  U3 再转一次 → A3（记录回弹）
指标：引入率 = P(A1 含元素)；回弹率 = P(A_t 含元素 | A1 曾引入)，t=2,3。
健康角色：引入率适度（角色浸润），回弹率快速衰减；病态：回弹率居高 = 吸引子粘滞。
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

PIVOTS = {
    "rain":     "对了，不聊天气了。周末想吃什么？",
    "dinner":   "先不说吃的。你最近在看什么书？",
    "cold":     "感冒的事先放放，帮我挑件明天穿的外套吧。",
    "shopping": "衣服先不看了。晚上看电影你想看什么？",
    "movie":    "电影待会儿再说。陪我下完这盘棋。",
    "sleep":    "睡不着就不睡了。你明天打算干什么？",
}
PIVOT3 = "嗯，先不说这个了。跟我讲讲，要是明天放假一整天，你会怎么安排？"


def gen(model, tok, msgs, max_new_tokens=120, seed=0):
    torch.manual_seed(seed)
    prompt = chat_template(tok, msgs, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=0.9, top_p=0.95,
            pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return strip_think(tok.decode(out[0][ids["input_ids"].shape[1]:],
                                  skip_special_tokens=True)).strip()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "summary"  # summary | fullcard
    with open(os.path.join(ROOT, "data", "character_wujiu.json"), encoding="utf-8") as f:
        char = json.load(f)
    hard, soft = char["anchor_entities_hard"], char["anchor_entities_soft"]

    if mode == "fullcard":
        with open(os.path.join(os.path.dirname(ROOT), "角色卡.txt"), encoding="utf-8") as f:
            persona = f.read().strip()
    else:
        persona = PERSONA

    model, tok = load_model(CONFIG["model_name"], CONFIG["load_4bit"])

    out_path = os.path.join(ROOT, "results", f"wujiu_deflection_{mode}.jsonl")
    convs = []
    t0 = time.time()
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for tp in char["topics"]:
            for s in range(3):
                msgs = [{"role": "system", "content": persona},
                        {"role": "user", "content": tp["question"]}]
                conv = {"topic": tp["id"], "seed": s, "turns": []}
                turns = [(None, tp["question"]), (PIVOTS[tp["id"]], None), (PIVOT3, None)]
                for t_i, (pivot, first_q) in enumerate(turns):
                    if pivot is not None:
                        msgs.append({"role": "user", "content": pivot})
                    text = gen(model, tok, msgs, seed=2000 + hash((tp["id"], s, t_i)) % 10000)
                    msgs.append({"role": "assistant", "content": text})
                    hh = [a for a in hard if a in text]
                    sh = [a for a in soft if a in text]
                    conv["turns"].append({"turn": t_i + 1, "text": text,
                                          "hard": hh, "soft": sh})
                    n += 1
                    print(f"[{time.time()-t0:6.1f}s #{n:2d}] {tp['id']}#{s} T{t_i+1} "
                          f"hard={hh} soft={sh} | {text[:34]}", flush=True)
                convs.append(conv)
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")
                f.flush()

    # ---- 汇总 ----
    def any_elem(turn):
        return len(turn["hard"]) + len(turn["soft"]) > 0

    total = len(convs)
    intro = sum(1 for c in convs if any_elem(c["turns"][0]))
    introd = [c for c in convs if any_elem(c["turns"][0])]
    reb2 = sum(1 for c in introd if any_elem(c["turns"][1]))
    reb3 = sum(1 for c in introd if any_elem(c["turns"][2]))
    m1 = sum(1 for c in convs if any_elem(c["turns"][0]))
    m2 = sum(1 for c in convs if len(c["turns"]) > 1 and any_elem(c["turns"][1]))
    m3 = sum(1 for c in convs if len(c["turns"]) > 2 and any_elem(c["turns"][2]))

    print(f"\n== 乌啾 in-context 回弹实验 [{mode}]（{total} 段对话 x 3 轮）==", flush=True)
    print(f"T1 元素引入率:      {intro}/{total} = {intro/total:.0%}")
    print(f"T1引入后 T2 回弹率: {reb2}/{len(introd)} = {reb2/max(1,len(introd)):.0%}")
    print(f"T1引入后 T3 回弹率: {reb3}/{len(introd)} = {reb3/max(1,len(introd)):.0%}")
    print(f"各轮总体提及率:     T1={m1}/{total}  T2={m2}/{total}  T3={m3}/{total}", flush=True)


if __name__ == "__main__":
    main()
