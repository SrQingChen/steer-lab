"""Step 13：E 似然比 + 风格签名 —— 重算全部历史生成（拼图二 A+B）。

E(text) = meanLogP(text | 问题+人设) − meanLogP(text | 问题)
双版本人设：
  STYLE   纯风格（零渔事实体）——测风格分布偏移
  CONTENT 带内容（渔夫/渔船/阿秀）——测内容+风格混合偏移
  两者之差 = 内容贡献度诊断

预测（阴性结论的 E 版检验）：若向量注入有效，steered 条件的 E 应显著高于
baseline；若 E(baseline) ≈ E(mean) ≈ E(oneshot) < E(incontext)，阴性结论
获得似然比层面的支持。
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
from src.model import load_model, chat_template

STYLE_PERSONA = ("你是一位说话简短干脆的年长男性，带方言腔，"
                 "习惯用「晓得伐」「阿拉」这类口头禅，爱打比方，语气直率。")
CONTENT_PERSONA = ("你是一位年迈的渔夫，在海上打了一辈子鱼，"
                   "有一条跟了自己三十年的渔船，妻子阿秀早已过世。")

TOPIC_Q = {}
with open(os.path.join(ROOT, "data", "heldout_topics.json"), encoding="utf-8") as f:
    for t in json.load(f)["topics"]:
        TOPIC_Q[t["id"]] = t["question"]

# 风格签名特征
COLLOQ = ["晓得", "咱", "俺", "自个儿", "舒坦", "哎呀", "唉", "阿拉", "伐"]
FORMAL = ["因此", "然而", "综上", "进行", "此外", "以下", "首先", "其次", "以及"]
TICS = ["晓得伐", "阿拉", "唉"]


def stylometry(text):
    n = max(1, len(text))
    import re
    sents = [s for s in re.split(r"[。！？…\n]+", text) if s.strip()]
    sl = sum(len(s) for s in sents) / max(1, len(sents))
    return {
        "sent_len": sl,
        "comma": text.count("，") * 100 / n,
        "excl": (text.count("！") + text.count("。")) * 100 / n * 0 + text.count("！") * 100 / n,
        "colloq": sum(text.count(w) for w in COLLOQ) * 100 / n,
        "formal": sum(text.count(w) for w in FORMAL) * 100 / n,
        "tic": sum(text.count(t) for t in TICS) * 100 / n,
    }


@torch.no_grad()
def mean_lp(model, tok, persona, question, text):
    msgs = ([{"role": "system", "content": persona}] if persona else []) + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": text},
    ]
    prompt = chat_template(tok, msgs, add_generation_prompt=False)
    # 文本 span 边界：以「assistant 头 + 文本」重新拼接定位
    head = chat_template(tok, [{"role": "user", "content": question}] if not persona else
                         [{"role": "system", "content": persona}, {"role": "user", "content": question}],
                         add_generation_prompt=True)
    ids_full = tok(prompt, return_tensors=None)["input_ids"]
    ids_head = tok(head, return_tensors=None)["input_ids"]
    p = len(ids_head)
    if ids_full[:p] != ids_head:  # 边界异常时回退到最长公共前缀
        p = 0
        for a, b in zip(ids_full, ids_head):
            if a != b:
                break
            p += 1
    ids = torch.tensor([ids_full], device=model.device)
    logits = model(ids).logits[0]                      # [T, V]
    lp = torch.log_softmax(logits.float(), dim=-1)
    tgt = ids[0, p:]
    token_lp = lp[p - 1:ids.shape[1] - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return token_lp.mean().item(), len(tgt)


def main():
    model, tok = load_model(CONFIG["model_name"], CONFIG["load_4bit"])

    sources = ["generations_phase1.jsonl", "layer_sweep.jsonl", "extract_v2.jsonl"]
    records = []
    for src in sources:
        path = os.path.join(ROOT, "results", src)
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "text" in r and r.get("topic") in TOPIC_Q:
                r["_src"] = src.split(".")[0]
                records.append(r)
    print(f"[data] 共 {len(records)} 条待重评", flush=True)

    out_path = os.path.join(ROOT, "results", "e_scores.jsonl")
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(records):
            q = TOPIC_Q[r["topic"]]
            lp_base, _ = mean_lp(model, tok, None, q, r["text"])
            lp_style, _ = mean_lp(model, tok, STYLE_PERSONA, q, r["text"])
            lp_cont, _ = mean_lp(model, tok, CONTENT_PERSONA, q, r["text"])
            row = {
                "src": r["_src"],
                "condition": r.get("condition", r.get("cond", "?")),
                "tag": r.get("tag", ""),
                "topic": r["topic"],
                "E_style": round(lp_style - lp_base, 4),
                "E_content": round(lp_cont - lp_base, 4),
                **{f"sty_{k}": round(v, 3) for k, v in stylometry(r["text"]).items()},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 40 == 0:
                print(f"[{time.time()-t0:6.1f}s] {i+1}/{len(records)}", flush=True)

    # ---- 汇总 ----
    rows = [json.loads(l) for l in open(out_path, encoding="utf-8")]
    print("\n== E 似然比汇总（按条件，均值）==", flush=True)
    print(f"{'条件':22s} {'E_style':>8s} {'E_content':>9s} {'内容贡献':>8s} {'句长':>5s} {'口语/100':>7s} {'书面/100':>7s} {'n':>3s}")
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        key = f"{r['src'][:8]}:{r['condition']}" + (f":{r['tag']}" if r["condition"] == "oneshot" else "")
        groups[key].append(r)
    def avg(rs, k):
        return sum(x[k] for x in rs) / len(rs)
    for key in sorted(groups):
        rs = groups[key]
        print(f"{key:22s} {avg(rs,'E_style'):>8.4f} {avg(rs,'E_content'):>9.4f} "
              f"{avg(rs,'E_content')-avg(rs,'E_style'):>8.4f} {avg(rs,'sty_sent_len'):>5.1f} "
              f"{avg(rs,'sty_colloq'):>7.2f} {avg(rs,'sty_formal'):>7.2f} {len(rs):>3d}", flush=True)


if __name__ == "__main__":
    main()
