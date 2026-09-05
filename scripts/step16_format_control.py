"""Step 16：格式混杂判别实验（用户盲测反馈触发的对照）。

用户发现：C vs baseline 的裁判对里，基线多为「123条目+加粗」的助手格式，
C 为口语段落——裁判可能判的是格式不是习语。
判别设计：
  对照条件 casual-base：system 要求"口语化连贯段落、不列条目"，零渔夫信息。
  新对集1（关键）：C-a0.5 vs casual-base（格式匹配，语域放宽匹配，无习语信息）
  新对集2（次要）：C-a0.5 vs 格式清洗后的 baseline（剥离条目/加粗标记）
预测：
  C 胜 casual-base 显著 >50% => 习语转移在格式之外成立
  C ≈ casual-base => 发现降级为格式逃逸+语域放宽
"""
import json
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import env_setup  # noqa: F401
import torch
from configs.phase1 import CONFIG
from src.model import load_model, chat_template, strip_think
from scripts.step13_e_score import TOPIC_Q, mean_lp, STYLE_PERSONA, CONTENT_PERSONA
from scripts.step15_judge_ds import EXAMPLES_REF, JUDGE_TMPL

CASUAL_SYS = ("你说话随和、口语化，像普通人聊天。用一段连贯的话回答，"
              "不要列条目，不要用标题和加粗。")


def strip_format(t):
    t = re.sub(r"#{1,4}\s*", "", t)
    t = re.sub(r"\*\*", "", t)
    t = re.sub(r"^\s*\d+[\.、)]\s*", "", t, flags=re.M)
    t = re.sub(r"^\s*[-•]\s*", "", t, flags=re.M)
    return t.strip()


def gen(model, tok, system, q, seed, max_new=120):
    torch.manual_seed(seed)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": q}]
    prompt = chat_template(tok, msgs, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=True,
                             temperature=0.8, top_p=0.9,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return strip_think(tok.decode(out[0][ids["input_ids"].shape[1]:],
                                  skip_special_tokens=True)).strip()


def ask_ds(env, a, b):
    body = json.dumps({"model": "deepseek-v4-pro",
                       "messages": [{"role": "user", "content": JUDGE_TMPL.format(
                           examples=EXAMPLES_REF, a=a[:400], b=b[:400])}],
                       "temperature": 0, "max_tokens": 400}).encode()
    req = urllib.request.Request(
        env["DEEPSEEK_BASE_URL"] + "/chat/completions", data=body,
        headers={"Authorization": "Bearer " + env["DEEPSEEK_API_KEY"],
                 "Content-Type": "application/json"})
    for attempt in range(3):
        try:
            resp = json.load(urllib.request.urlopen(req, timeout=60))
            msg = resp["choices"][0]["message"]
            text = (msg.get("content") or "") + "\n" + (msg.get("reasoning_content") or "")
            m = re.findall(r"答案[:：]\s*([AB])", text)
            if m:
                return m[-1]
            return "PARSE_FAIL"
        except Exception:
            if attempt == 2:
                return "ERR"
            time.sleep(2)


def main():
    env = {}
    for line in open(os.path.join(ROOT, ".env"), encoding="utf-8"):
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            env[k] = v

    model, tok = load_model(CONFIG["model_name"], CONFIG["load_4bit"])
    with open(os.path.join(ROOT, "data", "heldout_topics.json"), encoding="utf-8") as f:
        topics = json.load(f)["topics"]

    # 1) 生成 casual-base
    cb_path = os.path.join(ROOT, "results", "casual_base.jsonl")
    casual = []
    with open(cb_path, "w", encoding="utf-8") as f:
        for tp in topics:
            for s in range(4):
                text = gen(model, tok, CASUAL_SYS, tp["question"],
                           seed=91000 + hash((tp["id"], s)) % 10000)
                r = {"cond": "casual-base", "topic": tp["id"], "sample": s, "text": text}
                casual.append(r)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 2) E 评分 casual-base（对照 C 的 E_style）
    import statistics as st
    for r in casual:
        q = TOPIC_Q[r["topic"]]
        lp0, _ = mean_lp(model, tok, None, q, r["text"])
        lps, _ = mean_lp(model, tok, STYLE_PERSONA, q, r["text"])
        lpc, _ = mean_lp(model, tok, CONTENT_PERSONA, q, r["text"])
        r["E_style"] = lps - lp0
        r["E_content"] = lpc - lp0
    with open(cb_path, "w", encoding="utf-8") as f:
        for r in casual:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    confirm = [json.loads(l) for l in
               open(os.path.join(ROOT, "results", "confirm_C_L18.jsonl"), encoding="utf-8")]
    c = [r for r in confirm if r["cond"] == "C-a0.5"]
    es_c = st.mean([r["E_style"] for r in c])
    es_cb = st.mean([r["E_style"] for r in casual])
    ec_c = st.mean([r["E_content"] for r in c])
    ec_cb = st.mean([r["E_content"] for r in casual])
    print(f"[E] C-a0.5: E_style={es_c:+.4f}  casual-base: E_style={es_cb:+.4f}  "
          f"(原始 baseline 为 -0.64)", flush=True)
    print(f"[E] E_content: C={ec_c:+.4f} casual={ec_cb:+.4f}", flush=True)

    # 3) 构造判别对集并送 DS 裁判
    import random
    rnd = random.Random(123)
    cb_by_topic = {}
    for r in casual:
        cb_by_topic.setdefault(r["topic"], []).append(r)

    pairs = []
    for i, cr in enumerate(c):
        w = cb_by_topic[cr["topic"]][i % 4]
        flip = rnd.random() < 0.5
        pairs.append({"set": "C_vs_casual", "A": w["text"] if flip else cr["text"],
                      "B": cr["text"] if flip else w["text"], "A_is_C": not flip})
    base_by_topic = {}
    for r in confirm:
        if r["cond"] == "baseline":
            base_by_topic.setdefault(r["topic"], []).append(r)
    for i, cr in enumerate(c[:12]):
        b = base_by_topic[cr["topic"]][i % 4]
        flip = rnd.random() < 0.5
        pairs.append({"set": "C_vs_normbase",
                      "A": strip_format(b["text"]) if flip else cr["text"],
                      "B": cr["text"] if flip else strip_format(b["text"]),
                      "A_is_C": not flip})

    print(f"\n[judge] 送 DS 裁判 {len(pairs)} 对...", flush=True)
    results = []
    for i, p in enumerate(pairs):
        v = ask_ds(env, p["A"], p["B"])
        results.append({**p, "verdict": v})
        if (i + 1) % 8 == 0:
            print(f"  {i+1}/{len(pairs)}", flush=True)

    for s in ["C_vs_casual", "C_vs_normbase"]:
        rs = [r for r in results if r["set"] == s and r["verdict"] in "AB"]
        pick_c = sum(1 for r in rs if (r["verdict"] == "A") == r["A_is_C"])
        print(f"{s}: 裁判选C {pick_c}/{len(rs)} = {pick_c/max(1,len(rs)):.0%}", flush=True)

    json.dump(results, open(os.path.join(ROOT, "results", "format_control.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
