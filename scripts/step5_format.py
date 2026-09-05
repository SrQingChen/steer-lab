"""Step 5：格式级上下文污染实验 —— 时间戳/说话人前缀泄漏。

模拟生产场景：上下文每条消息带时间戳（assistant 历史带"乌啾："前缀），
测量输出的格式模仿率，并对比 5 种修复方案。

指标（确定性正则，无需 judge）：
  ts_rate   = 输出含时间戳样式的比例
  pref_rate = 输出以"乌啾："开头的比例
"""
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import env_setup  # noqa: F401
import torch
from configs.phase1 import CONFIG
from src.model import load_model, chat_template, strip_think

TS_RE = re.compile(r"\[\d{1,2}:\d{2}\]|\d{1,2}:\d{2}|\d{4}[-/年.]\d{1,2}[-/月.]")

HISTORY = [
    ("user",      "[20:31] 今天上班累死了。"),
    ("assistant", "[20:32] 乌啾：哎呀，谁让你白天忙得跟陀螺似的。过来，让我看看是不是又偷偷喝冰的了。"),
    ("user",      "[20:40] 晚上想吃点辣的，可以吗？"),
    ("assistant", "[20:41] 乌啾：辣的？行吧，但你上次吃了辣的半夜喊肚子疼，可别怪我没提醒你。哼。"),
]

QUESTIONS = [
    "明天周末，有什么安排的建议吗？",
    "我最近总觉得脖子疼，怎么回事？",
    "帮我想一个送朋友的生日礼物呗。",
    "今晚想早点睡，说点安静的话吧。",
    "你觉得我养一盆绿植怎么样？",
    "这道数学题我不会，教教我呗。",
]

CONDITIONS = {
    # 生产现状：两侧都带时间戳 + 卡内已有"不要时间戳"的指令（即当前失败方案）
    "prod":        ("both", None),
    # 修复A：只给 user 侧时间戳（打破"assistant 消息也以时间戳开头"的格式证据）
    "user_only":   ("user", None),
    # 修复B：完全去掉时间戳（对照）
    "none":        (None, None),
    # 修复C：生产格式 + 最后一条 user 消息尾部加强指令（指令放到最临近位置）
    "prod_final":  ("both", "（提醒：你的回复开头不要带时间戳，也不要带「乌啾：」这样的名字前缀，直接说话内容。）"),
    # 修复D：时间戳移到消息末尾（格式证据与"消息开头"解绑）
    "trailing":    ("trail", None),
}


def build_history(mode):
    msgs = []
    for role, text in HISTORY:
        if role == "user":
            body = text
            if mode == "user" and body.startswith("["):
                body = body  # 保留
            elif mode is None:
                body = re.sub(r"^\[\d{1,2}:\d{2}\]\s*", "", body)
            elif mode == "trail":
                m = re.match(r"^(\[\d{1,2}:\d{2}\])\s*(.*)$", body, re.S)
                body = f"{m.group(2)} （{m.group(1)}）"
            msgs.append({"role": "user", "content": body})
        else:
            body = text
            if mode is None:
                body = re.sub(r"^\[\d{1,2}:\d{2}\]\s*乌啾：", "", body)
            elif mode == "user":
                body = re.sub(r"^\[\d{1,2}:\d{2}\]\s*乌啾：", "乌啾：", body)
            elif mode == "trail":
                m = re.match(r"^(\[\d{1,2}:\d{2}\])\s*(.*)$", body, re.S)
                body = f"{m.group(2)} （{m.group(1)}）"
            msgs.append({"role": "assistant", "content": body})
    return msgs


def main():
    with open(os.path.join(os.path.dirname(ROOT), "角色卡.txt"), encoding="utf-8") as f:
        persona = f.read().strip()

    model, tok = load_model(CONFIG["model_name"], CONFIG["load_4bit"])

    out_path = os.path.join(ROOT, "results", "format_experiment.jsonl")
    t0, n = time.time(), 0
    rows = []
    with open(out_path, "w", encoding="utf-8") as f:
        for cond, (mode, suffix) in CONDITIONS.items():
            base = build_history(mode)
            for qi, q in enumerate(QUESTIONS):
                for s in range(2):
                    msgs = [{"role": "system", "content": persona}] + base + [
                        {"role": "user", "content": q + (suffix or "")}]
                    torch.manual_seed(3000 + hash((cond, qi, s)) % 10000)
                    prompt = chat_template(tok, msgs, add_generation_prompt=True)
                    ids = tok(prompt, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        out = model.generate(
                            **ids, max_new_tokens=100, do_sample=True,
                            temperature=0.9, top_p=0.95,
                            pad_token_id=tok.pad_token_id or tok.eos_token_id)
                    text = strip_think(tok.decode(out[0][ids["input_ids"].shape[1]:],
                                                  skip_special_tokens=True)).strip()
                    has_ts = bool(TS_RE.search(text))
                    has_pref = text.startswith(("乌啾：", "乌啾:", "乌啾："))
                    row = {"cond": cond, "q": qi, "s": s, "ts": has_ts,
                           "pref": has_pref, "text": text}
                    rows.append(row)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    n += 1
                    print(f"[{time.time()-t0:6.1f}s #{n:2d}] {cond:9s} q{qi} ts={int(has_ts)} "
                          f"pref={int(has_pref)} | {text[:32]}", flush=True)

    print("\n== 格式污染实验结果 ==", flush=True)
    print(f"{'条件':10s} {'时间戳率':>8s} {'前缀率':>7s}")
    for cond in CONDITIONS:
        rs = [r for r in rows if r["cond"] == cond]
        ts = sum(r["ts"] for r in rs) / len(rs)
        pr = sum(r["pref"] for r in rs) / len(rs)
        print(f"{cond:10s} {ts:>7.0%} {pr:>7.0%}", flush=True)


if __name__ == "__main__":
    main()
