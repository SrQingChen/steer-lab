"""Step 18：复现文本的 DS 裁判感知层抽检（乌啾 x 1.7B）。

12 对：C-L11a1.0 vs baseline（同话题），盲 A/B，参照 = 12 句真实游戏台词。
"""
import json
import os
import random
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import env_setup  # noqa: F401
from scripts.step15_judge_ds import JUDGE_TMPL


def main():
    env = {}
    for line in open(os.path.join(ROOT, ".env"), encoding="utf-8"):
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            env[k] = v

    char = json.load(open(os.path.join(ROOT, "data", "character_wujiu.json"),
                          encoding="utf-8"))
    examples_ref = "这个说话者的参考例句（她的典型说话方式）：\n" + \
        "\n".join(f"{i+1}. {s}" for i, s in enumerate(char["examples"]))
    tmpl = JUDGE_TMPL.replace("{examples}", "{examples}")  # keep placeholder

    rows = [json.loads(l) for l in
            open(os.path.join(ROOT, "results", "replication_wujiu.jsonl"),
                 encoding="utf-8")]
    c = [r for r in rows if r["cond"] == "C-L11a1.0"]
    base = {}
    for r in rows:
        if r["cond"] == "baseline":
            base.setdefault(r["topic"], []).append(r)

    rnd = random.Random(77)
    pairs = []
    for i, cr in enumerate(c[:12]):
        b = base[cr["topic"]][i % len(base[cr["topic"]])]
        flip = rnd.random() < 0.5
        pairs.append({"A": b["text"] if flip else cr["text"],
                      "B": cr["text"] if flip else b["text"],
                      "A_is_C": not flip})

    def ask(a, b):
        body = json.dumps({"model": "deepseek-v4-pro",
                           "messages": [{"role": "user", "content": tmpl.format(
                               examples=examples_ref, a=a[:400], b=b[:400])}],
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
                return m[-1] if m else "PARSE_FAIL"
            except Exception:
                if attempt == 2:
                    return "ERR"
                time.sleep(2)

    pick = n = 0
    out = []
    for i, p in enumerate(pairs):
        v = ask(p["A"], p["B"])
        out.append({"pair": i, "verdict": v, "A_is_C": p["A_is_C"]})
        if v in "AB":
            n += 1
            pick += (v == "A") == p["A_is_C"]
    json.dump(out, open(os.path.join(ROOT, "results", "judge_replication.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"DS 裁判（复现文本）：选C {pick}/{n} = {pick/max(1,n):.0%}", flush=True)


if __name__ == "__main__":
    main()
