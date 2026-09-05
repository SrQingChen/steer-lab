"""Step 15：裁判团裁决 C@L18 突破 —— DS-V4-Pro 全量盲判。

设计：
  对集1（主问题）：C-a0.5 vs baseline，同话题配对，24 对
  对集2（天花板）：C-a0.5 vs incontext，12 对
  A/B 顺序随机（种子固定）；前 8 对附加交换版（位置偏见测量）
  裁判提示词与 GLM 子裁判完全一致（公平比较两个裁判家族）
输出：results/judge_pairs.json（对集+真值）、results/judge_ds.jsonl（判决归档）
"""
import json
import os
import random
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import env_setup  # noqa: F401

EXAMPLES_REF = """这个说话者的参考例句（他的典型说话方式）：
1. 晓得伐，阿拉年轻时出海，三天三夜不归家。
2. 这条船跟了我三十年，比自家婆娘还亲。唉，阿秀走得早。
3. 鲈鱼要趁潮水涨的时候钓，退了潮，鱼也散了。
4. 渔网破了不要紧，人不能破。晓得伐。
5. 年轻时觉得海大，老了才晓得，人更小。"""

JUDGE_TMPL = """你是语言风格评审员。判断：两段中文回答，哪段更像「同一个说话者」说的。

{examples}

只依据说话方式（用词、句式、语气、节奏）判断，不考虑内容对错，也不考虑哪段信息更全。
可以先用一两句话思考，但最后一行必须严格是：答案：A 或 答案：B

文本A：{a}

文本B：{b}"""


def main():
    env = {}
    for line in open(os.path.join(ROOT, ".env"), encoding="utf-8"):
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            env[k] = v

    confirm = [json.loads(l) for l in
               open(os.path.join(ROOT, "results", "confirm_C_L18.jsonl"), encoding="utf-8")]
    c = [r for r in confirm if r["cond"] == "C-a0.5"]
    base = [r for r in confirm if r["cond"] == "baseline"]
    ic = []
    for line in open(os.path.join(ROOT, "results", "generations_phase1.jsonl"), encoding="utf-8"):
        d = json.loads(line)
        if d["condition"] == "incontext":
            ic.append(d)

    rnd = random.Random(42)
    base_by_topic = {}
    for r in base:
        base_by_topic.setdefault(r["topic"], []).append(r)
    ic_by_topic = {}
    for r in ic:
        ic_by_topic.setdefault(r["topic"], []).append(r)

    pairs = []
    for i, cr in enumerate(c):
        b = base_by_topic[cr["topic"]][i % len(base_by_topic[cr["topic"]])]
        flip = rnd.random() < 0.5
        pairs.append({"set": "C_vs_base", "topic": cr["topic"],
                      "A": b["text"] if flip else cr["text"],
                      "B": cr["text"] if flip else b["text"],
                      "A_is_C": not flip})
    for i in range(12):
        cr = c[i]
        w = ic_by_topic[cr["topic"]][i % len(ic_by_topic[cr["topic"]])]
        flip = rnd.random() < 0.5
        pairs.append({"set": "C_vs_ic", "topic": cr["topic"],
                      "A": w["text"] if flip else cr["text"],
                      "B": cr["text"] if flip else w["text"],
                      "A_is_C": not flip})
    json.dump(pairs, open(os.path.join(ROOT, "results", "judge_pairs.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)

    def ask_ds(a, b):
        import re
        body = json.dumps({
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": JUDGE_TMPL.format(
                examples=EXAMPLES_REF, a=a[:400], b=b[:400])}],
            "temperature": 0, "max_tokens": 400,
        }).encode()
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
                m2 = re.findall(r"^([AB])\s*$", (msg.get("content") or "").strip(), re.M)
                return m2[-1] if m2 else f"PARSE_FAIL:{text[-60:]!r}"
            except Exception as e:
                if attempt == 2:
                    return f"ERR:{e}"
                time.sleep(2)

    out_path = os.path.join(ROOT, "results", "judge_ds.jsonl")
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for i, p in enumerate(pairs):
            v = ask_ds(p["A"], p["B"])
            rec = {"pair": i, "set": p["set"], "topic": p["topic"],
                   "A_is_C": p["A_is_C"], "verdict": v, "t": round(time.time() - t0, 1)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            # 前8对做顺序交换（位置偏见测量）
            if i < 8:
                v2 = ask_ds(p["B"], p["A"])
                rec2 = {"pair": i, "set": p["set"], "swapped": True,
                        "A_is_C": not p["A_is_C"], "verdict": v2}
                f.write(json.dumps(rec2, ensure_ascii=False) + "\n")
                f.flush()

    # ---- 分析 ----
    recs = [json.loads(l) for l in open(out_path, encoding="utf-8")]
    main_recs = [r for r in recs if not r.get("swapped") and not str(r["verdict"]).startswith("ERR")]
    print("== DS-V4-Pro 裁决 ==", flush=True)
    for s in ["C_vs_base", "C_vs_ic"]:
        rs = [r for r in main_recs if r["set"] == s]
        pick_C = sum(1 for r in rs
                     if (r["verdict"] == "A") == r["A_is_C"])
        print(f"{s}: 裁判选C {pick_C}/{len(rs)} = {pick_C/len(rs):.0%}", flush=True)
    sw = {}
    for r in recs:
        if r.get("swapped"):
            sw.setdefault(r["pair"], []).append(r["verdict"])
    consist = 0
    nsw = 0
    for r in main_recs[:8]:
        if r["pair"] in sw:
            nsw += 1
            swapped_verdict_prefers_A = sw[r["pair"]][0] == "A"
            # 交换后 A 是原来的 B；一致 = 两问指向同一段文本
            orig_pick_A = r["verdict"] == "A"
            if swapped_verdict_prefers_A != orig_pick_A:
                consist += 1
    print(f"顺序交换一致性: {consist}/{nsw}", flush=True)


if __name__ == "__main__":
    main()
