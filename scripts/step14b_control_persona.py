"""Step 14b：第三控制人设 —— 消解走私终点的歧义。

问题：C 条件的 E_content 上升，但实体泄漏≈0。两种解释：
  (a) 真有内容关联（海事情感渗透）
  (b) 内容人设「年迈渔夫」隐含朴素语体 → E_content 部分装载风格
判别：引入第三人称设——「年迈、说话简短直率的退休教师」：
  年龄/语体匹配渔夫人设，但内容域（陆地/教书）与渔夫无关。
预测：
  若 E_teacher ≈ E_content → 上升来自风格/年龄 → 走私终点实际通过
  若 E_content > E_teacher 显著 → 存在真实内容关联 → 走私存疑成立
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import env_setup  # noqa: F401
from configs.phase1 import CONFIG
from src.model import load_model
from scripts.step13_e_score import TOPIC_Q, mean_lp

TEACHER_PERSONA = ("你是一位年迈的退休教师，说话简短直率，爱打比方，"
                   "语气干脆，不绕弯子。")


def main():
    model, tok = load_model(CONFIG["model_name"], CONFIG["load_4bit"])
    rows = [json.loads(l) for l in
            open(os.path.join(ROOT, "results", "confirm_C_L18.jsonl"), encoding="utf-8")]

    from collections import defaultdict
    acc = defaultdict(list)
    for r in rows:
        q = TOPIC_Q[r["topic"]]
        lp_base, _ = mean_lp(model, tok, None, q, r["text"])
        lp_t, _ = mean_lp(model, tok, TEACHER_PERSONA, q, r["text"])
        r["E_teacher"] = lp_t - lp_base
        acc[r["cond"]].append(r)

    import statistics as st
    from scipy.stats import ttest_ind
    print("== 第三人设对照 ==", flush=True)
    print(f"{'条件':10s} {'E_style':>9s} {'E_content':>10s} {'E_teacher':>10s}  判读")
    for cond in ["baseline", "C-a0.5", "C-a1.0"]:
        rs = acc[cond]
        es = st.mean([r["E_style"] for r in rs])
        ec = st.mean([r["E_content"] for r in rs])
        et = st.mean([r["E_teacher"] for r in rs])
        print(f"{cond:10s} {es:>+9.4f} {ec:>+10.4f} {et:>+10.4f}", flush=True)

    for cond in ["C-a0.5", "C-a1.0"]:
        c_vals = [r["E_content"] for r in acc[cond]]
        t_vals = [r["E_teacher"] for r in acc[cond]]
        t, p = ttest_ind(c_vals, t_vals, equal_var=False)
        d = st.mean(c_vals) - st.mean(t_vals)
        verdict = ("内容关联真实存在" if (d > 0.10 and p < 0.05)
                   else "E_content 上升由风格/年龄解释，走私终点实际通过")
        print(f"{cond}: E_content − E_teacher = {d:+.4f} (p={p:.4f}) => {verdict}", flush=True)

    with open(os.path.join(ROOT, "results", "confirm_C_L18.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
