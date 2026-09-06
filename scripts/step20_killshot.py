"""Step 20：杀手实验 —— CAST(LaMP-5 Scholarly Title) 协议复现 + 效应分解。

预注册（跑前固定）：
  复现验收：A臂(CAST向量) 相对 non-pers 的 ROUGE-L 提升为正且 >1%
  分解判据：若 B(全档案上下文) ≥ 0.8×A 且 C(错配向量) 不提升 →
           效应主要由"用户历史表层统计收敛"承载（机制无特异性）
  中介分析：METEOR 提升分解为功能词重合增量 vs 实义词重合增量
任务：LaMP-5 Scholarly Title（CAST 最大效应 +25.8% ROUGE-L）
模型：Llama-2-7b-chat 4-bit（CAST 用 8-bit；本机显存限制，差异已声明）
层位/α：CAST 验证集选层 → 我们在 {15,20,25}×{0.5,1,2} 上用前 8 用户小扫描
中性回复：本地模型贪心生成（CAST 用 gpt-3.5-turbo，差异已声明）
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import env_setup  # noqa: F401
import torch
from src.model import load_model, chat_template, strip_think
from src.resumable import ResumeStore

MODEL = "NousResearch/Llama-2-7b-chat-hf"
N_USERS = 50
MAX_HIST = 8
LAYERS_SWEEP = [15, 20, 25]
ALPHAS_SWEEP = [0.5, 1.0, 2.0]
N_SWEEP_USERS = 8


def load_lamp(path_q, path_h, n, max_hist):
    qs = {}
    for line in open(path_q, encoding="utf-8"):
        r = json.loads(line)
        qs[r["id"]] = r
    # 历史文件为扁平格式：每行一条记录(id, source, target, generation)，按 id 聚组
    from collections import defaultdict
    prof = defaultdict(list)
    for line in open(path_h, encoding="utf-8"):
        r = json.loads(line)
        prof[r["id"]].append({"source": r["source"], "target": r["target"]})
    users = []
    for uid, q in list(qs.items())[:n]:
        items = prof.get(uid, [])[:max_hist]
        if len(items) < 3:
            continue
        users.append({"uid": uid, "source": q["source"], "target": q["target"],
                      "profile": items})
    return users


def main():
    users = load_lamp(os.path.join(ROOT, "data/lamp/lamp5_test_query.jsonl"),
                      os.path.join(ROOT, "data/lamp/lamp5_test_history.jsonl"),
                      N_USERS, MAX_HIST)
    print(f"[data] {len(users)} 用户（CAST表4: 全集2500, 我们采样）", flush=True)

    model, tok = load_model(MODEL, True)  # 4-bit
    n_layers = model.config.num_hidden_layers

    # ---- 模型默认对话模板（Llama-2 无内置 chat template 时手工构造）----
    def wrap(sysmsg, user):
        s = f"<s>[INST] "
        if sysmsg:
            s += f"<<SYS>>\n{sysmsg}\n<</SYS>>\n\n"
        s += f"{user} [/INST]"
        return s

    @torch.no_grad()
    def gen_greedy(sysmsg, user, max_new=40):
        prompt = wrap(sysmsg, user)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        return strip_think(tok.decode(out[0][ids["input_ids"].shape[1]:],
                                      skip_special_tokens=True)).strip()

    @torch.no_grad()
    def last_act(sysmsg, user, reply, L):
        prompt = wrap(sysmsg, user) + " " + reply + " </s>"
        ids = tok(prompt, return_tensors="pt").to(model.device)
        out = model(**ids, output_hidden_states=True)
        return out.hidden_states[L][0, -1, :].float().cpu()

    # ================= 阶段1：中性回复（CAST 负臂）=================
    store_n = ResumeStore(os.path.join(ROOT, "results", "killshot_neutral.jsonl"),
                          ["uid", "hi"])
    print("[阶段1] 生成 style-agnostic 中性回复（本地贪心）...", flush=True)
    t0 = time.time()
    for u in users:
        for hi, piece in enumerate(u["profile"]):
            if store_n.seen({"uid": u["uid"], "hi": hi}):
                continue
            neutral = gen_greedy(None, piece["source"], max_new=40)
            store_n.append({"uid": u["uid"], "hi": hi, "source": piece["source"],
                            "user_reply": piece["target"], "neutral": neutral})
    store_n.close()
    print(f"[阶段1] 完成 {time.time()-t0:.0f}s", flush=True)

    # ================= 阶段2：层位/α 小扫描（前 N_SWEEP_USERS 用户）=================
    neutral_map = {}
    for line in open(os.path.join(ROOT, "results", "killshot_neutral.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        neutral_map[(r["uid"], r["hi"])] = r

    @torch.no_grad()
    def extract_vec(user, L):
        key = (user["uid"], L)
        if key in _VEC_CACHE:
            return _VEC_CACHE[key]
        pos_l, neg_l = [], []
        for hi, piece in enumerate(user["profile"]):
            n = neutral_map[(user["uid"], hi)]
            pos_l.append(last_act(None, piece["source"], piece["target"], L))
            neg_l.append(last_act(None, piece["source"], n["neutral"], L))
        v = torch.stack(pos_l).mean(0) - torch.stack(neg_l).mean(0)
        v = v / v.norm().clamp_min(1e-8)
        _VEC_CACHE[key] = v
        return v

    _VEC_CACHE = {}  # (uid, layer) -> 向量；进程内缓存，避免每次生成重复16次前向

    # HookManager 注入需要预挂层；小扫描逐层挂
    from src.model import HookManager
    hook = HookManager(model)
    hook.install(LAYERS_SWEEP)

    def steer_gen(user, L, alpha, sysmsg=None):
        vecs = {L: extract_vec(user, L).to(model.device)}
        hook.steering = {L: {"vec": vecs[L], "scale": alpha}}
        hook.mode = "steer"
        try:
            return gen_greedy(sysmsg, user["source"], max_new=40)
        finally:
            hook.mode = "idle"
            hook.steering.clear()

    # 小扫描：非个性化基线 vs 各(层,α)
    print(f"[阶段2] 层位/α 小扫描（{N_SWEEP_USERS} 用户）...", flush=True)
    sweep_store = ResumeStore(os.path.join(ROOT, "results", "killshot_sweep.jsonl"),
                              ["uid", "L", "alpha", "arm"])
    from rouge_score import rouge_scorer
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def rl(pred, ref):
        return rs.score(ref, pred)["rougeL"].fmeasure

    def base_gen(user, sysmsg=None):
        return gen_greedy(sysmsg, user["source"], max_new=40)

    for u in users[:N_SWEEP_USERS]:
        for arm in ["nonpers", "prompt_profile"] + \
                   [f"cast-L{L}a{a}" for L in LAYERS_SWEEP for a in ALPHAS_SWEEP]:
            if sweep_store.seen({"uid": u["uid"], "L": 0, "alpha": 0, "arm": arm}):
                continue
            if arm == "nonpers":
                text = base_gen(u)
            elif arm == "prompt_profile":
                prof = "\n".join(f"Abstract: {p['source'][:200]}\nYour title: {p['target']}"
                                 for p in u["profile"])
                text = base_gen(u, "Follow the title-writing style shown in these examples.\n" + prof)
            else:
                L = int(arm.split("L")[1].split("a")[0])
                a_ = float(arm.split("a")[-1])
                text = steer_gen(u, L, a_)
            sweep_store.append({"uid": u["uid"], "L": 0, "alpha": 0, "arm": arm,
                                "text": text, "rougeL": rl(text, u["target"])})
    sweep_store.close()

    # 选最优 (层, α)
    import statistics as st
    from collections import defaultdict
    g = defaultdict(list)
    for line in open(os.path.join(ROOT, "results", "killshot_sweep.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        g[r["arm"]].append(r["rougeL"])
    base_r = st.mean(g["nonpers"])
    best_cfg, best_r = None, -1
    for arm in [a for a in g if a.startswith("cast-")]:
        m = st.mean(g[arm])
        if m > best_r:
            best_r, best_cfg = m, arm
    print(f"[阶段2] nonpers RL={base_r:.4f}  最优配置={best_cfg} RL={best_r:.4f}", flush=True)
    json.dump({"best_cfg": best_cfg, "base_r": base_r, "best_r": best_r},
              open(os.path.join(ROOT, "results", "killshot_cfg.json"), "w"))

    # ================= 阶段3：全量四臂 =================
    L = int(best_cfg.split("L")[1].split("a")[0])
    A_ = float(best_cfg.split("a")[-1])
    full_store = ResumeStore(os.path.join(ROOT, "results", "killshot_main.jsonl"),
                             ["uid", "arm"])
    print(f"[阶段3] 全量 {len(users)} 用户 × 4 臂（L={L}, α={A_}）...", flush=True)
    # 错配用户向量：用户 i 用用户 (i+1)%n 的历史提取
    for i, u in enumerate(users):
        wrong = users[(i + 1) % len(users)]
        arms = {
            "nonpers": None,
            "cast": ("steer", u, L, A_),
            "prompt_profile": ("prompt", None, 0, 0),
            "wrong_user_vec": ("steer", wrong, L, A_),
        }
        for arm, spec in arms.items():
            if full_store.seen({"uid": u["uid"], "arm": arm}):
                continue
            if spec is None:
                text = base_gen(u)
            elif spec[0] == "prompt":
                prof = "\n".join(f"Abstract: {p['source'][:200]}\nYour title: {p['target']}"
                                 for p in u["profile"])
                text = base_gen(u, "Follow the title-writing style shown in these examples.\n" + prof)
            else:
                text = steer_gen(spec[1], spec[2], spec[3])
            full_store.append({"uid": u["uid"], "arm": arm, "text": text,
                               "rougeL": rl(text, u["target"])})
    full_store.close()

    # ================= 阶段4：汇总 + 分解 =================
    g2 = defaultdict(list)
    for line in open(os.path.join(ROOT, "results", "killshot_main.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        g2[r["arm"]].append(r["rougeL"])
    print("\n== 杀手实验主表（ROUGE-L, n=%d 用户）==" % len(users), flush=True)
    base_full = st.mean(g2["nonpers"])
    for arm in ["nonpers", "cast", "prompt_profile", "wrong_user_vec"]:
        m = st.mean(g2[arm])
        print(f"  {arm:16s} {m:.4f}  ({(m-base_full)/base_full:+.1%})", flush=True)
    r_cast = st.mean(g2["cast"])
    r_prompt = st.mean(g2["prompt_profile"])
    r_wrong = st.mean(g2["wrong_user_vec"])
    gain_cast = r_cast - base_full
    gain_prompt = r_prompt - base_full
    share = gain_prompt / gain_cast if abs(gain_cast) > 1e-9 else float("nan")
    print(f"\n[判决] CAST增益 {gain_cast:+.4f}；全档案prompt增益 {gain_prompt:+.4f}；"
          f"prompt份额 {share:.0%}；错配向量增益 {r_wrong-base_full:+.4f}", flush=True)
    verdict = ("表层收敛主导（机制无特异性）" if share > 0.8 and abs(r_wrong-base_full) < 0.3*abs(gain_cast)
               else "向量机制有增量" if share < 0.5 else "部分分解")
    print(f"  => {verdict}", flush=True)


if __name__ == "__main__":
    main()
