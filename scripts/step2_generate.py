"""Step 2：三条件生成 —— baseline / mean / oneshot_i，全部在 held-out 话题上。"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import env_setup  # noqa: F401
import torch
from configs.phase1 import CONFIG
from src.model import load_model, HookManager
from src.steer import generate as steer_generate

C = CONFIG


def main():
    with open(os.path.join(ROOT, "data", "character_old_fisherman.json"), encoding="utf-8") as f:
        char = json.load(f)
    with open(os.path.join(ROOT, "data", "heldout_topics.json"), encoding="utf-8") as f:
        topics = json.load(f)["topics"]
    vecs = torch.load(os.path.join(ROOT, "results", "vectors_phase1.pt"))

    try:
        model, tok = load_model(C["model_name"], C["load_4bit"])
    except Exception as e:
        print(f"[warn] 主模型加载失败（{e}），退回 {C.get('final_model', 'Qwen/Qwen2.5-3B-Instruct')}")
        model, tok = load_model(C.get("final_model", "Qwen/Qwen2.5-3B-Instruct"), False)

    from src.model import resolve_layers
    L = resolve_layers(model, C["layer_frac"])[0]
    if L not in vecs:
        raise RuntimeError(f"向量文件里没有 layer {L}（step1 提取的层: {list(vecs.keys())}），"
                           f"请确认 step1/step2 用了同一个模型")
    d = vecs[L]
    hnorm = {L: d["hidden_norm"]}

    hook = HookManager(model)
    hook.install([L])

    records = []
    import time
    out_path = os.path.join(ROOT, "results", "generations_phase1.jsonl")

    # 断点续跑：已完成的组合直接跳过（宿舍断电/中断只损失当前一条）
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["condition"], r.get("tag", ""), r["topic"], r["sample"]))
                    records.append(r)
                except json.JSONDecodeError:
                    pass  # 断电时可能截断的最后一行，丢弃重跑
        print(f"[resume] 已有 {len(done)} 条完成记录，将跳过", flush=True)

    outf = open(out_path, "a", encoding="utf-8")  # 追加模式 + 增量落盘
    t0 = time.time()

    def emit(rec):
        rec["elapsed_s"] = round(time.time() - t0, 1)
        records.append(rec)
        outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        outf.flush()
        r = rec
        print(f"[{rec['elapsed_s']:7.1f}s #{len(records):3d}] {r['condition']}"
              f"{(':' + r['tag']) if r['tag'] else ''} {r['topic']}#{r['sample']} | {r['text'][:36]}...",
              flush=True)

    def run(cond, vec, topic_list, n_samples, tag=""):
        for tp in topic_list:
            for s in range(n_samples):
                if (cond, tag, tp["id"], s) in done:
                    continue
                seed = C["seed"] + hash((cond, tp["id"], s)) % 10000
                text = steer_generate(
                    model, tok, hook, tp["question"], vec, hnorm,
                    steer_scale=C["steer_scale"], max_new_tokens=C["max_new_tokens"],
                    temperature=C["temperature"], top_p=C["top_p"], seed=seed)
                rec = {"condition": cond, "tag": tag, "topic": tp["id"],
                       "sample": s, "text": text}
                emit(rec)

    # 0) incontext：示例直接放 system prompt（模拟典型 AI 伴侣应用的做法，生产对照）
    ex_block = "\n".join(f"- {s}" for s in char["examples"])
    ic_system = (
        "你要扮演一个角色进行对话。以下是该角色的说话示例，"
        "请模仿它的语气和风格，但不要重复示例中的具体内容。\n" + ex_block
    )
    for tp in topics[: C["mean_topics"]]:
        for s in range(C["samples_per_topic"]):
            if ("incontext", "", tp["id"], s) in done:
                continue
            seed = C["seed"] + hash(("incontext", tp["id"], s)) % 10000
            torch.manual_seed(seed)
            msgs = [
                {"role": "system", "content": ic_system},
                {"role": "user", "content": tp["question"]},
            ]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **ids, max_new_tokens=C["max_new_tokens"], do_sample=True,
                    temperature=C["temperature"], top_p=C["top_p"],
                    pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
            from src.model import strip_think
            text = strip_think(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)).strip()
            emit({"condition": "incontext", "tag": "", "topic": tp["id"], "sample": s, "text": text})

    # 1) baseline：不注入
    run("baseline", {}, topics[: C["mean_topics"]], C["samples_per_topic"])

    # 2) mean：朴素平均向量
    run("mean", {L: d["mean"]}, topics[: C["mean_topics"]], C["samples_per_topic"])

    # 3) oneshot_i：每个单示例向量（18862 方差实验）
    for i in range(min(C["oneshot_examples"], d["oneshot"].shape[0])):
        run("oneshot", {L: d["oneshot"][i]}, topics[: C["oneshot_topics"]],
            C["oneshot_samples"], tag=str(i))

    hook.remove()
    outf.close()
    print(f"\n[ok] 共 {len(records)} 条生成已保存: {out_path}", flush=True)


if __name__ == "__main__":
    main()
