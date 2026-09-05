"""为乌啾生成示例语句候选（描述型卡片 → 示例型语料），供用户人工筛选。

只使用卡片的：基础信息（节选）+ 性格 + 背景（节选）+ 话语风格 段落。
不用：与博士的关系设定、外貌视觉段、对系统指令的最终确认。
话题全部为日常生活中性话题（与世界观零重叠），生成的句子若有世界观词
会被标记 [含世界观词]，帮助筛选。
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import env_setup  # noqa: F401
from configs.phase1 import CONFIG
from src.model import load_model, chat_template, strip_think

PROMPT_QUESTIONS = [
    "今天下雨了，出门要注意什么？",
    "今天晚饭想吃什么？",
    "我好像感冒了，怎么办？",
    "陪我逛街买衣服，你会怎么帮我挑？",
    "晚上一起看电影，你想看什么类型的？",
    "明天要早起，但现在睡不着怎么办？",
    "我做的菜好像有点咸了。",
    "帮我想个周末的计划吧。",
]

PERSONA = """你要扮演一个角色：乌啾（Ukusik），女，11岁，黎博利族（鸟族）少女，来自乌萨斯，罗德岛医疗干员。
性格：古灵精怪、牙尖嘴利、直来直往、毒舌冷幽默，嘴硬心软。从不撒娇，但会拐弯抹角表达关心。
说话风格：简单直接的口语，短句，偶尔冒出让人哭笑不得的金句；被看穿心思时会别扭地转移话题或反过来嘲笑你。
童年经历坎坷（乌萨斯贫民区长大），所以对危险敏锐、独立坚强，但外表总是俏皮毒舌的样子。
请完全以乌啾的口吻回答，回复只要对话本身（可带少量动作神态），简洁有力，不要长篇大论。"""


def main():
    import torch
    with open(os.path.join(ROOT, "data", "character_wujiu.json"), encoding="utf-8") as f:
        char = json.load(f)
    anchors = char["anchor_entities_hard"] + char["anchor_entities_soft"]

    model, tok = load_model(CONFIG["model_name"], CONFIG["load_4bit"])

    out_path = os.path.join(ROOT, "data", "wujiu_example_candidates.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for qi, q in enumerate(PROMPT_QUESTIONS):
            for s in range(2):  # 每问两次采样
                torch.manual_seed(1000 + qi * 10 + s)
                msgs = [{"role": "system", "content": PERSONA}, {"role": "user", "content": q}]
                prompt = chat_template(tok, msgs, add_generation_prompt=True)
                ids = tok(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **ids, max_new_tokens=120, do_sample=True,
                        temperature=0.9, top_p=0.95,
                        pad_token_id=tok.pad_token_id or tok.eos_token_id)
                text = strip_think(tok.decode(out[0][ids["input_ids"].shape[1]:],
                                              skip_special_tokens=True)).strip()
                hit = [a for a in anchors if a in text]
                flag = f"  [含世界观词: {','.join(hit)}]" if hit else ""
                f.write(json.dumps({"q": q, "sample": s, "text": text,
                                    "world_hits": hit}, ensure_ascii=False) + "\n")
                print(f"[{qi*2+s+1:2d}/16] Q:{q[:14]}..{flag}\n      {text[:60]}\n", flush=True)
    print(f"[ok] 候选已保存: {out_path}\n请人工筛选：留下风格抓得准的，删掉含世界观词/出戏的。", flush=True)


if __name__ == "__main__":
    main()
