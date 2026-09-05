"""Phase 1 实验配置：复现「示例语句内容走私」现象。"""

CONFIG = {
    # 迭代用 3B 全精度（bnb 4-bit 在本机仅 ~1 tok/s，不可迭代）；
    # 最终确认跑切回 "Qwen/Qwen3.5-9B" + load_4bit=True，挂机过夜
    "model_name": "Qwen/Qwen2.5-3B-Instruct",
    "load_4bit": False,
    "final_model": "Qwen/Qwen3.5-9B",

    # 残差流层位：按模型深度取比例（0.6 ≈ 中层），适配不同层数的模型
    "layer_frac": 0.60,

    # 注入强度：alpha = steer_scale * 该层激活的平均范数（相对单位化向量）
    "steer_scale": 0.10,

    # 生成参数
    "max_new_tokens": 160,
    "temperature": 0.8,
    "top_p": 0.9,
    "seed": 42,

    # 实验规模（控制 8GB 笔记本的时长）
    "mean_topics": 6,       # 平均向量 & 基线：全部 6 个 held-out 话题
    "samples_per_topic": 3,
    "oneshot_examples": 12, # 单示例方差实验：全部 12 个单句向量
    "oneshot_topics": 2,    # 只用前 2 个话题
    "oneshot_samples": 2,
}
