"""向量提取：把示例语句变成残差流方向。

Phase 1 的两种提取（对照消元配对版留给 Phase 2）：
  v_oneshot[i] = act(示例i) - mean(act(中性基线))    # 单示例向量
  v_mean       = normalize(mean_i(v_oneshot_raw))    # 朴素平均向量
提取格式：示例作为 assistant 发言，取其末 token 激活。
"""
import torch

from src.model import chat_template


@torch.no_grad()
def _last_token_acts(model, tok, hook, text: str) -> None:
    """一次前向，hook 侧记录末 token 激活。"""
    msgs = [
        {"role": "user", "content": "请一字不差地复述下面这句话。"},
        {"role": "assistant", "content": text},
    ]
    prompt = chat_template(tok, msgs, add_generation_prompt=False)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    model(**ids)


@torch.no_grad()
def extract_vectors(model, tok, hook, examples, baselines, layers):
    """返回 {layer: {"oneshot": Tensor(N,d) 单位化, "mean": Tensor(d) 单位化,
                     "hidden_norm": float, "cos_matrix": Tensor(N,N)}}"""
    hook.clear_captures()
    hook.mode = "capture"
    for t in list(examples) + list(baselines):
        _last_token_acts(model, tok, hook, t)
    hook.mode = "idle"

    n_ex = len(examples)
    out = {}
    for L in layers:
        acts = torch.cat(hook.captures[L], dim=0)      # (N_ex + N_base, d)
        act_ex, act_base = acts[:n_ex], acts[n_ex:]
        base_mean = act_base.mean(dim=0, keepdim=True)

        raw = act_ex - base_mean                        # 未归一化差分
        oneshot = raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        mean_raw = raw.mean(dim=0)
        mean_v = mean_raw / mean_raw.norm().clamp_min(1e-8)
        cos = (oneshot @ oneshot.T).clamp(-1, 1)

        out[L] = {
            "oneshot": oneshot,
            "mean": mean_v,
            "hidden_norm": sum(hook.norm_stats[L]) / len(hook.norm_stats[L]),
            "cos_matrix": cos,
        }
    hook.clear_captures()
    return out
