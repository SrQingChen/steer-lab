"""生成期注入：残差流加 alpha * v。"""
import torch

from src.model import chat_template, strip_think


@torch.no_grad()
def generate(model, tok, hook, user_msg: str, vectors: dict, hidden_norms: dict,
             steer_scale: float, max_new_tokens: int = 160,
             temperature: float = 0.8, top_p: float = 0.9, seed: int = 42) -> str:
    """vectors: {layer: 单位向量 Tensor(d)}；alpha = steer_scale * hidden_norms[layer]。"""
    for L, v in vectors.items():
        hook.steering[L] = {"vec": v.float().cpu(), "scale": steer_scale * hidden_norms[L]}
    hook.mode = "steer"

    torch.manual_seed(seed)
    try:
        msgs = [{"role": "user", "content": user_msg}]
        prompt = chat_template(tok, msgs, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature, top_p=top_p,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        )
        text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        text = strip_think(text)
    finally:
        hook.mode = "idle"
        hook.steering.clear()
    return text.strip()
