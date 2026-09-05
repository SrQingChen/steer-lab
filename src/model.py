"""模型加载与残差流 hook 基础设施。"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env_setup  # noqa: F401  必须最先执行（镜像与缓存路径）

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(model_name: str, load_4bit: bool = True):
    tok = AutoTokenizer.from_pretrained(model_name)
    kwargs = {"device_map": {"": 0}}
    if load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        try:
            kwargs["dtype"] = torch.bfloat16  # transformers v5
        except Exception:
            kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tok


def resolve_layers(model, frac: float = 0.6):
    """按深度比例取注入层（0=输入侧 1=输出侧），适配不同层数的模型。"""
    n = model.config.num_hidden_layers
    idx = min(n - 1, max(0, int(round(frac * n))))
    return [idx]


def chat_template(tok, msgs, **kw):
    """apply_chat_template 包装：对带思考模式的模型尽量关闭思考。"""
    try:
        return tok.apply_chat_template(msgs, tokenize=False, **{"enable_thinking": False, **kw})
    except (TypeError, Exception) as e:
        if "enable_thinking" in str(e) or isinstance(e, TypeError):
            return tok.apply_chat_template(msgs, tokenize=False, **kw)
        raise


def strip_think(text: str) -> str:
    """去掉生成里的 <think>...</think> 块（思考模式防御）。"""
    while "<think>" in text and "</think>" in text:
        a, b = text.index("<think>"), text.index("</think>") + len("</think>")
        text = text[:a] + text[b:]
    return text.strip()


class HookManager:
    """在 decoder layer 输出（=该层之后的残差流）上挂 hook。

    两种模式：
      capture —— 记录每次前向最后一个 token 的激活（提取期）
      steer   —— 在隐藏状态上加 scale * vec（生成期）
    """

    def __init__(self, model):
        self.model = model
        self.handles = []
        self.captures = {}   # layer -> list[Tensor(B, d)]
        self.norm_stats = {} # layer -> list[float]（末 token 激活范数，用于定标 alpha）
        self.steering = {}   # layer -> {"vec": Tensor(d), "scale": float}
        self.mode = "idle"
        self.capture_k = 1        # 提取时取末 K 个 token 的平均（>1 = 池化）
        self.steer_only_decode = False  # True = 只在解码步注入（seq_len==1），不扰 prefill

    def _make_hook(self, idx: int):
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, (tuple, list)) else output
            if self.mode == "capture":
                k = self.capture_k
                seg = hs[:, -k:, :].detach().float().cpu()
                self.captures.setdefault(idx, []).append(seg.mean(dim=1))
                self.norm_stats.setdefault(idx, []).append(
                    hs[:, -1, :].detach().float().norm(dim=-1).mean().item())
            elif self.mode == "steer" and idx in self.steering:
                if self.steer_only_decode and hs.shape[1] != 1:
                    return  # prefill 前向不注入，保护上下文读取
                cfg = self.steering[idx]
                hs = hs + cfg["scale"] * cfg["vec"].to(device=hs.device, dtype=hs.dtype)
                if isinstance(output, (tuple, list)):
                    return (hs,) + tuple(output[1:])
                return hs
        return hook

    def install(self, layers):
        for i in layers:
            self.handles.append(
                self.model.model.layers[i].register_forward_hook(self._make_hook(i)))

    def clear_captures(self):
        self.captures = {}
        self.norm_stats = {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.steering = {}
        self.mode = "idle"
