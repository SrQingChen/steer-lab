"""必须在所有 transformers/huggingface 导入之前 import 本模块。"""
import os

# huggingface.co 不通，走国内镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# C 盘紧张，HF 缓存放本项目内（E 盘）
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(_HERE, ".hf_cache"))
