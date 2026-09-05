"""断点续跑基建：生成实验的通用组件（step2 模式标准化）。

用法：
    from src.resumable import ResumeStore
    store = ResumeStore(path, key_fields=["condition", "topic", "sample"])
    if store.seen({...}): continue      # 跳过已完成组合
    store.append(record)                 # 逐条落盘+flush，断电/终止只损失当前一条
"""
import json
import os


class ResumeStore:
    def __init__(self, path, key_fields):
        self.path = path
        self.key_fields = key_fields
        self._seen = set()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self._seen.add(tuple(str(r.get(k)) for k in key_fields))
                    except json.JSONDecodeError:
                        pass  # 截断行丢弃重跑
        self._f = open(path, "a", encoding="utf-8")

    def seen(self, record):
        return tuple(str(record.get(k)) for k in self.key_fields) in self._seen

    def append(self, record):
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._f.flush()
        self._seen.add(tuple(str(record.get(k)) for k in self.key_fields))

    def close(self):
        self._f.close()
