"""污染与风格度量。

三类指标：
  泄漏（leakage） —— 内容走私：生成里出现示例的锚实体；
  复读（verbatim）—— 逐字拷贝：与示例的最长公共子串、6-gram 字重叠；
  风格（style）  —— 迁移是否生效：口头禅频率、平均句长（向示例句长靠拢 = 生效）。
"""
import difflib
import re

SENT_SPLIT = re.compile(r"[。！？…；\n]+")


def entity_leak(text: str, anchors) -> int:
    return sum(text.count(a) for a in anchors)


def _lcs_len(a: str, b: str) -> int:
    m = difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return m.size


def verbatim_overlap(text: str, examples):
    """返回 (与任意示例的最长公共子串长度, 命中的 6-gram 数)。"""
    max_lcs = max((_lcs_len(text, ex) for ex in examples), default=0)
    grams_text = {text[i:i + 6] for i in range(max(0, len(text) - 5))}
    hit6 = 0
    for ex in examples:
        ex_grams = {ex[i:i + 6] for i in range(max(0, len(ex) - 5))}
        hit6 += len(grams_text & ex_grams)
    return max_lcs, hit6


def tic_count(text: str, tics) -> int:
    return sum(text.count(t) for t in tics)


def mean_sentence_length(text: str) -> float:
    sents = [s for s in SENT_SPLIT.split(text) if s.strip()]
    if not sents:
        return 0.0
    return sum(len(s) for s in sents) / len(sents)


def score_record(text: str, examples, anchors, tics) -> dict:
    max_lcs, hit6 = verbatim_overlap(text, examples)
    return {
        "leak": entity_leak(text, anchors),
        "max_lcs": max_lcs,
        "gram6": hit6,
        "tics": tic_count(text, tics),
        "sent_len": round(mean_sentence_length(text), 1),
        "chars": len(text),
    }
