# 3. baselines.py
# Implements all baselines.

from utils import count_tokens

def full_context(segments, B):
    text = " ".join(segments)
    return text[:B], count_tokens(text)

def truncation(segments, B):
    text = " ".join(segments)
    words = text.split()[:B]
    return " ".join(words), len(words)

def retrieval_topk(segments, importance, B, k=5):
    ranked = sorted(list(zip(segments, importance)), key=lambda x: x[1], reverse=True)
    chosen = []
    total = 0
    for seg, score in ranked[:k]:
        cost = count_tokens(seg)
        if total + cost <= B:
            chosen.append(seg)
            total += cost
    return " ".join(chosen), total

def all_summaries(summaries, B):
    chosen = []
    total = 0
    for s in summaries:
        c = count_tokens(s)
        if total + c <= B:
            chosen.append(s)
            total += c
    return " ".join(chosen), total

def oracle_selection(segments, answers, B):
    """Oracle includes segments that contain gold answer string."""
    chosen, total = [], 0
    for seg in segments:
        if any(ans in seg for ans in answers):
            c = count_tokens(seg)
            if total + c <= B:
                chosen.append(seg)
                total += c
    return " ".join(chosen), total
