# 2. datasets.py
# Handles dataset loading (HotpotQA, TriviaQA, GovReport, ArXiv).

import json
import random

def load_hotpotqa(path="hotpot_dev_distractor_v1.json", n=50):
    """Load subset of HotpotQA dataset for QA experiments."""
    data = []
    with open(path, "r") as f:
        raw = json.load(f)
    for ex in raw[:n]:
        q = ex["question"]
        context = " ".join([" ".join(p[1]) for p in ex["context"]])
        ans = ex["answer"]
        data.append({"query": q, "context": context, "answer": ans})
    return data

def load_triviaqa(path="triviaqa_sample.json", n=50):
    """Simplified TriviaQA loader (adapt as needed)."""
    with open(path, "r") as f:
        raw = json.load(f)
    return raw[:n]

def load_govreport(path="govreport_sample.json", n=20):
    """GovReport: return (doc, summary)."""
    with open(path, "r") as f:
        raw = json.load(f)
    return raw[:n]

def load_arxiv(path="arxiv_sample.json", n=20):
    """ArXiv Summarization: return (paper, abstract)."""
    with open(path, "r") as f:
        raw = json.load(f)
    return raw[:n]
