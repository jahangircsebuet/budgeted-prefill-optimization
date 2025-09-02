# 2. datasets.py
# Handles dataset loading (HotpotQA, TriviaQA, GovReport, ArXiv).

import json
import random
from datasets import load_dataset

def load_hotpotqa(n=50):
    """
    Load HotpotQA (distractor setting) subset from Hugging Face.
    Returns: list of dicts {query, context, answer}
    """
    dataset = load_dataset("hotpot_qa", "distractor", split=f"validation[:{n}]")
    data = []
    for ex in dataset:
        q = ex["question"]
        # concatenate all context paragraphs
        context = " ".join([" ".join(p) for p in ex["context"]])
        ans = ex["answer"]
        data.append({"query": q, "context": context, "answer": ans})
    return data


def load_triviaqa(n=50):
    """
    Load TriviaQA (Wikipedia setting) subset from Hugging Face.
    Returns: list of dicts {query, context, answer}
    """
    dataset = load_dataset("trivia_qa", "rc", split=f"validation[:{n}]")
    data = []
    for ex in dataset:
        q = ex["question"]
        # concatenate all candidate evidence docs
        context = " ".join(ex["entity_pages"]["wiki_context"]) if "entity_pages" in ex else ""
        ans = ex["answer"]["value"] if "answer" in ex else ""
        data.append({"query": q, "context": context, "answer": ans})
    return data


def load_govreport(n=20):
    """
    Load GovReport Summarization dataset from Hugging Face.
    Returns: list of dicts {report, summary}
    """
    dataset = load_dataset("ccdv/govreport-summarization", split=f"test[:{n}]")
    data = []
    for ex in dataset:
        # fields are "report" and "summary"
        data.append({"report": ex["report"], "summary": ex["summary"]})
    return data


def load_arxiv(n=20):
    """
    Load ArXiv Summarization dataset from Hugging Face.
    Returns: list of dicts {report, summary}
    """
    dataset = load_dataset("ccdv/arxiv-summarization", split=f"test[:{n}]")
    data = []
    for ex in dataset:
        data.append({"report": ex["article"], "summary": ex["abstract"]})
    return data



