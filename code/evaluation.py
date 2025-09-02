# 4. evaluation.py
# Implements metrics (QA: EM/F1, Summarization: ROUGE, BERTScore).

import re
from sklearn.metrics import f1_score
from rouge_score import rouge_scorer
from bert_score import score as bertscore

def normalize_answer(s):
    return re.sub(r'\W+', ' ', s).strip().lower()

def qa_metrics(pred, gold):
    pred, gold = normalize_answer(pred), normalize_answer(gold)
    em = int(pred == gold)
    # F1 = token overlap
    pred_tokens, gold_tokens = pred.split(), gold.split()
    common = len(set(pred_tokens) & set(gold_tokens))
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return em, 0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    f1 = 2*precision*recall / (precision+recall+1e-6)
    return em, f1

def summarization_metrics(pred, ref):
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    rouge = scorer.score(ref, pred)
    P, R, F = bertscore([pred], [ref], lang="en")
    return rouge, F.mean().item()
