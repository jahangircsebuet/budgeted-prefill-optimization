# 3. scoring.py (importance scoring + summaries)
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import nltk
import numpy as np

# Initialize models
bm25_tokenizer = nltk.word_tokenize
embedder = SentenceTransformer("all-MiniLM-L6-v2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def compute_importance(segments, query):
    """Compute importance scores with BM25 + embeddings."""
    tokenized_segments = [bm25_tokenizer(seg.lower()) for seg in segments]
    bm25 = BM25Okapi(tokenized_segments)
    bm25_scores = bm25.get_scores(bm25_tokenizer(query.lower()))

    seg_emb = embedder.encode(segments, convert_to_tensor=True)
    query_emb = embedder.encode(query, convert_to_tensor=True)
    cosine_scores = (seg_emb @ query_emb) / (seg_emb.norm(dim=-1) * query_emb.norm())

    return 0.5 * np.array(bm25_scores) + 0.5 * cosine_scores.cpu().numpy()

def summarize_segments(segments, max_len=50, min_len=10):
    """Summarize each segment into shorter form."""
    summaries = []
    for seg in segments:
        try:
            summary = summarizer(seg, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
        except Exception:
            summary = seg[:100]  # fallback
        summaries.append(summary)
    return summaries
