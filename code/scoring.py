# 3. scoring.py (importance scoring + summaries)
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import nltk
import numpy as np

# Load environment variables from .env file
try:
    from load_env import load_dotenv
    load_dotenv()  # Load .env file at import time
except ImportError:
    # Fallback if load_env.py is not available
    pass

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
    """Summarize each segment into shorter form, adapting to input length."""
    summaries = []
    for seg in segments:
        input_len = len(seg.split())

        # If input is very short, skip summarization (use raw segment)
        if input_len < 20:
            summaries.append(seg)
            continue

        # Adaptive max/min
        adaptive_max = min(max_len, int(0.7 * input_len))
        adaptive_min = min_len if adaptive_max > min_len else max(5, int(0.3 * input_len))

        try:
            summary = summarizer(
                seg,
                max_length=adaptive_max,
                min_length=adaptive_min,
                do_sample=False
            )[0]['summary_text']
        except Exception:
            summary = seg[:100]  # fallback safe truncation
        summaries.append(summary)
    return summaries

