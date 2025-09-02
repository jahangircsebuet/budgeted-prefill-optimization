# 7. online_update.py
# Implements a toy online learning (bandit-style adjustment).

import numpy as np

class OnlineScorer:
    def __init__(self, alpha=0.1):
        self.weights = np.array([0.5, 0.5])  # BM25, Embedding
        self.alpha = alpha

    def score(self, bm25_score, emb_score):
        return np.dot(self.weights, np.array([bm25_score, emb_score]))

    def update(self, bm25_score, emb_score, success):
        grad = np.array([bm25_score, emb_score]) * (1 if success else -1)
        self.weights += self.alpha * grad
        self.weights = np.clip(self.weights, 0, 1)
        if self.weights.sum() > 0:
            self.weights /= self.weights.sum()
