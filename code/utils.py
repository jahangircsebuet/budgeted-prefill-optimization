# 1. utils.py (basic helpers)
import time
import nltk

def count_tokens(text: str) -> int:
    return len(text.split())

def load_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

