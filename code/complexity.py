# 8. complexity.py
# Benchmarks runtime.

import time
import numpy as np

def measure_complexity(n_segments=1000):
    data = ["segment text " + str(i) for i in range(n_segments)]
    start = time.time()
    sorted(data, key=lambda x: len(x))
    end = time.time()
    return {"n": n_segments, "runtime": end-start, "complexity": "O(n log n)"}
