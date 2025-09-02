# 6. ablations.py
# Runs experiments with multiple budgets (2048, 4000, 8192).

def run_with_budgets(run_func, budgets=[2048,4000,8192]):
    results = {}
    for B in budgets:
        res = run_func(B)
        results[B] = res
    return results
