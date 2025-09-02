# 4. selection.py (budgeted prefill optimization)
from utils import count_tokens

def budgeted_selection(segments, summaries, importance_scores, B):
    """Greedy budget allocation for full vs. summary segments."""
    segment_costs = [count_tokens(seg) for seg in segments]
    summary_costs = [count_tokens(s) for s in summaries]

    utilities_full = importance_scores
    utilities_summary = importance_scores * (np.array(summary_costs) / (np.array(segment_costs) + 1e-6))

    items = [{"id": i, "mode": "summary", "cost": summary_costs[i], "value": utilities_summary[i]} 
             for i in range(len(segments))]

    # Sort by value density
    items = sorted(items, key=lambda x: x["value"]/max(1, x["cost"]), reverse=True)

    chosen, total_cost, total_value = [], 0, 0
    for item in items:
        if total_cost + item["cost"] <= B:
            chosen.append(item)
            total_cost += item["cost"]
            total_value += item["value"]

    return chosen, total_cost, total_value
