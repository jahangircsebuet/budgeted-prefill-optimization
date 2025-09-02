# 5. main.py (or run_experiment.py)
# This ties everything together.

from utils import load_text, count_tokens
from segmentation import segment_text
from scoring import compute_importance, summarize_segments
from selection import budgeted_selection

# Step 1: Load input text
raw_text = load_text("sample_long_text.txt")

# Step 2: Segment text
segments = segment_text(raw_text)

# Step 3: Define query (for QA; for summarization use generic)
query = "What is the contribution of this paper?"

# Step 4: Importance scoring + summarization
importance_scores = compute_importance(segments, query)
summaries = summarize_segments(segments)

# Step 5: Run budgeted prefill optimization
budget = 400  # set token budget
selected, cost, val = budgeted_selection(segments, summaries, importance_scores, budget)

# Step 6: Build final prompt
final_prompt = []
for sel in selected:
    if sel["mode"] == "summary":
        final_prompt.append(summaries[sel["id"]])
    else:
        final_prompt.append(segments[sel["id"]])

final_prompt_text = "\n".join(final_prompt)
print("Final prompt length:", count_tokens(final_prompt_text))
print("\n==== Final Prompt ====\n", final_prompt_text[:1000], "...")
