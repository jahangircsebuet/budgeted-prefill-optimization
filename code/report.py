def make_qa_table(results, caption="QA Results (HotpotQA)", label="tab:qa"):
    """
    Convert QA results dict {method: (EM, F1)} into LaTeX table.
    """
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{|l|c|c|}")
    lines.append("\\hline")
    lines.append("\\textbf{Method} & \\textbf{EM (\\%)} & \\textbf{F1 (\\%)} \\\\")
    lines.append("\\hline")

    for method, (em, f1) in results.items():
        lines.append(f"{method:15s} & {em:.1f} & {f1:.1f} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def make_sum_table(results, caption="Summarization Results (GovReport)", label="tab:sum"):
    """
    Convert summarization results dict {method: (R1,R2,RL,BERT)} into LaTeX table.
    """
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{|l|c|c|c|c|}")
    lines.append("\\hline")
    lines.append("\\textbf{Method} & \\textbf{ROUGE-1} & \\textbf{ROUGE-2} & \\textbf{ROUGE-L} & \\textbf{BERTScore} \\\\")
    lines.append("\\hline")

    for method, (r1,r2,rL,bert) in results.items():
        lines.append(f"{method:15s} & {r1:.1f} & {r2:.1f} & {rL:.1f} & {bert:.3f} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def make_ablation_table(results, caption="Ablation Results", label="tab:ablation"):
    """
    results: dict {budget: {method: (EM,F1)}}
    """
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{|c|c|c|c|}")
    lines.append("\\hline")
    lines.append("\\textbf{Budget} & \\textbf{Method} & \\textbf{EM (\\%)} & \\textbf{F1 (\\%)} \\\\")
    lines.append("\\hline")

    for B, res in results.items():
        for method, (em,f1) in res.items():
            lines.append(f"{B:4d} & {method:10s} & {em:.1f} & {f1:.1f} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)

def save_latex_table(content: str, filename: str):
    """Save LaTeX table string into a .tex file."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[INFO] LaTeX table saved to {filename}")
