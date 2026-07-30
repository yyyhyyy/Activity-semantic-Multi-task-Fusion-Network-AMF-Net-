# -*- coding: utf-8 -*-
"""从评估输出生成论文常用图表（PNG/LaTeX/汇总表）。

输入目录（默认 outputs/eval_video/ 或 outputs/eval/）需要包含：
- summary.csv
- confusion_matrix_<task>.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def _latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    # 生成简单可直接粘贴论文的 LaTeX 表
    cols = ["task", "accuracy", "macro_f1", "n_samples"]
    d = df[cols].copy()
    d["accuracy"] = d["accuracy"].map(lambda x: f"{x:.4f}")
    d["macro_f1"] = d["macro_f1"].map(lambda x: f"{x:.4f}")
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\hline")
    lines.append("Task & Acc. & Macro-F1 & N \\\\")
    lines.append("\\hline")
    for _, r in d.iterrows():
        lines.append(f"{r['task']} & {r['accuracy']} & {r['macro_f1']} & {int(r['n_samples'])} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", type=str, default="./outputs/eval_video", help="eval.py 或 eval_video.py 的输出目录")
    ap.add_argument("--out_dir", type=str, default="", help="图表输出目录（默认 eval_dir/report）")
    ap.add_argument("--title", type=str, default="Classroom behavior recognition results", help="表格/图标题")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        raise FileNotFoundError(eval_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (eval_dir / "report")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = eval_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    df = pd.read_csv(summary_path)
    df = df.sort_values("task")

    # 保存更“论文友好”的汇总
    df.to_csv(out_dir / "summary_sorted.csv", index=False)

    # LaTeX 表
    tex = _latex_table(df, caption=args.title, label="tab:classroom_results")
    (out_dir / "results_table.tex").write_text(tex, encoding="utf-8")

    # 混淆矩阵热力图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm_files = sorted(eval_dir.glob("confusion_matrix_*.csv"))
    for cm_file in cm_files:
        task = cm_file.stem.replace("confusion_matrix_", "")
        cm_df = pd.read_csv(cm_file, index_col=0)
        cm = cm_df.values.astype(np.int64)
        # 归一化（按行）
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, np.maximum(row_sum, 1), dtype=np.float32)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=False, cmap="Blues", xticklabels=cm_df.columns, yticklabels=cm_df.index)
        plt.title(f"{args.title} - {task} (row-normalized)")
        plt.xlabel("Pred")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(out_dir / f"cm_{task}.png", dpi=300)
        plt.close()

    print("报告已生成到:", out_dir)
    print(" -", out_dir / "results_table.tex")
    print(" -", out_dir / "summary_sorted.csv")
    print(" -", out_dir / "cm_<task>.png")


if __name__ == "__main__":
    main()

