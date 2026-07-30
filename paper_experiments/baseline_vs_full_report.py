#!/usr/bin/env python3
"""Summarize plain backbone baselines against the proposed full model."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def prediction_signature(run_dir: Path) -> str:
    frames = []
    for path in sorted(run_dir.glob("fold*_discuss_predictions.csv")):
        df = pd.read_csv(path)
        if {"fold", "row_id", "true_idx", "pred_idx"}.issubset(df.columns):
            frames.append(df[["fold", "row_id", "true_idx", "pred_idx"]])
    if not frames:
        return ""
    all_df = pd.concat(frames, ignore_index=True).sort_values(["fold", "row_id"])
    payload = all_df.to_csv(index=False).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def read_discuss_summary(run_dir: Path, group: str, model_name: str) -> dict:
    row = {
        "group": group,
        "model": model_name,
        "run_dir": str(run_dir),
        "status": "missing_summary",
        "prediction_sha1": prediction_signature(run_dir),
    }
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        return row
    df = pd.read_csv(summary_path)
    discuss = df[df["task"].astype(str) == "discuss_type"]
    if discuss.empty:
        row["status"] = "missing_discuss_type"
        return row
    r = discuss.iloc[0]
    row.update({
        "status": "ok",
        "acc_mean": float(r["acc_mean"]),
        "acc_std": float(r["acc_std"]),
        "macro_f1_mean": float(r["mf1_mean"]),
        "macro_f1_std": float(r["mf1_std"]),
        "n_mean": float(r["n_mean"]),
    })
    return row


def read_class_summary(run_dir: Path, group: str, model_name: str) -> pd.DataFrame:
    path = run_dir / "discuss_type_class_summary_strict_primary.csv"
    if not path.exists():
        path = run_dir / "discuss_type_class_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    cols = [
        "discuss_type",
        "support_sum",
        "tp_sum",
        "fp_sum",
        "fn_sum",
        "recall_global",
        "precision_global",
        "f1_global",
        "f1_mean",
        "f1_std",
    ]
    out = df[[c for c in cols if c in df.columns]].copy()
    out.insert(0, "model", model_name)
    out.insert(0, "group", group)
    return out


def collect_runs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    class_frames = []
    for group_dir, group in ((root / "baselines", "plain_baseline"), (root / "full_model", "proposed_full")):
        if not group_dir.exists():
            continue
        for run_dir in sorted(group_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            model_name = run_dir.name
            rows.append(read_discuss_summary(run_dir, group, model_name))
            class_df = read_class_summary(run_dir, group, model_name)
            if not class_df.empty:
                class_frames.append(class_df)
    summary = pd.DataFrame(rows)
    class_summary = pd.concat(class_frames, ignore_index=True) if class_frames else pd.DataFrame()
    return summary, class_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    summary, class_summary = collect_runs(root)

    summary_path = root / "baseline_vs_full_summary.csv"
    class_path = root / "baseline_vs_full_class_summary.csv"
    summary.to_csv(summary_path, index=False)
    class_summary.to_csv(class_path, index=False)

    if not summary.empty:
        display_cols = [
            c for c in (
                "group",
                "model",
                "status",
                "acc_mean",
                "acc_std",
                "macro_f1_mean",
                "macro_f1_std",
                "prediction_sha1",
            )
            if c in summary.columns
        ]
        print("\n=== plain baselines vs proposed full model: discuss_type ===")
        print(summary[display_cols].to_string(index=False))
    print(f"\n[baseline_vs_full] wrote {summary_path}")
    print(f"[baseline_vs_full] wrote {class_path}")


if __name__ == "__main__":
    main()
