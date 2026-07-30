#!/usr/bin/env python3
"""Summarize final-backbone ablation runs for PMF-Net."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

ORDERS = {
    "paper_core": [
        "A0_plain",
        "A1_multitask_fusion",
        "A2_pedagogical_constraints",
        "A3_pmfn_net",
    ],
    "diagnostic_full": [
        "A0_plain",
        "A1_multitask_fusion",
        "D2_data_head",
        "D3_data_head_pedagogical",
        "D4_full_with_optional_data_head",
    ],
    "auto": [
        "A0_plain",
        "A1_multitask_fusion",
        "A2_pedagogical_constraints",
        "A3_pmfn_net",
        "D2_data_head",
        "D3_data_head_pedagogical",
        "D4_full_with_optional_data_head",
        # Backward-compatible names from older script versions.
        "A2_data_head",
        "A3_pedagogical_constraints",
        "A4_full_ours",
    ],
}

DISPLAY = {
    "A0_plain": "A0 Plain backbone",
    "A1_multitask_fusion": "A1 + multi-task fusion",
    "A2_pedagogical_constraints": "A2 + pedagogical prior/constraints",
    "A3_pmfn_net": "A3 PMF-Net",
    "D2_data_head": "D2 + optional data-specific head",
    "D3_data_head_pedagogical": "D3 + optional data head + pedagogical constraints",
    "D4_full_with_optional_data_head": "D4 full with optional data head",
    # Backward-compatible labels.
    "A2_data_head": "A2 + data-specific head",
    "A3_pedagogical_constraints": "A3 + data head + pedagogical constraints",
    "A4_full_ours": "A4 full Ours",
}


def prediction_signature(run_dir: Path) -> str:
    frames = []
    for path in sorted(run_dir.glob("fold*_discuss_predictions.csv")):
        df = pd.read_csv(path)
        if {"fold", "row_id", "true_idx", "pred_idx"}.issubset(df.columns):
            frames.append(df[["fold", "row_id", "true_idx", "pred_idx"]])
    if not frames:
        return ""
    all_df = pd.concat(frames, ignore_index=True).sort_values(["fold", "row_id"])
    return hashlib.sha1(all_df.to_csv(index=False).encode("utf-8")).hexdigest()[:12]


def read_summary(run_dir: Path, name: str) -> dict:
    row = {
        "variant": name,
        "display": DISPLAY.get(name, name),
        "run_dir": str(run_dir),
        "status": "missing_summary",
        "prediction_sha1": prediction_signature(run_dir),
    }
    path = run_dir / "summary.csv"
    if not path.exists():
        return row
    df = pd.read_csv(path)
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


def add_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "macro_f1_mean" not in summary.columns:
        return summary
    summary = summary.copy()
    ok = summary["status"].eq("ok")
    baseline = summary.loc[ok, "macro_f1_mean"].iloc[0] if ok.any() else pd.NA
    summary["delta_macro_f1_vs_prev"] = pd.NA
    summary["delta_macro_f1_vs_plain"] = pd.NA
    previous = None
    for idx, row in summary.iterrows():
        if row.get("status") != "ok" or pd.isna(row.get("macro_f1_mean")):
            continue
        current = float(row["macro_f1_mean"])
        if previous is not None:
            summary.at[idx, "delta_macro_f1_vs_prev"] = current - previous
        if not pd.isna(baseline):
            summary.at[idx, "delta_macro_f1_vs_plain"] = current - float(baseline)
        previous = current
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--mode", default="auto", choices=sorted(ORDERS))
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for name in ORDERS[args.mode]:
        run_dir = root / "runs" / name
        if run_dir.exists():
            rows.append(read_summary(run_dir, name))
    summary = add_deltas(pd.DataFrame(rows))
    out = root / "ablation_summary.csv"
    root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)

    if not summary.empty:
        cols = [c for c in (
            "variant", "display", "status", "acc_mean", "acc_std",
            "macro_f1_mean", "macro_f1_std", "delta_macro_f1_vs_prev",
            "delta_macro_f1_vs_plain", "prediction_sha1",
        ) if c in summary.columns]
        print("\n=== final-backbone ablation summary ===")
        print(summary[cols].to_string(index=False))
    print(f"\n[ablation_report] wrote {out}")


if __name__ == "__main__":
    main()