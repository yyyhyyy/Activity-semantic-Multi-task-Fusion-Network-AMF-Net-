#!/usr/bin/env python3
"""Summarize backbone sweep experiments with the full Ours modules."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

MODEL_LABELS = {
    "ours_s3d": "Ours (S3D)",
    "ours_r3d_18": "Ours (R3D-18)",
    "ours_mc3_18": "Ours (MC3-18)",
    "ours_r2plus1d_18": "Ours (R(2+1)D-18)",
    "ours_swin3d_t": "Ours (Swin3D-T)",
}

ORDER = ["ours_s3d", "ours_r3d_18", "ours_mc3_18", "ours_r2plus1d_18", "ours_swin3d_t"]


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


def read_summary(run_dir: Path, model_name: str) -> dict:
    row = {
        "model": model_name,
        "display": MODEL_LABELS.get(model_name, model_name),
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


def collect(root: Path) -> pd.DataFrame:
    rows = []
    runs = root / "runs"
    if not runs.exists():
        return pd.DataFrame()
    for model_name in ORDER:
        run_dir = runs / model_name
        if not run_dir.exists():
            continue
        rows.append(read_summary(run_dir, model_name))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    summary = collect(root)
    summary_path = root / "backbone_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)

    if not summary.empty:
        cols = [c for c in ("display", "status", "acc_mean", "acc_std", "macro_f1_mean", "macro_f1_std", "prediction_sha1") if c in summary.columns]
        print("\n=== backbone sweep with Ours ===")
        print(summary[cols].to_string(index=False))
    print(f"\n[backbone_sweep] wrote {summary_path}")


if __name__ == "__main__":
    main()
