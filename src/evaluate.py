# =========================
# evaluate.py
# Load a saved model and evaluate on the full dataset or a holdout split.
# =========================

from __future__ import annotations
import argparse
import json
import os

import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from utils_dataset import DatasetConfig, load_xy

import argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()

ap.add_argument("--dataset_root", default=str(ROOT / "dataset"))
ap.add_argument("--out_dir", default=str(ROOT / "artifacts"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default="dataset")
    ap.add_argument("--read_folder", default="raw", choices=["raw", "normalized", "scaled"])
    ap.add_argument("--feature_mode", default="scaled", choices=["raw", "normalized", "scaled"])
    ap.add_argument("--add_angles", action="store_true")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="artifacts")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = DatasetConfig(dataset_root=args.dataset_root, split_seed=args.seed, test_size=args.test_size)

    X, y, labels = load_xy(cfg, mode_folder=args.read_folder, feature_mode=args.feature_mode, add_angles=args.add_angles)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    model = joblib.load(args.model_path)
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred, target_names=labels))

    report = classification_report(
        y_test,
        y_pred,
        labels=range(len(labels)),
        target_names=labels,
        output_dict=True,
        zero_division=0
    )
    with open(os.path.join(args.out_dir, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "report": report}, f, indent=2)

    print(f"Saved eval report to {os.path.join(args.out_dir, 'eval_report.json')}")


if __name__ == "__main__":
    main()