# =========================
# plot_confusion.py
# Plot confusion matrix from a trained model (matplotlib only, no seaborn).
# =========================

from __future__ import annotations
import argparse
import os
import json

import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
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

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(labels))))

    fig = plt.figure(figsize=(12, 10))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest")
    plt.colorbar(im, ax=ax)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    # Write counts (optional but useful)
    thresh = cm.max() * 0.6 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val == 0:
                continue
            ax.text(j, i, str(val), ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=7)

    plt.tight_layout()

    out_png = os.path.join(args.out_dir, "confusion_matrix.png")
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

    # save raw matrix too
    out_npy = os.path.join(args.out_dir, "confusion_matrix.npy")
    np.save(out_npy, cm)

    print(f"Saved {out_png}")
    print(f"Saved {out_npy}")


if __name__ == "__main__":
    main()