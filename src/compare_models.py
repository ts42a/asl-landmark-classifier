# =========================
# compare_models.py
# Train & compare KNN vs SVM vs RF across feature modes (raw/normalized/scaled).
# Produces a results.json you can use in your paper.
# =========================

from __future__ import annotations
import argparse
import json
import os
from typing import Dict, Any, Tuple

import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from utils_dataset import DatasetConfig, load_xy


from pathlib import Path

# Get project root (one folder above src)
ROOT = Path(__file__).resolve().parent.parent

def grid_svm() -> Tuple[Pipeline, Dict[str, Any]]:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC())
    ])
    grid = {
        "clf__kernel": ["rbf"],
        "clf__C": [1, 3, 10, 30],
        "clf__gamma": ["scale", 0.01, 0.03, 0.1],
    }
    return pipe, grid


def grid_knn() -> Tuple[Pipeline, Dict[str, Any]]:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier())
    ])
    grid = {
        "clf__n_neighbors": [3, 5, 7, 9, 11],
        "clf__weights": ["uniform", "distance"],
    }
    return pipe, grid


def grid_rf() -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    clf = RandomForestClassifier(random_state=42)
    grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 12, 20],
        "min_samples_split": [2, 5],
    }
    return clf, grid


def run_one(X, y, model_name: str, seed: int, test_size: float) -> Dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    if model_name == "svm":
        model, grid = grid_svm()
    elif model_name == "knn":
        model, grid = grid_knn()
    elif model_name == "rf":
        model, grid = grid_rf()
    else:
        raise ValueError(model_name)

    search = GridSearchCV(model, grid, scoring="accuracy", cv=5, n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)

    best = search.best_estimator_
    y_pred = best.predict(X_test)

    return {
        "best_params": search.best_params_,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "n_features": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default="dataset")
    ap.add_argument("--read_folder", default="raw", choices=["raw", "normalized", "scaled"])
    ap.add_argument("--add_angles", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--out_dir", default="artifacts")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = DatasetConfig(dataset_root=args.dataset_root, split_seed=args.seed, test_size=args.test_size)

    feature_modes = ["raw", "normalized", "scaled"]
    models = ["knn", "svm", "rf"]

    results: Dict[str, Any] = {
        "read_folder": args.read_folder,
        "add_angles": bool(args.add_angles),
        "seed": args.seed,
        "test_size": args.test_size,
        "runs": {}
    }

    for fm in feature_modes:
        X, y, labels = load_xy(cfg, mode_folder=args.read_folder, feature_mode=fm, add_angles=args.add_angles)
        results["runs"][fm] = {}
        for m in models:
            r = run_one(X, y, m, seed=args.seed, test_size=args.test_size)
            results["runs"][fm][m] = r
            print(f"[{fm}] {m}: acc={r['accuracy']:.4f} macroF1={r['macro_f1']:.4f}")

    results["labels"] = labels

    out_path = os.path.join(args.out_dir, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()