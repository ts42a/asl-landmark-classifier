# =========================
# train_static.py
# Train ONE model and save it.
# =========================

from __future__ import annotations
import argparse
import os
import json

import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

import joblib

from utils_dataset import DatasetConfig, load_xy


def build_model(model_name: str):
    model_name = model_name.lower().strip()

    if model_name == "svm":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=False))
        ])
        param_grid = {
            "clf__kernel": ["rbf"],
            "clf__C": [1, 3, 10, 30],
            "clf__gamma": ["scale", 0.01, 0.03, 0.1],
        }
        return pipe, param_grid

    if model_name == "knn":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier())
        ])
        param_grid = {
            "clf__n_neighbors": [3, 5, 7, 9, 11],
            "clf__weights": ["uniform", "distance"],
            "clf__metric": ["minkowski"],
        }
        return pipe, param_grid

    if model_name == "rf":
        # Random Forest doesn't need scaling
        clf = RandomForestClassifier(random_state=42)
        param_grid = {
            "n_estimators": [200, 400],
            "max_depth": [None, 12, 20],
            "min_samples_split": [2, 5],
        }
        return clf, param_grid

    raise ValueError("model must be one of: svm, knn, rf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default="dataset", help="dataset root folder")
    ap.add_argument("--read_folder", default="raw", choices=["raw", "normalized", "scaled"],
                    help="folder to read samples from")
    ap.add_argument("--feature_mode", default="scaled", choices=["raw", "normalized", "scaled"],
                    help="feature transform applied to each sample")
    ap.add_argument("--add_angles", action="store_true", help="append finger angle features")
    ap.add_argument("--model", default="svm", choices=["svm", "knn", "rf"], help="model type")
    ap.add_argument("--out_dir", default="artifacts", help="output directory")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = DatasetConfig(dataset_root=args.dataset_root, split_seed=args.seed, test_size=args.test_size)

    X, y, labels = load_xy(cfg, mode_folder=args.read_folder, feature_mode=args.feature_mode, add_angles=args.add_angles)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    model, param_grid = build_model(args.model)
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)

    best = search.best_estimator_
    y_pred = best.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)

    print("\nBest Params:", search.best_params_)
    print("Test Accuracy:", acc)
    print(classification_report(y_test, y_pred, target_names=labels))

    # Save model + metadata
    model_path = os.path.join(args.out_dir, f"asl_{args.model}_{args.feature_mode}{'_ang' if args.add_angles else ''}.joblib")
    joblib.dump(best, model_path)

    meta = {
        "model": args.model,
        "read_folder": args.read_folder,
        "feature_mode": args.feature_mode,
        "add_angles": bool(args.add_angles),
        "labels": labels,
        "best_params": search.best_params_,
        "test_accuracy": acc,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "seed": args.seed,
        "test_size": args.test_size,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(args.out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved meta/report to: {args.out_dir}")


if __name__ == "__main__":
    main()