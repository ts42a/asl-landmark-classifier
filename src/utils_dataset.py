# =========================
# utils_dataset.py
# (helper module used by all scripts)
# Save this file next to your .py scripts.
# =========================

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass
class DatasetConfig:
    dataset_root: str = "dataset"  # contains raw/A/*.npy etc
    split_seed: int = 42
    test_size: float = 0.2
    allowed_exts: Tuple[str, ...] = (".npy",)


def ensure_63d(vec: np.ndarray) -> np.ndarray:
    """
    Accepts:
      - (63,) flat vector (x,y,z)*21
      - (21,3) landmarks
    Returns:
      - (63,) float32
    """
    vec = np.asarray(vec)
    if vec.shape == (63,):
        out = vec
    elif vec.shape == (21, 3):
        out = vec.reshape(-1)
    else:
        raise ValueError(f"Expected (63,) or (21,3), got {vec.shape}")
    return out.astype(np.float32, copy=False)


def to_landmarks_21x3(vec63: np.ndarray) -> np.ndarray:
    v = ensure_63d(vec63)
    return v.reshape(21, 3)


def normalize_wrist(vec63: np.ndarray, wrist_index: int = 0) -> np.ndarray:
    """
    Translation normalization: subtract wrist landmark.
    """
    lm = to_landmarks_21x3(vec63)
    wrist = lm[wrist_index].copy()
    lm = lm - wrist
    return lm.reshape(-1).astype(np.float32, copy=False)


def normalize_wrist_and_scale(vec63: np.ndarray, wrist_index: int = 0, eps: float = 1e-6) -> np.ndarray:
    """
    Translation + scale normalization:
      1) subtract wrist
      2) divide by max distance from origin
    """
    lm = to_landmarks_21x3(vec63)
    lm = lm - lm[wrist_index]
    d = np.linalg.norm(lm, axis=1)
    scale = float(np.max(d))
    if scale < eps:
        scale = 1.0
    lm = lm / scale
    return lm.reshape(-1).astype(np.float32, copy=False)


def compute_finger_angles(vec63: np.ndarray) -> np.ndarray:
    """
    Optional feature engineering:
    Returns a small angle feature vector (float32).
    Uses triplets along each finger to compute angles.
    Angles help with confusions like M/N, U/V, etc.

    This is safe for static ASL letters.
    """
    lm = to_landmarks_21x3(vec63)

    # MediaPipe hand landmark indices (commonly used)
    # Thumb: 1,2,3,4
    # Index: 5,6,7,8
    # Middle: 9,10,11,12
    # Ring: 13,14,15,16
    # Pinky: 17,18,19,20
    chains = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20],
    ]

    def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-9) -> float:
        # angle ABC at point B
        ba = a - b
        bc = c - b
        nba = np.linalg.norm(ba) + eps
        nbc = np.linalg.norm(bc) + eps
        cosv = float(np.dot(ba, bc) / (nba * nbc))
        cosv = max(-1.0, min(1.0, cosv))
        return float(np.arccos(cosv))

    angles: List[float] = []
    for ch in chains:
        # two joint angles per finger: at joint 2 and 3 (e.g., 1-2-3 and 2-3-4)
        angles.append(angle(lm[ch[0]], lm[ch[1]], lm[ch[2]]))
        angles.append(angle(lm[ch[1]], lm[ch[2]], lm[ch[3]]))

    return np.array(angles, dtype=np.float32)


def apply_feature_mode(vec63: np.ndarray, mode: str, add_angles: bool = False) -> np.ndarray:
    """
    mode: 'raw' | 'normalized' | 'scaled'
    add_angles: if True, append angle features to the 63D vector
    """
    mode = mode.lower().strip()
    if mode == "raw":
        x = ensure_63d(vec63)
    elif mode == "normalized":
        x = normalize_wrist(vec63)
    elif mode == "scaled":
        x = normalize_wrist_and_scale(vec63)
    else:
        raise ValueError("mode must be one of: raw, normalized, scaled")

    if add_angles:
        ang = compute_finger_angles(x)  # use whatever x currently is
        x = np.concatenate([x, ang], axis=0).astype(np.float32, copy=False)

    return x


def list_samples(dataset_root: str, mode: str) -> List[Tuple[str, str]]:
    """
    Returns list of (filepath, label) from dataset_root/mode/<LABEL>/*.npy
    Example: dataset/raw/A/xxx.npy
    """
    base = os.path.join(dataset_root, mode)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Missing folder: {base}")

    samples: List[Tuple[str, str]] = []
    for lab in LABELS:
        lab_dir = os.path.join(base, lab)
        if not os.path.isdir(lab_dir):
            continue
        for fn in os.listdir(lab_dir):
            if fn.lower().endswith(".npy"):
                samples.append((os.path.join(lab_dir, fn), lab))
    if not samples:
        raise RuntimeError(f"No .npy samples found under: {base}/<LABEL>/")
    return samples


def load_xy(cfg: DatasetConfig, mode_folder: str, feature_mode: str, add_angles: bool = False):
    """
    Loads samples from dataset/<mode_folder>/<LABEL>/*.npy
    Returns X, y, labels_present where labels_present matches the actual folders found.
    """
    samples = list_samples(cfg.dataset_root, mode_folder)

    # ✅ Use only labels that actually exist in the dataset
    labels_present = sorted({lab for _, lab in samples})
    label_to_id = {lab: i for i, lab in enumerate(labels_present)}

    X_list = []
    y_list = []

    for path, lab in samples:
        arr = np.load(path)
        feat = apply_feature_mode(arr, feature_mode, add_angles=add_angles)
        X_list.append(feat)
        y_list.append(label_to_id[lab])

    X = np.vstack([x.reshape(1, -1) for x in X_list]).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, labels_present