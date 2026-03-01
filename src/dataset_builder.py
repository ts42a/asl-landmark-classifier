# dataset_builder.py
# Research-first STATIC dataset builder for ASL Alphabet (A–Z)

import cv2
import os
import json
import time
import urllib.request
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision
except ImportError:
    raise SystemExit("Run: pip install mediapipe opencv-python numpy")


from pathlib import Path

# Get project root (one folder above src)
ROOT = Path(__file__).resolve().parent.parent

# ---------------- CONFIG ----------------
BASE_DIR = "dataset"                 # ✅ match your research repo structure
RAW_DIR = os.path.join(BASE_DIR, "raw")
META_FILE = os.path.join(BASE_DIR, "metadata.jsonl")

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

COUNTDOWN_SECONDS = 3
STATIC_CAPTURE_SECONDS = 3.0         # ✅ required by you
STABLE_VAR_THRESHOLD = 1e-4
STABILITY_WINDOW = 6                 # rolling window for stability
MIN_STABLE_FRAMES = 8

# Default: research = keep many
DEFAULT_SAVE_MODE = "all"            # "all" or "best3"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------- Helpers ----------
def get_hand_model_path():
    path = os.path.join(MODEL_DIR, "hand_landmarker.task")
    if not os.path.exists(path):
        print("Downloading MediaPipe hand_landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, path)
        print("Done.")
    return path


def sanitize_label(name: str) -> str:
    name = name.strip().upper()
    if len(name) == 1 and name.isalpha():
        return name
    raise ValueError("Label must be a single letter A–Z for this paper dataset.")


def ensure_label_folder(label: str):
    folder = os.path.join(RAW_DIR, label)
    os.makedirs(folder, exist_ok=True)
    return folder


def extract_hand_features(hand_landmarks) -> np.ndarray:
    """
    Returns (63,) float32:
    - wrist-centered
    - scale-normalized (2D scale)
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)

    # wrist-center
    pts -= pts[0:1, :]

    # scale-normalize using max 2D distance from wrist
    d = np.linalg.norm(pts[:, :2], axis=1)
    s = float(np.max(d)) if np.max(d) > 1e-6 else 1.0
    pts /= s

    return pts.reshape(-1).astype(np.float32)  # 63


def create_detector(num_hands=1):
    model_path = get_hand_model_path()
    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=num_hands,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return vision.HandLandmarker.create_from_options(options)


def draw_text(img, lines, x=10, y=30, gap=30):
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + i * gap),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def pick_best_k_by_centroid(samples: list[np.ndarray], k: int):
    X = np.stack(samples, axis=0)
    centroid = X.mean(axis=0)
    d = np.linalg.norm(X - centroid, axis=1)
    idx = np.argsort(d)[:k]
    return [samples[i] for i in idx]


def write_meta(record: dict):
    # JSON Lines format: one record per capture session
    with open(META_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------- Core capture (STATIC) ----------
def capture_static(label: str, save_mode: str = DEFAULT_SAVE_MODE):
    """
    save_mode:
      - "all": save all stable frames (recommended for research)
      - "best3": save only best 3 most consistent frames (demo only)
    """
    label = sanitize_label(label)
    folder = ensure_label_folder(label)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not opened. Try changing VideoCapture(0) to VideoCapture(1).")

    detector = create_detector(num_hands=1)
    mp_drawing = vision.drawing_utils
    mp_styles = vision.drawing_styles
    mp_connections = vision.HandLandmarksConnections

    win = f"STATIC dataset capture: {label}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # Countdown
    end = time.time() + COUNTDOWN_SECONDS
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rem = int(np.ceil(end - time.time()))
        if rem <= 0:
            break

        draw_text(frame, [f"LABEL: {label}", f"Ready in... {rem}", "Hold steady", "ESC/Q cancel"])
        cv2.imshow(win, frame)
        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            cap.release()
            cv2.destroyAllWindows()
            if hasattr(detector, "close"): detector.close()
            print("[CANCELED]")
            return

    # Capture window
    recent = []
    stable_samples = []
    detected_frames = 0

    start_t = time.time()
    end = start_t + STATIC_CAPTURE_SECONDS

    while time.time() < end:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect(mp_image)

        detected, stable = False, False
        if res.hand_landmarks:
            detected = True
            detected_frames += 1
            hand = res.hand_landmarks[0]

            mp_drawing.draw_landmarks(
                frame, hand, mp_connections.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            vec = extract_hand_features(hand)
            recent.append(vec)
            if len(recent) > STABILITY_WINDOW:
                recent.pop(0)

            if len(recent) >= 3:
                var = np.var(np.stack(recent), axis=0).mean()
                stable = var < STABLE_VAR_THRESHOLD

            if stable:
                stable_samples.append(vec)
        else:
            recent.clear()

        draw_text(frame, [
            f"ASL STATIC: {label}",
            f"Stable saved candidates: {len(stable_samples)}",
            f"Detected frames: {detected_frames}",
            f"Stable now: {'YES' if stable else 'NO'}",
            "ESC/Q cancel"
        ])
        cv2.imshow(win, frame)
        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    if hasattr(detector, "close"): detector.close()

    if len(stable_samples) < MIN_STABLE_FRAMES:
        print("[WARN] Too few stable frames. Try better lighting + keep hand steady.")
        return

    # Choose what to save
    if save_mode == "best3":
        chosen = pick_best_k_by_centroid(stable_samples, 3)
    else:
        chosen = stable_samples  # ✅ research: keep everything stable

    # Save files with session id
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_paths = []

    for i, vec in enumerate(chosen, start=1):
        fn = f"{label}_{session_id}_{i:04d}.npy"
        out = os.path.join(folder, fn)
        np.save(out, vec)
        saved_paths.append(out)

    # Write metadata (very useful for paper!)
    meta = {
        "session_id": session_id,
        "label": label,
        "capture_seconds": STATIC_CAPTURE_SECONDS,
        "save_mode": save_mode,
        "stable_candidates": len(stable_samples),
        "saved_count": len(chosen),
        "timestamp": session_id,
        "notes": "wrist+scale normalized (63D) from MediaPipe hand landmarks"
    }
    write_meta(meta)

    print(f"[OK] Saved {len(chosen)} samples for '{label}' in {folder}")
    print(f"     Metadata appended to: {META_FILE}")


def main():
    while True:
        print("\n=== ASL A–Z STATIC DATASET BUILDER (Research Mode) ===")
        print("1) Capture STATIC (save ALL stable frames) ✅ (recommended)")
        print("2) Capture STATIC (save best 3 only) (demo)")
        print("3) Exit")

        c = input("Select: ").strip()
        if c == "1":
            lb = input("Letter (A–Z): ").strip()
            capture_static(lb, save_mode="all")
        elif c == "2":
            lb = input("Letter (A–Z): ").strip()
            capture_static(lb, save_mode="best3")
        elif c == "3":
            break
        else:
            print("Invalid.")


if __name__ == "__main__":
    main()