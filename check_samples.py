import os, glob
import numpy as np

BASE = "dataset/raw"
LABELS = list("ABCDE")

def check_file(path):
    x = np.load(path)
    x = np.asarray(x)
    if x.shape == (21,3):
        x = x.reshape(-1)
    ok = (x.shape == (63,)) and np.isfinite(x).all()
    if not ok:
        return False, f"shape={x.shape}, finite={np.isfinite(x).all()}"
    lm = x.reshape(21,3)
    wrist_norm = float(np.linalg.norm(lm[0]))
    max_d2 = float(np.max(np.linalg.norm(lm[:,:2], axis=1)))
    return True, f"shape=(63,) wrist_norm={wrist_norm:.2e} max_2d_dist={max_d2:.3f} min={x.min():.3f} max={x.max():.3f}"

def main():
    total = 0
    for lab in LABELS:
        files = sorted(glob.glob(os.path.join(BASE, lab, "*.npy")))
        print(f"\n{lab}: {len(files)} file(s)")
        for f in files:
            total += 1
            ok, msg = check_file(f)
            print(("OK  " if ok else "BAD ") + os.path.basename(f) + " -> " + msg)
    print(f"\nChecked total: {total}")

if __name__ == "__main__":
    main()