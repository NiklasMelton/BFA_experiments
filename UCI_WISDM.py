from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass
from time import perf_counter
from typing import Optional, Tuple, Dict, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ART library (same imports you use)
from artlib import (
    FuzzyARTMAP, BinaryFuzzyARTMAP, ART1, SimpleARTMAP,
    complement_code, FuzzyART, BinaryFuzzyART
)

# ----------------------------
# Config
# ----------------------------

RANDOM_STATE = 42
CLASS_ORDERED_STREAM = True          # toggle
SHUFFLE_WITHIN_CLASS = True          # recommended

WISDM_RAW_PATH = "WISDM_ar_v1.1_raw.txt"  # put in same directory as script
WISDM_TRANSFORMED_ARFF_PATH = "WISDM_ar_v1.1_transformed.arff"

FS = 20  # Hz (as in the notebook)
FRAME_SECONDS = 4.0
HOP_SECONDS = 2.0
FRAME_SIZE = int(FS * FRAME_SECONDS)   # 80
HOP_SIZE = int(FS * HOP_SECONDS)       # 40

FRAME_REP = "features"   # "features", "rawflat", or "ARFF"

# If using ARFF, this is already 6-class; you can still keep this for consistency:
USE_6_CLASS_SUBSET = True
CLASS6 = ["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"]


# Thermometer encoding bits per feature
N_BITS = 1  # IMPORTANT: 1 is usually too lossy for WISDM

# ----------------------------
# Data container
# ----------------------------

@dataclass
class WISDMFrames:
    X: np.ndarray          # (N, D)
    y: np.ndarray          # (N,)
    subject: np.ndarray    # (N,)


# ----------------------------
# Thermometer encoding
# ----------------------------

def binarize_features_thermometer(data: np.ndarray, n_bits: int) -> np.ndarray:
    if n_bits <= 0:
        raise ValueError("n_bits must be a positive integer.")

    n, m = data.shape
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    ranges = np.where(max_vals - min_vals > 0, max_vals - min_vals, 1.0)
    normalized = (data - min_vals) / ranges

    if n_bits == 1:
        return (normalized > 0.5).astype(np.uint8)

    q = np.floor(normalized * n_bits).astype(int)
    out = np.zeros((n, m, n_bits), dtype=np.uint8)
    for i in range(n_bits):
        out[:, :, i] = (q > i).astype(np.uint8)
    return out.reshape(n, m * n_bits)


# ----------------------------
# WISDM raw parsing (same file format as notebook)
# ----------------------------

def load_wisdm_raw_txt(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parses WISDM_ar_v1.1_raw.txt lines like:
      user,activity,time,x,y,z;
    Returns arrays:
      user(int), activity(str), xyz(float32, shape (N,3))
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing WISDM raw file: {path}")

    users: List[int] = []
    acts: List[str] = []
    xyz: List[List[float]] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split(",")
                if len(parts) < 6:
                    continue

                user = int(float(parts[0]))
                act = parts[1].strip()

                # z often comes with trailing ';'
                z_str = parts[5].split(";")[0].strip()
                if z_str == "":
                    continue

                x = float(parts[3]); y = float(parts[4]); z = float(z_str)
                if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                    continue

                users.append(user)
                acts.append(act)
                xyz.append([x, y, z])
            except Exception:
                # keep going (the notebook does similar try/except) :contentReference[oaicite:4]{index=4}
                continue

    user_arr = np.asarray(users, dtype=np.int32)
    act_arr = np.asarray(acts, dtype=object)
    xyz_arr = np.asarray(xyz, dtype=np.float32)
    return user_arr, act_arr, xyz_arr

def load_wisdm_transformed_arff(arff_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, str], List[str]]:
    """
    Loads WISDM_ar_v1.1_transformed.arff (single file).
    Replaces missing feature values encoded as '?' with the column mean.

    Returns:
      X: (N, 43) float32 engineered features (excludes UNIQUE_ID, user, class)
      y: (N,) int32 class ids
      subject: (N,) int32 user ids
      label_map: {class_id: class_name}
      feature_names: list of 43 feature column names in order
    """
    # --- parse header for attribute names ---
    attr_names: List[str] = []
    in_data = False
    with open(arff_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            if s.lower().startswith("@data"):
                in_data = True
                break
            if s.lower().startswith("@attribute"):
                rest = s[len("@attribute"):].strip()
                if rest.startswith('"'):
                    name = rest.split('"', 2)[1]
                else:
                    name = rest.split(None, 1)[0].split("{", 1)[0]
                attr_names.append(name)

    if not in_data or len(attr_names) != 46:
        raise ValueError(f"Unexpected ARFF structure: found {len(attr_names)} attributes (expected 46).")

    if attr_names[0] != "UNIQUE_ID" or attr_names[1] != "user" or attr_names[-1] != "class":
        raise ValueError(
            f"Unexpected attribute layout: first={attr_names[0]}, second={attr_names[1]}, last={attr_names[-1]}"
        )

    feature_names = attr_names[2:-1]  # 43 features

    # --- parse data ---
    X_rows: List[np.ndarray] = []
    y_str: List[str] = []
    subj: List[int] = []

    total = 0
    missing_count = 0

    with open(arff_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip().lower().startswith("@data"):
                break

        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            parts = s.split(",")
            if len(parts) != 46:
                continue

            total += 1

            # user + class
            try:
                subj_id = int(float(parts[1]))
                cls = str(parts[-1]).strip()
            except Exception:
                continue

            feats = parts[2:-1]  # 43 strings
            row = np.empty((len(feats),), dtype=np.float32)

            for j, v in enumerate(feats):
                vv = v.strip()
                if vv == "?":
                    row[j] = np.nan
                    missing_count += 1
                else:
                    try:
                        row[j] = float(vv)
                    except Exception:
                        row[j] = np.nan
                        missing_count += 1

            X_rows.append(row)
            subj.append(subj_id)
            y_str.append(cls)

    if not X_rows:
        raise RuntimeError("No data rows parsed from ARFF.")

    X = np.vstack(X_rows).astype(np.float32)

    # --- mean imputation per column ---
    col_means = np.nanmean(X, axis=0)  # shape (43,)
    # If a column is entirely NaN (rare), nanmean -> NaN; replace those with 0.0
    col_means = np.where(np.isfinite(col_means), col_means, 0.0).astype(np.float32)

    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    subject = np.asarray(subj, dtype=np.int32)

    # label mapping (stable)
    classes = sorted(set(y_str))
    str_to_id = {c: i for i, c in enumerate(classes)}
    y = np.asarray([str_to_id[c] for c in y_str], dtype=np.int32)
    label_map = {i: c for c, i in str_to_id.items()}

    print(f"[ARFF] Rows: {total}, features: {X.shape[1]}, total missing values imputed: {missing_count}")

    return X, y, subject, label_map, feature_names


def filter_classes(user: np.ndarray, act: np.ndarray, xyz: np.ndarray):
    """
    Keep either 6-class subset or all classes, but DO NOT cap by row order.
    Returns user, y, xyz, label_map
    """
    act_str = act.astype(str)

    if USE_6_CLASS_SUBSET:
        keep_mask = np.isin(act_str, CLASS6)
        user2 = user[keep_mask]
        act2 = act_str[keep_mask]
        xyz2 = xyz[keep_mask]

        label_map = {c: i for i, c in enumerate(CLASS6)}
        y2 = np.asarray([label_map[a] for a in act2], dtype=np.int32)
        return user2, y2, xyz2, label_map

    # all activities encountered
    uniq = sorted(set(act_str.tolist()))
    label_map = {a: i for i, a in enumerate(uniq)}
    y = np.asarray([label_map[a] for a in act_str], dtype=np.int32)
    return user, y, xyz, label_map


def make_frames_per_user_activity(user: np.ndarray, y: np.ndarray, xyz: np.ndarray):
    """
    Standardize xyz globally, then create frames within each (user, activity).
    Label each frame by that activity (no voting needed).
    """
    scaler = StandardScaler()
    xyz_scaled = scaler.fit_transform(xyz).astype(np.float32)

    X_frames, y_frames, s_frames = [], [], []

    for sid in np.unique(user):
        idx_u = np.where(user == sid)[0]
        if idx_u.size < FRAME_SIZE:
            continue

        xu = xyz_scaled[idx_u]
        yu = y[idx_u]

        for cls in np.unique(yu):
            idx_c = np.where(yu == cls)[0]
            if idx_c.size < FRAME_SIZE:
                continue

            xc = xu[idx_c]  # contiguous in-file in many WISDM dumps, but not guaranteed; still safer than mixing classes
            # If your file interleaves activities, you can enforce contiguity by splitting runs; ask me if needed.

            for start in range(0, xc.shape[0] - FRAME_SIZE + 1, HOP_SIZE):
                w = xc[start:start + FRAME_SIZE]  # (frame,3)
                X_frames.append(w)
                y_frames.append(int(cls))
                s_frames.append(int(sid))

    Xw = np.stack(X_frames, axis=0)
    yw = np.asarray(y_frames, dtype=np.int32)
    sw = np.asarray(s_frames, dtype=np.int32)

    if FRAME_REP == "rawflat":
        return WISDMFrames(X=Xw.reshape(Xw.shape[0], -1).astype(np.float32), y=yw, subject=sw)

    # same cheap 24-dim features as before
    def frame_feats(w: np.ndarray) -> np.ndarray:
        eps = 1e-8
        mu = w.mean(axis=0)
        sd = w.std(axis=0)
        mn = w.min(axis=0)
        mx = w.max(axis=0)
        energy = (w ** 2).mean(axis=0)
        maa = np.mean(np.abs(w), axis=0)
        sgn = np.sign(w + eps)
        zcr = (sgn[1:] * sgn[:-1] < 0).mean(axis=0)

        def corr(a,b):
            a0 = a - a.mean()
            b0 = b - b.mean()
            return float((a0*b0).mean() / (a0.std()*b0.std() + eps))

        c_xy = corr(w[:,0], w[:,1])
        c_xz = corr(w[:,0], w[:,2])
        c_yz = corr(w[:,1], w[:,2])

        return np.concatenate([mu, sd, mn, mx, energy, maa, zcr, np.array([c_xy,c_xz,c_yz], np.float32)]).astype(np.float32)

    X_feat = np.stack([frame_feats(w) for w in Xw], axis=0)
    return WISDMFrames(X=X_feat, y=yw, subject=sw)

def load_wisdm_representation():
    """
    Returns X, y, subject, label_map
    """
    if FRAME_REP == "ARFF":
        X, y, subject, label_map, feature_names = load_wisdm_transformed_arff(WISDM_TRANSFORMED_ARFF_PATH)
        print(f"[ARFF] Loaded: X={X.shape} classes={len(np.unique(y))} subjects={len(np.unique(subject))}")
        print("[ARFF] Label map:", label_map)
        return X, y, subject, label_map

    # Otherwise fall back to your existing RAW pipeline:
    # - load_wisdm_raw_txt(...)
    # - filter_classes(...)
    # - make_frames_per_user_activity(...) OR rawflat
    user, act, xyz = load_wisdm_raw_txt(WISDM_RAW_PATH)
    user, y, xyz, label_map = filter_classes(user, act, xyz)

    if FRAME_REP == "rawflat":
        frames = make_frames_per_user_activity(user, y, xyz)  # ensure this respects FRAME_REP internally, or call rawflat explicitly
        return frames.X, frames.y, frames.subject, label_map

    # "features"
    frames = make_frames_per_user_activity(user, y, xyz)
    return frames.X, frames.y, frames.subject, label_map




# ----------------------------
# Experiments
# ----------------------------

def split_data(X: np.ndarray, y: np.ndarray, subject: np.ndarray, subject_holdout: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not subject_holdout:
        return train_test_split(X, y, test_size=0.5, random_state=RANDOM_STATE, stratify=y, shuffle=True)

    rng = np.random.default_rng(RANDOM_STATE)
    subs = np.unique(subject)
    n_test = max(1, int(round(0.3 * subs.size)))
    test_subs = set(rng.choice(subs, size=n_test, replace=False).tolist())
    mask = np.array([s in test_subs for s in subject], dtype=bool)
    return X[~mask], X[mask], y[~mask], y[mask]


def run_model(name: str, cls, X: np.ndarray, y: np.ndarray, subject: np.ndarray) -> None:
    print("=" * 30)
    print(name)

    X_bin = binarize_features_thermometer(X, N_BITS)
    X_bin = complement_code(X_bin)  # helps a lot for binned real-valued inputs

    X_prep = cls.prepare_data(X_bin)

    Xtr, Xte, ytr, yte = split_data(X_prep, y, subject, subject_holdout=False)
    print("Train/Test:", Xtr.shape, Xte.shape)

    t0 = perf_counter()
    cls.fit(Xtr, ytr)
    t1 = perf_counter()

    yhat = cls.predict(Xte)
    acc = accuracy_score(yte, yhat)
    print(f"Accuracy: {acc:.4f} | Train time: {t1 - t0:.3f}s")
    try:
        print("Clusters:", cls.module_a.n_clusters)
    except Exception:
        pass

def order_by_class_sequence(X, y, class_order, subject=None, shuffle_within_class=True, seed=42):
    rng = np.random.default_rng(seed)
    idx_all = []
    for c in class_order:
        idx = np.where(y == c)[0]
        if shuffle_within_class:
            rng.shuffle(idx)
        idx_all.append(idx)
    order = np.concatenate(idx_all)
    Xo, yo = X[order], y[order]
    so = None if subject is None else np.asarray(subject)[order]
    return Xo, yo, so


def main():
    X, y, subject, label_map = load_wisdm_representation()
    if CLASS_ORDERED_STREAM:
        rng = np.random.default_rng(RANDOM_STATE)
        order_idx = []
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            if SHUFFLE_WITHIN_CLASS:
                rng.shuffle(idx)
            order_idx.append(idx)
        order_idx = np.concatenate(order_idx)

        X = X[order_idx]
        y = y[order_idx]
        subject = subject[order_idx] if subject is not None else None

    # (Optional but recommended) quick sanity print
    print("Final dataset:", "X:", X.shape, "y:", y.shape,
          "classes:", len(np.unique(y)), "subjects:", len(np.unique(subject)))

    # Then run your same ART experiments:
    run_model("BinaryFuzzyARTMAP", BinaryFuzzyARTMAP(rho=0.0), X, y, subject)
    run_model("FuzzyARTMAP", FuzzyARTMAP(rho=0.0, alpha=1e-10, beta=1.0), X, y, subject)
    run_model("BinaryFuzzyARTMAP -- python", SimpleARTMAP(BinaryFuzzyART(rho=0.0)),
              X, y, subject)
    run_model("FuzzyARTMAP -- python", SimpleARTMAP(FuzzyART(rho=0.0, alpha=1e-10,
                                                             beta=1.0)), X, y, subject)

    # ART1 path (bool + complement)
    print("=" * 30)
    print("ART1 + ARTMAP")
    cls = SimpleARTMAP(ART1(rho=0.0, L=1.0))
    X_bin = binarize_features_thermometer(X, N_BITS)
    X_cc = complement_code(X_bin)
    X_prep = cls.prepare_data(X_cc).astype(bool)

    Xtr, Xte, ytr, yte = split_data(X_prep, y, subject, subject_holdout=False)
    t0 = perf_counter()
    cls.fit(Xtr, ytr)
    t1 = perf_counter()
    acc = accuracy_score(yte, cls.predict(Xte))
    print(f"Accuracy: {acc:.4f} | Train time: {t1 - t0:.3f}s | Clusters: {cls.module_a.n_clusters}")


if __name__ == "__main__":
    main()
