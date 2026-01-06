"""
UCI HAR (Human Activity Recognition Using Smartphones) + ART-based classifiers.

- Preferred loader: ucimlrepo (UCI ML Repository Python API)
- Fallback loader: download official UCI zip and parse train/test text files

This script mirrors the structure of your LFW/ArcFace pipeline:
    load -> (optional) preprocessing -> thermometer binarize -> artlib.prepare_data -> train/test -> fit/predict -> accuracy

Refs:
- UCI HAR dataset page (id=240): https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones
- ucimlrepo: https://github.com/uci-ml-repo/ucimlrepo
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Tuple, Optional

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Optional: simple preprocessing hooks (kept minimal)
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA

# ART library (same imports you use)
from artlib import (
    FuzzyARTMAP, BinaryFuzzyARTMAP
)
from artlib.optimized.backends.cpp.ART1MAP import ART1MAP

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from scipy import sparse


# ----------------------------
# Configuration
# ----------------------------

RANDOM_STATE = 42

# Thermometer encoding bits per feature:
# - 1 means simple threshold at 0.5 after min-max scaling (binary)
# - >1 expands each feature into n_bits bins ("thermometer code")
N_BITS = 1

TEST_SUBJECT_FRACTION = 0.30

RHO = 0.7


# ----------------------------
# Dataset loading
# ----------------------------

@dataclass
class HARData:
    X: np.ndarray          # (N, D)
    y: np.ndarray          # (N,)
    subject: np.ndarray    # (N,)
    feature_names: Optional[list[str]] = None
    label_names: Optional[dict[int, str]] = None

def load_uci_har(local_zip_path: str = "UCI HAR Dataset.zip") -> HARData:
    """
    Load UCI HAR from a local ZIP if present; otherwise download and load.
    Auto-detects internal prefix by locating train/X_train.txt.
    """
    import zipfile
    import numpy as np


    z = zipfile.ZipFile(local_zip_path, "r")

    names = z.namelist()

    # Find member ending with train/X_train.txt to detect prefix robustly
    xtrain_member = next((n for n in names if n.endswith("train/X_train.txt")), None)
    if xtrain_member is None:
        sample = "\n".join(names[:80])
        raise FileNotFoundError(
            "Could not find a member ending with 'train/X_train.txt' in the ZIP.\n\n"
            f"First entries:\n{sample}"
        )

    prefix = xtrain_member[: -len("train/X_train.txt")]  # e.g. "UCI HAR Dataset/" or "wrapper/UCI HAR Dataset/"

    def read_txt(rel_path: str, dtype=float) -> np.ndarray:
        with z.open(prefix + rel_path) as f:
            return np.loadtxt(f, dtype=dtype)

    X_train = read_txt("train/X_train.txt", dtype=np.float32)
    y_train = read_txt("train/y_train.txt", dtype=np.int32).reshape(-1)
    s_train = read_txt("train/subject_train.txt", dtype=np.int32).reshape(-1)

    X_test = read_txt("test/X_test.txt", dtype=np.float32)
    y_test = read_txt("test/y_test.txt", dtype=np.int32).reshape(-1)
    s_test = read_txt("test/subject_test.txt", dtype=np.int32).reshape(-1)

    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    subject = np.concatenate([s_train, s_test])

    # Feature names
    with z.open(prefix + "features.txt") as f:
        lines = f.read().decode("utf-8").strip().splitlines()
    feature_names = [ln.split(maxsplit=1)[1] for ln in lines]

    label_names = {
        1: "WALKING",
        2: "WALKING_UPSTAIRS",
        3: "WALKING_DOWNSTAIRS",
        4: "SITTING",
        5: "STANDING",
        6: "LAYING",
    }

    return HARData(
        X=X,
        y=y.astype(int),
        subject=subject.astype(int),
        feature_names=feature_names,
        label_names=label_names,
    )


# ----------------------------
# Thermometer encoding (same as your implementation)
# ----------------------------

def binarize_features_thermometer(data: np.ndarray, n_bits: int) -> np.ndarray:
    """Binarizes each feature in the data using thermometer encoding.

    Parameters:
        data: (n, m)
        n_bits: number of thermometer bits per feature (>=1)

    Returns:
        (n, m*n_bits) uint8
    """
    if n_bits <= 0:
        raise ValueError("n_bits must be a positive integer.")

    n, m = data.shape
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)

    ranges = np.where(max_vals - min_vals > 0, max_vals - min_vals, 1.0)
    normalized_data = (data - min_vals) / ranges

    if n_bits == 1:
        return (normalized_data > 0.5).astype(np.uint8)

    quantized_data = np.floor(normalized_data * n_bits).astype(int)
    thermometer_encoded = np.zeros((n, m, n_bits), dtype=np.uint8)

    for i in range(n_bits):
        thermometer_encoded[:, :, i] = (quantized_data > i).astype(np.uint8)

    return thermometer_encoded.reshape(n, m * n_bits)


# ----------------------------
# Optional preprocessing
# ----------------------------


# ----------------------------
# Train/eval helpers (mirrors your functions)
# ----------------------------

def split_data(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Either random stratified split, or subject-holdout split.
    """
    return train_test_split(
        X, y,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y,
        shuffle=True,
    )

def artmap_weight_size(W):
    return sum(w.size for w in W)

def train_FuzzyARTMAP_binary(X: np.ndarray, y: np.ndarray) -> None:
    print("=" * 20)
    print("FuzzyARTMAP -- binary")
    cls = FuzzyARTMAP(rho=RHO, alpha=1e-10, beta=1.0)

    X_bin = binarize_features_thermometer(X, N_BITS)
    X_prep = cls.prepare_data(X_bin)

    X_train, X_test, y_train, y_test = split_data(X_prep, y)
    print("Train/Test:", X_train.shape, X_test.shape)

    t0 = perf_counter()
    cls.fit(X_train, y_train)
    t1 = perf_counter()
    print(f"Training time: {t1 - t0:.3f}s")
    print("Clusters:", cls.module_a.n_clusters)

    y_pred = cls.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # float32 weights and integer map
    mem = 32*artmap_weight_size(cls.module_a.W) + np.ceil(np.log2(np.max(
        y)))*len(cls.map)
    print("Memory Size:", mem)

def train_FuzzyARTMAP_continuous(X: np.ndarray, y: np.ndarray) -> None:
    print("=" * 20)
    print("FuzzyARTMAP -- continuous")
    cls = FuzzyARTMAP(rho=RHO, alpha=1e-10, beta=1.0)

    X_prep = cls.prepare_data(X)

    X_train, X_test, y_train, y_test = split_data(X_prep, y)
    print("Train/Test:", X_train.shape, X_test.shape)

    t0 = perf_counter()
    cls.fit(X_train, y_train)
    t1 = perf_counter()
    print(f"Training time: {t1 - t0:.3f}s")
    print("Clusters:", cls.module_a.n_clusters)

    y_pred = cls.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # float32 weights and integer map
    mem = 32*artmap_weight_size(cls.module_a.W) + np.ceil(np.log2(np.max(y)))*len(cls.map)
    print("Memory Size:", mem)



def train_BinaryFuzzyARTMAP(X: np.ndarray, y: np.ndarray) -> None:
    print("=" * 20)
    print("BinaryFuzzyARTMAP")
    cls = BinaryFuzzyARTMAP(rho=RHO)

    X_bin = binarize_features_thermometer(X, N_BITS)
    X_prep = cls.prepare_data(X_bin)

    X_train, X_test, y_train, y_test = split_data(X_prep, y)
    print("Train/Test:", X_train.shape, X_test.shape)

    t0 = perf_counter()
    cls.fit(X_train, y_train)
    t1 = perf_counter()
    print(f"Training time: {t1 - t0:.3f}s")
    print("Clusters:", cls.module_a.n_clusters)

    y_pred = cls.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # binary weights and integer map
    mem = artmap_weight_size(cls.module_a.W) + np.ceil(np.log2(np.max(y)))*len(cls.map)
    print("Memory Size:", mem)



def train_ART1ARTMAP(X: np.ndarray, y: np.ndarray) -> None:
    print("=" * 20)
    print("ART1MAP")
    cls = ART1MAP(rho=RHO, L=1.0)

    X_bin = binarize_features_thermometer(X, N_BITS)
    X_prep = cls.prepare_data(X_bin).astype(np.int16)

    X_train, X_test, y_train, y_test = split_data(X_prep, y)
    print("Train/Test:", X_train.shape, X_test.shape)

    t0 = perf_counter()
    cls.fit(X_train, y_train)
    t1 = perf_counter()
    print(f"Training time: {t1 - t0:.3f}s")
    print("Clusters:", cls.module_a.n_clusters)

    y_pred = cls.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # float32 weights and integer map
    mem = 32*artmap_weight_size(cls.module_a.W) + np.ceil(np.log2(np.max(y)))*len(cls.map)
    print("Memory Size:", mem)


def _array_bits(a) -> int:
    """Bits used by a numpy array or scipy sparse matrix payload."""
    if a is None:
        return 0

    # scipy sparse: count data + indices + indptr
    if sparse.issparse(a):
        bits = 0
        bits += a.data.size    * a.data.dtype.itemsize * 8
        bits += a.indices.size * a.indices.dtype.itemsize * 8
        bits += a.indptr.size  * a.indptr.dtype.itemsize * 8
        return int(bits)

    a = np.asarray(a)
    return int(a.size * a.dtype.itemsize * 8)


def memory_bits_multinomial_nb(nb) -> int:
    """
    Estimate bits for MultinomialNB learned state.
    Uses the arrays typically present after fit().
    """
    bits = 0
    bits += _array_bits(getattr(nb, "feature_log_prob_", None))
    bits += _array_bits(getattr(nb, "class_log_prior_", None))

    # Counts may or may not be kept depending on version/settings; include if present.
    bits += _array_bits(getattr(nb, "feature_count_", None))
    bits += _array_bits(getattr(nb, "class_count_", None))

    # optional: store class labels (nb.classes_) - small but include for completeness
    bits += _array_bits(getattr(nb, "classes_", None))
    return int(bits)


def memory_bits_sgd_classifier(sgd) -> int:
    """
    Estimate bits for SGDClassifier learned state.
    coef_ + intercept_ dominate.
    """
    bits = 0
    bits += _array_bits(getattr(sgd, "coef_", None))
    bits += _array_bits(getattr(sgd, "intercept_", None))

    # optional: classes_ is small but include
    bits += _array_bits(getattr(sgd, "classes_", None))

    # scalars / ints (tiny). Include if you want:
    # t_ is float, n_iter_ is int; both negligible vs coef_
    for name in ("t_", "n_iter_"):
        if hasattr(sgd, name):
            val = getattr(sgd, name)
            # represent python scalar as 64-bit if you want a conservative bound
            bits += 64 if isinstance(val, (float, np.floating, int, np.integer)) else 0

    return int(bits)


def _as_nonnegative_counts(X: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """
    MultinomialNB expects non-negative "counts" (or at least non-negative features).
    If your X might include negatives (e.g., standardized features), this makes it safe.
    """
    X = np.asarray(X)
    if np.any(X < 0):
        # shift per-feature so min becomes eps (default 0.0)
        mins = X.min(axis=0, keepdims=True)
        X = X - mins + eps
    return X


def train_MultinomialNB_binary(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> None:
    """
    For binary / count-like features (e.g., binarized thermometer features, bag-of-words).
    If you already have a binarize_features_thermometer(), call it before this or adapt below.
    """
    print("=" * 20)
    print("MultinomialNB (binary/count)")
    cls = MultinomialNB(alpha=alpha)

    X_bin = binarize_features_thermometer(X, N_BITS)
    X_prep = X_bin

    # Ensure non-negative (required by MultinomialNB)
    X_prep = _as_nonnegative_counts(X_prep)

    X_train, X_test, y_train, y_test = split_data(X_prep, y)
    print("Train/Test:", X_train.shape, X_test.shape)

    t0 = perf_counter()
    cls.fit(X_train, y_train)  # effectively one pass (sufficient stats)
    t1 = perf_counter()
    print(f"Training time: {t1 - t0:.3f}s")

    y_pred = cls.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    print("Memory Size:", memory_bits_multinomial_nb(cls))


def train_SGDClassifier_binary(
    X: np.ndarray,
    y: np.ndarray,
    loss: str = "log_loss",
    alpha: float = 1e-4,
    random_state: int = 0,
) -> None:
    """
    One-pass SGD via partial_fit on a single batch (train split).
    For binary/count features you typically do NOT need standardization.
    """
    print("=" * 20)
    print(f"SGDClassifier (binary/count, one-pass partial_fit, loss={loss})")

    X_prep = binarize_features_thermometer(X, N_BITS)

    X_train, X_test, y_train, y_test = split_data(X_prep, y)
    print("Train/Test:", X_train.shape, X_test.shape)

    cls = SGDClassifier(loss=loss, alpha=alpha, random_state=random_state)

    classes = np.unique(y_train)

    t0 = perf_counter()
    # single "shot" update over the training set
    cls.partial_fit(X_train, y_train, classes=classes)
    t1 = perf_counter()
    print(f"Training time: {t1 - t0:.3f}s")

    y_pred = cls.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    print("Memory Size:", memory_bits_sgd_classifier(cls))


def train_SGDClassifier_continuous(
    X: np.ndarray,
    y: np.ndarray,
    loss: str = "log_loss",
    alpha: float = 1e-4,
    random_state: int = 0,
) -> None:
    """
    One-pass SGD via partial_fit. For continuous features, scaling is important.
    Note: Fitting a scaler itself requires a pass over training data; we fit on X_train only.
    This does not violate "single pass over training data" if you allow preprocessing to
    use the same pass, but strictly speaking it's an additional computation.
    If you want to be *very* strict, remove scaling and accept likely worse performance.
    """
    print("=" * 20)
    print(f"SGDClassifier (continuous, one-pass partial_fit, loss={loss})")

    X_min = np.min(X)
    X_max = np.max(X)

    # Apply the Min-Max formula
    X_normalized = (X - X_min) / (X_max - X_min)

    X_train, X_test, y_train, y_test = split_data(X_normalized, y)
    print("Train/Test:", X_train.shape, X_test.shape)


    cls = SGDClassifier(loss=loss, alpha=alpha, random_state=random_state)
    classes = np.unique(y_train)

    t0 = perf_counter()
    cls.partial_fit(X_train, y_train, classes=classes)  # single pass update
    t1 = perf_counter()
    print(f"Training time: {t1 - t0:.3f}s")

    y_pred = cls.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    print("Memory Size:", memory_bits_sgd_classifier(cls))

# ----------------------------
# Main
# ----------------------------

def main() -> None:
    # 1) Load UCI HAR
    data = load_uci_har()
    X, y, subject = data.X, data.y, data.subject
    print("n classes:", len(np.unique(y)))
    print(f"X: {X.shape}  y: {y.shape}  subject: {subject.shape}")


    # 2) random split eval across multiple models
    train_FuzzyARTMAP_binary(X, y)
    train_FuzzyARTMAP_continuous(X, y)
    train_BinaryFuzzyARTMAP(X, y)
    train_ART1ARTMAP(X, y)

    train_MultinomialNB_binary(X, y)
    train_SGDClassifier_binary(X, y)
    train_SGDClassifier_continuous(X, y)


if __name__ == "__main__":
    main()
