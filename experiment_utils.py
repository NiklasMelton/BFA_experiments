from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict, Any, Tuple, List

import os, sys
from contextlib import contextmanager

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, adjusted_rand_score, adjusted_mutual_info_score
)
from scipy import sparse

from sklearn.datasets import fetch_lfw_people
from insightface.app import FaceAnalysis

from artlib import FuzzyARTMAP, BinaryFuzzyARTMAP
from artlib.optimized.backends.cpp.ART1MAP import ART1MAP
from artlib.optimized.backends.cpp.FuzzyART import FuzzyART
from artlib.optimized.backends.cpp.BinaryFuzzyART import BinaryFuzzyART
from artlib.optimized.backends.cpp.ART1 import ART1


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier



@dataclass
class DatasetBundle:
    name: str
    X: np.ndarray
    y: np.ndarray
    meta: Dict[str, Any]

# ----------------------------
# stdout suppression
# ----------------------------
@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = devnull, devnull
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err

# ----------------------------
# UCI HAR loader
# ----------------------------

def load_uci_har_zip(local_zip_path: str = "UCI HAR Dataset.zip") -> DatasetBundle:
    """
    Load UCI HAR from a local ZIP. (Your existing loader logic, repackaged.)
    """
    import zipfile

    z = zipfile.ZipFile(local_zip_path, "r")
    names = z.namelist()

    xtrain_member = next((n for n in names if n.endswith("train/X_train.txt")), None)
    if xtrain_member is None:
        sample = "\n".join(names[:80])
        raise FileNotFoundError(
            "Could not find a member ending with 'train/X_train.txt' in the ZIP.\n\n"
            f"First entries:\n{sample}"
        )

    prefix = xtrain_member[: -len("train/X_train.txt")]

    def read_txt(rel_path: str, dtype=float) -> np.ndarray:
        with z.open(prefix + rel_path) as f:
            return np.loadtxt(f, dtype=dtype)

    X_train = read_txt("train/X_train.txt", dtype=np.float32)
    y_train = read_txt("train/y_train.txt", dtype=np.int32).reshape(-1)

    X_test = read_txt("test/X_test.txt", dtype=np.float32)
    y_test = read_txt("test/y_test.txt", dtype=np.int32).reshape(-1)

    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test]).astype(int)

    return DatasetBundle(
        name="uci_har",
        X=X,
        y=y,
        meta={
            "source": "zip",
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": int(len(np.unique(y))),
        },
    )


def load_uci_via_api_stub(dataset_id: int, dataset_name: str) -> DatasetBundle:
    """
    Load additional UCI datasets via python API.

    Recommended: ucimlrepo (pip install ucimlrepo)
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except Exception as e:
        raise RuntimeError(
            "ucimlrepo is not installed. Install it with: pip install ucimlrepo"
        ) from e

    ds = fetch_ucirepo(id=dataset_id)
    # Typical ucimlrepo structure: ds.data.features (pandas), ds.data.targets (pandas)
    X_df = ds.data.features
    y_df = ds.data.targets

    # Make y a 1D array if possible
    if hasattr(y_df, "shape") and len(y_df.shape) == 2 and y_df.shape[1] == 1:
        y_arr = y_df.iloc[:, 0].to_numpy()
    else:
        y_arr = np.asarray(y_df)

    X = X_df.to_numpy().astype(np.float32, copy=False)

    # Encode non-numeric labels if needed
    if y_arr.dtype.kind in {"U", "S", "O"}:
        _, y_enc = np.unique(y_arr, return_inverse=True)
        y = y_enc.astype(int)
    else:
        y = y_arr.astype(int, copy=False)

    return DatasetBundle(
        name=dataset_name,
        X=X,
        y=y,
        meta={
            "source": "ucimlrepo",
            "uci_id": int(dataset_id),
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": int(len(np.unique(y))),
        },
    )



# ----------------------------
# LFW / RetinaFace / InsightFace helpers
# ----------------------------
def load_lfw_subset(
    rng: np.random.Generator,
):
    """
    Fetch LFW (color images) and sample a subset of identities and images.

    Returns:
        images: np.ndarray (N, H, W, 3), float32, values in [0, 255]
        labels: np.ndarray (N,), reindexed labels in [0, num_ids_selected-1]
        label_names: list of original person names for each compact label
    """
    lfw = fetch_lfw_people(
        min_faces_per_person=2,
        resize=0.5,
        color=True,
        funneled=True,
    )

    images = lfw.images  # float32, usually in [0, 1]
    targets = lfw.target
    target_names = lfw.target_names
    print("Total Images:", len(targets))

    # Rescale to [0, 255]
    if images.max() <= 1.0:
        images = images * 255.0
    images = images.astype(np.float32)

    unique_labels = np.unique(targets)
    chosen = unique_labels

    label_map = {orig: new for new, orig in enumerate(chosen)}
    label_names = [target_names[orig] for orig in chosen]

    sampled_images: List[np.ndarray] = []
    sampled_labels: List[int] = []

    for orig_label in chosen:
        idxs = np.where(targets == orig_label)[0]
        if len(idxs) == 0:
            continue
        for idx in idxs:
            sampled_images.append(images[idx])
            sampled_labels.append(label_map[orig_label])

    images_array = np.stack(sampled_images, axis=0)
    labels_array = np.array(sampled_labels, dtype=int)

    return images_array, labels_array, label_names



def init_retinaface(
    lightweight: bool = False,
    ctx_id: int = 0,
    det_size: Tuple[int, int] | None = None,
) -> FaceAnalysis:
    """
    Initialize an InsightFace FaceAnalysis app with RetinaFace.

    Parameters
    ----------
    lightweight : bool, default=False
        If False, use the heavier, higher-accuracy model pack ('buffalo_l').
        If True, use a lighter model pack ('buffalo_s') that is more
        mobile/embedded-friendly.

    ctx_id : int, default=0
        Context ID for InsightFace:
            0  -> first GPU (good for Colab with GPU)
           -1  -> CPU only

    det_size : (int, int) or None, default=None
        Detection input size. If None, a sensible default is chosen depending
        on `lightweight`. You can override it manually if needed.

    Returns
    -------
    app : FaceAnalysis
        Configured InsightFace FaceAnalysis application.
    """
    if lightweight:
        model_name = "buffalo_s"   # smaller, faster model pack
        default_det_size = (192, 192)
    else:
        model_name = "buffalo_l"   # larger & more accurate
        default_det_size = (256, 256)

    if det_size is None:
        det_size = default_det_size

    app = FaceAnalysis(
        name=model_name,
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app



def prepare_for_detection(img_rgb: np.ndarray) -> np.ndarray:
    """
    Given an RGB float32 image in [0, 255], make a detector-friendly BGR uint8 image:
      - Ensure uint8
      - Scale up so longest side >= MIN_LONG_SIDE
      - Add black border around
    """
    MIN_LONG_SIDE = 256
    BORDER = 32

    img_uint8 = np.clip(img_rgb, 0, 255).astype(np.uint8)

    h, w, _ = img_uint8.shape
    long_side = max(h, w)
    scale = 1.0
    if long_side < MIN_LONG_SIDE:
        scale = MIN_LONG_SIDE / float(long_side)

    if scale != 1.0:
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img_uint8 = cv2.resize(img_uint8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Convert RGB -> BGR
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

    # Add border so the face isn't touching the frame
    img_bgr = cv2.copyMakeBorder(
        img_bgr,
        BORDER,
        BORDER,
        BORDER,
        BORDER,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return img_bgr


def compute_embeddings_from_lfw(
    app: FaceAnalysis,
    images_rgb: np.ndarray,  # (N, H, W, 3), float32 [0, 255]
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run RetinaFace on a batch of RGB images.

    Returns:
        embeddings: (M, D)
        labels_out: (M,)
    """
    embeddings: List[np.ndarray] = []
    labels_out: List[int] = []

    for img_rgb, label in zip(images_rgb, labels):
        img_bgr = prepare_for_detection(img_rgb)

        faces = app.get(img_bgr)
        if not faces:
            # Fallback: try with a slightly different size (optional tweak)
            faces = app.get(img_bgr, max_num=0)

        if not faces:
            print("[WARN] No face detected in one sample, skipping.")
            continue

        best_face = max(faces, key=lambda f: f.det_score)
        emb = best_face.normed_embedding.astype("float32")

        embeddings.append(emb)
        labels_out.append(label)

    if not embeddings:
        raise RuntimeError("No embeddings computed. Detection failed for all images.")

    X = np.stack(embeddings, axis=0)
    y = np.array(labels_out, dtype=int)
    return X, y


def filter_min_class_count(X, y, min_count=2):
    y = np.asarray(y)
    counts = np.bincount(y)
    keep_classes = np.where(counts >= min_count)[0]
    keep_mask = np.isin(y, keep_classes)
    X2 = X[keep_mask]
    y2 = y[keep_mask]

    # remap labels to 0..K-1
    _, y_remap = np.unique(y2, return_inverse=True)
    return X2, y_remap


def load_retinaface_lfw_dataset(random_state: int):
    rng = np.random.default_rng(random_state)
    images_rgb, labels, label_names = load_lfw_subset(rng=rng)
    app = init_retinaface(lightweight=True)
    X, y = compute_embeddings_from_lfw(app, images_rgb, labels)
    X, y = filter_min_class_count(X, y, min_count=4)
    return DatasetBundle(
        name="RetinaFace LFW",
        X=X,
        y=y,
        meta={
            "source": "ucimlrepo",
            "uci_id": int(-1),
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": int(len(np.unique(y))),
        },
    )

def load_mnist() -> DatasetBundle:
    """
    Load MNIST via scikit-learn/OpenML.

    Returns:
        DatasetBundle with:
          - X: float32, shape (70000, 784), values in [0, 255]
          - y: int32, shape (70000,)
    """
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml("mnist_784", version=1, as_frame=False)

    X = mnist.data.astype(np.float32, copy=False)
    y_raw = mnist.target

    # OpenML labels are typically strings; force integer class labels.
    y = y_raw.astype(np.int32) if y_raw.dtype.kind in {"U", "S", "O"} else y_raw.astype(np.int32, copy=False)

    return DatasetBundle(
        name="mnist",
        X=X,
        y=y,
        meta={
            "source": "MNIST",
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": int(len(np.unique(y))),
        },
    )


# ----------------------------
# Shared utilities
# ----------------------------

def _class_order_within_shuffle(X, y, rng, class_order=None):
    y = np.asarray(y)
    classes = np.unique(y) if class_order is None else np.asarray(class_order)

    idx_out = []
    for c in classes:
        idx_c = np.flatnonzero(y == c)
        rng.shuffle(idx_c)              # random within class
        idx_out.append(idx_c)

    idx_out = np.concatenate(idx_out)
    return X[idx_out], y[idx_out]

def split_data(X: np.ndarray, y: np.ndarray, random_state: int
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.5,
        random_state=random_state,
        stratify=y,
        shuffle=True,
    )

    rng = np.random.default_rng(random_state)
    X_train, y_train = _class_order_within_shuffle(X_train, y_train, rng)
    X_test,  y_test  = _class_order_within_shuffle(X_test,  y_test,  rng)

    return X_train, X_test, y_train, y_test



def binarize_features_thermometer(data: np.ndarray, n_bits: int) -> np.ndarray:
    if n_bits <= 0:
        raise ValueError("n_bits must be a positive integer.")

    X = np.asarray(data, dtype=np.float32)
    n, m = X.shape

    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    ranges = np.where(max_vals - min_vals > 0, max_vals - min_vals, 1.0)

    norm = (X - min_vals) / ranges

    if n_bits == 1:
        return (norm > 0.5).astype(np.uint8)

    q = np.floor(norm * n_bits).astype(np.uint16)
    q = np.clip(q, 0, n_bits - 1)

    out = np.empty((n, m * n_bits), dtype=np.bool)
    for b in range(n_bits):
        out[:, b*m:(b+1)*m] = (q > b)

    return out



def artmap_weight_size(W) -> int:
    return int(sum(w.size for w in W))


def _array_bits(a) -> int:
    if a is None:
        return 0
    if sparse.issparse(a):
        bits = 0
        bits += a.data.size    * a.data.dtype.itemsize * 8
        bits += a.indices.size * a.indices.dtype.itemsize * 8
        bits += a.indptr.size  * a.indptr.dtype.itemsize * 8
        return int(bits)
    a = np.asarray(a)
    return int(a.size * a.dtype.itemsize * 8)


def memory_bits_multinomial_nb(nb) -> int:
    bits = 0
    bits += _array_bits(getattr(nb, "feature_log_prob_", None))
    bits += _array_bits(getattr(nb, "class_log_prior_", None))
    bits += _array_bits(getattr(nb, "feature_count_", None))
    bits += _array_bits(getattr(nb, "class_count_", None))
    bits += _array_bits(getattr(nb, "classes_", None))
    return int(bits)


def memory_bits_sgd_classifier(sgd) -> int:
    bits = 0
    bits += _array_bits(getattr(sgd, "coef_", None))
    bits += _array_bits(getattr(sgd, "intercept_", None))
    bits += _array_bits(getattr(sgd, "classes_", None))
    for name in ("t_", "n_iter_"):
        if hasattr(sgd, name):
            val = getattr(sgd, name)
            bits += 64 if isinstance(val, (float, np.floating, int, np.integer)) else 0
    return int(bits)


def _as_nonnegative_counts(X: np.ndarray, eps: float = 0.0) -> np.ndarray:
    X = np.asarray(X, dtype=np.uint16)
    if np.any(X < 0):
        mins = X.min(axis=0, keepdims=True)
        X = X - mins + eps
    return X


def safe_save_csv(df: pd.DataFrame, path: str, temp_path: str) -> None:
    """
    Atomic-ish write: write temp then replace.
    """
    df.to_csv(temp_path, index=False)
    os.replace(temp_path, path)

# ----------------------------
# Test ART functions (return dicts)
# ----------------------------

def run_fuzzyart_binary(X: np.ndarray, y: np.ndarray, rho: float, n_bits: int,
                           random_state: int, **kwargs) -> Dict[str, Any]:
    cls = FuzzyART(rho=rho, alpha=1e-10, beta=1.0)

    X_bin = binarize_features_thermometer(X, n_bits)
    X_prep = cls.prepare_data(X_bin.astype(np.int8))
    del X_bin

    X_train, X_test, y_train, y_test = split_data(X_prep, y, random_state=random_state)
    del X_prep
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    t0 = perf_counter()
    cls.fit(X_train)
    t1 = perf_counter()

    p0 = perf_counter()
    y_pred = cls.predict(X_test)
    p1 = perf_counter()

    ari_train = adjusted_rand_score(y_train, cls.labels_)
    ami_train = adjusted_mutual_info_score(y_train, cls.labels_)
    ari_test = adjusted_rand_score(y_test, y_pred)
    ami_test = adjusted_mutual_info_score(y_test, y_pred)

    mem_bits = 32 * artmap_weight_size(cls.module_a.W) + \
               np.ceil(np.log2(np.max(y)+1)) * len(cls.map)

    return {
        "ari_train": float(ari_train),
        "ami_train": float(ami_train),
        "ari_test": float(ari_test),
        "ami_test": float(ami_test),
        "train_time_s": float(t1 - t0),
        "pred_time_s": float(p1 - p0),
        "n_clusters": int(cls.module_a.n_clusters),
        "memory_bits": float(mem_bits),
        "variant": "binary",
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test))
    }


def run_fuzzyart_continuous(X: np.ndarray, y: np.ndarray, rho: float,
                               random_state: int, **kwargs) -> Dict[str, Any]:
    cls = FuzzyART(rho=rho, alpha=1e-10, beta=1.0)
    X_prep = cls.prepare_data(X)

    X_train, X_test, y_train, y_test = split_data(X_prep, y, random_state=random_state)
    del X_prep

    t0 = perf_counter()
    cls.fit(X_train)
    t1 = perf_counter()

    p0 = perf_counter()
    y_pred = cls.predict(X_test)
    p1 = perf_counter()

    ari_train = adjusted_rand_score(y_train, cls.labels_)
    ami_train = adjusted_mutual_info_score(y_train, cls.labels_)
    ari_test = adjusted_rand_score(y_test, y_pred)
    ami_test = adjusted_mutual_info_score(y_test, y_pred)

    mem_bits = 32 * artmap_weight_size(cls.module_a.W) + \
               np.ceil(np.log2(np.max(y)+1)) * len(cls.map)

    return {
        "ari_train": float(ari_train),
        "ami_train": float(ami_train),
        "ari_test": float(ari_test),
        "ami_test": float(ami_test),
        "train_time_s": float(t1 - t0),
        "pred_time_s": float(p1 - p0),
        "n_clusters": int(cls.module_a.n_clusters),
        "memory_bits": float(mem_bits),
        "variant": "continuous",
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test))
    }


def run_binaryfuzzyart(X: np.ndarray, y: np.ndarray, rho: float, n_bits: int,
                          random_state: int, **kwargs) -> Dict[str, Any]:
    cls = BinaryFuzzyART(rho=rho)
    X_bin = binarize_features_thermometer(X, n_bits)
    X_prep = cls.prepare_data(X_bin)
    del X_bin

    X_train, X_test, y_train, y_test = split_data(X_prep, y, random_state=random_state)
    del X_prep
    X_train = X_train.astype(np.uint8)
    X_test = X_test.astype(np.uint8)

    t0 = perf_counter()
    cls.fit(X_train)
    t1 = perf_counter()

    p0 = perf_counter()
    y_pred = cls.predict(X_test)
    p1 = perf_counter()

    ari_train = adjusted_rand_score(y_train, cls.labels_)
    ami_train = adjusted_mutual_info_score(y_train, cls.labels_)
    ari_test = adjusted_rand_score(y_test, y_pred)
    ami_test = adjusted_mutual_info_score(y_test, y_pred)

    mem_bits = artmap_weight_size(cls.module_a.W) + \
               np.ceil(np.log2(np.max(y)+1)) * len(cls.map)
    compressed_mem_bits = 2*np.ceil(np.log2(n_bits)+1)*X.shape[1] + \
               np.ceil(np.log2(np.max(y)+1)) * len(cls.map)

    mem_bits = min(mem_bits, compressed_mem_bits)

    return {
        "ari_train": float(ari_train),
        "ami_train": float(ami_train),
        "ari_test": float(ari_test),
        "ami_test": float(ami_test),
        "train_time_s": float(t1 - t0),
        "pred_time_s": float(p1 - p0),
        "n_clusters": int(cls.module_a.n_clusters),
        "memory_bits": float(mem_bits),
        "variant": "binary",
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test))
    }


def run_art1(X: np.ndarray, y: np.ndarray, rho: float, n_bits: int,
                random_state: int, **kwargs) -> Dict[str, Any]:
    cls = ART1(rho=rho, L=1.0)
    X_bin = binarize_features_thermometer(X, n_bits)
    X_prep = cls.prepare_data(X_bin).astype(np.int16)
    del X_bin

    X_train, X_test, y_train, y_test = split_data(X_prep, y, random_state=random_state)
    del X_prep

    t0 = perf_counter()
    cls.fit(X_train)
    t1 = perf_counter()

    p0 = perf_counter()
    y_pred = cls.predict(X_test)
    p1 = perf_counter()

    ari_train = adjusted_rand_score(y_train, cls.labels_)
    ami_train = adjusted_mutual_info_score(y_train, cls.labels_)
    ari_test = adjusted_rand_score(y_test, y_pred)
    ami_test = adjusted_mutual_info_score(y_test, y_pred)

    mem_bits = 32 * artmap_weight_size(cls.module_a.W) + \
               np.ceil(np.log2(np.max(y)+1)) * len(cls.map)

    return {
        "ari_train": float(ari_train),
        "ami_train": float(ami_train),
        "ari_test": float(ari_test),
        "ami_test": float(ami_test),
        "train_time_s": float(t1 - t0),
        "pred_time_s": float(p1 - p0),
        "n_clusters": int(cls.module_a.n_clusters),
        "memory_bits": float(mem_bits),
        "variant": "binary",
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test))
    }

# ----------------------------
# Test ARTMAP functions (return dicts)
# ----------------------------

def run_fuzzyartmap_binary(X: np.ndarray, y: np.ndarray, rho: float, n_bits: int,
                           random_state: int, **kwargs) -> Dict[str, Any]:
    cls = FuzzyARTMAP(rho=rho, alpha=1e-10, beta=1.0)

    X_bin = binarize_features_thermometer(X, n_bits).astype(np.int8)
    # testing
    X_prep = cls.prepare_data(X_bin.astype(np.int8))
    del X_bin

    X_train, X_test, y_train, y_test = split_data(X_prep, y, random_state=random_state)
    del X_prep
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    t0 = perf_counter()
    cls.fit(X_train, y_train)
    t1 = perf_counter()

    p0 = perf_counter()
    y_pred = cls.predict(X_test)
    p1 = perf_counter()

    acc = accuracy_score(y_test, y_pred)

    mem_bits = 32 * artmap_weight_size(cls.module_a.W) + \
               np.ceil(np.log2(np.max(y)+1)) * len(cls.map)

    return {
        "accuracy": float(acc),
        "train_time_s": float(t1 - t0),
        "pred_time_s": float(p1 - p0),
        "n_clusters": int(cls.module_a.n_clusters),
        "memory_bits": float(mem_bits),
        "variant": "binary",
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test))
    }


def run_fuzzyartmap_continuous(X: np.ndarray, y: np.ndarray, rho: float,
                               random_state: int, **kwargs) -> Dict[str, Any]:
    cls = FuzzyARTMAP(rho=rho, alpha=1e-10, beta=1.0)
    X_prep = cls.prepare_data(X)

    X_train, X_test, y_train, y_test = split_data(X_prep, y, random_state=random_state)
    del X_prep

    t0 = perf_counter()
    cls.fit(X_train, y_train)
    t1 = perf_counter()

    p0 = perf_counter()
    y_pred = cls.predict(X_test)
    p1 = perf_counter()

    acc = accuracy_score(y_test, y_pred)
    mem_bits = 32 * artmap_weight_size(cls.module_a.W) + \
               np.ceil(np.log2(np.max(y)+1)) * len(cls.map)

    return {
        "accuracy": float(acc),
        "train_time_s": float(t1 - t0),
        "pred_time_s": float(p1 - p0),
        "n_clusters": int(cls.module_a.n_clusters),
        "memory_bits": float(mem_bits),
        "variant": "continuous",
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test))
    }


def run_binaryfuzzyartmap(X: np.ndarray, y: np.ndarray, rho: float, n_bits: int,
                          random_state: int, **kwargs) -> Dict[str, Any]:
    cls = BinaryFuzzyARTMAP(rho=rho)
    X_bin = binarize_features_thermometer(X, n_bits)
    X_prep = cls.prepare_data(X_bin)
    del X_bin

    X_train, X_test, y_train, y_test = split_data(X_prep, y, random_state=random_state)
    del X_prep
    X_train = X_train.astype(np.uint8)
    X_test = X_test.astype(np.uint8)

    t0 = perf_counter()
    cls.fit(X_train, y_train)
    t1 = perf_counter()

    p0 = perf_counter()
    y_pred = cls.predict(X_test)
    p1 = perf_counter()

    acc = accuracy_score(y_test, y_pred)

    mem_bits = artmap_weight_size(cls.module_a.W) + \
               np.ceil(np.log2(np.max(y)+1)) * len(cls.map)
    compressed_mem_bits = 2*np.ceil(np.log2(n_bits+1))*X.shape[1] + \
               np.ceil(np.log2(np.max(y)+1)) * len(cls.map)

    mem_bits = min(mem_bits, compressed_mem_bits)

    return {
        "accuracy": float(acc),
        "train_time_s": float(t1 - t0),
        "pred_time_s": float(p1 - p0),
        "n_clusters": int(cls.module_a.n_clusters),
        "memory_bits": float(mem_bits),
        "variant": "binary",
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test))
    }


def run_art1map(X: np.ndarray, y: np.ndarray, rho: float, n_bits: int,
                random_state: int, **kwargs) -> Dict[str, Any]:
    cls = ART1MAP(rho=rho, L=1.0)
    X_bin = binarize_features_thermometer(X, n_bits)
    X_prep = cls.prepare_data(X_bin).astype(np.int16)
    del X_bin
    X_train, X_test, y_train, y_test = split_data(X_prep, y, random_state=random_state)
    del X_prep
    X_train = X_train.astype(np.uint8)
    X_test = X_test.astype(np.uint8)

    t0 = perf_counter()
    cls.fit(X_train, y_train)
    t1 = perf_counter()

    p0 = perf_counter()
    y_pred = cls.predict(X_test)
    p1 = perf_counter()

    acc = accuracy_score(y_test, y_pred)
    mem_bits = 32 * artmap_weight_size(cls.module_a.W) + \
               np.ceil(np.log2(np.max(y)+1)) * len(cls.map)

    return {
        "accuracy": float(acc),
        "train_time_s": float(t1 - t0),
        "pred_time_s": float(p1 - p0),
        "n_clusters": int(cls.module_a.n_clusters),
        "memory_bits": float(mem_bits),
        "variant": "binary",
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test))
    }


def run_multinomial_nb_binary(X: np.ndarray, y: np.ndarray, n_bits: int,
                              random_state: int, alpha: float = 1.0, **kwargs) -> \
        Dict[str, Any]:
    cls = MultinomialNB(alpha=alpha)

    X_bin = binarize_features_thermometer(X, n_bits)
    X_prep = _as_nonnegative_counts(X_bin)
    del X_bin

    X_train, X_test, y_train, y_test = split_data(X_prep, y, random_state=random_state)
    del X_prep

    t0 = perf_counter()
    cls.fit(X_train, y_train)
    t1 = perf_counter()

    p0 = perf_counter()
    y_pred = cls.predict(X_test)
    p1 = perf_counter()

    acc = accuracy_score(y_test, y_pred)

    return {
        "accuracy": float(acc),
        "train_time_s": float(t1 - t0),
        "pred_time_s": float(p1 - p0),
        "memory_bits": float(memory_bits_multinomial_nb(cls)),
        "variant": "binary",
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test))
    }


def run_sgd_binary(X: np.ndarray, y: np.ndarray, n_bits: int, random_state: int,
                   loss: str = "log_loss", alpha: float = 1e-4, **kwargs) -> \
        Dict[str, Any]:
    cls = SGDClassifier(loss=loss, alpha=alpha, random_state=random_state)

    X_prep = binarize_features_thermometer(X, n_bits)
    X_train, X_test, y_train, y_test = split_data(X_prep, y, random_state=random_state)
    del X_prep
    X_train = X_train.astype(np.uint8)
    X_test = X_test.astype(np.uint8)
    classes = np.unique(y_train)

    t0 = perf_counter()
    cls.partial_fit(X_train, y_train, classes=classes)
    t1 = perf_counter()

    p0 = perf_counter()
    y_pred = cls.predict(X_test)
    p1 = perf_counter()

    acc = accuracy_score(y_test, y_pred)

    return {
        "accuracy": float(acc),
        "train_time_s": float(t1 - t0),
        "pred_time_s": float(p1 - p0),
        "memory_bits": float(memory_bits_sgd_classifier(cls)),
        "variant": f"binary(loss={loss})",
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test))
    }


def run_sgd_continuous(X: np.ndarray, y: np.ndarray, random_state: int,
                       loss: str = "log_loss", alpha: float = 1e-4, **kwargs) -> \
        Dict[str, Any]:
    cls = SGDClassifier(loss=loss, alpha=alpha, random_state=random_state)

    X_min = np.min(X)
    X_max = np.max(X)
    X_norm = (X - X_min) / (X_max - X_min + 1e-12)

    X_train, X_test, y_train, y_test = split_data(X_norm, y, random_state=random_state)
    del X_norm
    classes = np.unique(y_train)

    t0 = perf_counter()
    cls.partial_fit(X_train, y_train, classes=classes)
    t1 = perf_counter()

    p0 = perf_counter()
    y_pred = cls.predict(X_test)
    p1 = perf_counter()

    acc = accuracy_score(y_test, y_pred)

    return {
        "accuracy": float(acc),
        "train_time_s": float(t1 - t0),
        "pred_time_s": float(p1 - p0),
        "memory_bits": float(memory_bits_sgd_classifier(cls)),
        "variant": f"continuous(loss={loss})",
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test))
    }


# ----------------------------
# Experiment runner
# ----------------------------

@dataclass
class TestSpec:
    name: str
    fn: Callable[[np.ndarray, np.ndarray], Dict[str, Any]]
    depends_on_n_bits: bool = True
