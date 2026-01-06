#pip install git+https://github.com/NiklasMelton/AdaptiveResonanceLib@develop
#!pip install insightface onnxruntime-gpu scikit-learn opencv-python-headless

from typing import List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

from insightface.app import FaceAnalysis
from time import perf_counter

from artlib import FuzzyARTMAP, BinaryFuzzyARTMAP, ART1, SimpleARTMAP, \
    complement_code, FuzzyART, BinaryFuzzyART
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



# ----------------------------
# Configuration
# ----------------------------

NUM_IDENTITIES = 14000     # how many people to sample
IMAGES_PER_ID = 1000      # max images per person
RANDOM_STATE = 0

# Minimum target size for the *long side* before detection
MIN_LONG_SIDE = 256

# Extra border for detection context
BORDER = 32  # pixels on each side


# ----------------------------
# Dataset helpers
# ----------------------------

def load_lfw_subset(
    num_ids: int,
    images_per_id: int,
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
    if len(unique_labels) > num_ids:
        chosen = rng.choice(unique_labels, size=num_ids, replace=False)
    else:
        chosen = unique_labels

    label_map = {orig: new for new, orig in enumerate(chosen)}
    label_names = [target_names[orig] for orig in chosen]

    sampled_images: List[np.ndarray] = []
    sampled_labels: List[int] = []

    for orig_label in chosen:
        idxs = np.where(targets == orig_label)[0]
        if len(idxs) == 0:
            continue
        if len(idxs) > images_per_id:
            idxs = rng.choice(idxs, size=images_per_id, replace=False)

        for idx in idxs:
            sampled_images.append(images[idx])
            sampled_labels.append(label_map[orig_label])

    images_array = np.stack(sampled_images, axis=0)
    labels_array = np.array(sampled_labels, dtype=int)

    return images_array, labels_array, label_names


# ----------------------------
# Visualization helpers
# ----------------------------

def show_sample_images(
    images: np.ndarray,
    labels: np.ndarray,
    label_names: List[str],
    num_to_show: int = 9,
) -> None:
    num_to_show = min(num_to_show, images.shape[0])
    cols = 3
    rows = int(np.ceil(num_to_show / cols))

    plt.figure(figsize=(3 * cols, 3 * rows))
    for idx in range(num_to_show):
        img = images[idx]  # (H, W, 3), float32 in [0, 255]
        label = labels[idx]
        name = label_names[label]

        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)

        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(img_uint8)  # RGB
        ax.set_title(f"{name} (ID {label})", fontsize=8)
        ax.axis("off")

    plt.suptitle("Sample LFW images", fontsize=14)
    plt.tight_layout()
    plt.show()



# ----------------------------
# ArcFace / InsightFace helpers
# ----------------------------

def init_arcface(
    lightweight: bool = False,
    ctx_id: int = 0,
    det_size: Tuple[int, int] | None = None,
) -> FaceAnalysis:
    """
    Initialize an InsightFace FaceAnalysis app with ArcFace.

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
        model_name = "buffalo_l"   # your current default, larger & more accurate
        default_det_size = (256, 256)

    if det_size is None:
        det_size = default_det_size

    app = FaceAnalysis(
        name=model_name,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
    Run ArcFace on a batch of RGB images.

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


def binarize_features_thermometer(data: np.ndarray, n_bits: int) -> np.ndarray:
    """Binarizes each feature in the data using thermometer encoding.

    Parameters:
        data (np.ndarray): Input array of shape (n, m), where n is the number of samples and m is the number of features.
        n_bits (int): Number of bits to use for thermometer encoding.

    Returns:
        np.ndarray: A thermometer-coded representation of the input data with shape (n, m * n_bits).

    """
    if n_bits <= 0:
        raise ValueError("n_bits must be a positive integer.")

    n, m = data.shape
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)

    # Avoid division by zero in case of constant features
    ranges = np.where(max_vals - min_vals > 0, max_vals - min_vals, 1)

    # Normalize to [0, 1]
    normalized_data = (data - min_vals) / ranges

    if n_bits == 1:
        return (normalized_data > 0.5).astype(np.uint8)

    # Quantize into `n_bits` levels (instead of `2^n_bits` levels)
    quantized_data = np.floor(normalized_data * n_bits).astype(int)

    # Generate thermometer encoding: fill from left to right
    thermometer_encoded = np.zeros((n, m, n_bits), dtype=np.uint8)

    for i in range(n_bits):
        thermometer_encoded[:, :, i] = (quantized_data > i).astype(np.uint8)

    return thermometer_encoded.reshape(n, m * n_bits)


def train_FuzzyARTMAP(X, y):
    print("="*20)
    print("FuzzyART")
    cls = FuzzyARTMAP(rho=0.0, alpha=1e-10, beta=1.0)

    X_bin = binarize_features_thermometer(X, 1)
    X_prep = cls.prepare_data(X_bin)

    X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.5,
                                                        random_state=42, stratify=y,
                                                        shuffle=True)
    print(X_train.shape, X_test.shape)

    t0 = perf_counter()
    cls = cls.fit(X_train, y_train)
    t1 = perf_counter()
    print(f"Training time: {t1 - t0:.2f}s")
    print("Clusters: ", cls.module_a.n_clusters)

    y_pred = cls.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")


def train_FuzzyARTMAPpy(X, y):
    print("="*20)
    print("FuzzyART -- python")
    cls = SimpleARTMAP(FuzzyART(rho=0.0, alpha=1e-10, beta=1.0))

    X_bin = binarize_features_thermometer(X, 1)
    X_prep = cls.prepare_data(X_bin)

    X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.5,
                                                        random_state=42, stratify=y,
                                                        shuffle=True)
    print(X_train.shape, X_test.shape)

    t0 = perf_counter()
    cls = cls.fit(X_train, y_train)
    t1 = perf_counter()
    print(f"Training time: {t1 - t0:.2f}s")
    print("Clusters: ", cls.module_a.n_clusters)

    y_pred = cls.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")


def train_BinaryFuzzyARTMAP(X, y):
    print("="*20)
    print("BinaryFuzzyART")
    cls = BinaryFuzzyARTMAP(rho=0.0)

    X_bin = binarize_features_thermometer(X, 1)
    X_prep = cls.prepare_data(X_bin)

    X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.5, random_state=42, stratify=y, shuffle=True)
    print(X_train.shape, X_test.shape)

    # sidx = np.argsort(y_train)
    # X_train = X_train[sidx]
    # y_train = y_train[sidx]

    t0 = perf_counter()
    cls = cls.fit(X_train, y_train)
    t1 = perf_counter()
    print(f"Training time: {t1 - t0:.2f}s")
    print("Clusters: ", cls.module_a.n_clusters)

    y_pred = cls.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")


def train_BinaryFuzzyARTMAPpy(X, y):
    print("="*20)
    print("BinaryFuzzyART -- python")
    cls = SimpleARTMAP(BinaryFuzzyART(rho=0.0))

    X_bin = binarize_features_thermometer(X, 1)
    X_prep = cls.prepare_data(X_bin)

    X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.5, random_state=42, stratify=y, shuffle=True)
    print(X_train.shape, X_test.shape)

    # sidx = np.argsort(y_train)
    # X_train = X_train[sidx]
    # y_train = y_train[sidx]

    t0 = perf_counter()
    cls = cls.fit(X_train, y_train)
    t1 = perf_counter()
    print(f"Training time: {t1 - t0:.2f}s")
    print("Clusters: ", cls.module_a.n_clusters)

    y_pred = cls.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")


def train_ART1ARTMAP(X, y):
    print("="*20)
    print("ART1")
    cls = SimpleARTMAP(ART1(rho=0.0, L=1.0))

    X_bin = binarize_features_thermometer(X, 1)
    X_bin = complement_code(X_bin)
    X_prep = cls.prepare_data(X_bin).astype(bool)

    X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.5,
                                                        random_state=42, stratify=y,
                                                        shuffle=True)
    print(X_train.shape, X_test.shape)

    # sidx = np.argsort(y_train)
    # X_train = X_train[sidx]
    # y_train = y_train[sidx]

    t0 = perf_counter()
    cls = cls.fit(X_train, y_train)
    t1 = perf_counter()
    print(f"Training time: {t1 - t0:.2f}s")
    print("Clusters: ", cls.module_a.n_clusters)

    y_pred = cls.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")

def main():
    rng = np.random.default_rng(RANDOM_STATE)

    # 1. Load LFW subset
    print("Loading LFW subset via sklearn...")
    images_rgb, labels, label_names = load_lfw_subset(
        num_ids=NUM_IDENTITIES,
        images_per_id=IMAGES_PER_ID,
        rng=rng,
    )
    print(f"Loaded {images_rgb.shape[0]} images from {len(label_names)} identities.")
    print(f"Image shape: {images_rgb.shape[1:]} (H, W, C)")

    # 2. Show example images
    print("Showing sample images...")
    show_sample_images(images_rgb, labels, label_names, num_to_show=9)

    # 3. Initialize ArcFace
    print("Initializing ArcFace (InsightFace FaceAnalysis)...")
    app = init_arcface(lightweight=True)

    # 4. Compute embeddings
    print("Computing ArcFace embeddings...")
    X, y = compute_embeddings_from_lfw(app, images_rgb, labels)
    print(f"Got embeddings for {X.shape[0]} images, dim = {X.shape[1]}")

    # 5. ART tests
    train_FuzzyARTMAP(X, y)
    train_BinaryFuzzyARTMAP(X, y)
    train_FuzzyARTMAPpy(X, y)
    train_BinaryFuzzyARTMAPpy(X, y)
    train_ART1ARTMAP(X, y)

if __name__ == "__main__":
    main()