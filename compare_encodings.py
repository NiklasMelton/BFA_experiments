#!/usr/bin/env python3
"""
Distance–distance scatters under L1: continuous baseline vs multiple encodings,
and how behavior changes with N_BITS.

Requires:
  - numpy
  - matplotlib
  - your local module (rename it to avoid stdlib conflict!)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from sklearn.datasets import fetch_openml

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr


from binary_encodings import (
    binarize_features,
    binarize_features_thermometer,
    binarize_features_gray,
)

# =============================
# Fixed experiment parameters
# =============================
N_POINTS = 400
N_FEATURES = 16
KIND = "mixture"          # "gaussian" | "uniform" | "mixture"
CORR = 0.25
NOISE_STD = 0.05
N_PAIRS = 60_000
SEED = 0

# Sweep these:
N_BITS_LIST = [4, 16, 32]

FONT_SMALL = 20
FONT_MED = 24
FONT_LARGE = 30

plt.rcParams.update({
    "font.size": FONT_SMALL,
    "axes.titlesize": FONT_MED,
    "axes.labelsize": FONT_MED,
    "xtick.labelsize": FONT_SMALL,
    "ytick.labelsize": FONT_SMALL,
    "legend.fontsize": FONT_SMALL,
    "figure.titlesize": FONT_LARGE,
})

OUTPATH = "distance_distance_bitsweep.png"  # set None to show


# -----------------------------
# Synthetic data generation
# -----------------------------
def make_synthetic(n: int, d: int, kind: str, corr: float, noise: float, seed: int) -> np.ndarray:
    if not (0.0 <= corr < 0.99):
        raise ValueError("corr must be in [0, 0.99).")
    rng = np.random.default_rng(seed)

    if kind == "gaussian":
        X = rng.standard_normal((n, d))
    elif kind == "uniform":
        X = rng.random((n, d))
    elif kind == "mixture":
        z = rng.integers(0, 2, size=n)
        means = np.vstack([np.zeros(d), np.ones(d) * 2.0])
        X = rng.standard_normal((n, d)) + means[z]
    else:
        raise ValueError(f"Unknown kind={kind!r}.")

    if corr > 0:
        W = np.eye(d) + 0.1 * rng.standard_normal((d, d))
        X = (1.0 - corr) * X + corr * (X @ W)

    if noise > 0:
        X = X + noise * rng.standard_normal((n, d))

    return X.astype(np.float64)


def minmax_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    denom = np.maximum(x_max - x_min, eps)
    return (X - x_min) / denom


# -----------------------------
# Pair sampling + L1 distances
# -----------------------------
def sample_pairs(n: int, m: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=m, endpoint=False)
    j = rng.integers(0, n, size=m, endpoint=False)
    bad = (i == j)
    while np.any(bad):
        j[bad] = rng.integers(0, n, size=int(np.sum(bad)), endpoint=False)
        bad = (i == j)
    a = np.minimum(i, j)
    b = np.maximum(i, j)
    return a, b


def l1_distances_for_pairs(X: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(X[a] - X[b]), axis=1)


# -----------------------------
# Plotting
# -----------------------------
@dataclass(frozen=True)
class MethodSpec:
    name: str
    encode_fn: callable


def main() -> None:
    # Data
    # X = make_synthetic(N_POINTS, N_FEATURES, KIND, CORR, NOISE_STD, SEED)
    X, y = fetch_openml(
        name="isolet",
        version=1,
        as_frame=False,
        parser="auto",
        return_X_y=True,
    )
    N_PAIRS = 60_000
    Xn = minmax_normalize(X)

    # Pairs + baseline distances
    a, b = sample_pairs(X.shape[0], N_PAIRS, seed=SEED + 1)
    d_base = l1_distances_for_pairs(Xn, a, b)

    d_base = d_base / X.shape[1]

    methods: List[MethodSpec] = [
        MethodSpec("binary", binarize_features),
        MethodSpec("thermometer", binarize_features_thermometer),
        MethodSpec("gray", binarize_features_gray),
    ]

    # Correlation storage: one series per method over N_BITS_LIST
    pearson_series = {m.name: [] for m in methods}
    spearman_series = {m.name: [] for m in methods}

    n_rows = len(methods)
    n_cols = len(N_BITS_LIST)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.8 * n_cols, 4.4 * n_rows),
        constrained_layout=True,
        sharex=True
    )
    if n_rows == 1:
        axes = np.array([axes])


    for r, method in enumerate(methods):
        for c, n_bits in enumerate(N_BITS_LIST):
            ax = axes[r, c]

            E = np.asarray(method.encode_fn(Xn, n_bits), dtype=np.int32)
            d_enc = l1_distances_for_pairs(E, a, b)


            d_enc = d_enc / E.shape[1]

            # --- correlations vs baseline (SciPy) ---
            p = float(pearsonr(d_base, d_enc).statistic)
            s = float(spearmanr(d_base, d_enc).correlation)

            pearson_series[method.name].append(p)
            spearman_series[method.name].append(s)

            ax.plot([0, 1], [0, 1], color="red", linewidth=1.0, zorder=1)

            ax.scatter(d_base, d_enc, s=6.0, alpha=0.12, edgecolors="none")
            # ax.set_xlim(0., 0.8)
            # ax.set_ylim(0.0, 0.8)
            ax.grid(True, linewidth=0.5)

            if r == 0:
                ax.set_title(f"n_bits = {n_bits}")
            if c == 0:
                ax.set_ylabel(f"{method.name.title()}\nL1 Encoded")
            if r == n_rows - 1:
                ax.set_xlabel("L1 Continuous")

    fig.suptitle("Isolet L1 Distance-Distance Scatter",
                 fontsize=FONT_LARGE)

    if OUTPATH:
        fig.savefig(OUTPATH, dpi=200)
        print(f"Saved plot to: {OUTPATH}")
    else:
        plt.show()
    #
    # # =============================
    # # Correlation vs n_bits plots
    # # =============================
    # x = np.array(N_BITS_LIST, dtype=float)
    #
    # fig = plt.figure(figsize=(7.2, 4.6), constrained_layout=True)
    # for name, ys in pearson_series.items():
    #     plt.plot(x, ys, marker="o", label=name)
    # plt.xscale("log", base=2)
    # plt.ylim(-1.0, 1.0)
    # plt.xlabel("n_bits")
    # plt.ylabel("Pearson corr(dist_encoded, dist_continuous)")
    # plt.title("Pearson correlation vs n_bits")
    # plt.grid(True, linewidth=0.5)
    # plt.legend()
    # if OUTPATH:
    #     pearson_out = OUTPATH.replace(".png", "_pearson.png")
    #     plt.savefig(pearson_out, dpi=200)
    #     print(f"Saved plot to: {pearson_out}")
    # else:
    #     plt.show()
    #
    # plt.figure(figsize=(7.2, 4.6), constrained_layout=True)
    # for name, ys in spearman_series.items():
    #     plt.plot(x, ys, marker="o", label=name)
    # plt.xscale("log", base=2)
    # plt.ylim(-1.0, 1.0)
    # plt.xlabel("n_bits")
    # plt.ylabel("Spearman corr(dist_encoded, dist_continuous)")
    # plt.title("Spearman correlation vs n_bits")
    # plt.grid(True, linewidth=0.5)
    # plt.legend()
    # if OUTPATH:
    #     spearman_out = OUTPATH.replace(".png", "_spearman.png")
    #     plt.savefig(spearman_out, dpi=200)
    #     print(f"Saved plot to: {spearman_out}")
    # else:
    #     plt.show()


if __name__ == "__main__":
    main()
