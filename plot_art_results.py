import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 22,
    "axes.titlesize": 24,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "legend.title_fontsize": 18,
    "figure.titlesize": 26,
})

okabe_ito = [
    "#E69F00",  # orange
    "#000000",  # black
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#009E73",  # bluish green

]


def fix_mem_bits(df):
    # Work on a copy so caller's df isn't mutated unexpectedly.
    df = df.copy()

    # ART1
    m = df["model"].eq("ART1")
    df.loc[m, "memory_bits"] = (
        df.loc[m, "n_clusters"]
        * (df.loc[m, "n_features"] * df.loc[m, "n_bits"] * 4) * 32
        + np.ceil(np.log2(df.loc[m, "ds_n_classes"])) * df.loc[m, "n_clusters"]
    )

    # Fuzzy ART binary
    m = df["model"].eq("FuzzyART_binary")
    df.loc[m, "memory_bits"] = (
        df.loc[m, "n_clusters"]
        * (df.loc[m, "n_features"] * df.loc[m, "n_bits"] * 2) * 32
        + np.ceil(np.log2(df.loc[m, "ds_n_classes"])) * df.loc[m, "n_clusters"]
    )

    # Fuzzy ART continuous
    m = df["model"].eq("FuzzyART_continuous")
    df.loc[m, "memory_bits"] = (
        df.loc[m, "n_clusters"] * (df.loc[m, "n_features"] * 2) * 32
        + np.ceil(np.log2(df.loc[m, "ds_n_classes"])) * df.loc[m, "n_clusters"]
    )

    # Binary Fuzzy ART
    m = df["model"].eq("BinaryFuzzyART")
    mem_bits_a = (
        df.loc[m, "n_clusters"] * (df.loc[m, "n_features"] * df.loc[m, "n_bits"] * 2)
        + np.ceil(np.log2(df.loc[m, "ds_n_classes"] + 1)) * df.loc[m, "n_clusters"]
    )

    mem_bits_b = (
        2 * np.ceil(np.log2(df.loc[m, "n_bits"] + 1)) * df.loc[m, "n_features"]
        + np.ceil(np.log2(df.loc[m, "ds_n_classes"])) * df.loc[m, "n_clusters"]
    )

    df.loc[m, "memory_bits"] = np.minimum(mem_bits_a, mem_bits_b)

    return df



def load_and_aggregate_data():
    df = pd.read_csv("bfa_art_results.csv")
    df = fix_mem_bits(df)
    errors = df["error"].unique()
    print("Errors detected")
    print(errors)
    print("Filtering results")
    df = df[df["error"].isna()]


    group_cols = ["dataset", "model", "rho", "n_bits", "variant"]

    mean_std_cols = ["ari_test", "ari_train", "ami_train", "ami_test", "train_time_s",
                     "pred_time_s", "n_clusters", "memory_bits"]
    first_cols = ["n_samples", "n_features", "n_classes", "n_train", "n_test"]

    agg_spec = {c: ["mean", "std"] for c in mean_std_cols}
    agg_spec.update({c: "first" for c in first_cols})

    summary = (
        df.groupby(group_cols, dropna=False)
            .agg(agg_spec)
    )

    # flatten MultiIndex columns -> e.g., accuracy_mean, accuracy_std
    summary.columns = [
        f"{col}_{stat}" if stat else col
        for (col, stat) in summary.columns.to_flat_index()
    ]

    summary = summary.reset_index()

    return summary


def plot_ari_vs_memory_bits(df, n_bits_left=1, n_bits_right=16):
    """
    2x4 grid (4 datasets):
      - Columns 0-1: dataset panels for n_bits = n_bits_left
      - Columns 2-3: dataset panels for n_bits = n_bits_right
    Within each panel:
      x = memory_bits_mean, y = accuracy_mean
      color = (model, variant) pair
      marker = rho
      points only (no connecting lines).
    """
    required = {"dataset", "model", "variant", "rho", "n_bits", "ari_train_mean",
                "ami_train_mean",
                "memory_bits_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    # pick up to 4 datasets
    datasets = list(pd.unique(df["dataset"]))[:4]

    # aesthetics mappings computed once from full df (stable across panels)
    mv_pairs = df[["model", "variant"]].drop_duplicates().apply(tuple, axis=1).tolist()
    mv_pairs = sorted(mv_pairs)

    mv_to_color = {mv: okabe_ito[i % len(okabe_ito)] for i, mv in enumerate(mv_pairs)}

    rhos = sorted(pd.unique(df["rho"]))
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "H", "p", "8"]
    rho_to_marker = {rho: marker_cycle[i % len(marker_cycle)] for i, rho in enumerate(rhos)}

    # layout: 2 rows x 4 cols
    fig, axes = plt.subplots(2, 4, figsize=(22, 9), sharex=False, sharey=False)

    positions = [
        (0, 0, 0), (0, 1, 1),
        (1, 0, 2), (1, 1, 3),
    ]  # (row, col_left_base, dataset_index)

    handles_for_legend = {}

    def _plot_panel(ax, dsub_bits, dataset, bits_value):
        # points only; group only for labeling/consistent mapping
        for (model, variant, rho), g in dsub_bits.groupby(["model", "variant", "rho"], dropna=False):
            if g.empty:
                continue

            x = g["memory_bits_mean"].to_numpy()
            y = g["ari_train_mean"].to_numpy()

            # avoid issues with log scale if zeros slip in
            mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
            if not np.any(mask):
                continue

            color = mv_to_color[(model, variant)]

            model_clean = model.replace("_binary", "").replace("_continuous", "")
            variant_clean = variant.replace("(loss=log_loss)", "")
            label = f"{model_clean} | {variant_clean}"
            if "ART" in model_clean:
                label += f" | rho={rho}"
                marker = rho_to_marker[rho]
            else:
                marker = "X"

            sc = ax.scatter(
                x[mask],
                y[mask],
                color=color,
                marker=marker,
                s=45 if "Binary" not in model_clean else 90,
                alpha=0.9,
                label=label,
            )
            handles_for_legend.setdefault(label, sc)

        title_map = {
            "mnist": "MNIST",
            "uci_har": "HAR",
            "pendigits": "Pendigits",
            "Spambase": "Spambase",
        }
        dataset_title = title_map.get(str(dataset), str(dataset))

        ax.set_title(f"{dataset_title} | n_bits={bits_value}")
        ax.set_xlabel("Memory Bits")
        ax.set_ylabel("ARI (train)")
        ax.set_xscale("log")
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

        # set xlim based on data in this panel (robust)
        xs = dsub_bits["memory_bits_mean"].to_numpy()
        xs = xs[np.isfinite(xs) & (xs > 0)]
        if xs.size:
            lo, hi = np.min(xs), np.max(xs)
            # ax.set_xlim([lo / 1.2, hi * 1.2])

    # plot each dataset in left/right panels
    for row, col_left, di in positions:
        if di >= len(datasets):
            axes[row, col_left].axis("off")
            axes[row, col_left + 2].axis("off")
            continue

        dataset = datasets[di]
        dsub = df[df["dataset"] == dataset].copy()

        left = dsub[(dsub["n_bits"] == n_bits_left) | (dsub["n_bits"] == 0)]
        right = dsub[(dsub["n_bits"] == n_bits_right) | (dsub["n_bits"] == 0)]

        _plot_panel(axes[row, col_left], left, dataset, n_bits_left)
        _plot_panel(axes[row, col_left + 2], right, dataset, n_bits_right)

    # hide any unused panels if < 4 datasets
    for r in range(2):
        for c in range(4):
            if not axes[r, c].has_data():
                axes[r, c].axis("off")

    # legend + layout (keep legend inside canvas)
    fig.subplots_adjust(bottom=0.2)
    fig.legend(
        handles_for_legend.values(),
        handles_for_legend.keys(),
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.0),
        frameon=True,
    )

    fig.tight_layout(rect=[0, 0.15, 1, 1])
    return fig, axes


def plot_ami_vs_memory_bits(df, n_bits_left=1, n_bits_right=16):
    """
    2x4 grid (4 datasets):
      - Columns 0-1: dataset panels for n_bits = n_bits_left
      - Columns 2-3: dataset panels for n_bits = n_bits_right
    Within each panel:
      x = memory_bits_mean, y = accuracy_mean
      color = (model, variant) pair
      marker = rho
      points only (no connecting lines).
    """
    required = {"dataset", "model", "variant", "rho", "n_bits", "ari_train_mean",
                "ami_train_mean",
                "memory_bits_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    # pick up to 4 datasets
    datasets = list(pd.unique(df["dataset"]))[:4]

    # aesthetics mappings computed once from full df (stable across panels)
    mv_pairs = df[["model", "variant"]].drop_duplicates().apply(tuple, axis=1).tolist()
    mv_pairs = sorted(mv_pairs)

    mv_to_color = {mv: okabe_ito[i % len(okabe_ito)] for i, mv in enumerate(mv_pairs)}

    rhos = sorted(pd.unique(df["rho"]))
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "H", "p", "8"]
    rho_to_marker = {rho: marker_cycle[i % len(marker_cycle)] for i, rho in enumerate(rhos)}

    # layout: 2 rows x 4 cols
    fig, axes = plt.subplots(2, 4, figsize=(22, 9), sharex=False, sharey=False)

    positions = [
        (0, 0, 0), (0, 1, 1),
        (1, 0, 2), (1, 1, 3),
    ]  # (row, col_left_base, dataset_index)

    handles_for_legend = {}

    def _plot_panel(ax, dsub_bits, dataset, bits_value):
        # points only; group only for labeling/consistent mapping
        for (model, variant, rho), g in dsub_bits.groupby(["model", "variant", "rho"], dropna=False):
            if g.empty:
                continue

            x = g["memory_bits_mean"].to_numpy()
            y = g["ami_train_mean"].to_numpy()

            # avoid issues with log scale if zeros slip in
            mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
            if not np.any(mask):
                continue

            color = mv_to_color[(model, variant)]

            model_clean = model.replace("_binary", "").replace("_continuous", "")
            variant_clean = variant.replace("(loss=log_loss)", "")
            label = f"{model_clean} | {variant_clean}"
            if "ART" in model_clean:
                label += f" | rho={rho}"
                marker = rho_to_marker[rho]
            else:
                marker = "X"

            sc = ax.scatter(
                x[mask],
                y[mask],
                color=color,
                marker=marker,
                s=45 if "Binary" not in model_clean else 90,
                alpha=0.9,
                label=label,
            )
            handles_for_legend.setdefault(label, sc)

        title_map = {
            "mnist": "MNIST",
            "uci_har": "HAR",
            "pendigits": "Pendigits",
            "Spambase": "Spambase",
        }
        dataset_title = title_map.get(str(dataset), str(dataset))

        ax.set_title(f"{dataset_title} | n_bits={bits_value}")
        ax.set_xlabel("Memory Bits")
        ax.set_ylabel("AMI (train)")
        ax.set_xscale("log")
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

        # set xlim based on data in this panel (robust)
        xs = dsub_bits["memory_bits_mean"].to_numpy()
        xs = xs[np.isfinite(xs) & (xs > 0)]
        if xs.size:
            lo, hi = np.min(xs), np.max(xs)
            # ax.set_xlim([lo / 1.2, hi * 1.2])

    # plot each dataset in left/right panels
    for row, col_left, di in positions:
        if di >= len(datasets):
            axes[row, col_left].axis("off")
            axes[row, col_left + 2].axis("off")
            continue

        dataset = datasets[di]
        dsub = df[df["dataset"] == dataset].copy()

        left = dsub[(dsub["n_bits"] == n_bits_left) | (dsub["n_bits"] == 0)]
        right = dsub[(dsub["n_bits"] == n_bits_right) | (dsub["n_bits"] == 0)]

        _plot_panel(axes[row, col_left], left, dataset, n_bits_left)
        _plot_panel(axes[row, col_left + 2], right, dataset, n_bits_right)

    # hide any unused panels if < 4 datasets
    for r in range(2):
        for c in range(4):
            if not axes[r, c].has_data():
                axes[r, c].axis("off")

    # legend + layout (keep legend inside canvas)
    fig.subplots_adjust(bottom=0.2)
    fig.legend(
        handles_for_legend.values(),
        handles_for_legend.keys(),
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.0),
        frameon=True,
    )

    fig.tight_layout(rect=[0, 0.15, 1, 1])
    return fig, axes

def plot_fit_time_vs_predict_time(df, n_bits_left=1, n_bits_right=16):
    """
    2x4 grid (4 datasets):
      - Columns 0-1: dataset panels for n_bits = n_bits_left
      - Columns 2-3: dataset panels for n_bits = n_bits_right
    Within each panel:
      x = train_time_s_mean, y = accuracy_mean
      color = (model, variant) pair
      marker = rho
      points only (no connecting lines).
    """
    required = {"dataset", "model", "variant", "rho", "n_bits", "train_time_s_mean",
                "pred_time_s_mean", "n_test_first"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    # pick up to 4 datasets
    datasets = list(pd.unique(df["dataset"]))[:4]

    # aesthetics mappings computed once from full df (stable across panels)
    mv_pairs = df[["model", "variant"]].drop_duplicates().apply(tuple, axis=1).tolist()
    mv_pairs = sorted(mv_pairs)

    mv_to_color = {mv: okabe_ito[i % len(okabe_ito)] for i, mv in enumerate(mv_pairs)}

    rhos = sorted(pd.unique(df["rho"]))
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "H", "p", "8"]
    rho_to_marker = {rho: marker_cycle[i % len(marker_cycle)] for i, rho in enumerate(rhos)}

    # layout: 2 rows x 4 cols
    fig, axes = plt.subplots(2, 4, figsize=(22, 9), sharex=False, sharey=False)

    # map datasets to panel positions:
    # (row 0,col 0) dataset0 left, (row 0,col 2) dataset0 right
    # (row 0,col 1) dataset1 left, (row 0,col 3) dataset1 right
    # (row 1,col 0) dataset2 left, (row 1,col 2) dataset2 right
    # (row 1,col 1) dataset3 left, (row 1,col 3) dataset3 right
    positions = [
        (0, 0, 0), (0, 1, 1),
        (1, 0, 2), (1, 1, 3),
    ]  # (row, col_left_base, dataset_index)

    handles_for_legend = {}

    def _plot_panel(ax, dsub_bits, dataset, bits_value):
        # points only; group only for labeling/consistent mapping
        for (model, variant, rho), g in dsub_bits.groupby(["model", "variant", "rho"], dropna=False):
            if g.empty:
                continue
            color = mv_to_color[(model, variant)]
            model_clean = model.replace("_binary", "").replace("_continuous", "")
            variant_clean = variant.replace("(loss=log_loss)", "")
            label = f"{model_clean} | {variant_clean}"
            if "ART" in model_clean:
                label += f" | rho={rho}"
                marker = rho_to_marker[rho]
            else:
                marker = "X"
            pred_time_s_mean = g["pred_time_s_mean"].to_numpy()
            pred_time_s_mean_per = pred_time_s_mean/g["n_test_first"]
            sc = ax.scatter(
                pred_time_s_mean_per,
                g["train_time_s_mean"].to_numpy(),
                color=color,
                marker=marker,
                s=45 if "Binary" not in model_clean else 90,
                alpha=0.9,
                label=label,
            )

            handles_for_legend.setdefault(label, sc)
        title_map = {
            "mnist": "MNIST",
            "uci_har": "HAR",
            "pendigits": "Pendigits",
            "Spambase": "Spambase"
        }
        dataset_title = title_map[str(dataset)]
        ax.set_title(f"{dataset_title} | n_bits={bits_value}")
        ax.set_xlabel("Per-Sample Predict Time (secs)")
        ax.set_ylabel("Train Time (secs)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim([1e-7, 1])
        ax.set_ylim([1e-3, 1e4])
        ax.grid(True, alpha=0.3)

    # plot each dataset in left/right panels
    for row, col_left, di in positions:
        if di >= len(datasets):
            # hide both panels if dataset missing
            axes[row, col_left].axis("off")
            axes[row, col_left + 2].axis("off")
            continue

        dataset = datasets[di]
        dsub = df[df["dataset"] == dataset].copy()

        left = dsub[(dsub["n_bits"] == n_bits_left) | (dsub["n_bits"] == 0)]
        right = dsub[(dsub["n_bits"] == n_bits_right) | (dsub["n_bits"] == 0)]

        _plot_panel(axes[row, col_left], left, dataset, n_bits_left)
        _plot_panel(axes[row, col_left + 2], right, dataset, n_bits_right)

    # hide any unused panels if < 4 datasets (already handled above, but safe)
    for r in range(2):
        for c in range(4):
            if not axes[r, c].has_data():
                axes[r, c].axis("off")

    # single figure legend
    fig.legend(
        handles_for_legend.values(),
        handles_for_legend.keys(),
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.0),
        frameon=True,
    )

    fig.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    return fig, axes


if __name__ == "__main__":
    df = load_and_aggregate_data()
    filt = df[(df["rho"] == 0.5) | (df["rho"] == 0.75)]

    fig, axes = plot_ari_vs_memory_bits(filt, n_bits_left=1, n_bits_right=16)
    fig.savefig("art_ari_vs_mem_bits.png")

    fig, axes = plot_ami_vs_memory_bits(filt, n_bits_left=1, n_bits_right=16)
    fig.savefig("art_ami_vs_mem_bits.png")

    fig, axes = plot_fit_time_vs_predict_time(filt, n_bits_left=1, n_bits_right=16)
    fig.savefig("art_fit_vs_pred_time.png")

    plt.show()
