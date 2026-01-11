import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_and_aggregate_data():
    df = pd.read_csv("bfa_artmap_results.csv")
    errors = df["error"].unique()
    print("Errors detected")
    print(errors)
    print("Filtering results")
    df = df[df["error"].isna()]


    group_cols = ["dataset", "model", "rho", "n_bits", "variant"]

    mean_std_cols = ["accuracy", "train_time_s", "pred_time_s", "n_clusters",
                     "memory_bits"]
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


def plot_accuracy_vs_fit_time(df):
    """
    df: summary dataframe with columns:
        - dataset, model, variant, rho, n_bits
        - accuracy_mean, train_time_s_mean  (from your groupby/agg)
    Makes a 2x2 grid (4 datasets). Within each subplot:
      x = train_time_s_mean, y = accuracy_mean
      color = (model, variant) pair
      marker = rho
      lines connect points across n_bits within each (model, variant, rho)
    """
    # ---- basic checks / column names ----
    required = {"dataset", "model", "variant", "rho", "n_bits", "accuracy_mean", "train_time_s_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    # ---- determine datasets and layout ----
    datasets = list(pd.unique(df["dataset"]))
    if len(datasets) != 4:
        # still plot whatever you have, but keep 2x2 as requested
        datasets = datasets[:4]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=False, sharey=False)
    axes = axes.ravel()

    # ---- build aesthetics mappings ----
    # color by (model, variant)
    mv_pairs = df[["model", "variant"]].drop_duplicates().apply(tuple, axis=1).tolist()
    print(np.unique(mv_pairs))
    mv_pairs = sorted(mv_pairs)

    cmap = plt.get_cmap("tab20")  # enough distinct colors for many pairs
    mv_to_color = {mv: cmap(i % cmap.N) for i, mv in enumerate(mv_pairs)}

    # marker by rho
    rhos = sorted(pd.unique(df["rho"]))
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "H", "p", "8"]
    rho_to_marker = {rho: marker_cycle[i % len(marker_cycle)] for i, rho in enumerate(rhos)}

    # ---- plot ----
    handles_for_legend = {}
    for ax, dataset in zip(axes, datasets):
        dsub = df[df["dataset"] == dataset].copy()

        # group by (model, variant, rho) so within-group we connect across n_bits
        for (model, variant, rho), g in dsub.groupby(["model", "variant", "rho"], dropna=False):
            g = g.sort_values("n_bits")  # connect in increasing n_bits
            color = mv_to_color[(model, variant)]
            marker = rho_to_marker[rho]
            label = f"{model} | {variant} | rho={rho}"

            line = ax.plot(
                g["train_time_s_mean"].to_numpy(),
                g["accuracy_mean"].to_numpy(),
                color=color,
                marker=marker,
                linewidth=1.5,
                markersize=6,
                label=label,
            )[0]

            # store one handle per label so we can make a single figure-level legend
            handles_for_legend.setdefault(label, line)

        ax.set_title(str(dataset))
        ax.set_xlabel("train_time_s (mean)")
        ax.set_ylabel("accuracy (mean)")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    # If fewer than 4 datasets, hide unused axes (but keep the 2x2 figure)
    for j in range(len(datasets), 4):
        axes[j].axis("off")

    # ---- single legend for the whole figure ----
    fig.legend(
        handles_for_legend.values(),
        handles_for_legend.keys(),
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )

    fig.tight_layout()
    plt.subplots_adjust(bottom=0.20)  # make room for legend
    return fig, axes

if __name__ == "__main__":
    df = load_and_aggregate_data()
    fig, axes = plot_accuracy_vs_fit_time(df)
    plt.show()
