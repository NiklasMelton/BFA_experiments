# merge_parts.py
from __future__ import annotations

import argparse
import glob
import os
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts_dir", type=str, default="bfa_art_results_parts")
    ap.add_argument("--out", type=str, default="bfa_art_results.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.parts_dir, "part_*.csv")))
    if not paths:
        raise RuntimeError(f"No part files found in {args.parts_dir}")

    dfs = [pd.read_csv(p) for p in paths]
    out = pd.concat(dfs, ignore_index=True)

    # Optional: sort for readability
    sort_cols = [c for c in ["dataset", "model", "random_state", "n_bits", "rho"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    out.to_csv(args.out, index=False)
    print(f"[ok] merged {len(paths)} parts -> {args.out} (rows={len(out)})")


if __name__ == "__main__":
    main()
