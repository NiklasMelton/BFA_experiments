from __future__ import annotations

import os
from pathlib import Path
import argparse
import warnings
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional

import numpy as np
import pandas as pd

from experiment_utils import (
    DatasetBundle,
    suppress_stdout_stderr,
    load_mnist,
    cl_run_binaryfuzzyartmap,
)

# ----------------------------
# Thread pinning (per Slurm task / process)
# ----------------------------

def _pin_threads_from_slurm() -> None:
    """Pin BLAS/OpenMP/framework threads to SLURM_CPUS_PER_TASK (default 4 here)."""
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    os.environ["OMP_NUM_THREADS"] = str(cpus)
    os.environ["MKL_NUM_THREADS"] = str(cpus)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpus)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpus)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpus)


# ----------------------------
# Fixed experiment config
# ----------------------------

RANDOM_STATE_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, 8]
FIXED_N_BITS = 8
FIXED_RHO = 0.8


@dataclass(frozen=True)
class Trial:
    random_state: int


def _load_dataset() -> DatasetBundle:
    """MNIST only."""
    with suppress_stdout_stderr(), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The number of unique classes is greater than 50% of the number of samples\.",
            category=UserWarning,
            module=r"sklearn\.utils\.multiclass",
        )
        ds = load_mnist()
    return ds


def run_one_trial(trial: Trial) -> Dict[str, Any]:
    ds = _load_dataset()
    X, y = ds.X, ds.y

    # Keep your original "sort full dataset by label" behavior (optional).
    # Note: your CL function sorts X_train by class after splitting anyway,
    # but leaving this preserves your prior preprocessing convention.
    idx = np.argsort(y, kind="stable")
    X = X[idx]
    y = y[idx]

    row: Dict[str, Any] = {
        "dataset": ds.name,
        "model": "BinaryFuzzyARTMAP_CL",
        "random_state": int(trial.random_state),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(np.unique(y))),
        "n_bits": int(FIXED_N_BITS),
        "rho": float(FIXED_RHO),
        **{f"ds_{k}": v for k, v in ds.meta.items()},
    }

    try:
        with suppress_stdout_stderr(), warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The number of unique classes is greater than 50% of the number of samples\.",
                category=UserWarning,
                module=r"sklearn\.utils\.multiclass",
            )
            out = cl_run_binaryfuzzyartmap(
                X,
                y,
                rho=float(FIXED_RHO),
                n_bits=int(FIXED_N_BITS),
                random_state=int(trial.random_state),
                # batch_size can also be set via CLI (see main)
                # batch_size=<passed in>
            )

        # out is your reduced dict with arrays
        row.update(out)
        row["status"] = "ok"
        row["error"] = ""
        row["traceback"] = ""
    except Exception as e:
        row["status"] = "error"
        row["error"] = repr(e)
        row["traceback"] = traceback.format_exc()
        raise e

    return row


def _write_part(out_dir: str, task_id: int, row: Dict[str, Any]) -> str:
    """
    CSV will stringify numpy arrays.
    If you want to preserve arrays losslessly, also write a .npz alongside the CSV.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"part_{task_id:06d}.csv")

    # Ensure arrays are serializable in CSV (stringified)
    row = dict(row)
    if "class_acc_by_batch" in row and isinstance(row["class_acc_by_batch"], np.ndarray):
        row["class_acc_by_batch"] = row["class_acc_by_batch"].tolist()
    if "classes" in row and isinstance(row["classes"], np.ndarray):
        row["classes"] = row["classes"].tolist()

    pd.DataFrame([row]).to_csv(path, index=False)
    return path


def _write_npz(out_dir: str, task_id: int, row: Dict[str, Any]) -> str:
    """Write arrays losslessly."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"part_{task_id:06d}.npz")

    arrays = {}
    meta = {}
    for k, v in row.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        else:
            meta[k] = v

    # store meta as a 0-d object array
    np.savez_compressed(path, **arrays, meta=np.array(meta, dtype=object))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="bfa_cl_results_parts")
    parser.add_argument("--task_id", type=int, default=None,
                        help="Explicit task id. If omitted, uses SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--print_ntrials", action="store_true",
                        help="Print total # trials and exit (useful to size --array).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing part file for this task id.")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Batch size for partial_fit incremental training.")
    parser.add_argument("--write_npz", action="store_true",
                        help="Also write a .npz with arrays stored losslessly.")
    args = parser.parse_args()

    _pin_threads_from_slurm()

    trials: List[Trial] = [Trial(random_state=rs) for rs in RANDOM_STATE_VALUES]

    if args.print_ntrials:
        print(len(trials))
        return

    task_id: Optional[int] = args.task_id
    if task_id is None:
        env = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env is None:
            raise RuntimeError("No --task_id provided and SLURM_ARRAY_TASK_ID is not set.")
        task_id = int(env)

    if not (0 <= task_id < len(trials)):
        raise IndexError(f"task_id {task_id} out of range [0, {len(trials)-1}]")

    part_path_csv = os.path.join(args.out_dir, f"part_{task_id:06d}.csv")
    # part_path_npz = os.path.join(args.out_dir, f"part_{task_id:06d}.npz")

    if os.path.exists(part_path_csv) and not args.overwrite:
        print(f"[skip] {part_path_csv} exists (use --overwrite to rerun).")
        return

    # Run trial
    row = run_one_trial(trials[task_id])


    written_csv = _write_part(args.out_dir, task_id, row)
    print(f"[ok] wrote {written_csv}")

    if args.write_npz:
        written_npz = _write_npz(args.out_dir, task_id, row)
        print(f"[ok] wrote {written_npz}")


if __name__ == "__main__":
    main()
