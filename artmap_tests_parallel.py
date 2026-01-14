# artmap_array.py
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
    TestSpec,
    suppress_stdout_stderr,
    load_uci_via_api_stub,
    load_mnist,
    load_uci_har_zip,
    run_binaryfuzzyartmap,
    run_sgd_binary,
    run_fuzzyartmap_binary,
    run_art1map,
    run_sgd_continuous,
    run_fuzzyartmap_continuous,
    run_multinomial_nb_binary,
)

# ----------------------------
# Thread pinning (per Slurm task / process)
# ----------------------------

def _pin_threads_from_slurm() -> None:
    """Pin BLAS/OpenMP/framework threads to SLURM_CPUS_PER_TASK (default 1)."""
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    os.environ["OMP_NUM_THREADS"] = str(cpus)
    os.environ["MKL_NUM_THREADS"] = str(cpus)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpus)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpus)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpus)



# ----------------------------
# Configuration (same defaults as your original)
# ----------------------------

RANDOM_STATE_VALUES = [0, 1, 2, 3, 4]
N_BITS_VALUES = [1, 4, 8, 16]
RHO_VALUES = [0.0, 0.25, 0.5, 0.75, 0.9]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    loader: Callable[[], DatasetBundle]


@dataclass(frozen=True)
class Trial:
    dataset_name: str
    dataset_index: int
    random_state: int
    test_name: str
    test_index: int
    depends_on_n_bits: bool
    n_bits: int          # 0 means "N/A"
    rho: float


def _build_datasets() -> List[DatasetSpec]:
    """
    IMPORTANT: avoid anonymous lambdas in the trial identity.
    Keep a stable dataset_name for reproducible indexing.
    """
    pkg_dir = Path(__file__).resolve().parent  # /home/nmmz76/BFA_experiments/BFA_experiments
    uci_zip = pkg_dir / "UCI HAR Dataset.zip"
    return [
        # DatasetSpec(
        #     name="UCI_HAR_ZIP",
        #     loader=lambda p=str(uci_zip): load_uci_har_zip(p),
        # ),
        DatasetSpec(name="UCI_94_Spambase", loader=lambda: load_uci_via_api_stub(94, "Spambase")),
        DatasetSpec(name="UCI_81_PenDigits", loader=lambda: load_uci_via_api_stub(81, "pendigits")),
        # DatasetSpec(name="MNIST", loader=load_mnist),
        # DatasetSpec(name="LFW_RetinaFace", loader=lambda: load_retinaface_lfw_dataset(0)),
    ]


def _build_tests() -> List[TestSpec]:
    return [
        TestSpec("FuzzyARTMAP_binary", run_fuzzyartmap_binary, depends_on_n_bits=True),
        # TestSpec("FuzzyARTMAP_continuous", run_fuzzyartmap_continuous, depends_on_n_bits=False),
        #
        # TestSpec("BinaryFuzzyARTMAP", run_binaryfuzzyartmap, depends_on_n_bits=True),
        # TestSpec("ART1MAP", run_art1map, depends_on_n_bits=True),
        #
        # TestSpec("MultinomialNB", run_multinomial_nb_binary, depends_on_n_bits=True),
        #
        # TestSpec("SGDClassifier_binary", run_sgd_binary, depends_on_n_bits=True),
        # TestSpec("SGDClassifier_continuous", run_sgd_continuous, depends_on_n_bits=False),
    ]


def build_trials(
    datasets: List[DatasetSpec],
    tests: List[TestSpec],
    random_states: List[int],
    n_bits_values: List[int],
    rho_values: List[float],
) -> List[Trial]:
    trials: List[Trial] = []
    for di, ds in enumerate(datasets):
        for rs in random_states:
            for ti, test in enumerate(tests):
                bits_to_run = n_bits_values if test.depends_on_n_bits else [0]  # 0 = N/A
                # Match your original behavior: only ART* tests sweep rho
                rhos_to_run = rho_values if ("ART" in test.name) else [0.0]

                for rho in rhos_to_run:
                    for nb in bits_to_run:
                        trials.append(
                            Trial(
                                dataset_name=ds.name,
                                dataset_index=di,
                                random_state=int(rs),
                                test_name=test.name,
                                test_index=ti,
                                depends_on_n_bits=bool(test.depends_on_n_bits),
                                n_bits=int(nb),
                                rho=float(rho),
                            )
                        )
    return trials


def _load_dataset(datasets: List[DatasetSpec], dataset_index: int) -> DatasetBundle:
    ds_spec = datasets[dataset_index]
    with suppress_stdout_stderr(), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The number of unique classes is greater than 50% of the number of samples\.",
            category=UserWarning,
            module=r"sklearn\.utils\.multiclass",
        )
        ds = ds_spec.loader()
    return ds


def run_one_trial(
    trial: Trial,
    datasets: List[DatasetSpec],
    tests: List[TestSpec],
) -> Dict[str, Any]:
    ds = _load_dataset(datasets, trial.dataset_index)
    X, y = ds.X, ds.y

    # Match your original sorting-by-label behavior
    idx = np.argsort(y)
    X = X[idx]
    y = y[idx]

    test = tests[trial.test_index]

    # kwargs expected by your test functions
    kwargs = {
        "n_bits": int(trial.n_bits),
        "rho": float(trial.rho),
        "random_state": int(trial.random_state),
    }

    row: Dict[str, Any] = {
        "dataset": ds.name,
        "dataset_spec": trial.dataset_name,
        "model": test.name,
        "random_state": int(trial.random_state),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(np.unique(y))),
        "n_bits": int(trial.n_bits),   # 0 means "N/A"
        "rho": float(trial.rho),
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
            out = test.fn(X, y, **kwargs)

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
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"part_{task_id:06d}.csv")
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="bfa_artmap_results_parts")
    parser.add_argument("--task_id", type=int, default=None,
                        help="Explicit task id. If omitted, uses SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--print_ntrials", action="store_true",
                        help="Print total # trials and exit (useful to size --array).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing part file for this task id.")
    args = parser.parse_args()

    _pin_threads_from_slurm()

    datasets = _build_datasets()
    tests = _build_tests()
    trials = build_trials(
        datasets=datasets,
        tests=tests,
        random_states=RANDOM_STATE_VALUES,
        n_bits_values=N_BITS_VALUES,
        rho_values=RHO_VALUES,
    )

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

    part_path = os.path.join(args.out_dir, f"part_{task_id:06d}.csv")
    if os.path.exists(part_path) and not args.overwrite:
        print(f"[skip] {part_path} exists (use --overwrite to rerun).")
        return

    row = run_one_trial(trials[task_id], datasets=datasets, tests=tests)
    written = _write_part(args.out_dir, task_id, row)
    print(f"[ok] wrote {written}")


if __name__ == "__main__":
    main()
