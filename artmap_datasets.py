# Binary ARTMAP tests
from __future__ import annotations

from typing import Callable, Dict, Any, List

import os
import warnings
import traceback
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

from experiment_utils import (
DatasetBundle, TestSpec, split_data, suppress_stdout_stderr, safe_save_csv,
load_uci_via_api_stub, load_mnist, load_retinaface_lfw_dataset, load_uci_har_zip,
run_binaryfuzzyartmap, run_sgd_binary, run_fuzzyartmap_binary, run_art1map,
run_sgd_continuous, run_fuzzyartmap_continuous, run_multinomial_nb_binary
)


# ----------------------------
# Configuration
# ----------------------------

RANDOM_STATE = 0          # dont change
RANDOM_STATE_VALUES = [0, 1, 2, 3, 4]   # example: 5 seeds
N_BITS = 1                 # dont change
N_BITS_VALUES = [1, 4, 8, 16]
RHO = 0.0                  # dont change
RHO_VALUES = [0.0, 0.5, 0.7, 0.9]


RESULTS_PATH = "results.csv"      # simple + portable
RESULTS_PATH_TMP = "results.tmp.csv"


# ----------------------------
# Dataset containers
# ----------------------------


def run_all(
    datasets: List[Callable[[], DatasetBundle]],
    tests: List[TestSpec],
    n_bits_values: List[int],
    results_path: str = RESULTS_PATH,
) -> pd.DataFrame:
    global N_BITS  # we set this per-run for tests that use binarization

    # resume if file exists
    if os.path.exists(results_path):
        results = pd.read_csv(results_path)
    else:
        results = pd.DataFrame()

    for load_dataset in tqdm(datasets, desc="Datasets"):
        with suppress_stdout_stderr(), warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The number of unique classes is greater than 50% of the number of samples\.",
                category=UserWarning,
                module=r"sklearn\.utils\.multiclass",
            )
            ds = load_dataset()
        X, y = ds.X, ds.y
        idx = np.argsort(y)
        X = X[idx]
        y = y[idx]
        for rs in tqdm(RANDOM_STATE_VALUES, desc="Random state", leave=False):
            RANDOM_STATE = int(rs)
            for test in tqdm(tests, desc=f"Tests ({ds.name})", leave=False):
                bits_to_run = n_bits_values if test.depends_on_n_bits else [None]
                rho_values = RHO_VALUES if "ART" in test.name else [0.0]

                for rho in tqdm(rho_values, desc=f"rho ({test.name})", leave=False):
                    RHO = float(rho)

                    for n_bits in tqdm(bits_to_run, desc=f"N_BITS ({test.name})", leave=False):
                        kwargs = {
                            "n_bits": int(n_bits) if n_bits is not None else 0,
                            "rho": float(RHO),
                            "random_state": int(RANDOM_STATE)
                        }

                        row: Dict[str, Any] = {
                            "dataset": ds.name,
                            "model": test.name,
                            "random_state": int(RANDOM_STATE),
                            "n_samples": int(X.shape[0]),
                            "n_features": int(X.shape[1]),
                            "n_classes": int(len(np.unique(y))),
                            # For tests that don't depend on N_BITS, set 0 to mean "N/A"
                            "n_bits": int(N_BITS) if n_bits is not None else 0,
                            "rho": float(RHO),
                            **{f"ds_{k}": v for k, v in ds.meta.items()},
                        }

                        try:
                            X_tr, X_te, y_tr, y_te = split_data(X, y, random_state=RANDOM_STATE)
                            row["n_train"] = int(len(y_tr))
                            row["n_test"] = int(len(y_te))
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
                            print(f"Error running {ds.name}", e)

                        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
                        safe_save_csv(results, results_path, RESULTS_PATH_TMP)

    return results



def main() -> None:
    # Datasets: keep HAR now; add more later by appending lambdas that call load_uci_via_api_stub(...)
    DATASETS = [
        lambda: load_uci_har_zip("UCI HAR Dataset.zip"),
        # lambda: load_uci_via_api_stub(53, "iris"),
        # lambda: load_uci_via_api_stub(109, "wine"),
        # lambda: load_uci_via_api_stub(42, "Glass Identification"),
        # lambda: load_uci_via_api_stub(17, "Breast Cancer Wisconsin"),
        # lambda: load_uci_via_api_stub(52, "Ionosphere"),
        lambda: load_uci_via_api_stub(94, "Spambase"),
        # lambda: load_uci_via_api_stub(80, "optdigits"),
        lambda: load_uci_via_api_stub(81, "pendigits"),
        # lambda: load_uci_via_api_stub(159, "Magic Gamma Telescope"),
        # lambda: load_retinaface_lfw_dataset(RANDOM_STATE),
        load_mnist,
    ]

    TESTS = [
        TestSpec("FuzzyARTMAP", run_fuzzyartmap_binary, depends_on_n_bits=True),
        TestSpec("FuzzyARTMAP", run_fuzzyartmap_continuous, depends_on_n_bits=False),
        # run once
        TestSpec("BinaryFuzzyARTMAP", run_binaryfuzzyartmap, depends_on_n_bits=True),
        TestSpec("ART1MAP", run_art1map, depends_on_n_bits=True),
        TestSpec("MultinomialNB", run_multinomial_nb_binary, depends_on_n_bits=True),
        TestSpec("SGDClassifier", run_sgd_binary, depends_on_n_bits=True),
        TestSpec("SGDClassifier", run_sgd_continuous, depends_on_n_bits=False),
        # also run once
    ]

    run_all(DATASETS, TESTS, n_bits_values=N_BITS_VALUES, results_path=RESULTS_PATH)


if __name__ == "__main__":
    main()
