#!/usr/bin/env python3
from pathlib import Path
import math
import subprocess

MAX_ARRAY = 1001
THROTTLE = 25

def get_n_trials(repo_dir: Path) -> int:
    # run from inside the repo so relative imports/paths work
    out = subprocess.check_output(
        ["python3", "artmap_tests_parallel.py", "--print_ntrials"],
        text=True,
        cwd=str(repo_dir),
    )
    return int(out.strip())

def main():
    repo_dir = Path(__file__).resolve().parent           # .../BFA_experiments
    sbatch_file = repo_dir / "run_artmap_parallel.sbatch"

    n = get_n_trials(repo_dir)
    last = n - 1
    chunks = math.ceil(n / MAX_ARRAY)
    print(f"Total trials: {n} (0..{last}), chunks: {chunks}")

    for i in range(chunks):
        start = i * MAX_ARRAY
        end = min((i + 1) * MAX_ARRAY - 1, last)
        array_spec = f"{start}-{end}%{THROTTLE}"
        cmd = ["sbatch", f"--array={array_spec}", str(sbatch_file)]
        print(" ".join(cmd))
        subprocess.check_call(cmd)

if __name__ == "__main__":
    main()
