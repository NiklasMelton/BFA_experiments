#!/usr/bin/env python3
from pathlib import Path
import subprocess

MAX_ARRAY = 1001
THROTTLE = 25

def get_n_trials(repo_dir: Path) -> int:
    out = subprocess.check_output(
        ["python3", "cl_tests_parallel.py", "--print_ntrials"],
        cwd=str(repo_dir),
        text=True,
    )
    return int(out.strip())

def main():
    repo_dir = Path(__file__).resolve().parent           # .../BFA_experiments
    sbatch_file = repo_dir / "run_cl_parallel.sbatch"

    n = get_n_trials(repo_dir)
    last = n - 1
    print(f"Total trials: {n} (0..{last})")

    offset = 0
    while offset < n:
        count = min(MAX_ARRAY, n - offset)     # number of tasks in this chunk
        end = count - 1                        # 0-based end index for this array
        array_spec = f"0-{end}%{THROTTLE}"

        cmd = [
            "sbatch",
            f"--array={array_spec}",
            f"--export=ALL,OFFSET={offset}",
            str(sbatch_file),
        ]
        print(" ".join(cmd))
        subprocess.check_call(cmd)

        offset += count

if __name__ == "__main__":
    main()
