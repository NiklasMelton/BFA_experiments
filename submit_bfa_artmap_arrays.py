#!/usr/bin/env python3
import math
import subprocess

SBATCH_FILE = "BFA_experiments/run_artmap_parallel.sbatch"
MAX_ARRAY = 1001
THROTTLE = 25

def get_n_trials() -> int:
    # Change this to the correct script that prints N
    out = subprocess.check_output(["python", "artmap_array.py", "--print_ntrials"], text=True)
    return int(out.strip())

def main():
    n = get_n_trials()
    last = n - 1
    chunks = math.ceil(n / MAX_ARRAY)
    print(f"Total trials: {n} (0..{last}), chunks: {chunks}")

    for i in range(chunks):
        start = i * MAX_ARRAY
        end = min((i + 1) * MAX_ARRAY - 1, last)
        array_spec = f"{start}-{end}%{THROTTLE}"
        cmd = ["sbatch", f"--array={array_spec}", SBATCH_FILE]
        print(" ".join(cmd))
        subprocess.check_call(cmd)

if __name__ == "__main__":
    main()
