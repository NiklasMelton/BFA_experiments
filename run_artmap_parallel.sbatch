#!/bin/bash
#SBATCH --job-name=ARTMAP_exp
#SBATCH --partition=requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00


#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=nmmz76@umsystem.edu

module load python/3.12.1
source .venv/bin/activate

cd $SLURM_SUBMIT_DIR
mkdir -p logs

python artmap_tests_parallel.py --out_dir results_parts
