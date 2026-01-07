#!/bin/bash
#SBATCH --job-name=ARTMAP_exp
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --constraint=skylake&40CPU

#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=nmmz76@umsystem.edu

module load python/3.12.1

cd $SLURM_SUBMIT_DIR
mkdir -p logs bfa_artmap_results_parts

source /home/nmmz76/BFA_Experiments/.venv/bin/activate

OFFSET=${OFFSET:-0}
TASK_ID=$((OFFSET + SLURM_ARRAY_TASK_ID))

python3 BFA_experiments/artmap_tests_parallel.py --task_id ${TASK_ID}
