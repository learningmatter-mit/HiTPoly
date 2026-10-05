#!/bin/bash
# One SLURM array task per CSV row: build the box, then equilibrate + 100 ns production.
#
#   sbatch --array=0-27 run_copolymer_array.sh case_study_reruns_P5_P14.csv ~/rerun
#
# Rows whose simulation already finished exit right away, so the full range is safe.
# Per-row logs: <results>/logs/<jobid>_<task>.out; status: <results>/status/<name>.csv
#SBATCH --job-name=copolymer
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH --time=4-00:00:00
#SBATCH --output=copolymer_%A_%a.out

CSV=$(realpath "${1:?usage: sbatch --array=... run_copolymer_array.sh <csv> <results_dir>}")
RESULTS=$(realpath -m "${2:?results dir required}")
HITPOLY_DIR=${HITPOLY_DIR:-$SLURM_SUBMIT_DIR}

mkdir -p "$RESULTS/logs"
exec > "$RESULTS/logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>&1

source /etc/profile
source "$HOME/.bashrc"
conda activate htvs

python -u "$HITPOLY_DIR/run_copolymer.py" --csv "$CSV" --row "$SLURM_ARRAY_TASK_ID" --results "$RESULTS"
