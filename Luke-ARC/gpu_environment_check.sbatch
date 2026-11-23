#!/bin/bash
#SBATCH --job-name=pvnet-envcheck
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --output=pvnet-envcheck-%j.out

set -euo pipefail

if [[ -z "${PVNET_VENV:-}" ]]; then
    echo "Set PVNET_VENV to the ARC virtual-environment directory before submitting." >&2
    exit 2
fi
if [[ ! -f "$PVNET_VENV/bin/activate" ]]; then
    echo "No virtual environment activation script at $PVNET_VENV/bin/activate" >&2
    exit 2
fi

source "$PVNET_VENV/bin/activate"
cd "${SLURM_SUBMIT_DIR:?Submit this job from Luke-ARC}"
python check_environment.py
