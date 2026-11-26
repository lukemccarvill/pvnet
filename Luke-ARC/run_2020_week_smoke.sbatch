#!/bin/bash
#SBATCH --job-name=pvnet-week-smoke
#SBATCH --partition=medium
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=pvnet-week-smoke-%j.out

set -euo pipefail

if [[ -z "${PVNET_DATA_ROOT:-}" ]]; then
    echo "Set PVNET_DATA_ROOT before submitting (for example: export PVNET_DATA_ROOT=\"\$DATA/pvnet\")." >&2
    exit 2
fi
if [[ -z "${PVNET_VENV:-}" || ! -x "${PVNET_VENV}/bin/python" ]]; then
    echo "Set PVNET_VENV to the prepared PVNet virtual environment before submitting." >&2
    exit 2
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:?Submit from the PVNet repository or Luke-ARC directory}"
if [[ -f "$SUBMIT_DIR/pyproject.toml" ]]; then
    REPO_ROOT="$SUBMIT_DIR"
elif [[ -f "$SUBMIT_DIR/../pyproject.toml" ]]; then
    REPO_ROOT="$(cd "$SUBMIT_DIR/.." && pwd)"
else
    echo "Could not locate PVNet repository from SLURM_SUBMIT_DIR=$SUBMIT_DIR" >&2
    exit 2
fi
cd "$REPO_ROOT"
"${PVNET_VENV}/bin/python" "$REPO_ROOT/Luke-ARC/run_2020_week_smoke.py"
