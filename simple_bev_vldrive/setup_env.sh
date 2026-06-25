#!/usr/bin/env zsh
set -euo pipefail

ENV_NAME="simple_bev_vldrive"

if command -v conda >/dev/null 2>&1; then
  CONDA_CMD="conda"
elif command -v mamba >/dev/null 2>&1; then
  CONDA_CMD="mamba"
else
  echo "ERROR: conda or mamba is required to create the environment."
  exit 1
fi

echo "Creating or updating conda environment '${ENV_NAME}' from environment.yml..."
$CONDA_CMD env create -f environment.yml || $CONDA_CMD env update -f environment.yml

echo "Environment ready. Activate it with: conda activate ${ENV_NAME}"
echo "Then verify MPS support with: python check_mps.py"
