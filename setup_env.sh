#!/usr/bin/env zsh
# Create/update the unified `vldrive` conda env that runs all VLMProjects
# projects on an Apple-Silicon MacBook (MPS). See environment.yml for details.
set -euo pipefail

ENV_NAME="vldrive"
HERE="$(cd "$(dirname "$0")" && pwd)"

if command -v conda >/dev/null 2>&1; then
  CONDA_CMD="conda"
elif command -v mamba >/dev/null 2>&1; then
  CONDA_CMD="mamba"
else
  echo "ERROR: conda or mamba is required to create the environment."
  exit 1
fi

echo "Creating or updating conda environment '${ENV_NAME}' from environment.yml..."
$CONDA_CMD env create -f "${HERE}/environment.yml" \
  || $CONDA_CMD env update -f "${HERE}/environment.yml"

echo
echo "Environment ready. Activate it with: conda activate ${ENV_NAME}"
echo "Then verify MPS support with:        python ${HERE}/check_mps.py"
echo
echo "If you hit 'OMP: Error #15' on macOS, run with KMP_DUPLICATE_LIB_OK=TRUE"
echo "or apply the libomp symlink fix described in environment.yml."
