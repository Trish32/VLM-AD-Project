#!/usr/bin/env bash
# Extract the first N scenarios from the downloaded AV2 val.tar (sequential, so this only
# reads a prefix of the archive). Usage: ./extract_val_subset.sh [N] [AV2_ROOT]
set -e
N="${1:-150}"
ROOT="${2:-/Users/trish/Downloads/Argoverse 2}"
cd "$ROOT"
tar tf val.tar | grep -E '^val/[^/]+/$' | head -"$N" > /tmp/val_subset.txt
tar xf val.tar -T /tmp/val_subset.txt
echo "extracted $(ls val | wc -l) scenarios into $ROOT/val"
