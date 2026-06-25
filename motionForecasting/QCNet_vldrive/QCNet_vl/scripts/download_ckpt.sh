#!/usr/bin/env bash
# Download the official QCNet_AV2 checkpoint (used for both AV2 val and test in the README).
set -e
cd "$(dirname "$0")/.."
mkdir -p ckpt
gdown 1OKBytt6N6BdRa9FWmS7F1-YvF0YectBv -O ckpt/QCNet_AV2.ckpt
echo "checkpoint -> ckpt/QCNet_AV2.ckpt"
