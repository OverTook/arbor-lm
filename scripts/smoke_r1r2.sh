#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=true PYTHONPATH="$PWD"
rm -rf ~/arbor-data/smoke
~/arbor-venv/bin/python -m arbor.train.r1 --config configs/smoke_r1.yaml > /dev/null 2>&1
~/arbor-venv/bin/python -m arbor.train.r2 --config configs/smoke_r2.yaml > /dev/null 2>&1
echo SMOKE_DONE
