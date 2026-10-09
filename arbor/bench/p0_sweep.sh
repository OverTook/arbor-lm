#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
run() { timeout 2400 ~/arbor-venv/bin/python -m arbor.bench.train_throughput --opt adamw8bit --compile --steps 14 "$@" 2>&1 | grep -E '^\{|Error|Traceback'; }
run --size M --variants A3,A0 --micro 3
run --size S --variants A0,A1,A2,A3 --micro 3
run --size S --variants A0,A1,A2,A3 --micro 4
