#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for csv in "$SCRIPT_DIR"/timings/*.csv; do
    base="$(basename "$csv" .csv)"
    jpg="$SCRIPT_DIR/timings/$base.jpg"
    echo "Plotting $csv -> $jpg"
    python "$SCRIPT_DIR/plot_benchmark.py" "$csv" "$jpg"
done
