#!/usr/bin/env bash
# Builds, runs on a dataset (synthetic if data/ is empty), and packs proof
# of execution into proof_of_execution.tar.gz.
set -euo pipefail
cd "$(dirname "$0")"
INPUT_DIR="${1:-data}"
STREAMS="${2:-4}"

make -s
if ! ls "$INPUT_DIR"/*.png "$INPUT_DIR"/*.jpg >/dev/null 2>&1; then
  python3 scripts/make_synthetic.py --out "$INPUT_DIR" --count 200
fi

rm -rf results && mkdir -p results/samples_before
nvidia-smi > results/gpu_info.txt 2>&1 || true
./bin/edge_pipeline --input_dir "$INPUT_DIR" --output_dir results/output \
  --streams "$STREAMS" | tr '\r' '\n' | tee results/run.log
./bin/edge_pipeline --input_dir "$INPUT_DIR" --output_dir results/output_1stream \
  --streams 1 | tr '\r' '\n' | tail -n 1 | tee -a results/run.log
ls "$INPUT_DIR" | head -n 5 | while read -r f; do
  cp "$INPUT_DIR/$f" results/samples_before/
done
rm -rf results/output_1stream/*.png
tar -czf proof_of_execution.tar.gz results
echo "Wrote proof_of_execution.tar.gz"
