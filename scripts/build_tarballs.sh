#!/usr/bin/env bash
# Run this on the CHTC Linux submit node to build venv.tar.gz and code.tar.gz
# Prerequisites: Python 3.10 or 3.11 available on the submit node
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=== Building venv.tar.gz ==="
# Clean any old venv
rm -rf venv venv.tar.gz

# Create fresh venv
python3 -m venv venv
source venv/bin/activate

# Install PyTorch for CUDA 11.8 (P100 compatible)
pip install --upgrade pip
pip install torch==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install \
  transformers>=4.38 \
  peft>=0.11 \
  datasets>=2.18 \
  accelerate>=0.27 \
  safetensors \
  pot \
  geomloss \
  scipy \
  scikit-learn \
  numpy

deactivate

# Strip unnecessary files to reduce tarball size
find venv -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find venv -name "*.pyc" -delete 2>/dev/null || true
find venv -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true

tar -czf venv.tar.gz venv
echo "venv.tar.gz: $(du -h venv.tar.gz | cut -f1)"

echo ""
echo "=== Building code.tar.gz ==="
# Pack project code (flat — extracts into current dir on execute node)
rm -f code.tar.gz
tar -czf code.tar.gz \
  run.py \
  sweep.py \
  config.py \
  data.py \
  model.py \
  stage_a.py \
  stage_b.py \
  evaluate.py \
  otdd_utils.py \
  third_party/

echo "code.tar.gz: $(du -h code.tar.gz | cut -f1)"

echo ""
echo "=== Done ==="
echo "Transfer these to your submit directory:"
echo "  venv.tar.gz"
echo "  code.tar.gz"
echo ""
echo "Then submit:"
echo "  mkdir -p htcondor_logs"
echo "  condor_submit htcondor/sweep.sub"
