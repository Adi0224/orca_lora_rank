#!/usr/bin/env bash
# Pack code.tar.gz from repo root — flat layout so run.py is at top level after extract.
# Run from project root.
set -euo pipefail

STAMP=$(date +%Y%m%d_%H%M%S)

tar -czf "pack_code_${STAMP}.tar.gz" \
  --exclude='.git' \
  --exclude='venv' \
  --exclude='.venv' \
  --exclude='runs' \
  --exclude='__pycache__' \
  --exclude='*.tar.gz' \
  --exclude='htcondor_logs' \
  --exclude='.DS_Store' \
  --exclude='hf_cache' \
  run.py \
  sweep.py \
  config.py \
  data.py \
  model.py \
  stage_a.py \
  stage_b.py \
  evaluate.py \
  otdd_utils.py \
  requirements.txt \
  scripts/ \
  htcondor/ \
  third_party/

# Stable name for condor
ln -sf "pack_code_${STAMP}.tar.gz" code.tar.gz

echo "Created: pack_code_${STAMP}.tar.gz -> code.tar.gz"
ls -lh code.tar.gz
