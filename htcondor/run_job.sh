#!/usr/bin/env bash
# HTCondor executable: unpack venv + code, run one GPU job.
# Expects transfer_input_files: venv.tar.gz, code.tar.gz
# Environment variables set by submit file: METHOD, LORA_RANK, SEED, EMBED_EP, OUTPUT_TARBALL
set -euo pipefail

WORK="${_CONDOR_SCRATCH_DIR:-$(pwd)}"
cd "${WORK}"

# --- Resource logging ---
log_condor_resources() {
  local label="$1"
  local log_file="$2"
  mkdir -p "$(dirname "$log_file")"
  {
    echo "=== condor_resource_usage: ${label} ==="
    echo "date: $(date -Is 2>/dev/null || date)"
    echo "WORK=${WORK}"
    du -sh . 2>/dev/null || true
    du -sh venv hf_cache runs 2>/dev/null | sort -h || true
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader
    fi
    echo "--- end ${label} ---"
    echo ""
  } | tee -a "$log_file"
}

# --- Unpack ---
if [[ ! -f venv.tar.gz || ! -f code.tar.gz ]]; then
  echo "Missing transfer_input_files: venv.tar.gz and/or code.tar.gz" >&2
  exit 9
fi

tar -xzf venv.tar.gz
tar -xzf code.tar.gz

export PATH="${PWD}/venv/bin:${PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${PWD}/hf_cache}"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
mkdir -p "${HF_HOME}"

# --- GPU probe ---
python -c "
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
    print('capability', torch.cuda.get_device_capability(0))
    print('bf16_supported', torch.cuda.is_bf16_supported())
"

# --- Job parameters (from submit file environment) ---
METHOD="${METHOD:-lora_only}"
LORA_RANK="${LORA_RANK:-8}"
SEED="${SEED:-0}"
EMBED_EP="${EMBED_EP:-45}"
LORA_EPOCHS="${LORA_EPOCHS:-10}"
TEST_SAMPLES="${TEST_SAMPLES:-0}"
HARD_TRAIN_SAMPLES="${HARD_TRAIN_SAMPLES:-200}"
EASY_POOL_SAMPLES="${EASY_POOL_SAMPLES:-100}"
VAL_SAMPLES="${VAL_SAMPLES:-50}"

JOB_TAG="${JOB_TAG:-${METHOD}_r${LORA_RANK}_s${SEED}}"
# Store in results/lora/ or results/orca/
if [[ "${METHOD}" == "lora_only" ]]; then
  OUT="results/lora/${JOB_TAG}"
else
  OUT="results/orca/${JOB_TAG}"
fi
OUTPUT_TARBALL="${OUTPUT_TARBALL:-orca_${METHOD}_r${LORA_RANK}_s${SEED}.tar.gz}"
mkdir -p "${OUT}"

# --- Run experiment ---
set +e
python run.py \
  --method "${METHOD}" \
  --lora_r "${LORA_RANK}" \
  --lora_alpha "$(( LORA_RANK * 2 ))" \
  --seed "${SEED}" \
  --embedder_epochs "${EMBED_EP}" \
  --lora_epochs "${LORA_EPOCHS}" \
  --test_samples "${TEST_SAMPLES}" \
  --hard_train_samples "${HARD_TRAIN_SAMPLES}" \
  --easy_pool_samples "${EASY_POOL_SAMPLES}" \
  --val_samples "${VAL_SAMPLES}" \
  --output_dir "${OUT}" \
  --no-verbose \
  --device cuda
_rc=${?}
set -euo pipefail

echo "run.py exit_code=${_rc}" >> "${OUT}/condor_job_status.txt"
log_condor_resources "pre_pack" "${OUT}/condor_resource_usage.txt"

# --- Pack results ---
tar -czf "${OUTPUT_TARBALL}" "${OUT}"
ls -lh "${OUTPUT_TARBALL}"

exit "${_rc}"
