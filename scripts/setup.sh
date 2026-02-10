#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
MODEL_ID="${MODEL_ID:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
MODEL_OUT_DIR="${MODEL_OUT_DIR:-models/tinyllama}"
DATA_DIR="${DATA_DIR:-data}"
INDEX_PATH="${INDEX_PATH:-artifacts/index.pkl}"

echo "=== Rhythmx Setup (macOS/Linux) - HF Transformers Local LLM ==="
echo "Project root: $(pwd)"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment at ${VENV_DIR} ..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "Virtual environment already exists: ${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
pip install -r requirements.txt

mkdir -p "${MODEL_OUT_DIR}"
if [[ ! -f "${MODEL_OUT_DIR}/config.json" ]]; then
  echo "Downloading model ${MODEL_ID} to ${MODEL_OUT_DIR} ..."
  python scripts/download_hf_model.py --model-id "${MODEL_ID}" --out-dir "${MODEL_OUT_DIR}"
else
  echo "Model folder already looks populated: ${MODEL_OUT_DIR}"
fi

mkdir -p "$(dirname "${INDEX_PATH}")"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "WARNING: Data directory not found: ${DATA_DIR}"
  echo "Create it and copy your FHIR JSON files (*.json) there."
else
  shopt -s nullglob
  json_files=("${DATA_DIR}"/*.json)
  shopt -u nullglob
  if [[ ${#json_files[@]} -eq 0 ]]; then
    echo "WARNING: No *.json files found in ${DATA_DIR}. Copy FHIR JSON files there, then re-run setup."
  else
    echo "Building retrieval index from ${DATA_DIR} ..."
    python cli.py build-index --data-dir "${DATA_DIR}" --index-path "${INDEX_PATH}"
  fi
fi

echo ""
echo "Setup complete."
echo "Next:"
echo "  - Ask a question:"
echo "      ./scripts/run.sh --query "Does the patient have asthma?" --show-context"
echo "  - Or start UI:"
echo "      ./scripts/run.sh --ui"
