#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
INDEX_PATH="${INDEX_PATH:-artifacts/index.pkl}"
DATA_DIR="${DATA_DIR:-data}"
MODEL_PATH="${MODEL_PATH:-models/tinyllama}"
TOP_K="${TOP_K:-8}"
TEMPERATURE="${TEMPERATURE:-0.2}"
MAX_TOKENS="${MAX_TOKENS:-700}"

QUERY=""
SHOW_CONTEXT="false"
UI="false"

usage() {
  cat <<EOF
Usage:
  ./scripts/run.sh --query "..." [--show-context]
  ./scripts/run.sh --ui

Options:
  --query         Question to ask (CLI mode)
  --show-context  Print retrieved context chunks
  --ui            Launch Streamlit UI
  --model-path    Local HF model folder path (default: ${MODEL_PATH})
  --index-path    Index path (default: ${INDEX_PATH})
  --data-dir      Data directory with FHIR JSON (default: ${DATA_DIR})
  --top-k         Retrieval top-k (default: ${TOP_K})
  --temperature   LLM temperature (default: ${TEMPERATURE})
  --max-tokens    Max new tokens (default: ${MAX_TOKENS})
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --query) QUERY="$2"; shift 2 ;;
    --show-context) SHOW_CONTEXT="true"; shift 1 ;;
    --ui) UI="true"; shift 1 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --index-path) INDEX_PATH="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Virtual environment not found. Run: ./scripts/setup.sh"
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

if [[ ! -f "${INDEX_PATH}" ]]; then
  if [[ ! -d "${DATA_DIR}" ]]; then
    echo "Index not found and data dir missing: ${DATA_DIR}. Copy FHIR JSON files there and run setup."
    exit 1
  fi
  shopt -s nullglob
  json_files=("${DATA_DIR}"/*.json)
  shopt -u nullglob
  if [[ ${#json_files[@]} -eq 0 ]]; then
    echo "Index not found and no *.json files in ${DATA_DIR}. Copy FHIR JSON files there and rerun."
    exit 1
  fi
  mkdir -p "$(dirname "${INDEX_PATH}")"
  python cli.py build-index --data-dir "${DATA_DIR}" --index-path "${INDEX_PATH}"
fi

if [[ "${UI}" == "true" ]]; then
  streamlit run app.py
  exit 0
fi

if [[ -z "${QUERY}" ]]; then
  echo "Provide --query or use --ui."
  usage
  exit 1
fi

ARGS=(cli.py ask
  --index-path "${INDEX_PATH}"
  --query "${QUERY}"
  --top-k "${TOP_K}"
  --use-llm
  --llm-provider hf
  --llm-model-path "${MODEL_PATH}"
  --temperature "${TEMPERATURE}"
  --max-tokens "${MAX_TOKENS}"
)

if [[ "${SHOW_CONTEXT}" == "true" ]]; then
  ARGS+=(--show-context)
fi

python "${ARGS[@]}"
