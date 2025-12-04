#!/usr/bin/env bash
# Convenience wrapper to run tools/parse_inference_log.py
# Usage: tools/parse_inference_log.sh <inference.log> [output.csv]

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SELF_DIR/parse_inference_log.py"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <inference.log> [output.csv]"
  exit 1
fi

LOG_FILE="$1"
CSV_OUT="${2:-}"

if [[ ! -f "$LOG_FILE" ]]; then
  echo "Error: log file '$LOG_FILE' not found"
  exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is required but not found in PATH"
  exit 3
fi

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "Error: parser script not found at $PY_SCRIPT"
  exit 4
fi

if [[ -n "$CSV_OUT" ]]; then
  python3 "$PY_SCRIPT" "$LOG_FILE" "$CSV_OUT"
else
  python3 "$PY_SCRIPT" "$LOG_FILE"
fi
