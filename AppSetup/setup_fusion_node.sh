#!/usr/bin/env bash
# setup_fusion_node.sh — Bootstrap a fresh Linux/macOS node for the
# Fusion Centers FL Update.
#
# Provisions, in order:
#   1. System packages (curl, unzip, python3-venv) — Debian/Ubuntu only.
#      Other distros / macOS: skipped, you must have python3 + curl + unzip
#      installed already.
#   2. A local .venv/ at the project root.
#   3. requirements_fusion.txt into that venv.
#   4. The UCI Communities and Crime (Unnormalized) data file under
#      $HOME/datasets/CommunitiesCrime/.
#   5. The .names schema file via `python -m ...generate_names_file`.
#   6. (Optional, --verify) Runs `pytest tests/unit -q` to confirm the
#      install actually works.
#
# Usage:
#   bash AppSetup/setup_fusion_node.sh             # full setup
#   bash AppSetup/setup_fusion_node.sh --verify    # full setup + run tests
#   bash AppSetup/setup_fusion_node.sh --skip-data # don't touch ~/datasets
#   bash AppSetup/setup_fusion_node.sh --help      # show options
#
# Re-runs are safe: existing venv / data files are reused, not clobbered.
set -euo pipefail

# ── Resolve project root (parent of AppSetup/) ──────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements_fusion.txt"

# ── Defaults ────────────────────────────────────────────────────────────
VERIFY=0
SKIP_DATA=0
SKIP_SYS=0
VENV_DIR="$PROJECT_ROOT/.venv"
DATA_DIR="${DATA_DIR:-$HOME/datasets/CommunitiesCrime}"
UCI_ZIP_URL="https://archive.ics.uci.edu/static/public/211/communities+and+crime+unnormalized.zip"

usage() {
    sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//' | head -n -1
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --verify)    VERIFY=1; shift;;
        --skip-data) SKIP_DATA=1; shift;;
        --skip-sys)  SKIP_SYS=1; shift;;
        --venv)      VENV_DIR="$2"; shift 2;;
        -h|--help)   usage;;
        *) echo "Unknown option: $1" >&2; exit 2;;
    esac
done

log()  { printf '\n\033[1;36m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[warn]\033[0m  %s\n' "$*" >&2; }
fail() { printf '\n\033[1;31m[fail]\033[0m  %s\n' "$*" >&2; exit 1; }

# ── 1. System packages (best-effort) ────────────────────────────────────
if [[ "$SKIP_SYS" -eq 0 ]]; then
    if command -v apt-get >/dev/null 2>&1; then
        log "Installing system packages via apt..."
        sudo apt-get update -qq
        sudo apt-get install -y --no-install-recommends \
            python3 python3-venv python3-pip curl unzip ca-certificates
    elif command -v brew >/dev/null 2>&1; then
        log "macOS detected — assuming Homebrew Python is already installed."
        log "If not: brew install python@3.11 curl unzip"
    else
        warn "No apt-get or brew found — skipping system-package step."
        warn "Make sure python3, pip, curl, and unzip are on PATH."
    fi
else
    log "Skipping system-package step (--skip-sys)."
fi

# ── 2. Create venv ──────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    fail "python3 not found on PATH. Install Python 3.9+ first."
fi

PYTHON_VER="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
log "Using Python $PYTHON_VER from $(command -v "$PYTHON_BIN")"

if [[ ! -d "$VENV_DIR" ]]; then
    log "Creating venv at $VENV_DIR..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    log "Reusing existing venv at $VENV_DIR."
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
PIP_BIN="$VENV_DIR/bin/pip"

# ── 3. Python deps ──────────────────────────────────────────────────────
log "Upgrading pip + installing $REQUIREMENTS_FILE..."
"$PIP_BIN" install --upgrade pip wheel setuptools
"$PIP_BIN" install -r "$REQUIREMENTS_FILE"

# ── 4. Dataset acquisition ──────────────────────────────────────────────
if [[ "$SKIP_DATA" -eq 1 ]]; then
    log "Skipping dataset acquisition (--skip-data)."
else
    mkdir -p "$DATA_DIR"
    DATA_FILE="$DATA_DIR/CommViolPredUnnormalizedData.txt"

    if [[ -f "$DATA_FILE" ]]; then
        log "Data file already present at $DATA_FILE — skipping download."
    else
        log "Downloading UCI Communities and Crime (Unnormalized)..."
        TMP_ZIP="$(mktemp -t comm_crime.XXXXXX.zip)"
        TMP_EXTRACT="$(mktemp -d -t comm_crime.XXXXXX)"
        trap 'rm -rf "$TMP_ZIP" "$TMP_EXTRACT"' EXIT

        curl -L --fail --silent --show-error "$UCI_ZIP_URL" -o "$TMP_ZIP"
        unzip -q "$TMP_ZIP" -d "$TMP_EXTRACT"

        # The zip layout has varied; locate the data file wherever it landed.
        FOUND="$(find "$TMP_EXTRACT" -name 'CommViolPredUnnormalizedData.txt' -print -quit)"
        if [[ -z "$FOUND" ]]; then
            fail "Could not find CommViolPredUnnormalizedData.txt in the UCI zip."
        fi
        mv "$FOUND" "$DATA_FILE"
        log "Saved $DATA_FILE ($(wc -c < "$DATA_FILE") bytes)"
    fi

    NAMES_FILE="$DATA_DIR/communities_and_crime_unnormalized.names"
    if [[ -f "$NAMES_FILE" ]]; then
        log ".names schema already present — skipping generation."
    else
        log "Generating .names schema via ucimlrepo..."
        (
            cd "$PROJECT_ROOT"
            "$VENV_DIR/bin/python" -m \
                Config.DatasetConfig.CommunitiesCrime_Sampling.generate_names_file \
                --output "$NAMES_FILE"
        )
    fi

    # ── 5. Quick schema validation ──────────────────────────────────────
    log "Validating dataset schema..."
    (
        cd "$PROJECT_ROOT"
        "$VENV_DIR/bin/python" - <<PYEOF
from pathlib import Path
from Config.DatasetConfig.CommunitiesCrime_Sampling.commCrimeDatasetLoad import (
    parse_names_file, load_raw, CRIME_RATE_COLUMNS, SENSITIVE_COLUMNS,
)
d = Path(r"$DATA_DIR")
names = parse_names_file(d / "communities_and_crime_unnormalized.names")
df = load_raw(d / "CommViolPredUnnormalizedData.txt", names)
print(f"  rows={len(df)}  cols={df.shape[1]}")
missing_crime = [c for c in CRIME_RATE_COLUMNS if c not in df.columns]
missing_sens  = [c for c in SENSITIVE_COLUMNS  if c not in df.columns]
if missing_crime:
    print(f"  !! missing crime-rate cols: {missing_crime}")
else:
    print(f"  ok — all {len(CRIME_RATE_COLUMNS)} crime-rate columns present")
if missing_sens:
    print(f"  note: sensitive columns absent (will be silently skipped): {missing_sens}")
PYEOF
    )
fi

# ── 6. Optional: run test suite ─────────────────────────────────────────
if [[ "$VERIFY" -eq 1 ]]; then
    log "Running test suite (tests/unit) to verify install..."
    (
        cd "$PROJECT_ROOT"
        "$VENV_DIR/bin/pytest" tests/unit -q -k fusion
    )
fi

log "Done. Activate the venv with:"
echo "    source \"$VENV_DIR/bin/activate\""
log "Then follow DeveloperDocs/RUNNING_FUSION_EXPERIMENTS.md for the experiment matrix."
