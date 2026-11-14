#!/usr/bin/env bash
set -euo pipefail

# Turnkey cross-platform setup script for video-editing-ai (macOS & Linux)
# Installs system prerequisites (ffmpeg, libheif, compatible Python), creates .venv,
# installs all runtime/GPU/dev Python dependencies, and verifies via `ai-editor doctor`.

SKIP_SYSTEM_DEPS=0
INSTALL_DEV=1
EDITABLE=0
WITH_ANTIGRAVITY=0
SYNC_SKILL=0
VENV_DIR=".venv"

usage() {
  cat <<'EOF'
Usage: ./scripts/setup.sh [options]

Options:
  --skip-system-deps   Skip Homebrew / apt / dnf / pacman system package checks
  --no-dev             Install runtime dependencies only (skip pytest / dev extras)
  --editable           Install in editable mode (-e); avoid with dot-prefixed .venv on macOS Python 3.12+
  --with-antigravity   Also install optional [antigravity] SDK extra
  --sync-skill         Sync .agent/skills/ai-video-editor to ~/.gemini/config/skills
  --venv <path>        Virtual environment directory (default: .venv)
  -h, --help           Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-system-deps)
      SKIP_SYSTEM_DEPS=1
      shift
      ;;
    --no-dev)
      INSTALL_DEV=0
      shift
      ;;
    --editable)
      EDITABLE=1
      shift
      ;;
    --with-antigravity)
      WITH_ANTIGRAVITY=1
      shift
      ;;
    --sync-skill)
      SYNC_SKILL=1
      shift
      ;;
    --venv)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --venv requires a path argument" >&2
        exit 1
      fi
      VENV_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
echo "==> Setting up video-editing-ai in ${REPO_ROOT} (${OS_NAME} / ${ARCH_NAME})"

_python_is_compatible() {
  local py_bin="$1"
  command -v "${py_bin}" >/dev/null 2>&1 || [[ -x "${py_bin}" ]] || return 1
  "${py_bin}" - "$OS_NAME" "$ARCH_NAME" <<'PY' >/dev/null 2>&1
import sys
os_name, arch_name = sys.argv[1], sys.argv[2]
maj, min_ = sys.version_info[:2]
if (maj, min_) < (3, 10):
    sys.exit(1)
if os_name == "Darwin" and arch_name == "x86_64" and (maj, min_) >= (3, 13):
    sys.exit(1)
sys.exit(0)
PY
}

_find_compatible_python() {
  local candidates=(
    python3.12
    python3.11
    python3.10
    /opt/homebrew/opt/python@3.12/bin/python3.12
    /usr/local/opt/python@3.12/bin/python3.12
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12
    /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11
    python3
  )
  local cand
  for cand in "${candidates[@]}"; do
    if _python_is_compatible "${cand}"; then
      command -v "${cand}" 2>/dev/null || printf '%s\n' "${cand}"
      return 0
    fi
  done
  return 1
}

if [[ "${SKIP_SYSTEM_DEPS}" -eq 0 ]]; then
  if [[ "${OS_NAME}" == "Darwin" ]]; then
    if command -v brew >/dev/null 2>&1; then
      MISSING_BREW=()
      if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
        MISSING_BREW+=(ffmpeg)
      fi
      if ! brew list --formula libheif >/dev/null 2>&1; then
        MISSING_BREW+=(libheif)
      fi
      if [[ "${ARCH_NAME}" == "x86_64" ]] && ! _find_compatible_python >/dev/null 2>&1; then
        MISSING_BREW+=(python@3.12)
      fi
      if [[ ${#MISSING_BREW[@]} -gt 0 ]]; then
        echo "==> Installing macOS prerequisites via Homebrew: ${MISSING_BREW[*]}"
        brew install "${MISSING_BREW[@]}"
      fi
    else
      if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
        echo "[ERROR] ffmpeg/ffprobe not found and Homebrew is not installed." >&2
        echo "        Install Homebrew (https://brew.sh) and run: brew install ffmpeg libheif python@3.12" >&2
        exit 1
      fi
      echo "[WARN] Homebrew not found; assuming system libheif and compatible Python are installed."
    fi
  elif [[ "${OS_NAME}" == "Linux" ]]; then
    if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
      SUDO_CMD=""
      if [[ "$(id -u)" -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
        SUDO_CMD="sudo"
      fi
      if command -v apt-get >/dev/null 2>&1; then
        echo "==> Installing Linux prerequisites via apt-get (ffmpeg, libheif-dev, python3-venv)"
        ${SUDO_CMD} apt-get update
        ${SUDO_CMD} apt-get install -y ffmpeg libheif-dev python3-venv
      elif command -v dnf >/dev/null 2>&1; then
        echo "==> Installing Linux prerequisites via dnf (ffmpeg, libheif)"
        ${SUDO_CMD} dnf install -y ffmpeg libheif
      elif command -v pacman >/dev/null 2>&1; then
        echo "==> Installing Linux prerequisites via pacman (ffmpeg, libheif)"
        ${SUDO_CMD} pacman -Sy --noconfirm ffmpeg libheif
      else
        echo "[ERROR] ffmpeg/ffprobe missing and no supported package manager (apt-get/dnf/pacman) found." >&2
        exit 1
      fi
    fi
  fi
fi

if [[ -x "${VENV_DIR}/bin/python" ]] && _python_is_compatible "${VENV_DIR}/bin/python"; then
  echo "==> Reusing existing compatible virtual environment at ${VENV_DIR}"
else
  PYTHON_BIN="$(_find_compatible_python || true)"
  if [[ -z "${PYTHON_BIN}" ]]; then
    if [[ "${OS_NAME}" == "Darwin" && "${ARCH_NAME}" == "x86_64" ]]; then
      echo "[ERROR] Intel macOS requires Python 3.10–3.12 (PyTorch 2.2.2 MPS compatibility)." >&2
      echo "        Run: brew install python@3.12" >&2
    else
      echo "[ERROR] Python >=3.10 not found. Install Python 3.12 and retry." >&2
    fi
    exit 1
  fi
  echo "==> Creating virtual environment at ${VENV_DIR} using ${PYTHON_BIN}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PY="${VENV_DIR}/bin/python"
echo "==> Upgrading pip, setuptools, and wheel"
"${VENV_PY}" -m pip install --upgrade pip setuptools wheel

EXTRAS=""
if [[ "${INSTALL_DEV}" -eq 1 ]]; then
  EXTRAS="dev"
fi
if [[ "${WITH_ANTIGRAVITY}" -eq 1 ]]; then
  if [[ -n "${EXTRAS}" ]]; then
    EXTRAS="${EXTRAS},antigravity"
  else
    EXTRAS="antigravity"
  fi
fi

if [[ -n "${EXTRAS}" ]]; then
  INSTALL_TARGET=".[${EXTRAS}]"
else
  INSTALL_TARGET="."
fi

echo "==> Installing video-editing-ai (${INSTALL_TARGET})"
if [[ "${EDITABLE}" -eq 1 ]]; then
  "${VENV_PY}" -m pip install -e "${INSTALL_TARGET}"
  if [[ "${OS_NAME}" == "Darwin" ]]; then
    chflags nohidden "${VENV_DIR}"/lib/python*/site-packages/*.pth 2>/dev/null || true
  fi
else
  "${VENV_PY}" -m pip install "${INSTALL_TARGET}"
fi

if [[ "${SYNC_SKILL}" -eq 1 ]]; then
  echo "==> Synchronizing ai-video-editor skill globally"
  "${VENV_PY}" -m video_editing_ai.cli sync-skill
fi

echo "==> Verifying environment with ai-editor doctor"
"${VENV_PY}" -m video_editing_ai.cli doctor
echo "==> Setup complete! Activate with: source ${VENV_DIR}/bin/activate"
