#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Environment setup for "Entailed Opinion Matters" (AACL).
#
# Two environments are required, because they have conflicting pins:
#
#   entail-vllm         vllm 0.11.0 + transformers 4.57.0
#                       -> evidence classification, justification generation,
#                          ELM training, adapter inference
#
#   entail-llamafactory transformers <= 4.49 (LLaMA-Factory's own pin)
#                       -> LoRA / LoRA+ / QLoRA fine-tuning of the GLMs
#
# Usage:
#   bash env/setup.sh vllm          # create + populate entail-vllm
#   bash env/setup.sh llamafactory  # create + populate entail-llamafactory
#   bash env/setup.sh both
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
TARGET="${1:-both}"

have_conda() { command -v conda >/dev/null 2>&1; }

# vLLM JIT-compiles Triton kernels at import time and needs Python.h to do it.
check_python_headers() {
  local inc
  inc="$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["include"])' 2>/dev/null)"
  if [ -n "$inc" ] && [ ! -f "$inc/Python.h" ]; then
    echo
    echo "WARNING: $inc/Python.h is missing."
    echo "  vLLM will fail at startup with:"
    echo "    fatal error: Python.h: No such file or directory"
    echo "    torch._dynamo.exc.BackendCompilerFailed: backend='VllmBackend' raised ..."
    echo "  Fix with:  sudo apt-get install -y python3-dev build-essential"
    echo "  Or work around it with:  export VLLM_USE_V1=0   (slower, no torch.compile)"
    echo
  fi
}

make_env() {   # $1 = env name, $2 = python version
  if have_conda; then
    conda env list | grep -qE "^$1[[:space:]]" || conda create -y -n "$1" "python=$2"
    echo "conda run -n $1"
  else
    [ -d "$REPO/.venv-$1" ] || python3 -m venv "$REPO/.venv-$1"
    echo "$REPO/.venv-$1/bin/python -m"
  fi
}

setup_vllm() {
  echo "=== entail-vllm ==="
  check_python_headers
  if have_conda; then
    conda env list | grep -qE "^entail-vllm[[:space:]]" || conda create -y -n entail-vllm python=3.11
    conda run -n entail-vllm pip install -r "$HERE/requirements-vllm.txt"
    conda run -n entail-vllm python -m spacy download xx_sent_ud_sm || \
      echo "WARN: spaCy xx_sent_ud_sm not installed (multilingual chunking will fall back to blingfire/nltk)"
  else
    python3 -m venv "$REPO/.venv-vllm"
    "$REPO/.venv-vllm/bin/pip" install --upgrade pip
    "$REPO/.venv-vllm/bin/pip" install -r "$HERE/requirements-vllm.txt"
    "$REPO/.venv-vllm/bin/python" -m spacy download xx_sent_ud_sm || true
  fi
}

setup_llamafactory() {
  echo "=== entail-llamafactory ==="
  LF_DIR="${LLAMAFACTORY_DIR:-$REPO/third_party/LLaMA-Factory}"
  if [ ! -d "$LF_DIR" ]; then
    mkdir -p "$(dirname "$LF_DIR")"
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git "$LF_DIR"
  fi
  if have_conda; then
    conda env list | grep -qE "^entail-llamafactory[[:space:]]" || conda create -y -n entail-llamafactory python=3.11
    conda run -n entail-llamafactory pip install -e "$LF_DIR[torch,bitsandbytes]"
    conda run -n entail-llamafactory pip install -r "$HERE/requirements-llamafactory.txt"
  else
    python3 -m venv "$REPO/.venv-llamafactory"
    "$REPO/.venv-llamafactory/bin/pip" install --upgrade pip
    "$REPO/.venv-llamafactory/bin/pip" install -e "$LF_DIR[torch,bitsandbytes]"
    "$REPO/.venv-llamafactory/bin/pip" install -r "$HERE/requirements-llamafactory.txt"
  fi
  echo
  echo "LLaMA-Factory checked out at: $LF_DIR"
  echo "Now register this project's datasets and copy the configs:"
  echo "  bash env/link_llamafactory.sh \"$LF_DIR\""
}

case "$TARGET" in
  vllm)          setup_vllm ;;
  llamafactory)  setup_llamafactory ;;
  both)          setup_vllm; setup_llamafactory ;;
  *) echo "usage: bash env/setup.sh [vllm|llamafactory|both]"; exit 1 ;;
esac

echo
echo "Done."
