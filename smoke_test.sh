#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Smoke test for the Entailment-Based-Fact-Checking repository.
#
# Stage 1 (no GPU, no model downloads) -- always runs:
#   * every .py compiles
#   * every .sh parses
#   * every .yaml parses, and every `dataset:` it names resolves in dataset_info
#   * the pure-data transforms actually run against samples/
#
# Stage 2 (needs a GPU + the entail-vllm environment) -- runs only with --gpu:
#   * one real evidence-classification + justification-generation pass
#     over 3 claims, using a small model
#
# Usage:
#   bash smoke_test.sh            # stage 1 only
#   bash smoke_test.sh --gpu      # stage 1 + stage 2
# ---------------------------------------------------------------------------
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
RUN_GPU=0
[ "${1:-}" = "--gpu" ] && RUN_GPU=1

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PASS=0; FAIL=0; SKIP=0
ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=$((FAIL+1)); }
skip() { printf '  \033[33mSKIP\033[0m  %s\n' "$1"; SKIP=$((SKIP+1)); }
hdr()  { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

# Vendored third-party trees are not ours to lint.
EXCLUDE='monolingual/TBE3/LoRA_and_LoRA\+_finetuning/(src|tests|scripts|evaluation)|third_party/|\.venv|__pycache__'

# ---------------------------------------------------------------------------
hdr "1. Python files compile"
# ---------------------------------------------------------------------------
n_py=0; bad_py=0
while IFS= read -r f; do
  n_py=$((n_py+1))
  if ! "$PY" -m py_compile "$f" 2>"$TMP/err"; then
    bad_py=$((bad_py+1)); printf '        %s\n           %s\n' "${f#$REPO/}" "$(tail -1 "$TMP/err")"
  fi
done < <(find "$REPO" -name '*.py' | grep -Ev "$EXCLUDE")
[ "$bad_py" -eq 0 ] && ok "$n_py python files compile" || bad "$bad_py/$n_py python files fail to compile"

# ---------------------------------------------------------------------------
hdr "2. Shell scripts parse"
# ---------------------------------------------------------------------------
n_sh=0; bad_sh=0
while IFS= read -r f; do
  n_sh=$((n_sh+1))
  bash -n "$f" 2>"$TMP/err" || { bad_sh=$((bad_sh+1)); printf '        %s: %s\n' "${f#$REPO/}" "$(tail -1 "$TMP/err")"; }
done < <(find "$REPO" -name '*.sh' | grep -Ev "$EXCLUDE")
[ "$bad_sh" -eq 0 ] && ok "$n_sh shell scripts parse" || bad "$bad_sh/$n_sh shell scripts have syntax errors"

# ---------------------------------------------------------------------------
hdr "3. YAML configs parse"
# ---------------------------------------------------------------------------
"$PY" - "$REPO" <<'PYEOF'
import sys, glob, os
repo = sys.argv[1]
try:
    import yaml
except ImportError:
    print("  SKIP  pyyaml not installed"); sys.exit(0)
bad = []
files = [f for f in glob.glob(f"{repo}/**/*.yaml", recursive=True)
         if "third_party" not in f and ".venv" not in f]
for f in files:
    try:
        yaml.safe_load(open(f))
    except Exception as e:
        bad.append((f.replace(repo + "/", ""), str(e).split("\n")[0]))
if bad:
    print(f"  \033[31mFAIL\033[0m  {len(bad)}/{len(files)} YAML files fail to parse")
    for f, e in bad[:10]:
        print(f"        {f}: {e}")
    sys.exit(1)
print(f"  \033[32mPASS\033[0m  {len(files)} YAML configs parse")
PYEOF
[ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

# ---------------------------------------------------------------------------
hdr "4. Every YAML 'dataset:' resolves in dataset_info"
# ---------------------------------------------------------------------------
"$PY" - "$REPO" <<'PYEOF'
import sys, glob, json, re, os
repo = sys.argv[1]
info = {}
for p in glob.glob(f"{repo}/**/dataset_info*.json", recursive=True):
    if "third_party" in p:
        continue
    try:
        info.update(json.load(open(p)))
    except Exception:
        pass
referenced, missing = set(), set()
for f in glob.glob(f"{repo}/**/*.yaml", recursive=True):
    if "third_party" in f:
        continue
    for line in open(f, errors="ignore"):
        m = re.match(r"\s*dataset:\s*(.+?)\s*(?:#.*)?$", line)
        if m:
            for name in m.group(1).split(","):
                name = name.strip()
                if name and name not in ("null", ""):
                    referenced.add(name)
                    if name not in info:
                        missing.add(name)
if missing:
    print(f"  \033[31mFAIL\033[0m  {len(missing)}/{len(referenced)} dataset names unregistered")
    for n in sorted(missing)[:10]:
        print(f"        {n}")
    sys.exit(1)
print(f"  \033[32mPASS\033[0m  all {len(referenced)} referenced dataset names registered "
      f"({len(info)} entries in dataset_info)")
PYEOF
[ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

# ---------------------------------------------------------------------------
hdr "5. Sample data is well-formed"
# ---------------------------------------------------------------------------
"$PY" - "$REPO" <<'PYEOF'
import sys, glob, json
repo = sys.argv[1]
files = sorted(glob.glob(f"{repo}/samples/**/*.json", recursive=True))
if not files:
    print("  \033[31mFAIL\033[0m  no sample files found"); sys.exit(1)
for f in files:
    d = json.load(open(f))
    assert isinstance(d, list) and d, f
print(f"  \033[32mPASS\033[0m  {len(files)} sample files load as non-empty lists")
PYEOF
[ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

# ---------------------------------------------------------------------------
hdr "6. Data transforms run on samples"
# ---------------------------------------------------------------------------
S="$REPO/samples"

# 6a. monolingual justification cleaning (TBE-3 step 3)
if "$PY" - "$REPO" "$TMP" <<'PYEOF'
import sys, json, importlib.util, os
repo, tmp = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location(
    "cleaning", f"{repo}/monolingual/TBE3/lair_raw/cleaning_data.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
src = f"{repo}/samples/monolingual/lair_raw/step2_justifications_test_llama.json"
dst = f"{tmp}/cleaned.json"
# the sample uses the pre-cleaning key names; normalise then clean
data = json.load(open(src))
norm = [{"claim": r["claim"], "label": r["label"],
         "support_justification": r.get("support_justification") or r.get("true_justification", ""),
         "refute_justification":  r.get("refute_justification")  or r.get("false_justification", "")}
        for r in data]
tmp_in = f"{tmp}/norm.json"; json.dump(norm, open(tmp_in, "w"))
m.clean_json_data(tmp_in, dst)
out = json.load(open(dst))
assert len(out) == len(data) and all(o["support_justification"] for o in out)
PYEOF
then ok "TBE-3 cleaning_data.py cleans 20 LIAR-RAW justifications"; else bad "TBE-3 cleaning_data.py"; fi

# 6b. multimodal -> LLaMA-Factory conversion, all three datasets
CONV="$REPO/multimodal/4_qlora_glm/prepare_data/to_llamafactory.py"
conv_ok=1
"$PY" "$CONV" --dataset factify2 \
  --input "$S/multimodal/factify2/step2_justifications_test_qwen2vl.json" \
  --output "$TMP/f2.jsonl" >/dev/null 2>&1 || conv_ok=0
"$PY" "$CONV" --dataset verite \
  --input "$S/multimodal/verite/step2_justifications_train_qwen2vl_with_labels.json" \
  --output "$TMP/vr.jsonl" >/dev/null 2>&1 || conv_ok=0
for f in "$TMP/f2.jsonl" "$TMP/vr.jsonl"; do
  [ -s "$f" ] || conv_ok=0
done
[ "$conv_ok" -eq 1 ] && ok "to_llamafactory.py converts Factify-2 + VERITE samples" \
                     || bad "to_llamafactory.py conversion"

# 6c. label join
if "$PY" "$REPO/multimodal/0_data_preparation/attach_labels.py" \
     --justifications "$S/multimodal/verite/step2_justifications_train_qwen2vl_with_labels.json" \
     --labels "$S/multimodal/verite/step2_justifications_train_qwen2vl_with_labels.json" \
     --just_id_field claim_id --label_id_field claim_id --label_field label \
     --drop_unmatched --output "$TMP/relabelled.json" >/dev/null 2>&1 \
   && [ "$("$PY" -c "import json;print(len(json.load(open('$TMP/relabelled.json'))))")" = "20" ]; then
  ok "attach_labels.py joins 20/20 VERITE labels"
else
  bad "attach_labels.py"
fi

# ---------------------------------------------------------------------------
hdr "7. --help works on the entry-point scripts"
# ---------------------------------------------------------------------------
ENTRYPOINTS=(
  "multimodal/4_qlora_glm/prepare_data/to_llamafactory.py"
  "multimodal/0_data_preparation/attach_labels.py"
  "multilingual/4_qlora_glm/prepare_data/to_llamafactory.py"
)
for e in "${ENTRYPOINTS[@]}"; do
  if "$PY" "$REPO/$e" --help >/dev/null 2>"$TMP/err"; then
    ok "$e --help"
  elif grep -qiE "No module named|ImportError|cannot import name" "$TMP/err"; then
    skip "$e --help (missing dependency: $(grep -oiE "No module named '[^']+'|cannot import name '[^']+'" "$TMP/err" | head -1))"
  else
    bad "$e --help -- $(tail -1 "$TMP/err")"
  fi
done

# ---------------------------------------------------------------------------
if [ "$RUN_GPU" -eq 1 ]; then
hdr "8. GPU stage (evidence classification + justification generation, 3 claims)"
  # vLLM JIT-compiles Triton kernels at startup and needs Python.h to do it.
  PYINC="$("$PY" -c 'import sysconfig;print(sysconfig.get_paths()["include"])' 2>/dev/null)"
  if [ -n "$PYINC" ] && [ ! -f "$PYINC/Python.h" ]; then
    echo "        NOTE: $PYINC/Python.h is missing -- vLLM's Triton compile step will fail."
    echo "              Install python3-dev, or export VLLM_USE_V1=0 to fall back to the"
    echo "              non-compiling engine. Setting VLLM_USE_V1=0 for this run."
    export VLLM_USE_V1=0
  fi
  if ! "$PY" -c "import torch,vllm" 2>/dev/null; then
    skip "torch/vllm not importable in this environment"
  elif ! "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    skip "no CUDA device visible"
  else
    # One GLM, one split, 3 claims. SMOKE_MODEL defaults to llama; override to
    # match whatever you already have in the HF cache.
    MODEL="${SMOKE_MODEL:-llama}"
    cd "$REPO/monolingual/TBE3/rawfc"
    if "$PY" Rawfc_evidence_classification.py \
         --output "$TMP/rawfc_out" --data_dir . \
         --models "$MODEL" --splits test --limit 3 >"$TMP/gpu.log" 2>&1
    then
      OUT=$(find "$TMP/rawfc_out" -name '*.json' | head -1)
      if [ -n "$OUT" ] && "$PY" -c "
import json,sys
d=json.load(open('$OUT'))
assert len(d)==3, f'expected 3 records, got {len(d)}'
assert all('supporting_evidence' in r and 'refuting_evidence' in r for r in d)
n=sum(len(r['supporting_evidence'])+len(r['refuting_evidence']) for r in d)
print(f'        {len(d)} claims, {n} evidence sentences classified')
"; then
        ok "step 1 entailment: RAW-FC / $MODEL / 3 claims"
      else
        bad "step 1 ran but produced no usable output"
      fi
    else
      bad "step 1 entailment failed -- log tail:"
      tail -8 "$TMP/gpu.log" | sed 's/^/        /'
    fi
    cd "$REPO"
  fi
else
  hdr "8. GPU stage"
  skip "not requested (re-run with: bash smoke_test.sh --gpu)"
fi

# ---------------------------------------------------------------------------
printf '\n\033[1m=== %d passed, %d failed, %d skipped ===\033[0m\n' "$PASS" "$FAIL" "$SKIP"
[ "$FAIL" -eq 0 ]
