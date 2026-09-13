#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Wire this project's configs and datasets into a LLaMA-Factory checkout.
#
#   bash env/link_llamafactory.sh [/path/to/LLaMA-Factory]
#
# Defaults to third_party/LLaMA-Factory (where env/setup.sh clones it).
#
# What it does:
#   1. merges env/dataset_info.project.json into <LF>/data/dataset_info.json
#   2. symlinks this repo's training YAMLs under <LF>/examples/entailment/
#   3. reports which registered data files are not yet present
#
# After this you can run, from the LLaMA-Factory directory:
#   llamafactory-cli train examples/entailment/multimodal/factify2/qwen_data/<config>.yaml
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
LF="${1:-$REPO/third_party/LLaMA-Factory}"

[ -d "$LF" ] || { echo "ERROR: no LLaMA-Factory at $LF"; echo "Run: bash env/setup.sh llamafactory"; exit 1; }
[ -d "$LF/data" ] || { echo "ERROR: $LF does not look like a LLaMA-Factory checkout (no data/)"; exit 1; }

echo "LLaMA-Factory : $LF"

# --- 1. merge dataset_info ---------------------------------------------------
python3 - "$REPO" "$LF" <<'PY'
import json, os, sys, shutil
repo, lf = sys.argv[1], sys.argv[2]
target = os.path.join(lf, "data", "dataset_info.json")
project = json.load(open(os.path.join(repo, "env", "dataset_info.project.json")))
existing = json.load(open(target)) if os.path.exists(target) else {}

if os.path.exists(target) and not os.path.exists(target + ".orig"):
    shutil.copy2(target, target + ".orig")

added = [k for k in project if k not in existing]
clashed = [k for k in project if k in existing and existing[k] != project[k]]
existing.update(project)
json.dump(existing, open(target, "w"), indent=2, ensure_ascii=False)

print(f"dataset_info  : +{len(added)} new, {len(clashed)} overwritten, {len(existing)} total")
if clashed:
    print("                overwritten:", ", ".join(clashed[:5]) + ("..." if len(clashed) > 5 else ""))
PY

# --- 2. link the YAML configs ------------------------------------------------
mkdir -p "$LF/examples/entailment"
for src in "$REPO/multimodal/4_qlora_glm/configs" \
           "$REPO/multilingual/4_qlora_glm/configs"; do
  name="$(basename "$(dirname "$(dirname "$src")")")"   # multimodal | multilingual
  ln -sfn "$src" "$LF/examples/entailment/$name"
  echo "configs       : examples/entailment/$name -> ${src#$REPO/}"
done
if [ -d "$REPO/monolingual/TBE3/LoRA_and_LoRA+_finetuning/examples/train_lora" ]; then
  ln -sfn "$REPO/monolingual/TBE3/LoRA_and_LoRA+_finetuning/examples/train_lora" \
          "$LF/examples/entailment/monolingual"
  echo "configs       : examples/entailment/monolingual -> monolingual/TBE3/.../examples/train_lora"
fi

# --- 3. which registered data files are missing? -----------------------------
python3 - "$REPO" "$LF" <<'PY'
import json, os, sys
repo, lf = sys.argv[1], sys.argv[2]
info = json.load(open(os.path.join(repo, "env", "dataset_info.project.json")))
search = [
    os.path.join(lf, "data"),
    os.path.join(repo, "monolingual", "TBE3", "LoRA_and_LoRA+_finetuning", "data"),
]
missing = []
for name, spec in info.items():
    fn = spec.get("file_name")
    if not fn:
        continue
    if not any(os.path.exists(os.path.join(d, fn)) for d in search):
        missing.append((name, fn))
print(f"data files    : {len(info) - len(missing)}/{len(info)} present under {os.path.join(lf,'data')}")
if missing:
    print()
    print("  The following are registered but not on disk. Generate them with the")
    print("  prepare_data/to_llamafactory.py script of the matching section, then")
    print(f"  copy the output into {os.path.join(lf, 'data')}/ :")
    for name, fn in missing:
        print(f"    {name:60s} -> {fn}")
PY

echo
echo "Done. Train with, e.g.:"
echo "  cd $LF && llamafactory-cli train examples/entailment/multilingual/xfact/mistral_lora_sft.yaml"
