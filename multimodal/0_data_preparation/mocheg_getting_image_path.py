#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract image paths for MOCHEG without running models.
Reuses load_split() logic from evidence_classification.py.
"""

import json
import os
from evidence_classification import load_split

def main():
    data_root = "mocheg"     # path to your dataset root
    split = "train"            # or 'train' / 'test'
    output_json = f"outputs/{split}_image_paths_only.json"

    # Load just the data, no model inference
    claims, text_evi, image_evi = load_split(data_root=data_root, split=split)

    results = []
    for cid, claim_info in claims.items():
        ctext = claim_info["claim_text"]
        img_paths = image_evi.get(cid, [])
        results.append({
            "Id": cid,
            "claim_text": ctext,
            "document_image_path": img_paths[0] if img_paths else ""
        })

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Wrote {len(results)} entries with image paths to {output_json}")

if __name__ == "__main__":
    main()


