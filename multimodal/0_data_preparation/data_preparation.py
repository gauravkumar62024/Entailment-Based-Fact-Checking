import os
import json
import imghdr
from langdetect import detect
from typing import Dict, List, Any, Tuple

# ============================================================
# 0. CONFIG
# ============================================================

ROOT = "mr2"   # folder that has dataset_items_*.json + train/val/test
OUTPUT_DIR = os.path.join(ROOT, "prepared_json")  # where we will save the final files

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. BASIC HELPERS
# ============================================================

def is_english(text: str) -> bool:
    """Return True if text is detected as English."""
    try:
        return detect(text) == "en"
    except Exception:
        return False


def process_string(input_str: str) -> str:
    """Clean some HTML-like artifacts (same spirit as original code)."""
    input_str = input_str.replace('&#39;', ' ')
    input_str = input_str.replace('<b>', '')
    input_str = input_str.replace('</b>', '')
    return input_str.strip()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_join(*args) -> str:
    """Join paths and normalize."""
    return os.path.normpath(os.path.join(*args))


# ============================================================
# 2. EVIDENCE EXTRACTION HELPERS
# ============================================================

def collect_evidence_captions_from_inverse(inv_dict: Dict[str, Any]) -> List[str]:
    """
    Reimplementation of load_captions() from original code.
    Extract textual evidence from inverse_annotation.json.
    """
    captions = ['']
    pages_with_captions_keys = ['all_fully_matched_captions', 'all_partially_matched_captions']

    for key1 in pages_with_captions_keys:
        if key1 in inv_dict:
            for page in inv_dict[key1]:
                # title
                if 'title' in page:
                    item = process_string(page['title'])
                    captions.append(item)

                # caption dict
                if 'caption' in page:
                    sub_captions_list = []
                    unfiltered_captions = []
                    for key2 in page['caption']:
                        sub_caption = page['caption'][key2]
                        sub_caption_filter = process_string(sub_caption)
                        if sub_caption in unfiltered_captions:
                            continue
                        sub_captions_list.append(sub_caption_filter)
                        unfiltered_captions.append(sub_caption)
                    captions.extend(sub_captions_list)

    # pages with title only
    pages_with_title_only_keys = ['partially_matched_no_text', 'fully_matched_no_text']
    for key1 in pages_with_title_only_keys:
        if key1 in inv_dict:
            for page in inv_dict[key1]:
                if 'title' in page:
                    title = process_string(page['title'])
                    captions.append(title)

    return captions


def collect_evidence_captions_from_direct(direct_dict: Dict[str, Any]) -> List[str]:
    """
    Reimplementation of load_captions_weibo() from original code.
    Extract textual evidence from direct_annotation.json.
    """
    captions = ['']
    keys = ['images_with_captions', 'images_with_no_captions', 'images_with_caption_matched_tags']

    for key1 in keys:
        if key1 in direct_dict:
            for page in direct_dict[key1]:
                if 'page_title' in page:
                    item = process_string(page['page_title'])
                    captions.append(item)

                if 'caption' in page:
                    sub_captions_list = []
                    unfiltered_captions = []
                    for key2 in page['caption']:
                        sub_caption = page['caption'][key2]
                        sub_caption_filter = process_string(sub_caption)
                        if sub_caption in unfiltered_captions:
                            continue
                        sub_captions_list.append(sub_caption_filter)
                        unfiltered_captions.append(sub_caption)
                    captions.extend(sub_captions_list)

    return captions


def collect_evidence_image_paths(direct_folder: str, direct_dict: Dict[str, Any]) -> List[str]:
    """
    Reimplementation of load_imgs_direct_search() but only keeps paths, not tensors.
    Looks for images under direct_folder / basename(image_path)
    """
    image_paths: List[str] = []
    keys_to_check = ['images_with_captions', 'images_with_no_captions', 'images_with_caption_matched_tags']

    for key1 in keys_to_check:
        if key1 in direct_dict:
            for page in direct_dict[key1]:
                # original code: image_path = os.path.join(item_folder_path, page['image_path'].split('/')[-1])
                base_name = os.path.basename(page['image_path'])
                local_path = safe_join(direct_folder, base_name)

                # optional: only keep if file looks like an image & exists
                if os.path.exists(local_path) and imghdr.what(local_path) is not None:
                    image_paths.append(local_path)

    # deduplicate while preserving order
    seen = set()
    unique_paths = []
    for p in image_paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)
    return unique_paths


# ============================================================
# 3. BUILDING FINAL RECORDS FOR ONE SPLIT
# ============================================================

def prepare_split(
    items_dict: Dict[str, Any],
    split_name: str,
    root: str,
    only_english: bool = True,
    filter_evidence_to_english: bool = False
) -> List[Dict[str, Any]]:
    
    out_records: List[Dict[str, Any]] = []
    total = len(items_dict)
    print(f"Processing split '{split_name}' with {total} raw items ...")

    for idx, (sample_id, item) in enumerate(items_dict.items(), start=1):
        if idx % 500 == 0:
            print(f"  processed {idx}/{total} ...")

        claim_text = item.get("caption", "")
        if not isinstance(claim_text, str):
            continue

        if only_english:
            is_eng = is_english(claim_text)
        else:
            is_eng = N

        claim_text = process_string(claim_text)

        # ✔ If claim image missing -> keep None
        claim_image_path = None
        if "image_path" in item:
            candidate = safe_join(root, item["image_path"])
            if os.path.exists(candidate):
                claim_image_path = candidate

        # ✔ If annotations missing -> keep empty lists
        evidence_texts = []
        evidence_images = []

        direct_path = safe_join(root, item.get("direct_path", ""))
        inv_path    = safe_join(root, item.get("inv_path", ""))

        direct_anno_path = safe_join(direct_path, "direct_annotation.json")
        inv_anno_path    = safe_join(inv_path, "inverse_annotation.json")

        if os.path.exists(inv_anno_path):
            inv_dict = load_json(inv_anno_path)
            evidence_texts.extend(collect_evidence_captions_from_inverse(inv_dict))

        if os.path.exists(direct_anno_path):
            direct_dict = load_json(direct_anno_path)
            evidence_texts.extend(collect_evidence_captions_from_direct(direct_dict))
            evidence_images.extend(collect_evidence_image_paths(direct_path, direct_dict))

        # Deduplicate evidence text
        seen = set()
        cleaned_texts = []
        for t in evidence_texts:
            t = process_string(str(t))
            if not t:
                continue
            if filter_evidence_to_english and not is_english(t):
                continue
            if t not in seen:
                seen.add(t)
                cleaned_texts.append(t)

        # Label
        label = item.get("label", None)
        try:
            original_label = int(label) if label is not None else None
        except Exception:
            original_label = None

        out_records.append({
            "id": str(sample_id),
            "claim_text": claim_text,
            "claim_image": claim_image_path,  # can be None now
            "evidence_texts": cleaned_texts,  # may be []
            "evidence_images": evidence_images, # may be []
            "original_label": original_label
        })

    print(f"  -> kept {len(out_records)} items for split '{split_name}' after filtering")
    return out_records



# ============================================================
# 4. LOAD RAW JSONS
# ============================================================

train_raw = load_json(safe_join(ROOT, "dataset_items_train.json"))
val_raw   = load_json(safe_join(ROOT, "dataset_items_val.json"))
test_raw  = load_json(safe_join(ROOT, "dataset_items_test.json"))

print("Raw sizes:")
print("  Train:", len(train_raw))
print("  Val  :", len(val_raw))
print("  Test :", len(test_raw))


# ============================================================
# 5. PREPARE ALL SPLITS (ENGLISH-ONLY)
# ============================================================

train_prepared = prepare_split(train_raw, "train", ROOT, only_english=True)
val_prepared   = prepare_split(val_raw, "val", ROOT, only_english=True)
test_prepared  = prepare_split(test_raw, "test", ROOT, only_english=True)

# ============================================================
# 6. SAVE TO JSON FILES
# ============================================================

train_out_path = safe_join(OUTPUT_DIR, "mr2_en_train_prepared.json")
val_out_path   = safe_join(OUTPUT_DIR, "mr2_en_val_prepared.json")
test_out_path  = safe_join(OUTPUT_DIR, "mr2_en_test_prepared.json")

with open(train_out_path, "w", encoding="utf-8") as f:
    json.dump(train_prepared, f, ensure_ascii=False, indent=2)

with open(val_out_path, "w", encoding="utf-8") as f:
    json.dump(val_prepared, f, ensure_ascii=False, indent=2)

with open(test_out_path, "w", encoding="utf-8") as f:
    json.dump(test_prepared, f, ensure_ascii=False, indent=2)

print("Saved:")
print(" ", train_out_path)
print(" ", val_out_path)
print(" ", test_out_path)

