#!/usr/bin/env python3
"""
significance_factify2.py

Paired significance testing between two models on FACTIFY-2 (5-class):
  1) McNemar's test on *correct vs incorrect* (paired on the same examples)
     - Exact (two-sided binomial) p-value
     - Chi-square statistic (df=1) with and without continuity correction

  2) Bowker's test of symmetry on the *KxK paired prediction table* (multi-class)

Works with typical prediction CSVs that contain:
  - an example id (e.g., claim_id / id / sample_id)
  - true label column
  - predicted label column

If your two CSVs do NOT share a common id, you can align by (claim text + true label + duplicate index)
using --align_on claim (requires both CSVs to have a 'claim' column).
"""

import argparse
import os
import re
import sys
import unicodedata
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2

from sklearn.metrics import classification_report, accuracy_score, f1_score

# McNemar exact test (binomial)
try:
    from scipy.stats import binomtest
    _HAS_BINOMTEST = True
except Exception:
    _HAS_BINOMTEST = False


# =========================
# FACTIFY-2 label mapping
# =========================
LABELS = [
    "Support_Multimodal",
    "Support_Text",
    "Insufficient_Multimodal",
    "Insufficient_Text",
    "Refute",
]

# Numeric-id mapping (0..4)
ID2LABEL_DEFAULT = {
    0: "Support_Multimodal",
    1: "Support_Text",
    2: "Insufficient_Multimodal",
    3: "Insufficient_Text",
    4: "Refute",
}


def _norm_str(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s


def _to_str_label(x, id2label: dict) -> Optional[str]:
    """
    Convert input label into canonical FACTIFY-2 string label.

    Accepts:
      - numeric {0..4} (int/np.int/float integer-like, or digit string)
      - canonical strings like 'Support_Text' (case-insensitive; '_'/'-'/space tolerant)
      - abbreviations: ST, SM, IT, IM
      - a few common variants (supports/refutes/nei)
    """
    if pd.isna(x):
        return None

    # numeric-like
    if isinstance(x, (int, np.integer)):
        return id2label.get(int(x), None)

    if isinstance(x, (float, np.floating)) and float(x).is_integer():
        return id2label.get(int(x), None)

    if isinstance(x, str):
        xs = x.strip()
        if xs.isdigit():
            return id2label.get(int(xs), None)

        # sometimes saved as "3.0"
        try:
            xf = float(xs)
            if xf.is_integer():
                return id2label.get(int(xf), None)
        except Exception:
            pass

        s = _norm_str(xs)

        # abbreviations / common variants
        if s in {"st", "support_text"}:
            return "Support_Text"
        if s in {"sm", "support_multimodal", "support_multi_modal"}:
            return "Support_Multimodal"
        if s in {"it", "insufficient_text", "not_enough_info_text", "nei_text"}:
            return "Insufficient_Text"
        if s in {"im", "insufficient_multimodal", "insufficient_multi_modal", "nei_multimodal"}:
            return "Insufficient_Multimodal"

        if s in {"support", "supports"}:
            return "Support_Text"
        if s in {"refute", "refutes", "refuted"}:
            return "Refute"
        if s in {"nei", "not_enough_info", "insufficient"}:
            return "Insufficient_Text"

        canon = {
            "support_multimodal": "Support_Multimodal",
            "support_text": "Support_Text",
            "insufficient_multimodal": "Insufficient_Multimodal",
            "insufficient_text": "Insufficient_Text",
            "refute": "Refute",
        }
        return canon.get(s, None)

    return None


def _find_first_existing(df: pd.DataFrame, preferred: Optional[str], candidates) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find required column. Tried: {candidates}. Available: {list(df.columns)}")


def _find_id_col(df: pd.DataFrame, preferred: Optional[str]) -> str:
    return _find_first_existing(
        df,
        preferred,
        candidates=["claim_id", "Id", "id", "ID", "sample_id", "example_id", "uid", "guid"],
    )


def _find_true_col(df: pd.DataFrame, preferred: Optional[str]) -> str:
    return _find_first_existing(
        df,
        preferred,
        candidates=["true_label", "gold_label", "gold", "label", "Category", "y_true", "gt", "ground_truth"],
    )


def _find_pred_col(df: pd.DataFrame, preferred: Optional[str]) -> str:
    return _find_first_existing(
        df,
        preferred,
        candidates=["predicted_label", "prediction", "pred_label", "pred", "y_pred", "prediction_label"],
    )


def _standardize_df(
    df: pd.DataFrame,
    name: str,
    id_col: str,
    true_col: str,
    pred_col: str,
    id2label: dict,
) -> pd.DataFrame:
    """
    Standardize to columns: [_row, id, true_str, pred_str]
    where _row is the original row index from the source CSV.
    """
    out = df.copy().reset_index().rename(columns={"index": "_row"})

    out[id_col] = out[id_col].astype(str).str.strip()

    out["true_str"] = out[true_col].apply(lambda x: _to_str_label(x, id2label))
    out["pred_str"] = out[pred_col].apply(lambda x: _to_str_label(x, id2label))

    bad_true = int(out["true_str"].isna().sum())
    bad_pred = int(out["pred_str"].isna().sum())
    if bad_true or bad_pred:
        print(f"⚠️ [{name}] Unmapped labels -> true: {bad_true}, pred: {bad_pred}. Dropping those rows.")
        out = out.dropna(subset=["true_str", "pred_str"])

    return out[["_row", id_col, "true_str", "pred_str"]].rename(columns={id_col: "id"})


def _normalize_claim_for_alignment(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("\u200b", "")
    s = s.lower().strip()
    s = re.sub(r"https?://\S+", "", s)          # remove URLs
    s = re.sub(r"[^a-z0-9\s]+", " ", s)         # keep alphanum/spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _align(
    base_raw: pd.DataFrame,
    new_raw: pd.DataFrame,
    base_std: pd.DataFrame,
    new_std: pd.DataFrame,
    align_on: str,
    min_overlap: float,
) -> Tuple[pd.DataFrame, str]:
    """
    Returns (merged_df, mode_used).

    align_on:
      - "id": merge on 'id' (requires both ids refer to same examples)
      - "claim": merge using (normalized claim + true label + within-duplicate index)
      - "row": align by row index (unsafe unless both CSVs are already in identical order)
      - "auto": try id, then claim, then row (if possible)
    """

    def merge_on_id():
        return base_std.merge(new_std, on="id", suffixes=("_base", "_new"))

    def merge_on_claim():
        if "claim" not in base_raw.columns or "claim" not in new_raw.columns:
            raise ValueError("claim alignment requested but one of the CSVs lacks a 'claim' column.")

        b = base_raw.copy().reset_index().rename(columns={"index": "_row"}).merge(base_std, on="_row", how="inner")
        n = new_raw.copy().reset_index().rename(columns={"index": "_row"}).merge(new_std, on="_row", how="inner")

        b["claim_norm"] = b["claim"].apply(_normalize_claim_for_alignment)
        n["claim_norm"] = n["claim"].apply(_normalize_claim_for_alignment)

        b["key"] = b["claim_norm"] + "||" + b["true_str"]
        n["key"] = n["claim_norm"] + "||" + n["true_str"]

        b["kidx"] = b.groupby("key").cumcount()
        n["kidx"] = n.groupby("key").cumcount()

        b["merge_key"] = b["key"] + "||" + b["kidx"].astype(str)
        n["merge_key"] = n["key"] + "||" + n["kidx"].astype(str)

        merged = (
            b[["merge_key", "true_str", "pred_str"]]
            .rename(columns={"true_str": "true_str_base", "pred_str": "pred_str_base"})
            .merge(
                n[["merge_key", "true_str", "pred_str"]]
                .rename(columns={"true_str": "true_str_new", "pred_str": "pred_str_new"}),
                on="merge_key",
                how="inner",
            )
        )
        return merged

    def merge_on_row():
        if len(base_std) != len(new_std):
            raise ValueError(f"row alignment requires same length. baseline={len(base_std)}, new={len(new_std)}")
        return pd.DataFrame({
            "true_str_base": base_std["true_str"].values,
            "pred_str_base": base_std["pred_str"].values,
            "true_str_new": new_std["true_str"].values,
            "pred_str_new": new_std["pred_str"].values,
        })

    attempts = []
    if align_on == "id":
        attempts.append(("id", merge_on_id()))
    elif align_on == "claim":
        attempts.append(("claim", merge_on_claim()))
    elif align_on == "row":
        attempts.append(("row", merge_on_row()))
    elif align_on == "auto":
        for fn, name in [(merge_on_id, "id"), (merge_on_claim, "claim"), (merge_on_row, "row")]:
            try:
                attempts.append((name, fn()))
            except Exception:
                pass
    else:
        raise ValueError("--align_on must be one of: auto, id, claim, row")

    if not attempts:
        raise RuntimeError("No alignment strategy succeeded. Use --align_on id/claim/row explicitly.")

    mode_used, merged_best = max(attempts, key=lambda t: len(t[1]))

    denom = min(len(base_std), len(new_std))
    overlap_ratio = (len(merged_best) / denom) if denom > 0 else 0.0

    print("\n" + "=" * 90)
    print("ALIGNMENT SUMMARY")
    print("=" * 90)
    print(f"Alignment mode used: {mode_used}")
    print(f"Baseline standardized rows: {len(base_std)}")
    print(f"New standardized rows:      {len(new_std)}")
    print(f"Aligned (paired) rows:      {len(merged_best)}")
    print(f"Overlap ratio (aligned / min(rows)) = {overlap_ratio:.4f}")

    if overlap_ratio < min_overlap:
        raise RuntimeError(
            f"Alignment overlap too low ({overlap_ratio:.3f} < min_overlap={min_overlap}).\n"
            "Usually: the two CSVs are not the same test set, or IDs do not refer to the same examples.\n"
            "Fix by ensuring both CSVs share the same unique example id (recommended) and use --align_on id."
        )

    return merged_best, mode_used


def bowker_test_symmetry(ct: pd.DataFrame) -> Tuple[float, int, float]:
    """Bowker’s test of symmetry for a KxK paired contingency table."""
    k = ct.shape[0]
    chi2_stat = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            nij = ct.iat[i, j]
            nji = ct.iat[j, i]
            if nij + nji > 0:
                chi2_stat += (nij - nji) ** 2 / (nij + nji)

    df = k * (k - 1) // 2  # for K=5 => 10
    p_value = 1.0 - chi2.cdf(chi2_stat, df)
    return float(chi2_stat), int(df), float(p_value)


def mcnemar_exact_p(n01: int, n10: int) -> float:
    n = n01 + n10
    if n == 0:
        return 1.0
    if _HAS_BINOMTEST:
        res = binomtest(min(n01, n10), n=n, p=0.5, alternative="two-sided")
        return float(res.pvalue)

    stat = (abs(n01 - n10) - 1) ** 2 / n
    return float(1.0 - chi2.cdf(stat, df=1))


def mcnemar_chi_square(n01: int, n10: int, continuity: bool = True) -> Tuple[float, float]:
    n = n01 + n10
    if n == 0:
        return 0.0, 1.0
    if continuity:
        stat = (abs(n01 - n10) - 1) ** 2 / n
    else:
        stat = (n01 - n10) ** 2 / n
    p = 1.0 - chi2.cdf(stat, df=1)
    return float(stat), float(p)


def print_classification(name: str, y_true, y_pred) -> Tuple[float, float]:
    print("\n" + "=" * 90)
    print(f"{name} - CLASSIFICATION REPORT (FACTIFY-2)")
    print("=" * 90)
    print(
        classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            target_names=LABELS,
            digits=4,
            zero_division=0,
        )
    )
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    print(f"Accuracy: {acc:.4f} | Macro-F1: {f1m:.4f}")
    return float(acc), float(f1m)


class Tee:
    """Safe stdout tee."""
    def __init__(self, *files):
        self.files = list(files)

    def write(self, obj):
        for f in list(self.files):
            try:
                f.write(obj)
                f.flush()
            except Exception:
                try:
                    self.files.remove(f)
                except Exception:
                    pass

    def flush(self):
        for f in list(self.files):
            try:
                f.flush()
            except Exception:
                try:
                    self.files.remove(f)
                except Exception:
                    pass


def _parse_id_map(s: Optional[str]) -> dict:
    """
    Parse mapping like:
      "0=Support_Multimodal,1=Support_Text,2=Insufficient_Multimodal,3=Insufficient_Text,4=Refute"
    """
    if not s:
        return dict(ID2LABEL_DEFAULT)
    out = {}
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Bad --id_map entry '{p}'. Use 'int=Label'.")
        k, v = p.split("=", 1)
        k = int(k.strip())
        v = v.strip()
        if v not in LABELS:
            raise ValueError(f"Bad label '{v}' in --id_map. Must be one of: {LABELS}")
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_csv", required=True)
    ap.add_argument("--new_csv", required=True)

    ap.add_argument("--baseline_id_col", default=None)
    ap.add_argument("--new_id_col", default=None)
    ap.add_argument("--baseline_true_col", default=None)
    ap.add_argument("--baseline_pred_col", default=None)
    ap.add_argument("--new_true_col", default=None)
    ap.add_argument("--new_pred_col", default=None)

    ap.add_argument("--align_on", choices=["auto", "id", "claim", "row"], default="auto")
    ap.add_argument("--min_overlap", type=float, default=0.95)

    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--id_map", type=str, default=None)
    ap.add_argument("--output_txt", type=str, default=None)
    args = ap.parse_args()

    id2label = _parse_id_map(args.id_map)

    original_stdout = sys.stdout
    fout = None
    try:
        if args.output_txt:
            os.makedirs(os.path.dirname(args.output_txt) or ".", exist_ok=True)
            fout = open(args.output_txt, "w", encoding="utf-8")
            sys.stdout = Tee(original_stdout, fout)

        base_raw = pd.read_csv(args.baseline_csv)
        new_raw = pd.read_csv(args.new_csv)

        base_id = _find_id_col(base_raw, args.baseline_id_col)
        new_id = _find_id_col(new_raw, args.new_id_col)

        base_true = _find_true_col(base_raw, args.baseline_true_col)
        base_pred = _find_pred_col(base_raw, args.baseline_pred_col)

        new_true = _find_true_col(new_raw, args.new_true_col)
        new_pred = _find_pred_col(new_raw, args.new_pred_col)

        print("\n" + "=" * 90)
        print("COLUMN DETECTION")
        print("=" * 90)
        print(f"Baseline: id={base_id} | true={base_true} | pred={base_pred}")
        print(f"New     : id={new_id} | true={new_true} | pred={new_pred}")

        base_std = _standardize_df(base_raw, "baseline", base_id, base_true, base_pred, id2label)
        new_std = _standardize_df(new_raw, "new", new_id, new_true, new_pred, id2label)

        merged, _ = _align(base_raw, new_raw, base_std, new_std, args.align_on, args.min_overlap)

        mismatch_true = int((merged["true_str_base"] != merged["true_str_new"]).sum())
        if mismatch_true > 0:
            print(f"⚠️ WARNING: {mismatch_true} paired rows have different TRUE labels between files.")
            print("   Using baseline true labels (true_str_base) for evaluation/tests.")

        y_true = merged["true_str_base"].values
        y_pred_base = merged["pred_str_base"].values
        y_pred_new = merged["pred_str_new"].values

        acc_b, f1_b = print_classification("BASELINE", y_true, y_pred_base)
        acc_n, f1_n = print_classification("NEW MODEL", y_true, y_pred_new)

        print("\n" + "=" * 90)
        print("METRIC DELTAS (NEW - BASELINE)")
        print("=" * 90)
        print(f"Δ Accuracy : {acc_n - acc_b:+.4f}")
        print(f"Δ Macro-F1 : {f1_n - f1_b:+.4f}")

        # Bowker
        ct = pd.crosstab(
            pd.Categorical(y_pred_base, categories=LABELS, ordered=True),
            pd.Categorical(y_pred_new, categories=LABELS, ordered=True),
            rownames=["Baseline pred"],
            colnames=["New pred"],
            dropna=False,
        ).reindex(index=LABELS, columns=LABELS, fill_value=0)

        chi2_stat, df_b, p_bowker = bowker_test_symmetry(ct)

        print("\n" + "=" * 90)
        print("BOWKER’S TEST (multi-class paired symmetry on predictions)")
        print("=" * 90)
        print("Contingency table (baseline ↘ new):")
        print(ct)
        print(f"\nBowker χ² = {chi2_stat:.6f} | df = {df_b} | p = {p_bowker:.6g}")
        print(f"Reject symmetry at α={args.alpha}? {'YES' if p_bowker < args.alpha else 'NO'}")

        # McNemar
        base_correct = (y_pred_base == y_true)
        new_correct = (y_pred_new == y_true)

        n01 = int((~base_correct & new_correct).sum())
        n10 = int((base_correct & ~new_correct).sum())

        p_exact = mcnemar_exact_p(n01, n10)
        chi2_cc, p_cc = mcnemar_chi_square(n01, n10, continuity=True)
        chi2_nocc, p_nocc = mcnemar_chi_square(n01, n10, continuity=False)

        print("\n" + "=" * 90)
        print("MCNEMAR’S TEST (paired correctness: correct vs incorrect)")
        print("=" * 90)
        print("Discordant pairs:")
        print(f"  n01 (baseline wrong, new correct) = {n01}")
        print(f"  n10 (baseline correct, new wrong) = {n10}")
        print(f"  total discordant = {n01 + n10}")

        print("\nExact McNemar (two-sided binomial):")
        print(f"  p_exact = {p_exact:.6g}")
        print(f"  Significant at α={args.alpha}? {'YES' if p_exact < args.alpha else 'NO'}")

        print("\nMcNemar chi-square (df=1):")
        print(f"  χ² (with continuity corr)    = {chi2_cc:.6f} | p = {p_cc:.6g}")
        print(f"  χ² (without continuity corr) = {chi2_nocc:.6f} | p = {p_nocc:.6g}")

        n = n01 + n10
        delta_correct = n01 - n10
        rd_discordant = (delta_correct / n) if n > 0 else 0.0
        n01_s, n10_s = n01, n10
        if n01 == 0 or n10 == 0:
            n01_s = n01 + 0.5
            n10_s = n10 + 0.5
        odds_ratio = n01_s / n10_s

        print("\nEffect (discordant-only):")
        print(f"  delta_correct = n01 - n10 = {delta_correct}  (positive => NEW better)")
        print(f"  risk-diff on discordants = (n01 - n10)/(n01 + n10) = {rd_discordant:+.4f}")
        print(f"  discordant odds ratio (smoothed if needed) = {odds_ratio:.4f}")

        if args.output_txt:
            print(f"\n✅ Saved output log to: {args.output_txt}")

    finally:
        sys.stdout = original_stdout
        if fout is not None:
            try:
                fout.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
