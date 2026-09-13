# # significance_verite.py
# import argparse
# import pandas as pd
# import numpy as np
# import os

# from sklearn.metrics import classification_report, accuracy_score, f1_score
# from scipy.stats import chi2

# # McNemar exact test (binomial)
# try:
#     from scipy.stats import binomtest
#     _HAS_BINOMTEST = True
# except Exception:
#     _HAS_BINOMTEST = False


# # =========================
# # VERITE label mapping
# # =========================
# LABELS = ["true", "miscaptioned", "out-of-context"]
# LABEL2ID = {l: i for i, l in enumerate(LABELS)}
# ID2LABEL = {i: l for l, i in LABEL2ID.items()}


# def _to_str_label(x):
#     """
#     Convert input label into canonical VERITE string label.
#     Accepts:
#       - numeric {0,1,2} (int, numpy int, or digit string)
#       - canonical strings like "true", "miscaptioned", "out-of-context" (case-insensitive)
#       - a few common variants
#     """
#     if pd.isna(x):
#         return None

#     # numeric
#     if isinstance(x, (int, np.integer)) or (isinstance(x, str) and x.strip().isdigit()):
#         i = int(x)
#         return ID2LABEL.get(i, None)

#     # string
#     s = str(x).strip().lower()
#     s = s.replace("_", "-")
#     if s == "out of context":
#         s = "out-of-context"
#     if s in LABEL2ID:
#         return s

#     if s in {"mis-captioned", "mis-caption", "mis captioned"}:
#         return "miscaptioned"

#     return None


# def _find_id_col(df, preferred):
#     if preferred and preferred in df.columns:
#         return preferred
#     for c in ["claim_id", "id", "Id", "ID"]:
#         if c in df.columns:
#             return c
#     raise ValueError(f"Could not find an id column in: {list(df.columns)}")


# def _find_label_col(df, preferred, kind="true_label"):
#     if preferred and preferred in df.columns:
#         return preferred

#     # prefer these in order
#     candidates = [
#         kind, kind.lower(), kind.upper(),
#         "true_label", "gold", "y_true",
#         "pred_label", "prediction", "pred"
#     ]
#     for c in candidates:
#         if c in df.columns:
#             return c
#     raise ValueError(f"Could not find {kind} column in: {list(df.columns)}")


# def _standardize_df(df, name, id_col, true_col, pred_col):
#     out = df.copy()

#     out[id_col] = out[id_col].astype(str).str.strip()

#     out["true_str"] = out[true_col].apply(_to_str_label)
#     out["pred_str"] = out[pred_col].apply(_to_str_label)

#     bad_true = out["true_str"].isna().sum()
#     bad_pred = out["pred_str"].isna().sum()
#     if bad_true or bad_pred:
#         print(f"⚠️ [{name}] Unmapped labels -> true: {bad_true}, pred: {bad_pred}. Dropping those rows.")
#         out = out.dropna(subset=["true_str", "pred_str"])

#     return out[[id_col, "true_str", "pred_str"]].rename(columns={id_col: "id"})


# def bowker_test_symmetry(ct: pd.DataFrame):
#     """
#     Bowker’s test of symmetry for a KxK paired contingency table.
#     ct: DataFrame indexed/columned by same LABELS.
#     """
#     k = ct.shape[0]
#     chi2_stat = 0.0
#     for i in range(k):
#         for j in range(i + 1, k):
#             nij = ct.iat[i, j]
#             nji = ct.iat[j, i]
#             if nij + nji > 0:
#                 chi2_stat += (nij - nji) ** 2 / (nij + nji)

#     df = k * (k - 1) // 2
#     p_value = 1.0 - chi2.cdf(chi2_stat, df)
#     return chi2_stat, df, p_value


# def mcnemar_exact(n01, n10):
#     """
#     Exact McNemar test using a two-sided binomial test on discordant pairs.
#     Returns p-value.
#     """
#     n = n01 + n10
#     if n == 0:
#         return 1.0

#     if _HAS_BINOMTEST:
#         # Two-sided exact
#         res = binomtest(min(n01, n10), n=n, p=0.5, alternative="two-sided")
#         return float(res.pvalue)

#     # Fallback: chi-square approx w/ continuity correction
#     stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
#     return float(1.0 - chi2.cdf(stat, df=1))


# def mcnemar_stats(n01, n10):
#     """
#     Convenience: returns p-value + other helpful values.
#     """
#     n = n01 + n10
#     p = mcnemar_exact(n01, n10)

#     # effect direction on paired accuracy:
#     # positive means new corrects more cases than it breaks
#     delta_correct = n01 - n10

#     # normalized effect size (risk difference on discordant set)
#     rd_discordant = (delta_correct / n) if n > 0 else 0.0

#     # odds ratio on discordants (with Haldane-Anscombe correction if zeros)
#     # OR = n01/n10 (only uses discordants); if n10=0, OR is infinite -> smooth it
#     n01_s, n10_s = n01, n10
#     if n01 == 0 or n10 == 0:
#         n01_s = n01 + 0.5
#         n10_s = n10 + 0.5
#     odds_ratio = n01_s / n10_s

#     return {
#         "n01": int(n01),
#         "n10": int(n10),
#         "discordant": int(n),
#         "delta_correct": int(delta_correct),
#         "rd_discordant": float(rd_discordant),
#         "odds_ratio": float(odds_ratio),
#         "p_value": float(p),
#     }


# def print_classification(name, y_true, y_pred):
#     print("\n" + "=" * 90)
#     print(f"{name} - FULL CLASSIFICATION REPORT")
#     print("=" * 90)
#     print(
#         classification_report(
#             y_true,
#             y_pred,
#             labels=LABELS,
#             target_names=LABELS,
#             digits=4,
#             zero_division=0
#         )
#     )
#     acc = accuracy_score(y_true, y_pred)
#     f1m = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
#     print(f"Accuracy: {acc:.4f} | Macro-F1: {f1m:.4f}")
#     return acc, f1m


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--baseline_csv", required=True, help="Path to baseline predictions CSV")
#     ap.add_argument("--new_csv", required=True, help="Path to best-model predictions CSV")
#     ap.add_argument("--baseline_id_col", default=None, help="ID column in baseline CSV (default: auto)")
#     ap.add_argument("--new_id_col", default=None, help="ID column in new CSV (default: auto)")
#     ap.add_argument("--baseline_true_col", default=None, help="True label column in baseline CSV (default: auto)")
#     ap.add_argument("--baseline_pred_col", default=None, help="Pred label column in baseline CSV (default: auto)")
#     ap.add_argument("--new_true_col", default=None, help="True label column in new CSV (default: auto)")
#     ap.add_argument("--new_pred_col", default=None, help="Pred label column in new CSV (default: auto)")
#     ap.add_argument("--alpha", type=float, default=0.05, help="Significance level (default 0.05)")
#     ap.add_argument("--output_txt", type=str, default=None, help="Optional: save the whole output to a txt file")
#     args = ap.parse_args()

#     # optional: tee print to file
#     if args.output_txt:
#         os.makedirs(os.path.dirname(args.output_txt) or ".", exist_ok=True)
#         import sys
#         class Tee:
#             def __init__(self, *files):
#                 self.files = files
#             def write(self, obj):
#                 for f in self.files:
#                     f.write(obj)
#                     f.flush()
#             def flush(self):
#                 for f in self.files:
#                     f.flush()
#         fout = open(args.output_txt, "w", encoding="utf-8")
#         sys.stdout = Tee(sys.stdout, fout)

#     base_raw = pd.read_csv(args.baseline_csv)
#     new_raw = pd.read_csv(args.new_csv)

#     # detect columns
#     base_id = _find_id_col(base_raw, args.baseline_id_col)
#     new_id = _find_id_col(new_raw, args.new_id_col)

#     base_true = _find_label_col(base_raw, args.baseline_true_col, kind="true_label")
#     base_pred = _find_label_col(base_raw, args.baseline_pred_col, kind="pred_label")

#     new_true = _find_label_col(new_raw, args.new_true_col, kind="true_label")
#     new_pred = _find_label_col(new_raw, args.new_pred_col, kind="pred_label")

#     base = _standardize_df(base_raw, "baseline", base_id, base_true, base_pred)
#     new = _standardize_df(new_raw, "new", new_id, new_true, new_pred)

#     # align by intersection of ids
#     merged = base.merge(new, on="id", suffixes=("_base", "_new"))
#     base_only = set(base["id"]) - set(merged["id"])
#     new_only = set(new["id"]) - set(merged["id"])

#     print("\n" + "=" * 90)
#     print("ALIGNMENT SUMMARY")
#     print("=" * 90)
#     print(f"Baseline rows: {len(base)}")
#     print(f"New rows:      {len(new)}")
#     print(f"Aligned rows:  {len(merged)}")
#     print(f"Baseline-only (dropped): {len(base_only)}")
#     print(f"New-only (dropped):      {len(new_only)}")

#     if len(merged) == 0:
#         raise RuntimeError("No overlapping ids between the two CSVs. Check your id columns / files.")

#     # sanity: check true labels agreement
#     mismatch_true = (merged["true_str_base"] != merged["true_str_new"]).sum()
#     if mismatch_true > 0:
#         print(f"⚠️ WARNING: {mismatch_true} rows have different true labels between files. Using baseline true labels.")

#     # full reports (aligned set)
#     y_true = merged["true_str_base"].values
#     y_pred_base = merged["pred_str_base"].values
#     y_pred_new = merged["pred_str_new"].values

#     acc_b, f1_b = print_classification("BASELINE", y_true, y_pred_base)
#     acc_n, f1_n = print_classification("NEW/BEST MODEL", y_true, y_pred_new)

#     print("\n" + "=" * 90)
#     print("METRIC DELTAS (NEW - BASELINE)")
#     print("=" * 90)
#     print(f"Δ Accuracy  : {acc_n - acc_b:+.4f}")
#     print(f"Δ Macro-F1  : {f1_n - f1_b:+.4f}")

#     # -------------------------
#     # (A) Bowker test on predictions
#     # -------------------------
#     ct = pd.crosstab(
#         pd.Categorical(merged["pred_str_base"], categories=LABELS, ordered=True),
#         pd.Categorical(merged["pred_str_new"], categories=LABELS, ordered=True),
#         rownames=["Baseline pred"],
#         colnames=["New pred"],
#         dropna=False
#     ).reindex(index=LABELS, columns=LABELS, fill_value=0)

#     chi2_stat, df, p_bowker = bowker_test_symmetry(ct)

#     print("\n" + "=" * 90)
#     print("BOWKER’S TEST (multi-class paired symmetry on predictions)")
#     print("=" * 90)
#     print("Contingency table (baseline ↘ new):")
#     print(ct)
#     print(f"\nBowker χ² = {chi2_stat:.4f} | df = {df} | p = {p_bowker:.6g}")
#     print(f"Significant at α={args.alpha}?  {'YES' if p_bowker < args.alpha else 'NO'}")

#     # -------------------------
#     # (B) McNemar on correctness
#     # -------------------------
#     base_correct = (merged["pred_str_base"] == merged["true_str_base"])
#     new_correct = (merged["pred_str_new"] == merged["true_str_base"])

#     n01 = int((~base_correct & new_correct).sum())  # base wrong, new correct
#     n10 = int((base_correct & ~new_correct).sum())  # base correct, new wrong

#     mc = mcnemar_stats(n01, n10)

#     print("\n" + "=" * 90)
#     print("MCNEMAR’S TEST (paired correctness: correct vs incorrect)")
#     print("=" * 90)
#     print(f"Discordant pairs:")
#     print(f"  n01 (base wrong, new correct) = {mc['n01']}")
#     print(f"  n10 (base correct, new wrong) = {mc['n10']}")
#     print(f"  total discordant = {mc['discordant']}")
#     print(f"Exact McNemar p-value = {mc['p_value']:.6g}")
#     print(f"Significant at α={args.alpha}?  {'YES' if mc['p_value'] < args.alpha else 'NO'}")

#     # extra effect info (useful in paper)
#     print("\nEffect direction / magnitude (discordant-only):")
#     print(f"  delta_correct = n01 - n10 = {mc['delta_correct']}  (positive => NEW better)")
#     print(f"  risk-diff on discordants = (n01 - n10)/(n01 + n10) = {mc['rd_discordant']:+.4f}")
#     print(f"  discordant odds ratio (smoothed if needed) = {mc['odds_ratio']:.4f}")

#     if args.output_txt:
#         fout.close()


# if __name__ == "__main__":
#     main()

# significance_verite.py
import argparse
import os
import sys

import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, accuracy_score, f1_score
from scipy.stats import chi2

# McNemar exact test (binomial)
try:
    from scipy.stats import binomtest
    _HAS_BINOMTEST = True
except Exception:
    _HAS_BINOMTEST = False


# =========================
# VERITE label mapping
# =========================
LABELS = ["true", "miscaptioned", "out-of-context"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def _to_str_label(x):
    """
    Convert input label into canonical VERITE string label.
    Accepts:
      - numeric {0,1,2} (int, numpy int, or digit string)
      - canonical strings like "true", "miscaptioned", "out-of-context" (case-insensitive)
      - a few common variants
    """
    if pd.isna(x):
        return None

    # numeric
    if isinstance(x, (int, np.integer)) or (isinstance(x, str) and x.strip().isdigit()):
        i = int(x)
        return ID2LABEL.get(i, None)

    # string
    s = str(x).strip().lower()
    s = s.replace("_", "-")
    if s == "out of context":
        s = "out-of-context"
    if s in LABEL2ID:
        return s

    if s in {"mis-captioned", "mis-caption", "mis captioned"}:
        return "miscaptioned"

    return None


def _find_id_col(df, preferred):
    if preferred and preferred in df.columns:
        return preferred
    for c in ["claim_id", "id", "Id", "ID"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find an id column in: {list(df.columns)}")


def _find_label_col(df, preferred, kind="true_label"):
    if preferred and preferred in df.columns:
        return preferred

    # prefer these in order
    candidates = [
        kind, kind.lower(), kind.upper(),
        "true_label", "gold", "y_true",
        "pred_label", "prediction", "pred"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find {kind} column in: {list(df.columns)}")


def _standardize_df(df, name, id_col, true_col, pred_col):
    out = df.copy()

    out[id_col] = out[id_col].astype(str).str.strip()

    out["true_str"] = out[true_col].apply(_to_str_label)
    out["pred_str"] = out[pred_col].apply(_to_str_label)

    bad_true = out["true_str"].isna().sum()
    bad_pred = out["pred_str"].isna().sum()
    if bad_true or bad_pred:
        print(f"⚠️ [{name}] Unmapped labels -> true: {bad_true}, pred: {bad_pred}. Dropping those rows.")
        out = out.dropna(subset=["true_str", "pred_str"])

    return out[[id_col, "true_str", "pred_str"]].rename(columns={id_col: "id"})


def bowker_test_symmetry(ct: pd.DataFrame):
    """
    Bowker’s test of symmetry for a KxK paired contingency table.
    ct: DataFrame indexed/columned by same LABELS.
    """
    k = ct.shape[0]
    chi2_stat = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            nij = ct.iat[i, j]
            nji = ct.iat[j, i]
            if nij + nji > 0:
                chi2_stat += (nij - nji) ** 2 / (nij + nji)

    df = k * (k - 1) // 2
    p_value = 1.0 - chi2.cdf(chi2_stat, df)
    return chi2_stat, df, p_value


def mcnemar_exact_p(n01, n10):
    """
    Exact McNemar test using a two-sided binomial test on discordant pairs.
    Returns p-value.
    """
    n = n01 + n10
    if n == 0:
        return 1.0

    if _HAS_BINOMTEST:
        res = binomtest(min(n01, n10), n=n, p=0.5, alternative="two-sided")
        return float(res.pvalue)

    # fallback (chi-square w/ continuity correction)
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return float(1.0 - chi2.cdf(stat, df=1))


def mcnemar_chi_square(n01, n10, continuity=True):
    """
    McNemar chi-square statistic (df=1).
      - continuity=True  => (|n01-n10|-1)^2/(n01+n10)
      - continuity=False => (n01-n10)^2/(n01+n10)
    Returns (chi2_stat, p_value).
    """
    n = n01 + n10
    if n == 0:
        return 0.0, 1.0

    if continuity:
        stat = (abs(n01 - n10) - 1) ** 2 / n
    else:
        stat = (n01 - n10) ** 2 / n

    p = 1.0 - chi2.cdf(stat, df=1)
    return float(stat), float(p)


def mcnemar_stats(n01, n10):
    """
    Returns exact p + chi-square stats (with/without continuity) + effect info.
    """
    n = n01 + n10
    p_exact = mcnemar_exact_p(n01, n10)

    chi2_cc, p_cc = mcnemar_chi_square(n01, n10, continuity=True)
    chi2_nocc, p_nocc = mcnemar_chi_square(n01, n10, continuity=False)

    delta_correct = n01 - n10
    rd_discordant = (delta_correct / n) if n > 0 else 0.0

    # odds ratio on discordants (Haldane-Anscombe correction if zeros)
    n01_s, n10_s = n01, n10
    if n01 == 0 or n10 == 0:
        n01_s = n01 + 0.5
        n10_s = n10 + 0.5
    odds_ratio = n01_s / n10_s

    return {
        "n01": int(n01),
        "n10": int(n10),
        "discordant": int(n),
        "delta_correct": int(delta_correct),
        "rd_discordant": float(rd_discordant),
        "odds_ratio": float(odds_ratio),
        "p_exact": float(p_exact),
        "chi2_cc": float(chi2_cc),
        "p_cc": float(p_cc),
        "chi2_no_cc": float(chi2_nocc),
        "p_no_cc": float(p_nocc),
    }


def print_classification(name, y_true, y_pred):
    print("\n" + "=" * 90)
    print(f"{name} - FULL CLASSIFICATION REPORT")
    print("=" * 90)
    print(
        classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            target_names=LABELS,
            digits=4,
            zero_division=0
        )
    )
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    print(f"Accuracy: {acc:.4f} | Macro-F1: {f1m:.4f}")
    return acc, f1m


class Tee:
    """Safe stdout tee (won't crash if file is closed / removed)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_csv", required=True, help="Path to baseline predictions CSV")
    ap.add_argument("--new_csv", required=True, help="Path to best-model predictions CSV")
    ap.add_argument("--baseline_id_col", default=None, help="ID column in baseline CSV (default: auto)")
    ap.add_argument("--new_id_col", default=None, help="ID column in new CSV (default: auto)")
    ap.add_argument("--baseline_true_col", default=None, help="True label column in baseline CSV (default: auto)")
    ap.add_argument("--baseline_pred_col", default=None, help="Pred label column in baseline CSV (default: auto)")
    ap.add_argument("--new_true_col", default=None, help="True label column in new CSV (default: auto)")
    ap.add_argument("--new_pred_col", default=None, help="Pred label column in new CSV (default: auto)")
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance level (default 0.05)")
    ap.add_argument("--output_txt", type=str, default=None, help="Optional: save the whole output to a txt file")
    args = ap.parse_args()

    original_stdout = sys.stdout
    fout = None
    try:
        if args.output_txt:
            os.makedirs(os.path.dirname(args.output_txt) or ".", exist_ok=True)
            fout = open(args.output_txt, "w", encoding="utf-8")
            sys.stdout = Tee(original_stdout, fout)

        base_raw = pd.read_csv(args.baseline_csv)
        new_raw = pd.read_csv(args.new_csv)

        # detect columns
        base_id = _find_id_col(base_raw, args.baseline_id_col)
        new_id = _find_id_col(new_raw, args.new_id_col)

        base_true = _find_label_col(base_raw, args.baseline_true_col, kind="true_label")
        base_pred = _find_label_col(base_raw, args.baseline_pred_col, kind="pred_label")

        new_true = _find_label_col(new_raw, args.new_true_col, kind="true_label")
        new_pred = _find_label_col(new_raw, args.new_pred_col, kind="pred_label")

        base = _standardize_df(base_raw, "baseline", base_id, base_true, base_pred)
        new = _standardize_df(new_raw, "new", new_id, new_true, new_pred)

        # align by intersection of ids
        merged = base.merge(new, on="id", suffixes=("_base", "_new"))
        base_only = set(base["id"]) - set(merged["id"])
        new_only = set(new["id"]) - set(merged["id"])

        print("\n" + "=" * 90)
        print("ALIGNMENT SUMMARY")
        print("=" * 90)
        print(f"Baseline rows: {len(base)}")
        print(f"New rows:      {len(new)}")
        print(f"Aligned rows:  {len(merged)}")
        print(f"Baseline-only (dropped): {len(base_only)}")
        print(f"New-only (dropped):      {len(new_only)}")

        if len(merged) == 0:
            raise RuntimeError("No overlapping ids between the two CSVs. Check your id columns / files.")

        # sanity: check true labels agreement
        mismatch_true = (merged["true_str_base"] != merged["true_str_new"]).sum()
        if mismatch_true > 0:
            print(f"⚠️ WARNING: {mismatch_true} rows have different true labels between files. Using baseline true labels.")

        # full reports (aligned set)
        y_true = merged["true_str_base"].values
        y_pred_base = merged["pred_str_base"].values
        y_pred_new = merged["pred_str_new"].values

        acc_b, f1_b = print_classification("BASELINE", y_true, y_pred_base)
        acc_n, f1_n = print_classification("NEW/BEST MODEL", y_true, y_pred_new)

        print("\n" + "=" * 90)
        print("METRIC DELTAS (NEW - BASELINE)")
        print("=" * 90)
        print(f"Δ Accuracy  : {acc_n - acc_b:+.4f}")
        print(f"Δ Macro-F1  : {f1_n - f1_b:+.4f}")

        # -------------------------
        # (A) Bowker test on predictions
        # -------------------------
        ct = pd.crosstab(
            pd.Categorical(merged["pred_str_base"], categories=LABELS, ordered=True),
            pd.Categorical(merged["pred_str_new"], categories=LABELS, ordered=True),
            rownames=["Baseline pred"],
            colnames=["New pred"],
            dropna=False
        ).reindex(index=LABELS, columns=LABELS, fill_value=0)

        chi2_stat, df, p_bowker = bowker_test_symmetry(ct)

        print("\n" + "=" * 90)
        print("BOWKER’S TEST (multi-class paired symmetry on predictions)")
        print("=" * 90)
        print("Contingency table (baseline ↘ new):")
        print(ct)
        print(f"\nBowker χ² = {chi2_stat:.4f} | df = {df} | p = {p_bowker:.6g}")
        print(f"Significant at α={args.alpha}?  {'YES' if p_bowker < args.alpha else 'NO'}")

        # -------------------------
        # (B) McNemar on correctness
        # -------------------------
        base_correct = (merged["pred_str_base"] == merged["true_str_base"])
        new_correct = (merged["pred_str_new"] == merged["true_str_base"])

        n01 = int((~base_correct & new_correct).sum())   # base wrong, new correct
        n10 = int((base_correct & ~new_correct).sum())   # base correct, new wrong

        mc = mcnemar_stats(n01, n10)

        print("\n" + "=" * 90)
        print("MCNEMAR’S TEST (paired correctness: correct vs incorrect)")
        print("=" * 90)
        print("Discordant pairs:")
        print(f"  n01 (base wrong, new correct) = {mc['n01']}")
        print(f"  n10 (base correct, new wrong) = {mc['n10']}")
        print(f"  total discordant = {mc['discordant']}")

        print("\nExact McNemar (binomial) p-value:")
        print(f"  p_exact = {mc['p_exact']:.6g}")
        print(f"  Significant at α={args.alpha}?  {'YES' if mc['p_exact'] < args.alpha else 'NO'}")

        print("\nMcNemar chi-square (df=1):")
        print(f"  χ² (with continuity corr)    = {mc['chi2_cc']:.6f} | p = {mc['p_cc']:.6g}")
        print(f"  χ² (without continuity corr) = {mc['chi2_no_cc']:.6f} | p = {mc['p_no_cc']:.6g}")
        print(f"  Significant at α={args.alpha}?  {'YES' if mc['p_cc'] < args.alpha else 'NO'}  (using continuity-corrected p)")

        print("\nEffect direction / magnitude (discordant-only):")
        print(f"  delta_correct = n01 - n10 = {mc['delta_correct']}  (positive => NEW better)")
        print(f"  risk-diff on discordants = (n01 - n10)/(n01 + n10) = {mc['rd_discordant']:+.4f}")
        print(f"  discordant odds ratio (smoothed if needed) = {mc['odds_ratio']:.4f}")

        if args.output_txt:
            print(f"\n✅ Saved output log to: {args.output_txt}")

    finally:
        # restore stdout BEFORE closing file (prevents flush-on-closed-file crash)
        sys.stdout = original_stdout
        if fout is not None:
            try:
                fout.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()

