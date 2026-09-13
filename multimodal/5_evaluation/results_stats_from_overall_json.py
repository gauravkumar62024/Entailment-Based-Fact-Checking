import json
import argparse
import numpy as np

def parse_accuracy_and_macro_f1(report_str: str):
    """
    Parse accuracy and macro F1 from a sklearn classification_report string.

    Works with typical sklearn formatting like:
          accuracy                         0.5588       102
         macro avg     0.5624    0.5546    0.5582       102
    Returns: (accuracy, macro_f1)
    """
    acc = None
    macro_f1 = None

    lines = report_str.splitlines()
    for line in lines:
        stripped = line.strip()

        if stripped.startswith("accuracy"):
            # Example tokens: ['accuracy', '0.5588', '102']
            tokens = stripped.split()
            # best effort: find the first float in the line (usually accuracy value)
            for t in tokens:
                try:
                    acc = float(t)
                    break
                except ValueError:
                    continue

        elif stripped.startswith("macro avg"):
            # Example tokens: ['macro', 'avg', '0.5624', '0.5546', '0.5582', '102']
            tokens = stripped.split()
            # macro avg line: precision recall f1 support -> f1 is tokens[-2]
            try:
                macro_f1 = float(tokens[-2])
            except (ValueError, IndexError):
                pass

    if acc is None or macro_f1 is None:
        raise ValueError(
            "Could not parse accuracy and/or macro F1 from classification_report.\n"
            "Report was:\n" + report_str
        )

    return acc, macro_f1


def summarize_overall_results(json_path: str):
    """
    For a single overall_results_*.json file:
      - prints FULL classification_report for each seed
      - prints mean/std of accuracy and macro F1 across seeds
    """
    print("\n" + "=" * 100)
    print(f"Processing results file: {json_path}")
    print("=" * 100)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Expected structure:
    # { dataset_id: [ { 'seed': ..., 'test_report': '...' }, ... ] }
    for dataset_id, runs in data.items():
        print("\n" + "#" * 100)
        print(f"DATASET: {dataset_id}")
        print("#" * 100)

        accs = []
        f1s = []

        for run in runs:
            seed = run.get("seed", "NA")
            report_str = run.get("test_report", "")

            print("\n" + "-" * 100)
            print(f"SEED: {seed}")
            print("-" * 100)
            print(report_str.strip())  # ✅ FULL classification report output

            # Still compute stats
            acc, macro_f1 = parse_accuracy_and_macro_f1(report_str)
            accs.append(acc)
            f1s.append(macro_f1)

            print(f"\nParsed summary → accuracy = {acc:.4f}, macro F1 = {macro_f1:.4f}")

        accs = np.array(accs, dtype=float)
        f1s = np.array(f1s, dtype=float)

        print("\n" + "=" * 100)
        print(f"SUMMARY OVER SEEDS for dataset: {dataset_id}")
        print("=" * 100)
        print(f"Accuracy: mean = {accs.mean():.4f}, std = {accs.std(ddof=0):.4f}")
        print(f"Macro F1: mean = {f1s.mean():.4f}, std = {f1s.std(ddof=0):.4f}")
        print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Print full classification reports per seed + mean/std stats from overall_results_*.json"
    )
    parser.add_argument(
        "--results_files",
        type=str,
        nargs="+",
        required=True,
        help="One or more overall_results_*.json files."
    )

    args = parser.parse_args()

    for path in args.results_files:
        summarize_overall_results(path)


if __name__ == "__main__":
    main()
