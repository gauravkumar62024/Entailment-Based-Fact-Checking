import json
import os

def load_json_or_jsonl(path):
    """
    Robust loader for:
    - JSON array files
    - Single JSON object files
    - Proper JSONL files (one JSON object per line)
    """

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    data = []

    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        # Case 1: JSON array or single JSON object
        if first_char in ["[", "{"]:
            try:
                obj = json.load(f)
                if isinstance(obj, list):
                    return obj
                return [obj]
            except json.JSONDecodeError:
                # 🔑 IMPORTANT: rewind before JSONL parsing
                f.seek(0)

        # Case 2: JSONL (one JSON object per line)
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_no} in {path}:\n{line[:200]}"
                ) from e

    return data
