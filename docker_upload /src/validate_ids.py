from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_candidate_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            ids.add(json.loads(line)["candidate_id"])
    return ids


def validate_submission_ids(candidates_path: Path, submission_path: Path) -> list[str]:
    errors: list[str] = []
    valid_ids = load_candidate_ids(candidates_path)
    seen: set[str] = set()

    with submission_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["candidate_id", "rank", "score", "reasoning"]:
            errors.append("CSV header must be candidate_id,rank,score,reasoning")
            return errors
        rows = list(reader)

    if len(rows) != 100:
        errors.append(f"Expected 100 rows, found {len(rows)}")

    for row in rows:
        candidate_id = row["candidate_id"]
        if candidate_id not in valid_ids:
            errors.append(f"Unknown candidate_id: {candidate_id}")
        if candidate_id in seen:
            errors.append(f"Duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)
        if not row.get("reasoning", "").strip():
            errors.append(f"Empty reasoning for {candidate_id}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate submission candidate IDs against the source JSONL.")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--submission", required=True, type=Path)
    args = parser.parse_args()

    errors = validate_submission_ids(args.candidates, args.submission)
    if errors:
        print(f"Local validation failed ({len(errors)} issue(s)):")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("Local ID validation passed.")


if __name__ == "__main__":
    main()
