#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import heapq
import json
from pathlib import Path

from src.features import extract_features
from src.reasoning import build_reasoning
from src.scoring import ScoredCandidate, final_csv_scores, score_candidate, sort_candidates


KEEP_TOP_N = 500
FINAL_N = 100


def iter_candidates(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def rank_candidates(candidates_path: Path) -> list[ScoredCandidate]:
    heap: list[tuple[float, str, ScoredCandidate]] = []
    for candidate in iter_candidates(candidates_path):
        features = extract_features(candidate)
        score = score_candidate(features)
        item = ScoredCandidate(candidate_id=features.candidate_id, score=score, features=features)
        key = (score, _reverse_candidate_id(features.candidate_id), item)
        if len(heap) < KEEP_TOP_N:
            heapq.heappush(heap, key)
        else:
            if key > heap[0]:
                heapq.heapreplace(heap, key)

    candidates = [item for _, _, item in heap]
    return sort_candidates(candidates)[:FINAL_N]


def write_submission(scored: list[ScoredCandidate], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    display_scores = final_csv_scores(scored)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (item, display_score) in enumerate(zip(scored, display_scores), start=1):
            writer.writerow(
                [
                    item.candidate_id,
                    rank,
                    f"{display_score:.6f}",
                    build_reasoning(item.features, rank),
                ]
            )


def _reverse_candidate_id(candidate_id: str) -> int:
    """Heap helper: smaller IDs should win ties, so larger IDs are worse."""
    try:
        return -int(candidate_id.split("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank Redrob hackathon candidates.")
    parser.add_argument("--candidates", required=True, type=Path, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, type=Path, help="Output CSV path")
    args = parser.parse_args()

    scored = rank_candidates(args.candidates)
    write_submission(scored, args.out)
    print(f"Wrote {len(scored)} ranked candidates to {args.out}")


if __name__ == "__main__":
    main()
