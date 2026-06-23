# Redrob Hackathon Execution Plan

## Goal

Build a deterministic, CPU-only candidate ranker that reads the Redrob candidate pool and emits a valid top-100 CSV for the Senior AI Engineer founding-team JD. The implementation should favor candidates with real production search, retrieval, ranking, and applied ML experience over keyword-heavy profiles.

## Ranking Strategy

- Score structured evidence from the full profile, career history, skills, education, and Redrob behavioral signals.
- Prioritize production experience with embeddings, hybrid retrieval, vector search, recommendation systems, learning-to-rank, evaluation metrics, and A/B testing.
- Reward senior hands-on AI/ML/search titles, 5-9 years of experience, product-company experience, India/Pune/Noida logistics, open-to-work status, fast response behavior, and low notice periods.
- Penalize keyword stuffing, non-technical current titles, services-only careers, stale profiles, very long notice periods, pure research-only profiles, and impossible-looking skill claims.
- Generate short, fact-grounded reasoning for each selected candidate using only information present in the candidate record.

## Deliverables

- `rank.py`: command-line entrypoint.
- `src/features.py`: feature extraction and signal normalization.
- `src/scoring.py`: weighted scoring and final sorting.
- `src/reasoning.py`: grounded reasoning generation.
- `src/validate_ids.py`: local ID/output validation helpers.
- `submission_metadata.yaml`: metadata template to fill before upload.
- `README.md`: setup, reproduction, and validation commands.

## Validation

Run:

```bash
python rank.py --candidates ./data_redrob_hackathon/candidates.jsonl --out ./submission.csv
python src/validate_ids.py --candidates ./data_redrob_hackathon/candidates.jsonl --submission ./submission.csv
python data_redrob_hackathon/validate_submission.py submission.csv
```

The full ranking step must complete within 5 minutes on CPU, use no network access, and produce exactly 100 ranked candidates.
