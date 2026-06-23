# Redrob Hackathon Ranker

Deterministic CPU-only ranker for the Redrob Intelligent Candidate Discovery & Ranking Challenge.

The ranker uses structured feature engineering over candidate profile text, career history, skills, education, and Redrob behavioral signals. It is designed to prefer real production search/retrieval/ranking experience over keyword-stuffed profiles.

## Reproduce Submission

```bash
python rank.py --candidates ./data_redrob_hackathon/candidates.jsonl --out ./submission.csv
```

## Validate

```bash
python src/validate_ids.py --candidates ./data_redrob_hackathon/candidates.jsonl --submission ./submission.csv
python data_redrob_hackathon/validate_submission.py submission.csv
```

## Docker Sandbox

```bash
docker run --rm -p 7860:7860 namanmani/natanyx-redrob-ranker:latest
```

Open `http://localhost:7860`.

## Approach

- Core fit: production embeddings, retrieval, vector search, recommendation systems, learning-to-rank, and ranking evaluation.
- Career fit: senior hands-on AI/ML/search titles, 5-9 years of experience, product-company history.
- Shipping fit: deployment, scale, latency, user/recruiter workflows, A/B testing, feedback loops.
- Logistics: India/Pune/Noida fit, relocation, hybrid/flexible preferences, notice period.
- Redrob behavior: recent activity, open-to-work status, response rate, recruiter saves, verification, interview completion.
- Guardrails: penalties for non-AI titles with AI keyword stuffing, services-only careers, stale profiles, poor availability, and impossible-looking skill claims.

The ranking step uses no network access, no GPU, and no hosted LLM calls.
