from __future__ import annotations

import math
from dataclasses import dataclass

from .features import CandidateFeatures, clamp


@dataclass(slots=True)
class ScoredCandidate:
    candidate_id: str
    score: float
    features: CandidateFeatures


def score_candidate(features: CandidateFeatures) -> float:
    core = _core_technical_score(features) * 35.0
    role = _role_score(features) * 20.0
    shipping = _shipping_score(features) * 15.0
    logistics = features.location_score * 10.0
    behavior = features.behavior_score * 15.0
    validation = _validation_score(features) * 5.0

    score = core + role + shipping + logistics + behavior + validation
    score -= _penalties(features)
    return round(max(0.0, min(100.0, score)), 8)


def sort_candidates(candidates: list[ScoredCandidate]) -> list[ScoredCandidate]:
    return sorted(candidates, key=lambda item: (-item.score, item.candidate_id))


def final_csv_scores(scored: list[ScoredCandidate]) -> list[float]:
    """Return strictly descending display scores to avoid rounded tie violations."""
    scores: list[float] = []
    previous = 101.0
    for item in scored:
        display = round(item.score, 6)
        if display >= previous:
            display = previous - 0.000001
        display = max(0.0, display)
        scores.append(display)
        previous = display
    return scores


def _core_technical_score(features: CandidateFeatures) -> float:
    category_weights = {
        "retrieval": 0.22,
        "vector_db": 0.18,
        "ranking": 0.20,
        "eval": 0.18,
        "llm": 0.10,
        "python_ml": 0.12,
    }
    text_score = 0.0
    for category, weight in category_weights.items():
        hits = features.tech_hits.get(category, 0)
        text_score += weight * clamp(math.log1p(hits) / math.log1p(8))

    skill_score = clamp(features.trusted_skill_score / 35.0)
    evidence_bonus = 0.12 if _has_all_critical_evidence(features) else 0.0
    return clamp(0.72 * text_score + 0.28 * skill_score + evidence_bonus)


def _role_score(features: CandidateFeatures) -> float:
    if features.years <= 0:
        exp = 0.0
    elif 5.0 <= features.years <= 9.0:
        exp = 1.0
    elif 4.0 <= features.years < 5.0 or 9.0 < features.years <= 10.5:
        exp = 0.76
    elif 3.0 <= features.years < 4.0 or 10.5 < features.years <= 13.0:
        exp = 0.42
    else:
        exp = 0.18

    title = 1.0 if features.target_title else 0.2 if not features.bad_title else 0.0
    product = 1.0 if features.product_experience else 0.45
    senior_hands_on = 0.35
    title_lower = features.current_title.lower()
    if any(token in title_lower for token in ("senior", "lead", "staff")) and features.target_title:
        senior_hands_on = 1.0
    elif features.target_title:
        senior_hands_on = 0.75
    if features.manager_only:
        senior_hands_on *= 0.35

    return clamp(0.32 * exp + 0.34 * title + 0.20 * product + 0.14 * senior_hands_on)


def _shipping_score(features: CandidateFeatures) -> float:
    weights = {
        "shipping": 0.35,
        "scale": 0.22,
        "product": 0.20,
        "eval": 0.23,
    }
    score = 0.0
    for category, weight in weights.items():
        score += weight * clamp(math.log1p(features.production_hits.get(category, 0)) / math.log1p(8))
    if features.pure_research_hint and features.production_hits.get("shipping", 0) == 0:
        score *= 0.55
    return clamp(score)


def _validation_score(features: CandidateFeatures) -> float:
    return clamp(
        0.35 * features.education_score
        + 0.25 * features.github_score
        + 0.20 * clamp(features.saved_by_recruiters / 50.0)
        + 0.20 * clamp(features.interview_completion_rate)
    )


def _penalties(features: CandidateFeatures) -> float:
    penalty = 0.0
    production_evidence = sum(features.production_hits.values())
    tech_evidence = sum(features.tech_hits.values())

    if features.bad_title and features.ai_skill_count >= 5:
        penalty += 22.0
    elif features.bad_title:
        penalty += 12.0

    if features.service_only and tech_evidence < 10:
        penalty += 14.0
    elif features.service_only:
        penalty += 6.0

    if features.ai_skill_count >= 8 and production_evidence < 3:
        penalty += 15.0

    if features.manager_only and not features.target_title:
        penalty += 10.0

    if features.notice_days >= 150:
        penalty += 8.0
    elif features.notice_days >= 120:
        penalty += 5.0

    if not features.open_to_work:
        penalty += 5.0

    if features.last_active_days > 180:
        penalty += 6.0
    elif features.last_active_days > 90:
        penalty += 3.0

    if features.recruiter_response_rate < 0.20:
        penalty += 4.0

    if features.pure_research_hint and production_evidence < 3:
        penalty += 8.0

    for flag in features.anomaly_flags:
        if flag == "keyword-heavy non-AI current title":
            penalty += 9.0
        elif flag == "AI skills without production evidence":
            penalty += 8.0
        elif flag == "services-only career with weak ML evidence":
            penalty += 6.0
        elif flag == "too junior for claimed AI depth":
            penalty += 10.0
        elif flag == "expert skill with near-zero duration":
            penalty += 15.0
        elif flag == "poor availability":
            penalty += 5.0
        elif flag == "stale and low response":
            penalty += 7.0
        elif flag == "AI-curious rather than AI builder":
            penalty += 12.0
        elif flag == "experience-duration mismatch":
            penalty += 6.0

    if _has_all_critical_evidence(features) and features.target_title and features.years >= 4:
        penalty *= 0.55

    return penalty


def _has_all_critical_evidence(features: CandidateFeatures) -> bool:
    return (
        features.tech_hits.get("retrieval", 0) > 0
        and features.tech_hits.get("vector_db", 0) > 0
        and features.tech_hits.get("ranking", 0) > 0
        and features.tech_hits.get("eval", 0) > 0
        and features.production_hits.get("shipping", 0) > 0
    )
