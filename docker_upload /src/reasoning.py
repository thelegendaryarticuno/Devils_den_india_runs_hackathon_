from __future__ import annotations

from .features import CandidateFeatures


def build_reasoning(features: CandidateFeatures, rank: int) -> str:
    intro = _intro(features)
    evidence = _technical_evidence(features)
    logistics = _logistics_and_behavior(features)
    concern = _concern(features, rank)

    if rank <= 20:
        sentence1 = f"{intro}; {evidence}."
        sentence2 = f"{logistics}{concern}"
    elif rank <= 70:
        sentence1 = f"{intro} with relevant AI/search evidence: {evidence}."
        sentence2 = f"{logistics}{concern}"
    else:
        sentence1 = f"{intro}; included for adjacent fit because {evidence}."
        sentence2 = f"{logistics}{concern}"

    return " ".join([sentence1, sentence2]).replace("..", ".")


def _intro(features: CandidateFeatures) -> str:
    years = f"{features.years:.1f}".rstrip("0").rstrip(".")
    company = f" at {features.current_company}" if features.current_company else ""
    location = f" in {features.location}" if features.location else ""
    return f"{features.current_title}{company} with {years} years of experience{location}"


def _technical_evidence(features: CandidateFeatures) -> str:
    parts: list[str] = []
    if features.tech_hits.get("retrieval", 0):
        parts.append("retrieval/search")
    if features.tech_hits.get("ranking", 0):
        parts.append("ranking or recommendation systems")
    if features.tech_hits.get("vector_db", 0):
        parts.append("vector-search infrastructure")
    if features.tech_hits.get("eval", 0):
        parts.append("ranking evaluation")
    if features.tech_hits.get("llm", 0):
        parts.append("LLM/fine-tuning work")
    if features.production_hits.get("shipping", 0):
        parts.append("production shipping evidence")

    if not parts:
        parts.append("some applied ML/data-platform overlap")
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _logistics_and_behavior(features: CandidateFeatures) -> str:
    availability = []
    if features.open_to_work:
        availability.append("open to work")
    if features.notice_days <= 30:
        availability.append(f"{features.notice_days}-day notice")
    elif features.notice_days <= 60:
        availability.append(f"{features.notice_days}-day notice")
    if features.recruiter_response_rate >= 0.65:
        availability.append(f"{features.recruiter_response_rate:.0%} recruiter response rate")
    elif features.recruiter_response_rate >= 0.45:
        availability.append(f"{features.recruiter_response_rate:.0%} response rate")
    if features.saved_by_recruiters >= 20:
        availability.append(f"{features.saved_by_recruiters} recruiter saves in 30 days")
    if features.location_score >= 0.7:
        availability.append("location/relocation aligns with Pune-Noida hybrid needs")
    elif features.country.lower() == "india":
        availability.append("India-based")

    if not availability:
        availability.append("behavioral signals are acceptable but not standout")
    return "Signals: " + ", ".join(availability) + "."


def _concern(features: CandidateFeatures, rank: int) -> str:
    concerns: list[str] = []
    if features.notice_days >= 120:
        concerns.append(f"{features.notice_days}-day notice")
    if not features.open_to_work:
        concerns.append("not marked open to work")
    if features.country.lower() != "india" and features.location_score < 0.5:
        concerns.append("location may be harder")
    if features.service_only:
        concerns.append("services-heavy background")
    if features.years < 5:
        concerns.append("slightly below the preferred 5-9 year band")
    elif features.years > 9:
        concerns.append("above the preferred 5-9 year band")
    if features.anomaly_flags and rank > 40:
        concerns.append(features.anomaly_flags[0])

    if not concerns:
        return ""
    return " Concern: " + "; ".join(concerns[:2]) + "."
