from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any


REFERENCE_DATE = date(2026, 6, 23)

SERVICE_COMPANIES = {
    "accenture",
    "capgemini",
    "cognizant",
    "hcl",
    "infosys",
    "mindtree",
    "mphasis",
    "tcs",
    "tech mahindra",
    "wipro",
}

TARGET_TITLE_RE = re.compile(
    r"\b("
    r"senior ai engineer|lead ai engineer|ai engineer|machine learning engineer|"
    r"senior machine learning engineer|senior ml engineer|ml engineer|"
    r"search engineer|recommendation systems engineer|recommender systems engineer|"
    r"ranking engineer|nlp engineer|senior nlp engineer|applied scientist|"
    r"senior applied scientist|data scientist|senior data scientist"
    r")\b",
    re.I,
)

BAD_TITLE_RE = re.compile(
    r"\b("
    r"hr manager|human resources|marketing manager|sales executive|accountant|"
    r"graphic designer|content writer|operations manager|customer support|"
    r"civil engineer|mechanical engineer|project manager|business analyst"
    r")\b",
    re.I,
)

MANAGER_ONLY_RE = re.compile(r"\b(manager|director|head of|vp|chief)\b", re.I)

CORE_TECH_PATTERNS = {
    "retrieval": re.compile(r"\b(retrieval|semantic search|hybrid search|vector search|dense retrieval|bm25|rag)\b", re.I),
    "vector_db": re.compile(r"\b(faiss|pinecone|qdrant|milvus|weaviate|pgvector|elasticsearch|opensearch|vector database)\b", re.I),
    "ranking": re.compile(r"\b(ranking|learning[- ]to[- ]rank|ltr|recommendation systems?|recommender systems?|xgboost|lightgbm)\b", re.I),
    "eval": re.compile(r"\b(ndcg|mrr|map|recall@k|precision@k|offline evaluation|online evaluation|a/b|ab test|relevance judgment)\b", re.I),
    "llm": re.compile(r"\b(llm|large language model|fine[- ]?tuning|lora|qlora|peft|bge|e5|sentence[- ]?transformers?|embeddings?)\b", re.I),
    "python_ml": re.compile(r"\b(python|pytorch|tensorflow|scikit[- ]?learn|mlflow|kubeflow|bentoml|mlops)\b", re.I),
}

PRODUCTION_PATTERNS = {
    "shipping": re.compile(r"\b(production|deployed|shipped|served|serving|launched|rolled out|owned|operated)\b", re.I),
    "scale": re.compile(r"\b(\d+\s*(m|million|k|thousand)|p95|latency|queries|users|items|documents|scale)\b", re.I),
    "product": re.compile(r"\b(product|platform|marketplace|recruiter|candidate|customer|pm|feedback loop|metrics|engagement|conversion)\b", re.I),
    "eval": CORE_TECH_PATTERNS["eval"],
}

AI_SKILL_RE = re.compile(
    r"\b(ai|ml|machine learning|deep learning|nlp|retrieval|ranking|recommendation|"
    r"recommender|search|embedding|vector|llm|rag|fine[- ]?tuning|lora|qlora|"
    r"faiss|qdrant|milvus|weaviate|pinecone|elasticsearch|opensearch|xgboost|"
    r"pytorch|tensorflow|semantic search|information retrieval|learning to rank)\b",
    re.I,
)

PURE_RESEARCH_RE = re.compile(r"\b(research scientist|phd|academic lab|published papers?|theoretical|research-only)\b", re.I)

PRODUCT_INDUSTRIES = {
    "ai/ml",
    "conversational ai",
    "e-commerce",
    "edtech",
    "fintech",
    "food delivery",
    "healthtech",
    "healthtech ai",
    "insurance tech",
    "marketplace",
    "saas",
    "software",
    "transportation",
}

PREFERRED_CITIES = (
    "pune",
    "noida",
    "delhi",
    "gurgaon",
    "gurugram",
    "mumbai",
    "hyderabad",
    "bangalore",
    "bengaluru",
)

PROFICIENCY_WEIGHT = {
    "beginner": 0.25,
    "intermediate": 0.55,
    "advanced": 0.85,
    "expert": 1.0,
}


@dataclass(slots=True)
class CandidateFeatures:
    candidate_id: str
    current_title: str
    current_company: str
    location: str
    country: str
    years: float
    headline: str
    summary: str
    career_text: str
    all_text: str
    target_title: bool
    bad_title: bool
    manager_only: bool
    ai_skill_count: int
    strong_ai_skill_count: int
    trusted_skill_score: float
    tech_hits: dict[str, int] = field(default_factory=dict)
    production_hits: dict[str, int] = field(default_factory=dict)
    product_experience: bool = False
    service_only: bool = False
    pure_research_hint: bool = False
    location_score: float = 0.0
    behavior_score: float = 0.0
    education_score: float = 0.0
    github_score: float = 0.0
    notice_days: int = 180
    open_to_work: bool = False
    recruiter_response_rate: float = 0.0
    avg_response_hours: float = 999.0
    last_active_days: int = 999
    saved_by_recruiters: int = 0
    interview_completion_rate: float = 0.0
    offer_acceptance_rate: float = -1.0
    anomaly_flags: list[str] = field(default_factory=list)
    evidence: dict[str, str] = field(default_factory=dict)


def extract_features(candidate: dict[str, Any]) -> CandidateFeatures:
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])

    career_text = " ".join(
        " ".join(
            str(job.get(key, ""))
            for key in ("title", "company", "industry", "description")
        )
        for job in career
    )
    all_text = " ".join(
        [
            profile.get("headline", ""),
            profile.get("summary", ""),
            profile.get("current_title", ""),
            profile.get("current_industry", ""),
            career_text,
        ]
    )
    all_text_lower = all_text.lower()

    ai_skills = [skill for skill in skills if AI_SKILL_RE.search(skill.get("name", ""))]
    strong_ai_skills = [
        skill
        for skill in ai_skills
        if skill.get("proficiency") in {"advanced", "expert"} and _number(skill.get("duration_months"), 0) >= 18
    ]

    trusted_skill_score = 0.0
    for skill in skills:
        name = skill.get("name", "")
        if not AI_SKILL_RE.search(name):
            continue
        prof = PROFICIENCY_WEIGHT.get(skill.get("proficiency", ""), 0.0)
        duration = min(float(_number(skill.get("duration_months"), 0)), 96.0)
        endorsements = min(float(_number(skill.get("endorsements"), 0)), 80.0)
        trusted_skill_score += prof * (0.55 * math.log1p(duration) + 0.45 * math.log1p(endorsements))

    tech_hits = {name: _count_matches(pattern, all_text) for name, pattern in CORE_TECH_PATTERNS.items()}
    production_hits = {name: _count_matches(pattern, all_text) for name, pattern in PRODUCTION_PATTERNS.items()}

    product_experience = any(
        str(job.get("industry", "")).lower() in PRODUCT_INDUSTRIES
        for job in career
    ) or str(profile.get("current_industry", "")).lower() in PRODUCT_INDUSTRIES

    service_only = bool(career) and all(
        str(job.get("company", "")).lower() in SERVICE_COMPANIES
        or str(job.get("industry", "")).lower() in {"it services", "consulting"}
        for job in career
    )

    target_title = bool(TARGET_TITLE_RE.search(profile.get("current_title", "")))
    bad_title = bool(BAD_TITLE_RE.search(profile.get("current_title", "")))
    manager_only = bool(MANAGER_ONLY_RE.search(profile.get("current_title", ""))) and not target_title

    location_score = _location_score(profile, signals)
    behavior_score = _behavior_score(signals)
    education_score = _education_score(education, signals)
    github_score = _github_score(signals)
    notice_days = int(_number(signals.get("notice_period_days"), 180))
    last_active_days = _days_since(signals.get("last_active_date"))

    anomaly_flags = _anomaly_flags(
        profile=profile,
        career=career,
        skills=skills,
        signals=signals,
        tech_hits=tech_hits,
        production_hits=production_hits,
        ai_skill_count=len(ai_skills),
        strong_ai_skill_count=len(strong_ai_skills),
        all_text_lower=all_text_lower,
        bad_title=bad_title,
        service_only=service_only,
    )

    return CandidateFeatures(
        candidate_id=candidate["candidate_id"],
        current_title=profile.get("current_title", ""),
        current_company=profile.get("current_company", ""),
        location=profile.get("location", ""),
        country=profile.get("country", ""),
        years=float(_number(profile.get("years_of_experience"), 0.0)),
        headline=profile.get("headline", ""),
        summary=profile.get("summary", ""),
        career_text=career_text,
        all_text=all_text,
        target_title=target_title,
        bad_title=bad_title,
        manager_only=manager_only,
        ai_skill_count=len(ai_skills),
        strong_ai_skill_count=len(strong_ai_skills),
        trusted_skill_score=trusted_skill_score,
        tech_hits=tech_hits,
        production_hits=production_hits,
        product_experience=product_experience,
        service_only=service_only,
        pure_research_hint=bool(PURE_RESEARCH_RE.search(all_text)),
        location_score=location_score,
        behavior_score=behavior_score,
        education_score=education_score,
        github_score=github_score,
        notice_days=notice_days,
        open_to_work=bool(signals.get("open_to_work_flag")),
        recruiter_response_rate=float(_number(signals.get("recruiter_response_rate"), 0.0)),
        avg_response_hours=float(_number(signals.get("avg_response_time_hours"), 999.0)),
        last_active_days=last_active_days,
        saved_by_recruiters=int(_number(signals.get("saved_by_recruiters_30d"), 0)),
        interview_completion_rate=float(_number(signals.get("interview_completion_rate"), 0.0)),
        offer_acceptance_rate=float(_number(signals.get("offer_acceptance_rate"), -1.0)),
        anomaly_flags=anomaly_flags,
        evidence={},
    )


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _number(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _count_matches(pattern: re.Pattern[str], text: str, cap: int = 8) -> int:
    count = 0
    for _ in pattern.finditer(text):
        count += 1
        if count >= cap:
            return cap
    return count


def _days_since(date_text: str | None) -> int:
    if not date_text:
        return 999
    try:
        parsed = date.fromisoformat(date_text)
    except ValueError:
        return 999
    return max(0, (REFERENCE_DATE - parsed).days)


def _location_score(profile: dict[str, Any], signals: dict[str, Any]) -> float:
    location = str(profile.get("location", "")).lower()
    country = str(profile.get("country", "")).lower()
    willing = bool(signals.get("willing_to_relocate"))
    work_mode = str(signals.get("preferred_work_mode", "")).lower()

    score = 0.0
    if country == "india":
        score += 0.45
    if any(city in location for city in ("pune", "noida")):
        score += 0.35
    elif any(city in location for city in PREFERRED_CITIES):
        score += 0.25
    if willing:
        score += 0.20
    if work_mode in {"hybrid", "flexible", "onsite"}:
        score += 0.10
    if country != "india" and not willing:
        score -= 0.25
    return clamp(score)


def _behavior_score(signals: dict[str, Any]) -> float:
    completeness = clamp(float(_number(signals.get("profile_completeness_score"), 0.0)) / 100.0)
    response = clamp(float(_number(signals.get("recruiter_response_rate"), 0.0)))
    response_hours = float(_number(signals.get("avg_response_time_hours"), 999.0))
    response_speed = clamp(1.0 - (response_hours / 168.0))
    open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.35
    active = clamp(1.0 - (_days_since(signals.get("last_active_date")) / 180.0))
    notice = int(_number(signals.get("notice_period_days"), 180))
    notice_score = 1.0 if notice <= 30 else 0.8 if notice <= 60 else 0.55 if notice <= 90 else 0.2
    verified = (
        0.34 * bool(signals.get("verified_email"))
        + 0.33 * bool(signals.get("verified_phone"))
        + 0.33 * bool(signals.get("linkedin_connected"))
    )
    recruiter_pull = clamp(
        0.5 * math.log1p(float(_number(signals.get("saved_by_recruiters_30d"), 0))) / math.log1p(60)
        + 0.5 * math.log1p(float(_number(signals.get("profile_views_received_30d"), 0))) / math.log1p(120)
    )
    interview = clamp(float(_number(signals.get("interview_completion_rate"), 0.0)))
    offer = float(signals.get("offer_acceptance_rate") if signals.get("offer_acceptance_rate") is not None else -1)
    offer_score = 0.5 if offer < 0 else clamp(offer)

    return clamp(
        0.14 * completeness
        + 0.20 * response
        + 0.09 * response_speed
        + 0.16 * open_to_work
        + 0.13 * active
        + 0.12 * notice_score
        + 0.05 * verified
        + 0.05 * recruiter_pull
        + 0.04 * interview
        + 0.02 * offer_score
    )


def _education_score(education: list[dict[str, Any]], signals: dict[str, Any]) -> float:
    tier_score = 0.0
    field_score = 0.0
    for degree in education:
        tier = degree.get("tier")
        if tier == "tier_1":
            tier_score = max(tier_score, 1.0)
        elif tier == "tier_2":
            tier_score = max(tier_score, 0.75)
        elif tier == "tier_3":
            tier_score = max(tier_score, 0.45)
        field = f"{degree.get('degree', '')} {degree.get('field_of_study', '')}"
        if re.search(r"\b(computer science|machine learning|artificial intelligence|information technology|data science|mathematics|statistics)\b", field, re.I):
            field_score = max(field_score, 1.0)
    assessments = signals.get("skill_assessment_scores") or {}
    relevant_assessments = [
        float(score)
        for name, score in assessments.items()
        if AI_SKILL_RE.search(str(name)) and isinstance(score, (int, float))
    ]
    assessment_score = (sum(relevant_assessments) / len(relevant_assessments) / 100.0) if relevant_assessments else 0.0
    return clamp(0.45 * tier_score + 0.35 * field_score + 0.20 * assessment_score)


def _github_score(signals: dict[str, Any]) -> float:
    github = float(_number(signals.get("github_activity_score"), -1.0))
    if github < 0:
        return 0.0
    return clamp(github / 100.0)


def _anomaly_flags(
    *,
    profile: dict[str, Any],
    career: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    signals: dict[str, Any],
    tech_hits: dict[str, int],
    production_hits: dict[str, int],
    ai_skill_count: int,
    strong_ai_skill_count: int,
    all_text_lower: str,
    bad_title: bool,
    service_only: bool,
) -> list[str]:
    flags: list[str] = []
    years = float(_number(profile.get("years_of_experience"), 0.0))
    production_evidence = sum(production_hits.values())
    tech_evidence = sum(tech_hits.values())

    if bad_title and ai_skill_count >= 5:
        flags.append("keyword-heavy non-AI current title")
    if ai_skill_count >= 8 and production_evidence < 2:
        flags.append("AI skills without production evidence")
    if service_only and tech_evidence < 8:
        flags.append("services-only career with weak ML evidence")
    if years < 3.0 and strong_ai_skill_count >= 5:
        flags.append("too junior for claimed AI depth")
    if any(skill.get("proficiency") == "expert" and int(_number(skill.get("duration_months"), 0)) <= 3 for skill in skills):
        flags.append("expert skill with near-zero duration")
    if int(_number(signals.get("notice_period_days"), 180)) >= 120 and not signals.get("open_to_work_flag"):
        flags.append("poor availability")
    if _days_since(signals.get("last_active_date")) > 180 and float(_number(signals.get("recruiter_response_rate"), 0.0)) < 0.25:
        flags.append("stale and low response")
    if "curious about how ai tools" in all_text_lower and tech_evidence < 5:
        flags.append("AI-curious rather than AI builder")
    if career:
        career_months = sum(int(_number(job.get("duration_months"), 0)) for job in career)
        if career_months and abs((career_months / 12.0) - years) > 6.0:
            flags.append("experience-duration mismatch")
    return flags


def _evidence(candidate: dict[str, Any], all_text: str) -> dict[str, str]:
    evidence: dict[str, str] = {}
    for key, pattern in {**CORE_TECH_PATTERNS, **PRODUCTION_PATTERNS}.items():
        match = pattern.search(all_text)
        if match:
            evidence[key] = match.group(0)

    career = candidate.get("career_history", [])
    for job in career:
        description = str(job.get("description", ""))
        if any(pattern.search(description) for pattern in CORE_TECH_PATTERNS.values()):
            evidence["career_role"] = f"{job.get('title', '')} at {job.get('company', '')}".strip()
            evidence["career_snippet"] = _shorten(description)
            break
    return evidence


def _shorten(text: str, limit: int = 170) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rsplit(" ", 1)[0] + "."
