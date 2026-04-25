from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping

from .catalog import (
    CANDIDATE_FEATURE_SPECS,
    COMPANY_FEATURE_SPECS,
    INPUT_SCHEMA_VERSION,
    MARKET_DYNAMIC_SIGNAL_KEYS,
    MARKET_FEATURE_SPECS,
)


def _deterministic_value(index: int, base: float, span: float) -> float:
    bucket = ((index * 17 + 11) % 13) / 12
    return round(base + bucket * span, 6)


def _boost_domain(features: Dict[str, float], domain_prefix: str, value: float) -> None:
    for feature_name in features:
        if feature_name.startswith(f"{domain_prefix}_vector_"):
            features[feature_name] = value


def _default_candidate_profile() -> Dict[str, object]:
    return {
        "name": "Ziad Salah",
        "headline": "Independent Security Researcher | Vulnerability Researcher | Software Developer",
        "location": "Cairo, Egypt",
        "contact": {"email": "zs.01117875692@gmail.com", "phone": "+201117875692"},
        "links": {
            "github": "https://github.com/Zierax",
            "hackerone": "https://hackerone.com/0xzyo",
            "linkedin": "https://linkedin.com/in/z14d",
            "x": "https://x.com/Zierax_x",
            "orcid": "https://orcid.org/0009-0002-6813-2416",
        },
        "evidence_summary": [
            "Founder and lead researcher of Axiom Logic / Division-36.",
            "Built deterministic security and reasoning systems including Axiom-Qsecurity, Axiom-WAF, and Planck-99.",
            "Reported critical vulnerabilities on major VDP programs with high-severity impact.",
            "Published research accepted at IEEE AIITA 2026 and under Scientific Reports review.",
            "Maintains public benchmark repositories and technical writeups for reproducible validation.",
        ],
        "highlight_projects": [
            "Axiom-Math",
            "Axiom-Qsecurity",
            "Planck-99",
            "Axiom-Astrophysics",
            "Axiom-WAF",
            "Axiom-Vesuvius",
            "SYRTH",
            "Axiom-Zspace",
            "Axiom-LRM",
        ],
        "research_frameworks": [
            "PTRR Framework",
            "DVF Framework",
            "Truthimatics",
            "Axiom-Logic",
        ],
        "experience_highlights": [
            "Founder & Lead Researcher at Axiom Logic / Division-36.",
            "Vulnerability researcher on HackerOne VDP programs.",
            "Technical mentor for security research community.",
        ],
    }


def build_candidate_template() -> Dict[str, object]:
    features: Dict[str, float] = {}
    for index, feature_name in enumerate(CANDIDATE_FEATURE_SPECS.keys(), start=1):
        features[feature_name] = _deterministic_value(index=index, base=0.64, span=0.20)

    _boost_domain(features, "architecture", 0.94)
    _boost_domain(features, "delivery", 0.91)
    _boost_domain(features, "leadership", 0.90)
    _boost_domain(features, "ownership", 0.93)
    _boost_domain(features, "resilience", 0.92)
    _boost_domain(features, "communication", 0.87)
    _boost_domain(features, "security", 0.95)
    _boost_domain(features, "mentorship", 0.86)

    features.update(
        {
            "project_complexity": 0.97,
            "github_impact": 0.95,
            "code_latency_efficiency": 0.96,
            "code_memory_efficiency": 0.94,
            "algorithmic_depth": 0.97,
            "consistency_score": 0.90,
            "learning_velocity": 0.98,
            "execution_history": 0.93,
            "failure_recovery_capability": 0.92,
            "signal_to_noise_ratio": 0.91,
            "psychological_stability": 0.89,
            "innovation_index": 0.99,
            "real_world_deployment_impact": 0.96,
            "collaboration_reliability": 0.87,
            "ownership_depth": 0.95,
            "security_hardening": 0.98,
            "observability_rigor": 0.90,
            "documentation_quality": 0.84,
            "delivery_predictability": 0.92,
            "test_coverage_quality": 0.88,
            "incident_response_speed": 0.94,
            "technical_leadership": 0.93,
            "system_design_depth": 0.95,
            "domain_adaptability": 0.96,
            "ethical_judgment_score": 0.91,
            "resource_advantage_index": 0.50,
        }
    )
    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "candidate_id": "ziad-salah-default-profile",
        "default_candidate_profile": _default_candidate_profile(),
        "features": features,
    }


def build_company_template() -> Dict[str, object]:
    features: Dict[str, float] = {}
    for index, feature_name in enumerate(COMPANY_FEATURE_SPECS.keys(), start=1):
        features[feature_name] = _deterministic_value(index=index, base=0.38, span=0.36)
    features.update(
        {
            "hiring_latency": 0.72,
            "bureaucracy_level": 0.66,
            "engineering_depth": 0.77,
            "risk_tolerance": 0.63,
            "compensation_efficiency": 0.68,
            "internal_politics_factor": 0.54,
            "innovation_index": 0.73,
            "talent_retention": 0.69,
            "decision_making_entropy": 0.61,
            "technical_standards_strictness": 0.75,
            "growth_pressure": 0.81,
            "hiring_urgency": 0.84,
            "interview_loop_length": 0.70,
            "offer_approval_delay": 0.67,
            "managerial_bandwidth": 0.58,
            "recruiter_response_speed": 0.44,
            "candidate_experience_quality": 0.49,
        }
    )
    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "company_id": "company-sample-001",
        "features": features,
    }


def build_market_template() -> Dict[str, object]:
    features: Dict[str, float] = {}
    for index, feature_name in enumerate(MARKET_FEATURE_SPECS.keys(), start=1):
        features[feature_name] = _deterministic_value(index=index, base=0.45, span=0.33)
    features.update(
        {
            "demand_pressure": 0.86,
            "talent_scarcity": 0.82,
            "competition_density": 0.79,
            "economic_environment": 0.58,
            "time_to_hire_urgency": 0.88,
            "cost_of_delay": 0.84,
            "salary_inflation": 0.74,
            "candidate_mobility": 0.70,
        }
    )
    dynamic_signals = {key: features[key] for key in MARKET_DYNAMIC_SIGNAL_KEYS}
    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "market_id": "market-sample-2026-q2",
        "features": features,
        "dynamic_signals": dynamic_signals,
    }


def write_templates(target_directory: str | Path) -> Mapping[str, Path]:
    directory = Path(target_directory)
    directory.mkdir(parents=True, exist_ok=True)
    candidate_path = directory / "candidate.json"
    company_path = directory / "company.json"
    market_path = directory / "market.json"

    candidate_path.write_text(json.dumps(build_candidate_template(), indent=2, sort_keys=True), encoding="utf-8")
    company_path.write_text(json.dumps(build_company_template(), indent=2, sort_keys=True), encoding="utf-8")
    market_path.write_text(json.dumps(build_market_template(), indent=2, sort_keys=True), encoding="utf-8")

    return {
        "candidate": candidate_path,
        "company": company_path,
        "market": market_path,
    }
