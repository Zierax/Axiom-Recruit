from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

INPUT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    weight: float
    min_value: float = 0.0
    max_value: float = 1.0
    description: str = ""


def _build_core_specs(core_weights: Mapping[str, float], description_prefix: str) -> Dict[str, FeatureSpec]:
    return {
        name: FeatureSpec(
            name=name,
            weight=round(weight, 4),
            min_value=0.0,
            max_value=1.0,
            description=f"{description_prefix}::{name}",
        )
        for name, weight in core_weights.items()
    }


def _build_vector_specs(
    domains: Iterable[str],
    vectors_per_domain: int,
    base_weight: float,
    jitter: float,
    description_prefix: str,
) -> Dict[str, FeatureSpec]:
    specs: Dict[str, FeatureSpec] = {}
    ordinal = 0
    for domain_index, domain in enumerate(domains, start=1):
        for vector_index in range(1, vectors_per_domain + 1):
            ordinal += 1
            name = f"{domain}_vector_{vector_index:02d}"
            variance_bucket = ((domain_index * 31 + vector_index * 17 + ordinal) % 7) - 3
            weight = round(base_weight + variance_bucket * jitter, 4)
            specs[name] = FeatureSpec(
                name=name,
                weight=max(0.05, weight),
                min_value=0.0,
                max_value=1.0,
                description=f"{description_prefix}::{domain}",
            )
    return specs


CANDIDATE_CORE_WEIGHTS: Dict[str, float] = {
    "project_complexity": 1.42,
    "github_impact": 1.16,
    "code_latency_efficiency": 1.28,
    "code_memory_efficiency": 1.18,
    "algorithmic_depth": 1.34,
    "consistency_score": 1.39,
    "learning_velocity": 1.27,
    "execution_history": 1.33,
    "failure_recovery_capability": 1.24,
    "signal_to_noise_ratio": 1.14,
    "psychological_stability": 1.31,
    "innovation_index": 1.29,
    "real_world_deployment_impact": 1.41,
    "collaboration_reliability": 1.19,
    "ownership_depth": 1.21,
    "security_hardening": 1.25,
    "observability_rigor": 1.12,
    "documentation_quality": 0.94,
    "delivery_predictability": 1.22,
    "test_coverage_quality": 1.08,
    "incident_response_speed": 1.15,
    "technical_leadership": 1.2,
    "system_design_depth": 1.26,
    "domain_adaptability": 1.1,
    "ethical_judgment_score": 1.06,
    "resource_advantage_index": 1.45,
}

CANDIDATE_VECTOR_DOMAINS = (
    "architecture",
    "delivery",
    "leadership",
    "ownership",
    "collaboration",
    "product",
    "quality",
    "resilience",
    "communication",
    "mentorship",
    "security",
)

CANDIDATE_FEATURE_SPECS: Dict[str, FeatureSpec] = {
    **_build_core_specs(CANDIDATE_CORE_WEIGHTS, "candidate-core"),
    **_build_vector_specs(
        domains=CANDIDATE_VECTOR_DOMAINS,
        vectors_per_domain=9,
        base_weight=0.62,
        jitter=0.03,
        description_prefix="candidate-vector",
    ),
}


COMPANY_CORE_WEIGHTS: Dict[str, float] = {
    "hiring_latency": 1.45,
    "bureaucracy_level": 1.28,
    "engineering_depth": 1.41,
    "risk_tolerance": 1.1,
    "compensation_efficiency": 1.25,
    "internal_politics_factor": 1.2,
    "innovation_index": 1.32,
    "talent_retention": 1.29,
    "decision_making_entropy": 1.34,
    "technical_standards_strictness": 1.3,
    "growth_pressure": 1.22,
    "hiring_urgency": 1.37,
    "interview_loop_length": 1.17,
    "offer_approval_delay": 1.24,
    "managerial_bandwidth": 1.11,
    "roadmap_clarity": 1.07,
    "infra_maturity": 1.2,
    "delivery_reliability": 1.23,
    "compliance_rigor": 1.08,
    "onboarding_quality": 1.06,
    "role_clarity": 1.15,
    "performance_management_quality": 1.04,
    "knowledge_sharing_index": 0.98,
    "cross_team_alignment": 1.14,
    "technical_debt_pressure": 1.27,
    "budget_flexibility": 1.13,
    "team_stability": 1.18,
    "leadership_accountability": 1.19,
    "recruiter_response_speed": 1.26,
    "candidate_experience_quality": 1.16,
}

COMPANY_VECTOR_DOMAINS = (
    "process",
    "governance",
    "architecture",
    "delivery",
    "people",
    "finance",
    "security",
    "data",
    "quality",
    "innovation",
    "operations",
    "compliance",
    "planning",
    "recruiting",
    "retention",
    "leadership",
    "market",
    "product",
    "execution",
)

COMPANY_FEATURE_SPECS: Dict[str, FeatureSpec] = {
    **_build_core_specs(COMPANY_CORE_WEIGHTS, "company-core"),
    **_build_vector_specs(
        domains=COMPANY_VECTOR_DOMAINS,
        vectors_per_domain=10,
        base_weight=0.56,
        jitter=0.025,
        description_prefix="company-vector",
    ),
}


MARKET_CORE_WEIGHTS: Dict[str, float] = {
    "demand_pressure": 1.4,
    "talent_scarcity": 1.35,
    "competition_density": 1.2,
    "economic_environment": 1.18,
    "time_to_hire_urgency": 1.38,
    "salary_inflation": 1.09,
    "remote_talent_access": 1.02,
    "layoff_pressure": 0.97,
    "venture_funding_temperature": 1.01,
    "regulatory_pressure": 0.95,
    "cost_of_delay": 1.3,
    "skill_obsolescence_speed": 1.16,
    "outsourcing_pressure": 0.92,
    "geo_mobility_index": 1.0,
    "sector_volatility": 1.11,
    "ai_adoption_velocity": 1.21,
    "open_role_growth": 1.14,
    "candidate_mobility": 1.06,
    "offer_decline_rate": 1.08,
    "passive_talent_activity": 0.99,
}

MARKET_VECTOR_DOMAINS = ("macro", "labor", "salary", "technology", "regulation")

MARKET_FEATURE_SPECS: Dict[str, FeatureSpec] = {
    **_build_core_specs(MARKET_CORE_WEIGHTS, "market-core"),
    **_build_vector_specs(
        domains=MARKET_VECTOR_DOMAINS,
        vectors_per_domain=2,
        base_weight=0.72,
        jitter=0.035,
        description_prefix="market-vector",
    ),
}


MARKET_DYNAMIC_SIGNAL_KEYS = (
    "demand_pressure",
    "talent_scarcity",
    "competition_density",
    "economic_environment",
    "time_to_hire_urgency",
    "cost_of_delay",
)

CANDIDATE_WEIGHTS = {name: spec.weight for name, spec in CANDIDATE_FEATURE_SPECS.items()}
COMPANY_WEIGHTS = {name: spec.weight for name, spec in COMPANY_FEATURE_SPECS.items()}
MARKET_WEIGHTS = {name: spec.weight for name, spec in MARKET_FEATURE_SPECS.items()}

FEATURE_COUNT_SUMMARY = {
    "candidate": len(CANDIDATE_FEATURE_SPECS),
    "company": len(COMPANY_FEATURE_SPECS),
    "market": len(MARKET_FEATURE_SPECS),
}

assert FEATURE_COUNT_SUMMARY["candidate"] >= 100
assert FEATURE_COUNT_SUMMARY["company"] >= 200
