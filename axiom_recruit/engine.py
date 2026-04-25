from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Tuple

from .catalog import CANDIDATE_WEIGHTS, COMPANY_WEIGHTS, MARKET_WEIGHTS
from .whitebox_nn import evaluate_whitebox_edge_network


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


@dataclass(frozen=True)
class EvaluationOptions:
    explain_level: str = "summary"
    company_exposure_mode: bool = False
    self_evaluation_mode: bool = False


def _effective_weight(base_weight: float, adjustment: float) -> float:
    adjusted = base_weight + adjustment
    if adjusted < 0.01:
        return 0.01
    return adjusted


def _weighted_score(
    feature_values: Mapping[str, float],
    base_weights: Mapping[str, float],
    adjustments: Mapping[str, float],
    domain_name: str,
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    weighted_sum = 0.0
    total_weight = 0.0
    contribution_map: Dict[str, Dict[str, float]] = {}
    for feature_name, value in feature_values.items():
        weight = _effective_weight(base_weights[feature_name], adjustments.get(feature_name, 0.0))
        weighted_sum += weight * value
        total_weight += weight
        contribution_map[f"{domain_name}.{feature_name}"] = {
            "domain": domain_name,
            "feature": feature_name,
            "value": value,
            "weight": weight,
            "raw_contribution": weight * value,
        }
    score = weighted_sum / total_weight if total_weight else 0.0
    for entry in contribution_map.values():
        entry["normalized_contribution"] = entry["raw_contribution"] / total_weight if total_weight else 0.0
    return _clamp(score), contribution_map


def _neutral_company_features() -> Dict[str, float]:
    # Audit 3.6: Neutral features should be supportive for self-evaluation, not mediocre.
    neutral = {feature_name: 0.5 for feature_name in COMPANY_WEIGHTS}
    # Boost features that act as "good" environments for candidates
    neutral["compensation_efficiency"] = 0.7
    neutral["innovation_index"] = 0.7
    neutral["engineering_depth"] = 0.7
    neutral["hiring_latency"] = 0.3  # Low latency is better
    neutral["bureaucracy_level"] = 0.3  # Low bureaucracy is better
    return neutral


def _candidate_company_fit(candidate_features: Mapping[str, float], company_features: Mapping[str, float]) -> float:
    pairings = [
        ("project_complexity", "engineering_depth"),
        ("innovation_index", "innovation_index"),
        ("delivery_predictability", "delivery_reliability"),
        ("collaboration_reliability", "cross_team_alignment"),
        ("technical_leadership", "leadership_accountability"),
        ("signal_to_noise_ratio", "role_clarity"),
        ("security_hardening", "compliance_rigor"),
        ("ownership_depth", "managerial_bandwidth"),
        ("real_world_deployment_impact", "infra_maturity"),
    ]
    distances = [abs(candidate_features[left] - company_features[right]) for left, right in pairings]
    return _clamp(1.0 - (sum(distances) / len(distances)))


def _candidate_market_alignment(candidate_features: Mapping[str, float], market_features: Mapping[str, float]) -> float:
    candidate_supply_strength = _clamp(
        0.28 * candidate_features["project_complexity"]
        + 0.25 * candidate_features["algorithmic_depth"]
        + 0.20 * candidate_features["learning_velocity"]
        + 0.27 * candidate_features["real_world_deployment_impact"]
    )
    market_demand_strength = _clamp(
        0.24 * market_features["demand_pressure"]
        + 0.20 * market_features["talent_scarcity"]
        + 0.18 * market_features["competition_density"]
        + 0.20 * market_features["time_to_hire_urgency"]
        + 0.18 * market_features["cost_of_delay"]
    )
    return _clamp(1.0 - abs(candidate_supply_strength - market_demand_strength))


def _company_market_mismatch(company_features: Mapping[str, float], market_features: Mapping[str, float]) -> float:
    mismatch_terms = [
        abs(company_features["hiring_urgency"] - market_features["time_to_hire_urgency"]),
        abs(company_features["compensation_efficiency"] - market_features["salary_inflation"]),
        abs(company_features["hiring_latency"] - (1.0 - market_features["time_to_hire_urgency"])),
        abs(company_features["talent_retention"] - (1.0 - market_features["candidate_mobility"])),
        abs(company_features["risk_tolerance"] - market_features["sector_volatility"]),
    ]
    return _clamp(sum(mismatch_terms) / len(mismatch_terms))


def _candidate_excellence_index(candidate_features: Mapping[str, float], candidate_score: float) -> float:
    # Audit: Achievement vs Resources ratio.
    # High achievement with low resource access (e.g. 17yo in Egypt vs Harvard) = Higher potential.
    resource_advantage = candidate_features.get("resource_advantage_index", 0.1)
    potential_boost = (1.0 - resource_advantage) * 0.35
    return _clamp(
        0.45 * candidate_score
        + 0.20 * candidate_features["innovation_index"]
        + 0.15 * candidate_features["real_world_deployment_impact"]
        + 0.15 * candidate_features["system_design_depth"]
        + potential_boost
    )


def _company_readiness_index(company_features: Mapping[str, float], company_score: float) -> float:
    return _clamp(
        0.55 * company_score
        + 0.20 * (1.0 - company_features["hiring_latency"])
        + 0.15 * company_features["recruiter_response_speed"]
        + 0.10 * (1.0 - company_features["bureaucracy_level"])
    )


def _market_heat_index(market_features: Mapping[str, float]) -> float:
    return _clamp(
        0.35 * market_features["demand_pressure"]
        + 0.30 * market_features["time_to_hire_urgency"]
        + 0.20 * market_features["talent_scarcity"]
        + 0.15 * market_features["cost_of_delay"]
    )


def _acquisition_pressure_index(
    opportunity_loss_risk: float,
    candidate_dominance_index: float,
    market_heat_index: float,
) -> float:
    return _clamp(
        0.45 * opportunity_loss_risk
        + 0.30 * candidate_dominance_index
        + 0.25 * market_heat_index
    )


def _build_hard_constraints(
    candidate_features: Mapping[str, float],
    company_features: Mapping[str, float],
    market_features: Mapping[str, float],
    self_evaluation_mode: bool,
) -> List[Dict[str, Any]]:
    checks = [
        {
            "id": "HC-1",
            "description": "candidate.consistency_score >= 0.45",
            "satisfied": candidate_features["consistency_score"] >= 0.45,
            "value": round(candidate_features["consistency_score"], 6),
        },
        {
            "id": "HC-2",
            "description": "candidate.psychological_stability >= 0.35",
            "satisfied": candidate_features["psychological_stability"] >= 0.35,
            "value": round(candidate_features["psychological_stability"], 6),
        },
        {
            "id": "HC-3",
            "description": "candidate.execution_history >= 0.30",
            "satisfied": candidate_features["execution_history"] >= 0.30,
            "value": round(candidate_features["execution_history"], 6),
        },
        {
            "id": "HC-4",
            "description": "candidate.failure_recovery_capability >= 0.25",
            "satisfied": candidate_features["failure_recovery_capability"] >= 0.25,
            "value": round(candidate_features["failure_recovery_capability"], 6),
        },
        {
            "id": "HC-5",
            "description": "market.economic_environment >= 0.10",
            "satisfied": market_features["economic_environment"] >= 0.10,
            "value": round(market_features["economic_environment"], 6),
        },
    ]
    if not self_evaluation_mode:
        checks.extend(
            [
                {
                    "id": "HC-6",
                    "description": "company.engineering_depth >= 0.20",
                    "satisfied": company_features["engineering_depth"] >= 0.20,
                    "value": round(company_features["engineering_depth"], 6),
                },
                {
                    "id": "HC-7",
                    "description": "company.technical_standards_strictness >= 0.25",
                    "satisfied": company_features["technical_standards_strictness"] >= 0.25,
                    "value": round(company_features["technical_standards_strictness"], 6),
                },
            ]
        )
    return checks


def _build_soft_constraints(
    candidate_features: Mapping[str, float],
    company_features: Mapping[str, float],
    market_features: Mapping[str, float],
) -> List[Dict[str, Any]]:
    return [
        {
            "id": "SC-1",
            "description": "candidate.consistency_score < 0.55",
            "triggered": candidate_features["consistency_score"] < 0.55,
            "penalty": 0.015,
        },
        {
            "id": "SC-2",
            "description": "company.bureaucracy_level > 0.70",
            "triggered": company_features["bureaucracy_level"] > 0.70,
            "penalty": 0.018,
        },
        {
            "id": "SC-3",
            "description": "company.hiring_latency > 0.70",
            "triggered": company_features["hiring_latency"] > 0.70,
            "penalty": 0.018,
        },
        {
            "id": "SC-4",
            "description": "market.competition_density > 0.75",
            "triggered": market_features["competition_density"] > 0.75,
            "penalty": 0.010,
        },
        {
            "id": "SC-5",
            "description": "candidate.signal_to_noise_ratio < 0.45",
            "triggered": candidate_features["signal_to_noise_ratio"] < 0.45,
            "penalty": 0.014,
        },
    ]


def _build_contradictions(
    candidate_features: Mapping[str, float],
    company_features: Mapping[str, float],
    market_features: Mapping[str, float],
) -> Tuple[List[str], List[str]]:
    candidate_contradictions: List[str] = []
    company_contradictions: List[str] = []

    if company_features["innovation_index"] > 0.75 and company_features["hiring_latency"] > 0.70:
        company_contradictions.append("company innovation_index high while hiring_latency remains high.")
    if company_features["hiring_urgency"] > 0.80 and company_features["recruiter_response_speed"] < 0.40:
        company_contradictions.append("company hiring_urgency high while recruiter_response_speed is low.")
    if company_features["technical_standards_strictness"] > 0.80 and company_features["candidate_experience_quality"] < 0.35:
        company_contradictions.append("strict technical standards conflict with poor candidate_experience_quality.")
    if market_features["demand_pressure"] > 0.75 and company_features["compensation_efficiency"] < 0.40:
        company_contradictions.append("high market demand conflicts with low company compensation_efficiency.")

    if candidate_features["innovation_index"] > 0.80 and candidate_features["documentation_quality"] < 0.30:
        candidate_contradictions.append("candidate innovation_index high while documentation_quality is very low.")

    return candidate_contradictions, company_contradictions


def _map_affective_signals(
    candidate_company_fit: float,
    opportunity_loss_risk: float,
    acquisition_pressure_index: float,
    company_market_mismatch: float,
    hiring_inefficiency_drag: float,
    contradictions_found: bool,
) -> List[Dict[str, str]]:
    signals: List[Dict[str, str]] = []
    if candidate_company_fit >= 0.75:
        signals.append(
            {
                "signal": "Respect",
                "reason": "high candidate-company alignment detected through deterministic fit score.",
            }
        )
    if acquisition_pressure_index >= 0.78 or opportunity_loss_risk >= 0.72:
        signals.append(
            {
                "signal": "Urgency",
                "reason": "acquisition pressure exceeds threshold under market heat and candidate dominance.",
            }
        )
    if company_market_mismatch >= 0.60:
        signals.append(
            {
                "signal": "Risk",
                "reason": "company-market mismatch exceeds 0.60 and elevates deterministic failure risk index.",
            }
        )
    if hiring_inefficiency_drag >= 0.42:
        signals.append(
            {
                "signal": "Inefficiency",
                "reason": "recruiting workflow shows latency and bureaucracy concentration above threshold.",
            }
        )
    if contradictions_found:
        signals.append(
            {
                "signal": "Discrepancy",
                "reason": "contradiction engine detected conflicting company or candidate patterns.",
            }
        )
    return signals


def _decision_from_score(
    score: float,
    hard_constraints: List[Dict[str, Any]],
    opportunity_loss_risk: float,
    candidate_dominance_index: float,
    candidate_company_fit: float,
    candidate_excellence_index: float,
    acquisition_pressure_index: float,
) -> str:
    if any(not item["satisfied"] for item in hard_constraints):
        return "REJECT"
    if (
        score >= 0.85
        and acquisition_pressure_index >= 0.82
        and candidate_excellence_index >= 0.88
    ):
        return "MISSION_CRITICAL_ACQUISITION"
    if (
        score >= 0.76
        and acquisition_pressure_index >= 0.75
        and candidate_excellence_index >= 0.78
        and candidate_dominance_index >= 0.72
        and candidate_company_fit >= 0.55
    ):
        return "MANDATORY_ACQUISITION"
    if score >= 0.62:
        return "STRONG_HIRE"
    if score >= 0.44:
        return "CONSIDER"
    if opportunity_loss_risk >= 0.78 and candidate_dominance_index >= 0.85:
        return "CONSIDER"
    return "REJECT"


def _rank_key_factors(
    contribution_pool: Mapping[str, Dict[str, float]],
    limit: int = 12,
) -> List[Dict[str, Any]]:
    ranked = sorted(
        contribution_pool.values(),
        key=lambda item: abs(item["global_contribution"]),
        reverse=True,
    )[:limit]
    return [
        {
            "factor": f"{item['domain']}.{item['feature']}",
            "value": round(item["value"], 6),
            "weight": round(item["weight"], 6),
            "contribution": round(item["global_contribution"], 6),
        }
        for item in ranked
    ]


def evaluate_hiring_decision(
    candidate_payload: Mapping[str, Any],
    company_payload: Mapping[str, Any] | None,
    market_payload: Mapping[str, Any],
    weight_adjustments: Mapping[str, Mapping[str, float]] | None = None,
    options: EvaluationOptions | None = None,
) -> Dict[str, Any]:
    active_options = options or EvaluationOptions()
    candidate_features = candidate_payload["features"]
    market_features = market_payload["features"]
    if company_payload is None:
        company_features = _neutral_company_features()
        company_id = "neutral-company-profile"
    else:
        company_features = company_payload["features"]
        company_id = company_payload.get("company_id", "unknown-company")

    adjustments = weight_adjustments or {}
    candidate_adjustments = adjustments.get("candidate", {})
    company_adjustments = adjustments.get("company", {})
    market_adjustments = adjustments.get("market", {})

    candidate_score, candidate_contrib = _weighted_score(
        feature_values=candidate_features,
        base_weights=CANDIDATE_WEIGHTS,
        adjustments=candidate_adjustments,
        domain_name="candidate",
    )
    company_score, company_contrib = _weighted_score(
        feature_values=company_features,
        base_weights=COMPANY_WEIGHTS,
        adjustments=company_adjustments,
        domain_name="company",
    )
    market_score, market_contrib = _weighted_score(
        feature_values=market_features,
        base_weights=MARKET_WEIGHTS,
        adjustments=market_adjustments,
        domain_name="market",
    )

    candidate_company_fit = _candidate_company_fit(candidate_features, company_features)
    candidate_market_alignment = _candidate_market_alignment(candidate_features, market_features)
    company_market_mismatch = _company_market_mismatch(company_features, market_features)

    candidate_dominance_index = _clamp(
        0.45 * candidate_score
        + 0.30 * candidate_features["innovation_index"]
        + 0.25 * candidate_features["real_world_deployment_impact"]
    )
    opportunity_loss_risk = _clamp(
        0.38 * market_features["demand_pressure"]
        + 0.28 * candidate_dominance_index
        + 0.20 * company_features["hiring_latency"]
        + 0.14 * market_features["time_to_hire_urgency"]
    )
    decision_latency_impact = _clamp(
        company_features["hiring_latency"]
        * market_features["time_to_hire_urgency"]
        * (0.65 + 0.35 * company_features["bureaucracy_level"])
    )
    hiring_inefficiency_score = _clamp(
        0.42 * company_features["hiring_latency"]
        + 0.24 * company_features["bureaucracy_level"]
        + 0.20 * company_features["decision_making_entropy"]
        + 0.14 * (1.0 - company_features["recruiter_response_speed"])
    )
    candidate_excellence_index = _candidate_excellence_index(candidate_features, candidate_score)
    company_readiness_index = _company_readiness_index(company_features, company_score)
    market_heat_index = _market_heat_index(market_features)
    acquisition_pressure_index = _acquisition_pressure_index(
        opportunity_loss_risk=opportunity_loss_risk,
        candidate_dominance_index=candidate_dominance_index,
        market_heat_index=market_heat_index,
    )
    edge_network_trace = evaluate_whitebox_edge_network(
        {
            "candidate_excellence_index": candidate_excellence_index,
            "company_readiness_index": company_readiness_index,
            "market_heat_index": market_heat_index,
            "opportunity_loss_risk": opportunity_loss_risk,
            "hiring_inefficiency_drag": _clamp(hiring_inefficiency_score),
            "decision_latency_drag": _clamp(decision_latency_impact),
            "candidate_company_fit": candidate_company_fit,
            "company_market_mismatch": company_market_mismatch,
        }
    )
    edge_case_adjustment = float(edge_network_trace["edge_case_adjustment"])

    hard_constraints = _build_hard_constraints(
        candidate_features=candidate_features,
        company_features=company_features,
        market_features=market_features,
        self_evaluation_mode=active_options.self_evaluation_mode,
    )
    soft_constraints = _build_soft_constraints(
        candidate_features=candidate_features,
        company_features=company_features,
        market_features=market_features,
    )
    soft_penalty_total = sum(item["penalty"] for item in soft_constraints if item["triggered"])

    candidate_contradictions, company_contradictions = _build_contradictions(
        candidate_features=candidate_features,
        company_features=company_features,
        market_features=market_features,
    )
    # Only penalize candidate for their own contradictions; company contradictions are just flags.
    contradiction_penalty_base = min(0.12, len(candidate_contradictions) * 0.03)

    hiring_inefficiency_drag = _clamp(hiring_inefficiency_score * (1.0 - 0.45 * candidate_excellence_index))
    # Audit 3.2: slowness is MORE dangerous in a hot market.
    decision_latency_drag = _clamp(decision_latency_impact * (1.0 + 0.40 * market_heat_index))
    soft_penalty_adjusted = soft_penalty_total * (1.0 - 0.30 * candidate_excellence_index)
    contradiction_penalty_adjusted = contradiction_penalty_base * (1.0 - 0.25 * candidate_excellence_index)

    # Audit 3.1: Normalizing weights to sum to 1.0 (approximate redistributed values)
    score = _clamp(
        0.34 * candidate_score
        + 0.18 * company_score
        + 0.07 * market_score
        + 0.09 * candidate_company_fit
        + 0.06 * candidate_market_alignment
        - 0.04 * company_market_mismatch
        + 0.09 * candidate_dominance_index
        + 0.08 * opportunity_loss_risk
        + 0.05 * candidate_excellence_index
        + 0.04 * acquisition_pressure_index
        + edge_case_adjustment
        - 0.07 * hiring_inefficiency_drag
        - 0.05 * decision_latency_drag
        - soft_penalty_adjusted
        - contradiction_penalty_adjusted
    )
    # Audit: Achievement vs Resources multiplier (Grit Factor)
    # High achievement with low resource access (e.g. Egypt vs Harvard) boosts final score potential.
    resource_advantage = candidate_features.get("resource_advantage_index", 0.5)
    grit_multiplier = 1.0 + (1.0 - resource_advantage) * 0.15
    score = _clamp(score * grit_multiplier)

    # Non-linear boost for extreme excellence
    if candidate_excellence_index >= 0.88:
        score = _clamp(score + 0.04)

    if any(not item["satisfied"] for item in hard_constraints):
        score = min(score, 0.39)

    final_decision = _decision_from_score(
        score=score,
        hard_constraints=hard_constraints,
        opportunity_loss_risk=opportunity_loss_risk,
        candidate_dominance_index=candidate_dominance_index,
        candidate_company_fit=candidate_company_fit,
        candidate_excellence_index=candidate_excellence_index,
        acquisition_pressure_index=acquisition_pressure_index,
    )

    contribution_pool: MutableMapping[str, Dict[str, float]] = {}
    for feature_key, payload in candidate_contrib.items():
        payload["global_contribution"] = 0.34 * payload["normalized_contribution"]
        contribution_pool[feature_key] = payload
    for feature_key, payload in company_contrib.items():
        payload["global_contribution"] = 0.18 * payload["normalized_contribution"]
        contribution_pool[feature_key] = payload
    for feature_key, payload in market_contrib.items():
        payload["global_contribution"] = 0.07 * payload["normalized_contribution"]
        contribution_pool[feature_key] = payload

    contribution_pool["derived.candidate_company_fit"] = {
        "domain": "derived",
        "feature": "candidate_company_fit",
        "value": candidate_company_fit,
        "weight": 0.09,
        "normalized_contribution": candidate_company_fit,
        "raw_contribution": candidate_company_fit * 0.09,
        "global_contribution": candidate_company_fit * 0.09,
    }
    contribution_pool["derived.candidate_market_alignment"] = {
        "domain": "derived",
        "feature": "candidate_market_alignment",
        "value": candidate_market_alignment,
        "weight": 0.06,
        "normalized_contribution": candidate_market_alignment,
        "raw_contribution": candidate_market_alignment * 0.06,
        "global_contribution": candidate_market_alignment * 0.06,
    }
    contribution_pool["derived.company_market_mismatch"] = {
        "domain": "derived",
        "feature": "company_market_mismatch",
        "value": company_market_mismatch,
        "weight": -0.04,
        "normalized_contribution": company_market_mismatch,
        "raw_contribution": -company_market_mismatch * 0.04,
        "global_contribution": -company_market_mismatch * 0.04,
    }
    contribution_pool["derived.hiring_inefficiency_drag"] = {
        "domain": "derived",
        "feature": "hiring_inefficiency_drag",
        "value": hiring_inefficiency_drag,
        "weight": -0.07,
        "normalized_contribution": hiring_inefficiency_drag,
        "raw_contribution": -hiring_inefficiency_drag * 0.07,
        "global_contribution": -hiring_inefficiency_drag * 0.07,
    }
    contribution_pool["derived.decision_latency_drag"] = {
        "domain": "derived",
        "feature": "decision_latency_drag",
        "value": decision_latency_drag,
        "weight": -0.05,
        "normalized_contribution": decision_latency_drag,
        "raw_contribution": -decision_latency_drag * 0.05,
        "global_contribution": -decision_latency_drag * 0.05,
    }
    contribution_pool["derived.opportunity_loss_risk"] = {
        "domain": "derived",
        "feature": "opportunity_loss_risk",
        "value": opportunity_loss_risk,
        "weight": 0.08,
        "normalized_contribution": opportunity_loss_risk,
        "raw_contribution": opportunity_loss_risk * 0.08,
        "global_contribution": opportunity_loss_risk * 0.08,
    }
    contribution_pool["derived.candidate_dominance_index"] = {
        "domain": "derived",
        "feature": "candidate_dominance_index",
        "value": candidate_dominance_index,
        "weight": 0.09,
        "normalized_contribution": candidate_dominance_index,
        "raw_contribution": candidate_dominance_index * 0.09,
        "global_contribution": candidate_dominance_index * 0.09,
    }
    contribution_pool["derived.candidate_excellence_index"] = {
        "domain": "derived",
        "feature": "candidate_excellence_index",
        "value": candidate_excellence_index,
        "weight": 0.05,
        "normalized_contribution": candidate_excellence_index,
        "raw_contribution": candidate_excellence_index * 0.05,
        "global_contribution": candidate_excellence_index * 0.05,
    }
    contribution_pool["derived.acquisition_pressure_index"] = {
        "domain": "derived",
        "feature": "acquisition_pressure_index",
        "value": acquisition_pressure_index,
        "weight": 0.04,
        "normalized_contribution": acquisition_pressure_index,
        "raw_contribution": acquisition_pressure_index * 0.04,
        "global_contribution": acquisition_pressure_index * 0.04,
    }
    contribution_pool["derived.edge_case_adjustment"] = {
        "domain": "derived",
        "feature": "edge_case_adjustment",
        "value": edge_case_adjustment,
        "weight": 1.0,
        "normalized_contribution": edge_case_adjustment,
        "raw_contribution": edge_case_adjustment,
        "global_contribution": edge_case_adjustment,
    }
    contribution_pool["derived.soft_penalty_adjusted"] = {
        "domain": "derived",
        "feature": "soft_penalty_adjusted",
        "value": soft_penalty_adjusted,
        "weight": -1.0,
        "normalized_contribution": soft_penalty_adjusted,
        "raw_contribution": -soft_penalty_adjusted,
        "global_contribution": -soft_penalty_adjusted,
    }
    contribution_pool["derived.contradiction_penalty_adjusted"] = {
        "domain": "derived",
        "feature": "contradiction_penalty_adjusted",
        "value": contradiction_penalty_adjusted,
        "weight": -1.0,
        "normalized_contribution": contradiction_penalty_adjusted,
        "raw_contribution": -contradiction_penalty_adjusted,
        "global_contribution": -contradiction_penalty_adjusted,
    }

    key_factors = _rank_key_factors(contribution_pool=contribution_pool)

    risks: List[str] = []
    if opportunity_loss_risk >= 0.65:
        risks.append("opportunity_loss_risk exceeds 0.65.")
    if company_market_mismatch >= 0.55:
        risks.append("company_market_mismatch exceeds 0.55.")
    if decision_latency_impact >= 0.50:
        risks.append("decision_latency_impact exceeds 0.50.")
    if acquisition_pressure_index >= 0.80 and final_decision not in {
        "MANDATORY_ACQUISITION",
        "MISSION_CRITICAL_ACQUISITION",
    }:
        risks.append("acquisition_pressure_index exceeds 0.80 while decision is below mandatory level.")
    if any(not item["satisfied"] for item in hard_constraints):
        risks.append("one or more hard constraints failed.")

    inefficiencies: List[str] = []
    if hiring_inefficiency_score >= 0.55:
        inefficiencies.append("hiring_inefficiency_score exceeds 0.55.")
    if company_features["hiring_latency"] >= 0.65:
        inefficiencies.append("company hiring_latency exceeds 0.65.")
    if company_features["bureaucracy_level"] >= 0.65:
        inefficiencies.append("company bureaucracy_level exceeds 0.65.")
    if hiring_inefficiency_drag >= 0.40:
        inefficiencies.append("adjusted hiring_inefficiency_drag exceeds 0.40.")

    logical_trace = [
        f"Step 1: candidate_score = {candidate_score:.6f}.",
        f"Step 2: company_score = {company_score:.6f}.",
        f"Step 3: market_score = {market_score:.6f}.",
        f"Step 4: candidate_company_fit = {candidate_company_fit:.6f}.",
        f"Step 5: candidate_market_alignment = {candidate_market_alignment:.6f}.",
        f"Step 6: company_market_mismatch = {company_market_mismatch:.6f}.",
        f"Step 7: candidate_dominance_index = {candidate_dominance_index:.6f}.",
        f"Step 8: candidate_excellence_index = {candidate_excellence_index:.6f}.",
        f"Step 9: market_heat_index = {market_heat_index:.6f}.",
        f"Step 10: opportunity_loss_risk = {opportunity_loss_risk:.6f}.",
        f"Step 11: acquisition_pressure_index = {acquisition_pressure_index:.6f}.",
        f"Step 12: edge_case_adjustment = {edge_case_adjustment:.6f}.",
        f"Step 13: hiring_inefficiency_drag = {hiring_inefficiency_drag:.6f}; decision_latency_drag = {decision_latency_drag:.6f}.",
        (
            f"Step 14: soft_penalty_adjusted = {soft_penalty_adjusted:.6f}; "
            f"contradiction_penalty_adjusted = {contradiction_penalty_adjusted:.6f}."
        ),
        f"Step 15: final_score = {score:.6f}.",
        f"Step 16: final_decision = {final_decision}.",
    ]

    proof = [
        {"type": "definition", "statement": "D1: CandidateScore = SUM(wi * xi) / SUM(wi)."},
        {"type": "definition", "statement": "D2: CompanyScore = SUM(wi * xi) / SUM(wi)."},
        {"type": "definition", "statement": "D3: MarketScore = SUM(wi * xi) / SUM(wi)."},
        {
            "type": "definition",
            "statement": "D4: FinalScore = weighted sum of domain scores, cross-domain terms, derived metrics, and deterministic penalties.",
        },
        {
            "type": "definition",
            "statement": "D5: AcquisitionPressureIndex = weighted sum of opportunity loss, candidate dominance, and market heat.",
        },
        {
            "type": "definition",
            "statement": "D6: EdgeCaseAdjustment = deterministic white-box neural correction bounded in [-0.03, 0.03].",
        },
        {"type": "given", "statement": f"G1: candidate_id = {candidate_payload.get('candidate_id', 'unknown-candidate')}."},
        {"type": "given", "statement": f"G2: company_id = {company_id}."},
        {"type": "given", "statement": f"G3: market_id = {market_payload.get('market_id', 'unknown-market')}."},
        {
            "type": "deduction",
            "statement": (
                "S1: candidate.project_complexity "
                f"({candidate_features['project_complexity']:.6f}) compared with company.engineering_depth "
                f"({company_features['engineering_depth']:.6f})."
            ),
        },
        {
            "type": "deduction",
            "statement": (
                "S2: market.time_to_hire_urgency "
                f"({market_features['time_to_hire_urgency']:.6f}) and company.hiring_latency "
                f"({company_features['hiring_latency']:.6f}) produce decision_latency_drag "
                f"({decision_latency_drag:.6f})."
            ),
        },
        {
            "type": "deduction",
            "statement": (
                "S3: acquisition_pressure_index "
                f"({acquisition_pressure_index:.6f}) combines opportunity_loss_risk ({opportunity_loss_risk:.6f}), "
                f"candidate_dominance_index ({candidate_dominance_index:.6f}), and market_heat_index ({market_heat_index:.6f})."
            ),
        },
        {
            "type": "deduction",
            "statement": (
                "S4: edge_case_adjustment "
                f"({edge_case_adjustment:.6f}) produced by whitebox_edge_net_v1 with explicit visible weights."
            ),
        },
        {
            "type": "deduction",
            "statement": f"S5: hard constraint failures = {sum(1 for item in hard_constraints if not item['satisfied'])}.",
        },
        {"type": "conclusion", "statement": f"C1: final_score = {score:.6f}."},
        {"type": "conclusion", "statement": f"C2: final_decision = {final_decision}."},
    ]

    if active_options.explain_level != "full":
        logical_trace = logical_trace[:8]
        proof = proof[:7] + proof[-2:]

    affective_signals = _map_affective_signals(
        candidate_company_fit=candidate_company_fit,
        opportunity_loss_risk=opportunity_loss_risk,
        acquisition_pressure_index=acquisition_pressure_index,
        company_market_mismatch=company_market_mismatch,
        hiring_inefficiency_drag=hiring_inefficiency_drag,
        contradictions_found=bool(candidate_contradictions or company_contradictions),
    )

    result: Dict[str, Any] = {
        "final_decision": final_decision,
        "score": round(score, 6),
        "confidence": 1.0,
        "key_factors": key_factors,
        "risks": risks,
        "inefficiencies": inefficiencies,
        "contradictions": candidate_contradictions + company_contradictions,
        "candidate_contradictions": candidate_contradictions,
        "company_contradictions": company_contradictions,
        "logical_trace": logical_trace,
        "affective_signals": affective_signals,
        "proof": proof,
        "whitebox_neural_trace": edge_network_trace,
        "hard_constraints": hard_constraints,
        "soft_constraints": soft_constraints,
        "derived_metrics": {
            "opportunity_loss_risk": round(opportunity_loss_risk, 6),
            "hiring_inefficiency_score": round(hiring_inefficiency_score, 6),
            "hiring_inefficiency_drag": round(hiring_inefficiency_drag, 6),
            "candidate_dominance_index": round(candidate_dominance_index, 6),
            "candidate_excellence_index": round(candidate_excellence_index, 6),
            "acquisition_pressure_index": round(acquisition_pressure_index, 6),
            "edge_case_adjustment": round(edge_case_adjustment, 6),
            "company_readiness_index": round(company_readiness_index, 6),
            "market_heat_index": round(market_heat_index, 6),
            "decision_latency_impact": round(decision_latency_impact, 6),
            "decision_latency_drag": round(decision_latency_drag, 6),
            "candidate_company_fit": round(candidate_company_fit, 6),
            "candidate_market_alignment": round(candidate_market_alignment, 6),
            "company_market_mismatch": round(company_market_mismatch, 6),
            "candidate_score": round(candidate_score, 6),
            "company_score": round(company_score, 6),
            "market_score": round(market_score, 6),
            "soft_penalty_adjusted": round(soft_penalty_adjusted, 6),
            "contradiction_penalty_adjusted": round(contradiction_penalty_adjusted, 6),
        },
        "cross_domain_analysis": {
            "candidate_vs_company_fit": round(candidate_company_fit, 6),
            "candidate_vs_market_alignment": round(candidate_market_alignment, 6),
            "company_vs_market_mismatch": round(company_market_mismatch, 6),
            "candidate_excellence_vs_company_readiness_gap": round(
                candidate_excellence_index - company_readiness_index,
                6,
            ),
        },
        "_internal": {
            "contribution_ranking": [
                {
                    "domain": entry["domain"],
                    "feature": entry["feature"],
                    "contribution": entry["global_contribution"],
                }
                for entry in sorted(
                    contribution_pool.values(),
                    key=lambda item: abs(item["global_contribution"]),
                    reverse=True,
                )
            ],
            "score_formula": {
                "candidate_score_weight": 0.34,
                "company_score_weight": 0.18,
                "market_score_weight": 0.07,
                "candidate_company_fit_weight": 0.09,
                "candidate_market_alignment_weight": 0.06,
                "company_market_mismatch_weight": -0.04,
                "candidate_dominance_index_weight": 0.09,
                "opportunity_loss_risk_weight": 0.08,
                "candidate_excellence_index_weight": 0.05,
                "acquisition_pressure_index_weight": 0.04,
                "edge_case_adjustment_weight": 1.0,
                "hiring_inefficiency_drag_weight": -0.07,
                "decision_latency_drag_weight": -0.05,
                "soft_penalty_adjusted_weight": -1.0,
                "contradiction_penalty_adjusted_weight": -1.0,
            },
            "whitebox_neural_trace": edge_network_trace,
        },
    }

    if active_options.company_exposure_mode:
        result["company_exposure"] = {
            "inefficiency_score": round(hiring_inefficiency_score, 6),
            "inefficiency_drag": round(hiring_inefficiency_drag, 6),
            "latency_detector": round(decision_latency_impact, 6),
            "latency_drag": round(decision_latency_drag, 6),
            "loss_risk_if_delayed": round(opportunity_loss_risk, 6),
            "acquisition_pressure_index": round(acquisition_pressure_index, 6),
            "exposed_patterns": [
                "high-value candidate plus delayed process"
                if acquisition_pressure_index >= 0.80 and company_features["hiring_latency"] >= 0.60
                else "delay-risk below exposure threshold",
                "bureaucratic drag present" if company_features["bureaucracy_level"] >= 0.65 else "bureaucratic drag controlled",
                "response bottleneck detected" if company_features["recruiter_response_speed"] < 0.45 else "recruiter response within bound",
            ],
        }

    if active_options.self_evaluation_mode:
        improvement_targets = sorted(
            ({"feature": name, "value": value} for name, value in candidate_features.items()),
            key=lambda item: item["value"],
        )[:8]
        result["self_evaluation"] = {
            "readiness_score": round(_clamp(0.65 * candidate_score + 0.35 * candidate_market_alignment), 6),
            "candidate_excellence_index": round(candidate_excellence_index, 6),
            "market_alignment": round(candidate_market_alignment, 6),
            "improvement_targets": improvement_targets,
        }

    return result
