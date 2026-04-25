from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

from .catalog import (
    CANDIDATE_FEATURE_SPECS,
    COMPANY_FEATURE_SPECS,
    INPUT_SCHEMA_VERSION,
    MARKET_DYNAMIC_SIGNAL_KEYS,
    MARKET_FEATURE_SPECS,
)
from .templates import build_candidate_template, build_company_template, build_market_template


CANDIDATE_LEVELS = (
    "level_1_foundation",
    "level_2_progressing",
    "level_3_advanced",
    "level_4_expert",
    "level_5_principal",
)

COMPANY_NAMES = ("meta", "google", "coveo", "lazarda", "nexthink")

COUNTRIES = ("canada", "usa", "netherlands", "egypt", "singapore")
MARKET_SCENARIOS = ("ai-current", "ml-2016", "security-current")


DEMO_JOE_DOE_PROFILE = {
    "name": "Joe Doe",
    "headline": "Demo Candidate for Axiom-Recruit calibration",
    "location": "Toronto, Canada",
    "summary": "Synthetic demonstration profile used for deterministic level calibration and regression testing.",
}


REAL_WORLD_COMPANY_CONTEXT = {
    "meta": {
        "official_name": "Meta Platforms",
        "hq": "Menlo Park, USA",
        "industry": "Consumer and enterprise technology",
        "notes": "Large-scale product engineering with high standards and multi-stage hiring loops.",
    },
    "google": {
        "official_name": "Google",
        "hq": "Mountain View, USA",
        "industry": "Search, cloud, AI, and platform infrastructure",
        "notes": "Strong engineering depth with rigorous process and globally competitive talent market.",
    },
    "coveo": {
        "official_name": "Coveo",
        "hq": "Quebec City, Canada",
        "industry": "Enterprise search and AI relevance platforms",
        "notes": "Product-oriented engineering organization with faster loops than hyperscalers.",
    },
    "lazarda": {
        "official_name": "Lazarda",
        "hq": "Singapore",
        "industry": "E-commerce platform and digital operations",
        "notes": "Growth-heavy execution environment with mixed process maturity across teams.",
    },
    "nexthink": {
        "official_name": "Nexthink",
        "hq": "Lausanne, Switzerland",
        "industry": "Digital employee experience and endpoint analytics",
        "notes": "Technical B2B product environment with solid engineering standards and observability culture.",
    },
}


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def _noise(token: str, amplitude: float = 0.03) -> float:
    ascii_sum = sum(ord(character) for character in token)
    bucket = (ascii_sum % 1000) / 999
    return (bucket - 0.5) * 2 * amplitude


def _blend(low_value: float, high_value: float, ratio: float) -> float:
    return low_value + (high_value - low_value) * ratio


def _base_candidate_profile() -> Dict[str, object]:
    return build_candidate_template()


def _base_company_profile() -> Dict[str, object]:
    return build_company_template()


def _base_market_profile() -> Dict[str, object]:
    return build_market_template()


def build_main_ziad_profile() -> Dict[str, object]:
    # Sync with candidate.json if it exists in the root directory
    candidate_file = Path("candidate.json")
    features = {}
    profile_data = {}
    
    if candidate_file.exists():
        try:
            with candidate_file.open("r", encoding="utf-8") as f:
                source = json.load(f)
                features = source.get("features", {})
                # Try to get the detailed profile info if present
                profile_data = source.get("default_candidate_profile", source.get("candidate_profile", {}))
        except Exception:
            pass

    # Build base from template
    payload = _base_candidate_profile()
    payload["candidate_id"] = "main_ziad_salah"
    payload["profile_level"] = "principal"
    payload["profile_kind"] = "main_candidate"
    
    if profile_data:
        payload["candidate_profile"] = profile_data
    
    if features:
        # Inject features from the source of truth
        for k, v in features.items():
            if k in CANDIDATE_FEATURE_SPECS:
                payload["features"][k] = v
    else:
        # Fallback values if candidate.json is missing
        payload["features"]["resource_advantage_index"] = 0.15
        payload["features"]["innovation_index"] = 0.99

    return payload


def build_candidate_level_profile(level_index: int) -> Dict[str, object]:
    if level_index < 1 or level_index > 5:
        raise ValueError("level_index must be between 1 and 5.")
    level_ratio = (level_index - 1) / 4
    base_candidate = _base_candidate_profile()
    base_features = base_candidate["features"]
    features: Dict[str, float] = {}

    for feature_name in CANDIDATE_FEATURE_SPECS:
        high_value = base_features[feature_name]
        low_value = 0.28 + 0.20 * ((sum(ord(char) for char in feature_name) % 11) / 10)
        # Principal level (5) should be near perfect.
        upper_limit = 0.98 if level_index == 5 else 0.88
        blended = _blend(low_value, high_value * upper_limit, level_ratio)
        blended += _noise(f"candidate:{feature_name}:L{level_index}", amplitude=0.015)
        features[feature_name] = _clamp(blended)

    features["consistency_score"] = _clamp(_blend(0.42, 0.83, level_ratio))
    features["psychological_stability"] = _clamp(_blend(0.45, 0.82, level_ratio))
    features["execution_history"] = _clamp(_blend(0.36, 0.86, level_ratio))
    features["failure_recovery_capability"] = _clamp(_blend(0.33, 0.84, level_ratio))
    features["learning_velocity"] = _clamp(_blend(0.48, 0.88, level_ratio))
    features["project_complexity"] = _clamp(_blend(0.38, 0.90, level_ratio))
    features["innovation_index"] = _clamp(_blend(0.40, 0.89, level_ratio))
    features["real_world_deployment_impact"] = _clamp(_blend(0.32, 0.87, level_ratio))
    features["resource_advantage_index"] = _clamp(_blend(0.20, 0.90, level_ratio))

    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "candidate_id": f"demo_joe_doe_level_{level_index}",
        "profile_level": level_index,
        "profile_kind": "demo_level_candidate",
        "candidate_profile": DEMO_JOE_DOE_PROFILE,
        "features": features,
    }


def _company_core_targets() -> Mapping[str, Mapping[str, Mapping[str, float]]]:
    return {
        "meta": {
            "engineering_depth": {"low": 0.62, "high": 0.93},
            "innovation_index": {"low": 0.68, "high": 0.95},
            "hiring_latency": {"low": 0.76, "high": 0.52},
            "bureaucracy_level": {"low": 0.74, "high": 0.57},
            "recruiter_response_speed": {"low": 0.42, "high": 0.77},
            "compensation_efficiency": {"low": 0.61, "high": 0.88},
            "candidate_experience_quality": {"low": 0.49, "high": 0.82},
            "hiring_urgency": {"low": 0.64, "high": 0.86},
        },
        "google": {
            "engineering_depth": {"low": 0.66, "high": 0.96},
            "innovation_index": {"low": 0.71, "high": 0.95},
            "hiring_latency": {"low": 0.79, "high": 0.60},
            "bureaucracy_level": {"low": 0.78, "high": 0.64},
            "recruiter_response_speed": {"low": 0.40, "high": 0.74},
            "compensation_efficiency": {"low": 0.67, "high": 0.92},
            "candidate_experience_quality": {"low": 0.46, "high": 0.79},
            "hiring_urgency": {"low": 0.60, "high": 0.82},
        },
        "coveo": {
            "engineering_depth": {"low": 0.58, "high": 0.90},
            "innovation_index": {"low": 0.60, "high": 0.89},
            "hiring_latency": {"low": 0.66, "high": 0.43},
            "bureaucracy_level": {"low": 0.60, "high": 0.38},
            "recruiter_response_speed": {"low": 0.55, "high": 0.88},
            "compensation_efficiency": {"low": 0.56, "high": 0.83},
            "candidate_experience_quality": {"low": 0.60, "high": 0.90},
            "hiring_urgency": {"low": 0.58, "high": 0.80},
        },
        "lazarda": {
            "engineering_depth": {"low": 0.50, "high": 0.82},
            "innovation_index": {"low": 0.55, "high": 0.85},
            "hiring_latency": {"low": 0.73, "high": 0.52},
            "bureaucracy_level": {"low": 0.71, "high": 0.53},
            "recruiter_response_speed": {"low": 0.45, "high": 0.81},
            "compensation_efficiency": {"low": 0.53, "high": 0.81},
            "candidate_experience_quality": {"low": 0.49, "high": 0.84},
            "hiring_urgency": {"low": 0.62, "high": 0.84},
        },
        "nexthink": {
            "engineering_depth": {"low": 0.60, "high": 0.90},
            "innovation_index": {"low": 0.62, "high": 0.90},
            "hiring_latency": {"low": 0.67, "high": 0.44},
            "bureaucracy_level": {"low": 0.64, "high": 0.41},
            "recruiter_response_speed": {"low": 0.52, "high": 0.86},
            "compensation_efficiency": {"low": 0.59, "high": 0.85},
            "candidate_experience_quality": {"low": 0.56, "high": 0.88},
            "hiring_urgency": {"low": 0.57, "high": 0.81},
        },
    }


def build_company_level_profile(company_name: str, level_index: int) -> Dict[str, object]:
    if company_name not in COMPANY_NAMES:
        raise ValueError(f"unsupported company_name '{company_name}'.")
    if level_index < 1 or level_index > 5:
        raise ValueError("level_index must be between 1 and 5.")

    level_ratio = (level_index - 1) / 4
    base_payload = _base_company_profile()
    base_features = base_payload["features"]
    negative_features = {
        "hiring_latency",
        "bureaucracy_level",
        "decision_making_entropy",
        "interview_loop_length",
        "offer_approval_delay",
        "internal_politics_factor",
        "technical_debt_pressure",
    }

    features: Dict[str, float] = {}
    for feature_name in COMPANY_FEATURE_SPECS:
        baseline = base_features[feature_name]
        if feature_name in negative_features:
            value = baseline - 0.20 * (level_ratio - 0.5)
        else:
            value = baseline + 0.24 * (level_ratio - 0.5)
        value += _noise(f"company:{company_name}:{feature_name}:L{level_index}", amplitude=0.02)
        features[feature_name] = _clamp(value)

    core_targets = _company_core_targets()[company_name]
    for feature_name, bounds in core_targets.items():
        features[feature_name] = _clamp(_blend(bounds["low"], bounds["high"], level_ratio))

    features["technical_standards_strictness"] = _clamp(_blend(0.54, 0.92, level_ratio))
    features["team_stability"] = _clamp(_blend(0.45, 0.90, level_ratio))
    features["leadership_accountability"] = _clamp(_blend(0.50, 0.91, level_ratio))

    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "company_id": f"{company_name}_level_{level_index}",
        "company_name": company_name,
        "profile_level": level_index,
        "profile_kind": "company_level_profile",
        "features": features,
    }


def build_company_real_world_profile(company_name: str) -> Dict[str, object]:
    context = REAL_WORLD_COMPANY_CONTEXT[company_name]
    base_profile = build_company_level_profile(company_name=company_name, level_index=4)
    features = dict(base_profile["features"])
    if company_name in {"meta", "google"}:
        features["hiring_latency"] = max(features["hiring_latency"], 0.58)
        features["bureaucracy_level"] = max(features["bureaucracy_level"], 0.60)
        features["engineering_depth"] = max(features["engineering_depth"], 0.89)
        features["compensation_efficiency"] = max(features["compensation_efficiency"], 0.86)
    if company_name == "coveo":
        features["hiring_latency"] = min(features["hiring_latency"], 0.47)
        features["recruiter_response_speed"] = max(features["recruiter_response_speed"], 0.82)
        features["engineering_depth"] = max(features["engineering_depth"], 0.86)
        features["innovation_index"] = max(features["innovation_index"], 0.82)
        features["candidate_experience_quality"] = max(features["candidate_experience_quality"], 0.82)
    if company_name == "lazarda":
        features["growth_pressure"] = max(features["growth_pressure"], 0.82)
        features["hiring_urgency"] = max(features["hiring_urgency"], 0.78)
    if company_name == "nexthink":
        features["candidate_experience_quality"] = max(features["candidate_experience_quality"], 0.74)

    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "company_id": f"{company_name}_real_world_profile",
        "company_name": company_name,
        "profile_kind": "real_world_company_profile",
        "real_world_context": context,
        "features": features,
    }


def _country_biases() -> Mapping[str, Mapping[str, float]]:
    return {
        "canada": {
            "demand_pressure": -0.04,
            "talent_scarcity": -0.02,
            "competition_density": -0.06,
            "economic_environment": 0.06,
            "salary_inflation": -0.05,
            "candidate_mobility": -0.04,
            "ai_adoption_velocity": -0.03,
            "remote_talent_access": 0.10,
            "venture_funding_temperature": -0.03,
            "regulatory_pressure": 0.05,
        },
        "usa": {
            "demand_pressure": 0.10,
            "talent_scarcity": 0.06,
            "competition_density": 0.12,
            "economic_environment": 0.03,
            "salary_inflation": 0.08,
            "candidate_mobility": 0.09,
            "ai_adoption_velocity": 0.09,
            "remote_talent_access": 0.06,
            "venture_funding_temperature": 0.11,
            "regulatory_pressure": -0.02,
        },
        "netherlands": {
            "demand_pressure": -0.03,
            "talent_scarcity": -0.05,
            "competition_density": -0.07,
            "economic_environment": 0.08,
            "salary_inflation": -0.07,
            "candidate_mobility": -0.05,
            "ai_adoption_velocity": -0.05,
            "remote_talent_access": 0.09,
            "venture_funding_temperature": -0.04,
            "regulatory_pressure": 0.07,
        },
        "egypt": {
            "demand_pressure": -0.18,
            "talent_scarcity": 0.22,
            "competition_density": -0.25,
            "economic_environment": -0.32,
            "salary_inflation": 0.08,
            "candidate_mobility": 0.12,
            "ai_adoption_velocity": -0.45,
            "remote_talent_access": 0.18,
            "venture_funding_temperature": -0.28,
            "regulatory_pressure": -0.08,
        },
        "singapore": {
            "demand_pressure": 0.08,
            "talent_scarcity": 0.09,
            "competition_density": 0.09,
            "economic_environment": 0.11,
            "salary_inflation": 0.06,
            "candidate_mobility": 0.02,
            "ai_adoption_velocity": 0.08,
            "remote_talent_access": 0.07,
            "venture_funding_temperature": 0.07,
            "regulatory_pressure": 0.02,
        },
    }


def _scenario_baselines() -> Mapping[str, Mapping[str, float]]:
    return {
        "ai-current": {
            "demand_pressure": 0.82,
            "talent_scarcity": 0.79,
            "competition_density": 0.78,
            "economic_environment": 0.66,
            "time_to_hire_urgency": 0.84,
            "salary_inflation": 0.72,
            "remote_talent_access": 0.72,
            "layoff_pressure": 0.38,
            "venture_funding_temperature": 0.71,
            "regulatory_pressure": 0.54,
            "cost_of_delay": 0.85,
            "skill_obsolescence_speed": 0.83,
            "outsourcing_pressure": 0.55,
            "geo_mobility_index": 0.66,
            "sector_volatility": 0.61,
            "ai_adoption_velocity": 0.92,
            "open_role_growth": 0.80,
            "candidate_mobility": 0.68,
            "offer_decline_rate": 0.66,
            "passive_talent_activity": 0.74,
        },
        "security-current": {
            "demand_pressure": 0.77,
            "talent_scarcity": 0.76,
            "competition_density": 0.69,
            "economic_environment": 0.64,
            "time_to_hire_urgency": 0.80,
            "salary_inflation": 0.68,
            "remote_talent_access": 0.68,
            "layoff_pressure": 0.33,
            "venture_funding_temperature": 0.62,
            "regulatory_pressure": 0.64,
            "cost_of_delay": 0.81,
            "skill_obsolescence_speed": 0.72,
            "outsourcing_pressure": 0.49,
            "geo_mobility_index": 0.63,
            "sector_volatility": 0.56,
            "ai_adoption_velocity": 0.70,
            "open_role_growth": 0.74,
            "candidate_mobility": 0.61,
            "offer_decline_rate": 0.58,
            "passive_talent_activity": 0.66,
        },
        "ml-2016": {
            "demand_pressure": 0.34,
            "talent_scarcity": 0.32,
            "competition_density": 0.28,
            "economic_environment": 0.59,
            "time_to_hire_urgency": 0.31,
            "salary_inflation": 0.34,
            "remote_talent_access": 0.29,
            "layoff_pressure": 0.47,
            "venture_funding_temperature": 0.35,
            "regulatory_pressure": 0.41,
            "cost_of_delay": 0.30,
            "skill_obsolescence_speed": 0.37,
            "outsourcing_pressure": 0.58,
            "geo_mobility_index": 0.41,
            "sector_volatility": 0.44,
            "ai_adoption_velocity": 0.24,
            "open_role_growth": 0.36,
            "candidate_mobility": 0.38,
            "offer_decline_rate": 0.29,
            "passive_talent_activity": 0.35,
        },
    }


def _vector_value(feature_name: str, features: Mapping[str, float], country: str, scenario: str) -> float:
    if feature_name.startswith("macro_vector"):
        base = 0.55 * features["economic_environment"] + 0.45 * (1 - features["sector_volatility"])
    elif feature_name.startswith("labor_vector"):
        base = 0.60 * features["demand_pressure"] + 0.40 * features["talent_scarcity"]
    elif feature_name.startswith("salary_vector"):
        base = 0.65 * features["salary_inflation"] + 0.35 * features["offer_decline_rate"]
    elif feature_name.startswith("technology_vector"):
        base = 0.70 * features["ai_adoption_velocity"] + 0.30 * features["skill_obsolescence_speed"]
    else:
        base = 0.60 * features["regulatory_pressure"] + 0.40 * (1 - features["outsourcing_pressure"])
    base += _noise(f"vector:{country}:{scenario}:{feature_name}", amplitude=0.012)
    return _clamp(base)


def build_market_profile(country: str, scenario: str) -> Dict[str, object]:
    if country not in COUNTRIES:
        raise ValueError(f"unsupported country '{country}'.")
    if scenario not in MARKET_SCENARIOS:
        raise ValueError(f"unsupported scenario '{scenario}'.")

    _ = _base_market_profile()
    scenario_base = _scenario_baselines()[scenario]
    country_bias = _country_biases()[country]
    features: Dict[str, float] = {}

    for feature_name in MARKET_FEATURE_SPECS:
        if "_vector_" in feature_name:
            continue
        base = scenario_base.get(feature_name, 0.50)
        bias = country_bias.get(feature_name, 0.0)
        value = base + bias + _noise(f"market:{country}:{scenario}:{feature_name}", amplitude=0.015)
        if scenario == "ml-2016" and country == "egypt" and feature_name == "competition_density":
            value = 0.17
        features[feature_name] = _clamp(value)

    for feature_name in MARKET_FEATURE_SPECS:
        if "_vector_" in feature_name:
            features[feature_name] = _vector_value(feature_name, features, country, scenario)

    dynamic_signals = {key: features[key] for key in MARKET_DYNAMIC_SIGNAL_KEYS}
    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "market_id": f"{scenario}_{country}",
        "country": country,
        "scenario": scenario,
        "profile_kind": "market_real_world_profile",
        "features": features,
        "dynamic_signals": dynamic_signals,
    }


def write_profile_library(root_directory: str | Path) -> Mapping[str, List[Path]]:
    root = Path(root_directory)
    profiles_root = root / "profiles"
    if profiles_root.exists():
        shutil.rmtree(profiles_root)
    candidates_root = profiles_root / "candidates"
    companies_root = profiles_root / "companies"
    markets_root = profiles_root / "markets"

    candidates_root.mkdir(parents=True, exist_ok=True)
    companies_root.mkdir(parents=True, exist_ok=True)
    markets_root.mkdir(parents=True, exist_ok=True)

    created_candidates: List[Path] = []
    created_companies: List[Path] = []
    created_markets: List[Path] = []

    for level_index, level_name in enumerate(CANDIDATE_LEVELS, start=1):
        payload = build_candidate_level_profile(level_index)
        path = candidates_root / f"{level_name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        created_candidates.append(path)

    main_candidate_path = candidates_root / "main_ziad_salah.json"
    main_candidate_path.write_text(
        json.dumps(build_main_ziad_profile(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    created_candidates.append(main_candidate_path)

    for company_name in COMPANY_NAMES:
        company_folder = companies_root / company_name
        company_folder.mkdir(parents=True, exist_ok=True)
        real_profile_path = company_folder / "real_world_profile.json"
        real_profile_path.write_text(
            json.dumps(build_company_real_world_profile(company_name), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        created_companies.append(real_profile_path)

    for country in COUNTRIES:
        country_folder = markets_root / country
        country_folder.mkdir(parents=True, exist_ok=True)
        for scenario in MARKET_SCENARIOS:
            payload = build_market_profile(country, scenario)
            path = country_folder / f"{scenario}.json"
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            created_markets.append(path)

    manifest = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "candidates": [str(path.relative_to(root)) for path in created_candidates],
        "companies": [str(path.relative_to(root)) for path in created_companies],
        "markets": [str(path.relative_to(root)) for path in created_markets],
    }
    manifest_path = profiles_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "candidates": created_candidates,
        "companies": created_companies,
        "markets": created_markets,
        "manifest": [manifest_path],
    }


def iter_profile_payloads(root_directory: str | Path) -> Iterable[Mapping[str, object]]:
    root = Path(root_directory)
    manifest_path = root / "profiles" / "manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payloads: List[Mapping[str, object]] = []
    for bucket in ("candidates", "companies", "markets"):
        for relative_path in manifest.get(bucket, []):
            payloads.append(json.loads((root / relative_path).read_text(encoding="utf-8")))
    return payloads
