from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

from .catalog import (
    CANDIDATE_FEATURE_SPECS,
    COMPANY_FEATURE_SPECS,
    INPUT_SCHEMA_VERSION,
    MARKET_DYNAMIC_SIGNAL_KEYS,
    MARKET_FEATURE_SPECS,
)


class InputValidationError(ValueError):
    pass


def _validate_numeric_value(entity: str, feature_name: str, raw_value: Any, minimum: float, maximum: float) -> float:
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise InputValidationError(
            f"{entity}.features.{feature_name} must be numeric in [{minimum}, {maximum}], got {type(raw_value).__name__}."
        )
    numeric_value = float(raw_value)
    if math.isnan(numeric_value) or math.isinf(numeric_value):
        raise InputValidationError(f"{entity}.features.{feature_name} must be finite.")
    if numeric_value < minimum or numeric_value > maximum:
        raise InputValidationError(
            f"{entity}.features.{feature_name}={numeric_value} is out of bounds [{minimum}, {maximum}]."
        )
    return numeric_value


def _validate_feature_map(entity: str, features: Any, specs: Mapping[str, Any]) -> Dict[str, float]:
    if not isinstance(features, Mapping):
        raise InputValidationError(f"{entity}.features must be an object.")
    feature_keys = set(features.keys())
    expected_keys = set(specs.keys())
    missing = sorted(expected_keys - feature_keys)
    unknown = sorted(feature_keys - expected_keys)
    if missing:
        raise InputValidationError(f"{entity}.features missing {len(missing)} keys. First missing keys: {missing[:8]}")
    if unknown:
        raise InputValidationError(f"{entity}.features contains unknown keys. First unknown keys: {unknown[:8]}")
    validated: Dict[str, float] = {}
    for feature_name, spec in specs.items():
        validated[feature_name] = _validate_numeric_value(
            entity=entity,
            feature_name=feature_name,
            raw_value=features[feature_name],
            minimum=spec.min_value,
            maximum=spec.max_value,
        )
    return validated


def _validate_schema_header(entity: str, payload: Any, id_key: str) -> str:
    if not isinstance(payload, Mapping):
        raise InputValidationError(f"{entity} payload must be an object.")
    schema_version = payload.get("schema_version")
    if schema_version != INPUT_SCHEMA_VERSION:
        raise InputValidationError(
            f"{entity}.schema_version must be '{INPUT_SCHEMA_VERSION}', got '{schema_version}'."
        )
    identifier = payload.get(id_key) or payload.get("id")
    if not isinstance(identifier, str) or not identifier.strip():
        raise InputValidationError(f"{entity}.{id_key} must be a non-empty string.")
    return identifier.strip()


def _validate_candidate_consistency(features: Mapping[str, float]) -> None:
    if features["execution_history"] < 0.2 and features["real_world_deployment_impact"] > 0.85:
        raise InputValidationError(
            "candidate data inconsistency: deployment impact cannot exceed 0.85 when execution_history is below 0.2."
        )
    if features["project_complexity"] > 0.9 and features["system_design_depth"] < 0.25:
        raise InputValidationError(
            "candidate data inconsistency: project_complexity above 0.9 requires system_design_depth at least 0.25."
        )
    if features["learning_velocity"] > 0.9 and features["consistency_score"] < 0.15:
        raise InputValidationError(
            "candidate data inconsistency: learning_velocity above 0.9 conflicts with consistency_score below 0.15."
        )


def _validate_company_consistency(features: Mapping[str, float]) -> None:
    if (
        features["hiring_urgency"] > 0.85
        and features["hiring_latency"] > 0.9
        and features["recruiter_response_speed"] < 0.2
    ):
        raise InputValidationError(
            "company data inconsistency: urgent hiring cannot coexist with extreme latency and near-zero recruiter response."
        )
    if features["technical_standards_strictness"] > 0.9 and features["engineering_depth"] < 0.2:
        raise InputValidationError(
            "company data inconsistency: strict technical standards require engineering_depth at least 0.2."
        )
    if features["innovation_index"] > 0.9 and features["budget_flexibility"] < 0.1:
        raise InputValidationError(
            "company data inconsistency: innovation_index above 0.9 requires budget_flexibility at least 0.1."
        )


def _validate_market_consistency(features: Mapping[str, float]) -> None:
    if features["time_to_hire_urgency"] > 0.85 and features["demand_pressure"] < 0.3 and features["cost_of_delay"] < 0.2:
        raise InputValidationError(
            "market data inconsistency: high hiring urgency conflicts with low demand_pressure and low cost_of_delay."
        )
    if features["demand_pressure"] < 0.2 and features["talent_scarcity"] > 0.9 and features["competition_density"] < 0.2:
        raise InputValidationError(
            "market data inconsistency: scarcity above 0.9 conflicts with demand_pressure and competition_density both below 0.2."
        )


def validate_candidate_payload(payload: Any) -> Dict[str, Any]:
    candidate_id = _validate_schema_header(entity="candidate", payload=payload, id_key="candidate_id")
    features = _validate_feature_map(entity="candidate", features=payload.get("features"), specs=CANDIDATE_FEATURE_SPECS)
    _validate_candidate_consistency(features)
    return {"schema_version": INPUT_SCHEMA_VERSION, "candidate_id": candidate_id, "features": features}


def validate_company_payload(payload: Any) -> Dict[str, Any]:
    company_id = _validate_schema_header(entity="company", payload=payload, id_key="company_id")
    features = _validate_feature_map(entity="company", features=payload.get("features"), specs=COMPANY_FEATURE_SPECS)
    _validate_company_consistency(features)
    return {"schema_version": INPUT_SCHEMA_VERSION, "company_id": company_id, "features": features}


def validate_market_payload(payload: Any) -> Dict[str, Any]:
    market_id = _validate_schema_header(entity="market", payload=payload, id_key="market_id")
    features = _validate_feature_map(entity="market", features=payload.get("features"), specs=MARKET_FEATURE_SPECS)
    dynamic_signals = payload.get("dynamic_signals")
    if dynamic_signals is not None:
        if not isinstance(dynamic_signals, Mapping):
            raise InputValidationError("market.dynamic_signals must be an object when provided.")
        for key in MARKET_DYNAMIC_SIGNAL_KEYS:
            if key not in dynamic_signals:
                raise InputValidationError(f"market.dynamic_signals missing key '{key}'.")
            signal_value = _validate_numeric_value(
                entity="market.dynamic_signals",
                feature_name=key,
                raw_value=dynamic_signals[key],
                minimum=0.0,
                maximum=1.0,
            )
            if abs(signal_value - features[key]) > 1e-9:
                raise InputValidationError(
                    f"market.dynamic_signals.{key} must match market.features.{key} exactly."
                )
    _validate_market_consistency(features)
    return {"schema_version": INPUT_SCHEMA_VERSION, "market_id": market_id, "features": features}


def load_json(path: str | Path) -> Dict[str, Any]:
    resolved = Path(path)
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise InputValidationError(f"Input file not found: {resolved}") from exc
    except json.JSONDecodeError as exc:
        raise InputValidationError(f"Invalid JSON in {resolved}: {exc.msg} at line {exc.lineno}, column {exc.colno}.") from exc
    if not isinstance(payload, dict):
        raise InputValidationError(f"Top-level JSON object expected in {resolved}.")
    return payload
