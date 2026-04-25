from __future__ import annotations

import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from .catalog import CANDIDATE_WEIGHTS, COMPANY_WEIGHTS, MARKET_WEIGHTS

LEDGER_SCHEMA_VERSION = "1.0"


def _clamp(value: float, minimum: float, maximum: float) -> float:
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def _base_weight_snapshot() -> Dict[str, Dict[str, float]]:
    return {
        "candidate": deepcopy(CANDIDATE_WEIGHTS),
        "company": deepcopy(COMPANY_WEIGHTS),
        "market": deepcopy(MARKET_WEIGHTS),
    }


def _empty_adjustment_snapshot() -> Dict[str, Dict[str, float]]:
    return {
        "candidate": {feature_name: 0.0 for feature_name in CANDIDATE_WEIGHTS},
        "company": {feature_name: 0.0 for feature_name in COMPANY_WEIGHTS},
        "market": {feature_name: 0.0 for feature_name in MARKET_WEIGHTS},
    }


def create_fresh_ledger() -> Dict[str, Any]:
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "base_weights": _base_weight_snapshot(),
        "adjustments": _empty_adjustment_snapshot(),
        "history": [],
    }


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_valid_adjustment_snapshot(adjustments: Any) -> bool:
    if not isinstance(adjustments, dict):
        return False
    expected_domains = {
        "candidate": CANDIDATE_WEIGHTS,
        "company": COMPANY_WEIGHTS,
        "market": MARKET_WEIGHTS,
    }
    if set(adjustments.keys()) != set(expected_domains.keys()):
        return False
    for domain_name, expected_features in expected_domains.items():
        domain_adjustments = adjustments.get(domain_name)
        if not isinstance(domain_adjustments, dict):
            return False
        if set(domain_adjustments.keys()) != set(expected_features.keys()):
            return False
        for value in domain_adjustments.values():
            if not _is_finite_number(value):
                return False
    return True


def load_weight_ledger(path: str | Path) -> Dict[str, Any]:
    ledger_path = Path(path)
    if not ledger_path.exists():
        return create_fresh_ledger()
    try:
        with ledger_path.open("r", encoding="utf-8") as handle:
            ledger = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return create_fresh_ledger()
    if not isinstance(ledger, dict):
        return create_fresh_ledger()
    if ledger.get("schema_version") != LEDGER_SCHEMA_VERSION:
        return create_fresh_ledger()
    if not _is_valid_adjustment_snapshot(ledger.get("adjustments")):
        return create_fresh_ledger()
    if not isinstance(ledger.get("history"), list):
        return create_fresh_ledger()
    return ledger


def save_weight_ledger(path: str | Path, ledger: Mapping[str, Any]) -> None:
    ledger_path = Path(path)
    with ledger_path.open("w", encoding="utf-8") as handle:
        json.dump(ledger, handle, indent=2, sort_keys=True)


def apply_feedback(
    ledger: Dict[str, Any],
    rating: int,
    score: float,
    contribution_ranking: List[Mapping[str, Any]],
    reason: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if rating < 1 or rating > 10:
        raise ValueError("rating must be between 1 and 10 inclusive.")

    normalized_rating = (rating - 1) / 9
    divergence = round(normalized_rating - score, 6)
    raw_delta = divergence * 0.05
    bounded_delta = _clamp(raw_delta, -0.03, 0.03)

    top_ranked = [
        item
        for item in contribution_ranking
        if item.get("domain") in ("candidate", "company", "market")
    ][:12]
    max_contribution = max((abs(item.get("contribution", 0.0)) for item in top_ranked), default=1.0)
    if max_contribution == 0:
        max_contribution = 1.0

    adjustment_records: List[Dict[str, Any]] = []
    for item in top_ranked:
        domain = item["domain"]
        feature = item["feature"]
        contribution = float(item.get("contribution", 0.0))
        scale = abs(contribution) / max_contribution
        delta = round(bounded_delta * scale, 8)

        base_weight = float(ledger["base_weights"][domain][feature])
        previous_adjustment = float(ledger["adjustments"][domain][feature])
        bound = round(base_weight * 0.25, 8)
        updated_adjustment = round(_clamp(previous_adjustment + delta, -bound, bound), 8)
        ledger["adjustments"][domain][feature] = updated_adjustment

        adjustment_records.append(
            {
                "domain": domain,
                "feature": feature,
                "base_weight": base_weight,
                "previous_adjustment": previous_adjustment,
                "delta": delta,
                "updated_adjustment": updated_adjustment,
                "bound": bound,
            }
        )

    history_entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rating": rating,
        "normalized_rating": round(normalized_rating, 6),
        "score": round(score, 6),
        "divergence": divergence,
        "bounded_delta": round(bounded_delta, 8),
        "reason": reason.strip() if reason else "no reason provided",
        "adjustments": adjustment_records,
    }
    ledger["history"].append(history_entry)

    summary = {
        "rating": rating,
        "normalized_rating": round(normalized_rating, 6),
        "score": round(score, 6),
        "divergence": divergence,
        "divergence_detected": abs(divergence) >= 0.20,
        "adjustment_count": len(adjustment_records),
        "bounded_delta": round(bounded_delta, 8),
        "reason": history_entry["reason"],
    }
    return ledger, summary


def revert_last_feedback(ledger: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    history = ledger.get("history", [])
    if not history:
        return ledger, {"reverted": False, "message": "no history entry available for rollback."}
    entry = history.pop()
    for adjustment in entry.get("adjustments", []):
        domain = adjustment["domain"]
        feature = adjustment["feature"]
        ledger["adjustments"][domain][feature] = adjustment["previous_adjustment"]
    return ledger, {"reverted": True, "message": "last feedback entry reverted deterministically."}
