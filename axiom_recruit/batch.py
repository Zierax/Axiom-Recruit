from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from .engine import EvaluationOptions, evaluate_hiring_decision
from .validation import InputValidationError, load_json, validate_candidate_payload


def evaluate_candidate_directory(
    candidates_directory: str | Path,
    company_payload: Mapping[str, Any] | None,
    market_payload: Mapping[str, Any],
    weight_adjustments: Mapping[str, Mapping[str, float]] | None = None,
    options: EvaluationOptions | None = None,
) -> Dict[str, Any]:
    root = Path(candidates_directory)
    if not root.exists() or not root.is_dir():
        raise InputValidationError(f"candidate directory not found: {root}")

    candidate_files = sorted(path for path in root.glob("*.json") if path.is_file())
    if not candidate_files:
        raise InputValidationError(f"candidate directory has no JSON files: {root}")

    results = []
    errors = []
    for file_path in candidate_files:
        try:
            candidate_payload = validate_candidate_payload(load_json(file_path))
            evaluation = evaluate_hiring_decision(
                candidate_payload=candidate_payload,
                company_payload=company_payload,
                market_payload=market_payload,
                weight_adjustments=weight_adjustments,
                options=options,
            )
            evaluation["candidate_id"] = candidate_payload["candidate_id"]
            evaluation["candidate_file"] = str(file_path)
            results.append(evaluation)
        except (InputValidationError, KeyError, ValueError) as exc:
            errors.append({"candidate_file": str(file_path), "error": str(exc)})

    ranked_results = sorted(results, key=lambda item: item["score"], reverse=True)
    summary = [
        {
            "rank": index + 1,
            "candidate_id": item["candidate_id"],
            "score": item["score"],
            "decision": item["final_decision"],
            "candidate_file": item["candidate_file"],
        }
        for index, item in enumerate(ranked_results)
    ]

    return {
        "mode": "batch",
        "evaluated_count": len(results),
        "failed_count": len(errors),
        "results": ranked_results,
        "summary": summary,
        "errors": errors,
    }
