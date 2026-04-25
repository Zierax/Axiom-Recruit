#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Dict

from axiom_recruit.batch import evaluate_candidate_directory
from axiom_recruit.engine import EvaluationOptions, evaluate_hiring_decision
from axiom_recruit.feedback import apply_feedback, load_weight_ledger, revert_last_feedback, save_weight_ledger
from axiom_recruit.prompt_generator import (
    build_candidate_extraction_prompt,
    build_company_extraction_prompt,
    build_external_ai_prompt,
    build_market_extraction_prompt,
)
from axiom_recruit.profile_library import write_profile_library
from axiom_recruit.templates import write_templates
from axiom_recruit.validation import (
    InputValidationError,
    load_json,
    validate_candidate_payload,
    validate_company_payload,
    validate_market_payload,
)


def _build_parser() -> argparse.ArgumentParser:
    description = (
        "Axiom-Recruit: deterministic, white-box, reproducible hiring audit engine.\n"
        "This system uses explicit weighted math, hard constraints, soft constraints,\n"
        "cross-domain analysis, contradiction detection, and proof-like trace output."
    )
    epilog = (
        "Input format guide:\n"
        "  candidate.json -> {schema_version, candidate_id, features{...}}\n"
        "  company.json   -> {schema_version, company_id, features{...}}\n"
        "  market.json    -> {schema_version, market_id, features{...}, dynamic_signals{...}}\n"
        "  All feature values must be numeric in [0,1].\n\n"
        "Example usage:\n"
        "  python axiom-recruit.py --candidate candidate.json --company company.json --market market.json --output verdict.json --explain full\n"
        "  python axiom-recruit.py --candidates-dir profiles/candidates --company profiles/companies/google/real_world_profile.json --market profiles/markets/usa/ai-current.json --output batch_verdict.json\n"
        "  python axiom-recruit.py --generate-input-templates\n"
        "  python axiom-recruit.py --generate-profiles-library\n"
        "  python axiom-recruit.py --generate-candidate-prompt\n"
        "  python axiom-recruit.py --generate-company-prompt\n"
        "  python axiom-recruit.py --generate-ai-prompt\n"
        "  python axiom-recruit.py --candidate candidate.json --company company.json --market market.json --rating 8 --feedback-reason \"strong hire expected\"\n\n"
        "AI prompt generator instructions:\n"
        "  --generate-candidate-prompt prints prompt #1 for candidate profile extraction.\n"
        "  --generate-company-prompt prints prompt #2 for company profile extraction.\n"
        "  --generate-market-prompt prints prompt #3 for market profile extraction.\n"
        "  --generate-profiles-library creates /profiles with level-based candidates, one real-world profile per company, and market scenarios.\n"
        "  --generate-ai-prompt prints the full prompt bundle."
    )
    parser = argparse.ArgumentParser(
        prog="axiom-recruit.py",
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--candidate", help="Path to candidate.json input.")
    parser.add_argument("--candidates-dir", help="Path to folder of candidate JSON files for batch evaluation.")
    parser.add_argument("--company", help="Path to company.json input.")
    parser.add_argument("--market", help="Path to market.json input.")
    parser.add_argument("--output", default="verdict.json", help="Path to output verdict JSON file.")
    parser.add_argument("--explain", choices=("summary", "full"), default="summary", help="Reasoning trace depth.")
    parser.add_argument("--interactive", action="store_true", help="Prompt for missing inputs and optional feedback rating.")
    parser.add_argument("--generate-ai-prompt", action="store_true", help="Print external AI extraction prompt.")
    parser.add_argument("--generate-candidate-prompt", action="store_true", help="Print candidate-only extraction prompt.")
    parser.add_argument("--generate-company-prompt", action="store_true", help="Print company-only extraction prompt.")
    parser.add_argument("--generate-market-prompt", action="store_true", help="Print market-only extraction prompt.")
    parser.add_argument("--generate-input-templates", action="store_true", help="Generate candidate.json, company.json, market.json templates in current directory.")
    parser.add_argument("--generate-profiles-library", action="store_true", help="Generate deterministic /profiles library with levels and country-market scenarios.")
    parser.add_argument("--company-exposure-mode", action="store_true", help="Expose deterministic company inefficiency diagnostics.")
    parser.add_argument("--self-evaluation-mode", action="store_true", help="Evaluate candidate readiness against market using neutral company profile.")
    parser.add_argument("--rating", type=int, help="Feedback rating from 1 to 10.")
    parser.add_argument("--feedback-reason", default="", help="Deterministic explanation for rating disagreement.")
    parser.add_argument("--weight-ledger", default="weight_ledger.json", help="Path to reversible weight adjustment ledger.")
    parser.add_argument(
        "--ignore-weight-ledger",
        action="store_true",
        help="Ignore stored feedback adjustments and evaluate using baseline weights only.",
    )
    parser.add_argument("--revert-last-feedback", action="store_true", help="Rollback most recent feedback adjustment entry.")
    return parser


def _strip_internal_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(result)
    sanitized.pop("_internal", None)
    return sanitized


def _maybe_prompt(prompt_text: str, current_value: str | None) -> str | None:
    if current_value:
        return current_value
    answer = input(prompt_text).strip()
    return answer or None


def _count_non_zero_adjustments(adjustments: Dict[str, Dict[str, float]]) -> int:
    count = 0
    for domain_adjustments in adjustments.values():
        for value in domain_adjustments.values():
            if abs(float(value)) > 1e-12:
                count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.generate_candidate_prompt:
        print(build_candidate_extraction_prompt())
    if args.generate_company_prompt:
        print(build_company_extraction_prompt())
    if args.generate_market_prompt:
        print(build_market_extraction_prompt())
    if args.generate_ai_prompt:
        print(build_external_ai_prompt())

    generated_prompt_only = any(
        [
            args.generate_ai_prompt,
            args.generate_candidate_prompt,
            args.generate_company_prompt,
            args.generate_market_prompt,
        ]
    )
    if generated_prompt_only and not any(
        [
            args.candidate,
            args.company,
            args.market,
            args.interactive,
            args.generate_input_templates,
            args.generate_profiles_library,
        ]
    ):
        return 0

    if args.generate_input_templates:
        paths = write_templates(Path.cwd())
        print(f"Generated templates: {paths['candidate']}, {paths['company']}, {paths['market']}")
        if not any([args.candidate, args.company, args.market, args.interactive, args.generate_profiles_library]):
            return 0

    if args.generate_profiles_library:
        profile_paths = write_profile_library(Path.cwd())
        print(
            "Generated profile library: "
            f"{len(profile_paths['candidates'])} candidate files, "
            f"{len(profile_paths['companies'])} company files, "
            f"{len(profile_paths['markets'])} market files."
        )
        if not any([args.candidate, args.company, args.market, args.interactive]):
            return 0

    ledger = load_weight_ledger(args.weight_ledger)
    if args.revert_last_feedback:
        ledger, rollback_summary = revert_last_feedback(ledger)
        save_weight_ledger(args.weight_ledger, ledger)
        print(rollback_summary["message"])
        if not any([args.candidate, args.company, args.market, args.interactive]):
            return 0

    ledger_adjustments = ledger.get("adjustments", {})
    non_zero_adjustment_count = _count_non_zero_adjustments(ledger_adjustments)
    if args.ignore_weight_ledger:
        active_adjustments = {}
        if non_zero_adjustment_count:
            print(
                "WeightLedger=IGNORED "
                f"NonZeroAdjustments={non_zero_adjustment_count} "
                f"HistoryEntries={len(ledger.get('history', []))}"
            )
    else:
        active_adjustments = ledger_adjustments
        if non_zero_adjustment_count:
            print(
                "WeightLedger=ACTIVE "
                f"NonZeroAdjustments={non_zero_adjustment_count} "
                f"HistoryEntries={len(ledger.get('history', []))}"
            )

    candidate_path = args.candidate
    candidates_dir = args.candidates_dir
    company_path = args.company
    market_path = args.market
    if args.interactive:
        candidate_path = _maybe_prompt("Candidate JSON path: ", candidate_path)
        if not args.self_evaluation_mode:
            company_path = _maybe_prompt("Company JSON path: ", company_path)
        market_path = _maybe_prompt("Market JSON path: ", market_path)

    if candidate_path and candidates_dir:
        parser.error("--candidate and --candidates-dir are mutually exclusive.")
    if not candidate_path and not candidates_dir:
        parser.error("--candidate or --candidates-dir is required unless using prompt/template/profile generation only.")
    if not market_path:
        parser.error("--market is required for evaluation mode.")
    if not args.self_evaluation_mode and not company_path:
        parser.error("--company is required for evaluation mode unless --self-evaluation-mode is enabled.")

    try:
        company_payload = None if args.self_evaluation_mode else validate_company_payload(load_json(company_path))
        market_payload = validate_market_payload(load_json(market_path))
    except InputValidationError as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        return 2

    evaluation_options = EvaluationOptions(
        explain_level=args.explain,
        company_exposure_mode=args.company_exposure_mode,
        self_evaluation_mode=args.self_evaluation_mode,
    )

    start = perf_counter()
    if candidates_dir:
        try:
            result = evaluate_candidate_directory(
                candidates_directory=candidates_dir,
                company_payload=company_payload,
                market_payload=market_payload,
                weight_adjustments=active_adjustments,
                options=evaluation_options,
            )
        except InputValidationError as exc:
            print(f"Validation error: {exc}", file=sys.stderr)
            return 2
    else:
        try:
            candidate_payload = validate_candidate_payload(load_json(candidate_path))
        except InputValidationError as exc:
            print(f"Validation error: {exc}", file=sys.stderr)
            return 2
        result = evaluate_hiring_decision(
            candidate_payload=candidate_payload,
            company_payload=company_payload,
            market_payload=market_payload,
            weight_adjustments=active_adjustments,
            options=evaluation_options,
        )
    elapsed_ms = (perf_counter() - start) * 1000
    result["performance"] = {"execution_ms": round(elapsed_ms, 6)}

    rating = args.rating
    feedback_reason = args.feedback_reason
    if args.interactive and rating is None:
        raw_rating = input("Feedback rating 1-10 (optional, press enter to skip): ").strip()
        if raw_rating:
            rating = int(raw_rating)
            feedback_reason = input("Feedback reason (optional): ").strip()

    if rating is not None and not candidates_dir:
        internal = result.get("_internal", {})
        contribution_ranking = internal.get("contribution_ranking", [])
        try:
            ledger, feedback_summary = apply_feedback(
                ledger=ledger,
                rating=rating,
                score=float(result["score"]),
                contribution_ranking=contribution_ranking,
                reason=feedback_reason,
            )
        except ValueError as exc:
            print(f"Feedback error: {exc}", file=sys.stderr)
            return 2
        save_weight_ledger(args.weight_ledger, ledger)
        result["feedback"] = feedback_summary
    elif rating is not None and candidates_dir:
        print("Feedback update skipped in batch mode (requires single-candidate evaluation).")

    if candidates_dir:
        output_payload = dict(result)
        cleaned_results = []
        for item in result["results"]:
            cleaned_results.append(_strip_internal_fields(item))
        output_payload["results"] = cleaned_results
    else:
        output_payload = _strip_internal_fields(result)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(output_payload, indent=2, sort_keys=True), encoding="utf-8")
    if candidates_dir:
        print(
            f"BatchEvaluated={output_payload['evaluated_count']} "
            f"Failed={output_payload['failed_count']} "
            f"ExecutionMs={output_payload['performance']['execution_ms']:.3f}"
        )
        if output_payload["summary"]:
            best = output_payload["summary"][0]
            print(
                f"TopCandidate={best['candidate_id']} Score={best['score']:.6f} Decision={best['decision']}"
            )
    else:
        print(
            f"Decision={output_payload['final_decision']} Score={output_payload['score']:.6f} "
            f"ExecutionMs={output_payload['performance']['execution_ms']:.3f}"
        )
    print(f"Verdict written to {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
