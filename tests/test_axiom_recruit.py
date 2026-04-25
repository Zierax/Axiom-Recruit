from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict

from axiom_recruit.batch import evaluate_candidate_directory
from axiom_recruit.catalog import (
    CANDIDATE_FEATURE_SPECS,
    COMPANY_FEATURE_SPECS,
    FEATURE_COUNT_SUMMARY,
    MARKET_DYNAMIC_SIGNAL_KEYS,
    MARKET_FEATURE_SPECS,
)
from axiom_recruit.engine import EvaluationOptions, evaluate_hiring_decision
from axiom_recruit.feedback import (
    apply_feedback,
    create_fresh_ledger,
    load_weight_ledger,
    revert_last_feedback,
)
from axiom_recruit.profile_library import (
    CANDIDATE_LEVELS,
    COMPANY_NAMES,
    COUNTRIES,
    MARKET_SCENARIOS,
    build_candidate_level_profile,
    build_company_real_world_profile,
    build_main_ziad_profile,
    build_market_profile,
    write_profile_library,
)
from axiom_recruit.prompt_generator import (
    build_candidate_extraction_prompt,
    build_company_extraction_prompt,
    build_market_extraction_prompt,
)
from axiom_recruit.templates import build_candidate_template, build_company_template, build_market_template
from axiom_recruit.validation import (
    InputValidationError,
    validate_candidate_payload,
    validate_company_payload,
    validate_market_payload,
)


DECISIONS = {
    "REJECT",
    "CONSIDER",
    "STRONG_HIRE",
    "MANDATORY_ACQUISITION",
    "MISSION_CRITICAL_ACQUISITION",
}


def _deterministic_value(seed: int, index: int, minimum: float = 0.0, maximum: float = 1.0) -> float:
    span = maximum - minimum
    bucket = ((seed * 104729 + index * 130363 + 17) % 1000) / 999
    return minimum + span * bucket


def _make_candidate(seed: int) -> Dict[str, object]:
    features: Dict[str, float] = {}
    for index, feature_name in enumerate(CANDIDATE_FEATURE_SPECS, start=1):
        features[feature_name] = _deterministic_value(seed, index)
    features["consistency_score"] = max(features["consistency_score"], 0.40)
    features["psychological_stability"] = max(features["psychological_stability"], 0.35)
    features["execution_history"] = max(features["execution_history"], 0.30)
    features["failure_recovery_capability"] = max(features["failure_recovery_capability"], 0.25)
    features["system_design_depth"] = max(features["system_design_depth"], 0.25)
    features["documentation_quality"] = max(features["documentation_quality"], 0.32)
    features["project_complexity"] = min(features["project_complexity"], 0.90)
    return {
        "schema_version": "1.0",
        "candidate_id": f"fuzz-candidate-{seed}",
        "features": features,
    }


def _make_company(seed: int) -> Dict[str, object]:
    features: Dict[str, float] = {}
    for index, feature_name in enumerate(COMPANY_FEATURE_SPECS, start=1):
        features[feature_name] = _deterministic_value(seed + 97, index)
    features["engineering_depth"] = max(features["engineering_depth"], 0.20)
    features["technical_standards_strictness"] = max(features["technical_standards_strictness"], 0.25)
    features["recruiter_response_speed"] = max(features["recruiter_response_speed"], 0.21)
    features["budget_flexibility"] = max(features["budget_flexibility"], 0.11)
    if features["hiring_urgency"] > 0.85 and features["hiring_latency"] > 0.90:
        features["recruiter_response_speed"] = max(features["recruiter_response_speed"], 0.22)
    return {
        "schema_version": "1.0",
        "company_id": f"fuzz-company-{seed}",
        "features": features,
    }


def _make_market(seed: int) -> Dict[str, object]:
    features: Dict[str, float] = {}
    for index, feature_name in enumerate(MARKET_FEATURE_SPECS, start=1):
        features[feature_name] = _deterministic_value(seed + 193, index)
    features["economic_environment"] = max(features["economic_environment"], 0.10)
    if features["time_to_hire_urgency"] > 0.85 and features["demand_pressure"] < 0.30 and features["cost_of_delay"] < 0.20:
        features["cost_of_delay"] = 0.22
    if features["demand_pressure"] < 0.20 and features["talent_scarcity"] > 0.90 and features["competition_density"] < 0.20:
        features["competition_density"] = 0.22
    dynamic_signals = {key: features[key] for key in MARKET_DYNAMIC_SIGNAL_KEYS}
    return {
        "schema_version": "1.0",
        "market_id": f"fuzz-market-{seed}",
        "features": features,
        "dynamic_signals": dynamic_signals,
    }


class AxiomRecruitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidate = validate_candidate_payload(build_candidate_template())
        self.company = validate_company_payload(build_company_template())
        self.market = validate_market_payload(build_market_template())

    def test_feature_counts(self) -> None:
        self.assertGreaterEqual(FEATURE_COUNT_SUMMARY["candidate"], 100)
        self.assertGreaterEqual(FEATURE_COUNT_SUMMARY["company"], 200)

    def test_validation_rejects_invalid_bounds(self) -> None:
        bad_candidate = build_candidate_template()
        bad_candidate["features"]["project_complexity"] = 1.5
        with self.assertRaises(InputValidationError):
            validate_candidate_payload(bad_candidate)

    def test_validation_rejects_boolean_as_numeric(self) -> None:
        bad_company = build_company_template()
        bad_company["features"]["engineering_depth"] = True
        with self.assertRaises(InputValidationError):
            validate_company_payload(bad_company)

    def test_validation_rejects_dynamic_signal_mismatch(self) -> None:
        bad_market = build_market_template()
        bad_market["dynamic_signals"]["demand_pressure"] = 0.1
        with self.assertRaises(InputValidationError):
            validate_market_payload(bad_market)

    def test_evaluation_is_deterministic(self) -> None:
        result_one = evaluate_hiring_decision(
            candidate_payload=self.candidate,
            company_payload=self.company,
            market_payload=self.market,
            options=EvaluationOptions(explain_level="full", company_exposure_mode=True),
        )
        result_two = evaluate_hiring_decision(
            candidate_payload=self.candidate,
            company_payload=self.company,
            market_payload=self.market,
            options=EvaluationOptions(explain_level="full", company_exposure_mode=True),
        )
        self.assertEqual(result_one["final_decision"], result_two["final_decision"])
        self.assertEqual(result_one["score"], result_two["score"])
        self.assertEqual(result_one["derived_metrics"], result_two["derived_metrics"])

    def test_default_candidate_profile_snapshot_exists(self) -> None:
        template = build_candidate_template()
        self.assertEqual(template["candidate_id"], "ziad-salah-default-profile")
        profile = template.get("default_candidate_profile")
        self.assertIsInstance(profile, dict)
        self.assertEqual(profile.get("name"), "Ziad Salah")

    def test_split_prompts_are_domain_specific(self) -> None:
        candidate_prompt = build_candidate_extraction_prompt()
        company_prompt = build_company_extraction_prompt()
        market_prompt = build_market_extraction_prompt()
        self.assertIn("Candidate Profile Mode", candidate_prompt)
        self.assertIn("Never emit company fields.", candidate_prompt)
        self.assertIn("Company Profile Mode", company_prompt)
        self.assertIn("Never emit candidate fields.", company_prompt)
        self.assertIn("Market Profile Mode", market_prompt)
        self.assertIn("dynamic_signals", market_prompt)

    def test_output_shape_contains_required_fields(self) -> None:
        result = evaluate_hiring_decision(
            candidate_payload=self.candidate,
            company_payload=self.company,
            market_payload=self.market,
            options=EvaluationOptions(explain_level="full"),
        )
        required_keys = {
            "final_decision",
            "score",
            "confidence",
            "key_factors",
            "risks",
            "inefficiencies",
            "contradictions",
            "logical_trace",
            "affective_signals",
            "proof",
            "derived_metrics",
            "cross_domain_analysis",
            "whitebox_neural_trace",
        }
        self.assertTrue(required_keys.issubset(set(result.keys())))
        self.assertIn(result["final_decision"], DECISIONS)
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)
        self.assertEqual(result["whitebox_neural_trace"]["model"], "whitebox_edge_net_v1")

    def test_feedback_is_bounded_and_reversible(self) -> None:
        ledger = create_fresh_ledger()
        result = evaluate_hiring_decision(
            candidate_payload=self.candidate,
            company_payload=self.company,
            market_payload=self.market,
            weight_adjustments=ledger["adjustments"],
            options=EvaluationOptions(explain_level="summary"),
        )
        contribution_ranking = result["_internal"]["contribution_ranking"]
        ledger, feedback_summary = apply_feedback(
            ledger=ledger,
            rating=9,
            score=float(result["score"]),
            contribution_ranking=contribution_ranking,
            reason="validation",
        )
        self.assertGreaterEqual(feedback_summary["adjustment_count"], 1)
        self.assertEqual(len(ledger["history"]), 1)
        last_entry = ledger["history"][-1]
        for adjustment in last_entry["adjustments"]:
            self.assertLessEqual(abs(adjustment["updated_adjustment"]), adjustment["bound"] + 1e-9)
        ledger, revert_summary = revert_last_feedback(ledger)
        self.assertTrue(revert_summary["reverted"])
        self.assertEqual(len(ledger["history"]), 0)

    def test_malformed_weight_ledger_falls_back_to_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            ledger_path = Path(temp_directory) / "weight_ledger.json"
            ledger_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "adjustments": {
                            "candidate": {"project_complexity": "bad-type"},
                        },
                        "history": "not-a-list",
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_weight_ledger(ledger_path)
            self.assertEqual(loaded["schema_version"], "1.0")
            self.assertEqual(loaded["history"], [])
            self.assertEqual(loaded["adjustments"]["candidate"]["project_complexity"], 0.0)

    def test_main_candidate_now_scores_high(self) -> None:
        result = evaluate_hiring_decision(
            candidate_payload=self.candidate,
            company_payload=self.company,
            market_payload=self.market,
            options=EvaluationOptions(explain_level="summary"),
        )
        self.assertGreaterEqual(result["score"], 0.70)
        self.assertIn(
            result["final_decision"],
            {"STRONG_HIRE", "MANDATORY_ACQUISITION", "MISSION_CRITICAL_ACQUISITION"},
        )

    def test_candidate_levels_are_demo_joe_doe(self) -> None:
        for level_index in range(1, 6):
            candidate = build_candidate_level_profile(level_index)
            self.assertTrue(candidate["candidate_id"].startswith("demo_joe_doe_level_"))
            self.assertEqual(candidate["candidate_profile"]["name"], "Joe Doe")
        main_profile = build_main_ziad_profile()
        self.assertEqual(main_profile["candidate_id"], "main_ziad_salah")

    def test_candidate_levels_monotonic_against_fixed_context(self) -> None:
        market = validate_market_payload(build_market_profile("usa", "ai-current"))
        company = validate_company_payload(build_company_real_world_profile("google"))
        scores = []
        for level_index in range(1, 6):
            candidate = validate_candidate_payload(build_candidate_level_profile(level_index))
            result = evaluate_hiring_decision(candidate, company, market, options=EvaluationOptions(explain_level="summary"))
            scores.append(result["score"])
        self.assertEqual(scores, sorted(scores))

    def test_real_world_company_profiles_score_order(self) -> None:
        market = validate_market_payload(build_market_profile("singapore", "security-current"))
        candidate = validate_candidate_payload(build_main_ziad_profile())
        scored = []
        for company_name in COMPANY_NAMES:
            company = validate_company_payload(build_company_real_world_profile(company_name))
            result = evaluate_hiring_decision(candidate, company, market, options=EvaluationOptions(explain_level="summary"))
            scored.append(result["score"])
        self.assertTrue(all(0.0 <= value <= 1.0 for value in scored))

    def test_real_world_company_profiles_validate(self) -> None:
        for company_name in COMPANY_NAMES:
            payload = build_company_real_world_profile(company_name)
            validated = validate_company_payload(payload)
            self.assertEqual(payload["company_name"], company_name)
            self.assertIn("real_world_context", payload)
            self.assertEqual(validated["schema_version"], "1.0")

    def test_market_realism_for_egypt_ml_2016(self) -> None:
        payload = build_market_profile("egypt", "ml-2016")
        self.assertLess(payload["features"]["competition_density"], 0.25)
        self.assertLess(payload["features"]["ai_adoption_velocity"], 0.25)
        self.assertLess(payload["features"]["demand_pressure"], 0.35)
        validate_market_payload(payload)

    def test_market_pressure_ordering(self) -> None:
        candidate = validate_candidate_payload(build_main_ziad_profile())
        company = validate_company_payload(build_company_real_world_profile("coveo"))
        low_market = validate_market_payload(build_market_profile("canada", "ml-2016"))
        high_market = validate_market_payload(build_market_profile("canada", "ai-current"))

        low_result = evaluate_hiring_decision(candidate, company, low_market, options=EvaluationOptions(explain_level="summary"))
        high_result = evaluate_hiring_decision(candidate, company, high_market, options=EvaluationOptions(explain_level="summary"))

        self.assertGreaterEqual(
            high_result["derived_metrics"]["opportunity_loss_risk"],
            low_result["derived_metrics"]["opportunity_loss_risk"],
        )
        self.assertGreaterEqual(
            high_result["derived_metrics"]["acquisition_pressure_index"],
            low_result["derived_metrics"]["acquisition_pressure_index"],
        )

    def test_profile_library_generation_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            generated = write_profile_library(root)

            self.assertEqual(len(generated["candidates"]), 6)
            self.assertEqual(len(generated["companies"]), 5)
            self.assertEqual(len(generated["markets"]), 15)

            manifest = json.loads((root / "profiles" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["candidates"]), 6)
            self.assertEqual(len(manifest["companies"]), 5)
            self.assertEqual(len(manifest["markets"]), 15)

            for relative_path in manifest["candidates"]:
                payload = json.loads((root / relative_path).read_text(encoding="utf-8"))
                validate_candidate_payload(payload)
            for relative_path in manifest["companies"]:
                payload = json.loads((root / relative_path).read_text(encoding="utf-8"))
                validate_company_payload(payload)
            for relative_path in manifest["markets"]:
                payload = json.loads((root / relative_path).read_text(encoding="utf-8"))
                validate_market_payload(payload)

            self.assertTrue((root / "profiles" / "candidates" / "main_ziad_salah.json").exists())
            self.assertTrue((root / "profiles" / "companies" / "meta" / "real_world_profile.json").exists())
            self.assertFalse((root / "profiles" / "companies" / "meta" / "level_1.json").exists())

    def test_batch_folder_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            candidate_dir = root / "candidates"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            for level_index in range(1, 4):
                payload = build_candidate_level_profile(level_index)
                (candidate_dir / f"candidate_{level_index}.json").write_text(
                    json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            bad_payload = build_candidate_level_profile(2)
            bad_payload["features"]["project_complexity"] = 1.8
            (candidate_dir / "bad_candidate.json").write_text(
                json.dumps(bad_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            company = validate_company_payload(build_company_real_world_profile("google"))
            market = validate_market_payload(build_market_profile("usa", "ai-current"))
            report = evaluate_candidate_directory(
                candidates_directory=candidate_dir,
                company_payload=company,
                market_payload=market,
                options=EvaluationOptions(explain_level="summary"),
            )
            self.assertEqual(report["evaluated_count"], 3)
            self.assertEqual(report["failed_count"], 1)
            self.assertEqual(len(report["summary"]), 3)
            self.assertEqual(report["summary"][0]["rank"], 1)

    def test_profile_library_expected_dimensions(self) -> None:
        self.assertEqual(len(CANDIDATE_LEVELS), 5)
        self.assertEqual(len(COMPANY_NAMES), 5)
        self.assertEqual(len(COUNTRIES), 5)
        self.assertGreaterEqual(len(MARKET_SCENARIOS), 3)

    def test_deterministic_fuzz_sweep_edge_cases(self) -> None:
        for seed in range(1, 151):
            candidate_payload = validate_candidate_payload(_make_candidate(seed))
            company_payload = validate_company_payload(_make_company(seed))
            market_payload = validate_market_payload(_make_market(seed))

            result_one = evaluate_hiring_decision(
                candidate_payload=candidate_payload,
                company_payload=company_payload,
                market_payload=market_payload,
                options=EvaluationOptions(explain_level="summary"),
            )
            result_two = evaluate_hiring_decision(
                candidate_payload=candidate_payload,
                company_payload=company_payload,
                market_payload=market_payload,
                options=EvaluationOptions(explain_level="summary"),
            )

            self.assertEqual(result_one["final_decision"], result_two["final_decision"])
            self.assertEqual(result_one["score"], result_two["score"])
            self.assertIn(result_one["final_decision"], DECISIONS)
            self.assertGreaterEqual(result_one["score"], 0.0)
            self.assertLessEqual(result_one["score"], 1.0)

            derived = result_one["derived_metrics"]
            self.assertGreaterEqual(derived["opportunity_loss_risk"], 0.0)
            self.assertLessEqual(derived["opportunity_loss_risk"], 1.0)
            self.assertGreaterEqual(derived["acquisition_pressure_index"], 0.0)
            self.assertLessEqual(derived["acquisition_pressure_index"], 1.0)
            self.assertGreaterEqual(derived["hiring_inefficiency_drag"], 0.0)
            self.assertLessEqual(derived["hiring_inefficiency_drag"], 1.0)


if __name__ == "__main__":
    unittest.main()
