from __future__ import annotations

import unittest
from typing import Dict, Any
from axiom_recruit.engine import evaluate_hiring_decision, EvaluationOptions
from axiom_recruit.validation import validate_candidate_payload, validate_company_payload, validate_market_payload
from axiom_recruit.templates import build_candidate_template, build_company_template, build_market_template
from axiom_recruit.feedback import apply_feedback, create_fresh_ledger

class AuditRobustnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidate = validate_candidate_payload(build_candidate_template())
        self.company = validate_company_payload(build_company_template())
        self.market = validate_market_payload(build_market_template())

    def test_latency_drag_market_heat_correlation(self) -> None:
        """Audit 3.2: Verify slowness is MORE dangerous in a hot market."""
        # Scenario 1: Low market heat
        low_heat_market = build_market_template()
        low_heat_market["features"]["demand_pressure"] = 0.1
        low_heat_market["features"]["time_to_hire_urgency"] = 0.1
        # Sync dynamic signals
        for k in low_heat_market["dynamic_signals"]:
            low_heat_market["dynamic_signals"][k] = low_heat_market["features"][k]
        low_heat_market = validate_market_payload(low_heat_market)
        
        # Scenario 2: High market heat
        high_heat_market = build_market_template()
        high_heat_market["features"]["demand_pressure"] = 0.9
        high_heat_market["features"]["time_to_hire_urgency"] = 0.9
        # Sync dynamic signals
        for k in high_heat_market["dynamic_signals"]:
            high_heat_market["dynamic_signals"][k] = high_heat_market["features"][k]
        high_heat_market = validate_market_payload(high_heat_market)
        
        # Constant company with high latency
        slow_company = build_company_template()
        slow_company["features"]["hiring_latency"] = 0.8
        slow_company = validate_company_payload(slow_company)
        
        res_low = evaluate_hiring_decision(self.candidate, slow_company, low_heat_market)
        res_high = evaluate_hiring_decision(self.candidate, slow_company, high_heat_market)
        
        drag_low = res_low["derived_metrics"]["decision_latency_drag"]
        drag_high = res_high["derived_metrics"]["decision_latency_drag"]
        
        # In a high heat market, the penalty (drag) should be higher
        self.assertGreater(drag_high, drag_low, "Latency drag should increase with market heat")

    def test_contradiction_attribution(self) -> None:
        """Audit 3.3: Verify company contradictions don't penalize candidate score as much as candidate contradictions."""
        # Base case
        base_res = evaluate_hiring_decision(self.candidate, self.company, self.market)
        
        # Case 1: Company contradiction (Innovation high, Latency high)
        conflicted_company = build_company_template()
        conflicted_company["features"]["innovation_index"] = 0.9
        conflicted_company["features"]["hiring_latency"] = 0.9
        conflicted_company = validate_company_payload(conflicted_company)
        
        res_company_conflict = evaluate_hiring_decision(self.candidate, conflicted_company, self.market)
        
        # Case 2: Candidate contradiction (Innovation high, Documentation low)
        conflicted_candidate = build_candidate_template()
        conflicted_candidate["features"]["innovation_index"] = 0.9
        conflicted_candidate["features"]["documentation_quality"] = 0.1
        conflicted_candidate = validate_candidate_payload(conflicted_candidate)
        
        res_candidate_conflict = evaluate_hiring_decision(conflicted_candidate, self.company, self.market)
        
        # Contradiction penalty should be applied to candidate conflict
        self.assertGreater(res_candidate_conflict["derived_metrics"]["contradiction_penalty_adjusted"], 0.0)
        
        # Check if score reduction is primarily from candidate
        # (Note: score will drop for company conflict too because of latency itself, but not the contradiction penalty)
        self.assertEqual(res_company_conflict["derived_metrics"]["contradiction_penalty_adjusted"], 0.0, 
                         "Company contradictions should not trigger candidate score penalty")

    def test_feedback_convergence_speed(self) -> None:
        """Audit 3.4: Verify feedback results in faster convergence (delta > 0.01)."""
        ledger = create_fresh_ledger()
        
        # Result with high score
        result = evaluate_hiring_decision(self.candidate, self.company, self.market)
        score = result["score"]
        
        # Give a very low rating (1) to create maximum divergence
        # Rating 1 -> normalized 0.0. Divergence = 0.0 - score
        ledger, summary = apply_feedback(
            ledger=ledger,
            rating=1,
            score=score,
            contribution_ranking=result["_internal"]["contribution_ranking"],
            reason="test"
        )
        
        # Before it was capped at 0.01. Now it should be up to 0.03
        self.assertGreater(abs(summary["bounded_delta"]), 0.01, "Feedback convergence should be faster than 0.01")
        self.assertLessEqual(abs(summary["bounded_delta"]), 0.03)

    def test_hc1_double_jeopardy(self) -> None:
        """Audit 3.7: Verify HC-1 threshold and SC-1 interaction."""
        # Case 1: Just below HC-1 (0.45)
        bad_candidate = build_candidate_template()
        bad_candidate["features"]["consistency_score"] = 0.44
        bad_candidate = validate_candidate_payload(bad_candidate)
        res_fail = evaluate_hiring_decision(bad_candidate, self.company, self.market)
        self.assertEqual(res_fail["final_decision"], "REJECT")
        
        # Case 2: Just above HC-1 (0.45) but below SC-1 (0.55)
        marginal_candidate = build_candidate_template()
        marginal_candidate["features"]["consistency_score"] = 0.46
        marginal_candidate = validate_candidate_payload(marginal_candidate)
        res_pass = evaluate_hiring_decision(marginal_candidate, self.company, self.market)
        
        # Should pass HC but have SC penalty
        hc1 = next(c for c in res_pass["hard_constraints"] if c["id"] == "HC-1")
        sc1 = next(c for c in res_pass["soft_constraints"] if c["id"] == "SC-1")
        
        self.assertTrue(hc1["satisfied"])
        self.assertTrue(sc1["triggered"])
        self.assertGreater(res_pass["derived_metrics"]["soft_penalty_adjusted"], 0.0)

    def test_self_evaluation_neutrality_boost(self) -> None:
        """Audit 3.6: Verify self-evaluation uses supportive neutral values."""
        # Self evaluation mode
        res_self = evaluate_hiring_decision(self.candidate, None, self.market, options=EvaluationOptions(self_evaluation_mode=True))
        
        # Manual evaluation with a truly "mediocre" (0.5 everywhere) company
        mediocre_company = {
            "schema_version": "1.0",
            "company_id": "mediocre",
            "features": {f: 0.5 for f in self.company["features"]}
        }
        res_mediocre = evaluate_hiring_decision(self.candidate, mediocre_company, self.market)
        
        # Self evaluation should be better because we boosted neutral company features
        self.assertGreater(res_self["score"], res_mediocre["score"], "Self-evaluation should benefit from supportive neutral values")

    def test_mandatory_acquisition_balance(self) -> None:
        """Audit 3.5: Verify top-tier acquisition outcomes are reachable with high scores."""
        # Create a near-perfect candidate
        top_candidate = build_candidate_template()
        for f in top_candidate["features"]:
            top_candidate["features"][f] = 0.95
        top_candidate = validate_candidate_payload(top_candidate)
        
        # Create a matching company
        top_company = build_company_template()
        for f in top_company["features"]:
            top_company["features"][f] = 0.9
        # Latency should be low
        top_company["features"]["hiring_latency"] = 0.1
        top_company = validate_company_payload(top_company)
        
        res = evaluate_hiring_decision(top_candidate, top_company, self.market)
        
        self.assertIn(res["final_decision"], {"MANDATORY_ACQUISITION", "MISSION_CRITICAL_ACQUISITION"})
        self.assertGreaterEqual(res["score"], 0.8)

if __name__ == "__main__":
    unittest.main()
