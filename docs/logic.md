# Axiom-Recruit Logical Path

This document describes the deterministic logical path used by Axiom-Recruit to transform input profiles into a final hiring verdict.

## 1. Input and Validation

1. CLI receives `candidate`, `company`, and `market` JSON inputs (or batch candidate directory).
2. Files are loaded and validated for:
   - schema version
   - required feature keys
   - numeric bounds `[0, 1]`
   - domain-specific consistency rules
3. Optional runtime modes are applied:
   - `--self-evaluation-mode` replaces company input with a neutral-supportive company baseline
   - `--company-exposure-mode` adds explicit inefficiency diagnostics

## 2. Weight Resolution

1. Base feature weights come from `axiom_recruit/catalog.py`.
2. Optional adjustments are loaded from `weight_ledger.json`.
3. Effective feature weight = `base_weight + adjustment`, with a deterministic floor.
4. If `--ignore-weight-ledger` is used, scoring runs on baseline weights only.

## 3. Domain Scoring

1. Candidate score: weighted average across candidate features.
2. Company score: weighted average across company features.
3. Market score: weighted average across market features.
4. Per-feature contribution traces are recorded for auditability.

## 4. Cross-Domain Metrics

The engine computes cross-context metrics:

- candidate-company fit
- candidate-market alignment
- company-market mismatch
- candidate dominance index
- candidate excellence index
- market heat index
- opportunity loss risk
- acquisition pressure index
- hiring inefficiency score and drag
- decision latency impact and drag

## 5. Constraint and Contradiction Layer

1. Hard constraints are evaluated first (minimum thresholds).
2. Soft constraints are evaluated as additive penalties.
3. Contradictions are detected:
   - candidate contradictions contribute score penalty
   - company contradictions are flagged as risk signals

## 6. White-Box Neural Edge Adjustment

1. Derived metrics are passed to a deterministic white-box edge network (`whitebox_nn.py`).
2. Hidden nodes compute alignment, urgency, and instability terms.
3. Output is a bounded correction in `[-0.03, 0.03]`.

## 7. Final Score Assembly

1. Final score is assembled from:
   - weighted domain scores
   - cross-domain terms
   - derived metric terms
   - edge adjustment
   - soft and contradiction penalties
2. Grit multiplier is applied using `resource_advantage_index`.
3. Extreme-excellence bonus can apply.
4. If any hard constraint fails, score is capped and decision is forced to reject path.

## 8. Decision Mapping

Final decision is deterministic and tiered:

- `MISSION_CRITICAL_ACQUISITION`
- `MANDATORY_ACQUISITION`
- `STRONG_HIRE`
- `CONSIDER`
- `REJECT`

Thresholds depend on both score and key derived indices (not score alone).

## 9. Output and Trace

The engine writes a verdict JSON with:

- final decision and score
- logical trace steps
- proof-style deductions
- key factors and risks
- derived metrics
- white-box neural trace
- optional exposure/self-evaluation sections (mode-dependent)

This guarantees transparent, reproducible, and auditable hiring decisions.
