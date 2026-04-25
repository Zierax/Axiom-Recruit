# Axiom-Recruit Engine Logic & Deterministic Math

Axiom-Recruit is a white-box hiring audit engine designed for high-fidelity, reproducible decision-making. Unlike black-box AI systems, every decision is backed by explicit weighted calculations, hard constraints, and a deterministic correction layer.

## 1. Scoring Architecture

The engine operates on a multi-layered scoring hierarchy:

### Layer 1: Domain Scoring
Features are grouped into three primary domains: **Candidate**, **Company**, and **Market**. Each domain score is a weighted average of its internal features, validated against the `catalog.py` schema.

### Layer 2: Derived Metrics
The engine calculates higher-order heuristics that represent complex state:
- **Excellence Index**: Combines core candidate scores with innovation and real-world impact.
- **Dominance Index**: Measures the candidate's market-share potential.
- **Opportunity Loss Risk**: Evaluates the cost of failing to hire the candidate under current market conditions.

### Layer 3: The Grit Factor (Resource Ratio)
One of the most critical logic blocks is the **Achievement-to-Resource Ratio**. 
- **Logic**: We cannot compare a student at an elite Ivy League institution with an independent researcher in a developing economy using the same raw metric.
- **The Multiplier**: The engine applies a `Grit Factor` multiplier: `1.0 + (1.0 - resource_advantage_index) * 0.15`.
- **Impact**: High achievements with low resource access (Grit) are mathematically boosted, rewarding resilience and pure potential.

## 2. Constraints and Penalties

### Hard Constraints (HC)
Binary pass/fail checks (e.g., minimum psychological stability, technical standards). If any HC fails, the decision is capped at **REJECT** and the score is limited to 0.39.

### Soft Constraints (SC)
Linear penalties for suboptimal patterns (e.g., high bureaucracy, low consistency). These subtract from the score but do not force a rejection.

### Contradiction Detection
The engine looks for conflicting data points (e.g., "high innovation" vs "low documentation quality"). 
- **Candidate Contradictions**: Lower the score directly.
- **Company Contradictions**: Are flagged as risks but do not penalize the candidate's score.

## 3. White-box Neural Correction
A deterministic 3-node hidden layer hidden "neural" net provides fine-grained adjustments ([-0.03, 0.03]) based on non-linear patterns like "Urgency Pressure" or "Alignment Guards." Every weight and bias in this network is visible and auditable.

## 4. Decision Tiers

| Decision | Threshold | Description |
| :--- | :--- | :--- |
| **MISSION_CRITICAL_ACQUISITION** | 0.85+ | Elite talent with high Grit and Excellence. |
| **MANDATORY_ACQUISITION** | 0.76+ | Strong fit with high opportunity loss risk. |
| **STRONG_HIRE** | 0.62+ | High-quality candidate meeting all standards. |
| **CONSIDER** | 0.44+ | Potential fit requiring further evaluation. |
| **REJECT** | < 0.44 | Failed constraints or insufficient score. |
