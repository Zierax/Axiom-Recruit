# Axiom-Recruit

**Deterministic, White-box, Reproducible Hiring Audit Engine.**

Axiom-Recruit is a senior-level logic engine designed to evaluate hiring decisions through the lens of mathematical excellence, market alignment, and the "Grit Factor." It eliminates black-box bias by providing a full, auditable logic trace for every hiring verdict.

## ًںڑ€ Key Features

- **Grit Factor (Resource Ratio)**: Automatically boosts candidates who achieve high technical excellence in low-resource environments (e.g., Egypt vs. Silicon Valley).
- **Multi-Layer Logic**: Combines Domain Scoring, Derived Heuristics, and Deterministic Constraint checking.
- **Deterministic Trace**: Every decision generates a `verdict.json` containing the full math, proofs, and a white-box neural correction trace.
- **Dynamic Market Scenarios**: Model decisions across different countries (Egypt, USA, Canada, etc.) and technology eras (AI-Current, ML-2016).
- **AI Extraction Prompts**: Built-in utilities to extract deterministic feature vectors from unstructured CVs and company descriptions.

## ًں›  Installation

```bash
git clone https://github.com/Zierax/Axiom-Recruit
cd Axiom-Recruit
# No heavy dependencies; standard Python 3.10+
```

## ًں“– Usage

### Standard Evaluation
Evaluate a candidate against a company and market:
```bash
python axiom-recruit.py --candidate candidate.json --company profiles/companies/coveo/real_world_profile.json --market profiles/markets/canada/ai-current.json
```

Run with baseline weights only (ignore saved feedback adjustments in `weight_ledger.json`):
```bash
python axiom-recruit.py --candidate candidate.json --company profiles/companies/coveo/real_world_profile.json --market profiles/markets/canada/ai-current.json --ignore-weight-ledger
```

### Self-Evaluation Mode
Evaluate your own readiness against a specific market using a neutral company baseline:
```bash
python axiom-recruit.py --candidate candidate.json --market profiles/markets/egypt/ai-current.json --self-evaluation-mode
```

### Generating the Profile Library
Initialize or update the `/profiles` folder with synthetic levels and real-world company data:
```bash
python axiom-recruit.py --generate-profiles-library
```

## ًں“‚ Documentation

### Weight Ledger (Calibration Memory)
`weight_ledger.json` stores feedback-driven weight adjustments and history.
This means scores can change over time for the same input if calibration has been applied.

- Adaptive mode (default): `--weight-ledger weight_ledger.json`
- Baseline deterministic mode: `--ignore-weight-ledger`
- Roll back last calibration event: `--revert-last-feedback`

Detailed documentation is available in the `/docs` folder:
- [Engine Logic & Math](docs/engine_logic.md) - How the scoring and Grit Factor work.
- [Profiles & Markets](docs/profiles_and_markets.md) - Details on synthetic vs. real-world data.

## âڑ–ï¸ڈ Ethics & Transparency

Axiom-Recruit is a **white-box system**. It does not "predict" performance; it **audits** feature sets against explicit corporate and market standards. All weights are adjustable and transparent through the `catalog.py` and `feedback.py` systems.

---
**Founder & Lead Researcher**: Ziad Salah (Axiom Logic / Division-36)
**License**: MIT

