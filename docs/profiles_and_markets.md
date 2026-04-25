# Profiles, Markets & Realism

Axiom-Recruit provides a comprehensive library of deterministic profiles for calibration and testing.

## 1. Candidate Profiles

### Synthetic Profiles
Most candidate profiles in `/profiles/candidates` are **synthetic**. They are generated using a deterministic blending logic across 5 levels (Level 1: Foundation to Level 5: Principal). These serve as baselines for engine calibration.

### The "Ziad Salah" Profile
The `main_ziad_salah.json` profile is unique. It was built using **real-world extraction prompts** provided within the system itself (using the `--generate-candidate-prompt` utility). 
- **Source of Truth**: This profile is synced directly with the `candidate.json` file in the project root.
- **Role**: It represents a high-grit, principal-level security researcher profile used to validate the engine's sensitivity to extreme excellence under low-resource conditions.

## 2. Market Scenarios

The system models global markets using a baseline + country-bias approach.

### Country Biases
Countries like the **USA** and **Singapore** have high demand and economic environment biases. Countries like **Egypt** are modeled with significant challenges:
- **Talent Scarcity (+0.22)**: High brain-drain and rare high-tier talent.
- **AI Adoption Velocity (-0.45)**: Lower local adoption rates.
- **Economic Environment (-0.32)**: High volatility and limited local funding.

Evaluating a candidate in the "Egypt" market under "AI-Current" scenario highlights the engine's ability to recognize talent that thrives in restrictive environments.

## 3. Company Profiles
Real-world company profiles (e.g., Meta, Google, Coveo) are modeled based on public engineering culture data, hiring latency, and technical standards.

## 4. Usage Recommendation

> [!IMPORTANT]
> While the included profiles are high-fidelity, they are deterministic templates. For real-world usage, users are strongly recommended to update feature data with real-time market metrics and specific candidate evidence using the provided AI extraction prompts.
