from __future__ import annotations

from .catalog import (
    CANDIDATE_FEATURE_SPECS,
    COMPANY_FEATURE_SPECS,
    FEATURE_COUNT_SUMMARY,
    INPUT_SCHEMA_VERSION,
    MARKET_DYNAMIC_SIGNAL_KEYS,
    MARKET_FEATURE_SPECS,
)


def build_candidate_extraction_prompt() -> str:
    candidate_keys = ", ".join(list(CANDIDATE_FEATURE_SPECS.keys())[:25]) + ", ..."
    return f"""You are a structured extraction assistant for Axiom-Recruit (Candidate Profile Mode).

Goal:
- Convert CV text, GitHub repositories, publications, portfolio notes, and interview notes into one deterministic file: candidate.json

Mandatory rules:
- Output valid JSON only.
- Use schema_version "{INPUT_SCHEMA_VERSION}".
- Use numeric values only for candidate.features.
- Every candidate feature must be normalized to [0, 1].
- If evidence is missing, set value to 0.5 and add the feature name to an "assumptions" list.
- Do not hallucinate data. Do not invent evidence.
- Preserve traceability: every non-default value must map to a source statement.
- Never emit company fields.
- Never emit market fields.

Feature volume requirements:
- candidate.json must contain at least {FEATURE_COUNT_SUMMARY["candidate"]} features.

Output contract:
- Return exactly one JSON object with keys:
  {{
    "candidate": {{
      "schema_version": "{INPUT_SCHEMA_VERSION}",
      "candidate_id": "string",
      "features": {{ ...{FEATURE_COUNT_SUMMARY["candidate"]}+ numeric keys... }},
      "default_candidate_profile": {{ optional context block }}
    }},
    "assumptions": [ ... ],
    "evidence_map": {{ "feature_name": "source excerpt" }}
  }}

Normalization guidance:
- Convert ordinal labels to deterministic values:
  low=0.25, medium=0.50, high=0.75, very_high=0.90.
- If numerical source value has a known max, normalize as value/max.
- If no max is known, use bounded rubric and include rubric in assumptions.

Validation reminders:
- Candidate keys must include:
  {candidate_keys}
- No values outside [0, 1].
"""


def build_company_extraction_prompt() -> str:
    company_keys = ", ".join(list(COMPANY_FEATURE_SPECS.keys())[:25]) + ", ..."
    return f"""You are a structured extraction assistant for Axiom-Recruit (Company Profile Mode).

Goal:
- Convert recruiter notes, job descriptions, hiring process docs, leadership interviews, and team operating signals into one deterministic file: company.json

Mandatory rules:
- Output valid JSON only.
- Use schema_version "{INPUT_SCHEMA_VERSION}".
- Use numeric values only for company.features.
- Every company feature must be normalized to [0, 1].
- If evidence is missing, set value to 0.5 and add the feature name to an "assumptions" list.
- Do not hallucinate data. Do not invent evidence.
- Preserve traceability: every non-default value must map to a source statement.
- Never emit candidate fields.
- Never emit market fields.

Feature volume requirements:
- company.json must contain at least {FEATURE_COUNT_SUMMARY["company"]} features.

Output contract:
- Return exactly one JSON object with keys:
  {{
    "company": {{
      "schema_version": "{INPUT_SCHEMA_VERSION}",
      "company_id": "string",
      "features": {{ ...{FEATURE_COUNT_SUMMARY["company"]}+ numeric keys... }}
    }},
    "assumptions": [ ... ],
    "evidence_map": {{ "feature_name": "source excerpt" }}
  }}

Normalization guidance:
- Convert ordinal labels to deterministic values:
  low=0.25, medium=0.50, high=0.75, very_high=0.90.
- If a source gives a process time in days, normalize to [0,1] with explicit baseline and document the baseline in assumptions.

Validation reminders:
- Company keys must include:
  {company_keys}
- No values outside [0, 1].
"""


def build_market_extraction_prompt() -> str:
    market_keys = ", ".join(list(MARKET_FEATURE_SPECS.keys())[:20]) + ", ..."
    dynamic_keys = ", ".join(MARKET_DYNAMIC_SIGNAL_KEYS)
    return f"""You are a structured extraction assistant for Axiom-Recruit (Market Profile Mode).

Goal:
- Convert labor market notes, compensation trends, hiring competition signals, and macro conditions into one deterministic file: market.json

Mandatory rules:
- Output valid JSON only.
- Use schema_version "{INPUT_SCHEMA_VERSION}".
- Use numeric values only for market.features and market.dynamic_signals.
- Every market value must be normalized to [0, 1].
- If evidence is missing, set value to 0.5 and add the feature name to an "assumptions" list.
- Do not hallucinate data. Do not invent evidence.
- Preserve traceability: every non-default value must map to a source statement.

Feature volume requirements:
- market.json must contain at least {FEATURE_COUNT_SUMMARY["market"]} features.

Output contract:
- Return exactly one JSON object with keys:
  {{
    "market": {{
      "schema_version": "{INPUT_SCHEMA_VERSION}",
      "market_id": "string",
      "features": {{ ...{FEATURE_COUNT_SUMMARY["market"]}+ numeric keys... }},
      "dynamic_signals": {{ "{MARKET_DYNAMIC_SIGNAL_KEYS[0]}": number, ... }}
    }},
    "assumptions": [ ... ],
    "evidence_map": {{ "feature_name": "source excerpt" }}
  }}

Validation reminders:
- Market keys must include:
  {market_keys}
- dynamic_signals keys must include exactly:
  {dynamic_keys}
- dynamic_signals values must match the same keys in market.features.
- No values outside [0, 1].
"""


def build_external_ai_prompt() -> str:
    return (
        "Axiom-Recruit Prompt Bundle\n"
        "===========================\n\n"
        "[Prompt 1: Candidate Profile]\n"
        f"{build_candidate_extraction_prompt()}\n\n"
        "[Prompt 2: Company Profile]\n"
        f"{build_company_extraction_prompt()}\n\n"
        "[Prompt 3: Market Profile]\n"
        f"{build_market_extraction_prompt()}\n"
    )
