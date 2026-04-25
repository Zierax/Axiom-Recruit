from __future__ import annotations

from typing import Dict, Mapping


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def _relu(value: float) -> float:
    return value if value > 0 else 0.0


def _tanh_like(value: float) -> float:
    return value / (1 + abs(value))


EDGE_CASE_NET_SPEC: Dict[str, object] = {
    "name": "whitebox_edge_net_v1",
    "inputs": (
        "candidate_excellence_index",
        "company_readiness_index",
        "market_heat_index",
        "opportunity_loss_risk",
        "hiring_inefficiency_drag",
        "decision_latency_drag",
        "candidate_company_fit",
        "company_market_mismatch",
    ),
    "hidden": {
        "alignment_guard": {
            "weights": {
                "candidate_excellence_index": 1.10,
                "company_readiness_index": 0.25,
                "candidate_company_fit": 0.70,
                "company_market_mismatch": -0.60,
                "hiring_inefficiency_drag": -0.45,
            },
            "bias": -0.75,
            "activation": "relu",
        },
        "urgency_pressure": {
            "weights": {
                "market_heat_index": 1.05,
                "opportunity_loss_risk": 1.10,
                "decision_latency_drag": 0.30,
                "company_readiness_index": -0.25,
            },
            "bias": -0.68,
            "activation": "relu",
        },
        "instability_penalty": {
            "weights": {
                "hiring_inefficiency_drag": 1.05,
                "decision_latency_drag": 0.95,
                "company_market_mismatch": 0.75,
                "candidate_company_fit": -0.40,
            },
            "bias": -0.55,
            "activation": "relu",
        },
    },
    "output": {
        "weights": {
            "alignment_guard": 0.72,
            "urgency_pressure": 0.64,
            "instability_penalty": -0.88,
        },
        "bias": -0.08,
        "activation": "tanh_like",
        "max_adjustment": 0.03,
    },
}


def evaluate_whitebox_edge_network(inputs: Mapping[str, float]) -> Dict[str, object]:
    hidden_outputs: Dict[str, float] = {}
    hidden_trace = []
    hidden_spec = EDGE_CASE_NET_SPEC["hidden"]
    for node_name, node_spec in hidden_spec.items():
        weights = node_spec["weights"]
        weighted_sum = node_spec["bias"]
        terms = []
        for feature_name, weight in weights.items():
            value = float(inputs[feature_name])
            contribution = weight * value
            weighted_sum += contribution
            terms.append(
                {
                    "feature": feature_name,
                    "weight": weight,
                    "value": round(value, 6),
                    "contribution": round(contribution, 6),
                }
            )
        node_output = _relu(weighted_sum)
        hidden_outputs[node_name] = node_output
        hidden_trace.append(
            {
                "node": node_name,
                "activation": node_spec["activation"],
                "bias": node_spec["bias"],
                "weighted_sum": round(weighted_sum, 6),
                "output": round(node_output, 6),
                "terms": terms,
            }
        )

    output_spec = EDGE_CASE_NET_SPEC["output"]
    raw_output = output_spec["bias"]
    output_terms = []
    for node_name, weight in output_spec["weights"].items():
        value = hidden_outputs[node_name]
        contribution = value * weight
        raw_output += contribution
        output_terms.append(
            {
                "node": node_name,
                "weight": weight,
                "value": round(value, 6),
                "contribution": round(contribution, 6),
            }
        )

    activated_output = _tanh_like(raw_output)
    max_adjustment = float(output_spec["max_adjustment"])
    edge_adjustment = _clamp(max_adjustment * activated_output, -max_adjustment, max_adjustment)

    return {
        "model": EDGE_CASE_NET_SPEC["name"],
        "inputs": {key: round(float(value), 6) for key, value in inputs.items()},
        "hidden_layer": hidden_trace,
        "output_layer": {
            "activation": output_spec["activation"],
            "bias": output_spec["bias"],
            "weighted_sum": round(raw_output, 6),
            "activated": round(activated_output, 6),
            "terms": output_terms,
        },
        "edge_case_adjustment": round(edge_adjustment, 6),
    }
