from typing import Any, Dict, List, Mapping, Optional


LONG_CONTEXT_BOUNDARY_PRESET = "long-context-boundary"
PRESET_CASES: Dict[str, List[dict]] = {
    LONG_CONTEXT_BOUNDARY_PRESET: [
        {
            "shape_q": (1, 8192, 8, 128),
            "num_threads": 1,
            "dropout_p": 0.0,
            "attn_bias_name": "lower_triangular",
            "dtype_name": "float16",
            "Hkv": 8,
        }
    ]
}


def get_benchmark_cases(default_cases: List[dict], preset: Optional[str]) -> List[dict]:
    if preset is None:
        return default_cases
    return PRESET_CASES[preset]


def resolve_preset_cases(
    preset_cases: List[dict],
    *,
    dtype_by_name: Mapping[str, Any],
    attn_bias_by_name: Mapping[str, Any],
) -> List[dict]:
    resolved_cases = []
    for case in preset_cases:
        resolved_case = case.copy()
        resolved_case["dtype"] = dtype_by_name[resolved_case.pop("dtype_name")]
        resolved_case["attn_bias_cfg"] = attn_bias_by_name[
            resolved_case.pop("attn_bias_name")
        ]
        resolved_cases.append(resolved_case)
    return resolved_cases
