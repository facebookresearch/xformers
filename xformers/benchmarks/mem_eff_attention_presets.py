from typing import Dict, List, Optional


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
