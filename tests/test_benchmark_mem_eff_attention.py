from xformers.benchmarks.mem_eff_attention_presets import (
    LONG_CONTEXT_BOUNDARY_PRESET,
    get_benchmark_cases,
    resolve_preset_cases,
)


def test_long_context_boundary_preset_is_stable():
    cases = get_benchmark_cases([], LONG_CONTEXT_BOUNDARY_PRESET)

    assert len(cases) == 1

    case = cases[0]
    assert case["shape_q"] == (1, 8192, 8, 128)
    assert case["num_threads"] == 1
    assert case["dropout_p"] == 0.0
    assert case["attn_bias_name"] == "lower_triangular"
    assert case["dtype_name"] == "float16"
    assert case["Hkv"] == 8


def test_long_context_boundary_preset_resolution():
    cases = get_benchmark_cases([], LONG_CONTEXT_BOUNDARY_PRESET)

    resolved = resolve_preset_cases(
        cases,
        dtype_by_name={"float16": "f16"},
        attn_bias_by_name={"lower_triangular": ("causal_mask", False)},
    )

    assert resolved == [
        {
            "shape_q": (1, 8192, 8, 128),
            "num_threads": 1,
            "dropout_p": 0.0,
            "dtype": "f16",
            "attn_bias_cfg": ("causal_mask", False),
            "Hkv": 8,
        }
    ]
