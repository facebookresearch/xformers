from xformers.benchmarks.mem_eff_attention_presets import (
    LONG_CONTEXT_BOUNDARY_PRESET,
    get_benchmark_cases,
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
