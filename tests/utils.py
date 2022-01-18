def assert_eq(actual, expected, msg="", rtol=None, atol=None):
    """Asserts two things are equal with nice support for lists and tensors

    It also gives prettier error messages than assert a == b
    """
    # This does a lot of CPU work even when running in PYTHONOPTIMIZE mode otherwise
    if not __debug__:
        return

    if not msg:
        msg = f"Values are not equal: \n\ta={actual} \n\tb={expected}"

    if isinstance(actual, torch.Size):
        actual = list(actual)
    if isinstance(expected, torch.Size):
        expected = list(expected)

    if isinstance(actual, tuple):
        actual = list(actual)
    if isinstance(expected, tuple):
        expected = list(expected)

    if isinstance(actual, torch.Tensor):
        if rtol is None and atol is None:
            rtol, atol = _get_default_rtol_and_atol(actual=actual, expected=expected)
        torch.testing.assert_allclose(actual, expected, msg=msg, rtol=rtol, atol=atol)
        return
    if isinstance(actual, np.ndarray):
        np.testing.assert_allclose(actual, expected, rtol=rtol or 0, atol=atol or 0)
        return
    if isinstance(actual, torch.Size) or isinstance(expected, torch.Size):
        assert actual == expected, msg
        return
    if isinstance(actual, dict):
        assert isinstance(expected, dict)
        assert actual.keys() == expected.keys(), msg
        for key in actual.keys():
            assert_eq(actual[key], expected[key], msg=msg, rtol=rtol, atol=atol)
        return
    if isinstance(actual, (tuple, list, set)):
        assert isinstance(expected, type(actual))
        assert len(actual) == len(expected), msg
        for ai, bi in zip(actual, expected):
            assert_eq(ai, bi, msg=msg, rtol=rtol, atol=atol)
        return

    if rtol is None and atol is None:
        assert actual == expected, f"{actual} != {expected}"
    else:
        atol = 0 if atol is None else atol
        rtol = 0 if rtol is None else rtol
        assert (
            abs(actual - expected) <= atol + expected * rtol
        ), f"{actual} != {expected}"
