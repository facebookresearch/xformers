# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import importlib.util
import logging
from unittest import mock

import xformers.ops


def _reload_ops_without_mslk():
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "mslk":
            return None
        return real_find_spec(name, *args, **kwargs)

    with mock.patch("importlib.util.find_spec", side_effect=fake_find_spec):
        importlib.reload(xformers.ops)


def test_warns_when_mslk_is_missing(caplog):
    """A missing mslk strips memory_efficient_attention from xformers.ops.

    Without a diagnostic the only symptom is `AttributeError: module 'xformers.ops'
    has no attribute 'memory_efficient_attention'` at first use, with nothing
    naming the cause. See issue #1399.
    """
    try:
        with caplog.at_level(logging.WARNING, logger="xformers"):
            _reload_ops_without_mslk()

        assert not hasattr(xformers.ops, "memory_efficient_attention")
        messages = [record.getMessage() for record in caplog.records]
        assert any(
            "mslk" in message for message in messages
        ), "expected a warning naming mslk, got: " + repr(messages)
    finally:
        importlib.reload(xformers.ops)
