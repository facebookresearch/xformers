# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
import pkgutil
import subprocess
import sys
import textwrap

import pytest

import xformers.ops as xops

if xops._HAS_MSLK:
    import xformers.ops.fmha

    _REEXPORTING_MODULES = [
        m.name
        for m in pkgutil.walk_packages(xformers.ops.fmha.__path__, "xformers.ops.fmha.")
    ] + ["xformers.ops.tree_attention", "xformers.attn_bias_utils"]
else:
    _REEXPORTING_MODULES = []


@pytest.mark.parametrize("module", _REEXPORTING_MODULES)
def test_reexports_exist_in_mslk(module: str) -> None:
    # These modules re-export names from mslk: catch them going missing there
    importlib.import_module(module)


def _run(code: str, cwd) -> str:
    # In a subprocess, as importing mslk and xformers.ops has process-wide
    # effects (including, here, maybe having set MSLK_PYTHON_ONLY). The working
    # directory comes first in sys.path.
    env = {k: v for k, v in os.environ.items() if k != "MSLK_PYTHON_ONLY"}
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stderr


def test_import_without_mslk(tmp_path) -> None:
    _run(
        """
        import sys
        sys.modules["mslk"] = None  # as if mslk wasn't installed
        import xformers.ops as xops
        assert not xops._HAS_MSLK
        assert not hasattr(xops, "memory_efficient_attention")
        assert hasattr(xops, "RMSNorm")
        try:
            xops.memory_efficient_attention
            raise AssertionError("expected an AttributeError")
        except AttributeError as e:
            assert "pip install -U mslk" in str(e), e
        from xformers.ops import *
        """,
        tmp_path,
    )


def test_mslk_native_library_fails_to_load(tmp_path) -> None:
    (tmp_path / "mslk").mkdir()
    (tmp_path / "mslk" / "__init__.py").write_text(
        "import os\n"
        "if os.environ.get('MSLK_PYTHON_ONLY') != '1':\n"
        "    raise OSError('undefined symbol: fake')\n"
    )
    stderr = _run(
        """
        import os
        import xformers.ops
        assert os.environ["MSLK_PYTHON_ONLY"] == "1"
        """,
        tmp_path,
    )
    assert "python-only mode" in stderr
    assert "undefined symbol: fake" in stderr
