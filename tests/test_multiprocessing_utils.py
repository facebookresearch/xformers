# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List

from .multiprocessing_utils import launch_subprocesses


def inner_test(present_parent_keys: List[str] = [], absent_parent_keys: List[str] = []):
    # each time the process pool submits a job to the child processes, it will also transfer the
    # environment variables of the parent process to the child process.
    # we make sure they are available to the child process
    for parent_key in present_parent_keys:
        assert parent_key in os.environ

    # if keys are updated in the parent process this should also be reflected on the child process
    # missing keys should not be available to the child process
    for parent_key in absent_parent_keys:
        assert parent_key not in os.environ

    # any inserted local env vars will be removed by our process pool manager at the end of the job
    # the process pool will restore the original environment variables at the subprocess initialisation
    assert "var_temp" not in os.environ

    # INSERT LOCAL ENV VAR
    os.environ["var_temp"] = "1"


def test_env_vars():
    # insert global env var
    os.environ["var_1"] = "1"
    os.environ["var_2"] = "1"

    # first job submit => triggers subprocess creation
    launch_subprocesses(world_size=1, fn=inner_test, present_parent_keys=["var_1"])

    # delete global env var
    del os.environ["var_2"]

    # insert new global env var
    os.environ["var_3"] = "1"

    # second job submit => reuses the subprocess created before
    launch_subprocesses(
        world_size=1,
        fn=inner_test,
        present_parent_keys=["var_1", "var_3"],
        absent_parent_keys=["var_2"],
    )
