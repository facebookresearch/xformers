# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path

import git


@dataclass
class ComponentInfo:
    """
    A component is deemed to have changed if any of its
    files or dependencies have changed.
    If it has not changed, its files will be removed.
    """

    name: str
    # These files will be deleted if the component is not enabled
    files: list[str]
    dependencies: list[str]
    disable_set_env: dict[str, str] = field(default_factory=dict)


COMMON_PATTERNS = [
    # All components will be tested if something in there changes
    "setup.py",
]

COMPONENTS = [
    ComponentInfo(
        name="attention",
        files=[
            "tests/test_mem_eff_attention.py",
            "tests/test_find_sparse_locations*.py",
            "tests/test_block_sparse_mem_eff_attention*.py",
            "tests/test_attention_patterns.py",
            "tests/test_rope_padded.py",
            "tests/test_tree_attention*.py",
            "tests/test_fmha*.py",
        ],
        dependencies=[
            "xformers/ops/fmha/*",
            "third_party/cutlass",
            "third_party/flash-attention",
            "third_party/composable_kernel_tiled",
            "xformers/csrc/attention/*",
            "xformers/triton/*",
        ],
        disable_set_env={
            "XFORMERS_DISABLE_FLASH_ATTN": "1",
        },
    ),
    ComponentInfo(
        name="sp24",
        files=[
            "tests/test_sparsity24.py",
            "xformers/csrc/sparse24/*",
        ],
        dependencies=[
            "xformers/ops/sp24.py",
        ],
    ),
    ComponentInfo(
        name="sequence_parallel_fused",
        files=[
            "tests/test_seqpar.py",
            "tests/test_sequence_parallel_fused_ops.py",
            "tests/test_tiled_matmul.py",
        ],
        dependencies=[
            "tests/multiprocessing_utils.py",
            "xformers/ops/sequence_parallel_fused_ops.py",
        ],
    ),
]

repo_root_path = Path(__file__).parent.parent.parent.resolve().absolute()
repo = git.Repo(repo_root_path)


def list_files_in_commit(commit: git.Commit):
    file_list = []
    stack = [commit.tree]
    while len(stack) > 0:
        tree = stack.pop()
        # enumerate blobs (files) at this level
        for b in tree.blobs:
            file_list.append(str(Path(b.path).absolute().relative_to(repo_root_path)))
        for subtree in tree.trees:
            stack.append(subtree)
    # you can return dir_list if you want directories too
    return file_list


def check_patterns_are_valid(patterns):
    # Only check patterns in `fairinternal` repo
    if os.environ.get("GITHUB_REPOSITORY", "") != "fairinternal/xformers":
        return
    found_patterns = set()
    for f in all_files:
        for pattern in patterns:
            if fnmatch.fnmatch(f, pattern):
                found_patterns.add(pattern)
    for pattern in patterns:
        if pattern not in found_patterns:
            assert False, f"Pattern does not match any file: `{pattern}`"


parser = argparse.ArgumentParser("xFormers selective CI")
parser.add_argument("--base_commit", default="origin/main")
args = parser.parse_args()

base_commit = repo.rev_parse(args.base_commit)
all_files = list_files_in_commit(repo.head.commit) + [sm.path for sm in repo.submodules]
all_modified_files = set()
for item in repo.head.commit.diff(base_commit):
    if item.a_path is not None:
        all_modified_files.add(item.a_path)
    if item.b_path is not None:
        all_modified_files.add(item.b_path)

check_patterns_are_valid(COMMON_PATTERNS)
for component in COMPONENTS:
    # Sanity check that all files exist
    check_patterns_are_valid(component.files + component.dependencies)

    # Check if module is updated
    skip_module = True
    for pattern in COMMON_PATTERNS + component.files + component.dependencies:
        for f in all_modified_files:
            if fnmatch.fnmatch(f, pattern):
                skip_module = False
                break
    print(component.name, "SKIP" if skip_module else "TEST")
    if not skip_module:
        continue

    # Delete component files
    for f in all_files:
        for pattern in component.files:
            if fnmatch.fnmatch(f, pattern):
                if Path(f).exists():
                    Path(f).unlink()

    # Set env variable
    for env_k, env_v in component.disable_set_env.items():
        if "GITHUB_ENV" not in os.environ:
            print(f"{env_k}={env_v}")
            continue
        with open(os.environ["GITHUB_ENV"], "a") as fd:
            fd.write(f"{env_k}={env_v}\n")
