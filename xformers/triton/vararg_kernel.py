# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import ast
import copy
import functools
import linecache
import os
import sys
import tempfile
from enum import Enum
from typing import Any, Dict, List

import triton


class _ForLoopUnroller(ast.NodeTransformer):
    def __init__(self, target, inline_variables, loop_iter):
        self.loop_iter = loop_iter
        self.target = target
        self.inline_variables = inline_variables

    def visit_Name(self, node):
        if node.id != self.target:
            return node
        return ast.Name(str(self.loop_iter))

    def visit_Subscript(self, node):
        # Pattern-matching `value[slice]`
        if (
            isinstance(node.slice, ast.Name)
            and node.slice.id == self.target
            and isinstance(node.value, ast.Name)
            and node.value.id in self.inline_variables
        ):
            return ast.Name(f"{node.value.id}{self.loop_iter}")
        return node


class _VisitorVarargKernel(ast.NodeTransformer):
    def __init__(self, N):
        self.inline_variables = set()
        self.N = N

    def visit_AnnAssign(self, node):
        # Pattern-matching:
        # var_name: "VAR_ARGS_ARRAY"
        if (
            node.value is None
            and node.simple == 1
            and isinstance(node.target, ast.Name)
            and isinstance(node.annotation, ast.Constant)
            and node.annotation.value == "VAR_ARGS_ARRAY"
        ):
            self.inline_variables.add(node.target.id)
            return []
        if node.value is not None:
            node.value = self.visit(node.value)
        if node.annotation is not None:
            node.annotation = self.visit(node.annotation)
        if node.target is not None:
            node.target = self.visit(node.target)
        return node

    def visit_arguments(self, node):
        # Replace `args` annotated with `VAR_ARGS_ARRAY`
        new_args = []
        for arg in node.args:
            if (
                arg.annotation is not None
                and isinstance(arg.annotation, ast.Constant)
                and arg.annotation.value == "VAR_ARGS_ARRAY"
            ):
                self.inline_variables.add(arg.arg)
                new_args += [ast.arg(f"{arg.arg}{i}") for i in range(self.N)]
                continue
            new_args.append(arg)
        if node.vararg is not None:
            self.inline_variables.add(node.vararg.arg)
            new_args += [ast.arg(f"{node.vararg.arg}{i}") for i in range(self.N)]
            node.vararg = None
            new_args += node.kwonlyargs
            node.kwonlyargs = []
        node.args = new_args
        return node


class _VisitorUnrollKernel(_VisitorVarargKernel):
    def visit_For(self, node):
        if (
            not isinstance(node.iter, ast.Call)
            or node.iter.func.id != "range"
            or len(node.iter.args) != 1
            or not isinstance(node.iter.args[0], ast.Call)
            or node.iter.args[0].func.id != "len"
            or len(node.iter.args[0].args) != 1
            or node.iter.args[0].args[0].id not in self.inline_variables
        ):
            node.body = [self.visit(x) for x in node.body]
            return node
        # We know we have to modify this loop
        new_nodes = []
        for i in range(self.N):
            unroller = _ForLoopUnroller(
                target=node.target.id,
                inline_variables=self.inline_variables,
                loop_iter=i,
            )
            for body in node.body:
                body = copy.deepcopy(body)
                new_node = ast.fix_missing_locations(unroller.visit(body))
                new_node = self.visit(new_node)
                new_nodes.append(new_node)
        return new_nodes


class _VisitorConditionalKernel(_VisitorVarargKernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_nodes = None

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Subscript):
            node.value = self.visit_Subscript(node.value)
            return node
        if not isinstance(node.value, ast.Name):
            return node
        if node.value.id in self.inline_variables and isinstance(node.slice, ast.Name):
            # given `a[i]`, replace with `res`, where `res` is:
            # a0 if i == 0 else a1 if i== 1 else a2 if i == 2 ...
            if_statements = [None] * self.N
            if_statements[-1] = ast.Name(f"{node.value.id}{self.N - 1}")

            for i in reversed(range(self.N - 1)):
                test = ast.Compare(node.slice, [ast.Eq()], [ast.Constant(i)])
                body = ast.Name(f"{node.value.id}{i}")
                if_statements[i] = ast.IfExp(
                    test=test,
                    body=body,
                    orelse=if_statements[i + 1],
                )

            return if_statements[0]
        return node

    def visit_Call(self, node):
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "len"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in self.inline_variables
        ):
            return ast.Constant(self.N)
        self.generic_visit(node)
        return node


# Hackfix to get access to get source-code for
# `exec`-created functions - see https://stackoverflow.com/a/69668999
_getlines_orig = None
_FILENAME_TO_SRC: Dict[str, List[str]] = {}

# Materializing the codegen to disk can be useful for external tools, e.g. ncu
# Disabled by default because writing to disk at module import time is unexpected and error-prone.
_should_materialize_codegen = os.environ.get("XFORMERS_MATERIALIZE_CODEGEN") == "1"
_should_keep_materialized_source = os.environ.get("XFORMERS_KEEP_CODEGEN") == "1"
_tmp_dir = None


def _monkey_patched_getlines(filename, module_globals=None):
    if filename in _FILENAME_TO_SRC:
        return _FILENAME_TO_SRC[filename]
    else:
        return _getlines_orig(filename, module_globals)  # type: ignore


class VarargMode(Enum):
    UNROLL = "unroll"
    CONDITIONAL = "conditional"


@functools.lru_cache(None)
def unroll_varargs(kernel, N: int, mode: VarargMode = VarargMode.UNROLL):
    """
    Specializes a triton kernel with variable number of inputs
    to a specific number of inputs `N`.

    `mode` can either be `UNROLL` or `CONDITIONAL`. Both options
    implement the same functionality, but have different implementations
    and can have different performance. In `UNROLL` mode, any loops that
    loop over the varargs will be unrolled. In `CONDITIONAL` mode,
    indexing into the list of varargs is replaced with conditional
    statements like `a0 if i==0 else a1 if i==1 else a2...`.
    `CONDITIONAL` mode is generally better if `N` is large, because it
    generates a smaller triton kernel that should fit in the
    instruction cache and will compile faster.

    NOTE: Because it's quite costly to call `triton.jit`,
    we cache the returned value with `lru_cache`
    """
    global _FILENAME_TO_SRC, _getlines_orig, _tmp_dir

    k = triton.JITFunction(kernel.fn)
    parsed = ast.parse(k.src)
    if mode == VarargMode.UNROLL:
        nodeVisitor: _VisitorVarargKernel = _VisitorUnrollKernel(N=N)
    elif mode == VarargMode.CONDITIONAL:
        nodeVisitor = _VisitorConditionalKernel(N=N)
    parsed = nodeVisitor.visit(parsed)
    parsed = ast.fix_missing_locations(parsed)

    # NOTE: `ast.unparse` requires python 3.9+
    if (sys.version_info.major, sys.version_info.minor) <= (3, 8):
        raise RuntimeError("Error: This functionality requires python 3.9 or above")
    new_src = ast.unparse(parsed)  # type: ignore

    # Now we want to `eval` the function, but we need all this
    # boilerplate code to make sure triton can run `inspect.getsource`

    fn_basename = f"unroll_varargs-{kernel.fn.__name__}-{mode.value}-{N}"
    if _should_materialize_codegen:
        if not _tmp_dir:
            _tmp_dir = tempfile.TemporaryDirectory()
        fn_filename = os.path.join(_tmp_dir.name, f"{fn_basename}.py")
        if _should_keep_materialized_source:
            # destroy the TemporaryDirectory object
            _tmp_dir = None
            # create path if not exists
            os.makedirs(os.path.dirname(fn_filename), exist_ok=True)
        with open(fn_filename, "w") as f:
            f.write(new_src)
    else:
        # Patch `getlines` only the first time
        if not _FILENAME_TO_SRC:
            _getlines_orig = linecache.getlines
            linecache.getlines = _monkey_patched_getlines
        fn_filename = f"<{fn_basename}>"
        _FILENAME_TO_SRC[fn_filename] = new_src.splitlines(keepends=True)

    # Create function given source
    code = compile(new_src, fn_filename, "exec")

    _locals: Dict[str, Any] = {}
    exec(code, kernel.fn.__globals__, _locals)
    assert len(_locals) == 1, len(_locals)
    fn = next(iter(_locals.values()))

    jitted_fn = triton.jit(fn)
    if not hasattr(jitted_fn, "_unsafe_update_src"):
        # Triton older than 3.2
        jitted_fn.src = new_src
    return jitted_fn


# Note: just import this to make mypy happy
# when annotating variables with `VAR_ARGS_ARRAY`
VAR_ARGS_ARRAY = List[Any]
