# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import ast
import copy
import functools
import linecache
import sys
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


class _VisitorUnrollKernel(ast.NodeTransformer):
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


# Hackfix to get access to get source-code for
# `exec`-created functions - see https://stackoverflow.com/a/69668999
_getlines_orig = None
_FILENAME_TO_SRC: Dict[str, str] = {}


def _monkey_patched_getlines(filename, module_globals=None):
    if filename in _FILENAME_TO_SRC:
        return _FILENAME_TO_SRC[filename]
    else:
        return _getlines_orig(filename, module_globals)  # type: ignore


@functools.lru_cache(None)
def unroll_varargs(kernel, N: int):
    """
    Specializes a triton kernel with variable number of inputs
    to a specific number of inputs `N`.
    NOTE: Because it's quite costly to call `triton.jit`,
    we cache the returned value with `lru_cache`
    """
    global _FILENAME_TO_SRC, _getlines_orig

    k = triton.JITFunction(kernel.fn)
    parsed = ast.parse(k.src)
    nodeVisitor = _VisitorUnrollKernel(N=N)
    parsed = nodeVisitor.visit(parsed)
    parsed = ast.fix_missing_locations(parsed)

    # NOTE: `ast.unparse` requires python 3.9+
    if (sys.version_info.major, sys.version_info.minor) <= (3, 8):
        raise RuntimeError("Error: This functionality requires python 3.9 or above")
    new_src = ast.unparse(parsed)  # type: ignore

    # Now we want to `eval` the function, but we need all this
    # boilerplate code to make sure triton can run `inspect.getsource`

    fn_filename = f"<unroll_varargs-{kernel.fn.__name__}-{N}>"

    # Create function given source
    code = compile(new_src, fn_filename, "exec")

    _locals: Dict[str, Any] = {}
    exec(code, kernel.fn.__globals__, _locals)
    assert len(_locals) == 1, len(_locals)
    fn = next(iter(_locals.values()))
    # Patch `getlines` only the first time
    if not _FILENAME_TO_SRC:
        _getlines_orig = linecache.getlines
        linecache.getlines = _monkey_patched_getlines
    _FILENAME_TO_SRC[fn_filename] = new_src

    jitted_fn = triton.jit(fn)
    jitted_fn.src = new_src
    return jitted_fn


# Note: just import this to make mypy happy
# when annotating variables with `VAR_ARGS_ARRAY`
VAR_ARGS_ARRAY = List[Any]
