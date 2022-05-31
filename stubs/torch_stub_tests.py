# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Tuple, TypeVar

import torch
import torch.nn as nn
from pyre_extensions import TypeVarTuple, Unpack
from torch import Tensor
from typing_extensions import Literal as L

Ts = TypeVarTuple("Ts")
N = TypeVar("N", bound=int)

# flake8: noqa

"""
Tensor shape signatures can get complicated and hard to debug. We are basically
writing code at the level of types.

It's helpful to have type-level unit tests for the stubs.

Take care to add both a positive and a negative test for your stub. That way,
even if someone changes the stub to return a bad type like `Any`, we will still
be warned by an unused-ignore error. Otherwise, `y: Tensor[int, L[2], L[3]] =
foo(x)` would silently pass because `Any` is compatible with any type.

Use `pyre --output=json | pyre-upgrade` to add the `pyre-fixme` comment for you.
"""


def test_sin() -> None:
    x: Tensor[int, L[2], L[3]]
    same_shape_as_x: Tensor[int, L[2], L[3]]
    not_same_shape_as_x: Tensor[int, L[2], L[99]]
    y: Tensor[int, L[2], L[3]] = torch.sin(x)
    # pyre-fixme[9]: y2 has type `Tensor[int, typing_extensions.Literal[2],
    #  typing_extensions.Literal[4]]`; used as `Tensor[int,
    #  typing_extensions.Literal[2], typing_extensions.Literal[3]]`.
    y2: Tensor[int, L[2], L[4]] = torch.sin(x)

    y3: Tensor[int, L[2], L[3]] = torch.sin(x, out=same_shape_as_x)
    # pyre-fixme[6]: Expected `Tensor[Variable[torch.DType], *torch.Ts]` for 2nd
    #  param but got `Tensor[int, int, int]`.
    # pyre-fixme[9]: y4 has type `Tensor[int, typing_extensions.Literal[2],
    #  typing_extensions.Literal[4]]`; used as `Tensor[int,
    #  typing_extensions.Literal[2], typing_extensions.Literal[3]]`.
    y4: Tensor[int, L[2], L[4]] = torch.sin(x, out=not_same_shape_as_x)
    y5: Tensor[int, L[2], L[3]] = torch.sin(x, out=None)


def test_unsqueeze() -> None:
    x: Tensor[int, L[2], L[3]]
    y: Tensor[int, L[1], L[2], L[3]] = x.unsqueeze(0)
    y_torch_function: Tensor[int, L[1], L[2], L[3]] = torch.unsqueeze(x, 0)
    y2: Tensor[int, L[2], L[1], L[3]] = x.unsqueeze(1)
    y3: Tensor[int, L[2], L[3], L[1]] = x.unsqueeze(-1)
    # pyre-fixme[9]: y4 has type `Tensor[int, typing_extensions.Literal[99]]`; used
    #  as `Tensor[int, typing_extensions.Literal[1], typing_extensions.Literal[2],
    #  typing_extensions.Literal[3]]`.
    y4: Tensor[int, L[99]] = x.unsqueeze(0)

    empty: Tensor[int]
    y5: Tensor[int, L[1]] = empty.unsqueeze(0)
    # pyre-fixme[6]: Expected `typing_extensions.Literal[0]` for 1st param but got
    #  `typing_extensions.Literal[1]`.
    y6: Tensor[int, L[1]] = empty.unsqueeze(1)
    y7: Tensor[int, L[2], L[3], L[1]] = x.unsqueeze(2)


def test_unsqueeze_() -> None:
    x: Tensor[int, L[2], L[3]]
    y: Tensor[int, L[1], L[2], L[3]] = x.unsqueeze_(0)
    y_error: Tensor[int, L[1], L[2], L[3]] = x.unsqueeze_(0)

    # pyre-ignore[9]: `unsqueeze_` is an in-place shape-transforming function. But Pyre cannot
    # update a variable's shape type.
    z: Tensor[int, L[1], L[2], L[3]] = x


def test_squeeze_() -> None:
    x: Tensor[int, L[1], L[2], L[3]]
    out: Tensor

    y: Tensor[int, L[2], L[3]] = x.squeeze_(out=out)
    # pyre-ignore[9]: Expected error.
    y_error: Tensor[int, L[2], L[99]] = x.squeeze_()
    y2: Tensor[int, L[2], L[3]] = x.squeeze_().squeeze_()

    x2: Tensor[int, L[2], L[3], L[1], L[1]]
    x3: Tensor[int, L[2], L[3], L[1]]
    y3: Tensor[int, L[2], L[3]] = x2.squeeze_()
    y4: Tensor[int, L[2], L[3]] = x3.squeeze_()
    y5: Tensor[int, L[2], L[3]] = x.squeeze_(0)
    y6: Tensor[int, L[2], L[3], L[1]] = x2.squeeze_(-1)


def test_squeeze() -> None:
    x: Tensor[int, L[1], L[2], L[3]]
    out: Tensor

    y: Tensor[int, L[2], L[3]] = x.squeeze(out=out)
    # pyre-ignore[9]: Expected error.
    y_error: Tensor[int, L[2], L[99]] = x.squeeze()
    y2: Tensor[int, L[2], L[3]] = x.squeeze().squeeze()

    x2: Tensor[int, L[2], L[3], L[1], L[1]]
    x3: Tensor[int, L[2], L[3], L[1]]
    y3: Tensor[int, L[2], L[3]] = x2.squeeze()
    y4: Tensor[int, L[2], L[3]] = x3.squeeze()
    y5: Tensor[int, L[2], L[3]] = x.squeeze(0)
    y6: Tensor[int, L[2], L[3], L[1]] = x2.squeeze(-1)


def test_repeat() -> None:
    x: Tensor[int, L[2], L[3]]
    y: Tensor[int, L[8], L[15]] = x.repeat(4, 5)
    # pyre-fixme[9]
    y2: Tensor[int, L[8], L[16]] = x.repeat(4, 5)

    # TODO(T96315150): This is passing by coincidence right now.
    y3: Tensor[int, L[4], L[10], L[18]] = x.repeat(4, 5, 6)
    # pyre-ignore[9]: Doesn't error as expected because we have limited overloads.
    y3_error: Tensor[int, L[4], L[10], L[99]] = x.repeat(4, 5, 6)

    # pyre-ignore[9, 19]
    not_yet_supported: Tensor[int, L[4], L[5], L[12], L[21]] = x.repeat(4, 5, 6, 7)

    # Fewer dimensions than the Tensor. Should raise a different error.
    x.repeat(2)

    one_dimension: Tensor[int, L[2]]
    y4: Tensor[int, L[8]] = x.repeat(4)
    # pyre-ignore[9]
    y4_error: Tensor[int, L[99]] = x.repeat(4)


def test_multiply() -> None:
    x: Tensor[torch.int64, L[2], L[3]]

    y: Tensor[torch.float32, L[2], L[3]] = x * 2
    # pyre-fixme[9]: y_error has type `Tensor[torch.bool,
    #  typing_extensions.Literal[2], typing_extensions.Literal[99]]`; used as
    #  `Tensor[torch.float32, typing_extensions.Literal[2],
    #  typing_extensions.Literal[3]]`.
    y_error: Tensor[torch.bool, L[2], L[99]] = x * 2

    y2: Tensor[torch.float32, L[2], L[3]] = 2 * x
    # pyre-fixme[9]: y2_error has type `Tensor[torch.bool,
    #  typing_extensions.Literal[2], typing_extensions.Literal[99]]`; used as
    #  `Tensor[torch.float32, typing_extensions.Literal[2],
    #  typing_extensions.Literal[3]]`.
    y2_error: Tensor[torch.bool, L[2], L[99]] = 2 * x

    y3: Tensor[torch.float32, L[2], L[3]] = x * 2.0
    # pyre-fixme[9]: y3_error has type `Tensor[torch.float32,
    #  typing_extensions.Literal[2], typing_extensions.Literal[4]]`; used as
    #  `Tensor[torch.float32, typing_extensions.Literal[2],
    #  typing_extensions.Literal[3]]`.
    y3_error: Tensor[torch.float32, L[2], L[4]] = x * 2.0

    z: Tensor[torch.int64, L[4], L[1], L[1]]
    z_bad: Tensor[torch.int64, L[4], L[2], L[99]]
    y4: Tensor[torch.int64, L[4], L[2], L[3]] = x * z
    # pyre-fixme[2001]: Broadcast error at expression `x.__mul__(z_bad)`; types
    #  `Tuple[typing_extensions.Literal[2], typing_extensions.Literal[3]]` and
    #  `Tuple[typing_extensions.Literal[4], typing_extensions.Literal[2],
    #  typing_extensions.Literal[99]]` cannot be broadcasted together.
    x * z_bad

    x4: Tensor[torch.float32, L[2], L[3]]
    x5: Tensor[torch.float32, L[2], L[3]]
    x5_bad: Tensor[torch.float32, L[2], L[99]]
    x4 *= x5
    x4 *= 4
    y5: Tensor[torch.float32, L[2], L[3]] = x5

    # pyre-fixme[2001]: Broadcast error at expression `x4.__imul__(x5_bad)`; types
    #  `Tuple[typing_extensions.Literal[2], typing_extensions.Literal[3]]` and
    #  `Tuple[typing_extensions.Literal[2], typing_extensions.Literal[99]]` cannot be
    #  broadcasted together.
    x4 *= x5_bad


def test_floor_division() -> None:
    x: Tensor[torch.int64, L[2], L[3]]
    x2: Tensor[torch.int64, L[2], L[1]]
    y: Tensor[torch.int64, L[2], L[3]] = x // 2
    # pyre-fixme[9]: y_error has type `Tensor[torch.bool,
    #  typing_extensions.Literal[2], typing_extensions.Literal[99]]`; used as
    #  `Tensor[torch.float32, typing_extensions.Literal[2],
    #  typing_extensions.Literal[3]]`.
    y_error: Tensor[torch.bool, L[2], L[99]] = x // 2

    y2: Tensor[torch.int64, L[2], L[3]] = 2 // x
    # pyre-fixme[9]: y2_error has type `Tensor[torch.bool,
    #  typing_extensions.Literal[2], typing_extensions.Literal[99]]`; used as
    #  `Tensor[torch.float32, typing_extensions.Literal[2],
    #  typing_extensions.Literal[3]]`.
    y2_error: Tensor[torch.bool, L[2], L[99]] = 2 // x

    y3: Tensor[torch.int64, L[2], L[3]] = x // x2

    x3: Tensor[torch.float32, L[2], L[3]]
    x4: Tensor[torch.float32, L[2], L[3]]
    x4_bad: Tensor[torch.float32, L[2], L[99]]
    x3 //= x4
    x3 //= 4
    y5: Tensor[torch.float32, L[2], L[3]] = x3

    # pyre-fixme[2001]: Broadcast error at expression `x3.__ifloordiv__(x4_bad)`;
    #  types `Tuple[typing_extensions.Literal[2], typing_extensions.Literal[3]]` and
    #  `Tuple[typing_extensions.Literal[2], typing_extensions.Literal[99]]` cannot be
    #  broadcasted together.
    x3 //= x4_bad


def test_division() -> None:
    x: Tensor[torch.int64, L[2], L[3]]
    x2: Tensor[torch.int64, L[2], L[1]]
    y: Tensor[torch.float32, L[2], L[3]] = x / 2
    # pyre-fixme[9]: y_error has type `Tensor[torch.bool,
    #  typing_extensions.Literal[2], typing_extensions.Literal[99]]`; used as
    #  `Tensor[torch.float32, typing_extensions.Literal[2],
    #  typing_extensions.Literal[3]]`.
    y_error: Tensor[torch.bool, L[2], L[99]] = x / 2

    y2: Tensor[torch.float32, L[2], L[3]] = 2 / x
    # pyre-fixme[9]: y2_error has type `Tensor[torch.bool,
    #  typing_extensions.Literal[2], typing_extensions.Literal[99]]`; used as
    #  `Tensor[torch.float32, typing_extensions.Literal[2],
    #  typing_extensions.Literal[3]]`.
    y2_error: Tensor[torch.bool, L[2], L[99]] = 2 / x

    x3: Tensor[torch.float32, L[2], L[3]]
    y3: Tensor[torch.float32, L[2], L[3]] = x3 / 2
    y4: Tensor[torch.float32, L[2], L[3]] = 2 / x3

    y5: Tensor[torch.float32, L[2], L[3]] = x / x2

    x5: Tensor[torch.float32, L[2], L[3]]
    x6: Tensor[torch.float32, L[2], L[3]]
    x6_bad: Tensor[torch.float32, L[2], L[99]]
    x5 /= x6
    x5 /= 4
    y6: Tensor[torch.float32, L[2], L[3]] = x5

    # pyre-fixme[2001]: Broadcast error at expression `x5.__itruediv__(x6_bad)`;
    #  types `Tuple[typing_extensions.Literal[2], typing_extensions.Literal[3]]` and
    #  `Tuple[typing_extensions.Literal[2], typing_extensions.Literal[99]]` cannot be
    #  broadcasted together.
    x5 /= x6_bad


def test_setitem() -> None:
    x: Tensor[torch.int64, L[2], L[3]]
    x[0, 0] = 1


def test_arange(n: N) -> None:
    y: Tensor[torch.int64, L[5]] = torch.arange(5)
    # pyre-fixme[9]: y_error has type `Tensor[torch.int64,
    #  typing_extensions.Literal[99]]`; used as `Tensor[torch.int64,
    #  typing_extensions.Literal[5]]`.
    y_error: Tensor[torch.int64, L[99]] = torch.arange(5)
    y2: Tensor[torch.int64, L[4]] = torch.arange(1, 5)
    y3: Tensor[torch.int64, L[2]] = torch.arange(1, 6, 2)

    y_float: Tensor[torch.float32, L[5]] = torch.arange(5, dtype=torch.float32)
    y_float2: Tensor[torch.float32, L[2]] = torch.arange(1, 6, 2, dtype=torch.float32)

    device: torch.device
    y_generic: Tensor[torch.float32, N] = torch.arange(
        0, n, device=device, dtype=torch.float32
    )
    # pyre-fixme[9]: Expected error.
    y_generic_error: Tensor[torch.float32, L[99]] = torch.arange(
        0, n, device=device, dtype=torch.float32
    )


def test_embedding() -> None:
    embedding = nn.Embedding(10, 20)
    y: Tensor[torch.float32, L[10], L[20]] = embedding.weight
    # pyre-fixme[9]: y_error has type `Tensor[torch.float32,
    #  typing_extensions.Literal[10], typing_extensions.Literal[99]]`; used as
    #  `Tensor[torch.float32, typing_extensions.Literal[10],
    #  typing_extensions.Literal[20]]`.
    y_error: Tensor[torch.float32, L[10], L[99]] = embedding.weight

    x: Tensor[torch.float32, L[2], L[3], L[4]]
    y2: Tensor[torch.float32, L[2], L[3], L[4], L[20]] = embedding(x)
    # pyre-fixme[9]: y2_error has type `Tensor[torch.float32, typing_extensions.Liter...
    y2_error: Tensor[torch.float32, L[2], L[3], L[4], L[99]] = embedding(x)

    weight: Tensor[torch.float32, L[3], L[4]]
    embedding2: nn.Embedding[L[3], L[4]] = nn.Embedding.from_pretrained(weight)
    # pyre-fixme[9]: embedding2_error has type
    #  `Embedding[typing_extensions.Literal[3], typing_extensions.Literal[99]]`; used
    #  as `Embedding[typing_extensions.Literal[3], typing_extensions.Literal[4]]`.
    embedding2_error: nn.Embedding[L[3], L[99]] = nn.Embedding.from_pretrained(weight)
    y3: Tensor[torch.float32, L[2], L[3], L[4], L[4]] = embedding2(x)


def test_init_normal() -> None:
    x: Tensor[torch.float32, L[5], L[10]]
    y: Tensor[torch.float32, L[5], L[10]] = nn.init.normal_(x)
    # pyre-fixme[9]: y_error has type `Tensor[torch.float32,
    #  typing_extensions.Literal[5], typing_extensions.Literal[99]]`; used as
    #  `Tensor[torch.float32, typing_extensions.Literal[5],
    #  typing_extensions.Literal[10]]`.
    y_error: Tensor[torch.float32, L[5], L[99]] = nn.init.normal_(x)


def test_view() -> None:
    x: Tensor[torch.float32, L[4], L[4]]
    y: Tensor[torch.float32, L[16]] = x.view(16)
    # pyre-fixme[9]: y_error has type `Tensor[torch.float32,
    #  typing_extensions.Literal[99]]`; used as `Tensor[torch.float32,
    #  typing_extensions.Literal[16]]`.
    y_error: Tensor[torch.float32, L[99]] = x.view(16)
    # Should be an error because 4 * 4 != 99. Don't think this is going to be
    # feasible any time soon.
    y_error2: Tensor[torch.float32, L[99]] = x.view(99)
    y_error3: Tensor[torch.float32, L[2], L[3], L[4], L[5]] = x.view(2, 3, 4, 5)

    y2: Tensor[torch.float32, L[2], L[8]] = x.view(-1, 8)
    # pyre-fixme[9]: y2_error has type `Tensor[torch.float32,
    #  typing_extensions.Literal[2], typing_extensions.Literal[99]]`; used as
    #  `Tensor[torch.float32, typing_extensions.Literal[2],
    #  typing_extensions.Literal[8]]`.
    y2_error: Tensor[torch.float32, L[2], L[99]] = x.view(-1, 8)

    x3: Tensor[torch.float32, L[2], L[3], L[4]]
    y3: Tensor[torch.float32, L[24]] = x3.view(-1)
    y4: Tensor[torch.float32, L[8], L[3]] = x3.view(-1, 3)
    # pyre-fixme[9]: y4_error has type `Tensor[torch.float32,
    #  typing_extensions.Literal[99], typing_extensions.Literal[3]]`; used as
    #  `Tensor[torch.float32, typing_extensions.Literal[8],
    #  typing_extensions.Literal[3]]`.
    y4_error: Tensor[torch.float32, L[99], L[3]] = x3.view(-1, 3)
    y5: Tensor[torch.float32, L[2], L[6], L[2]] = x3.view(2, -1, 2)

    x4: Tensor[torch.float32, L[2], L[3], L[4], L[5]]
    y6: Tensor[torch.float32, L[3], L[5], L[8]] = x4.view(3, 5, -1)


def test_reshape() -> None:
    x: Tensor[torch.float32, L[2], L[3], L[4]]
    y: Tensor[torch.float32, L[24]] = torch.reshape(x, (-1,))
    y2: Tensor[torch.float32, L[8], L[3]] = torch.reshape(x, (-1, 3))
    # pyre-fixme[9]: y2_error has type `Tensor[torch.float32,
    #  typing_extensions.Literal[99], typing_extensions.Literal[3]]`; used as
    #  `Tensor[torch.float32, typing_extensions.Literal[8],
    #  typing_extensions.Literal[3]]`.
    y2_error: Tensor[torch.float32, L[99], L[3]] = torch.reshape(x, (-1, 3))
    y3: Tensor[torch.float32, L[6], L[2], L[2]] = torch.reshape(x, (-1, 2, 2))
    y4: Tensor[torch.float32, L[2], L[6], L[2]] = torch.reshape(x, (2, -1, 2))
    y5: Tensor[torch.float32, L[4], L[3], L[2]] = torch.reshape(x, (4, 3, 2))


def test_transpose() -> None:
    x: Tensor[torch.float32, L[2], L[3], L[4], L[5], L[6]]
    y: Tensor[torch.float32, L[2], L[3], L[4], L[6], L[5]] = x.transpose(-2, -1)
    y_function: Tensor[torch.float32, L[2], L[3], L[4], L[6], L[5]] = torch.transpose(
        x, -2, -1
    )
    # pyre-fixme[9]: y_error has type `Tensor[torch.float32, typing_extensions.Litera...
    y_error: Tensor[torch.float32, L[2], L[4], L[99]] = x.transpose(-2, -1)

    y2: Tensor[torch.float32, L[2], L[4], L[3], L[5], L[6]] = x.transpose(1, 2)
    y3: Tensor[torch.float32, L[3], L[2], L[4], L[5], L[6]] = x.transpose(0, 1)
    y4: Tensor[torch.float32, L[3], L[2], L[4], L[5], L[6]] = x.transpose(1, 0)
    y5: Tensor[torch.float32, L[2], L[3], L[4], L[6], L[5]] = x.transpose(-1, -2)
    not_yet_supported: Tensor[
        torch.float32,
        L[3],
        L[2],
        L[4],
        L[5],
        L[6]
        # pyre-fixme[6]: Expected `typing_extensions.Literal[0]` for 2nd param but got
        #  `typing_extensions.Literal[4]`.
    ] = x.transpose(1, 4)


def test_flatten() -> None:
    x: Tensor[torch.float32, L[2], L[3]]
    x_large: Tensor[torch.float32, L[2], L[3], L[4], L[5]]
    y: Tensor[torch.float32, L[6]] = x.flatten()
    y_default: Tensor[torch.float32, L[6]] = torch.flatten(x)
    y_large: Tensor[torch.float32, L[120]] = x_large.flatten()
    # pyre-fixme[9]: y_error has type `Tensor[torch.float32,
    #  typing_extensions.Literal[99]]`; used as `Tensor[torch.float32,
    #  typing_extensions.Literal[6]]`.
    y_error: Tensor[torch.float32, L[99]] = x.flatten()

    z: Tensor[torch.float32, L[2], L[3], L[4]]

    y2: Tensor[torch.float32, L[6], L[4]] = z.flatten(0, 1)
    y2_keyword: Tensor[torch.float32, L[6], L[4]] = z.flatten(start_dim=0, end_dim=1)
    y3: Tensor[torch.float32, L[2], L[12]] = z.flatten(1, 2)
    y3_large: Tensor[torch.float32, L[2], L[12], L[5]] = x_large.flatten(1, 2)

    y4: Tensor[torch.float32, L[2], L[3], L[20]] = x_large.flatten(2, 3)

    x_6d: Tensor[torch.float32, L[2], L[3], L[4], L[5], L[6], L[7]]
    y4_large: Tensor[torch.float32, L[2], L[3], L[20], L[6], L[7]] = x_6d.flatten(2, 3)

    # Out of bounds.
    # pyre-fixme[9]: y5_error has type `Tensor[torch.float32,
    #  typing_extensions.Literal[2], typing_extensions.Literal[12]]`; used as
    #  `Tensor[torch.float32, typing_extensions.Literal[6]]`.
    # pyre-fixme[6]: Expected `typing_extensions.Literal[0]` for 1st param but got
    #  `typing_extensions.Literal[99]`.
    y5_error: Tensor[torch.float32, L[2], L[12]] = x.flatten(99, 100)

    x_0d: Tensor[torch.float32]
    y_0d: Tensor[torch.float32, L[1]] = x_0d.flatten()


def test_empty() -> None:
    x: Tuple[L[1], L[2], L[3]]
    y: Tensor
    device: torch.device

    result1: torch.Tensor[torch.float32, L[1], L[2], L[3]] = torch.empty(
        *x,
        device=device,
        layout=torch.strided,
        requires_grad=True,
        out=y,
        pin_memory=False,
        memory_format=torch.memory_format(),
    )
    # pyre-fixme[9]: bad1 has type `Tensor[torch.float32, typing_extensions.Literal[9...
    bad1: torch.Tensor[torch.float32, L[99], L[2], L[3]] = torch.empty(*x)

    result2: torch.Tensor[torch.float32, L[1], L[2], L[3]] = torch.empty(
        *x, device=device, layout=torch.strided, requires_grad=True, out=y
    )
    # pyre-fixme[9]: bad2 has type `Tensor[torch.float32, typing_extensions.Literal[9...
    bad2: torch.Tensor[torch.float32, L[99], L[2], L[3]] = torch.empty(
        *x, device=device, layout=torch.strided, requires_grad=True, out=y
    )

    result4: torch.Tensor[torch.float32, L[1], L[2], L[3]] = torch.empty(x)
    result5: torch.Tensor[torch.float32, L[4]] = torch.empty(4)

    result6: torch.Tensor[torch.int64, L[1], L[2], L[3]] = torch.empty(
        x, dtype=torch.int64
    )
    result7: torch.Tensor[torch.int64, L[1], L[2], L[3]] = torch.empty(
        *x, dtype=torch.int64
    )


def test_empty_like() -> None:
    x: torch.Tensor[torch.float32, L[1], L[2], L[3]]
    out: Tensor
    device: torch.device

    y1: torch.Tensor[torch.float32, L[1], L[2], L[3]] = torch.empty_like(
        x, device=device, layout=torch.strided, requires_grad=True, out=out
    )
    # pyre-fixme[9]: Expected error.
    y1_error: torch.Tensor[torch.float32, L[99], L[2], L[3]] = torch.empty_like(
        x, device=device, layout=torch.strided, requires_grad=True, out=out
    )
    y2: torch.Tensor[torch.int64, L[1], L[2], L[3]] = torch.empty_like(
        x,
        dtype=torch.int64,
        device=device,
        layout=torch.strided,
        requires_grad=True,
        out=out,
    )


def test_randn() -> None:
    x: Tuple[L[1], L[2], L[3]]
    y: Tensor
    device: torch.device

    result1: torch.Tensor[torch.float32, L[1], L[2], L[3]] = torch.randn(
        *x, device=device, layout=torch.strided, requires_grad=True, out=y
    )
    # pyre-fixme[9]: bad1 has type `Tensor[torch.float32, typing_extensions.Literal[9...
    bad1: torch.Tensor[torch.float32, L[99], L[2], L[3]] = torch.randn(*x)

    result2: torch.Tensor[torch.float32, L[1], L[2], L[3]] = torch.randn(
        *x, device=device, layout=torch.strided, requires_grad=True, out=y
    )
    # pyre-fixme[9]: bad2 has type `Tensor[torch.float32, typing_extensions.Literal[9...
    bad2: torch.Tensor[torch.float32, L[99], L[2], L[3]] = torch.randn(
        *x, device=device, layout=torch.strided, requires_grad=True, out=y
    )

    result4: torch.Tensor[torch.float32, L[1], L[2], L[3]] = torch.randn(x)
    result5: torch.Tensor[torch.float32, L[4]] = torch.randn(4)

    result6: torch.Tensor[torch.int64, L[1], L[2], L[3]] = torch.randn(
        x, dtype=torch.int64
    )
    result7: torch.Tensor[torch.int64, L[1], L[2], L[3]] = torch.randn(
        *x, dtype=torch.int64
    )


def test_all() -> None:
    x: torch.Tensor[torch.float32, L[1], L[2], L[3]]
    device: torch.device

    y: torch.Tensor[torch.bool, L[1]] = torch.all(x)
    # pyre-fixme[9]: bad1 has type `Tensor[torch.bool,
    #  typing_extensions.Literal[99]]`; used as `Tensor[torch.bool,
    #  typing_extensions.Literal[1]]`.
    y_error: torch.Tensor[torch.bool, L[99]] = torch.all(x)
    y2: torch.Tensor[torch.bool, L[2], L[3]] = torch.all(x, dim=0)
    y3: torch.Tensor[torch.bool, L[1], L[3]] = torch.all(x, dim=1)
    y4: torch.Tensor[torch.bool, L[1]] = x.all()


def test_where() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3]]

    good: Tuple[torch.LongTensor[int, int], torch.LongTensor[int, int]] = torch.where(x)
    bad: Tuple[
        torch.LongTensor[int, int], torch.LongTensor[int, int], L[99]
    ] = torch.where(x)

    y: torch.Tensor[torch.float32, L[2], L[1]]
    not_broadcastable: torch.Tensor[torch.float32, L[2], L[99]]

    good: Tuple[torch.LongTensor[int, int], torch.LongTensor[int, int]] = torch.where(x)
    good2: torch.Tensor[torch.float32, L[2], L[3]] = torch.where(x > 0, x, y)
    # pyre-fixme[9]: bad2 has type `Tensor[torch.float32,
    #  typing_extensions.Literal[2], typing_extensions.Literal[99]]`; used as
    #  `Tensor[torch.float32, typing_extensions.Literal[2],
    #  typing_extensions.Literal[3]]`.
    bad2: torch.Tensor[torch.float32, L[2], L[99]] = torch.where(x > 0, x, y)
    # pyre-fixme[2001]: Broadcast error at expression `torch.where(x > 0, x,
    #  not_broadcastable)`; types `Tuple[typing_extensions.Literal[2],
    #  typing_extensions.Literal[3]]` and `Tuple[typing_extensions.Literal[2],
    #  typing_extensions.Literal[99]]` cannot be broadcasted together.
    z = torch.where(x > 0, x, not_broadcastable)


def test_getitem() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]

    good1: torch.Tensor[torch.float32, L[3], L[4]] = x[0]
    # pyre-fixme[9]: bad1 has type `Tensor[torch.float32,
    #  typing_extensions.Literal[99], typing_extensions.Literal[4]]`; used as
    #  `Tensor[torch.float32, typing_extensions.Literal[3],
    #  typing_extensions.Literal[4]]`.
    bad1: torch.Tensor[torch.float32, L[99], L[4]] = x[0]

    good2: torch.Tensor[torch.float32, L[1], L[2], L[3], L[4]] = x[None]
    # pyre-fixme[9]: bad2 has type `Tensor[torch.float32, typing_extensions.Literal[9...
    bad2: torch.Tensor[torch.float32, L[99], L[2], L[3], L[4]] = x[None]

    mask: torch.Tensor[torch.bool, L[2], L[3], L[4]]
    good3: torch.Tensor[torch.float32, int] = x[mask]
    # pyre-fixme[9]: bad3 has type `Tensor[torch.float32,
    #  typing_extensions.Literal[99]]`; used as `Tensor[torch.float32, int]`.
    bad3: torch.Tensor[torch.float32, L[99]] = x[mask]

    any1: Tuple[int, str, float] = x[2]
    any2: Tuple[float, str, int] = x[2]


def test_expand() -> None:
    x: torch.Tensor[torch.float32, L[1], L[2], L[3]]
    shape: Tuple[L[4], L[1], L[3]]

    good1: torch.Tensor[torch.float32, L[4], L[2], L[3]] = x.expand(shape)
    # pyre-fixme[9]: bad1 has type `Tensor[torch.float32, typing_extensions.Literal[9...
    bad1: torch.Tensor[torch.float32, L[99], L[2], L[3]] = x.expand(shape)
    # pyre-fixme[2001]: Broadcast error at expression `x.expand((4, 99, 3))`; types `...
    x.expand((4, 99, 3))

    good2: torch.Tensor[torch.float32, L[4], L[2], L[3]] = x.expand(4, 1, 3)


def test_to() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]

    good1: torch.Tensor[torch.int32, L[2], L[3], L[4]] = x.to(torch.int32)
    # pyre-fixme[9]: bad1 has type `Tensor[torch.int32, typing_extensions.Literal[99]...
    bad1: torch.Tensor[torch.int32, L[99], L[3], L[4]] = x.to(torch.int32)

    device: torch.device
    good2: torch.Tensor[torch.float32, L[2], L[3], L[4]] = x.to(device)
    # pyre-fixme[9]: bad2 has type `Tensor[torch.float32, typing_extensions.Literal[9...
    bad2: torch.Tensor[torch.float32, L[99], L[3], L[4]] = x.to(device)

    y: torch.Tensor[torch.int32, L[2], L[3], L[4]]
    good3: torch.Tensor[torch.float32, L[2], L[3], L[4]] = y.to(torch.float32, device)
    # pyre-fixme[9]: bad3 has type `Tensor[torch.float32, typing_extensions.Literal[9...
    bad3: torch.Tensor[torch.float32, L[99], L[3], L[4]] = y.to(torch.float32, device)


def test_Linear_to() -> None:
    linear: nn.Linear[L[10], L[20]]
    device: torch.device

    linear.to(dtype=torch.int64, device=device)


def test_Module_eval() -> None:
    module: nn.Module
    module.eval()


def test_Module_train() -> None:
    module: nn.Module
    module.train(mode=True)
    y: bool = module.training


def test_Linear_bias() -> None:
    linear: nn.Linear[L[10], L[20]]

    x: nn.Parameter = linear.bias


def test_sum() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]

    y1: torch.Tensor[torch.float32, L[2], L[3]] = x.sum(-1, dtype=None)
    # pyre-fixme[9]: y_error has type `Tensor[torch.float32,
    #  typing_extensions.Literal[99], typing_extensions.Literal[3]]`; used as
    #  `Tensor[torch.float32, typing_extensions.Literal[2],
    #  typing_extensions.Literal[3]]`.
    y_error: torch.Tensor[torch.float32, L[99], L[3]] = x.sum(-1, dtype=None)

    y2: torch.Tensor[torch.float32, L[2], L[4]] = x.sum(-2)
    y3: torch.Tensor[torch.float32] = x.sum()
    y4: torch.Tensor[torch.float32, L[3], L[4]] = x.sum(0)
    y5: torch.Tensor[torch.float32, L[2], L[4]] = x.sum(1)
    y6: torch.Tensor[torch.float32] = torch.sum(x)


def test_cumsum() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]

    good1: torch.Tensor[torch.float32, L[2], L[3], L[4]] = x.cumsum()
    # pyre-fixme[9]: bad1 has type `Tensor[torch.float32, typing_extensions.Literal[9...
    bad1: torch.Tensor[torch.float32, L[99], L[3], L[4]] = x.cumsum()

    good2: torch.Tensor[torch.float32, L[2], L[3], L[4]] = x.cumsum(dim=0)
    # pyre-fixme[9]: bad2 has type `Tensor[torch.float32, typing_extensions.Literal[9...
    bad2: torch.Tensor[torch.float32, L[99], L[3], L[4]] = x.cumsum(dim=0)

    good3: torch.Tensor[torch.float32, L[2], L[3], L[4]] = x.cumsum(dtype=None)
    # pyre-fixme[9]: bad3 has type `Tensor[torch.float32, typing_extensions.Literal[9...
    bad3: torch.Tensor[torch.float32, L[99], L[3], L[4]] = x.cumsum(dtype=None)


def test_contiguous() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]

    good: torch.Tensor[torch.float32, L[2], L[3], L[4]] = x.contiguous()
    # pyre-fixme[9]: bad has type `Tensor[torch.float32, typing_extensions.Literal[99...
    bad: torch.Tensor[torch.float32, L[99], L[3], L[4]] = x.contiguous()


def test_diff() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]

    good: torch.Tensor[torch.float32, L[2], L[3], L[3]] = torch.diff(x)
    # pyre-fixme[9]: bad has type `Tensor[torch.float32, typing_extensions.Literal[99...
    bad: torch.Tensor[torch.float32, L[99], L[3], L[3]] = torch.diff(x)
    good2: torch.Tensor[torch.float32, L[1], L[3], L[4]] = torch.diff(x, dim=0)
    good3: torch.Tensor[torch.float32, L[2], L[2], L[4]] = torch.diff(x, dim=1)
    good4: torch.Tensor[torch.float32, L[2], L[3], L[3]] = torch.diff(x, dim=-1)
    good5: torch.Tensor[torch.float32, L[2], L[2], L[4]] = torch.diff(x, dim=-2)
    good6: torch.Tensor[torch.float32, L[2], L[2], L[4]] = x.diff(dim=-2)


def test_argsort() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]

    good1: torch.Tensor[torch.float32, L[2], L[3], L[4]] = torch.argsort(x)
    # pyre-fixme[9]: bad1 has type `LongTensor[torch.float32, typing_extensions.Liter...
    bad1: torch.Tensor[torch.float32, L[99], L[3], L[4]] = torch.argsort(x)

    good2: torch.Tensor[torch.float32, L[2], L[3], L[4]] = torch.argsort(x, dim=0)
    # pyre-fixme[9]: bad2 has type `LongTensor[torch.float32, typing_extensions.Liter...
    bad2: torch.Tensor[torch.float32, L[99], L[3], L[4]] = torch.argsort(x, dim=0)

    good3: torch.Tensor[torch.float32, L[2], L[3], L[4]] = torch.argsort(
        x, descending=True
    )
    # pyre-fixme[9]: bad3 has type `LongTensor[torch.float32, typing_extensions.Liter...
    bad3: torch.Tensor[torch.float32, L[99], L[3], L[4]] = torch.argsort(
        x, descending=True
    )
    good4: torch.Tensor[torch.float32, L[2], L[3], L[4]] = x.argsort(dim=-1)


def test_functional_pad() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]

    good: torch.Tensor[torch.float32, L[2], L[3], L[5]] = nn.functional.pad(x, (1, 0))
    bad: torch.Tensor[torch.float32, L[99], L[3], L[5]] = nn.functional.pad(x, (1, 0))
    good2: torch.Tensor[torch.float32, L[2], L[10], L[7]] = nn.functional.pad(
        x, (1, 2, 3, 4), "constant", value=0.0
    )


def test_allclose() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3]]
    y: torch.Tensor[torch.float32, L[2], L[1]]
    not_broadcastable: torch.Tensor[torch.float32, L[3], L[4]]
    good: bool = torch.allclose(x, y, atol=0.0, rtol=0.0, equal_nan=True)
    # This should complain about non-broadcastable tensors but we don't have a
    # way to constrain two parameter types to be broadcastable.
    should_error: bool = torch.allclose(x, not_broadcastable)


def test_new_ones() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3]]

    y: torch.Tensor[torch.float32, L[8], L[9]] = x.new_ones((8, 9))
    # pyre-fixme[9]: Expected error.
    y_error: torch.Tensor[torch.float32, L[8], L[99]] = x.new_ones((8, 9))
    y2: torch.Tensor[torch.int64, L[8], L[9]] = x.new_ones(
        (8, 9), dtype=torch.int64, device="cuda", requires_grad=True
    )


def test_ones_like() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3]]
    device: torch.device

    good: torch.Tensor[torch.int64, L[2], L[3]] = torch.ones_like(
        x, dtype=torch.int64, device=device
    )
    # pyre-fixme[9]: bad has type `Tensor[torch.int64,
    #  typing_extensions.Literal[99], typing_extensions.Literal[3]]`; used as
    #  `Tensor[torch.int64, typing_extensions.Literal[2],
    #  typing_extensions.Literal[3]]`.
    bad: torch.Tensor[torch.int64, L[99], L[3]] = torch.ones_like(
        x, dtype=torch.int64, device=device
    )
    bad2: torch.Tensor[torch.float32, L[2], L[3]] = torch.ones_like(
        x,
    )


def test_sparse_softmax() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3]]
    y: torch.Tensor[torch.float32, L[2], L[3]] = torch.sparse.softmax(x, dim=-1)
    # pyre-fixme[9]: y_error has type `Tensor[torch.float32,
    #  typing_extensions.Literal[99], typing_extensions.Literal[3]]`; used as
    #  `Tensor[torch.float32, typing_extensions.Literal[2],
    #  typing_extensions.Literal[3]]`.
    y_error: torch.Tensor[torch.float32, L[99], L[3]] = torch.sparse.softmax(x, dim=-1)
    dtype: torch.int64
    y2: torch.Tensor[torch.int64, L[2], L[3]] = torch.sparse.softmax(
        x, dim=-1, dtype=dtype
    )


def test_eye() -> None:
    y: torch.Tensor[torch.int64, L[2], L[3]] = torch.eye(2, 3, dtype=torch.int64)
    # pyre-fixme[9]: y_error has type `Tensor[torch.int64,
    #  typing_extensions.Literal[99], typing_extensions.Literal[3]]`; used as
    #  `Tensor[torch.int64, typing_extensions.Literal[2],
    #  typing_extensions.Literal[3]]`.
    y_error: torch.Tensor[torch.int64, L[99], L[3]] = torch.eye(2, 3, dtype=torch.int64)
    y2: torch.Tensor[torch.float32, L[3], L[3]] = torch.eye(3)


def test_adaptive_average_pool2d() -> None:
    model: nn.AdaptiveAvgPool2d[L[5], L[7]] = nn.AdaptiveAvgPool2d((5, 7))
    # pyre-fixme[9]: model_error has type
    #  `AdaptiveAvgPool2d[typing_extensions.Literal[5],
    #  typing_extensions.Literal[99]]`; used as
    #  `AdaptiveAvgPool2d[typing_extensions.Literal[5], typing_extensions.Literal[7]]`.
    model_error: nn.AdaptiveAvgPool2d[L[5], L[99]] = nn.AdaptiveAvgPool2d((5, 7))
    model2: nn.AdaptiveAvgPool2d[L[5], L[5]] = nn.AdaptiveAvgPool2d(5)
    # TODO(T100083794): This should be an error.
    model2_error: nn.AdaptiveAvgPool2d[L[5], L[99]] = nn.AdaptiveAvgPool2d(5)
    model3: nn.AdaptiveAvgPool2d[L[5], L[-1]] = nn.AdaptiveAvgPool2d((5, None))
    # TODO(T100083794): This should be an error.
    model3_error: nn.AdaptiveAvgPool2d[L[5], L[99]] = nn.AdaptiveAvgPool2d((5, None))

    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]
    y: torch.Tensor[torch.float32, L[2], L[5], L[7]] = model(x)
    # pyre-fixme[9]: y_error has type `Tensor[torch.float32, typing_extensions.Litera...
    y_error: torch.Tensor[torch.float32, L[2], L[99], L[7]] = model(x)
    y2: torch.Tensor[torch.float32, L[2], L[5], L[5]] = model2(x)
    y3: torch.Tensor[torch.float32, L[2], L[5], L[4]] = model3(x)


def test_randperm() -> None:
    y: torch.Tensor[torch.int64, L[10]] = torch.randperm(10, dtype=torch.int64)
    # pyre-fixme[9]: y_error has type `Tensor[torch.int64,
    #  typing_extensions.Literal[99]]`; used as `Tensor[torch.int64,
    #  typing_extensions.Literal[10]]`.
    y_error: torch.Tensor[torch.int64, L[99]] = torch.randperm(10, dtype=torch.int64)


def test_sqrt() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3]]
    y: torch.Tensor[torch.float32, L[2], L[3]] = torch.sqrt(x)
    # pyre-fixme[9]: y_error has type `Tensor[torch.float32,
    #  typing_extensions.Literal[2], typing_extensions.Literal[99]]`; used as
    #  `Tensor[torch.float32, typing_extensions.Literal[2],
    #  typing_extensions.Literal[3]]`.
    y_error: torch.Tensor[torch.float32, L[2], L[99]] = torch.sqrt(x)


def test_multinomial() -> None:
    x: torch.Tensor[torch.float32, L[2], L[4]]
    y: torch.Tensor[torch.float32, L[2], L[3]] = torch.multinomial(x, 3)
    # pyre-fixme[9]: y_error has type `Tensor[torch.float32,
    #  typing_extensions.Literal[2], typing_extensions.Literal[99]]`; used as
    #  `Tensor[torch.float32, typing_extensions.Literal[2],
    #  typing_extensions.Literal[3]]`.
    y_error: torch.Tensor[torch.float32, L[2], L[99]] = torch.multinomial(x, 3)

    x2: torch.Tensor[torch.float32, L[4]]
    y2: torch.Tensor[torch.float32, L[3]] = torch.multinomial(x2, 3)
    y2: torch.Tensor[torch.float32, L[3]] = x2.multinomial(3)


def test_bmm() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]
    matrix: torch.Tensor[torch.float32, L[2], L[4], L[5]]
    y: torch.Tensor[torch.float32, L[2], L[3], L[5]] = torch.bmm(x, matrix)
    y2: torch.Tensor[torch.float32, L[2], L[3], L[5]] = x.bmm(matrix)
    # pyre-fixme[9]: y_error has type `Tensor[torch.float32, typing_extensions.Litera...
    y_error: torch.Tensor[torch.float32, L[2], L[3], L[99]] = torch.bmm(x, matrix)

    bad_matrix: torch.Tensor[torch.float32, L[2], L[99], L[5]]
    # Should raise an error but doesn't because we solve `L[99] <: M && L[4] <:
    # M` to be M = int.
    torch.bmm(x, bad_matrix)


def test_subtract() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[1]]
    x2: torch.Tensor[torch.float32, L[2], L[1], L[4]]
    y: torch.Tensor[torch.float32, L[2], L[3], L[4]] = x - x2
    # pyre-fixme[9]: y_error has type `Tensor[torch.float32, typing_extensions.Litera...
    y_error: torch.Tensor[torch.float32, L[2], L[3], L[99]] = x - x2
    y2: torch.Tensor[torch.float32, L[2], L[3], L[4]] = x2 - x
    y3: torch.Tensor[torch.float32, L[2], L[3], L[1]] = x - 42.0
    y4: torch.Tensor[torch.float32, L[2], L[3], L[1]] = 42.0 - x

    z: Any
    # Should not error.
    x - z

    x5: Tensor[torch.float32, L[2], L[3]]
    x6: Tensor[torch.float32, L[2], L[3]]
    x6_bad: Tensor[torch.float32, L[2], L[99]]
    x5 -= x6
    x5 -= 4
    y5: Tensor[torch.float32, L[2], L[3]] = x5

    # pyre-fixme[2001]: Broadcast error at expression `x5.__isub__(x6_bad)`; types
    #  `Tuple[typing_extensions.Literal[2], typing_extensions.Literal[3]]` and
    #  `Tuple[typing_extensions.Literal[2], typing_extensions.Literal[99]]` cannot be
    #  broadcasted together.
    x5 -= x6_bad


def test_add() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[1]]
    x2: torch.Tensor[torch.float32, L[2], L[1], L[4]]
    y: torch.Tensor[torch.float32, L[2], L[3], L[4]] = x + x2
    # pyre-fixme[9]: y_error has type `Tensor[torch.float32, typing_extensions.Litera...
    y_error: torch.Tensor[torch.float32, L[2], L[3], L[99]] = x + x2
    y2: torch.Tensor[torch.float32, L[2], L[3], L[4]] = x2 + x
    y3: torch.Tensor[torch.float32, L[2], L[3], L[1]] = x + 42.0
    y4: torch.Tensor[torch.float32, L[2], L[3], L[1]] = 42.0 + x

    x5: Tensor[torch.float32, L[2], L[3]]
    x6: Tensor[torch.float32, L[2], L[3]]
    x6_bad: Tensor[torch.float32, L[2], L[99]]
    x5 += x6
    x5 += 4
    y5: Tensor[torch.float32, L[2], L[3]] = x5

    # pyre-fixme[2001]: Broadcast error at expression `x5.__iadd__(x6_bad)`; types
    #  `Tuple[typing_extensions.Literal[2], typing_extensions.Literal[3]]` and
    #  `Tuple[typing_extensions.Literal[2], typing_extensions.Literal[99]]` cannot be
    #  broadcasted together.
    x5 += x6_bad


def test_torch_fft() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]
    y: torch.Tensor[torch.complex64, L[2], L[3], L[4]] = torch.fft.fft(x)
    # pyre-fixme[9]: y_error has type `Tensor[torch.complex64, typing_extensions.Lite...
    y_error: torch.Tensor[torch.complex64, L[2], L[3], L[99]] = torch.fft.fft(x)
    y2: torch.Tensor[torch.complex64, L[2], L[3], L[4]] = torch.fft.fft(x, dim=-2)


def test_torch_real() -> None:
    x: torch.Tensor[torch.complex64, L[2], L[3], L[4]]
    y: torch.Tensor[torch.float32, L[2], L[3], L[4]] = torch.real(x)
    # pyre-fixme[9]: y_error has type `Tensor[torch.float32, typing_extensions.Litera...
    y_error: torch.Tensor[torch.float32, L[2], L[3], L[99]] = torch.real(x)
    x2: torch.Tensor[torch.complex128, L[2], L[3], L[4]]
    y2: torch.Tensor[torch.float64, L[2], L[3], L[4]] = torch.real(x2)
    bad: torch.Tensor[torch.float32, L[2], L[3], L[4]]
    # pyre-fixme[6]: Expected `Tensor[torch.complex64, *torch.Ts]` for 1st param but
    #  got `Tensor[torch.float32, int, int, int]`.
    torch.real(bad)


def test_logical_and() -> None:
    x: torch.Tensor[torch.complex64, L[2], L[1], L[4]]
    x2: torch.Tensor[torch.float32, L[2], L[3], L[1]]
    y: torch.Tensor[torch.bool, L[2], L[3], L[4]] = torch.logical_and(x, x2)
    # pyre-fixme[9]: y_error has type `Tensor[torch.bool, typing_extensions.Literal[2...
    y_error: torch.Tensor[torch.bool, L[2], L[3], L[99]] = torch.logical_and(x, x2)
    y2: torch.Tensor[torch.bool, L[2], L[3], L[4]] = x.logical_and(x2)
    not_broadcastable: torch.Tensor[torch.float32, L[2], L[3], L[99]]
    # pyre-fixme[2001]: Broadcast error at expression `torch.logical_and(x, not_broad...
    torch.logical_and(x, not_broadcastable)

    x3: torch.Tensor[torch.complex64, L[2], L[1], L[1]]
    # In-place version.
    x.logical_and_(x3)
    # This is actually an error because the output type (2, 3, 4) is not
    # assignable to x. But we can't catch that because the typechecker doesn't
    # know this is an in-place operator. Leaving this as is for now.
    x.logical_and_(x2)


def test_and() -> None:
    x_bool: torch.Tensor[torch.bool, L[2], L[1], L[4]]
    x_bool2: torch.Tensor[torch.bool, L[2], L[3], L[1]]
    y3: torch.Tensor[torch.bool, L[2], L[3], L[4]] = x_bool & x_bool2

    # This broadcasts to (2, 1, 4), which is assignable to x_bool.
    x_bool3: torch.Tensor[torch.bool, L[2], L[1], L[1]]
    x_bool &= x_bool3
    # This broadcasts to (2, 3, 4), which is not assignable to x_bool.
    # pyre-fixme[9]: x_bool has type `Tensor[torch.bool, typing_extensions.Literal[2]...
    x_bool &= x_bool2

    x: torch.Tensor[torch.complex64, L[2], L[1], L[4]]
    x2: torch.Tensor[torch.float32, L[2], L[3], L[1]]
    # pyre-fixme[58]: `&` is not supported for operand types
    #  `Tensor[torch.complex64, int, int, int]` and `Tensor[torch.float32, int, int,
    #  int]`.
    x & x2


def test_linalg_pinv() -> None:
    x: torch.Tensor[torch.float32, L[2], L[2], L[3], L[4]]
    y: torch.Tensor[torch.float32, L[2], L[2], L[4], L[3]] = torch.linalg.pinv(x)
    # pyre-fixme[9]: y_error has type `Tensor[torch.float32, typing_extensions.Litera...
    y_error: torch.Tensor[torch.float32, L[2], L[4], L[99]] = torch.linalg.pinv(x)
    wrong_datatype: torch.Tensor[torch.bool, L[2], L[3], L[4]]
    # pyre-fixme[6]: Expected `Tensor[Variable[torch.linalg.FloatOrDouble <:
    #  [torch.float32, torch.float64, torch.complex64, torch.complex128]],
    #  *torch.linalg.Ts, Variable[N1 (bound to int)], Variable[N2 (bound to int)]]` for
    #  1st param but got `Tensor[torch.bool, int, int, int]`.
    torch.linalg.pinv(wrong_datatype)

    torch.linalg.pinv(x, hermitian=True)
    # Last two dimensions have to be equal.
    x_square: torch.Tensor[torch.float32, L[2], L[3], L[4], L[4]]
    y2: torch.Tensor[torch.float32, L[2], L[3], L[4], L[4]] = torch.linalg.pinv(
        x_square, hermitian=True
    )


def test_linalg_qr() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]
    y: Tuple[
        torch.Tensor[torch.float32, L[2], L[3], L[3]],
        torch.Tensor[torch.float32, L[2], L[3], L[4]],
    ] = torch.linalg.qr(x)
    # pyre-fixme[9]: y_error has type `Tuple[Tensor[torch.float32, typing_extensions....
    y_error: Tuple[
        torch.Tensor[torch.float32, L[2], L[3], L[99]],
        torch.Tensor[torch.float32, L[2], L[3], L[4]],
    ] = torch.linalg.qr(x)
    y2: Tuple[
        torch.Tensor[torch.float32, L[2], L[3], L[3]],
        torch.Tensor[torch.float32, L[2], L[3], L[4]],
    ] = torch.linalg.qr(x, mode="complete")


def test_torch_matmul() -> None:
    x: torch.Tensor[torch.float32, L[2], L[1], L[3], L[4]]
    x2: torch.Tensor[torch.float32, L[1], L[5], L[4], L[3]]
    y: torch.Tensor[torch.float32, L[2], L[5], L[3], L[3]] = torch.matmul(x, x2)
    # pyre-fixme[9]: y_error has type `Tensor[torch.float32, typing_extensions.Litera...
    y_error: torch.Tensor[torch.float32, L[2], L[5], L[3], L[99]] = torch.matmul(x, x2)
    y2: torch.Tensor[torch.float32, L[2], L[5], L[3], L[3]] = x.matmul(x2)
    y3: torch.Tensor[torch.float32, L[2], L[5], L[3], L[3]] = x.__matmul__(x2)

    bad_x: torch.Tensor[torch.float32, L[1], L[5], L[99], L[3]]
    torch.matmul(x, bad_x)

    x_1d: torch.Tensor[torch.float32, L[3]]
    x2_1d: torch.Tensor[torch.float32, L[3]]
    y4: torch.Tensor[torch.float32] = torch.matmul(x_1d, x2_1d)
    x3_1d_different: torch.Tensor[torch.float32, L[1]]
    torch.matmul(x_1d, x3_1d_different)


def test_torch_optim() -> None:
    block_parameters: Any
    torch.optim.SGD(block_parameters, lr=1.0)


def test_torch_cuda() -> None:
    torch.cuda.reset_peak_memory_stats()


def test_torch_profiler() -> None:
    torch.profiler.profile()


def test_mse_loss() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3]]
    x2: torch.Tensor[torch.float32, L[2], L[3]]
    y: torch.Tensor[torch.float32] = nn.MSELoss(
        size_average=True, reduce=True, reduction="mean"
    )(x, x2)
    # pyre-fixme[9]: y_error has type `Tensor[torch.float32,
    #  typing_extensions.Literal[99]]`; used as `Tensor[torch.float32]`.
    y_error: torch.Tensor[torch.float32, L[99]] = nn.MSELoss()(x, x2)


def test_clip_grad_norm() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3]]
    y: torch.Tensor = nn.utils.clip_grad_norm_(
        x, max_norm=0.0, norm_type=0.0, error_if_nonfinite=True
    )
    # pyre-fixme[9]: y_error has type `int`; used as `Tensor[typing.Any,
    #  *Tuple[typing.Any, ...]]`.
    y_error: int = nn.utils.clip_grad_norm_(
        x, max_norm=0.0, norm_type=0.0, error_if_nonfinite=True
    )


def test_clip_grad_value() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3]]
    nn.utils.clip_grad_value_([x], clip_value=0.0)


def test_bitwise_not() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3]]
    y: torch.Tensor[torch.float32, L[2], L[3]] = torch.bitwise_not(x)
    # pyre-fixme[9]: y_error has type `Tensor[torch.float32,
    #  typing_extensions.Literal[2], typing_extensions.Literal[99]]`; used as
    #  `Tensor[torch.float32, typing_extensions.Literal[2],
    #  typing_extensions.Literal[3]]`.
    y_error: torch.Tensor[torch.float32, L[2], L[99]] = torch.bitwise_not(x)
    y2: torch.Tensor[torch.float32, L[2], L[3]] = x.bitwise_not()
    # In-place.
    y3: torch.Tensor[torch.float32, L[2], L[3]] = x.bitwise_not_()
    y4: torch.Tensor[torch.float32, L[2], L[3]] = ~x


def test_cdist() -> None:
    x: torch.Tensor[torch.float32, L[5], L[1], L[2], L[3]]
    x2: torch.Tensor[torch.float32, L[1], L[7], L[4], L[3]]
    y: torch.Tensor[torch.float32, L[5], L[7], L[2], L[4]] = torch.cdist(x, x2)
    # pyre-fixme[9]: y_error has type `Tensor[torch.float32, typing_extensions.Litera...
    y_error: torch.Tensor[torch.float32, L[5], L[7], L[2], L[99]] = torch.cdist(x, x2)

    not_broadcastable: torch.Tensor[torch.float32, L[99], L[1], L[2], L[3]]
    # pyre-fixme[2001]: Broadcast error at expression `torch.cdist(x,
    #  not_broadcastable)`; types `Tuple[typing_extensions.Literal[5],
    #  typing_extensions.Literal[1]]` and `Tuple[typing_extensions.Literal[99],
    #  typing_extensions.Literal[1]]` cannot be broadcasted together.
    torch.cdist(x, not_broadcastable)


def test_random_manual_seed() -> None:
    torch.random.manual_seed(42)


def test_clone() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3]]
    y: torch.Tensor[torch.float32, L[2], L[3]] = torch.clone(x)
    # pyre-fixme[9]: y_error has type `Tensor[torch.float32,
    #  typing_extensions.Literal[2], typing_extensions.Literal[99]]`; used as
    #  `Tensor[torch.float32, typing_extensions.Literal[2],
    #  typing_extensions.Literal[3]]`.
    y_error: torch.Tensor[torch.float32, L[2], L[99]] = torch.clone(x)
    y2: torch.Tensor[torch.float32, L[2], L[3]] = x.clone()


def test_equal() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3]]
    y: torch.Tensor[torch.bool, L[2], L[3]] = x == 42
    # pyre-fixme[9]: y_error has type `Tensor[torch.bool,
    #  typing_extensions.Literal[2], typing_extensions.Literal[99]]`; used as
    #  `Tensor[torch.bool, typing_extensions.Literal[2],
    #  typing_extensions.Literal[3]]`.
    y_error: torch.Tensor[torch.bool, L[2], L[99]] = x == 42
    # This doesn't return a Tensor as expected because `int.__eq__` accepts `object`.
    y2: int = 42 == x

    x2: torch.Tensor[torch.float32, L[2], L[1]]
    x3: torch.Tensor[torch.float32, L[1], L[3]]
    y3: torch.Tensor[torch.bool, L[2], L[3]] = x2 == x3


def test_diag_embed() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]
    y: torch.Tensor = torch.diag_embed(x)


def test_unbind() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]
    y: Tuple[torch.Tensor[torch.float32, L[2], L[4]], ...] = torch.unbind(x, dim=1)
    # pyre-fixme[9]: y_error has type `Tuple[Tensor[torch.float32,
    #  typing_extensions.Literal[2], typing_extensions.Literal[99]], ...]`; used as
    #  `Tuple[Tensor[torch.float32, typing_extensions.Literal[2],
    #  typing_extensions.Literal[4]], ...]`.
    y_error: Tuple[torch.Tensor[torch.float32, L[2], L[99]], ...] = torch.unbind(
        x, dim=1
    )
    y2: Tuple[torch.Tensor[torch.float32, L[2], L[3]], ...] = torch.unbind(x, dim=-1)
    y3: Tuple[torch.Tensor[torch.float32, L[3], L[4]], ...] = torch.unbind(x)
    y4: Tuple[torch.Tensor[torch.float32, L[3], L[4]], ...] = x.unbind()


def test_size() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]
    y: Tuple[L[2], L[3], L[4]] = x.size()
    # pyre-fixme[9]: y_error has type `Tuple[typing_extensions.Literal[2],
    #  typing_extensions.Literal[3], typing_extensions.Literal[99]]`; used as
    #  `Tuple[typing_extensions.Literal[2], typing_extensions.Literal[3],
    #  typing_extensions.Literal[4]]`.
    y_error: Tuple[L[2], L[3], L[99]] = x.size()
    y2: L[2] = x.size(0)
    y3: L[3] = x.size(1)
    y4: L[4] = x.size(-1)
    y5: L[3] = x.size(-2)


def test_stack(
    arbitary_length_tuple: Tuple[torch.Tensor[torch.float32, L[3], L[4], L[5]], ...],
    variadic_tuple: Tuple[Unpack[Ts]],
) -> None:
    x: torch.Tensor[torch.float32, L[3], L[4], L[5]]
    x_incompatible: torch.Tensor[torch.float32, L[3], L[4], L[99]]
    y: torch.Tensor[torch.float32, L[2], L[3], L[4], L[5]] = torch.stack((x, x))
    # pyre-fixme[9]: y_error has type `Tensor[torch.float32, typing_extensions.Litera...
    y_error: torch.Tensor[torch.float32, L[2], L[3], L[4], L[99]] = torch.stack((x, x))
    y_incompatible_tensors: torch.Tensor = torch.stack((x, x_incompatible))
    y2: torch.Tensor[torch.float32, L[3], L[2], L[4], L[5]] = torch.stack((x, x), dim=1)
    y3: torch.Tensor[torch.float32, L[3], L[3], L[4], L[5]] = torch.stack(
        (x, x, x), dim=1
    )
    y4: torch.Tensor[torch.float32, L[3], L[3], L[4], L[5]] = torch.stack((x, x, x))

    # Arbitrary-length tuples make it return an arbitrary Tensor.
    y5: torch.Tensor = torch.stack(arbitary_length_tuple)
    y6: torch.Tensor = torch.stack(variadic_tuple)


def test_repeat_interleave() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]
    repeats: torch.Tensor[torch.float32, L[2]]
    y: torch.Tensor[torch.float32, L[72]] = torch.repeat_interleave(x, 3)
    # pyre-fixme[9]: y_error has type `Tensor[torch.float32,
    #  typing_extensions.Literal[99]]`; used as `Tensor[torch.float32,
    #  typing_extensions.Literal[72]]`.
    y_error: torch.Tensor[torch.float32, L[99]] = torch.repeat_interleave(x, 3)
    y2: torch.Tensor[torch.float32, L[4], L[3], L[4]] = torch.repeat_interleave(
        x, 2, dim=0
    )
    y3: torch.Tensor[torch.float32, L[2], L[6], L[4]] = torch.repeat_interleave(
        x, 2, dim=1
    )
    y4: torch.Tensor[torch.float32, L[2], L[3], L[8]] = torch.repeat_interleave(
        x, 2, dim=-1
    )

    # Too dynamic because the output shape depends on the contents of repeats.

    y5: torch.Tensor[torch.float32, L[0], L[3], L[4]] = torch.repeat_interleave(
        x, repeats, dim=0
    )
    y6: torch.Tensor[torch.float32, L[2], L[3], L[8]] = x.repeat_interleave(2, dim=-1)


def test_meshgrid() -> None:
    x1: torch.Tensor[torch.float32, L[2]]
    x2: torch.Tensor[torch.float32, L[3]]
    x3: torch.Tensor[torch.float32, L[4]]
    y: Tuple[
        torch.Tensor[torch.float32, L[2], L[3], L[4]],
        torch.Tensor[torch.float32, L[2], L[3], L[4]],
        torch.Tensor[torch.float32, L[2], L[3], L[4]],
    ] = torch.meshgrid(x1, x2, x3)
    # pyre-fixme[9]: y_error has type `Tuple[Tensor[torch.float32, typing_extensions....
    y_error: Tuple[
        torch.Tensor[torch.float32, L[2], L[3], L[4]],
        torch.Tensor[torch.float32, L[2], L[3], L[4]],
        torch.Tensor[torch.float32, L[2], L[3], L[99]],
    ] = torch.meshgrid(x1, x2, x3)

    y2: Tuple[
        torch.Tensor[torch.float32, L[2], L[3]],
        torch.Tensor[torch.float32, L[2], L[3]],
    ] = torch.meshgrid(x1, x2)
    y3: Tuple[
        torch.Tensor[torch.float32, L[2]],
    ] = torch.meshgrid(x1)

    x4: Tensor
    xs = tuple(x4 for _ in range(5))
    y4: Tuple[torch.Tensor, ...] = torch.meshgrid(*xs)
    xs2 = [x4 for _ in range(5)]
    y5: Tuple[torch.Tensor, ...] = torch.meshgrid(*xs2)


def test_argmax() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]
    y: torch.LongTensor[torch.int64] = torch.argmax(x)
    # pyre-fixme[9]: y_error has type `Tensor[torch.int64,
    #  typing_extensions.Literal[99]]`; used as `Tensor[torch.int64]`.
    y_error: torch.LongTensor[torch.int64, L[99]] = torch.argmax(x)
    y2: torch.LongTensor[torch.int64, L[3], L[4]] = torch.argmax(x, dim=0)
    y3: torch.LongTensor[torch.int64, L[1], L[3], L[4]] = torch.argmax(
        x, dim=0, keepdim=True
    )
    y4: torch.LongTensor[torch.int64, L[2], L[4]] = torch.argmax(x, dim=1)
    y5: torch.LongTensor[torch.int64, L[2], L[1], L[4]] = torch.argmax(
        x, dim=1, keepdim=True
    )
    y6: torch.LongTensor[torch.int64, L[2], L[3]] = torch.argmax(x, dim=2)
    y7: torch.LongTensor[torch.int64, L[2], L[3], L[1]] = torch.argmax(
        x, dim=2, keepdim=True
    )
    y8: torch.LongTensor[torch.int64, L[2], L[3]] = torch.argmax(x, dim=-1)
    y9: torch.LongTensor[torch.int64, L[2], L[3], L[1]] = torch.argmax(
        x, dim=-1, keepdim=True
    )
    y10: torch.LongTensor[torch.int64, L[2], L[3], L[1]] = x.argmax(
        dim=-1, keepdim=True
    )

    # pyre-fixme[6]: Expected `typing_extensions.Literal[0]` for 2nd param but got
    #  `typing_extensions.Literal[3]`.
    torch.argmax(x, dim=3)


def test_argmin() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]
    y: torch.LongTensor[torch.int64] = torch.argmin(x)
    # pyre-fixme[9]: y_error has type `Tensor[torch.int64,
    #  typing_extensions.Literal[99]]`; used as `Tensor[torch.int64]`.
    y_error: torch.LongTensor[torch.int64, L[99]] = torch.argmin(x)
    y2: torch.LongTensor[torch.int64, L[3], L[4]] = torch.argmin(x, dim=0)
    y3: torch.LongTensor[torch.int64, L[1], L[3], L[4]] = torch.argmin(
        x, dim=0, keepdim=True
    )
    y4: torch.LongTensor[torch.int64, L[2], L[4]] = torch.argmin(x, dim=1)
    y5: torch.LongTensor[torch.int64, L[2], L[1], L[4]] = torch.argmin(
        x, dim=1, keepdim=True
    )
    y6: torch.LongTensor[torch.int64, L[2], L[3]] = torch.argmin(x, dim=2)
    y7: torch.LongTensor[torch.int64, L[2], L[3], L[1]] = torch.argmin(
        x, dim=2, keepdim=True
    )
    y8: torch.LongTensor[torch.int64, L[2], L[3]] = torch.argmin(x, dim=-1)
    y9: torch.LongTensor[torch.int64, L[2], L[3], L[1]] = torch.argmin(
        x, dim=-1, keepdim=True
    )
    y10: torch.LongTensor[torch.int64, L[2], L[3], L[1]] = x.argmin(
        dim=-1, keepdim=True
    )

    # pyre-fixme[6]: Expected `typing_extensions.Literal[0]` for 2nd param but got
    #  `typing_extensions.Literal[3]`.
    torch.argmin(x, dim=3)


def test_mean() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]
    y: torch.Tensor[torch.float32] = torch.mean(x)
    # pyre-fixme[9]: y_error has type `Tensor[torch.float32,
    #  typing_extensions.Literal[99]]`; used as `Tensor[torch.float32]`.
    y_error: torch.Tensor[torch.float32, L[99]] = torch.mean(x)
    y2: torch.Tensor[torch.float32, L[3], L[4]] = torch.mean(x, dim=0)
    y3: torch.Tensor[torch.float32, L[1], L[3], L[4]] = torch.mean(
        x, dim=0, keepdim=True
    )
    y4: torch.Tensor[torch.float32, L[2], L[4]] = torch.mean(x, dim=1)
    y5: torch.Tensor[torch.float32, L[2], L[1], L[4]] = torch.mean(
        x, dim=1, keepdim=True
    )
    y6: torch.Tensor[torch.float32, L[2], L[3]] = torch.mean(x, dim=2)
    y7: torch.Tensor[torch.float32, L[2], L[3], L[1]] = torch.mean(
        x, dim=2, keepdim=True
    )
    y8: torch.Tensor[torch.float32, L[2], L[3]] = torch.mean(x, dim=-1)
    y9: torch.Tensor[torch.float32, L[2], L[3], L[1]] = torch.mean(
        x, dim=-1, keepdim=True
    )
    y10: torch.Tensor[torch.float32, L[2], L[3], L[1]] = x.mean(dim=-1, keepdim=True)

    # pyre-fixme[6]: Expected `typing_extensions.Literal[0]` for 2nd param but got
    #  `typing_extensions.Literal[3]`.
    torch.mean(x, dim=3)


def test_count_nonzero() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]
    y: torch.Tensor[torch.int64] = torch.count_nonzero(x)
    # pyre-fixme[9]: y_error has type `Tensor[torch.int64,
    #  typing_extensions.Literal[99]]`; used as `Tensor[torch.int64]`.
    y_error: torch.Tensor[torch.int64, L[99]] = torch.count_nonzero(x)
    y2: torch.Tensor[torch.int64, L[3], L[4]] = torch.count_nonzero(x, dim=0)
    y3: torch.Tensor[torch.int64, L[2], L[4]] = torch.count_nonzero(x, dim=1)
    y4: torch.Tensor[torch.int64, L[2], L[3]] = torch.count_nonzero(x, dim=2)
    y5: torch.Tensor[torch.int64, L[2], L[3]] = x.count_nonzero(dim=-1)

    # pyre-fixme[6]: Expected `typing_extensions.Literal[0]` for 2nd param but got
    #  `typing_extensions.Literal[3]`.
    torch.count_nonzero(x, dim=3)


def test_cat() -> None:
    x1: torch.Tensor[torch.float32, L[2], L[3], L[4]]
    x1_first_is_3: torch.Tensor[torch.float32, L[3], L[3], L[4]]
    x1_first_is_4: torch.Tensor[torch.float32, L[4], L[3], L[4]]
    x1_second_is_4: torch.Tensor[torch.float32, L[2], L[4], L[4]]
    x1_second_is_5: torch.Tensor[torch.float32, L[2], L[5], L[4]]
    x1_last_is_5: torch.Tensor[torch.float32, L[2], L[3], L[5]]
    x1_last_is_6: torch.Tensor[torch.float32, L[2], L[3], L[6]]

    # 2-element tuple.
    y: torch.Tensor[torch.float32, L[5], L[3], L[4]] = torch.cat((x1, x1_first_is_3))
    # pyre-fixme[9]: y_error has type `Tensor[torch.float32, typing_extensions.Litera...
    y_error: torch.Tensor[torch.float32, L[99], L[3], L[4]] = torch.cat(
        (x1, x1_first_is_3)
    )
    y2: torch.Tensor[torch.float32, L[2], L[7], L[4]] = torch.cat(
        (x1, x1_second_is_4), dim=1
    )
    y3: torch.Tensor[torch.float32, L[2], L[3], L[9]] = torch.cat(
        (x1, x1_last_is_5), dim=-1
    )
    y3_shape_mismatch: torch.Tensor[torch.float32, Unpack[Tuple[Any, ...]]] = torch.cat(
        (x1, x1_second_is_4), dim=-1
    )

    # 3-element tuple.
    y4: torch.Tensor[torch.float32, L[9], L[3], L[4]] = torch.cat(
        (x1, x1_first_is_3, x1_first_is_4)
    )
    y5: torch.Tensor[torch.float32, L[2], L[12], L[4]] = torch.cat(
        (x1, x1_second_is_4, x1_second_is_5), dim=1
    )
    y6: torch.Tensor[torch.float32, L[2], L[3], L[15]] = torch.cat(
        (x1, x1_last_is_5, x1_last_is_6), dim=-1
    )

    y_many_element_tuple: torch.Tensor[
        torch.float32, Unpack[Tuple[Any, ...]]
    ] = torch.cat((x1, x1, x1, x1))
    y_list: torch.Tensor[torch.float32, Unpack[Tuple[Any, ...]]] = torch.cat([x1, x1])


def test_sign() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]
    y: torch.Tensor[torch.float32, L[2], L[3], L[4]] = torch.sign(x)
    # pyre-fixme[9]: y_error has type `Tensor[torch.float32, typing_extensions.Litera...
    y_error: torch.Tensor[torch.float32, L[2], L[3], L[99]] = torch.sign(x)
    y2: torch.Tensor[torch.float32, L[2], L[3], L[4]] = x.sign()


def test_diagonal() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4], L[5]]
    y: torch.Tensor = torch.diagonal(x)


def test_diag() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3]]
    y: torch.Tensor = torch.diag(x)


def test_module_list() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3]]

    modules = nn.ModuleList([nn.AdaptiveAvgPool2d(0), nn.AdaptiveAvgPool2d(1)])
    for module in modules:
        y: Tensor = module(x)

    z: int = len(modules)


def test_sparse_coo_tensor() -> None:
    y: torch.Tensor[torch.float32, L[2], L[3]] = torch.sparse_coo_tensor(
        torch.randn(5), [6, 7, 8], size=(2, 3)
    )
    # pyre-fixme[9]: y_error has type `Tensor[torch.float32,
    #  typing_extensions.Literal[2], typing_extensions.Literal[99]]`; used as
    #  `Tensor[torch.float32, typing_extensions.Literal[2],
    #  typing_extensions.Literal[3]]`.
    y_error: torch.Tensor[torch.float32, L[2], L[99]] = torch.sparse_coo_tensor(
        torch.randn(5), [6, 7, 8], size=(2, 3)
    )
    y2: torch.Tensor = torch.sparse_coo_tensor(torch.randn(5), [6, 7, 8])


def test_max() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]
    y: torch.Tensor[torch.float32] = torch.max(x)

    y2: torch.Tensor[torch.float32, L[3], L[4]] = torch.max(x, dim=0).values
    y2_indices: torch.Tensor[torch.int64, L[3], L[4]] = torch.max(x, dim=0).indices
    y2_getitem: torch.Tensor[torch.int64, L[3], L[4]] = torch.max(x, dim=0)[1]
    y3: torch.Tensor[torch.float32, L[1], L[3], L[4]] = torch.max(
        x, dim=0, keepdim=True
    ).values
    y4: torch.Tensor[torch.float32, L[2], L[4]] = torch.max(x, dim=1).values
    y5: torch.Tensor[torch.float32, L[2], L[1], L[4]] = torch.max(
        x, dim=1, keepdim=True
    ).values
    y6: torch.Tensor[torch.float32, L[2], L[3]] = torch.max(x, dim=2).values
    y7: torch.Tensor[torch.float32, L[2], L[3], L[1]] = torch.max(
        x, dim=2, keepdim=True
    ).values
    y8: torch.Tensor[torch.float32, L[2], L[3]] = torch.max(x, dim=-1).values
    y9: torch.Tensor[torch.float32, L[2], L[3], L[1]] = torch.max(
        x, dim=-1, keepdim=True
    ).values
    y10: torch.Tensor[torch.float32, L[2], L[4]] = torch.max(x, dim=-2).values
    y11: torch.Tensor[torch.float32, L[2], L[1], L[4]] = torch.max(
        x, dim=-2, keepdim=True
    ).values
    y12: torch.Tensor[torch.float32, L[2], L[3], L[1]] = x.max(
        dim=-1, keepdim=True
    ).values

    # pyre-fixme[6]: Expected `typing_extensions.Literal[0]` for 2nd param but got
    #  `typing_extensions.Literal[3]`.
    torch.max(x, dim=3).values


def test_einsum() -> None:
    x: Tensor = torch.einsum("ii", torch.randn(4, 4))


def test_type_as() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]
    x2: torch.Tensor[torch.int64, L[2], L[3], L[4]]
    y: torch.Tensor[torch.int64, L[2], L[3], L[4]] = x.type_as(x2)


def test_softmax() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]
    y: torch.Tensor[torch.float32, L[2], L[3], L[4]] = torch.softmax(x, dim=1)
    # pyre-fixme[9]: y_error has type `Tensor[torch.float32, typing_extensions.Litera...
    y_error: torch.Tensor[torch.float32, L[2], L[3], L[99]] = torch.softmax(x, dim=1)
    y2: torch.Tensor[torch.int64, L[2], L[3], L[4]] = torch.softmax(
        x, dim=1, dtype=torch.int64
    )
    y3: torch.Tensor[torch.float32, L[2], L[3], L[4]] = x.softmax(dim=1)


def test_conv2d() -> None:
    x: Tensor[torch.float32, L[20], L[16], L[50], L[100]]

    y7: Tensor[torch.float32, L[20], L[33], L[56], L[100]] = nn.Conv2d(
        16, 33, (3, 5), padding=(4, 2), bias=False
    )(x)
    # pyre-fixme[9]: y7_error has type `Tensor[torch.float32, typing_extensions.Liter...
    y7_error: Tensor[torch.float32, L[20], L[33], L[56], L[99]] = nn.Conv2d(
        16, 33, (3, 5), padding=(4, 2)
    )(x)

    module: nn.Module = nn.Conv2d(16, 33, (3, 5), padding=(4, 2))


def test_nn_Parameter() -> None:
    x: Tensor[torch.float32, L[2], L[3], L[4]]

    y: Tensor[torch.float32, L[2], L[3], L[4]] = nn.Parameter(x)
    # pyre-fixme[9]: Expected error.
    y_error: Tensor[torch.float32, L[2], L[3], L[99]] = nn.Parameter(x)


def test_torch_datatypes() -> None:
    x: torch.float16
    x2: torch.int


def test_norm() -> None:
    x: Tensor[torch.float32, L[2], L[3], L[4]]
    x_out: Tensor[torch.float32, L[2], L[3], L[4]]

    y1: Tensor[torch.float32] = torch.norm(x)
    y2: Tensor[torch.float32, L[3], L[4]] = torch.norm(x, dim=0, out=x_out, p=1)
    # pyre-fixme[9]: Expected error.
    y2_error: Tensor[torch.float32, L[3], L[99]] = torch.norm(x, dim=0)
    y3: Tensor[torch.float32, L[1], L[3], L[4]] = torch.norm(x, dim=0, keepdim=True)


def test_rand() -> None:
    x: Tensor[torch.float32, L[2], L[3], L[4]]
    x_out: Tensor[torch.float32, L[2], L[3], L[4]]
    device: torch.device

    y1: Tensor[torch.float32, L[2], L[3], L[4]] = torch.rand(2, 3, 4)
    # pyre-fixme[9]: Expected Error.
    y1_error: Tensor[torch.float32, L[2], L[3], L[99]] = torch.rand(2, 3, 4)
    y2: Tensor[torch.int64, L[2], L[3], L[4]] = torch.rand(
        2,
        3,
        4,
        dtype=torch.int64,
        device=device,
        layout=torch.strided,
        out=x_out,
        requires_grad=True,
        generator=torch.default_generator,
    )
    y3: Tensor[torch.float32, L[2], L[3], L[4]] = torch.rand((2, 3, 4))


def test_randint() -> None:
    x: Tensor[torch.float32, L[2], L[3], L[4]]
    x_out: Tensor[torch.float32, L[2], L[3], L[4]]
    device: torch.device

    y1: Tensor[torch.int64, L[2], L[3], L[4]] = torch.randint(0, 3, (2, 3, 4))
    # pyre-fixme[9]: Expected error.
    y1_error: Tensor[torch.int64, L[2], L[3], L[99]] = torch.randint(0, 3, (2, 3, 4))
    y2: Tensor[torch.int64, L[2], L[3], L[4]] = torch.randint(
        3,
        (2, 3, 4),
        dtype=torch.int64,
        device=device,
        layout=torch.strided,
        out=x_out,
        requires_grad=True,
        generator=torch.default_generator,
    )


def test_zeros() -> None:
    x: Tensor[torch.float32, L[2], L[3], L[4]]
    x_out: Tensor[torch.float32, L[2], L[3], L[4]]
    device: torch.device

    y1: Tensor[torch.float32, L[2], L[3], L[4]] = torch.zeros(2, 3, 4)
    # pyre-fixme[9]: Expected Error.
    y1_error: Tensor[torch.float32, L[2], L[3], L[99]] = torch.zeros(2, 3, 4)
    y2: Tensor[torch.int64, L[2], L[3], L[4]] = torch.zeros(
        2,
        3,
        4,
        dtype=torch.int64,
        device=device,
        layout=torch.strided,
        out=x_out,
        requires_grad=True,
    )
    y3: Tensor[torch.float32, L[2], L[3], L[4]] = torch.zeros((2, 3, 4))


def test_stride() -> None:
    x: Tensor[torch.float32, L[2], L[3], L[4]]

    y: Tuple[L[2], L[3], L[4]] = x.stride()
    # pyre-fixme[9]: Expected error.
    y_error: Tuple[L[2], L[3], L[99]] = x.stride()
    y2: L[12] = x.stride(0)
    y3: L[4] = x.stride(1)
    y4: L[1] = x.stride(2)


def test_chunk() -> None:
    x: Tensor[torch.float32, L[2], L[3], L[4]]

    y: Tuple[
        Tensor[torch.float32, L[2], L[3], L[2]], Tensor[torch.float32, L[2], L[3], L[2]]
    ] = torch.chunk(x, 2, dim=-1)
    # pyre-fixme[9]: Expected error.
    y_error: Tuple[
        Tensor[torch.float32, L[2], L[3], L[99]],
        Tensor[torch.float32, L[2], L[3], L[2]],
    ] = torch.chunk(x, 2, dim=-1)
    y2: Tuple[
        Tensor[torch.float32, L[1], L[3], L[4]], Tensor[torch.float32, L[1], L[3], L[4]]
    ] = torch.chunk(x, 2, dim=0)
    y3: Tuple[
        Tensor[torch.float32, L[1], L[3], L[4]], Tensor[torch.float32, L[1], L[3], L[4]]
    ] = x.chunk(2, dim=0)


def test_abs() -> None:
    x: Tensor[torch.float32, L[2], L[3], L[4]]

    y: Tensor[torch.float32, L[2], L[3], L[4]] = x.abs()
    # pyre-fixme[9]: Expected error.
    y_error: Tensor[torch.float32, L[2], L[3], L[99]] = x.abs()


def test_enable_grad() -> None:
    with torch.enable_grad():
        pass


def test_normal() -> None:
    y: Tensor[torch.float32, L[2], L[3], L[4]] = torch.normal(
        0, 1, size=(2, 3, 4), device="cuda", requires_grad=True
    )
    # pyre-fixme[9]: Expected error.
    y_error: Tensor[torch.float32, L[2], L[3], L[99]] = torch.normal(
        0, 1, size=(2, 3, 4), device="cuda", requires_grad=True
    )


def test_dim() -> None:
    x0: Tensor[torch.float32]
    x1: Tensor[torch.float32, L[2]]
    x2: Tensor[torch.float32, L[2], L[3]]
    x3: Tensor[torch.float32, L[2], L[3], L[4]]

    y: L[3] = x3.dim()
    # pyre-fixme[9]: Expected error.
    y_error: L[5] = x3.dim()
    y2: L[0] = x0.dim()
    y3: L[1] = x1.dim()
    y4: L[2] = x2.dim()


def test_is_cuda() -> None:
    x: Tensor[torch.float32, L[2], L[3], L[4]]
    y: bool = x.is_cuda


def test_autograd_backward() -> None:
    x: Tensor[torch.float32, L[2], L[3], L[4]]
    torch.autograd.backward(x, x)


def test_linalg_norm() -> None:
    x: Tensor[torch.float32, L[2], L[3], L[4]]
    y: Tensor[torch.float32, L[2]] = torch.linalg.norm(x, dim=(-2, -1))
    # pyre-fixme[9]: Expected error.
    y_error: Tensor[torch.float32, L[99]] = torch.linalg.norm(x, dim=(-2, -1))


def test_Sized() -> None:
    x: torch.Size = torch.Size((2, 3, 4))


def test_initial_seed() -> None:
    x: int = torch.initial_seed()


def test_log_softmax() -> None:
    x: Tensor[torch.float32, L[2], L[3], L[4]]

    y: Tensor[torch.float32, L[2], L[3], L[4]] = torch.log_softmax(x, dim=1)
    # pyre-fixme[9]: Expected error.
    y_error: Tensor[torch.float32, L[2], L[3], L[99]] = torch.log_softmax(x, dim=1)
    y2: Tensor[torch.int64, L[2], L[3], L[4]] = torch.log_softmax(
        x, dtype=torch.int64, dim=1
    )


def test_masked_select() -> None:
    x: Tensor[torch.float32, L[2], L[3], L[4]]
    mask: Tensor[torch.bool, L[2], L[3], L[4]]
    out: Tensor[torch.float32, L[2], L[3], L[4]]

    y: Tensor = x.masked_select(mask, out=out)
    y2: Tensor = torch.masked_select(x, mask, out=out)


def test__lt__() -> None:
    x: Tensor[torch.float32, L[2], L[3], L[4]]

    y: Tensor[torch.bool, L[2], L[3], L[4]] = x < 3.0


def test_pow() -> None:
    x: Tensor[torch.float32, L[2], L[3], L[4]]

    y: Tensor[torch.float32, L[2], L[3], L[4]] = x**4
    # pyre-fixme[9]: Expected error.
    y_error: Tensor[torch.float32, L[2], L[3], L[99]] = x**4


def test_item() -> None:
    x: Tensor[torch.float32]
    x2: Tensor[torch.float32, L[1]]

    y: torch.float32 = x.item()
    # pyre-fixme[9]: Expected error.
    y_error: torch.int64 = x.item()
    y2: torch.float32 = x.item()


def test_uniform_() -> None:
    x: Tensor[torch.float32, L[2], L[3], L[4]]
    y: Tensor[torch.float32, L[2], L[3], L[4]] = nn.init.uniform_(x, a=1.0, b=2.0)
    # pyre-fixme[9]: Expected error.
    y_error: Tensor[torch.float32, L[2], L[3], L[99]] = nn.init.uniform_(
        x, a=1.0, b=2.0
    )


def test_kaiming_uniform_() -> None:
    x: Tensor[torch.float32, L[2], L[3], L[4]]
    y: Tensor[torch.float32, L[2], L[3], L[4]] = nn.init.kaiming_uniform_(
        x, a=1.0, mode="fan_in", nonlinearity="leaky_relu"
    )
    # pyre-fixme[9]: Expected error.
    y_error: Tensor[torch.float32, L[2], L[3], L[99]] = nn.init.kaiming_uniform_(x)


def test_constant_() -> None:
    x: Tensor[torch.float32, L[2], L[3], L[4]]
    y: Tensor[torch.float32, L[2], L[3], L[4]] = nn.init.constant_(x, val=1.0)
    # pyre-fixme[9]: Expected error.
    y_error: Tensor[torch.float32, L[2], L[3], L[99]] = nn.init.constant_(x, val=1.0)


def test_leaky_relu() -> None:
    x: Tensor[torch.float32, L[2], L[3], L[4]]
    y: Tensor[torch.float32, L[2], L[3], L[4]] = nn.LeakyReLU(
        negative_slope=1.0, inplace=True
    )(x)
    # pyre-fixme[9]: Expected error.
    y_error: Tensor[torch.float32, L[2], L[3], L[99]] = nn.LeakyReLU(
        negative_slope=1.0, inplace=True
    )(x)


def test_fft_fft2() -> None:
    x: Tensor[torch.complex64, L[2], L[3], L[4]]
    y: Tensor[torch.complex64, L[2], L[3], L[4]] = torch.fft.fft2(x)
    # pyre-fixme[9]: Expected error.
    y_error: Tensor[torch.complex64, L[2], L[3], L[99]] = torch.fft.fft2(x)


def test_real() -> None:
    x: Tensor[torch.complex64, L[2], L[3], L[4]]
    y: Tensor[torch.float32, L[2], L[3], L[4]] = x.real
    # pyre-fixme[9]: Expected error.
    y_error: Tensor[torch.float32, L[2], L[3], L[99]] = x.real
    x2: Tensor[torch.complex128, L[2], L[3], L[4]]
    y2: Tensor[torch.float64, L[2], L[3], L[4]] = x2.real

    not_complex: Tensor[torch.float64, L[2], L[3], L[4]]
    # Should error but we don't have overloads for @property.
    not_complex.real


def test_Tensor_init() -> None:
    x: Tensor[torch.float32, L[2], L[3], L[4]]

    # pyre-fixme[9]: Unexpected error because the constructor doesn't bind DType.
    y: Tensor[torch.float32, L[2], L[3], L[4]] = Tensor((2, 3, 4), device="cuda")
    # pyre-fixme[9]: Expected error.
    y_error: Tensor[torch.float32, L[2], L[3], L[99]] = Tensor((2, 3, 4), device="cuda")
    y2: Tensor[torch.float32, L[2], L[3], L[4]] = Tensor(2, 3, 4, device="cuda")
    y3: Tensor[torch.float32, L[2], L[3], L[4]] = Tensor(x)


def test_reflection_pad2d() -> None:
    module: nn.Module = nn.ReflectionPad2d(4)
    x: Tensor[torch.float32, L[20], L[16], L[50], L[100]]

    y: Tensor[torch.float32, L[20], L[16], L[58], L[108]] = nn.ReflectionPad2d(4)(x)
    # pyre-fixme[9]: Expected error.
    y_error: Tensor[torch.float32, L[20], L[16], L[58], L[99]] = nn.ReflectionPad2d(4)(
        x
    )


def test_half() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]

    good1: torch.Tensor[torch.float16, L[2], L[3], L[4]] = x.half(torch.memory_format())
    # pyre-fixme[9]: Expected error.
    bad1: torch.Tensor[torch.float16, L[99], L[3], L[4]] = x.half()


def test_is_contiguous() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]
    y: bool = x.is_contiguous(torch.memory_format())


def test_scatter() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]

    # We don't really check for the shape of index or src.
    index: torch.LongTensor[torch.float32, L[99]]
    src: torch.Tensor[torch.float32, L[99], L[99]]
    y: torch.Tensor[torch.float32, L[2], L[3], L[4]] = x.scatter(0, index, src)
    # pyre-fixme[9]: Expected error.
    y_error: torch.Tensor[torch.float32, L[2], L[3], L[99]] = x.scatter(0, index, src)
    y2: torch.Tensor[torch.float32, L[2], L[3], L[4]] = x.scatter(2, index, src)


def test_scatter_() -> None:
    x: torch.Tensor[torch.float32, L[2], L[3], L[4]]

    # We don't really check for the shape of index or src.
    index: torch.LongTensor[torch.float32, L[99]]
    src: torch.Tensor[torch.float32, L[99], L[99]]
    y: torch.Tensor[torch.float32, L[2], L[3], L[4]] = x.scatter_(0, index, src)
    # pyre-fixme[9]: Expected error.
    y_error: torch.Tensor[torch.float32, L[2], L[3], L[99]] = x.scatter_(0, index, src)
    y2: torch.Tensor[torch.float32, L[2], L[3], L[4]] = x.scatter_(2, index, src)


def test_bool() -> None:
    x: Tensor[torch.float32, L[2], L[3], L[4]]

    y: Tensor[torch.bool, L[2], L[3], L[4]] = x.bool()
    # pyre-fixme[9]: Expected error.
    y_error: Tensor[torch.bool, L[2], L[3], L[99]] = x.bool()
