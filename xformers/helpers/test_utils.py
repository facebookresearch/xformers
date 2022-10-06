# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import tempfile

import torch

import os
is_windows = False
if (os.environ.get('OS','') == 'Windows_NT'): #pytorch on windows uses gloo not ncll
    is_windows = True

def init_torch_distributed_local():
    if torch.distributed.is_initialized():
        return

    init_url = "file://" + tempfile.mkstemp()[1]
    backend = (
        torch.distributed.Backend.NCCL
        if torch.cuda.is_available() and not is_windows
        else torch.distributed.Backend.GLOO
    )
    torch.distributed.init_process_group(
        backend=backend,
        rank=0,
        world_size=1,
        init_method=init_url,
    )
