Custom parts reference
======================

Sparse CUDA kernels
####################


1. Building the kernels
***********************

xFormers transparently supports CUDA kernels to implement sparse attention computations, some of which are based on Sputnik_.
These kernels require xFormers to be installed from source, and the recipient machine to be able to compile CUDA source code.

.. code-block:: bash

    git clone git@github.com:fairinternal/xformers.git
    conda create --name xformer_env python=3.8
    conda activate xformer_env
    cd xformers
    pip install -r requirements.txt
    pip install -e .



Common issues are related to:

* NVCC and the current CUDA runtime match. You can often change the CUDA runtime with `module unload cuda module load cuda/xx.x`, possibly also `nvcc`
* the version of GCC that you're using matches the current NVCC capabilities
* the `TORCH_CUDA_ARCH_LIST` env variable is set to the architures that you want to support. A suggested setup (slow to build but comprehensive) is `export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.2;8.0;8.6"`

2. Usage
********

The sparse attention computation is automatically triggered when using the **scaled dot product** attention (see_), and a sparse enough mask (currently less than 30% of true values).
There is nothing specific to do, and a couple of examples are provided in the tutorials.


.. _Triton: https://triton-lang.org/
.. _Sputnik: https://github.com/google-research/sputnik
.. _see: https://github.com/facebookresearch/xformers/blob/main/xformers/components/attention/scaled_dot_product.py
