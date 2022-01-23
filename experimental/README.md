# Experimental

The kernels in this folder are experimental, and for now not tied to the rest of the repo.

## Requirements

- Checkout the [Triton v2.0 branch](git@github.com:openai/triton.git)
`git clone git@github.com:openai/triton.git; cd triton; git checkout v2.0`

- Install it in a dedicated environement. Note that you will need pytorch, and a matching GCC/G++/CUDA
`cd triton/python; pip install -e .`

- (Optional) install the kernels present here by running `pip install -e .` in this folder

- (Optional) Check that the installation is successful by running `pytest tests` in this folder
