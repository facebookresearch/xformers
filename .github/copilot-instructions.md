---
description: AI rules derived by SpecStory from the project AI interaction history
globs: *
---

## PROJECT RULES & CODING STANDARDS

## WORKFLOW & RELEASE RULES

## TECH STACK
- PyTorch (verified version 2.8.0 or higher)
- OpenMP (required for xFormers installation on macOS, install via `brew install libomp`)
- xFormers (installed from source, version 0.0.33+c159edc0.d20250903 or later)

## PROJECT DOCUMENTATION & CONTEXT SYSTEM

## DEBUGGING

## BEST PRACTICES
- When installing xFormers from source on macOS, ensure OpenMP is installed and the `setup.py` file is modified to use the correct compiler flags (`-Xpreprocessor -fopenmp -lomp`).
- Be aware that xFormers is primarily designed for CUDA-enabled systems, and some advanced features may not be available without CUDA. When running on Apple Silicon, focus on utilizing Metal (MPS) for GPU acceleration.
- When building xFormers from source on macOS, use `pip install -v --no-build-isolation -e .` to build and install it in editable mode.
- Basic Metal (MPS) tensor detection has been implemented in `dispatch.py` for xFormers. MPS tensors are now routed to MPS operators.
- When running on Apple Silicon, avoid unnecessary checks for CUDA availability, and prioritize leveraging Metal (MPS) for GPU acceleration where possible.
- xFormers now detects MPS tensors and routes them to MPS operators.
- The `memory_efficient_attention` function works on MPS devices.
- Causal masks work correctly with MPS.
- Tensors stay on MPS throughout the computation.
- MPS-specific operators should be prioritized for MPS tensors.
- When working on Apple Silicon, do not check for CUDA availability. Focus on Metal (MPS) acceleration.