
   1. #> pip install -e ./

   2. verify testing for generic fmha inference on ROCM

      #> pytest tests/test_mem_eff_attention.py::test_forward

   3. verify testing for decoder fmha inference on ROCM

      #> pytest tests/test_mem_eff_attention.py::test_decoder
      #> pytest tests/test_mem_eff_attention.py::test_splitk_decoder
