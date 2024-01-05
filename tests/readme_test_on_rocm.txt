
   1. pip install -e ./

   2. verify testing for memory_efficient_attention inference 

      pytest tests/test_mem_eff_attention_ck.py::test_forward
      pytest tests/test_mem_eff_attention.py::test_forward -k ckF 

   3. The following tests in tests/memory_eff_attention_ck.py have passed

      * test_forward
      * test_key_query_all_ones
      * test_logsumexp
      * test_attn_bias 
        - test_attn_bias_causal
        - test_attn_bias_torch_tensor 
        - test_attn_bias_blockdiag
        - test_attn_bias_blockdiag_batched
        - test_attn_bias_blockdiag_crossattn_causal
        - test_attn_bias_blockdiag_crossattn_causal_with_prefix_qk_cond
        - test_attn_bias_blockdiag_crossattn_causal_with_prefix()
        - test_attn_bias_padded
        - test_attn_bias_from_seqlens
        - test_attn_bias_blockdiag_doc
      * test_unsupported_cpu
      * test_unsupported_stride_lastdim
      * test_unsupported_stride_alignment
      * test_cuda_streams
      * test_dropout
      * test_backward
      * test_decoder

   4. verify testing for memory_efficient_attention forward (with dropout)

      pytest tests/test_mem_eff_attention_ck.py::test_dropout

