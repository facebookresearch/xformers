Using BlockSparseAttention
==========================

BlockSparse attention uses Triton_ to limit the attention computations to some tiles, which you define at construction time.
A simple example is that of a causal attention: just compute the lower triangular tiles! The tile size can be changed, the minimum being 16 coefficients on one dimension.

.. _Triton: https://github.com/openai/triton

If you already have a per-coefficient pattern in mind and this is not a perfect match with a block pattern, this is probably fine,
BlockSparse is fast enough so that dropping some of the computations after the fact with a fine-grained mask is still probably better than dense computations.
We provide a small helper (this is just maxpooling) to convert in between a per coefficient binary mask and the layout that you will need to build a block sparse attention.

*Please note that for now blocksparse attention requires the sequence length to be a power of two*.

Let's look at an example:

.. code-block:: python

    import torch

    from xformers.components import MultiHeadDispatch
    from xformers.components.attention import BlockSparseAttention

    BATCH = 2
    HEADS = 8
    SEQ = 2048
    EMB = 1024
    BLOCK_SIZE = 32
    DROPOUT = 0.1
    dtype = torch.float16

    # Let's try out a causal mask, but really it could be anything "block sparse enough"
    causal_mask = torch.tril(torch.ones((SEQ, SEQ), device=torch.device("cuda"), dtype=dtype))

    blocks = SEQ // BLOCK_SIZE
    causal_layout = torch.tril(torch.ones([HEADS, blocks, blocks]))

    # Let's build our blocksparse attention. Please note that the layout can be
    # [SEQ//BLOCK_SIZE, SEQ//BLOCK_SIZE] or  [HEADS, SEQ//BLOCK_SIZE, SEQ//BLOCK_SIZE]
    # so that _you can pass a different layout per head_
    attention = BlockSparseAttention(layout=causal_layout, block_size=BLOCK_SIZE, dropout=DROPOUT, num_heads=HEADS)

    # Out of commodity, let's build our multihead attention now
    # "multi_head" will be responsible for the forward
    multi_head = (
        MultiHeadDispatch(
            seq_len=SEQ,
            dim_model=EMB,
            residual_dropout=DROPOUT,
            num_heads=HEADS,
            attention=attention,
        )
        .cuda()
        .half()
    )

    # Now FW some random data
    # Note that passing a per-coefficient mask makes it possible to remove extra coefficients,
    # which where required by the blockification
    query = torch.randn((BATCH, SEQ, EMB), requires_grad=True, device=torch.device("cuda"), dtype=dtype)

    # Self attention in this particular example, no limitations really
    att_val = multi_head(query=query, key=query, value=query, att_mask=causal_mask)


    #########################################
    # Bonus: compare the memory use vs dense:
    def mem_use(fn, kwargs, title):
        # bookeeping
        import time

        start = time.time()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # actually run the function
        fn(**kwargs)
        torch.cuda.synchronize()
        stop = time.time()

        # now report
        max_memory = torch.cuda.max_memory_allocated() // 2 ** 20
        print(f"{title} - Peak memory use: {max_memory}MB - {round((stop-start)*1e6)/1e3}ms")


    pytorch_multihead = torch.nn.MultiheadAttention(
        EMB, HEADS, batch_first=True, device=torch.device("cuda"), dtype=torch.float16
    )

    mem_use(multi_head, {"query": query, "key": query, "value": query, "att_mask": causal_mask}, "Blocksparse")
    mem_use(pytorch_multihead, {"query": query, "key": query, "value": query, "attn_mask": causal_mask}, "PyTorch")

On a V100, with PyTorch 1.9, Triton 1.1 and xFormers 0.0.2 this reports something along the lines of:

.. code-block:: bash

    Blocksparse - Peak memory use: 151MB - 6.619ms
    PyTorch - Peak memory use: 393MB - 6.837ms

Note that the pattern here is not that sparse (half of the matrix is empty), the more sparse it gets the more biased the result will get towards BlockSparseAttention.
