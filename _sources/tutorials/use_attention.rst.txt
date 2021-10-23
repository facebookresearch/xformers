I'm only interested in testing out the attention mechanisms that are hosted here
================================================================================


That's completely fine! There are two paths to do this:

- Either you import the attention mechanisms that you're interested in directly
    in your code base, their API should be very similar and you would own everything.
    The dimensions expectations are that, depending on whether the attentions expose the `requires_head_dimension` flag,
    the input data would be either `[Batch, Heads, Sequence, Head dimension]`, or `[Batch x Heads, Sequence, Head dimension]`.

- Alternatively, a `build_attention` helper is provided, which takes a dict as an input.
    In that case, you defer a lot of the instantiation work to xFormers,
    which makes it a little more obscure although the parameters are hopefully straightforward.
    This was initially built for internal use in xFormers, to make sure that we can programatically
    build and test all possible combinations.
    In turn this should allow you to do sweeps or architecture search, given that the multihead attention definition
    becomes something like:

.. code-block:: python

  from xformers.components import MultiHeadDispatch, build_attention
  SEQ = 1024
  MODEL = 384
  HEADS = 16
  DROPOUT = 0.1

  my_config = {
      "name": attention_name,  # you can easily make this dependent on a file, sweep,..
      "dropout": DROPOUT,
      "seq_len": SEQ,
      "attention_query_mask": torch.rand((SEQ, 1)) < 0.3, # some dummy mask
  }

  attention = build_attention(my_config)

  # build a multi head dispatch to test this attention mechanism
  multi_head = MultiHeadDispatch(
      seq_len=SEQ,
      dim_model=MODEL,
      residual_dropout=DROPOUT,
      num_heads=HEADS,
      attention=attention,
  ).to(device)

  # do something with my new multi-head attention
  #...
