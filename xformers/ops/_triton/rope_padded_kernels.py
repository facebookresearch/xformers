# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import triton  # type: ignore
import triton.language as tl  # type: ignore

from xformers.triton.importing import libdevice_find

pow = libdevice_find("pow")


@triton.jit
def _rope_padded_kernel(
    xq,
    xk,
    xv,
    out_q,
    cache_k,
    cache_v,
    seqstartq,
    seqstartk,
    seqlenk,
    theta,
    linear_scale,
    use_dynamic_scaling: tl.constexpr,
    dynamic_old_context_len: tl.constexpr,
    dynamic_scale_factor: tl.constexpr,
    dynamic_low_freq_factor: tl.constexpr,
    dynamic_high_freq_factor: tl.constexpr,
    first_seqpos,
    seqpos,
    k_start: tl.constexpr,
    v_start: tl.constexpr,
    n_groups,
    dim: tl.constexpr,  # dimension of each head
    stride_xqM,
    stride_xqG,
    stride_xqH,
    stride_xkM,
    stride_xkG,
    stride_xkH,
    stride_xvM,
    stride_xvG,
    stride_xvH,
    stride_cachekM,
    stride_cachekG,
    stride_cachekH,
    stride_cachevM,
    stride_cachevG,
    stride_cachevH,
    stride_seqstartq,
    stride_seqstartk,
    stride_seqlenk,
    stride_outqM,
    stride_outqG,
    stride_outqH,
    stride_seqpos,
    internal_dtype: tl.constexpr,
    # If True, seqstartq and seqstartk are not used but rather we
    # assume that every batch element has the same number of
    # queries (i.e. num_queries := tl.num_programs(1) )
    # and the same cache space cache_padding_length.
    # Always False when called below.
    const_batch_strides: tl.constexpr,
    # If const_batch_strides==True, the common cache length for each batch element.
    # (Only the first seqlenk[i] elements are actually in use, and only the last
    #  num_queries of those are actually written to.)
    cache_padding_length,
    # offset added to all values in seqlenk before using them.
    # Always 0 when called below.
    seqlenk_shift: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    adjacents: tl.constexpr,
):
    """
    Each letter in this diagram is a whole row of length dim.

     INPUT      xq        xk       xv

        head_dim ─►

      batch   qqqqqq      kk       vv
        │     qqqqqq      kk       vv
        ▼     qqqqqq      kk       vv

    head_idx:  (goes across all heads of all 3 inputs)
              ▲     ▲     ▲ ▲      ▲ ▲
              │     │     │ │      │ │
                          │        │
              0  k_start  │v_start │n_total_heads
                          │        │
                          │        │
                      k_start    v_start

    Output is to out_q (same shape as xq), an xk-shaped part
    of cache_k and an xv-shaped part of cache_v
    """
    query_pos_in_batch_elt = tl.program_id(0)
    batch_elt = tl.program_id(1)
    group_head_idx = tl.program_id(2)
    group_idx = group_head_idx % n_groups
    head_idx = group_head_idx // n_groups

    if internal_dtype == "f32":
        theta = theta.to(tl.float32)
    elif internal_dtype == "f64":
        theta = theta.to(tl.float64)

    if const_batch_strides:
        query_pos = query_pos_in_batch_elt + tl.num_programs(1) * batch_elt
        end_query_pos = tl.num_programs(1) * (batch_elt + 1)
    else:
        query_pos = query_pos_in_batch_elt + tl.load(
            seqstartq + batch_elt * stride_seqstartq
        )
        end_query_pos = tl.load(seqstartq + (batch_elt + 1) * stride_seqstartq)
        if query_pos >= end_query_pos:
            return

    is_q = head_idx < k_start
    is_v = head_idx >= v_start

    xq += query_pos * stride_xqM + head_idx * stride_xqH + group_idx * stride_xqG
    out_q += (
        query_pos * stride_outqM + head_idx * stride_outqH + group_idx * stride_outqG
    )

    if const_batch_strides:
        cache_start = cache_padding_length * batch_elt
    else:
        cache_start = tl.load(seqstartk + batch_elt * stride_seqstartk)
    end_of_batch_elt_cache = (
        cache_start + tl.load(seqlenk + batch_elt * stride_seqlenk) + seqlenk_shift
    )

    cache_pos = end_of_batch_elt_cache - (end_query_pos - query_pos)
    if seqpos is not None:
        seq_pos = tl.load(seqpos + query_pos * stride_seqpos)
    else:
        seq_pos = cache_pos - cache_start
        if first_seqpos is not None:
            seq_pos += tl.load(first_seqpos + batch_elt * stride_seqpos)
    cache_k += (
        (head_idx - k_start) * stride_cachekH
        + cache_pos * stride_cachekM
        + group_idx * stride_cachekG
    )
    xk += (
        query_pos * stride_xkM
        + (head_idx - k_start) * stride_xkH
        + group_idx * stride_xkG
    )
    in_qk = tl.where(is_q, xq, xk)
    out_qk = tl.where(is_q, out_q, cache_k)

    cache_v += (
        (head_idx - v_start) * stride_cachevH
        + cache_pos * stride_cachevM
        + group_idx * stride_cachevG
    )
    xv += (
        query_pos * stride_xvM
        + (head_idx - v_start) * stride_xvH
        + group_idx * stride_xvG
    )

    out = tl.where(is_v, cache_v, out_qk)
    x_in = tl.where(is_v, xv, in_qk)

    for offset in range(0, dim // 2, BLOCK_SIZE // 2):
        c = tl.arange(0, BLOCK_SIZE // 2)
        powers = (offset + c) * 2.0
        if adjacents:
            cols_re = (offset + c) * 2
            cols_im = cols_re + 1
        else:
            cols_re = offset + c
            cols_im = cols_re + dim // 2

        mask = cols_im < dim

        re_x = tl.load(x_in + cols_re, mask=mask)
        im_x = tl.load(x_in + cols_im, mask=mask)
        # freqs = seq_pos / (theta ** (powers / dim))
        freqs = pow(theta, powers / (-dim))

        if use_dynamic_scaling:
            lo_freq_wavelen = dynamic_old_context_len / dynamic_low_freq_factor
            hi_freq_wavelen = dynamic_old_context_len / dynamic_high_freq_factor

            wavelens = 6.28318530718 / freqs  # 2*pi
            is_low_freq = wavelens > lo_freq_wavelen
            freqs = tl.where(is_low_freq, freqs / dynamic_scale_factor, freqs)

            is_mid_freq = hi_freq_wavelen < wavelens and wavelens <= lo_freq_wavelen

            smooth = (dynamic_old_context_len / wavelens - dynamic_low_freq_factor) / (
                dynamic_high_freq_factor - dynamic_low_freq_factor
            )
            freqs = tl.where(
                is_mid_freq,
                (1 - smooth) * freqs / dynamic_scale_factor + smooth * freqs,
                freqs,
            )

        freqs = seq_pos * freqs / linear_scale
        sines = tl.sin(freqs)
        cosines = tl.cos(freqs)
        re_out = re_x * cosines - im_x * sines
        im_out = im_x * cosines + re_x * sines

        re_out_ = tl.where(is_v, re_x, re_out)
        im_out_ = tl.where(is_v, im_x, im_out)
        if internal_dtype == "f64":
            if re_x.dtype == tl.bfloat16:
                # triton 2.0.0 crashes if you try to convert
                # float64 directly to bfloat16, so make an intermediate step.
                re_out_ = re_out_.to(tl.float32)
                im_out_ = im_out_.to(tl.float32)
        tl.store(out + cols_re, re_out_, mask=mask)
        tl.store(out + cols_im, im_out_, mask=mask)
