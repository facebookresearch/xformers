from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from xformers.components.attention import ATTENTION_REGISTRY, Attention, AttentionConfig, register_attention
from transformers.models.reformer.modeling_reformer import LSHSelfAttention, ReformerConfig, ReverseSort


@dataclass
class LSHSelfAttentionConfig(AttentionConfig):
    num_hash: int
    num_heads: int
    dim_model: int
    seq_len: int
    num_buckets: int
    chunk_length: int

@register_attention("lsh_reformer", LSHSelfAttentionConfig)
class LSHAttention(Attention):
    """
    Using implementation from huggingface transformers
    LSH attention mechanism, from
    "
    Reformer: The Efficient Transformer
    Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya (2020)
    "
    ArXiv: https://arxiv.org/abs/2001.04451
    Reference repository: https://huggingface.co/transformers/model_doc/reformer.html
    """
    def __init__(
        self,
        num_heads: int,
        dim_model: int,
        num_hash: int = 4,
        seq_len: int = 4096,
        dropout: float = 0.0,
        num_buckets: int = None,
        chunk_length: int = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        attn_config = ReformerConfig()
        attn_config.num_attention_heads = num_heads
        attn_config.attn_layers = ["lsh"]
        attn_config.is_decoder = False
        attn_config.num_hashes = num_hash
        attn_config.max_position_embeddings = seq_len
        attn_config.attention_head_size = dim_model // num_heads
        # attn_config.feed_forward_size = 1
        if chunk_length:
            attn_config.lsh_attn_chunk_length = chunk_length
        attn_config.num_buckets = num_buckets
        self.attn = ReformerAttention(attn_config)

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        att_mask: Optional[torch.Tensor] = None, 
        key_padding_mask: Optional[Tensor] = None,
        *args, **kwargs
    ):
        return self.attn(q, k, v, att_mask, key_padding_mask)


class ReformerAttention(LSHSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        # self.query_key = None
        # self.value = None
        del self.query_key
        del self.value
        self.num_heads = self.num_attention_heads

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        # attention_mask=None,
        # head_mask=None,
        # num_hashes=None,
        # buckets=None,
        # past_buckets_states=None,
        # use_cache=False,
        # output_attentions=False,
        *args,
        **kwargs,
    ):
        orig_sequence_length, head_dim = q.shape[1], q.shape[2]

        # padding to factors of self.chunk_length

        # needs centain sequence length to make the block wise local attention work
        def _pad_to_window_size(x, window_size):
            seq_len = x.size(-2)
            pad_len = (window_size - seq_len % window_size) % window_size
            return F.pad(x, (0,0,0,pad_len), value=0), pad_len
        q, _ = _pad_to_window_size(q, self.chunk_length)
        v, _ = _pad_to_window_size(v, self.chunk_length)

        key_padding_mask = key_padding_mask.to(q)

        if key_padding_mask.shape[1] % self.chunk_length != 0:
            pad_len = (self.chunk_length - key_padding_mask.shape[1] % self.chunk_length) % self.chunk_length
            # key padding mask: 1 means padding tokens
            key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_ones(key_padding_mask.size(0), pad_len).to(key_padding_mask)], dim=1) 
        sequence_length = q.shape[1]

        batch_size = q.shape[0] // self.num_attention_heads

        # num hashes can optionally be overwritten by user
        num_hashes = self.num_hashes

        # @xwhan reformer needs the key and query vectors to be the same
        # project hidden_states to query_key and value
        query_key_vectors = q.view(batch_size, self.num_attention_heads, sequence_length, head_dim)
        value_vectors = v.view(batch_size, self.num_attention_heads, sequence_length, head_dim)

        do_standard_self_attention = (sequence_length <= self.chunk_length)

        # xformer format, att_mask: B x seq_len x seq_len #TODO: better solution?
        if key_padding_mask is not None:
            attention_mask = key_padding_mask != 1
        elif att_mask is not None:
            assert len(att_mask.shape) == 3
            att_mask = att_mask.reshape(batch_size, self.num_heads, -1, sequence_length)[:,0,:,:]
            attention_mask = att_mask.sum(1) > 0
        else:
            attention_mask = None

        # LSH attention only makes sense if chunked attention should be performed
        if not do_standard_self_attention:
            # set `num_buckets` on the fly, recommended way to do it
            if self.num_buckets is None:
                self._set_num_buckets(sequence_length.item())

            # hash query key vectors into buckets
            buckets = self._hash_vectors(query_key_vectors, num_hashes, attention_mask)

            assert (
                int(buckets.shape[-1]) == num_hashes * sequence_length
            ), f"last dim of buckets is {buckets.shape[-1]}, but should be {num_hashes * sequence_length}"

            sorted_bucket_idx, undo_sorted_bucket_idx = self._get_sorted_bucket_idx_and_undo_sorted_bucket_idx(
                sequence_length, buckets, num_hashes
            )

            # make sure bucket idx is not longer then sequence length
            sorted_bucket_idx_per_hash = sorted_bucket_idx % sequence_length

            # cluster query key value vectors according to hashed buckets
            query_key_vectors = self._gather_by_expansion(query_key_vectors, sorted_bucket_idx_per_hash, num_hashes)
            value_vectors = self._gather_by_expansion(value_vectors, sorted_bucket_idx_per_hash, num_hashes)
            query_key_vectors = self._split_seq_length_dim_to(
                query_key_vectors,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                self.attention_head_size,
            )
            value_vectors = self._split_seq_length_dim_to(
                value_vectors,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                self.attention_head_size,
            )

            if self.chunk_length is None:
                assert (
                    self.num_chunks_before == 0 and self.num_chunks_after == 0
                ), "If `config.chunk_length` is `None`, make sure `config.num_chunks_after` and `config.num_chunks_before` are set to 0."

        else:
            # get sequence length indices
            sorted_bucket_idx_per_hash = torch.arange(sequence_length, device=query_key_vectors.device).repeat(
                batch_size, self.num_attention_heads, 1
            )

        # scale key vectors
        key_vectors = self._len_and_dim_norm(query_key_vectors)

        # set query_vectors to query key vectors if LSH self attention
        query_vectors = query_key_vectors

        # free memory
        del query_key_vectors

        # get attention probs
        out_vectors, logits, attention_probs = self._attend(
            query_vectors=query_vectors,
            key_vectors=key_vectors,
            value_vectors=value_vectors,
            sorted_bucket_idx_per_hash=sorted_bucket_idx_per_hash,
            attention_mask=attention_mask,
            head_mask=None,
            do_standard_self_attention=do_standard_self_attention,
            do_cached_attention=False
        )

        # free memory
        del key_vectors, value_vectors

        # re-order out_vectors and logits
        if not do_standard_self_attention:
            # sort clusters back to correct ordering
            out_vectors, logits = ReverseSort.apply(out_vectors, logits, sorted_bucket_idx, undo_sorted_bucket_idx)

        if not do_standard_self_attention:
            # sum up all hash rounds
            if num_hashes > 1:
                out_vectors = self._split_seq_length_dim_to(
                    out_vectors,
                    num_hashes,
                    sequence_length,
                    self.num_attention_heads,
                    self.attention_head_size,
                )
                logits = self._split_seq_length_dim_to(
                    logits,
                    num_hashes,
                    sequence_length,
                    self.num_attention_heads,
                    self.attention_head_size,
                ).unsqueeze(-1)

                probs_vectors = torch.exp(logits - torch.logsumexp(logits, dim=2, keepdim=True))
                out_vectors = torch.sum(out_vectors * probs_vectors, dim=2)
                # free memory
                del probs_vectors

            # free memory
            del logits

        assert out_vectors.shape == (
            batch_size,
            self.num_attention_heads,
            sequence_length,
            self.attention_head_size,
        ), "out_vectors have be of shape `[batch_size, config.num_attention_heads, sequence_length, config.attention_head_size]`."

        # print(out_vectors.view(-1, sequence_length, head_dim).size())
        # import pdb;pdb.set_trace()
        # out_vectors = self._merge_hidden_size_dims(out_vectors, self.num_attention_heads, self.attention_head_size)
    
        return out_vectors.view(-1, sequence_length, head_dim)[:,:orig_sequence_length,:].contiguous()

        # return LSHSelfAttentionOutput(hidden_states=out_vectors, attention_probs=attention_probs, buckets=buckets)
