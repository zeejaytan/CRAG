from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import FP32LayerNorm
import torch
import torch.nn as nn

from source.modules.attention_processor import CragVarlenFlashAttentionProcessor


class DiTFusionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_attention_heads: int,
        use_cross_attention: bool = True,
        cross_attention_dim: int | None = None,
        norm_elementwise_affine: bool = True,
        skip: bool = False,
    ):
        super().__init__()
        assert dim % num_attention_heads == 0, "dim must be divisible by num_attention_heads"
        self.dim = dim
        self.num_heads = num_attention_heads
        self.dim_head = dim // num_attention_heads
        self.use_cross_attention = use_cross_attention

        # Skip Connection
        if skip:
            self.skip_norm = FP32LayerNorm(dim, elementwise_affine=True)
            self.skip_linear = nn.Linear(2 * dim, dim)
        else:
            self.skip_linear = None

        # Self-Attention
        self.self_attn_prenorm = FP32LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.self_attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=self.dim_head,
            heads=self.num_heads,
            qk_norm="rms_norm",
            eps=1e-6,
            bias=False,
            elementwise_affine=norm_elementwise_affine,
            processor=CragVarlenFlashAttentionProcessor(),
        )

        # Global-Attention
        self.global_attn_prenorm = FP32LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.global_attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=self.dim_head,
            heads=self.num_heads,
            qk_norm="rms_norm",
            eps=1e-6,
            bias=False,
            elementwise_affine=norm_elementwise_affine,
            processor=CragVarlenFlashAttentionProcessor(),
        )

        # Cross Attn
        self.cross_attn = None
        if use_cross_attention:
            self.cross_attn_prenorm = FP32LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            self.cross_attn = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                dim_head=self.dim_head,
                heads=self.num_heads,
                qk_norm="rms_norm",
                eps=1e-6,
                bias=False,
                elementwise_affine=norm_elementwise_affine,
                processor=CragVarlenFlashAttentionProcessor(),
            )

        # FFN
        self.ff_prenorm = FP32LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 4, dim),
        )

    def forward_self_attention(
        self,
        x: torch.Tensor,  # (seq_len, dim)
        cu_seqlens: torch.Tensor,  # (batch_size + 1,)
        max_seqlen: int,  # max sequence length in the batch
    ):
        h = self.self_attn_prenorm(x)
        return x + self.self_attn(
            hidden_states=h,
            cu_seqlens_q=cu_seqlens,
            max_seqlen_q=max_seqlen,
        )

    def forward_global_attention(
        self,
        x: torch.Tensor,  # (seq_len, dim)
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ):
        normed_states = self.global_attn_prenorm(x)
        return x + self.global_attn(
            hidden_states=normed_states,
            cu_seqlens_q=cu_seqlens,
            max_seqlen_q=max_seqlen,
        )

    def forward_cross_attention(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_k: torch.Tensor,
    ):
        normed_states = self.cross_attn_prenorm(x)
        return x + self.cross_attn(
            hidden_states=normed_states,
            encoder_hidden_states=cond,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
        )

    def forward_ffn(
        self,
        x: torch.Tensor,
    ):
        return x + self.ff(self.ff_prenorm(x))

    def forward(
        self,
        x: torch.Tensor,
        self_cu_seqlens: torch.Tensor,
        self_max_seqlen: int,
        global_cu_seqlens: torch.Tensor,
        global_max_seqlen: int,
        context: torch.Tensor | None = None,
        context_cu_seqlens: torch.Tensor | None = None,
        context_max_seqlen: int | None = None,
        skip_input: torch.Tensor | None = None,
    ):
        # 0. Skip Connection
        if self.skip_linear is not None:
            skip_input = self.skip_linear(torch.cat([x, skip_input], dim=-1))
            x = self.skip_norm(skip_input)

        # 1. Self-Attention
        x = self.forward_self_attention(
            x,
            cu_seqlens=self_cu_seqlens,
            max_seqlen=self_max_seqlen,
        )

        # 2. Global-Attention
        x = self.forward_global_attention(
            x,
            cu_seqlens=global_cu_seqlens,
            max_seqlen=global_max_seqlen,
        )

        # 3. Cross-Attention
        if self.cross_attn is not None and context is not None:
            assert context_cu_seqlens is not None
            assert context_max_seqlen is not None
            x = self.forward_cross_attention(
                x,
                cond=context,
                cu_seqlens_q=global_cu_seqlens,
                max_seqlen_q=global_max_seqlen,
                cu_seqlens_k=context_cu_seqlens,
                max_seqlen_k=context_max_seqlen,
            )

        # 4. FFN
        x = self.forward_ffn(
            x,
        )

        return x  # noqa: RET504
