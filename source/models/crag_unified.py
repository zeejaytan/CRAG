import itertools

from diffusers import ConfigMixin
from diffusers import ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.models.embeddings import Timesteps
from diffusers.models.normalization import FP32LayerNorm
import einops
from hydra.utils import instantiate
import lightning
import numpy as np
import pytorch3d.transforms as p3dt
import torch
import torch.nn as nn
from transformers import Dinov2Model
import trimesh

from source.modules.attention_processor import CragVarlenFlashAttentionProcessor
from source.modules.attention_processor import flash_attn  # import from our processor to make use of flash attn 3
from source.modules.autoencoders.triposg_varlen_autoencoder import TripoSGVarlenVAEModel
from source.modules.embeddings import AbsolutePositionalEmbedding
from source.modules.fusion_block import DiTFusionBlock
from source.modules.scheduler import SE3FlowMatchEulerContinuousScheduler
from source.utils.evaluation import calc_part_acc
from source.utils.evaluation import scatter_mean
from third_party.tripo_sg.triposg.inference_utils import flash_extract_geometry
from third_party.tripo_sg.triposg.models.embeddings import FrequencyPositionalEmbedding
from third_party.tripo_sg.triposg.models.transformers import TripoSGDiTModel
from third_party.tripo_sg.triposg.models.transformers.triposg_transformer import DiTBlock


class JointAdapter(nn.Module):
    def __init__(
        self,
        *,
        assembly_dim: int,
        generation_dim: int | None,
        width: int = 2048,
        num_heads: int = 16,
        use_cross_attention: bool = True,
        use_joint_attention: bool = False,
    ):
        super().__init__()
        self.assembly_dim = assembly_dim
        self.generation_dim = generation_dim
        self.width = width
        self.num_heads = num_heads
        self.use_cross_attention = use_cross_attention
        self.use_joint_attention = use_joint_attention

        assert not (use_cross_attention and use_joint_attention), "Cannot use both cross attention and joint attention."

        if use_cross_attention:
            assert generation_dim is not None, "Generation dimension must be specified if cross attention is enabled."
            self.gen_to_assembly_prenorm = FP32LayerNorm(generation_dim)
            self.gen_to_assembly_cross_attn = Attention(
                query_dim=generation_dim,
                cross_attention_dim=assembly_dim,
                dim_head=width // num_heads,
                heads=num_heads,
                qk_norm="rms_norm",
                eps=1e-6,
                bias=False,
                processor=CragVarlenFlashAttentionProcessor(),
            )

            self.assembly_to_generation_prenorm = FP32LayerNorm(assembly_dim)
            self.assembly_to_generation_cross_attn = Attention(
                query_dim=assembly_dim,
                cross_attention_dim=generation_dim,
                dim_head=width // num_heads,
                heads=num_heads,
                qk_norm="rms_norm",
                eps=1e-6,
                bias=False,
                processor=CragVarlenFlashAttentionProcessor(),
            )

            nn.init.zeros_(self.gen_to_assembly_cross_attn.to_out[0].weight)
            nn.init.zeros_(self.assembly_to_generation_cross_attn.to_out[0].weight)
            nn.init.zeros_(self.gen_to_assembly_cross_attn.to_out[0].bias)
            nn.init.zeros_(self.assembly_to_generation_cross_attn.to_out[0].bias)

        if use_joint_attention:
            assert generation_dim is not None, "Generation dimension must be specified if joint attention is enabled."
            self.gen_joint_prenorm = FP32LayerNorm(generation_dim)
            self.assembly_joint_prenorm = FP32LayerNorm(assembly_dim)

            # qkv projection
            self.gen_joint_to_qkv = nn.Linear(generation_dim, 3 * width, bias=False)
            self.assembly_joint_to_qkv = nn.Linear(assembly_dim, 3 * width, bias=False)
            # qk norm
            self.gen_q_norm = nn.RMSNorm(width // num_heads, eps=1e-6)
            self.gen_k_norm = nn.RMSNorm(width // num_heads, eps=1e-6)
            self.assembly_q_norm = nn.RMSNorm(width // num_heads, eps=1e-6)
            self.assembly_k_norm = nn.RMSNorm(width // num_heads, eps=1e-6)
            # out projection
            self.gen_joint_to_out = nn.Linear(width, generation_dim, bias=False)
            self.assembly_joint_to_out = nn.Linear(width, assembly_dim, bias=False)

            nn.init.zeros_(self.gen_joint_to_out.weight)
            nn.init.zeros_(self.assembly_joint_to_out.weight)

    def forward(
        self,
        assembly_block: DiTFusionBlock,
        generation_block: DiTBlock | None,
        assembly_inputs: dict,
        generation_inputs: dict,
        joint_attn_kwargs: dict,
    ):
        assembly_hidden_states = assembly_inputs["x"]
        z_hidden_states = generation_inputs["hidden_states"]

        # 0. Assembly Skip Connection
        if assembly_block.skip_linear is not None:
            skip = torch.cat(
                [assembly_hidden_states, assembly_inputs["skip"]],
                dim=-1,
            )
            skip = assembly_block.skip_linear(skip)
            assembly_hidden_states = assembly_block.skip_norm(skip)

        # 1. Assembly Self-Attention
        assembly_hidden_states = assembly_block.forward_self_attention(
            assembly_hidden_states,
            cu_seqlens=assembly_inputs["self_cu_seqlens"],
            max_seqlen=assembly_inputs["self_max_seqlen"],
        )

        # 2. Assembly Global-Attention
        assembly_hidden_states = assembly_block.forward_global_attention(
            assembly_hidden_states,
            cu_seqlens=assembly_inputs["global_cu_seqlens"],
            max_seqlen=assembly_inputs["global_max_seqlen"],
        )

        # 2.1. Assembly Cross-Attention (Optional)
        if assembly_block.cross_attn is not None:
            cond = generation_inputs["encoder_hidden_states"]
            assembly_hidden_states = assembly_block.forward_cross_attention(
                assembly_hidden_states,
                cond=cond.flatten(0, 1),
                cu_seqlens_q=assembly_inputs["global_cu_seqlens"],
                max_seqlen_q=assembly_inputs["global_max_seqlen"],
                cu_seqlens_k=torch.arange(
                    0,
                    cond.shape[0] + 1,
                    device=assembly_hidden_states.device,
                ).int()
                * cond.shape[1],
                max_seqlen_k=cond.shape[1],
            )

        if generation_block is not None:
            # 3. Generation Skip Connection
            if generation_block.skip_linear is not None:
                cat = torch.cat(
                    (
                        [generation_inputs["skip"], generation_inputs["hidden_states"]]
                        if generation_block.skip_concat_front
                        else [generation_inputs["hidden_states"], generation_inputs["skip"]]
                    ),
                    dim=-1,
                )
                if generation_block.skip_norm_last:
                    z_hidden_states = generation_block.skip_linear(cat)
                    z_hidden_states = generation_block.skip_norm(z_hidden_states)
                else:
                    z_hidden_states = generation_block.skip_norm(cat)
                    z_hidden_states = generation_block.skip_linear(z_hidden_states)

            # 4. Generation Self-Attention
            if generation_block.use_self_attention:
                norm_z_hidden_states = generation_block.norm1(z_hidden_states)
                attn_output = generation_block.attn1(
                    norm_z_hidden_states,
                )
                z_hidden_states = z_hidden_states + attn_output

            # 5. Generation Cross-Attention
            if generation_block.use_cross_attention:
                z_hidden_states = z_hidden_states + generation_block.attn2(
                    generation_block.norm2(z_hidden_states),
                    encoder_hidden_states=generation_inputs["encoder_hidden_states"],
                )

            # 6. Bi-Directional Cross-Attention
            if self.use_cross_attention:
                # 6.1. Pre-norm on both sides
                norm_z_hidden_states = self.gen_to_assembly_prenorm(z_hidden_states)
                norm_assembly_hidden_states = self.assembly_to_generation_prenorm(assembly_hidden_states)

                # 6.2. Generation to Assembly
                gen_to_assembly_attn_output = self.gen_to_assembly_cross_attn(
                    norm_z_hidden_states.flatten(0, 1),
                    encoder_hidden_states=norm_assembly_hidden_states,
                    cu_seqlens_q=torch.arange(
                        0, norm_z_hidden_states.shape[0] + 1, device=norm_z_hidden_states.device
                    ).int()
                    * norm_z_hidden_states.shape[1],
                    max_seqlen_q=norm_z_hidden_states.shape[1],
                    cu_seqlens_k=assembly_inputs["global_cu_seqlens"],
                    max_seqlen_k=assembly_inputs["global_max_seqlen"],
                ).view_as(z_hidden_states)

                # 6.3. Assembly to Generation
                assembly_to_generation_attn_output = self.assembly_to_generation_cross_attn(
                    norm_assembly_hidden_states,
                    encoder_hidden_states=norm_z_hidden_states.flatten(0, 1),
                    cu_seqlens_q=assembly_inputs["global_cu_seqlens"],
                    max_seqlen_q=assembly_inputs["global_max_seqlen"],
                    cu_seqlens_k=torch.arange(0, z_hidden_states.shape[0] + 1, device=z_hidden_states.device).int()
                    * z_hidden_states.shape[1],
                    max_seqlen_k=z_hidden_states.shape[1],
                )

                # 6.4. Residual Connection
                assembly_hidden_states = assembly_hidden_states + assembly_to_generation_attn_output
                z_hidden_states = z_hidden_states + gen_to_assembly_attn_output

            # 7. Joint Attention
            if self.use_joint_attention:
                # 7.1. Pre-norm
                norm_z_hidden_states = self.gen_joint_prenorm(z_hidden_states)
                norm_assembly_hidden_states = self.assembly_joint_prenorm(assembly_hidden_states)

                # 7.2. QKV projection
                gen_qkv = self.gen_joint_to_qkv(norm_z_hidden_states)
                assembly_qkv = self.assembly_joint_to_qkv(norm_assembly_hidden_states)
                gen_qkv = einops.rearrange(
                    gen_qkv,
                    "b s (three h d) -> b s three h d",
                    three=3,
                    h=self.num_heads,
                )
                assembly_qkv = einops.rearrange(
                    assembly_qkv,
                    "s (three h d) -> s three h d",
                    three=3,
                    h=self.num_heads,
                )

                # 7.3. QK Norm
                gen_qkv[:, :, 0] = self.gen_q_norm(gen_qkv[:, :, 0])
                gen_qkv[:, :, 1] = self.gen_k_norm(gen_qkv[:, :, 1])
                assembly_qkv[:, :, 0] = self.assembly_q_norm(assembly_qkv[:, :, 0])
                assembly_qkv[:, :, 1] = self.assembly_k_norm(assembly_qkv[:, :, 1])

                # 7.4. Concatenate to form joint QKV
                mask = joint_attn_kwargs.get("mask")
                joint_qkv = torch.empty(
                    gen_qkv.shape[0] * gen_qkv.shape[1] + assembly_qkv.shape[0],
                    *gen_qkv.shape[2:],
                    dtype=gen_qkv.dtype,
                    device=gen_qkv.device,
                )
                joint_qkv[mask] = assembly_qkv
                joint_qkv[~mask] = einops.rearrange(
                    gen_qkv,
                    "b s three h d -> (b s) three h d",
                )
                attn_output = flash_attn.flash_attn_varlen_func(
                    q=joint_qkv[:, 0],
                    k=joint_qkv[:, 1],
                    v=joint_qkv[:, 2],
                    cu_seqlens_q=joint_attn_kwargs["cu_seqlens"],
                    max_seqlen_q=joint_attn_kwargs["max_seqlen"],
                    cu_seqlens_k=joint_attn_kwargs["cu_seqlens"],
                    max_seqlen_k=joint_attn_kwargs["max_seqlen"],
                )
                attn_output = attn_output.reshape(-1, self.width)
                attn_output = attn_output.to(joint_qkv.dtype)

                # 7.5. Out projection
                assembly_attn_out = self.assembly_joint_to_out(attn_output[mask]).view_as(assembly_hidden_states)
                gen_attn_out = self.gen_joint_to_out(attn_output[~mask]).view_as(z_hidden_states)

                # 7.6. Residual Connection
                assembly_hidden_states = assembly_hidden_states + assembly_attn_out
                z_hidden_states = z_hidden_states + gen_attn_out

        assembly_hidden_states = assembly_block.forward_ffn(
            assembly_hidden_states,
        )

        if generation_block is not None:
            # 8. Generation FFN
            z_hidden_states = z_hidden_states + generation_block.ff(generation_block.norm3(z_hidden_states))

        return assembly_hidden_states, z_hidden_states


class CragTransformer(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        *,
        in_dim: int = 1024,
        width: int = 512,
        num_layers: int = 21,
        num_heads: int = 8,
        enable_scale_emb: bool = True,
        enable_skip: bool = True,
        enable_cross_attention: bool = False,
        cross_attention_dim: int | None = None,
    ):
        super().__init__()

        if enable_cross_attention:
            assert cross_attention_dim is not None, (
                "Cross attention dimension must be specified if cross attention is enabled."
            )

        self.time_embed = Timesteps(
            width,
            flip_sin_to_cos=False,
            downscale_freq_shift=0,
        )
        self.time_proj = TimestepEmbedding(
            in_channels=width,
            time_embed_dim=width * 4,
            act_fn="gelu",
            out_dim=width,
        )

        self.anchor_embedding = nn.Embedding(2, width)
        self.parts_embedding = AbsolutePositionalEmbedding(max_length=100, embedding_dim=width)
        self.coord_embeddings = FrequencyPositionalEmbedding(
            num_freqs=8,
            input_dim=6 + (1 if enable_scale_emb else 0),
            logspace=True,
            include_input=True,
            include_pi=False,
        )

        self.transform_proj_in = nn.Linear(7, width)
        self.proj_in = nn.Linear(
            self.coord_embeddings.out_dim + in_dim,
            width,
        )

        self.blocks = nn.ModuleList(
            [
                DiTFusionBlock(
                    dim=width,
                    num_attention_heads=num_heads,
                    use_cross_attention=enable_cross_attention,
                    cross_attention_dim=cross_attention_dim,
                    norm_elementwise_affine=True,
                    skip=layer > num_layers // 2 and enable_skip,
                )
                for layer in range(num_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(width, elementwise_affine=True)
        self.proj_out = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width // 2),
            nn.SiLU(),
            nn.Linear(width // 2, 6, bias=False),
        )

    @torch.inference_mode()
    def apply_transform(
        self, coord: torch.Tensor, normal: torch.Tensor, points_per_part: torch.Tensor, transform: torch.Tensor
    ):
        # After scaling the part pointclouds, do not apply translation.
        with torch.autocast(enabled=False, device_type=transform.device.type):
            broadcasted_transform = transform.repeat_interleave(points_per_part, dim=0)
            coord = p3dt.quaternion_apply(
                broadcasted_transform[:, 3:],
                coord,
            )
            if not self.config.get("enable_scale_emb", True):
                coord = coord + broadcasted_transform[:, :3]
            normal = p3dt.quaternion_apply(
                broadcasted_transform[:, 3:],
                normal,
            )
            return coord, normal

    # @torch.compile()
    def prepare_inputs(
        self,
        parts_condition: dict[str, torch.Tensor],
        transform_t: torch.Tensor,
        t: torch.Tensor,
    ):
        # input embedding and projection
        coord, normal = self.apply_transform(
            parts_condition["coord"],
            parts_condition["normal"],
            parts_condition["points_per_part"],
            transform_t,
        )

        coord_normal = torch.cat([coord, normal], dim=-1)
        if self.config.get("enable_scale_emb", True):
            coord_normal = torch.cat(
                [
                    coord_normal,
                    parts_condition["scales"]
                    .repeat_interleave(parts_condition["points_per_part"], dim=0)
                    .unsqueeze(-1),
                ],
                dim=-1,
            )
        coord_emb = self.coord_embeddings(coord_normal)
        hidden_states = self.proj_in(torch.cat([parts_condition["feat"], coord_emb], dim=-1))

        # part embedding
        parts_embedding = self.parts_embedding(
            torch.cat(
                [torch.arange(n, device=parts_condition["num_parts"].device) for n in parts_condition["num_parts"]]
            )
        ).repeat_interleave(parts_condition["points_per_part"], dim=0)
        hidden_states = hidden_states + parts_embedding

        # transform embedding
        transform_emb = self.transform_proj_in(transform_t)
        hidden_states = hidden_states + transform_emb.repeat_interleave(parts_condition["points_per_part"], dim=0)

        # anchor embedding
        anchor = parts_condition["ref_part"].int()
        anchor_tokens = self.anchor_embedding(anchor)

        # time embedding
        time_emb = self.time_proj(self.time_embed(t))
        time_tokens = time_emb.repeat_interleave(parts_condition["num_parts"], dim=0)

        # rearrange tokens
        PREPEND_TOKEN_NUM = 2
        TOTAL_PARTS = parts_condition["num_parts"].sum()
        TOTAL_SEQ_LEN = hidden_states.shape[0] + PREPEND_TOKEN_NUM * TOTAL_PARTS

        extended_batch = torch.arange(TOTAL_PARTS, device=hidden_states.device).repeat_interleave(
            parts_condition["points_per_part"] + PREPEND_TOKEN_NUM
        )
        extended_indices = torch.arange(TOTAL_SEQ_LEN, device=hidden_states.device)
        extended_indices = extended_indices - PREPEND_TOKEN_NUM * (extended_batch + 1)
        extended_hidden_states = hidden_states[extended_indices]

        # calculate attention sequence lengths
        part_indices = torch.repeat_interleave(
            torch.arange(len(parts_condition["num_parts"]), device=parts_condition["num_parts"].device),
            parts_condition["num_parts"],
        )
        self_attn_seqlens = parts_condition["points_per_part"] + PREPEND_TOKEN_NUM
        self_attn_max_seqlen = self_attn_seqlens.max()
        self_attn_cu_seqlens = nn.functional.pad(self_attn_seqlens.cumsum(0), (1, 0), value=0).int()
        global_attn_seqlens = torch.zeros(
            len(parts_condition["num_parts"]), dtype=self_attn_seqlens.dtype, device=self_attn_seqlens.device
        )
        global_attn_seqlens.scatter_add_(0, part_indices, self_attn_seqlens)
        global_attn_max_seqlen = global_attn_seqlens.max()
        global_attn_cu_seqlens = nn.functional.pad(global_attn_seqlens.cumsum(0), (1, 0), value=0).int()

        extended_hidden_states[self_attn_cu_seqlens[:-1]] = anchor_tokens.to(extended_hidden_states.dtype)
        extended_hidden_states[self_attn_cu_seqlens[:-1] + 1] = time_tokens.to(extended_hidden_states.dtype)

        return extended_hidden_states, {
            "self_seqlens": self_attn_seqlens,
            "self_cu_seqlens": self_attn_cu_seqlens,
            "self_max_seqlen": self_attn_max_seqlen,
            "global_seqlens": global_attn_seqlens,
            "global_cu_seqlens": global_attn_cu_seqlens,
            "global_max_seqlen": global_attn_max_seqlen,
        }


class CragUnifiedTransformer(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        *,
        assembly_transformer_kwargs: dict | None = None,
        generation_pretrained_model_name_or_path: str = "VAST-AI/TripoSG",
        generation_transformer_kwargs: dict | None = None,
        joint_adapter_kwargs: dict | None = None,
        freeze_generation_transformer: bool = True,
        enable_generation: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.assembly_transformer = CragTransformer(**(assembly_transformer_kwargs or {}))
        if self.config.get("enable_generation", True):
            self.generation_transformer = TripoSGDiTModel.from_pretrained(
                generation_pretrained_model_name_or_path,
                **(generation_transformer_kwargs or {}),
            )
            # freeze generation transformer
            if freeze_generation_transformer:
                self.generation_transformer.eval()
                for param in self.generation_transformer.parameters():
                    param.requires_grad = False

        self.num_layers = len(self.assembly_transformer.blocks)

        # Joint Adapters
        self.joint_adapters = nn.ModuleList(
            [
                JointAdapter(
                    **(joint_adapter_kwargs or {}),
                    assembly_dim=self.assembly_transformer.config.get("width"),
                    generation_dim=self.generation_transformer.config.get("width")
                    if self.config.get("enable_generation", True)
                    else None,
                )
                for _ in range(self.num_layers)
            ]
        )

        # # cast weights to float16 for faster training
        # if self.config.get("enable_generation", True):
        #     self.generation_transformer.to(torch.bfloat16)

        # self.joint_adapters.to(torch.float16)
        # self.assembly_transformer.to(torch.float16)

    def forward(
        self,
        *,
        # latents
        transform_t: torch.Tensor,
        z_t: torch.Tensor | None,
        # timesteps
        t: torch.Tensor,
        # conditions
        image_condition: torch.Tensor,
        parts_condition: dict[str, torch.Tensor],
    ):
        # prepare assembly inputs
        assembly_hidden_states, seq_info = self.assembly_transformer.prepare_inputs(
            parts_condition=parts_condition,
            transform_t=transform_t,
            t=t,
        )

        # prepare generation inputs
        z_hidden_states = None
        if self.config.get("enable_generation", True):
            z_t_emb = self.generation_transformer.time_proj(self.generation_transformer.time_embed(t))
            z_t_emb = z_t_emb.unsqueeze(1)
            z_hidden_states = self.generation_transformer.proj_in(z_t)
            z_hidden_states = torch.cat([z_t_emb, z_hidden_states], dim=1)

        joint_attn_kwargs = {}
        if self.config.get("joint_adapter_kwargs", {}).get("use_joint_attention", False):
            # prepare masks & sequence lengths for joint attention
            joint_cu_seqlens = (
                seq_info["global_cu_seqlens"]
                + torch.arange(
                    0,
                    z_hidden_states.shape[0] + 1,
                    device=z_hidden_states.device,
                ).int()
                * z_hidden_states.shape[1]
            )
            joint_max_seqlen = seq_info["global_max_seqlen"] + z_hidden_states.shape[1]
            # mask for generation tokens
            mask = torch.ones(joint_cu_seqlens[-1], dtype=torch.bool, device=self.device)
            indices = torch.arange(z_hidden_states.shape[0] * z_hidden_states.shape[1], device=self.device)
            indices += seq_info["global_cu_seqlens"][1:].repeat_interleave(z_hidden_states.shape[1])
            mask[indices] = False
            joint_attn_kwargs = {
                "cu_seqlens": joint_cu_seqlens,
                "max_seqlen": joint_max_seqlen,
                "mask": mask,
            }

        skips = []
        assembly_skips = []
        for i in range(self.num_layers):
            skip = None if i <= self.num_layers // 2 else skips.pop()
            assembly_skip = None if i <= self.num_layers // 2 else assembly_skips.pop()

            assembly_hidden_states, z_hidden_states = self.joint_adapters[i](
                self.assembly_transformer.blocks[i],
                self.generation_transformer.blocks[i] if self.config.get("enable_generation", True) else None,
                {
                    "x": assembly_hidden_states,
                    "self_cu_seqlens": seq_info["self_cu_seqlens"],
                    "self_max_seqlen": seq_info["self_max_seqlen"],
                    "global_cu_seqlens": seq_info["global_cu_seqlens"],
                    "global_max_seqlen": seq_info["global_max_seqlen"],
                    "skip": assembly_skip,
                },
                {
                    "hidden_states": z_hidden_states,
                    "encoder_hidden_states": image_condition,
                    "skip": skip,
                },
                joint_attn_kwargs,
            )

            if i < self.num_layers // 2:
                skips.append(z_hidden_states)
                assembly_skips.append(assembly_hidden_states)

        assembly_hidden_states = assembly_hidden_states[seq_info["self_cu_seqlens"][:-1]]
        # Assembly output
        assembly_hidden_states = self.assembly_transformer.norm_out(assembly_hidden_states)
        pred_v_transform = self.assembly_transformer.proj_out(assembly_hidden_states)

        # Generation output
        pred_v_z = None
        if self.config.get("enable_generation", True):
            z_hidden_states = self.generation_transformer.norm_out(z_hidden_states)
            z_hidden_states = z_hidden_states[:, 1:, :]
            pred_v_z = self.generation_transformer.proj_out(z_hidden_states)

        return pred_v_transform, pred_v_z


class CragUnifiedModel(lightning.LightningModule, ConfigMixin):
    config_name = "CragUnifiedModel"

    @register_to_config
    def __init__(
        self,
        *,
        noise_scheduler: dict,
        pretrained_model_name_or_path: str = "VAST-AI/TripoSG",
        transformer_kwargs: dict | None = None,
        enable_generation: bool = True,
        optimizer: dict | None = None,
        lr_scheduler: dict | None = None,
        inference_kwargs: dict | None = None,
        image_drop_rate: float = 0.1,
        lr_grouping: dict | None = None,
        use_decoded_features: bool = False,
        use_post_kl: bool = False,
        part_scale_ratio: float = 1.0,
    ):
        super().__init__()
        self.noise_scheduler: SE3FlowMatchEulerContinuousScheduler = instantiate(noise_scheduler)
        # Load pretrained models
        self.image_encoder = Dinov2Model.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="image_encoder_dinov2",
        )
        self.shape_vae = TripoSGVarlenVAEModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
        )

        # Freeze pretrained models
        self.image_encoder.eval()
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        self.shape_vae.eval()
        for param in self.shape_vae.parameters():
            param.requires_grad = False

        # Initialize our own modules
        self.transformer = CragUnifiedTransformer(
            **(transformer_kwargs or {}),
            enable_generation=enable_generation,
        )
        self.inference_kwargs = {
            "num_inference_steps": 50,
            "shift_t": 1.0,
            **(inference_kwargs or {}),
        }

    @torch.inference_mode()
    def encode_image(
        self,
        images: torch.Tensor,  # (B, V, 3, H, W)
        *,
        all_views: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, V = images.shape[:2]
        if not all_views:
            # Along V, random select one view
            view_indices = torch.randint(0, V, (batch_size,), device=images.device)
            images = images[torch.arange(batch_size, device=images.device), view_indices]
            # TODO: should the masking be curriculum based?
            mask = torch.rand(len(images), device=images.device) < self.config.get("image_drop_rate", 0.1)
            images[mask] = 0.0
            view_indices[mask] = -1  # Mark dropped views with -1

        if all_views:
            neg_cond = torch.zeros((batch_size, 1, 3, 224, 224), device=images.device)
            images = torch.cat([images, neg_cond], dim=1)  # (batch_size, V+1, 3, 224, 224)
            images = images.permute(1, 0, 2, 3, 4)  # (V+1, batch_size, 3, 224, 224)
            images = images.reshape(-1, 3, 224, 224)  # (batch_size*(V+1), 3, 224, 224)
            view_indices = torch.arange(V + 1, device=images.device)
            view_indices[-1] = -1  # Mark the extra view with -1
            view_indices = view_indices.repeat(batch_size, 1).reshape(-1)  # (batch_size*(V+1),)

        outputs = self.image_encoder(pixel_values=images)
        return outputs.last_hidden_state, view_indices.reshape(batch_size, -1)

    @torch.inference_mode()
    def encode_parts(
        self,
        coord: torch.Tensor,
        normal: torch.Tensor,
        num_parts: torch.Tensor,
        ref_part: torch.Tensor,
        points_per_part: torch.Tensor,
        scales: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        input_feat = torch.cat([coord * self.config.get("part_scale_ratio", 1.0), normal], dim=-1)
        feat, post_kl, mapped_indices, sampled_points_per_part = self.shape_vae.encode_parts(
            input_feat,
            num_tokens=points_per_part,
            ratio=0.25,
            use_decoded_features=self.config.get("use_decoded_features", False),
        )

        if self.config.get("use_post_kl", False):
            feat = post_kl

        return {
            "feat": feat,
            "post_kl": post_kl,
            "coord": coord[mapped_indices],
            "normal": normal[mapped_indices],
            "num_parts": num_parts,
            "points_per_part": sampled_points_per_part,
            "ref_part": ref_part,
            "scales": scales,
        }

    @torch.inference_mode()
    def encode_shape(
        self,
        coord: torch.Tensor,
        normal: torch.Tensor,
        batch_size: int,
        num_tokens: int = 512,
    ) -> torch.Tensor:
        input_feat = torch.cat([coord, normal], dim=-1)
        _, posterior = self.shape_vae.encode_shape(
            input_feat,
            batch_size=batch_size,
            num_tokens=num_tokens,
        )
        sampled_posterior = posterior.sample()

        return sampled_posterior.reshape(batch_size, -1, sampled_posterior.shape[-1])

    @torch.inference_mode()
    def decode_shape(
        self,
        latents: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        meshes = []
        for latent in latents:
            try:
                mesh_v_f = flash_extract_geometry(
                    latents=latent.unsqueeze(0),
                    vae=self.shape_vae,
                    bounds=(-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
                    octree_depth=9,
                )[0]
                mesh = trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1])
                meshes.append(mesh)
            except Exception as e:
                print(f"Failed to decode latent: {e}")
                meshes.append(trimesh.Trimesh())
        return meshes

    @torch.inference_mode()
    def sample(
        self,
        batch: dict[str, torch.Tensor],
        *,
        decode_shape: bool = True,
        all_views: bool = False,
    ):
        batch_size = len(batch["num_parts"])

        # 1. Encode image and parts to get condition
        image_condition, image_indices = self.encode_image(batch["renderings"], all_views=all_views)
        parts_condition = self.encode_parts(
            coord=batch["pointclouds"],
            normal=batch["pointclouds_normals"],
            num_parts=batch["num_parts"],
            ref_part=batch["ref_part"],
            points_per_part=batch["points_per_part"],
            scales=batch["scales"],
        )
        views = image_indices.shape[-1]

        # 2. Prepare the latents
        t = self.noise_scheduler.sample_t(batch_size=batch_size).to(self.device)

        transform_0 = torch.cat([batch["translations"], batch["quaternions"]], dim=-1)
        _, transform_t = self.noise_scheduler.scale_noise_se3(
            x_0=transform_0, t=t.repeat_interleave(batch["num_parts"])
        )
        transform_t[batch["ref_part"]] = transform_0[batch["ref_part"]]

        z_t = None
        if self.config.get("enable_generation", True):
            z_0 = self.encode_shape(
                coord=batch["full_pointcloud"] * 1.9,
                normal=batch["full_pointcloud_normal"],
                batch_size=batch_size,
                num_tokens=2048,
            )
            _, z_t = self.noise_scheduler.scale_noise(x_0=z_0, t=t)

        # 3. Repeat for views to align with image_condition
        if z_t is not None:
            z_t = z_t.repeat(views, 1, 1)
        transform_0 = transform_0.repeat(views, 1)
        transform_t = transform_t.repeat(views, 1)
        parts_condition["points_per_part"] = parts_condition["points_per_part"].repeat(views)
        parts_condition["feat"] = parts_condition["feat"].repeat(views, 1)
        parts_condition["coord"] = parts_condition["coord"].repeat(views, 1)
        parts_condition["normal"] = parts_condition["normal"].repeat(views, 1)
        parts_condition["num_parts"] = parts_condition["num_parts"].repeat(views)
        parts_condition["ref_part"] = parts_condition["ref_part"].repeat(views)
        parts_condition["scales"] = parts_condition["scales"].repeat(views)
        ref_part = parts_condition["ref_part"]
        num_parts = parts_condition["num_parts"]

        # 4. Inference loop
        t_seq = torch.linspace(1, 0, self.inference_kwargs["num_inference_steps"] + 1, device=self.device)
        t_seq = self.inference_kwargs["shift_t"] * t_seq / (1 + (self.inference_kwargs["shift_t"] - 1) * t_seq)
        all_pred_transforms = [transform_t.detach()]
        for t, t_prev in itertools.pairwise(t_seq):
            v_transform_pred, v_z_pred = self.transformer(
                transform_t=transform_t,
                z_t=z_t,
                t=torch.full((batch_size * views,), t * 1000, device=self.device),
                image_condition=image_condition,
                parts_condition=parts_condition,
            )

            transform_t = self.noise_scheduler.step_se3(
                v_pred=v_transform_pred,
                t=t,
                t_prev=t_prev,
                x_t=transform_t,
            )
            if z_t is not None:
                z_t = self.noise_scheduler.step(
                    v_pred=-v_z_pred,  # triposg is trained to predict -v
                    t=t,
                    t_prev=t_prev,
                    x_t=z_t,
                )
            transform_t[ref_part] = transform_0[ref_part]
            all_pred_transforms.append(transform_t.detach())

        # 5. Compute metrics
        pred_trans = transform_t[..., :3].detach()
        pred_quat = transform_t[..., 3:].detach()

        trans_rmse = scatter_mean(
            (pred_trans - batch["translations"].repeat(views, 1)).pow(2).mean(dim=-1) ** 0.5,
            num_parts,
        )
        gt_degrees = torch.rad2deg(
            p3dt.matrix_to_euler_angles(
                matrix=p3dt.quaternion_to_matrix(batch["quaternions"].repeat(views, 1)),
                convention="XYZ",
            )
        )
        pred_degrees = torch.rad2deg(
            p3dt.matrix_to_euler_angles(
                matrix=p3dt.quaternion_to_matrix(pred_quat),
                convention="XYZ",
            )
        )
        rot_rmse = scatter_mean(
            (
                torch.min(
                    torch.abs(gt_degrees - pred_degrees),
                    360 - torch.abs(gt_degrees - pred_degrees),
                )
                .pow(2)
                .mean(dim=-1)
                ** 0.5
            ),
            num_parts,
        )

        part_acc, shape_chamfer = calc_part_acc(
            pcd=parts_condition["coord"],
            num_parts=num_parts,
            points_per_part=parts_condition["points_per_part"],
            gt_transform=transform_0,
            pred_transform=transform_t,
            scales=parts_condition["scales"],
        )

        # 6. Save results
        meshes = []
        if self.config.get("enable_generation", True) and decode_shape:
            self.shape_vae.set_flash_decoder()
            self.shape_vae.set_tripo_decoder()
            meshes = self.decode_shape(z_t)
            self.shape_vae.set_varlen_decoder()

        return {
            "pred_transform_steps": all_pred_transforms,
            "pred_transform": transform_t,
            "gt_transform": transform_0,
            "rmse_r": rot_rmse,
            "rmse_t": trans_rmse,
            "part_acc": part_acc,
            "shape_chamfer": shape_chamfer,
            "meshes": meshes,
            "image_indices": image_indices,
        }

    def forward_step(self, batch, batch_idx):
        batch_size = len(batch["num_parts"])

        # 1. Encode image and parts to get condition
        image_condition, _ = self.encode_image(batch["renderings"])
        parts_condition = self.encode_parts(
            coord=batch["pointclouds"],
            normal=batch["pointclouds_normals"],
            num_parts=batch["num_parts"],
            ref_part=batch["ref_part"],
            points_per_part=batch["points_per_part"],
            scales=batch["scales"],
        )

        # 2. Prepare the latents at t=0
        if self.config.get("enable_generation", True):
            z_0 = self.encode_shape(
                coord=batch["full_pointcloud"] * 1.9,
                normal=batch["full_pointcloud_normal"],
                batch_size=batch_size,
                num_tokens=2048,
            )

        transform_0 = torch.cat([batch["translations"], batch["quaternions"]], dim=-1)

        # 3. Sample timesteps, add noise and get velocities
        t = self.noise_scheduler.sample_t(batch_size=batch_size).to(self.device)
        transform_t, transform_noise = self.noise_scheduler.scale_noise_se3(
            x_0=transform_0, t=t.repeat_interleave(batch["num_parts"])
        )
        v_transform_gt = self.noise_scheduler.get_v_se3(
            x_0=transform_0, noise=transform_noise, t=t.repeat_interleave(batch["num_parts"])
        )
        transform_t[batch["ref_part"]] = transform_0[batch["ref_part"]]
        v_transform_gt[batch["ref_part"]] = 0.0

        z_t, z_noise, v_z_gt = None, None, None
        if self.config.get("enable_generation", True):
            z_t, z_noise = self.noise_scheduler.scale_noise(x_0=z_0, t=t)
            v_z_gt = self.noise_scheduler.get_v(x_0=z_0, noise=z_noise, t=t)

        # 4. Predict the velocities
        v_transform_pred, v_z_pred = self.transformer(
            transform_t=transform_t,
            z_t=z_t,
            t=t * 1000,
            image_condition=image_condition,
            parts_condition=parts_condition,
        )

        return {
            "v_transform_pred": v_transform_pred,
            "v_transform_gt": v_transform_gt,
            "v_z_pred": -v_z_pred
            if self.config.get("enable_generation", True)
            else None,  # triposg is trained to predict -v
            "v_z_gt": v_z_gt if self.config.get("enable_generation", True) else None,
        }

    def loss(
        self,
        input_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        mode: str = "train",
    ) -> torch.Tensor:
        v_transform_mse_loss = nn.functional.mse_loss(
            output_batch["v_transform_pred"],
            output_batch["v_transform_gt"],
            reduction="mean",
        )
        v_z_mse_loss = (
            nn.functional.mse_loss(
                output_batch["v_z_pred"],
                output_batch["v_z_gt"],
                reduction="mean",
            )
            if output_batch["v_z_pred"] is not None and output_batch["v_z_gt"] is not None
            else 0.0
        )

        loss = v_transform_mse_loss + v_z_mse_loss

        self.log_dict(
            {
                f"{mode}/loss": loss,
                f"{mode}/v_transform_mse_loss": v_transform_mse_loss,
                f"{mode}/v_z_mse_loss": v_z_mse_loss,
            },
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
            batch_size=len(input_batch["num_parts"]),
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self.loss(input_batch=batch, output_batch=self.forward_step(batch, batch_idx=batch_idx), mode="train")

    def validation_step(self, batch, batch_idx):
        output = self.sample(
            batch,
            decode_shape=batch_idx == 0 and self.global_step >= self.config.get("assembly_warm_up_steps", 0),
        )

        for b in range(len(batch["num_parts"])):
            category = batch["category"][b]
            self.log(
                f"eval_{category}/part_acc",
                output["part_acc"][b],
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                batch_size=1,
            )
            self.log(
                f"eval_{category}/shape_chamfer",
                output["shape_chamfer"][b],
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                batch_size=1,
            )
            self.log(
                f"eval_{category}/rmse_t",
                output["rmse_t"][b],
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                batch_size=1,
            )
            self.log(
                f"eval_{category}/rmse_r",
                output["rmse_r"][b],
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                batch_size=1,
            )

        return output

    def test_step(self, batch, batch_idx):
        output = self.sample(
            batch,
            decode_shape=True,
            all_views=False,
        )

        for b in range(len(batch["num_parts"])):
            category = batch["category"][b]
            self.log(
                f"eval_{category}/part_acc",
                output["part_acc"][b],
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                batch_size=1,
            )
            self.log(
                f"eval_{category}/shape_chamfer",
                output["shape_chamfer"][b],
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                batch_size=1,
            )
            self.log(
                f"eval_{category}/rmse_t",
                output["rmse_t"][b],
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                batch_size=1,
            )
            self.log(
                f"eval_{category}/rmse_r",
                output["rmse_r"][b],
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                batch_size=1,
            )

        return output

    # def on_before_optimizer_step(self, optimizer):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     norms = grad_norm(self, norm_type=2)
    #     self.log_dict(norms)

    def configure_optimizers(self):
        if self.config.get("optimizer") is not None:
            params = [
                {
                    "params": self.transformer.assembly_transformer.parameters(),
                    "lr": self.config.get("lr_grouping", {}).get("assembly_lr") or self.config.get("lr"),
                },
                {
                    "params": self.transformer.joint_adapters.parameters(),
                    "lr": self.config.get("lr_grouping", {}).get("joint_adapter_lr") or self.config.get("lr"),
                },
            ]

            if self.config.get("enable_generation", True):
                params.append(
                    {
                        "params": self.transformer.generation_transformer.parameters(),
                        "lr": self.config.get("lr_grouping", {}).get("generation_lr") or self.config.get("lr"),
                    }
                )

            optimizer = instantiate(
                self.config.get("optimizer"),
                params=params,
            )

            if self.config.get("lr_scheduler") is None:
                return optimizer

            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer=optimizer,
                    lr_lambda=instantiate(
                        self.config.get("lr_scheduler"),
                        optimizer=optimizer,
                        max_decay_steps=self.trainer.max_steps,
                    ),
                ),
                "interval": "step",
                "frequency": 1,
            }

            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
            }

        return None

    def compile(self):
        self.transformer.compile()

    def on_save_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = {
            **self.transformer.assembly_transformer.state_dict(prefix="transformer.assembly_transformer."),
            **self.transformer.joint_adapters.state_dict(prefix="transformer.joint_adapters."),
        }
        if not self.config.get("transformer_kwargs", {}).get("freeze_generation_transformer") and self.config.get(
            "enable_generation", True
        ):
            checkpoint["state_dict"].update(
                self.transformer.generation_transformer.state_dict(prefix="transformer.generation_transformer.")
            )
        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        # merge all missing keys into the checkpoint state dict
        model_state_dict = self.state_dict()
        for k in model_state_dict:
            if k not in checkpoint["state_dict"]:
                checkpoint["state_dict"][k] = model_state_dict[k]

        return super().on_load_checkpoint(checkpoint)
