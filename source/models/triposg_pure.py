import lightning as L
import numpy as np
from pytorch3d.loss.chamfer import chamfer_distance
import torch
from transformers import Dinov2Model
import trimesh

from third_party.tripo_sg.triposg.inference_utils import flash_extract_geometry
from third_party.tripo_sg.triposg.models.autoencoders import TripoSGVAEModel
from third_party.tripo_sg.triposg.models.transformers import TripoSGDiTModel
from third_party.tripo_sg.triposg.schedulers.scheduling_rectified_flow import RectifiedFlowScheduler


class TripoSGPureModel(L.LightningModule):
    """
    Pure TripoSG image-to-3D generation benchmark.

    Loads the same pretrained components as CragUnifiedModel (image encoder,
    generation transformer, VAE) but runs the generation branch only — no
    assembly, no part transforms.  Implements the TripoSG denoising loop with
    classifier-free guidance inline instead of going through TripoSGPipeline.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "VAST-AI/TripoSG",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        view_index: int = 0,
        num_sample_points: int = 8192,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.view_index = view_index
        self.num_sample_points = num_sample_points

        # --- Load pretrained components (same source as CragUnifiedModel) ---
        self.image_encoder = Dinov2Model.from_pretrained(
            pretrained_model_name_or_path, subfolder="image_encoder_dinov2"
        )
        self.image_encoder.eval()
        for p in self.image_encoder.parameters():
            p.requires_grad = False

        self.transformer = TripoSGDiTModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="transformer"
        )
        self.transformer.eval()
        for p in self.transformer.parameters():
            p.requires_grad = False

        self.vae = TripoSGVAEModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae"
        )
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        self.scheduler = RectifiedFlowScheduler()

    # ------------------------------------------------------------------
    # Image encoding (with CFG)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def encode_image(self, pixel_values: torch.Tensor):
        """Encode a pre-processed image tensor through DINOv2, returning (cond, uncond) embeddings.

        Args:
            pixel_values: (1, 3, H, W) — already normalised by the CRAG dataloader.
        """
        dtype = next(self.image_encoder.parameters()).dtype
        pixel_values = pixel_values.to(device=self.device, dtype=dtype)
        image_embeds = self.image_encoder(pixel_values=pixel_values).last_hidden_state  # (1, N, D)
        uncond_embeds = torch.zeros_like(image_embeds)
        return image_embeds, uncond_embeds

    # ------------------------------------------------------------------
    # Generation inference loop
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(self, pixel_values: torch.Tensor):
        """Run the full TripoSG denoising loop on a pre-processed image tensor → trimesh.

        Args:
            pixel_values: (1, 3, H, W) — already normalised by the dataloader.
        """
        image_embeds, uncond_embeds = self.encode_image(pixel_values)

        do_cfg = self.guidance_scale > 1.0
        if do_cfg:
            encoder_hidden_states = torch.cat([uncond_embeds, image_embeds], dim=0)
        else:
            encoder_hidden_states = image_embeds

        # Prepare timesteps
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # Initialise latents from pure noise
        num_tokens = 2048
        latent_channels = self.transformer.config.in_channels
        latents = torch.randn(
            1, num_tokens, latent_channels,
            device=self.device, dtype=encoder_hidden_states.dtype,
        )

        # Denoising loop
        for t in timesteps:
            latent_input = torch.cat([latents] * 2) if do_cfg else latents
            t_input = t.expand(latent_input.shape[0])

            noise_pred = self.transformer(
                latent_input,
                t_input,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]

            if do_cfg:
                pred_uncond, pred_cond = noise_pred.chunk(2)
                noise_pred = pred_uncond + self.guidance_scale * (pred_cond - pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Decode latents → mesh
        self.vae.set_flash_decoder()
        try:
            mesh_v_f = flash_extract_geometry(
                latents=latents,
                vae=self.vae,
                bounds=(-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
                octree_depth=9,
            )[0]
            mesh = trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1])
        except:
            mesh = trimesh.Trimesh()
        return mesh

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_points(pts: torch.Tensor) -> torch.Tensor:
        """Center and scale to [-1, 1]^3 for coordinate-agnostic Chamfer comparison."""
        center = pts.mean(dim=0)
        pts = pts - center
        scale = pts.abs().max().clamp(min=1e-6)
        return pts / scale

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def test_step(self, batch, batch_idx):
        batch_size = len(batch["name"])

        meshes: list[trimesh.Trimesh] = []
        shape_chamfers: list[torch.Tensor] = []

        for i in range(batch_size):
            try:
                # 1. Pick a single rendered view (pre-processed tensor from dataloader).
                pixel_values = batch["renderings"][i, self.view_index].unsqueeze(0)  # (1, 3, H, W)

                # 2. Run TripoSG generation.
                gen_mesh = self.generate(pixel_values)
                meshes.append(gen_mesh)

                # 3. Compute Chamfer distance vs GT full mesh.
                gt_mesh: trimesh.Trimesh = batch["full_mesh"][i]

                gen_pts, _ = trimesh.sample.sample_surface(gen_mesh, self.num_sample_points)
                gt_pts, _ = trimesh.sample.sample_surface(gt_mesh, self.num_sample_points)

                gen_pts_t = self._normalize_points(
                    torch.tensor(gen_pts, dtype=torch.float32, device=self.device)
                )
                gt_pts_t = self._normalize_points(
                    torch.tensor(gt_pts, dtype=torch.float32, device=self.device)
                )

                chamfer_val, _ = chamfer_distance(
                    gen_pts_t.unsqueeze(0),
                    gt_pts_t.unsqueeze(0),
                    single_directional=False,
                    point_reduction="mean",
                    batch_reduction=None,
                )
                shape_chamfers.append(chamfer_val[0])
            except:
                shape_chamfers.append(None)

        # 4. Log per-category metrics (mirrors CragUnifiedModel logging convention).
        for i in range(len(shape_chamfers)):
            category = batch["category"][i]
            self.log(
                f"eval_{category}/shape_chamfer",
                shape_chamfers[i],
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                batch_size=1,
            )

        return {
            "meshes": meshes,
            "shape_chamfer": shape_chamfers,
            "name": batch["name"],
            "category": batch["category"],
        }

    def configure_optimizers(self):
        return []
