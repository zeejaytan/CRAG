import math
from typing import Literal

from diffusers import ConfigMixin
from diffusers import SchedulerMixin
from diffusers.configuration_utils import register_to_config
import pytorch3d.transforms as T
import torch


class SE3FlowMatchEulerContinuousScheduler(SchedulerMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        sigma_min=1e-5,
        t_schedule: Literal["linear", "logit_normal"] = "linear",
        t_schedule_kwargs: dict | None = None,
    ):
        if t_schedule_kwargs is None:
            t_schedule_kwargs = {}
        super().__init__()

    def sample_t(
        self,
        batch_size: int,
    ):
        if self.config.get("t_schedule") == "linear":
            t = torch.rand(batch_size)
        elif self.config.get("t_schedule") == "logit_normal":
            mean = self.config.get("t_schedule_kwargs").get("mean", 0.0)
            std = self.config.get("t_schedule_kwargs").get("std", 1.0)
            t = torch.sigmoid(torch.randn(batch_size) * std + mean)
        elif self.config.get("t_schedule") == "u_shape":
            t = torch.rand(batch_size) * 2 - 1
            a = self.config.get("t_schedule_kwargs").get("a", 4.0)
            t = torch.asinh(t * math.sinh(a)) / a
            t = (t + 1) / 2
        else:
            raise ValueError(f"Unknown t_schedule: {self.config.get('t_schedule')}")
        return t

    def scale_noise(
        self,
        *,
        x_0: torch.FloatTensor,
        t: torch.FloatTensor,  # (B,)
        noise: torch.FloatTensor | None = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        if noise is None:
            noise = torch.randn_like(x_0)
        assert noise.shape == x_0.shape, (
            f"noise and x_0 must have the same shape, but got {noise.shape} and {x_0.shape}"
        )

        t = t.view(-1, *([1] * (len(x_0.shape) - 1)))  # (B, 1, ..., 1)
        return (1 - t) * x_0 + (self.config.get("sigma_min") + (1 - self.config.get("sigma_min")) * t) * noise, noise

    def scale_noise_se3(
        self,
        *,
        x_0: torch.FloatTensor,
        t: torch.FloatTensor,  # (B,)
        noise: torch.FloatTensor | None = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        with torch.autocast(device_type=x_0.device.type, enabled=False, dtype=torch.float32):
            assert x_0.shape == (x_0.shape[0], 7), f"x_0 must have shape (B, 7), but got {x_0.shape}"
            if noise is None:
                batch_size = x_0.shape[0]
                noisy_trans = torch.randn(batch_size, 3, dtype=x_0.dtype, device=x_0.device)
                noisy_quat = T.random_quaternions(batch_size, dtype=x_0.dtype, device=x_0.device)
                noise = torch.cat([noisy_trans, noisy_quat], dim=-1)

            assert noise.shape == x_0.shape, (
                f"noise and x_0 must have the same shape, but got {noise.shape} and {x_0.shape}"
            )
            t = t.view(-1, 1)  # (B, 1)

            # cast to float32 to avoid numerical issues
            x_0 = x_0.float()
            t = t.float()
            noise = noise.float()

            # Scale translation and rotation separately
            # For translation, it is same as normal interpolation
            x_t_trans = (1 - t) * x_0[:, :3] + (
                self.config.get("sigma_min") + (1 - self.config.get("sigma_min")) * t
            ) * noise[:, :3]

            # For rotation, map to the tangent space, do interpolation there, and map back
            x_0_rot_mat = T.quaternion_to_matrix(x_0[:, 3:])
            x_1_rot_mat = T.quaternion_to_matrix(noise[:, 3:])
            rot_rel = torch.matmul(x_1_rot_mat, x_0_rot_mat.transpose(1, 2))
            rot_rel_log = T.so3_log_map(rot_rel)
            rot_rel_log_scaled = (self.config.get("sigma_min") + (1 - self.config.get("sigma_min")) * t) * rot_rel_log
            scaled_relative_rot = T.so3_exp_map(rot_rel_log_scaled)
            x_t_rot = torch.matmul(scaled_relative_rot, x_0_rot_mat)
            x_t_quat = T.matrix_to_quaternion(x_t_rot)

            return torch.cat([x_t_trans, x_t_quat], dim=-1), noise

    def get_v(self, x_0: torch.FloatTensor, noise: torch.FloatTensor, t: torch.FloatTensor):
        return (1 - self.config.get("sigma_min")) * noise - x_0

    def get_v_se3(self, x_0: torch.FloatTensor, noise: torch.FloatTensor, t: torch.FloatTensor):
        with torch.autocast(device_type=x_0.device.type, enabled=False, dtype=torch.float32):
            # cast to float32 to avoid numerical issues
            x_0 = x_0.float()
            noise = noise.float()
            t = t.float()

            v_trans = self.get_v(x_0[:, :3], noise[:, :3], t)
            v_rot = T.so3_log_map(
                torch.matmul(
                    T.quaternion_to_matrix(noise[:, 3:]),
                    T.quaternion_to_matrix(x_0[:, 3:]).transpose(1, 2),
                )
            )
            return torch.cat([v_trans, v_rot], dim=-1)

    def step(
        self,
        v_pred: torch.FloatTensor,
        t: float,
        t_prev: float,
        x_t: torch.FloatTensor,
    ):
        # t > t_prev
        return x_t + (t_prev - t) * v_pred

    def step_se3(
        self,
        v_pred: torch.FloatTensor,
        t: float,
        t_prev: float,
        x_t: torch.FloatTensor,
    ):
        with torch.autocast(device_type=v_pred.device.type, enabled=False, dtype=torch.float32):
            # cast to float32 to avoid numerical issues
            v_pred = v_pred.float()
            x_t = x_t.float()

            delta_t = t_prev - t
            x_prev_trans = self.step(v_pred[:, :3], t, t_prev, x_t[:, :3])
            delta_rot = T.so3_exp_map(delta_t * v_pred[:, 3:])
            x_t_rot_mat = T.quaternion_to_matrix(x_t[:, 3:])
            x_prev_rot_mat = torch.matmul(delta_rot, x_t_rot_mat)
            x_prev_quat = T.matrix_to_quaternion(x_prev_rot_mat)
            return torch.cat([x_prev_trans, x_prev_quat], dim=-1)
