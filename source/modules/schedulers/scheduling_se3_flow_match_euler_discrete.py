from dataclasses import dataclass

from diffusers import ConfigMixin
from diffusers import SchedulerMixin
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput
from pytorch3d import transforms as p3dt
import torch


@dataclass
class SE3FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, 7)` for translations and scalar first quaternions):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor
    prev_additional_sample: torch.FloatTensor | None = None


class SE3FlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        *,
        shift: float = 1.0,
    ):
        super().__init__()
        timesteps = torch.flip(torch.linspace(1, num_train_timesteps, num_train_timesteps), dims=[0])
        sigmas = torch.tensor([t / num_train_timesteps for t in timesteps], dtype=torch.float32)
        sigmas = self.time_shift(sigmas)
        self.timesteps = sigmas * num_train_timesteps
        self.sigmas = sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

        self._step_index = None
        self._begin_index = None

    def time_shift(self, t: torch.Tensor):
        return self.config.get("shift", 1.0) * t / (1 + (self.config.get("shift", 1.0) - 1) * t)

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def _sigma_to_t(self, sigma: torch.FloatTensor) -> torch.FloatTensor:
        return sigma * self.config.get("num_train_timesteps")

    def set_timesteps(
        self,
        num_inference_steps: int = 50,
        sigmas: torch.FloatTensor = None,
    ):
        """
        Set the timesteps for the scheduler.

        By default, we use linspace from sigma_max to sigma_min with num_inference_steps.
        If sigmas is provided, we will use it directly.
        """
        if sigmas is None:
            timesteps = torch.flip(torch.linspace(1, num_inference_steps, num_inference_steps), dims=[0])
            sigmas = torch.tensor([t / num_inference_steps for t in timesteps], dtype=torch.float32)
            sigmas = self.time_shift(sigmas)
        else:
            num_inference_steps = len(sigmas)

        self.timesteps = sigmas * self.config.get("num_train_timesteps")

        sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
        self.sigmas = sigmas
        self._step_index = None
        self._begin_index = None

    def _scale_noise_for_translation(
        self,
        x_0_trans: torch.FloatTensor,  # (B, 3)
        sigma: torch.FloatTensor,  # (B)
        x_1_trans: torch.FloatTensor,  # (B, 3)
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Forward process for translations from x_0 to x_1.
        Args:
            x_0_trans: (B, 3) tensor, ground truth translations.
            sigma: (B) tensor, sigmas for each sample
            x_1_trans: (B, 3) tensor, pure noise translations.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]:
                - x_t_trans: (B, 3) tensor, noisy translations.
                - trans_vec_field: (B, 3) tensor, translation vector field.
        """
        sigma = sigma.unsqueeze(-1)
        x_t_trans = (1 - sigma) * x_0_trans + sigma * x_1_trans
        trans_vec_field = x_1_trans - x_0_trans
        return (x_t_trans, trans_vec_field)

    def _scale_noise_for_rotation(
        self,
        x_0_rot: torch.FloatTensor,  # (B, 4)
        sigma: torch.FloatTensor,  # (B)
        x_1_rot: torch.FloatTensor,  # (B, 4)
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Forward process for rotations from x_0 to x_1.
        Args:
            x_0_rot: (B, 4) tensor, ground truth rotations, scalar first quaternions.
            sigma: (B) tensor, sigmas for each sample
            x_1_rot: (B, 4) tensor, pure noise rotations, scalar first quaternions.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]:
                - x_t_rot: (B, 4) tensor, noisy rotations, scalar first quaternions.
                - rot_vec_field: (B, 3) tensor, rotation vector field.
        """
        sigma = sigma.unsqueeze(-1)
        x_0_rot_mat = p3dt.quaternion_to_matrix(x_0_rot)
        x_1_rot_mat = p3dt.quaternion_to_matrix(x_1_rot)

        # Calculate the rotation vector field
        rot_vec_field = p3dt.matrix_to_axis_angle(x_1_rot_mat)
        x_t_rot_mat = p3dt.axis_angle_to_matrix(sigma * rot_vec_field) @ x_0_rot_mat
        x_t_rot = p3dt.matrix_to_quaternion(x_t_rot_mat)
        return (x_t_rot, rot_vec_field)

    def scale_noise(
        self,
        sample: torch.FloatTensor,  # (B, 7)
        timestep: torch.FloatTensor,  # (B)
        noise: torch.FloatTensor,  # (B, 7)
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Args:
            sample (`torch.FloatTensor`): (B, 7) tensor, translations and scalar first quaternions.
            timesteps (`torch.FloatTensor`): (B) tensor, timesteps for each sample.
            noise (`torch.FloatTensor`): (B, 7) tensor, noise for each sample.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]:
                - x_t: (B, 7) tensor, noisy translations and scalar first quaternions.
                - vec_field: (B, 6) tensor, translation and rotation vector fields.
        """
        sigmas = self.sigmas.to(device=sample.device, dtype=sample.dtype)
        schedule_timesteps = self.timesteps.to(sample.device)
        timestep = timestep.to(sample.device)

        step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timestep]
        sigma = sigmas[step_indices].flatten()

        x_t_trans, trans_vec_field = self._scale_noise_for_translation(sample[..., :3], sigma, noise[..., :3])
        x_t_rots, rot_vec_field = self._scale_noise_for_rotation(sample[..., 3:], sigma, noise[..., 3:])

        return torch.cat([x_t_trans, x_t_rots], dim=-1), torch.cat([trans_vec_field, rot_vec_field], dim=-1)

    def scale_non_se3_noise(
        self,
        sample: torch.FloatTensor,  # (B, ...)
        timestep: torch.FloatTensor,  # (B)
        noise: torch.FloatTensor,  # (B, ...)
    ) -> torch.FloatTensor:
        sigmas = self.sigmas.to(device=sample.device, dtype=sample.dtype)
        schedule_timesteps = self.timesteps.to(sample.device)

        step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timestep]
        sigma = sigmas[step_indices].flatten()

        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)

        return (1.0 - sigma) * sample + sigma * noise

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def _step_for_translation(
        self,
        vec_field: torch.FloatTensor,
        delta_sigma: torch.FloatTensor,
        sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Args:
            vec_field (`torch.FloatTensor`): (B, 3) tensor, translation vector field.
            delta_sigma (`torch.FloatTensor`): (B) tensor.
            sample (`torch.FloatTensor`): (B, 3) tensor, sample translations.

        Returns:
            prev_sample (`torch.FloatTensor`): (B, 3) tensor, denoised translations.
        """
        return sample + delta_sigma * vec_field

    def _step_for_rotation(
        self,
        vec_field: torch.FloatTensor,
        delta_sigma: torch.FloatTensor,
        sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Args:
            vec_field (`torch.FloatTensor`): (B, 3) tensor, rotation vector field.
            delta_sigma (`torch.FloatTensor`): (B) tensor.
            sample (`torch.FloatTensor`): (B, 4) tensor, sample rotations, scalar first quaternions.

        Returns:
            prev_sample (`torch.FloatTensor`): (B, 4) tensor, denoised rotations, scalar first quaternions.
        """
        with torch.autocast(device_type="cuda", enabled=False):
            prev_sample = p3dt.axis_angle_to_matrix(delta_sigma * vec_field) @ p3dt.quaternion_to_matrix(sample)

            return p3dt.matrix_to_quaternion(prev_sample)

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: float | torch.FloatTensor,
        sample: torch.FloatTensor,
        model_additional_output: torch.FloatTensor | None = None,
        additional_sample: torch.FloatTensor | None = None,
    ) -> SE3FlowMatchEulerDiscreteSchedulerOutput:
        """
        Args:
            model_output (`torch.FloatTensor`):
                The model output. Should be a tuple of (trans_vec_field, rot_vec_field), each of shape (B, 3).
            timestep (`Union[float, torch.FloatTensor]`):
                The current timestep.
            sample (`torch.FloatTensor`):
                The sample for the current timestep of shape (B, 7).
                First 3 elements are translation
                Last 4 elements are quaternion, scalar first.

        Returns:
            `torch.FloatTensor`:
                The denoised sample.
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        # Avoid precision issues
        sample = sample.to(torch.float32)
        model_output = model_output.to(torch.float32)
        # sigma_next - sigma < 0
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        delta_sigma = sigma_next - sigma

        prev_sample_trans = self._step_for_translation(
            model_output[..., :3],
            delta_sigma,
            sample[..., :3],
        )

        prev_sample_rot = self._step_for_rotation(
            model_output[..., 3:],
            delta_sigma,
            sample[..., 3:],
        )

        prev_additional_sample = None
        if additional_sample is not None and model_additional_output is not None:
            # avoid precision issues
            additional_sample = additional_sample.to(torch.float32)
            model_additional_output = model_additional_output.to(torch.float32)
            prev_additional_sample = additional_sample + delta_sigma * model_additional_output
            prev_additional_sample = prev_additional_sample.to(model_additional_output.dtype)

        prev_sample = torch.cat([prev_sample_trans, prev_sample_rot], dim=-1)
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        return SE3FlowMatchEulerDiscreteSchedulerOutput(
            prev_sample=prev_sample,
            prev_additional_sample=prev_additional_sample,
        )

    def __len__(self):
        return self.config.num_train_timesteps


if __name__ == "__main__":
    scheduler = SE3FlowMatchEulerDiscreteScheduler()
    print(scheduler.sigmas)
