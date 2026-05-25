import lightning as L
import rerun as rr

from source.utils.rerun_utils import log_pose_result


class RerunVisualizationCallback(L.Callback):
    def __init__(
        self,
        *,
        application_id: str,
        grpc_port: int | None = None,
        save_dir: str | None = None,
    ):
        super().__init__()
        self.application_id = application_id
        self.grpc_port = grpc_port
        self.save_dir = save_dir
        rr.init(application_id=self.application_id)
        if self.grpc_port is not None:
            rr.serve_grpc(grpc_port=self.grpc_port)

    def on_test_start(self, trainer, pl_module):
        rr.init(application_id=self.application_id)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        log_pose_result(
            num_parts=batch["num_parts"],
            points_per_part=batch["points_per_part"],
            transforms=outputs["pred_transform_steps"],
            pointclouds=batch["pointclouds"],
            pointclouds_gt=batch["pointclouds_gt"],
            name=batch["name"],
        )
