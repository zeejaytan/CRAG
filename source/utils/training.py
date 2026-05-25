from pathlib import Path


def find_unused_params(model):
    """
    Find unused parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to check.

    Returns:
        list: A list of unused parameter names.
    """
    unused_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            # If the parameter requires gradient but has no gradient, it's unused
            unused_params.append(name)
    return unused_params


def find_wandb_run_id(ckpt_path: str) -> str | None:
    """Find the latest wandb run ID from the checkpoint path."""
    ckpt_dir = Path(ckpt_path).parent
    wandb_dir = ckpt_dir / "wandb"
    if (wandb_dir / "latest-run").exists():
        run_log_path = next((wandb_dir / "latest-run").glob("run-*.wandb"))
        run_id = run_log_path.stem.split("-")[-1]
        if len(run_id) == 8:
            return run_id
    return None
