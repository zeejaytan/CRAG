#!/bin/bash
set -e

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
source .venv/bin/activate

export HYDRA_FULL_ERROR=1

# Root directory holding the source HDF5 datasets. Override with: DATA_ROOT=/path/to/data
DATA_ROOT="${DATA_ROOT:-./data}"

export CKPT_PATH=null
export EXPERIMENT=$1
export EXPERIMENT_NAME=$2
# If experiment name is not provided, use the experiment name as the experiment
if [ -z "$EXPERIMENT_NAME" ]; then
    export EXPERIMENT_NAME=$EXPERIMENT
fi

export SLURM_JOB_ID="${SLURM_JOB_ID:-local}"

# if output/${EXPERIMENT_NAME}/last.ckpt exists, set CKPT_PATH to that path
if [ -f output/${EXPERIMENT_NAME}/last.ckpt ]; then
    export CKPT_PATH=output/${EXPERIMENT_NAME}/last.ckpt
    echo "Resuming from checkpoint: ${CKPT_PATH}"
else
    echo "No checkpoint found, starting from scratch."
fi

# Export environment variables for distributed training
eval $(tr '\0' '\n' < /proc/1/cmdline | grep '^export ' | sed 's/^export //g' | xargs -d '\n' printf 'export %s; ')

# Copy data to /dev/shm
if [ -f /dev/shm/PartNeXt.hdf5 ]; then
    echo "Data already exists in /dev/shm, skipping copy."
else
    echo "Copying data to /dev/shm..."
    dd if="$DATA_ROOT/PartNeXt.hdf5" of=/dev/shm/PartNeXt.hdf5 bs=10M status=progress &
    wait
fi

torchrun \
    --nproc_per_node=$PET_NPROC_PER_NODE \
    --nnodes=$PET_NNODES \
    --node_rank=$PET_NODE_RANK \
    --master_addr=$PET_MASTER_ADDR \
    --master_port=$PET_MASTER_PORT \
    --log-dir=logs/$EXPERIMENT \
    train.py \
        experiment=$EXPERIMENT \
        experiment_name=$EXPERIMENT_NAME \
        trainer.num_nodes=$PET_NNODES \
        project_name=CRAG \
        data.data_root.PartNeXt=/dev/shm/PartNeXt.hdf5 \
        data.categories="[PartNeXt]" \
        loggers=wandb \
        ++loggers.wandb.offline=true \
        ++trainer.num_sanity_val_steps=1 \
        ckpt_path=${CKPT_PATH}