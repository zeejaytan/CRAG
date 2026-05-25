source .venv/bin/activate
export SLURM_JOB_ID=local

python test.py \
    experiment=stg_2_partnext \
    ckpt_path="./partnext.ckpt" \
    data.batch_size=4 \
    data.categories="['PartNeXt']" \
    loggers=csv \
    trainer.devices=[0] \
    model.image_drop_rate=0.0 \
    trainer.num_nodes=1