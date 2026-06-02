CUDA_VISIBLE_DEVICES=0,1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    torchrun --nproc_per_node=2 main.py \
    --base=configs/train_vac.yaml \
    --name=TrackMyLift-AI \
    --projectname TrackMyLift-AI \
    --seed=1234 \
    --num_nodes=1 \
    --wandb=1 \
    --no-test true \
    data.params.frame_width=384 \
    data.params.frame_height=384 \
    data.params.batch_size=1 \
    data.params.num_workers=4 \
    data.params.data_gpu=0 \
    lightning.callbacks.image_logger.params.batch_frequency=200 \
    lightning.trainer.devices="0,1" \
    lightning.trainer.accumulate_grad_batches=2