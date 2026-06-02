CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --base=configs/train_vac.yaml \
    --name=TrackMyLift-AI --projectname=TrackMyLift-AI --seed=1234 --num_nodes=1 --wandb=1 \
    model.base_learning_rate=2e-5 \
    model.params.optimizer_config.params.foreach=False \
    data.params.dset_root=/media/pc/hdd2/data-students/nicolasdelplanque/TrackMyLift-AI/data/datasets/custom \
    data.params.frame_width=384 \
    data.params.frame_height=384 \
    data.params.batch_size=1 \
    data.params.num_workers=4 \
    data.params.data_gpu=0 \
    data.params.train_path=/media/pc/hdd2/data-students/nicolasdelplanque/TrackMyLift-AI/data/datasets/custom/train.json \
    data.params.val_path=/media/pc/hdd2/data-students/nicolasdelplanque/TrackMyLift-AI/data/datasets/custom/val.json \
    lightning.callbacks.image_logger.params.batch_frequency=200 