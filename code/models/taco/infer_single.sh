CUDA_VISIBLE_DEVICES=0 \
    python scripts/infer.py --gpus=0 \
    --frame_width=384 \
    --frame_height=384 \
    --config_path=configs/infer_vac.yaml \
    --model_path=../../checkpoints/taco/lora.ckpt \
    --output=../output2 \
    --frame_start=0 --frame_stride=2 --frame_rate=12 \
    --input_example=nico_1 \
    --autocast=1 --num_samples=10 --num_steps=50
