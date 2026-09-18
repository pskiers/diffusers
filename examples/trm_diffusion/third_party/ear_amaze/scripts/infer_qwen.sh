accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 infer_qwen.py \
    --dataset_path /root/private_data/circle/maze-dataset \
    --model_path /root/private_data/model/huggingface/hub/models--Qwen--Qwen-Image-Edit/snapshots/ac7f9318f633fc4b5778c59367c8128225f1e3de \
    --output_dir ./results/qwen_result/circle_5 \
    --split test \
    --save_images \
    --num_inference_steps 10 \
    --max_samples 700 \
    --num_attempts 5 \
    --max_samples_per_size 5 