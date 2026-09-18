SHAPES=(hexagon square triangle)
for shape in ${SHAPES[@]}; do
    python api_infer/inference_api.py \
        --api_key sk-PMcNW-gkgydkI1NvN0OQFw \
        --dataset_path /media/raid/workspace/zhaoyanpeng/model/maze_dataset/${shape}/maze-dataset \
        --output_dir api_results/gpt-image-1/${shape}/3_16 \
        --model gpt-image-1 \
        --base_url "https://litellm.mybigai.ac.cn/" \
        --api_provider openai \
        --split test \
        --num_attempts 5 \
        --num_threads 16 \
        --filter_size_min 3 \
        --filter_size_max 16 \
        --samples_per_size 5 \
        --resume_dir api_results/gpt-image-1/${shape}/3_16 \
        --resolution 1024 \
        --image_size 1024x1024
done