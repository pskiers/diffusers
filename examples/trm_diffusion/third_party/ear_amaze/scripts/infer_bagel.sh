#!/bin/bash

# ==================== 内存优化配置 ====================
# 设置随机端口，防止进程残留导致端口冲突
MASTER_PORT=$((($RANDOM % 9000) + 20000 ))

# GPU 设备选择
export CUDA_VISIBLE_DEVICES=0,1,2

# 离线模式
export TRANSFORMERS_OFFLINE=1
export MASTER_PORT=$MASTER_PORT

# ==================== OOM 优化环境变量 ====================
# 限制 PyTorch 的 CUDA 内存分配器
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 启用 CUDA 内存碎片整理（可能略微影响性能，但能减少OOM）
export CUDA_LAUNCH_BLOCKING=0

# 设置 PyTorch 使用更保守的内存分配策略
# export TORCH_CUDNN_V8_API_ENABLED=1

# 限制每个进程的最大内存使用（根据你的GPU显存调整，例如80GB显存可以设置为70GB）
# export CUDA_MEMORY_FRACTION=0.6


checkpoint_paths=(
    "/media/raid/workspace/zhaoyanpeng/code/amaze/results/sft_1600/sft_20260125_112343/checkpoints/0000800"
    # "/media/raid/workspace/zhaoyanpeng/model/0000211"
    # "/media/raid/workspace/zhaoyanpeng/model/0000211"
    # "/media/raid/workspace/zhaoyanpeng/model/0000211"
)

datasets=(

    "/media/raid/workspace/zhaoyanpeng/model/maze_dataset/hexagon/maze-dataset"
    # "/media/raid/workspace/zhaoyanpeng/model/maze_dataset/circle/maze-dataset"

    # "/media/raid/workspace/zhaoyanpeng/model/maze_dataset/square/maze-dataset"
    # "/media/raid/workspace/zhaoyanpeng/model/maze_dataset/triangle/maze-dataset"
)

logdir=(
    # "/media/raid/workspace/zhaoyanpeng/code/flowgrpo/amaze/output_images/hexagon_3_16"
    # "/media/raid/workspace/zhaoyanpeng/code/flowgrpo/amaze/output_images/circle_3_16"
    "/media/raid/workspace/zhaoyanpeng/code/flowgrpo/amaze/output_images/hexagon1600_800/hexagon_3_16"
    # "/media/raid/workspace/zhaoyanpeng/code/flowgrpo/amaze/output_images/circle_1600_500/square_3_16"
    # "/media/raid/workspace/zhaoyanpeng/code/flowgrpo/amaze/output_images/circle_1600_500/triangle_3_16"
)

# 设置日志保存目录
LOG_DIR="./eval_results_logs"
mkdir -p "$LOG_DIR"

echo "------------------------------------------------"
echo "🚀 开始推理: 处理所有配置"
echo "📊 共 ${#checkpoint_paths[@]} 个配置需要处理"
echo "使用端口: $MASTER_PORT"
echo "日志目录: $LOG_DIR"
echo "------------------------------------------------"

# 循环处理每个配置
for i in ${!checkpoint_paths[@]}; do
    # 生成日志文件名
    LOG_FILE="${LOG_DIR}/eval_config_circle_1600_$((i+1)).log"
    
    echo ""
    echo "================================================"
    echo "📝 处理第 $((i+1))/${#checkpoint_paths[@]} 个配置"
    echo "Checkpoint: ${checkpoint_paths[$i]}"
    echo "Dataset: ${datasets[$i]}"
    echo "Logdir: ${logdir[$i]}"
    echo "日志文件: $LOG_FILE"
    echo "================================================"
    
    # 为每次运行生成新的随机端口，防止端口冲突
    MASTER_PORT=$((($RANDOM % 9000) + 20000))
    export MASTER_PORT=$MASTER_PORT
    echo "使用端口: $MASTER_PORT"
    
    # 执行推理，同时输出到终端和日志文件
    accelerate launch \
        --config_file scripts/accelerate_configs/fsdp_eval.yaml \
        --main_process_port $MASTER_PORT \
        infer_bagel.py -- \
        --config config/maze.py:maze_sft_lora_eval \
        --config.sample.eval_num_steps=10 \
        --config.pretrained.checkpoint_path="${checkpoint_paths[$i]}" \
        --config.sample.filter_size_min=3 \
        --config.sample.filter_size_max=16 \
        --config.logdir="${logdir[$i]}" \
        --config.dataset="${datasets[$i]}" \
        2>&1 | tee "$LOG_FILE"
    
    # 检查上一个命令是否成功
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ 完成: 第 $((i+1)) 个配置推理成功"
    else
        echo "❌ 失败: 第 $((i+1)) 个配置推理过程中出现错误，请检查日志 $LOG_FILE"
        exit 1
    fi
    
    # 清理GPU内存并等待，确保资源释放
    echo "🧹 清理GPU内存..."
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 5
done

echo ""
echo "================================================"
echo "🎉 所有配置推理完成！"
echo "================================================"

# # 1. 定义需要测试的 eval_num_steps 列表
# EVAL_STEPS_LIST=(5 10 20 30 40 50)

# # 2. 设置日志保存目录
# LOG_DIR="./eval_results_logs"
# mkdir -p "$LOG_DIR"

# # 3. 设置通用的环境变量
# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
# export TRANSFORMERS_OFFLINE=1

# echo "🚀 开始自动化测试，共 ${#EVAL_STEPS_LIST[@]} 个配置..."

# # 4. 循环执行测试
# for STEPS in "${EVAL_STEPS_LIST[@]}"
# do
#     # 动态生成随机端口，防止进程残留导致端口冲突
#     MASTER_PORT=$((($RANDOM % 9000) + 20000 ))
    
#     # 定义日志文件名 (例如: eval_steps_10.log)
#     LOG_FILE="${LOG_DIR}/eval_steps_${STEPS}.log"
    
#     echo "------------------------------------------------"
#     echo "正在推理: eval_num_steps = $STEPS"
#     echo "使用端口: $MASTER_PORT"
#     echo "日志路径: $LOG_FILE"
#     echo "------------------------------------------------"

#     # 执行命令
#     # 使用 2>&1 将标准错误和标准输出都记录到日志中
#     accelerate launch \
#         --config_file scripts/accelerate_configs/fsdp.yaml \
#         --main_process_port $MASTER_PORT \
#         inference_multi_grpo.py \
#         --config config/maze.py:maze_eval \
#         --config.sample.eval_num_steps=$STEPS \
#         > "$LOG_FILE" 2>&1

#     # 检查上一个命令是否成功
#     if [ $? -eq 0 ]; then
#         echo "✅ 完成: eval_num_steps = $STEPS"
#     else
#         echo "❌ 失败: eval_num_steps = $STEPS，请检查日志 $LOG_FILE"
#     fi

#     # 稍微等待几秒，确保 GPU 内存释放和进程完全退出
#     sleep 5
# done

# echo "🎉 所有测试执行完毕！"