#!/bin/bash

# 批量迷宫生成脚本
# 为每种类型的迷宫从3x3到10x10生成40张不重复的图片

# set -e  # 遇到错误立即退出
# 1. 先判断连通性；2. 再看起点和终点位置
# 配置参数
BATCH_SIZE=10
MIN_MAZE_SIZE=5
MAX_MAZE_SIZE=6
OUTPUT_DIR="./test/generated_mazes"
SOLUTION_DIR="./test/generated_solutions"
NO_MARKER_DIR="./test/generated_mazes_no_markers"
MASK_DIR="./test/generated_masks"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_dependencies() {
    log_info "检查依赖..."

    if ! command -v node &> /dev/null; then
        log_error "Node.js 未安装，请先安装 Node.js"
        exit 1
    fi

    if [ ! -f "batch-maze-generator_test.js" ]; then
        log_error "批量生成器文件不存在: batch-maze-generator_test.js"
        exit 1
    fi

    if [ ! -f "package.json" ]; then
        log_error "package.json 文件不存在"
        exit 1
    fi

    # 检查 node_modules
    if [ ! -d "node_modules" ]; then
        log_info "安装 npm 依赖..."
        npm install
    fi

    log_success "依赖检查完成"
}

# 创建输出目录
create_directories() {
    log_info "创建输出目录..."
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$SOLUTION_DIR"
    mkdir -p "$NO_MARKER_DIR"
    mkdir -p "$MASK_DIR"
    log_success "输出目录创建完成"
}

# 生成时间戳
get_timestamp() {
    date +"%Y%m%d_%H%M%S"
}

# 生成随机种子
get_random_seed() {
    echo $((RANDOM * 32768 + RANDOM))
}

# 生成单类型迷宫批次
generate_maze_batch() {
    local maze_type="$1"
    local shape="$2"
    local algorithm="$3"
    local exit_config="$4"
    local size_param="$5"

    log_info "开始生成 ${MAZE_SIZE}x${MAZE_SIZE} ${maze_type} 迷宫批次 (${BATCH_SIZE}张)..."

    local success_count=0
    local fail_count=0
    local timestamp=$(get_timestamp)

    for i in $(seq 1 $BATCH_SIZE); do
        local seed=$(get_random_seed)
        local padded_num=$(printf "%03d" $i)
        local filename="${maze_type}_${shape}_${algorithm}_${MAZE_SIZE}x${MAZE_SIZE}_${padded_num}_${timestamp}_${exit_config}.png"

        # 构建命令参数
        local cmd_args="single --shape $shape --algorithm $algorithm --exitConfig $exit_config --seed $seed --filename $filename"

        # 根据形状添加尺寸参数
        if [ "$shape" = "circle" ]; then
            cmd_args="$cmd_args --layers $MAZE_SIZE"
        else
            cmd_args="$cmd_args --width $MAZE_SIZE --height $MAZE_SIZE"
        fi

        # 执行生成命令
        if node batch-maze-generator_test.js $cmd_args > /dev/null 2>&1; then
            ((success_count++))
            printf "\r${GREEN}[${MAZE_SIZE}x${MAZE_SIZE} ${maze_type}]${NC} 进度: %d/%d (成功: %d, 失败: %d)" $i $BATCH_SIZE $success_count $fail_count
        else
            ((fail_count++))
            printf "\r${RED}[${MAZE_SIZE}x${MAZE_SIZE} ${maze_type}]${NC} 进度: %d/%d (成功: %d, 失败: %d)" $i $BATCH_SIZE $success_count $fail_count
        fi
    done

    echo ""
    log_success "${MAZE_SIZE}x${MAZE_SIZE} ${maze_type} 批次完成: 成功 $success_count, 失败 $fail_count"
}

# 主函数
main() {
    local start_time=$(date +%s)

    echo "=================================="
    echo "     批量迷宫生成脚本 v1.0"
    echo "=================================="
    echo ""

    log_info "开始批量生成迷宫..."
    log_info "每种类型生成数量: $BATCH_SIZE"
    log_info "迷宫尺寸范围: ${MIN_MAZE_SIZE}x${MIN_MAZE_SIZE} 到 ${MAX_MAZE_SIZE}x${MAX_MAZE_SIZE}"
    echo ""

    # 检查依赖和创建目录
    check_dependencies
    create_directories

    echo ""
    log_info "开始生成四种类型的迷宫，尺寸从 ${MIN_MAZE_SIZE}x${MIN_MAZE_SIZE} 到 ${MAX_MAZE_SIZE}x${MAX_MAZE_SIZE}..."
    echo ""

    # 遍历所有尺寸
    for (( MAZE_SIZE=$MIN_MAZE_SIZE; MAZE_SIZE<=$MAX_MAZE_SIZE; MAZE_SIZE++ ))
    do
        log_info "开始生成 ${MAZE_SIZE}x${MAZE_SIZE} 尺寸的迷宫..."
        echo ""

        # 1. 生成方形迷宫 (递归回溯)
        generate_maze_batch "square" "square" "recursiveBacktrack" "vertical"
        generate_maze_batch "square" "square" "recursiveBacktrack" "hardest"
        generate_maze_batch "square" "square" "recursiveBacktrack" "horizontal"

        # 2. 生成三角形迷宫 (递归回溯)
        # generate_maze_batch "triangle" "triangle" "recursiveBacktrack" "vertical"

        # # 3. 生成六边形迷宫 (真Prims算法)
        # generate_maze_batch "hexagon" "hexagon" "truePrims" "hardest"

        # # 4. 生成圆形迷宫 (递归回溯)
        # generate_maze_batch "circle" "circle" "recursiveBacktrack" "vertical"
        
        echo ""
    done

    # 计算总用时
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))

    echo ""
    echo "=================================="
    log_success "所有迷宫生成完成!"
    echo "=================================="
    log_info "总用时: ${minutes}分${seconds}秒"
    log_info "迷宫文件位置: $OUTPUT_DIR"
    log_info "解答文件位置: $SOLUTION_DIR"
    log_info "掩码文件位置: $MASK_DIR"

    # 统计生成的文件数量
    local maze_count=$(find "$OUTPUT_DIR" -name "*.png" 2>/dev/null | wc -l)
    local solution_count=$(find "$SOLUTION_DIR" -name "*.png" 2>/dev/null | wc -l)
    local no_maze_count=$(find "$NO_MARKER_DIR" -name "*.png" 2>/dev/null | wc -l)
    local mask_count=$(find "$MASK_DIR" -name "*.png" 2>/dev/null | wc -l)

    log_info "生成统计:"
    log_info "  - 迷宫文件: $maze_count 个"
    log_info "  - 解答文件: $solution_count 个"
    log_info "  - no marker文件: $no_maze_count 个"
    log_info "  - 掩码文件: $mask_count 个"
    log_info "  - 总计: $((maze_count + solution_count + no_maze_count + mask_count)) 个文件"
    log_info "  - 尺寸范围: ${MIN_MAZE_SIZE}x${MIN_MAZE_SIZE} 到 ${MAX_MAZE_SIZE}x${MAX_MAZE_SIZE}"
    log_info "  - 每尺寸类型数: 4"
    log_info "  - 每类型生成数: $BATCH_SIZE"

    echo ""
    log_success "批量生成任务已完成! 🎉"
}

# 处理中断信号
trap 'log_warning "收到中断信号，正在清理..."; exit 130' INT TERM

# 运行主函数
main "$@"