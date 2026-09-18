#!/usr/bin/env python3
"""
迷宫图像转Parquet数据集转换器
将generated_mazes和generated_solutions目录中的图像转换为Parquet格式数据集,直接存储pil图像的版本
"""

import os
import sys
import argparse
import pandas as pd
from PIL import Image
import re
from pathlib import Path
import uuid
from tqdm import tqdm
import pickle
import base64

def serialize_pil_image(image):
    """
    将PIL Image对象序列化为可存储的格式

    Args:
        image (PIL.Image.Image): PIL图像对象

    Returns:
        str: Base64编码的图像数据
    """
    try:
        # 将PIL Image保存到字节流
        from io import BytesIO
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)

        # 编码为base64字符串
        image_bytes = buffer.getvalue()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        return image_b64
    except Exception as e:
        print(f"错误: 无法序列化PIL图像: {e}")
        return None

def deserialize_pil_image(image_b64):
    """
    从base64字符串反序列化PIL Image对象

    Args:
        image_b64 (str): Base64编码的图像数据

    Returns:
        PIL.Image.Image: PIL图像对象
    """
    try:
        from io import BytesIO
        # 解码base64字符串
        image_bytes = base64.b64decode(image_b64)

        # 从字节流创建PIL Image
        buffer = BytesIO(image_bytes)
        image = Image.open(buffer)
        return image
    except Exception as e:
        print(f"错误: 无法反序列化PIL图像: {e}")
        return None

def load_image_as_pil(image_path):
    """
    加载图像并返回PIL Image对象

    Args:
        image_path (str): 图像文件路径

    Returns:
        PIL.Image.Image: PIL图像对象
    """
    try:
        img = Image.open(image_path)
        # 转换为RGB模式确保一致性
        img = img.convert('RGB')
        return img
    except Exception as e:
        print(f"错误: 无法加载图像 {image_path}: {e}")
        return None

def find_matching_solution(maze_filename, solution_dir):
    """
    根据迷宫文件名找到对应的解答文件

    Args:
        maze_filename (str): 迷宫文件名
        solution_dir (str): 解答目录路径

    Returns:
        str or None: 匹配的解答文件路径，如果没找到返回None
    """
    # 从迷宫文件名提取基础名称（去掉.png后缀）
    base_name = os.path.splitext(maze_filename)[0]

    # 可能的解答文件名模式
    possible_names = [
        f"{base_name}_solution.png",
        f"{base_name}_sol.png",
        # 如果原文件名已经包含时间戳等信息，尝试提取核心部分
    ]

    for possible_name in possible_names:
        solution_path = os.path.join(solution_dir, possible_name)
        if os.path.exists(solution_path):
            return solution_path

    # 如果直接匹配失败，尝试模糊匹配
    # 提取文件名中的关键信息（形状、算法、尺寸等）
    try:
        # 匹配模式: shape_algorithm_size_difficulty_score_timestamp
        pattern = r'([^_]+)_([^_]+)_([^_]+)_([^_]+)_(\d+)_(.+)'
        match = re.match(pattern, base_name)

        if match:
            shape, algorithm, size, difficulty, score, timestamp = match.groups()

            # 在解答目录中查找包含相同特征的文件
            for solution_file in os.listdir(solution_dir):
                if (shape in solution_file and
                    algorithm in solution_file and
                    size in solution_file and
                    'solution' in solution_file):
                    return os.path.join(solution_dir, solution_file)
    except:
        pass

    return None

def extract_maze_info(filename):
    """
    从文件名提取迷宫信息

    Args:
        filename (str): 文件名

    Returns:
        dict: 包含迷宫信息的字典
    """
    base_name = os.path.splitext(filename)[0]

    # 默认信息
    info = {
        'shape': 'unknown',
        'algorithm': 'unknown',
        'size': 'unknown',
        'difficulty': 'unknown',
        'score': 0
    }

    try:
        # 尝试解析文件名模式
        parts = base_name.split('_')

        if len(parts) >= 3:
            info['shape'] = parts[0] if parts[0] else 'unknown'
            info['algorithm'] = parts[1] if parts[1] else 'unknown'
            info['size'] = parts[2] if parts[2] else 'unknown'

        if len(parts) >= 5:
            info['difficulty'] = parts[3] if parts[3] else 'unknown'
            try:
                info['score'] = int(parts[4]) if parts[4].isdigit() else 0
            except:
                info['score'] = 0

    except Exception as e:
        print(f"警告: 无法解析文件名 {filename}: {e}")

    return info

def create_parquet_dataset(maze_dir, solution_dir, output_path, instruction="Add the solution path for the maze"):
    """
    创建Parquet格式数据集

    Args:
        maze_dir (str): 迷宫图像目录
        solution_dir (str): 解答图像目录
        output_path (str): 输出Parquet文件路径
        instruction (str): 指令文本

    Returns:
        bool: 是否成功创建数据集
    """

    # 检查目录是否存在
    if not os.path.exists(maze_dir):
        print(f"错误: 迷宫目录不存在: {maze_dir}")
        return False

    if not os.path.exists(solution_dir):
        print(f"错误: 解答目录不存在: {solution_dir}")
        return False

    # 获取所有迷宫图像文件
    maze_files = [f for f in os.listdir(maze_dir) if f.lower().endswith('.png')]

    if not maze_files:
        print(f"错误: 在 {maze_dir} 中没有找到PNG文件")
        return False

    print(f"找到 {len(maze_files)} 个迷宫文件")

    # 存储数据集记录
    dataset_records = []
    successful_pairs = 0
    failed_pairs = 0

    # 处理每个迷宫文件
    for maze_file in tqdm(maze_files, desc="处理迷宫文件"):
        maze_path = os.path.join(maze_dir, maze_file)

        # 查找对应的解答文件
        solution_path = find_matching_solution(maze_file, solution_dir)

        if solution_path is None:
            print(f"警告: 找不到 {maze_file} 对应的解答文件")
            failed_pairs += 1
            continue

        # 加载图像
        original_img = load_image_as_pil(maze_path)
        solution_img = load_image_as_pil(solution_path)

        if original_img is None or solution_img is None:
            print(f"警告: 无法加载图像对: {maze_file}")
            failed_pairs += 1
            continue

        # 提取迷宫信息
        maze_info = extract_maze_info(maze_file)

        # 生成唯一ID
        unique_id = str(uuid.uuid4())

        # 创建数据记录 - 直接存储PIL Image对象
        record = {
            'id': unique_id,
            'original_img': original_img,  # 直接存储PIL Image对象
            'instruction': instruction,
            'sol_img': solution_img,  # 直接存储PIL Image对象
            'maze_file': maze_file,
            'solution_file': os.path.basename(solution_path),
            'shape': maze_info['shape'],
            'algorithm': maze_info['algorithm'],
            'size': maze_info['size'],
            'difficulty': maze_info['difficulty'],
            'difficulty_score': maze_info['score'],
            'original_img_size': original_img.size,  # (width, height)
            'sol_img_size': solution_img.size  # (width, height)
        }

        dataset_records.append(record)
        successful_pairs += 1

    if not dataset_records:
        print("错误: 没有成功处理任何图像对")
        return False

    print(f"\n数据集统计:")
    print(f"  ✓ 成功处理: {successful_pairs} 对")
    print(f"  ❌ 处理失败: {failed_pairs} 对")
    print(f"  📊 总计样本: {len(dataset_records)} 个")

    # 创建DataFrame
    print("创建DataFrame...")
    df = pd.DataFrame(dataset_records)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存为Parquet格式
    print(f"保存数据集到: {output_path}")
    try:
        df.to_parquet(output_path, index=False, compression='snappy')
        print(f"✓ 数据集已成功保存: {output_path}")

        # 输出数据集信息
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"  📁 文件大小: {file_size:.2f} MB")
        print(f"  📋 列数: {len(df.columns)}")
        print(f"  📄 行数: {len(df)}")

        return True

    except Exception as e:
        print(f"错误: 保存Parquet文件失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='迷宫图像转Parquet数据集转换器')
    parser.add_argument('--maze-dir', default='./generated_mazes',
                       help='迷宫图像目录路径 (默认: ./generated_mazes)')
    parser.add_argument('--solution-dir', default='./generated_solutions',
                       help='解答图像目录路径 (默认: ./generated_solutions)')
    parser.add_argument('--output', default='./maze_dataset.parquet',
                       help='输出Parquet文件路径 (默认: ./maze_dataset.parquet)')
    parser.add_argument('--instruction', default='Add the solution path for the maze',
                       help='指令文本 (默认: "Add the solution path for the maze")')

    args = parser.parse_args()

    print("迷宫图像转Parquet数据集转换器")
    print("=" * 50)
    print(f"迷宫目录: {args.maze_dir}")
    print(f"解答目录: {args.solution_dir}")
    print(f"输出文件: {args.output}")
    print(f"指令文本: {args.instruction}")
    print()

    # 转换为绝对路径
    maze_dir = os.path.abspath(args.maze_dir)
    solution_dir = os.path.abspath(args.solution_dir)
    output_path = os.path.abspath(args.output)

    # 执行转换
    success = create_parquet_dataset(maze_dir, solution_dir, output_path, args.instruction)

    if success:
        print("\n🎉 数据集创建完成!")
        sys.exit(0)
    else:
        print("\n❌ 数据集创建失败!")
        sys.exit(1)

if __name__ == '__main__':
    main()