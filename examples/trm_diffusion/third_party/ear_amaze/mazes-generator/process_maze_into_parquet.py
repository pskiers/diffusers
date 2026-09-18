#!/usr/bin/env python3
"""
迷宫图像转Parquet数据集转换器
将generated_mazes和generated_solutions目录中的图像转换为Parquet格式数据集
NEW
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
import random
import json
import hashlib

def serialize_pil_image(image):
    """
    将PIL Image对象序列化为可存储的格式
    """
    try:
        from io import BytesIO
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        return image_b64
    except Exception as e:
        print(f"错误: 无法序列化PIL图像: {e}")
        return None

def deserialize_pil_image(image_b64):
    """
    从base64字符串反序列化PIL Image对象
    """
    try:
        from io import BytesIO
        image_bytes = base64.b64decode(image_b64)
        buffer = BytesIO(image_bytes)
        image = Image.open(buffer)
        return image
    except Exception as e:
        print(f"错误: 无法反序列化PIL图像: {e}")
        return None

def load_image_as_pil(image_path):
    """
    加载图像并返回PIL Image对象
    """
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        return img
    except Exception as e:
        print(f"错误: 无法加载图像 {image_path}: {e}")
        return None

def find_matching_solution(maze_filename, solution_dir):
    """
    根据迷宫文件名找到对应的解答文件
    """
    base_name = os.path.splitext(maze_filename)[0]
    possible_names = [
        f"{base_name}_solution.png",
        f"{base_name}_sol.png",
    ]
    for possible_name in possible_names:
        solution_path = os.path.join(solution_dir, possible_name)
        if os.path.exists(solution_path):
            return solution_path

    try:
        pattern = r'([^_]+)_([^_]+)_([^_]+)_([^_]+)_(\d+)_(.+)'
        match = re.match(pattern, base_name)
        if match:
            shape, algorithm, size, difficulty, score, timestamp = match.groups()
            for solution_file in os.listdir(solution_dir):
                if (shape in solution_file and
                    algorithm in solution_file and
                    size in solution_file and
                    'solution' in solution_file):
                    return os.path.join(solution_dir, solution_file)
    except:
        pass

    return None

def find_matching_m_maze(maze_filename, solution_dir):
    """
    根据迷宫文件名找到对应的解答文件
    """
    base_name = os.path.splitext(maze_filename)[0]
    possible_names = [
        f"{base_name}_solution.png",
        f"{base_name}_sol.png",
    ]
    for possible_name in possible_names:
        solution_path = os.path.join(solution_dir, possible_name)
        if os.path.exists(solution_path):
            return solution_path

    try:
        pattern = r'([^_]+)_([^_]+)_([^_]+)_([^_]+)_(\d+)_(.+)'
        match = re.match(pattern, base_name)
        if match:
            shape, algorithm, size, difficulty, score, timestamp = match.groups()
            for solution_file in os.listdir(solution_dir):
                if (shape in solution_file and
                    algorithm in solution_file and
                    size in solution_file and
                    'solution' in solution_file):
                    return os.path.join(solution_dir, solution_file)
    except:
        pass

    return None

def find_matching_mask(maze_filename, mask_dir):
    """
    根据迷宫文件名找到对应的解空间mask文件（已废弃，保留用于向后兼容）
    """
    base_name = os.path.splitext(maze_filename)[0]
    possible_names = [
        f"{base_name}_mask.png",
    ]
    for possible_name in possible_names:
        mask_path = os.path.join(mask_dir, possible_name)
        if os.path.exists(mask_path):
            return mask_path

    try:
        pattern = r'([^_]+)_([^_]+)_([^_]+)_([^_]+)_(\d+)_(.+)'
        match = re.match(pattern, base_name)
        if match:
            shape, algorithm, size, difficulty, score, timestamp = match.groups()
            for mask_file in os.listdir(mask_dir):
                if (shape in mask_file and
                    algorithm in mask_file and
                    size in mask_file and
                    'mask' in mask_file):
                    return os.path.join(mask_dir, mask_file)
    except:
        pass

    return None

def find_matching_metadata_files(maze_filename, metadata_dir):
    """
    根据迷宫文件名找到对应的metadata文件（path_mask, cell_map, json）
    返回: (path_mask_path, cell_map_path, metadata_json_path)
    """
    base_name = os.path.splitext(maze_filename)[0]

    path_mask_path = os.path.join(metadata_dir, f"{base_name}_path_mask.png")
    cell_map_path = os.path.join(metadata_dir, f"{base_name}_cell_map.png")
    metadata_json_path = os.path.join(metadata_dir, f"{base_name}.json")

    # 检查文件是否存在
    results = (
        path_mask_path if os.path.exists(path_mask_path) else None,
        cell_map_path if os.path.exists(cell_map_path) else None,
        metadata_json_path if os.path.exists(metadata_json_path) else None
    )

    return results


def extract_maze_info(filename):
    """
    从文件名提取迷宫信息
    """
    base_name = os.path.splitext(filename)[0]
    info = {
        'shape': 'unknown',
        'algorithm': 'unknown',
        'size': 'unknown',
        'difficulty': 'unknown',
        'score': 0
    }
    try:
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

def assign_split_by_filename(filename, train_ratio, seed=42):
    """
    根据文件名哈希值决定是 train 还是 val，确保可重复性
    返回: 'train' 或 'val'
    """
    # 使用文件名和种子生成哈希值
    hash_input = f"{filename}_{seed}".encode('utf-8')
    hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
    # 归一化到 [0, 1)
    normalized = (hash_value % 10000) / 10000.0
    return 'train' if normalized < train_ratio else 'val'

def save_and_verify(df: pd.DataFrame, out_path: str, label: str):
    """保存为parquet并打印基本信息和前5行"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"保存{label}到: {out_path}")
    df.to_parquet(out_path, index=False, compression='snappy')
    print(f"✓ 已保存 {label}: {out_path}")
    file_size = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  📁 文件大小: {file_size:.2f} MB")
    print(f"  📋 列数: {len(df.columns)}")
    print(f"  📄 行数: {len(df)}")
    print(f"\n验证{label}内容：")
    df_loaded = pd.read_parquet(out_path)
    print(df_loaded.head())

def create_parquet_dataset(maze_dir, no_markers_dir, solution_dir, metadata_dir, output_path,
                           instruction="Add the solution path for the maze",
                           max_samples=None, train_ratio=0.8, seed=42):
    """
    创建Parquet数据集，支持选择样本数量并按比例切分训练/测试
    新增metadata支持：包括path_mask, cell_map, 和元数据JSON
    """
    if not os.path.exists(maze_dir):
        print(f"错误: 迷宫目录不存在: {maze_dir}")
        return False
    if not os.path.exists(no_markers_dir):
        print(f"错误: 无标记迷宫目录不存在: {no_markers_dir}")
        return False
    if not os.path.exists(solution_dir):
        print(f"错误: 解答目录不存在: {solution_dir}")
        return False
    if not os.path.exists(metadata_dir):
        print(f"错误: 元数据目录不存在: {metadata_dir}")
        return False

    # 使用无标记迷宫目录作为主要数据源
    maze_files = [f for f in os.listdir(no_markers_dir) if f.lower().endswith('.png')]
    if not maze_files:
        print(f"错误: 在 {no_markers_dir} 中没有找到PNG文件")
        return False

    # 按文件名排序，确保处理顺序一致
    maze_files.sort()

    print(f"找到 {len(maze_files)} 个迷宫文件")
    if max_samples is not None and max_samples > 0:
        print(f"计划最多转换成功样本数: {max_samples}")

    train_records = []
    val_records = []
    successful_pairs = 0
    failed_pairs = 0

    for maze_file in tqdm(maze_files, desc="处理迷宫文件"):
        if max_samples is not None and max_samples > 0 and successful_pairs >= max_samples:
            break

        # 使用无标记迷宫作为 original_img
        maze_path = os.path.join(no_markers_dir, maze_file)
        solution_path = find_matching_solution(maze_file, solution_dir)
        # 从带标记迷宫目录加载带标记图像作为 m_original_img
        m_maze_path = os.path.join(maze_dir, maze_file)
        # 从metadata目录加载path_mask、cell_map和元数据JSON
        path_mask_path, cell_map_path, metadata_json_path = find_matching_metadata_files(maze_file, metadata_dir)

        if solution_path is None:
            print(f"警告: 找不到 {maze_file} 对应的解答文件")
            failed_pairs += 1
            continue

        if path_mask_path is None or cell_map_path is None or metadata_json_path is None:
            print(f"警告: 找不到 {maze_file} 对应的metadata文件")
            failed_pairs += 1
            continue

        # 加载无标记迷宫（作为original_img）
        original_img = serialize_pil_image(load_image_as_pil(maze_path))
        # 加载带标记迷宫（作为m_original_img）
        m_original_img = None
        if os.path.exists(m_maze_path):
            m_original_img = serialize_pil_image(load_image_as_pil(m_maze_path))
        # 加载解答图像
        solution_img = serialize_pil_image(load_image_as_pil(solution_path))
        # 加载path mask图像
        path_mask_img = serialize_pil_image(load_image_as_pil(path_mask_path))
        # 加载cell map图像
        cell_map_img = serialize_pil_image(load_image_as_pil(cell_map_path))
        # 加载元数据JSON
        metadata_json = None
        try:
            with open(metadata_json_path, 'r') as f:
                metadata_json = json.load(f)
        except Exception as e:
            print(f"警告: 无法加载元数据JSON {metadata_json_path}: {e}")
            failed_pairs += 1
            continue

        if original_img is None or solution_img is None or path_mask_img is None or cell_map_img is None:
            print(f"警告: 无法加载图像对: {maze_file}")
            failed_pairs += 1
            continue

        record = {
            'id': str(uuid.uuid4()),
            'original_img': original_img,      # 无标记迷宫
            'm_original_img': m_original_img,  # 带标记迷宫
            'instruction': instruction,
            'sol_img': solution_img,           # 解答图像
            'mask_img': path_mask_img,         # GT路径mask（保持字段名为mask_img）
            'cell_map': cell_map_img,          # 格子分割图
            'metadata': json.dumps(metadata_json),  # 元数据JSON（序列化为字符串）
        }
        
        # 根据文件名分配到 train 或 val
        split = assign_split_by_filename(maze_file, train_ratio, seed)
        if split == 'train':
            train_records.append(record)
        else:
            val_records.append(record)
        successful_pairs += 1

    n_total = len(train_records) + len(val_records)
    if n_total == 0:
        print("错误: 没有成功处理任何图像对")
        return False

    print(f"\n数据集统计:")
    print(f"  ✓ 成功处理: {successful_pairs} 对")
    print(f"  ❌ 处理失败: {failed_pairs} 对")
    print(f"  📊 总计样本: {n_total} 个")
    print(f"\n根据文件名分配: 训练集 {len(train_records)} / 验证集 {len(val_records)} （训练占比: {train_ratio:.2f}）")

    df_train = pd.DataFrame(train_records)
    df_val = pd.DataFrame(val_records)

    # 根据 --output 基名生成两个文件
    out_dir = os.path.dirname(output_path)
    base = os.path.basename(output_path)
    stem, _ext = os.path.splitext(base)
    train_path = os.path.join(out_dir, f"{stem}_train.parquet")
    val_path = os.path.join(out_dir, f"{stem}_val.parquet")

    try:
        if len(train_records) > 0:
            save_and_verify(df_train, train_path, "训练集")
        if len(val_records) > 0:
            save_and_verify(df_val, val_path, "验证集")
        return True
    except Exception as e:
        print(f"错误: 保存Parquet文件失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='迷宫图像转Parquet数据集转换器')
    parser.add_argument('--maze-dir', default='./generated_mazes',
                        help='迷宫图像目录路径 (默认: ./generated_mazes)')
    parser.add_argument('--no-markers-dir', default='./generated_mazes_no_markers',
                        help='无标记迷宫图像目录路径 (默认: ./generated_mazes_no_markers)')
    parser.add_argument('--solution-dir', default='./generated_solutions',
                        help='解答图像目录路径 (默认: ./generated_solutions)')
    parser.add_argument('--metadata-dir', default='./generated_metadata',
                        help='元数据目录路径，包含path_mask、cell_map和JSON (默认: ./generated_metadata)')
    parser.add_argument('--output', default='./maze-dataset/maze_dataset.parquet',
                        help='输出Parquet基名 (将生成 *_train.parquet 和 *_val.parquet)')
    parser.add_argument('--instruction',
                        default=("Add the blue solution path for the maze, connect start point (solid red circle) "
                                 "to end point (red 'X' mark). Ensure all original maze elements "
                                 "(walls, points, etc.) remain unchanged—only add the path."),
                        help='指令文本')

    # 新增参数
    parser.add_argument('-n', '--max-samples', type=int, default=None,
                        help='最多转换的成功样本数（成功匹配且可加载的图像对）。默认不限制')
    parser.add_argument('--train-ratio', type=float, default=1.0,
                        help='训练集比例，默认0.8')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子，默认42')

    args = parser.parse_args()

    print("迷宫图像转Parquet数据集转换器")
    print("=" * 50)
    print(f"带标记迷宫目录: {args.maze_dir}")
    print(f"无标记迷宫目录: {args.no_markers_dir}")
    print(f"解答目录: {args.solution_dir}")
    print(f"元数据目录: {args.metadata_dir}")
    print(f"输出基名: {args.output}  →  *_train.parquet 和 *_val.parquet")
    print(f"指令文本: {args.instruction}")
    if args.max_samples:
        print(f"最大样本数: {args.max_samples}")
    print(f"训练集比例: {args.train_ratio} (根据文件名哈希分配)")
    print(f"随机种子: {args.seed} (用于文件名哈希)\n")

    maze_dir = os.path.abspath(args.maze_dir)
    no_markers_dir = os.path.abspath(args.no_markers_dir)
    solution_dir = os.path.abspath(args.solution_dir)
    metadata_dir = os.path.abspath(args.metadata_dir)
    output_path = os.path.abspath(args.output)

    success = create_parquet_dataset(
        maze_dir, no_markers_dir, solution_dir, metadata_dir, output_path,
        instruction=args.instruction,
        max_samples=args.max_samples,
        train_ratio=args.train_ratio,
        seed=args.seed
    )

    if success:
        print("\n🎉 数据集创建完成!")
        sys.exit(0)
    else:
        print("\n❌ 数据集创建失败!")
        sys.exit(1)

if __name__ == '__main__':
    main()