#!/usr/bin/env python3
"""
Cell Map 可视化工具
生成完整对比图：迷宫 + cell map + path mask + 叠加显示
支持批量处理整个文件夹
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import argparse
import os
import glob
from pathlib import Path

def decode_cell_map(cell_map_path):
    """解码cell map PNG为ID数组"""
    # OpenCV默认BGR格式
    cell_map_rgb = cv2.imread(cell_map_path)
    if cell_map_rgb is None:
        raise FileNotFoundError(f"无法加载: {cell_map_path}")

    # 解码为ID
    cell_map = (cell_map_rgb[:,:,2].astype(np.uint32) |
                (cell_map_rgb[:,:,1].astype(np.uint32) << 8) |
                (cell_map_rgb[:,:,0].astype(np.uint32) << 16))

    return cell_map

def visualize_cell_map(cell_map, colormap='tab20b'):
    """
    将cell map可视化为彩色图像
    """
    unique_ids = np.unique(cell_map)
    num_cells = len(unique_ids)

    # 归一化ID到[0, 1]用于colormap
    normalized = np.zeros_like(cell_map, dtype=np.float32)
    for i, cell_id in enumerate(unique_ids):
        mask = cell_map == cell_id
        normalized[mask] = i / max(1, num_cells - 1)

    # 应用colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(normalized)

    # 转换为8位RGB图像
    rgb_image = (colored[:, :, :3] * 255).astype(np.uint8)

    return rgb_image, num_cells

def create_visualization(maze_path, cell_map_path, path_mask_path, metadata_path,
                        output_path, colormap='tab20b', verbose=True):
    """
    生成完整对比图：迷宫 + cell map + path mask + 路径格子叠加
    """
    try:
        # 读取所有数据
        maze_img = cv2.imread(maze_path)
        if maze_img is None:
            if verbose:
                print(f"  ⚠️  无法加载迷宫图像: {maze_path}")
            return False

        cell_map = decode_cell_map(cell_map_path)
        path_mask = cv2.imread(path_mask_path, cv2.IMREAD_GRAYSCALE)

        with open(metadata_path) as f:
            metadata = json.load(f)

        # 可视化cell map
        cell_map_vis, num_cells = visualize_cell_map(cell_map, colormap)

        # 在cell map上叠加路径（标记路径经过的格子）
        cell_map_with_path = cell_map_vis.copy()
        path_cell_ids = set(metadata["path_cell_ids"])
        # 将路径经过的所有格子标记为红色
        # 注意：ID=0保留给背景，所有有效格子ID从1开始，因此不会误标记背景
        for cell_id in path_cell_ids:
            if cell_id != 0:  # 双重保护：虽然有效格子ID已从1开始，仍过滤ID=0以防万一
                cell_map_with_path[cell_map == cell_id] = [255, 0, 0]  # 整个格子变红色

        # 创建2x2网格显示
        fig, axes = plt.subplots(2, 3, figsize=(20, 20))

        # 1. 原始迷宫
        axes[0, 0].imshow(cv2.cvtColor(maze_img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('迷宫图像')
        axes[0, 0].axis('off')

        # 2. Cell Map可视化
        axes[0, 1].imshow(cell_map_vis)
        axes[0, 1].set_title(f'Cell Map ({num_cells}个格子)')
        axes[0, 1].axis('off')

        # 3. Path Mask
        axes[1, 0].imshow(path_mask, cmap='gray')
        axes[1, 0].set_title(f'GT路径Mask ({len(metadata["path_cell_ids"])}个格子)')
        axes[1, 0].axis('off')

        # 4. Cell Map + Path叠加
        axes[1, 1].imshow(cell_map_with_path)
        axes[1, 1].set_title('Cell Map + path（red）with num')
        axes[1, 1].axis('off')
        
        # 在每个格子上标注ID
        unique_ids = np.unique(cell_map)
        for cell_id in unique_ids:
            if cell_id == 0:  # 跳过背景
                continue
            # 找到该格子的所有像素位置
            mask = cell_map == cell_id
            if not np.any(mask):
                continue
            # 计算格子的质心（中心点）
            y_coords, x_coords = np.where(mask)
            center_y = np.mean(y_coords)
            center_x = np.mean(x_coords)
            # 绘制ID文本，使用白色文字配黑色边框以确保可见性
            axes[1, 1].text(center_x, center_y, str(cell_id), 
                           ha='center', va='center',
                           fontsize=8, fontweight='bold',
                           color='white',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='black', 
                                   alpha=0.6, 
                                   edgecolor='white', 
                                   linewidth=0.5))

        # 4. Cell Map + Path叠加
        axes[1, 2].imshow(cell_map_with_path)
        axes[1, 2].set_title('Cell Map + path（red）')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        if verbose:
            print(f"  ✓ 可视化已保存: {output_path}")
        plt.close()
        return True

    except Exception as e:
        if verbose:
            print(f"  ❌ 处理失败: {e}")
        return False

def process_folder(metadata_dir, maze_dir, output_dir, colormap='tab20b'):
    """批量处理文件夹下所有cell map并生成完整对比图"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 查找所有cell map文件
    pattern = os.path.join(metadata_dir, '*_cell_map.png')
    cell_map_files = glob.glob(pattern)

    if not cell_map_files:
        print(f"未找到cell map文件: {pattern}")
        return

    print(f"找到 {len(cell_map_files)} 个cell map文件")
    print(f"输出目录: {output_dir}")

    success_count = 0
    for cell_map_path in sorted(cell_map_files):
        # 提取base name
        filename = os.path.basename(cell_map_path)
        base_name = filename.replace('_cell_map.png', '')

        print(f"\n处理: {base_name}")

        # 构建文件路径
        maze_path = os.path.join(maze_dir, f'{base_name}.png')
        path_mask_path = os.path.join(metadata_dir, f'{base_name}_path_mask.png')
        metadata_path = os.path.join(metadata_dir, f'{base_name}.json')

        # 检查所有文件是否存在
        if not all(os.path.exists(p) for p in [maze_path, path_mask_path, metadata_path]):
            print(f"  ⚠️  跳过: 缺少必要文件")
            continue

        # 输出路径
        output_path = os.path.join(output_dir, f'{base_name}_visualization.png')

        if create_visualization(maze_path, cell_map_path, path_mask_path,
                               metadata_path, output_path, colormap):
            success_count += 1

    print(f"\n✓ 完成: 成功处理 {success_count}/{len(cell_map_files)} 个文件")

def main():
    parser = argparse.ArgumentParser(description='Cell Map完整对比图生成工具')
    parser.add_argument('--metadata-dir', default='generated_metadata',
                        help='元数据文件夹路径 (默认: generated_metadata)')
    parser.add_argument('--maze-dir', default='generated_mazes',
                        help='迷宫图像文件夹路径 (默认: generated_mazes)')
    parser.add_argument('--output-dir', default='visualizations',
                        help='可视化输出目录 (默认: visualizations)')
    parser.add_argument('--colormap', default='tab20b',
                        help='Matplotlib colormap (默认: tab20b, 可选: hsv, rainbow等)')

    args = parser.parse_args()

    print("=" * 60)
    print("Cell Map 完整对比图生成工具")
    print("=" * 60)

    process_folder(args.metadata_dir, args.maze_dir, args.output_dir, args.colormap)

if __name__ == '__main__':
    main()
