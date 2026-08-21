"""
python api_infer/inference_api.py \
    --api_key your_api_key \
    --dataset_path /media/raid/workspace/zhaoyanpeng/model/maze_dataset/hexagon/maze-dataset \
    --output_dir api_results/gpt-image-1/hexagon/3_16 \
    --model gpt-image-1 \
    --base_url "your_api_url" \
    --api_provider openai \
    --split test \
    --num_attempts 5 \
    --num_threads 16 \
    --filter_size_min 3 \
    --filter_size_max 16 \
    --samples_per_size 5 \
    --resume_dir api_results/gpt-image-1/hexagon/3_16 \
    --resolution 1024 \
    --image_size 1024x1024
"""



import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import base64
import json
import time
import threading
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Optional, List, Dict, Any

import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

IS_CIRCLE_DATASET = False
if IS_CIRCLE_DATASET:
    from flow_grpo.maze_rewards_circle import maze_reward
    import torch
else:
    from flow_grpo.maze_rewards import maze_reward
    import torch

try:
    from openai import OpenAI
except ImportError:
    print("请安装 openai 库: pip install openai")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("请安装 requests 库: pip install requests")
    sys.exit(1)

# 导入 Google Gemini API
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("警告: 无法导入 google.genai 库，Gemini API 将不可用")
    print("   安装命令: pip install google-genai")
    GEMINI_AVAILABLE = False

# 导入火山引擎 API
try:
    from volcenginesdkarkruntime import Ark
    VOLCANO_AVAILABLE = True
except ImportError:
    print("警告: 无法导入 volcenginesdkarkruntime 库，火山引擎 API 将不可用")
    print("   安装命令: pip install 'volcengine-python-sdk[ark]'")
    VOLCANO_AVAILABLE = False

from dataset.maze_dataset import MazeDataset

# # 导入奖励函数
# from concurrent import futures  # 确保 futures 模块总是可用
# if 'circle/maze-dataset' in dataset_path:
#     from flow_grpo.maze_rewards import maze_reward_circle as maze_reward
#     import torch


# try:
#     from flow_grpo.maze_rewards import maze_reward
#     import torch
# except ImportError as e:
#     print(f"警告: 无法导入奖励函数模块: {e}")
#     maze_reward = None


def decode_base64_image(base64_str):
    """Decode base64 encoded image string to PIL Image."""
    if isinstance(base64_str, str):
        img_bytes = base64.b64decode(base64_str)
    else:
        img_bytes = base64_str
    return Image.open(BytesIO(img_bytes)).convert('RGB')


def encode_image_to_base64(image: Image.Image, format='PNG') -> str:
    """Encode PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


def image_to_base64_file(image_path: str) -> Optional[str]:
    """Read image file and convert to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"错误：无法读取图像文件 {image_path}: {e}")
        return None


class MazePromptImageDataset(Dataset):
    def __init__(self, dataset_path, split='train', filter_size_min=None, filter_size_max=None, 
                 samples_per_size=None, resolution=None):
        """
        Args:
            dataset_path: Path to maze dataset
            split: 'train' or 'test'
            filter_size_min: Optional, minimum maze size (e.g., 5 for 5x5) or layers for circle
            filter_size_max: Optional, maximum maze size (e.g., 10 for 10x10) or layers for circle
            samples_per_size: Optional, number of samples to select per size
            resolution: Optional, target image resolution
        """
        self.dataset_path = dataset_path
        self.maze_dataset = MazeDataset(dataset_path, split=split)
        self.filter_size_min = filter_size_min
        self.filter_size_max = filter_size_max
        self.samples_per_size = samples_per_size
        self.resolution = resolution
        
        # 检测是否是圆形数据集
        self.is_circle_dataset = dataset_path and 'circle/maze-dataset' in dataset_path
        
        # Build filtered indices if filter is specified
        if filter_size_min is not None or filter_size_max is not None or samples_per_size is not None:
            if self.is_circle_dataset:
                if filter_size_min is not None and filter_size_max is not None:
                    print(f"🔍 按layers范围过滤数据集（圆形）: [{filter_size_min}, {filter_size_max}]")
                elif filter_size_min is not None:
                    print(f"🔍 按最小layers过滤数据集（圆形）: >= {filter_size_min}")
                elif filter_size_max is not None:
                    print(f"🔍 按最大layers过滤数据集（圆形）: <= {filter_size_max}")
            else:
                if filter_size_min is not None and filter_size_max is not None:
                    print(f"🔍 按尺寸范围过滤数据集: [{filter_size_min}, {filter_size_max}]")
                elif filter_size_min is not None:
                    print(f"🔍 按最小尺寸过滤数据集: >= {filter_size_min}")
                elif filter_size_max is not None:
                    print(f"🔍 按最大尺寸过滤数据集: <= {filter_size_max}")
            if samples_per_size is not None:
                print(f"🔍 每个尺寸选择 {samples_per_size} 个样本")
            self.filtered_indices = self._build_filtered_indices()
            print(f"   找到 {len(self.filtered_indices)} 个符合过滤条件的样本")
        else:
            self.filtered_indices = list(range(len(self.maze_dataset)))
    
    def _build_filtered_indices(self):
        """Build list of indices that match the size filter criteria."""
        samples_by_size = {}
        
        for idx in range(len(self.maze_dataset)):
            maze_item = self.maze_dataset[idx]
            maze_config = self._extract_maze_config(maze_item)
            
            if self.is_circle_dataset:
                # 圆形数据集按layers分组
                layers = maze_config.get('layers', 0)
                
                # Check if layers is within the specified range
                size_ok = True
                if self.filter_size_min is not None:
                    size_ok = size_ok and (layers >= self.filter_size_min)
                
                if self.filter_size_max is not None:
                    size_ok = size_ok and (layers <= self.filter_size_max)
                
                if size_ok:
                    size_key = f"layers_{layers}"
                    if size_key not in samples_by_size:
                        samples_by_size[size_key] = []
                    samples_by_size[size_key].append(idx)
            else:
                # 其他数据集按width/height分组
                width = maze_config.get('width', 0)
                height = maze_config.get('height', 0)
                
                # Only include hexagon mazes (width == height)
                if width != height:
                    continue
                
                # Check if size is within the specified range
                size_ok = True
                if self.filter_size_min is not None:
                    size_ok = size_ok and (width >= self.filter_size_min)
                
                if self.filter_size_max is not None:
                    size_ok = size_ok and (width <= self.filter_size_max)
                
                if size_ok:
                    size_key = f"{width}x{height}"
                    if size_key not in samples_by_size:
                        samples_by_size[size_key] = []
                    samples_by_size[size_key].append(idx)
        
        # Second pass: select samples (all or per-size limit)
        filtered = []
        if self.samples_per_size is not None:
            for size_key in sorted(samples_by_size.keys()):
                available = samples_by_size[size_key]
                selected = available[:self.samples_per_size]
                filtered.extend(selected)
                print(f"   {size_key}: 选择了 {len(selected)}/{len(available)} 个样本")
        else:
            for size_key in sorted(samples_by_size.keys()):
                filtered.extend(samples_by_size[size_key])
        # input()
        
        return filtered
    
    def _extract_maze_config(self, maze_item):
        """Extract maze_config from maze_item metadata."""
        maze_config = {}
        metadata = maze_item.get("metadata", {})
        
        # Try to parse nested metadata JSON string
        if 'metadata' in metadata:
            try:
                if isinstance(metadata['metadata'], str):
                    parsed_meta = json.loads(metadata['metadata'])
                    maze_config = parsed_meta.get('maze_config', {})
                elif isinstance(metadata['metadata'], dict):
                    maze_config = metadata['metadata'].get('maze_config', {})
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Fallback: try direct maze_config
        if not maze_config:
            maze_config = metadata.get('maze_config', {})
        
        return maze_config

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        # Map filtered index to original dataset index
        original_idx = self.filtered_indices[idx]
        maze_item = self.maze_dataset[original_idx]

        # Try to use actual images from the maze dataset if available
        image = None
        try:
            if 'm_original_img' in maze_item["metadata"]:
                # Decode the base64 image
                image = decode_base64_image(maze_item["metadata"]['m_original_img'])
                if self.resolution:
                    image = image.resize((self.resolution, self.resolution))
        except Exception as e:
            print(f"警告: 无法解码样本 {idx} 的图像, 错误: {e}")
            # Skip this sample by returning next one
            if idx + 1 < len(self):
                return self.__getitem__((idx + 1) % len(self))
            else:
                raise

        # Create a unique identifier
        if 'id' in maze_item["metadata"]:
            prompt_with_image_path = f"{maze_item['metadata']['id']}"
        else:
            prompt_with_image_path = f"{maze_item['prompt']}_maze_{idx}"

        # Prepare metadata
        processed_metadata = maze_item["metadata"].copy()

        # Decode all image fields from base64 to PIL Image
        image_fields = ['original_img', 'm_original_img', 'sol_img', 'mask_img', 'cell_map']
        for field in image_fields:
            if field in processed_metadata and isinstance(processed_metadata[field], str):
                try:
                    pil_image = decode_base64_image(processed_metadata[field])
                    if self.resolution:
                        if field == 'cell_map':
                            pass
                        else:
                            pil_image = pil_image.resize((self.resolution, self.resolution))
                    processed_metadata[field] = pil_image
                except Exception as e:
                    print(f"警告: 无法解码 {field} 对于样本 {idx}: {e}")

        item = {
            "id": maze_item["metadata"].get("id", f"{idx}"),
            "prompt": maze_item["prompt"],
            "metadata": processed_metadata,
            "image": image,
            "prompt_with_image_path": prompt_with_image_path
        }
        return item

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        images = [example["image"] for example in examples]
        ids = [example["id"] for example in examples]
        return prompts, metadatas, images, ids


def call_openai_image_edit(
    client: OpenAI,
    image: Image.Image,
    prompt: str,
    mask: Optional[Image.Image] = None,
    model: str = "dall-e-2",
    n: int = 1,
    size: str = "1024x1024",
    max_retries: int = 3
) -> Optional[List[Dict[str, str]]]:
    """
    调用 OpenAI 图像编辑 API
    
    Args:
        client: OpenAI 客户端
        image: 输入图像 (PIL Image)
        prompt: 编辑提示
        mask: 可选的遮罩图像 (PIL Image)
        model: 模型名称 ("dall-e-2", "dall-e-3", "gpt-image-1")
        n: 生成图像数量
        size: 图像尺寸
        max_retries: 最大重试次数
    
    Returns:
        生成的图像信息列表，每个元素是 {"type": "url"|"base64", "data": "..."}，失败返回 None
    """
    for attempt in range(max_retries):
        try:
            # 将 PIL Image 转换为文件对象
            image_buffer = BytesIO()
            image.save(image_buffer, format='PNG')
            image_buffer.seek(0)
            file_tuple = ("image.png", image_buffer, "image/png")
            
            # 调用 API
            if model == "gpt-image-1" or model == "gpt-image-1.5":
                # gpt-image-1 使用不同的 API 格式
                # image 参数是一个列表，可以包含多个参考图像
                # 返回 base64 编码的图像
                edit_params = {
                    "model": model,
                    "image": file_tuple,  # 列表格式
                    "prompt": prompt
                }
                # 如果有遮罩，也可以作为参考图像添加
                # if mask is not None:
                #     mask_buffer = BytesIO()
                #     mask.save(mask_buffer, format='PNG')
                #     mask_buffer.seek(0)
                #     edit_params["image"].append(mask_buffer)
                
                response = client.images.edit(**edit_params)
                
                # 提取 base64 编码的图像
                results = []
                # print(response.data)
                for item in response.data:
                    if hasattr(item, 'b64_json') and item.b64_json:
                        results.append({"type": "base64", "data": item.b64_json})
                return results if results else None
                
            elif model == "dall-e-2":
                # DALL-E 2 使用传统格式
                edit_params = {
                    "image": image_buffer,
                    "prompt": prompt,
                    "n": n,
                    "size": size
                }
                
                # 如果有遮罩，添加遮罩
                if mask is not None:
                    mask_buffer = BytesIO()
                    mask.save(mask_buffer, format='PNG')
                    mask_buffer.seek(0)
                    edit_params["mask"] = mask_buffer
                
                response = client.images.edit(**edit_params)
                print(response.data)
                
                # 提取图像 URL
                results = []
                for item in response.data:
                    if hasattr(item, 'url') and item.url:
                        results.append({"type": "url", "data": item.url})
                return results if results else None
            else:
                # DALL-E 3 或其他模型使用生成 API
                print(f"警告: {model} 可能不支持图像编辑 API，使用生成 API")
                response = client.images.generate(
                    model=model,
                    prompt=prompt,
                    n=n,
                    size=size
                )
                
                # 提取图像 URL
                results = []
                for item in response.data:
                    if hasattr(item, 'url') and item.url:
                        results.append({"type": "url", "data": item.url})
                return results if results else None
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(10)
            else:
                print(f"API 调用失败，已达到最大重试次数: {e}")
                import traceback
                traceback.print_exc()
                return None
    
    return None


def download_image_from_url(url: str, output_path: str, max_retries: int = 3) -> bool:
    """从 URL 下载图像并保存到本地"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"下载图像失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                time.sleep(wait_time)
            else:
                print(f"下载图像失败，已达到最大重试次数: {e}")
                return False
    
    return False


def call_gemini_image_edit(
    client: Any,
    image: Image.Image,
    prompt: str,
    mask: Optional[Image.Image] = None,
    model: str = "gemini-2.5-flash-image",
    max_retries: int = 3
) -> Optional[List[Dict[str, Any]]]:
    """
    调用 Google Gemini 图像编辑 API
    
    Args:
        client: Google Gemini 客户端 (genai.Client)
        image: 输入图像 (PIL Image)
        prompt: 编辑提示
        mask: 可选的遮罩图像 (PIL Image) - Gemini 可能不支持，但保留接口兼容性
        model: 模型名称 (默认: "gemini-2.5-flash-image")
        max_retries: 最大重试次数
    
    Returns:
        生成的图像信息列表，每个元素是 {"type": "pil_image", "data": PIL.Image}，失败返回 None
    """
    if not GEMINI_AVAILABLE:
        print("错误: Gemini API 不可用，请安装 google-genai 库")
        return None
    
    for attempt in range(max_retries):
        try:
            # 准备输入内容：prompt 和图像
            contents = [prompt, image]
            
            # 如果有遮罩图像，也可以作为输入添加（如果 API 支持）
            if mask is not None:
                # 注意：Gemini API 可能不支持 mask，这里先不添加
                # 如果需要，可以尝试将 mask 作为额外的图像输入
                pass
            
            # 调用 Gemini API
            response = client.models.generate_content(
                model=model,
                contents=contents,
            )
            
            # 提取生成的图像
            results = []
            # print(response)
            # input()
            for part in response.parts:
                if part.inline_data is not None:
                    # 获取生成的图像 - 转换为 PIL Image
                    gemini_img = part.as_image()
                    # 将 google.genai.types.Image 转换为 PIL Image
                    if hasattr(gemini_img, 'to_pil'):
                        generated_img = gemini_img.to_pil()
                    elif hasattr(part.inline_data, 'data'):
                        # 从 inline_data 获取字节数据并转换为 PIL Image
                        image_bytes = part.inline_data.data
                        generated_img = Image.open(BytesIO(image_bytes)).convert('RGB')
                    else:
                        # 尝试直接转换
                        generated_img = Image.fromarray(gemini_img) if hasattr(gemini_img, '__array__') else Image.open(BytesIO(bytes(gemini_img))).convert('RGB')
                    results.append({"type": "pil_image", "data": generated_img})
                elif part.text is not None:
                    # 如果有文本输出，可以记录（可选）
                    print(f"Gemini 文本输出: {part.text}")
            
            return results if results else None
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"Gemini API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"Gemini API 调用失败，已达到最大重试次数: {e}")
                import traceback
                traceback.print_exc()
                return None
    
    return None


def call_volcano_image_edit(
    client: Any,
    image: Image.Image,
    prompt: str,
    mask: Optional[Image.Image] = None,
    model: str = "doubao-seedream-4-5-251128",
    size: str = "2K",
    response_format: str = "url",
    watermark: bool = False,
    max_retries: int = 3
) -> Optional[List[Dict[str, Any]]]:
    """
    调用火山引擎图像编辑 API
    
    Args:
        client: 火山引擎客户端 (Ark)
        image: 输入图像 (PIL Image)
        prompt: 编辑提示
        mask: 可选的遮罩图像 (PIL Image) - 火山引擎可能不支持，但保留接口兼容性
        model: 模型名称 (默认: "doubao-seedream-4-5-251128")
        size: 图像尺寸 (默认: "2K", 可选: "1K", "2K", "4K")
        response_format: 返回格式 (默认: "url", 可选: "url", "b64_json")
        watermark: 是否添加水印 (默认: False)
        max_retries: 最大重试次数
    
    Returns:
        生成的图像信息列表，每个元素是 {"type": "url"|"base64", "data": "..."}，失败返回 None
    """
    if not VOLCANO_AVAILABLE:
        print("错误: 火山引擎 API 不可用，请安装 volcenginesdkarkruntime 库")
        return None
    
    for attempt in range(max_retries):
        try:
            # 将 PIL Image 转换为 base64 编码的字符串
            # 火山引擎 API 需要 base64 编码的图像或 URL
            image_base64 = encode_image_to_base64(image, format='PNG')
            # 构造 data URI 格式: data:image/png;base64,{base64_string}
            image_data_uri = f"data:image/png;base64,{image_base64}"
            
            # 调用火山引擎 API
            response = client.images.generate(
                model=model,
                prompt=prompt,
                image=image_data_uri,  # 使用 data URI 格式
                size=size,
                response_format=response_format,
                watermark=watermark
            )
            print(response)
            
            # 提取生成的图像
            results = []
            if hasattr(response, 'data') and response.data:
                for item in response.data:
                    if hasattr(item, 'url') and item.url:
                        # URL 格式
                        results.append({"type": "url", "data": item.url})
                    elif hasattr(item, 'b64_json') and item.b64_json:
                        # Base64 格式
                        results.append({"type": "base64", "data": item.b64_json})
                    elif hasattr(item, 'revised_prompt') and hasattr(item, 'url'):
                        # 某些响应可能包含 revised_prompt
                        if item.url:
                            results.append({"type": "url", "data": item.url})
            
            return results if results else None
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"火山引擎 API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"火山引擎 API 调用失败，已达到最大重试次数: {e}")
                import traceback
                traceback.print_exc()
                return None
    
    return None


def process_single_sample(
    client: Any,
    sample: Dict[str, Any],
    output_dir: str,
    model: str,
    size: str,
    num_attempts: int,
    lock: threading.Lock,
    results: List[Dict],
    eval_reward_fn: Optional[Any] = None,
    executor: Optional[Any] = None,
    device: str = "cpu",
    api_provider: str = "openai",
    pbar: Optional[tqdm] = None,
    completed_attempts: Optional[set] = None,
    json_path: Optional[str] = None,
    is_circle_dataset: bool = False
):
    """处理单个样本"""
    sample_id = sample["id"]
    prompt = sample["prompt"]
    image = sample["image"]
    metadata = sample["metadata"]
    
    # 提取 maze_config 信息
    maze_config = {}
    if 'metadata' in metadata:
        try:
            if isinstance(metadata['metadata'], str):
                parsed_meta = json.loads(metadata['metadata'])
                maze_config = parsed_meta.get('maze_config', {})
            elif isinstance(metadata['metadata'], dict):
                maze_config = metadata['metadata'].get('maze_config', {})
        except (json.JSONDecodeError, TypeError):
            pass
    
    if not maze_config:
        maze_config = metadata.get('maze_config', {})
    
    # 圆形数据集按layers分组，其他数据集按width/height分组
    if is_circle_dataset:
        layers = maze_config.get('layers', 0)
        size_key = f"layers_{layers}"
        width = 10  # 圆形数据集固定为10
        height = 10
    else:
        width = maze_config.get('width', 0)
        height = maze_config.get('height', 0)
        size_key = f"{width}x{height}"
    
    # 检查图像是否存在
    if image is None:
        print(f"警告: 样本 {sample_id} 没有图像，跳过")
        # 为每个 attempt 创建记录（即使跳过）
        for attempt_idx in range(1, num_attempts + 1):
            result_record = {
                "id": sample_id,
                "attempt": attempt_idx,
                "width": width,
                "height": height,
                "rewards": {}
            }
            # 如果是圆形数据集，添加layers字段
            if is_circle_dataset:
                layers = maze_config.get('layers', 0)
                result_record["layers"] = layers
            with lock:
                results.append(result_record)
        if pbar:
            pbar.update(1)
        return
    
    # 获取遮罩图像（如果存在）
    mask = metadata.get('mask_img', None)
    if mask is None:
        mask = metadata.get('mask', None)
    
    sol_img = metadata.get('sol_img', None)
    # 保存输入图像（仅第一次）
    filename_base = f"{size_key}_{sample_id}"
    input_path = os.path.join(output_dir, f"{filename_base}_input.jpg")
    gt_path = os.path.join(output_dir, f"{filename_base}_gt.jpg")
    
    with lock:
        if not os.path.exists(input_path):
            image.save(input_path)
            sol_img.save(gt_path)
    
    # 处理多次尝试 - 每个 attempt 创建单独的记录（与 inference_sft_lora.py 格式一致）
    if completed_attempts is None:
        completed_attempts = set()
    
    for attempt_idx in range(1, num_attempts + 1):
        # 检查是否已完成（load_existing_results 已经验证了文件存在性）
        if (sample_id, attempt_idx) in completed_attempts:
            print(f"⏭️  跳过已完成的 attempt: 样本 {sample_id}, attempt {attempt_idx}")
            # 跳过时不创建新记录，已有结果会在合并时保留
            continue
        
        sample_reward = {}
        generated_image = None
        output_path = None
        
        try:
            # 根据 API 提供商调用不同的 API
            if api_provider == "gemini":
                # 调用 Gemini API
                image_results = call_gemini_image_edit(
                    client=client,
                    image=image,
                    prompt=prompt,
                    mask=mask,
                    model=model,
                )
            elif api_provider == "volcano":
                # 调用火山引擎 API
                # 将 size 参数转换为火山引擎格式 (如 "1024x1024" -> "2K")
                volcano_size = size
                if size == "1024x1024":
                    volcano_size = "2K"
                elif size == "512x512":
                    volcano_size = "1K"
                elif size == "256x256":
                    volcano_size = "1K"
                else:
                    volcano_size = "2K"  # 默认使用 2K
                
                image_results = call_volcano_image_edit(
                    client=client,
                    image=image,
                    prompt=prompt,
                    mask=mask,
                    model=model,
                    size=volcano_size,
                    response_format="url",
                    watermark=False
                )
            else:
                # 调用 OpenAI API (默认)
                image_results = call_openai_image_edit(
                    client=client,
                    image=image,
                    prompt=prompt,
                    mask=mask,
                    model=model,
                    n=1,
                    size=size
                )
            
            if image_results and len(image_results) > 0:
                # 处理返回结果（可能是 URL、base64 或 PIL Image）
                output_path = os.path.join(output_dir, f"{filename_base}_attempt{attempt_idx:03d}.jpg")
                first_result = image_results[0]
                
                if first_result["type"] == "pil_image":
                    # 处理 Gemini 返回的 PIL Image
                    try:
                        generated_image = first_result["data"]
                        generated_image.save(output_path)
                    except Exception as e:
                        print(f"警告: 无法保存 Gemini 生成的图像，样本 {sample_id} 尝试 {attempt_idx}: {e}")
                        generated_image = None
                elif first_result["type"] == "base64":
                    # 处理 base64 编码的图像
                    try:
                        image_bytes = base64.b64decode(first_result["data"])
                        with open(output_path, 'wb') as f:
                            f.write(image_bytes)
                        # 加载生成的图像用于 rewards 计算
                        generated_image = Image.open(output_path).convert('RGB')
                    except Exception as e:
                        print(f"警告: 无法保存 base64 图像，样本 {sample_id} 尝试 {attempt_idx}: {e}")
                        generated_image = None
                elif first_result["type"] == "url":
                    # 处理 URL 图像
                    if download_image_from_url(first_result["data"], output_path):
                        # 加载生成的图像用于 rewards 计算
                        try:
                            generated_image = Image.open(output_path).convert('RGB')
                        except Exception as e:
                            print(f"警告: 无法加载生成的图像用于 rewards 计算: {e}")
                            generated_image = None
                    else:
                        print(f"警告: 下载图像失败，样本 {sample_id} 尝试 {attempt_idx}")
                else:
                    print(f"警告: 未知的图像返回类型: {first_result.get('type')}")
                    generated_image = None
        except Exception as e:
            print(f"处理样本 {sample_id} 尝试 {attempt_idx} 时出错: {e}")
        
        # 计算 rewards（如果提供了 eval_reward_fn）
        if eval_reward_fn is not None and generated_image is not None and executor is not None:
            try:
                # 将图像转换为 tensor 格式（与 inference_sft_lora.py 一致）
                # maze_reward 函数期望 images 是 (B, C, H, W) 格式的 tensor，值范围 [0, 1]
                from torchvision import transforms
                
                # 转换为 tensor (C, H, W)，值范围 [0, 1]
                transform = transforms.Compose([
                    transforms.ToTensor(),  # 自动转换为 [0, 1] 范围
                ])
                image_tensor = transform(generated_image).unsqueeze(0)  # (1, C, H, W)
                
                # 确保在正确的设备上
                if device != "cpu" and torch.cuda.is_available():
                    image_tensor = image_tensor.to(device)
                else:
                    image_tensor = image_tensor.cpu()
                
                # 调用 reward 函数
                # maze_reward 返回的函数签名: (images, prompts, metadata, ref_images=None, only_strict=True)
                images_batch = image_tensor
                prompts_batch = [prompt]
                metadatas_batch = [metadata]
                
                # 使用 executor 异步计算（与 inference_sft_lora.py 一致）
                reward_future = executor.submit(eval_reward_fn, images_batch, prompts_batch, metadatas_batch, only_strict=False)
                rewards_scores, rewards_dict = reward_future.result()
                print(f"Rewards: {rewards_scores}")
                # input()
                
                # rewards_dict 包含所有指标，格式与 inference_sft_lora.py 一致
                # 提取单个样本的 rewards（batch_size=1，所以取第一个元素）
                for key, value in rewards_dict.items():
                    if isinstance(value, (np.ndarray, list, tuple)):
                        sample_reward[key] = float(value[0]) if len(value) > 0 else None
                    elif isinstance(value, torch.Tensor):
                        sample_reward[key] = float(value[0].item()) if value.numel() > 0 else None
                    else:
                        sample_reward[key] = float(value) if value is not None else None
                        
            except Exception as e:
                print(f"警告: 计算 rewards 时出错 (样本 {sample_id}, 尝试 {attempt_idx}): {e}")
                import traceback
                traceback.print_exc()
                sample_reward = {}
        
        # 创建记录（格式与 inference_sft_lora.py 一致）
        result_record = {
            "id": sample_id,
            "attempt": attempt_idx,
            "width": width,
            "height": height,
            "rewards": sample_reward
        }
        
        # 如果是圆形数据集，添加layers字段
        if is_circle_dataset:
            layers = maze_config.get('layers', 0)
            result_record["layers"] = layers
        
        with lock:
            results.append(result_record)
            # 实时写入 JSON（如果提供了 json_path）
            if json_path is not None:
                try:
                    # 读取现有结果
                    all_results_dict = {}
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, 'r', encoding='utf-8') as f:
                                existing = json.load(f)
                                for r in existing:
                                    r_id = r.get("id")
                                    r_attempt = r.get("attempt")
                                    if r_id is not None and r_attempt is not None:
                                        all_results_dict[(r_id, r_attempt)] = r
                        except Exception as e:
                            # 如果读取失败，继续使用空字典
                            pass
                    
                    # 添加新结果（覆盖已有结果）
                    all_results_dict[(sample_id, attempt_idx)] = convert_to_native(result_record)
                    
                    # 写入文件
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(list(all_results_dict.values()), f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"警告: 实时写入 JSON 失败: {e}")
    
    if pbar:
        pbar.update(1)


def convert_to_native(obj):
    """转换 numpy 类型为 Python 原生类型（与 inference_sft_lora.py 一致）"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    return obj


def load_existing_results(resume_dir: str, is_circle_dataset: bool = False) -> tuple:
    """
    加载已有的结果，返回已完成的 attempt 集合和已有结果列表
    
    Args:
        resume_dir: 恢复目录路径
        is_circle_dataset: 是否是圆形数据集
        
    Returns:
        (completed_attempts, existing_results): 
        - completed_attempts: 集合，包含 (sample_id, attempt_idx) 元组
        - existing_results: 已有结果列表
    """
    completed_attempts = set()
    existing_results = []
    
    # 检查 resume_dir 是否存在
    if not os.path.exists(resume_dir):
        print(f"⚠️ 警告: resume_dir 不存在: {resume_dir}，将从头开始")
        return completed_attempts, existing_results
    
    # 加载已有的 sample_rewards.json
    results_json_path = os.path.join(resume_dir, "sample_rewards.json")
    if os.path.exists(results_json_path):
        try:
            with open(results_json_path, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            
            # 从 JSON 结果中提取已完成的 attempt
            # 同时验证对应的图像文件是否存在
            for result in existing_results:
                sample_id = result.get("id")
                attempt_idx = result.get("attempt")
                if sample_id is not None and attempt_idx is not None:
                    # 检查对应的图像文件是否存在
                    # 根据数据集类型构建文件名
                    if is_circle_dataset:
                        # 圆形数据集: layers_{layers}_{sample_id}_attempt{attempt_idx:03d}.jpg
                        layers = result.get("layers", 0)
                        if layers > 0:
                            size_key = f"layers_{layers}"
                            filename_base = f"{size_key}_{sample_id}"
                            output_path = os.path.join(resume_dir, f"{filename_base}_attempt{attempt_idx:03d}.jpg")
                        else:
                            # 如果没有layers字段，尝试从文件名推断
                            output_path = None
                    else:
                        # 其他数据集: {width}x{height}_{sample_id}_attempt{attempt_idx:03d}.jpg
                        width = result.get("width", 0)
                        height = result.get("height", 0)
                        size_key = f"{width}x{height}"
                        filename_base = f"{size_key}_{sample_id}"
                        output_path = os.path.join(resume_dir, f"{filename_base}_attempt{attempt_idx:03d}.jpg")
                    
                    # 如果图像文件存在且不为空，则认为已完成
                    if output_path and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        completed_attempts.add((sample_id, attempt_idx))
            
            print(f"📂 从 {resume_dir} 加载了 {len(existing_results)} 条已有结果")
            print(f"   发现 {len(completed_attempts)} 个已完成的 attempt（图像文件存在）")
        except Exception as e:
            print(f"⚠️ 警告: 无法加载已有结果文件: {e}")
            existing_results = []
    else:
        # 如果没有 JSON 文件，扫描图像文件
        print(f"⚠️ 未找到 sample_rewards.json 文件: {results_json_path}")
        print(f"   将扫描图像文件...")
        
        # 扫描所有 attempt 图像文件
        image_files = [f for f in os.listdir(resume_dir) if f.endswith('.jpg') and '_attempt' in f]
        
        if len(image_files) > 0:
            print(f"   找到 {len(image_files)} 个图像文件")
            
            # 解析文件名
            for img_file in image_files:
                try:
                    # 解析文件名
                    base_name = img_file.replace('.jpg', '')
                    if '_attempt' in base_name:
                        parts = base_name.split('_attempt')
                        if len(parts) == 2:
                            sample_part = parts[0]
                            attempt_str = parts[1]
                            attempt_idx = int(attempt_str)
                            
                            if is_circle_dataset:
                                # 圆形数据集: layers_{layers}_{sample_id}
                                # 例如: "layers_16_fe716a14-c3d0-4d29-ab82-17ea3fa9dc97"
                                if sample_part.startswith('layers_'):
                                    parts_sample = sample_part.split('_', 2)
                                    if len(parts_sample) == 3:
                                        layers = parts_sample[1]
                                        sample_id = parts_sample[2]
                                        img_path = os.path.join(resume_dir, img_file)
                                        if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                                            completed_attempts.add((sample_id, attempt_idx))
                            else:
                                # 其他数据集: {width}x{height}_{sample_id}
                                # 例如: "3x3_sample123"
                                parts_sample = sample_part.split('_', 1)
                                if len(parts_sample) == 2:
                                    size_key = parts_sample[0]
                                    sample_id = parts_sample[1]
                                    
                                    # 解析 width 和 height
                                    if 'x' in size_key:
                                        # 检查图像文件是否存在
                                        img_path = os.path.join(resume_dir, img_file)
                                        if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                                            completed_attempts.add((sample_id, attempt_idx))
                except Exception as e:
                    print(f"警告: 解析文件名失败 {img_file}: {e}")
            
            print(f"   扫描完成，发现 {len(completed_attempts)} 个已完成的 attempt")
            print(f"   将在后续处理时重新计算 reward")
    
    return completed_attempts, existing_results
    
    
def eval_with_api(
    api_provider: str,
    base_url: Optional[str],
    api_key: str,
    model_name: str,
    dataset_path: str,
    split: str,
    output_dir: str,
    num_attempts: int = 1,
    num_threads: int = 4,
    filter_size_min: Optional[int] = None,
    filter_size_max: Optional[int] = None,
    samples_per_size: Optional[int] = None,
    resolution: Optional[int] = None,
    image_size: str = "1024x1024",
    reward_fn: Optional[Dict] = None,
    device: str = "cuda",
    resume_dir: Optional[str] = None
):
    """
    使用 API（OpenAI、Google Gemini 或火山引擎）进行图像编辑评估
    
    Args:
        api_provider: API 提供商 ("openai", "gemini" 或 "volcano")
        base_url: API base URL (可选，None 使用默认)
        api_key: API key
        model_name: 模型名称
        dataset_path: 数据集路径
        split: 数据集分割 ('train' 或 'test')
        output_dir: 输出目录
        num_attempts: 每个样本的尝试次数
        num_threads: 线程数
        filter_size_min: 最小迷宫尺寸
        filter_size_max: 最大迷宫尺寸
        samples_per_size: 每个尺寸的样本数
        resolution: 图像分辨率
        image_size: 生成图像尺寸
        reward_fn: 奖励函数配置（未使用，保留兼容性）
        device: 计算 rewards 的设备
        resume_dir: 恢复目录路径，如果提供，将跳过已完成的 attempt
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 保存图像到: {output_dir}")
    print(f"🔄 每个样本将生成 {num_attempts} 次")
    print(f"🧵 使用 {num_threads} 个线程")
    print(f"🔌 使用 API 提供商: {api_provider}")
    
    # 初始化 API 客户端
    if api_provider == "gemini":
        if not GEMINI_AVAILABLE:
            print("❌ 错误: Gemini API 不可用，请安装 google-genai 库")
            print("   安装命令: pip install google-genai")
            sys.exit(1)
        # 初始化 Gemini 客户端
        # 注意：Gemini API key 可以通过环境变量 GOOGLE_API_KEY 设置
        # 或者通过 client = genai.Client(api_key=api_key) 设置
        if api_key:
            client = genai.Client(api_key=api_key, 
            http_options={
            "base_url": "https://api.zhizengzeng.com/google"
        })
        else:
            # 尝试从环境变量获取
            client = genai.Client()
        print(f"✅ Gemini 客户端初始化成功 (模型: {model_name})")
    elif api_provider == "volcano":
        if not VOLCANO_AVAILABLE:
            print("❌ 错误: 火山引擎 API 不可用，请安装 volcenginesdkarkruntime 库")
            print("   安装命令: pip install 'volcengine-python-sdk[ark]'")
            sys.exit(1)
        # 初始化火山引擎客户端
        # base_url 默认为 "https://ark.cn-beijing.volces.com/api/v3"
        volcano_base_url = "https://api.zhizengzeng.com/bytedance/api/v3/images/generations"
        client = Ark(
            base_url=volcano_base_url,
            api_key=api_key
        )
        print(f"✅ 火山引擎客户端初始化成功 (模型: {model_name}, base_url: {volcano_base_url})")
    else:
        # 初始化 OpenAI 客户端
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
        print(f"✅ OpenAI 客户端初始化成功 (模型: {model_name})")
    
    # 初始化奖励函数（如果可用）
    eval_reward_fn = None
    reward_executor = None
    if maze_reward is not None:
        try:
            print(f"🎯 初始化奖励函数 (device: {device})...")
            eval_reward_fn = maze_reward(device)
            reward_executor = ThreadPoolExecutor(max_workers=8)
            print("✅ 奖励函数初始化成功")
        except Exception as e:
            print(f"⚠️ 警告: 无法初始化奖励函数: {e}")
            print("   将继续运行，但不计算 rewards")
    else:
        print("⚠️ 警告: 奖励函数模块未导入，将不计算 rewards")
    
    # 检测是否是圆形数据集
    is_circle_dataset = dataset_path and 'circle/maze-dataset' in dataset_path
    
    # 加载数据集
    print(f"📂 加载数据集: {dataset_path} (split: {split})")
    if is_circle_dataset:
        print(f"🔵 检测到圆形数据集，将使用 layers 进行分类")
    dataset = MazePromptImageDataset(
        dataset_path,
        split=split,
        filter_size_min=filter_size_min,
        filter_size_max=filter_size_max,
        samples_per_size=samples_per_size,
        resolution=resolution
    )
    
    print(f"📊 数据集大小: {len(dataset)}")
    
    # 加载已有结果（如果提供了 resume_dir）
    completed_attempts = set()
    existing_results = []
    if resume_dir is not None:
        print(f"🔄 恢复模式: 从 {resume_dir} 加载已有结果...")
        completed_attempts, existing_results = load_existing_results(resume_dir, is_circle_dataset)
        
        # 如果没有 JSON 文件，但图像文件存在，重新计算 reward
        results_json_path = os.path.join(resume_dir, "sample_rewards.json")
        if not os.path.exists(results_json_path) and len(completed_attempts) > 0:
            print(f"   未找到 JSON 文件，开始重新计算已有 attempt 的 reward...")
            
            # 创建样本 ID 到索引的映射
            sample_id_to_idx = {}
            for idx in range(len(dataset)):
                try:
                    sample = dataset[idx]
                    sample_id = sample.get("id")
                    if sample_id:
                        sample_id_to_idx[sample_id] = idx
                except:
                    continue
            
            # 为每个已完成的 attempt 重新计算 reward
            for (sample_id, attempt_idx) in completed_attempts:
                if sample_id in sample_id_to_idx:
                    try:
                        # 获取样本
                        sample = dataset[sample_id_to_idx[sample_id]]
                        prompt = sample["prompt"]
                        metadata = sample["metadata"]
                        
                        # 构建图像文件路径
                        # 需要从文件名中提取 size_key
                        image_files = [f for f in os.listdir(resume_dir) 
                                     if f.endswith('.jpg') and '_attempt' in f and sample_id in f]
                        
                        for img_file in image_files:
                            # 解析文件名获取 attempt_idx
                            base_name = img_file.replace('.jpg', '')
                            # 支持两种格式: {size_key}_{sample_id}_attempt{attempt_idx:03d} 或 layers_{layers}_{sample_id}_attempt{attempt_idx:03d}
                            if f'_attempt{attempt_idx:03d}' in base_name:
                                img_path = os.path.join(resume_dir, img_file)
                                
                                # 加载图像
                                generated_image = Image.open(img_path).convert('RGB')
                                
                                # 计算 reward
                                if eval_reward_fn is not None and reward_executor is not None:
                                    try:
                                        from torchvision import transforms
                                        transform = transforms.Compose([
                                            transforms.ToTensor(),
                                        ])
                                        image_tensor = transform(generated_image).unsqueeze(0)
                                        
                                        if device != "cpu" and torch.cuda.is_available():
                                            image_tensor = image_tensor.to(device)
                                        else:
                                            image_tensor = image_tensor.cpu()
                                        
                                        images_batch = image_tensor
                                        prompts_batch = [prompt]
                                        metadatas_batch = [metadata]
                                        
                                        reward_future = reward_executor.submit(
                                            eval_reward_fn, images_batch, prompts_batch, 
                                            metadatas_batch, only_strict=False
                                        )
                                        rewards_scores, rewards_dict = reward_future.result()
                                        
                                        # 提取 maze_config
                                        maze_config = {}
                                        if 'metadata' in metadata:
                                            try:
                                                if isinstance(metadata['metadata'], str):
                                                    parsed_meta = json.loads(metadata['metadata'])
                                                    maze_config = parsed_meta.get('maze_config', {})
                                                elif isinstance(metadata['metadata'], dict):
                                                    maze_config = metadata['metadata'].get('maze_config', {})
                                            except (json.JSONDecodeError, TypeError):
                                                pass
                                        
                                        if not maze_config:
                                            maze_config = metadata.get('maze_config', {})
                                        
                                        width = maze_config.get('width', 0)
                                        height = maze_config.get('height', 0)
                                        
                                        # 构建 reward 字典
                                        sample_reward = {}
                                        for key, value in rewards_dict.items():
                                            if isinstance(value, (np.ndarray, list, tuple)):
                                                sample_reward[key] = float(value[0]) if len(value) > 0 else None
                                            elif isinstance(value, torch.Tensor):
                                                sample_reward[key] = float(value[0].item()) if value.numel() > 0 else None
                                            else:
                                                sample_reward[key] = float(value) if value is not None else None
                                        
                                        # 添加到已有结果
                                        result_record = {
                                            "id": sample_id,
                                            "attempt": attempt_idx,
                                            "width": width,
                                            "height": height,
                                            "rewards": sample_reward
                                        }
                                        
                                        # 如果是圆形数据集，添加layers字段
                                        if is_circle_dataset:
                                            layers = maze_config.get('layers', 0)
                                            result_record["layers"] = layers
                                        
                                        existing_results.append(result_record)
                                        
                                        print(f"   ✓ 重新计算 reward: 样本 {sample_id}, attempt {attempt_idx}")
                                    except Exception as e:
                                        print(f"   警告: 重新计算 reward 失败 (样本 {sample_id}, attempt {attempt_idx}): {e}")
                                
                                break
                    except Exception as e:
                        print(f"   警告: 处理样本 {sample_id} 时出错: {e}")
            
            # 保存重新计算的结果
            if len(existing_results) > 0:
                results_json_path = os.path.join(resume_dir, "sample_rewards.json")
                with open(results_json_path, 'w', encoding='utf-8') as f:
                    json.dump(convert_to_native(existing_results), f, indent=2, ensure_ascii=False)
                print(f"   💾 已保存 {len(existing_results)} 条重新计算的结果到 {results_json_path}")
        
        if len(completed_attempts) > 0:
            print(f"   将跳过 {len(completed_attempts)} 个已完成的 attempt")
    else:
        print("🆕 新建模式: 从头开始处理所有样本")
    
    # 使用延迟加载：不在主线程一次性加载所有样本
    # 而是在工作线程中按需加载，这样可以并行解码图片，避免阻塞
    
    # 多线程处理
    results = []
    lock = threading.Lock()
    
    # JSON 文件路径（用于实时写入）
    json_path = os.path.join(output_dir, "sample_rewards.json")
    
    def load_and_process_sample(idx):
        """在工作线程中加载样本并处理"""
        try:
            sample = dataset[idx]
            process_single_sample(
                client,
                sample,
                output_dir,
                model_name,
                image_size,
                num_attempts,
                lock,
                results,
                eval_reward_fn,
                reward_executor,
                device,
                api_provider,
                pbar,
                completed_attempts,
                json_path,
                is_circle_dataset
            )
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"🚀 开始处理 {len(dataset)} 个样本（使用延迟加载）...")
    
    with tqdm(total=len(dataset), desc="处理进度") as pbar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_list = []
            for idx in range(len(dataset)):
                future = executor.submit(load_and_process_sample, idx)
                future_list.append(future)
            
            # 等待所有任务完成
            for future in as_completed(future_list):
                try:
                    future.result()
                except Exception as e:
                    print(f"任务执行出错: {e}")
        
    # 合并已有结果和新结果
    # 创建一个字典，以 (id, attempt) 为键，避免重复
    all_results_dict = {}
    
    # 先添加已有结果
    for result in existing_results:
        sample_id = result.get("id")
        attempt_idx = result.get("attempt")
        if sample_id is not None and attempt_idx is not None:
            all_results_dict[(sample_id, attempt_idx)] = result
    
    # 再添加新结果（会覆盖已有结果中相同的键，这样新结果优先）
    sample_results_native = convert_to_native(results)
    for result in sample_results_native:
        sample_id = result.get("id")
        attempt_idx = result.get("attempt")
        if sample_id is not None and attempt_idx is not None:
            all_results_dict[(sample_id, attempt_idx)] = result
    
    # 转换为列表
    all_results = list(all_results_dict.values())
    
    # 保存为 sample_rewards.json（与 inference_sft_lora.py 一致）
    results_json_path = os.path.join(output_dir, "sample_rewards.json")
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    new_count = len(sample_results_native)
    existing_count = len(existing_results)
    total_count = len(all_results)
    print(f"💾 保存结果到: {results_json_path}")
    print(f"   新结果: {new_count} 条")
    print(f"   已有结果: {existing_count} 条")
    print(f"   总计: {total_count} 条")
    
    # 打印统计信息（使用合并后的结果）
    print("\n" + "="*80)
    print("📊 评估结果统计:")
    print("="*80)
    total_samples = len(set(r["id"] for r in all_results))
    total_attempts = len(all_results)
    
    # 统计成功次数（有 rewards 且不是空的）
    successful_attempts = sum(
        1 for r in all_results 
        if r.get("rewards") and len(r.get("rewards", {})) > 0
    )
    success_rate = (successful_attempts / total_attempts * 100) if total_attempts > 0 else 0
    
    print(f"  总样本数: {total_samples}")
    print(f"  总尝试次数: {total_attempts}")
    print(f"  成功计算 rewards 的次数: {successful_attempts}")
    print(f"  成功率: {success_rate:.2f}%")
    
    # 如果有 rewards，打印平均 rewards
    if successful_attempts > 0:
        all_rewards_keys = set()
        for r in all_results:
            if r.get("rewards"):
                all_rewards_keys.update(r["rewards"].keys())
        
        if all_rewards_keys:
            print("\n  平均 Rewards:")
            for key in sorted(all_rewards_keys):
                values = [
                    r["rewards"][key] 
                    for r in all_results 
                    if r.get("rewards") and key in r["rewards"] and r["rewards"][key] is not None
                ]
                if values:
                    avg_value = np.mean(values)
                    print(f"    {key}: {avg_value:.4f}")
    
    print("="*80)
    print(f"\n✅ 图像已保存到: {output_dir}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="使用 OpenAI API 进行图像编辑测试")
    
    # 必需参数
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API 密钥")
    parser.add_argument("--dataset_path", type=str, required=True, help="数据集路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    
    # 可选参数
    parser.add_argument("--base_url", type=str, default="https://litellm.mybigai.ac.cn/", help="OpenAI API base URL (可选)")
    parser.add_argument("--model", type=str, default="gpt-image-1", help="模型名称 (默认: dall-e-2)")
    parser.add_argument("--split", type=str, default="test_hexagon", choices=["train", "test", "test_hexagon"], 
                       help="数据集分割 (默认: test_hexagon)")
    parser.add_argument("--num_attempts", type=int, default=1, help="每个样本的尝试次数 (默认: 1)")
    parser.add_argument("--num_threads", type=int, default=4, help="线程数 (默认: 4)")
    parser.add_argument("--filter_size_min", type=int, default=None, help="最小迷宫尺寸")
    parser.add_argument("--filter_size_max", type=int, default=None, help="最大迷宫尺寸")
    parser.add_argument("--samples_per_size", type=int, default=None, help="每个尺寸的样本数")
    parser.add_argument("--resolution", type=int, default=None, help="图像分辨率")
    parser.add_argument("--image_size", type=str, default="1024x1024", 
                       choices=["256x256", "512x512", "1024x1024"],
                       help="生成图像尺寸 (默认: 1024x1024)")
    parser.add_argument("--device", type=str, default="cpu", 
                       choices=["cpu", "cuda"],
                       help="计算 rewards 的设备 (默认: cpu)")
    parser.add_argument("--api_provider", type=str, default="openai",
                       choices=["openai", "gemini", "volcano"],
                       help="API 提供商 (默认: openai)")
    parser.add_argument("--resume_dir", type=str, default=None,
                       help="恢复目录路径，如果提供，将跳过已完成的 attempt (默认: None)")
    
    args = parser.parse_args()
    
    # 运行评估
    eval_with_api(
        api_provider=args.api_provider,
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=args.model,
        dataset_path=args.dataset_path,
        split=args.split,
        output_dir=args.output_dir,
        num_attempts=args.num_attempts,
        num_threads=args.num_threads,
        filter_size_min=args.filter_size_min,
        filter_size_max=args.filter_size_max,
        samples_per_size=args.samples_per_size,
        resolution=args.resolution,
        image_size=args.image_size,
        device=args.device,
        resume_dir=args.resume_dir
    )


if __name__ == "__main__":
    main()
