from collections import defaultdict
import os
import sys
import datetime
from concurrent import futures
import time
import json
from safetensors.torch import load_file
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator, init_empty_weights, load_checkpoint_in_model
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from accelerate.hooks import remove_hook_from_module

# bagel
from infer.bagel.data.data_utils import add_special_tokens
from infer.bagel.data.transforms import ImageTransform
from infer.bagel.modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from infer.bagel.modeling.qwen2 import Qwen2Tokenizer
from infer.bagel.modeling.autoencoder import load_ae
from infer.bagel.inferencer import InterleaveInferencer

import numpy as np
from infer import maze_metrics
import torch
from functools import partial
import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import snapshot_download

from dataset.maze_dataset import MazeDataset


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

def decode_base64_image(base64_str):
    """Decode base64 encoded image string to PIL Image."""
    import base64
    from io import BytesIO
    img_bytes = base64.b64decode(base64_str)
    return Image.open(BytesIO(img_bytes)).convert('RGB')

class MazePromptImageDataset(Dataset):
    def __init__(self, dataset_path, split='train', filter_size_min=None, filter_size_max=None, samples_per_size=None, filter_shape=None, resolution=None):
        """
        Args:
            dataset_path: Path to maze dataset
            split: 'train' or 'test'
            filter_size_min: Optional, minimum maze size (e.g., 5 for 5x5)
            filter_size_max: Optional, maximum maze size (e.g., 10 for 10x10)
            samples_per_size: Optional, number of samples to select per size (e.g., 3 means 3 samples each for 5x5, 6x6, etc.)
            filter_shape: Optional, maze shape type to filter (e.g., 'triangle', 'square', 'hexagon')
            If both are set, only includes mazes where width and height are in [min, max] range
        """
        self.dataset_path = dataset_path
        self.maze_dataset = MazeDataset(dataset_path, split=split)
        self.filter_size_min = filter_size_min
        self.filter_size_max = filter_size_max
        self.samples_per_size = samples_per_size
        self.filter_shape = filter_shape
        self.resolution = resolution
        # Build filtered indices if filter is specified
        if filter_size_min is not None or filter_size_max is not None or samples_per_size is not None or filter_shape is not None:
            if filter_shape is not None:
                print(f"🔍 Filtering dataset by shape: {filter_shape}")
            if filter_size_min is not None and filter_size_max is not None:
                print(f"🔍 Filtering dataset by size range: [{filter_size_min}, {filter_size_max}]")
            elif filter_size_min is not None:
                print(f"🔍 Filtering dataset by minimum size: >= {filter_size_min}")
            elif filter_size_max is not None:
                print(f"🔍 Filtering dataset by maximum size: <= {filter_size_max}")
            if samples_per_size is not None:
                print(f"🔍 Selecting {samples_per_size} samples per size")
            self.filtered_indices = self._build_filtered_indices()
            print(f"   Found {len(self.filtered_indices)} samples matching filter criteria")
        else:
            self.filtered_indices = list(range(len(self.maze_dataset)))
    
    def _build_filtered_indices(self):
        """Build list of indices that match the size filter criteria.
        
        Filtering logic matches inference_multi.py:
        - Requires width == height (square mazes only)
        - Checks if shape matches filter_shape (if specified)
        - Checks if size is within [filter_size_min, filter_size_max] range
        - If samples_per_size is set, only selects first N samples per size
        """
        # First pass: categorize samples by size
        samples_by_size = {}
        
        for idx in range(len(self.maze_dataset)):
            maze_item = self.maze_dataset[idx]
            maze_config = self._extract_maze_config(maze_item)
            width = maze_config.get('width', 0)
            height = maze_config.get('height', 0)
            shape = maze_config.get('shape', '')
            
            # Only include square mazes (width == height)
            if width != height:
                continue
            
            # Check if shape matches filter_shape (if specified)
            if self.filter_shape is not None:
                if shape != self.filter_shape:
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
            # Select first N samples per size
            for size_key in sorted(samples_by_size.keys()):
                available = samples_by_size[size_key]
                selected = available[:self.samples_per_size]
                filtered.extend(selected)
                print(f"   {size_key}: selected {len(selected)}/{len(available)} samples")
        else:
            # Select all samples
            for size_key in sorted(samples_by_size.keys()):
                filtered.extend(samples_by_size[size_key])
        
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
        try:
            # Check if original_img is in metadata and is a base64 string
            if 'm_original_img' in maze_item["metadata"]:
                # Decode the base64 image
                image = decode_base64_image(maze_item["metadata"]['m_original_img'])
                # Resize to match the expected resolution
                # image = image.resize((self.resolution, self.resolution))
                
        except Exception as e:
            # If any error occurs in decoding, use placeholder
            print(f"Warning: Could not decode image for sample {idx}, Error: {e}")
            return self.__getitem__((idx + 1) % len(self))


        # Create a unique identifier for the prompt with image path
        if 'id' in maze_item["metadata"]:
            prompt_with_image_path = f"{maze_item['metadata']['id']}"
        else:
            prompt_with_image_path = f"{maze_item['prompt']}_maze_{idx}"

        # Prepare metadata for maze reward function - decode all base64 images to PIL
        processed_metadata = maze_item["metadata"].copy()
        # print(f"Processed Metadata: {processed_metadata}")
        # print(f"Maze Item: {maze_item}")

        # Decode all image fields from base64 to PIL Image for reward function
        image_fields = ['original_img', 'm_original_img', 'sol_img', 'mask_img', 'cell_map']
        for field in image_fields:
            if field in processed_metadata and isinstance(processed_metadata[field], str):
                try:
                    # Decode base64 to PIL Image
                    pil_image = decode_base64_image(processed_metadata[field])
                    # pil_image = pil_image.resize((self.resolution, self.resolution))
                    processed_metadata[field] = pil_image
                except Exception as e:
                    print(f"Warning: Could not decode {field} for sample {idx}: {e}")
                    # Keep the original base64 string if decoding fails
                    pass

        # 确保metadata JSON字符串也被包含在metadata中
        # 注意：这里不需要解码，直接保持字符串格式
        if 'metadata' in maze_item["metadata"] and isinstance(maze_item["metadata"]['metadata'], str):
            # metadata字段已经存在，保持不变
            pass

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



def eval(inferencer, inference_hyper, test_dataloader, tokenizer, config, accelerator, eval_reward_fn, executor, autocast, output_dir, num_attempts=1):
    """Run evaluation on test dataset and save images to output_dir.
    
    Args:
        output_dir: Directory to save generated images
        num_attempts: Number of generation attempts per sample (different random seeds)
    """
    import random
    import math
    
    dataset = test_dataloader.dataset
    indices = list(range(len(dataset)))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    if accelerator.is_main_process:
        
        print(f"📁 Saving images to: {output_dir}")
        print(f"🔄 Each sample will be generated {num_attempts} times")
    
    # ========== 多卡并行：每张卡负责不同的 attempts ==========
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    
    # 计算每张卡负责的 attempts 范围
    attempts_per_gpu = math.ceil(num_attempts / world_size)
    start_attempt = rank * attempts_per_gpu + 1
    end_attempt = min((rank + 1) * attempts_per_gpu, num_attempts)
    
    # 当前卡负责的 attempts 列表
    my_attempts = list(range(start_attempt, end_attempt + 1))
    
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"🚀 Multi-GPU Attempt Distribution:")
        print(f"   Total attempts: {num_attempts}")
        print(f"   Number of GPUs: {world_size}")
        print(f"   Attempts per GPU: ~{attempts_per_gpu}")
        for r in range(world_size):
            r_start = r * attempts_per_gpu + 1
            r_end = min((r + 1) * attempts_per_gpu, num_attempts)
            print(f"   GPU {r}: attempts {r_start}-{r_end}")
        print(f"{'='*60}\n")
    
    print(f"[GPU {rank}] Will process attempts: {my_attempts}")
    # ====================================================
    
    # ========== 按尺寸分组，然后创建batch ==========
    # 检查是否是圆形数据集（按layers分组）
    dataset_path = getattr(test_dataloader.dataset, 'dataset_path', None)
    is_circle_dataset = dataset_path and 'circle/maze-dataset' in dataset_path
    
    # 先按尺寸分组样本
    samples_by_size = defaultdict(list)
    for idx in indices:
        sample = dataset[idx]
        metadata = sample.get("metadata", {})
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
        else:
            width = maze_config.get('width', 0)
            height = maze_config.get('height', 0)
            size_key = f"{width}x{height}"
        samples_by_size[size_key].append(idx)
    
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        if is_circle_dataset:
            print(f"📊 按layers分组结果（圆形数据集）:")
        else:
            print(f"📊 按尺寸分组结果:")
        for size_key in sorted(samples_by_size.keys()):
            print(f"   {size_key}: {len(samples_by_size[size_key])} 个样本")
        print(f"{'='*60}\n")
    
    # 对每个尺寸组分别创建batch
    eval_batches = []
    # 如果设置了 test_batch_size_per_size，使用它；否则使用 test_batch_size
    bs = getattr(config.sample, 'test_batch_size_per_size', None)
    if bs is None:
        bs = config.sample.test_batch_size
    if accelerator.is_main_process:
        print(f"📦 每个尺寸内的batch大小: {bs} (test_batch_size_per_size={getattr(config.sample, 'test_batch_size_per_size', None)}, test_batch_size={config.sample.test_batch_size})")
    for size_key in sorted(samples_by_size.keys()):
        size_indices = samples_by_size[size_key]
        for i in range(0, len(size_indices), bs):
            batch_indices = size_indices[i : i + bs]
            batch_samples = [dataset[idx] for idx in batch_indices]
            eval_batches.append((batch_indices, test_dataloader.collate_fn(batch_samples), size_key))
    # ====================================================

    all_rewards = defaultdict(list)
    sample_results = []  # 存储每个样本的 id 和 rewards
    
    # Outer loop: iterate over attempts assigned to this GPU
    for attempt_idx in my_attempts:
        print(f"[GPU {rank}] 🎲 Processing Attempt {attempt_idx}/{num_attempts}")
        
        # Set different random seed for each attempt
        attempt_seed = config.seed + attempt_idx * 1000
        torch.manual_seed(attempt_seed)
        torch.cuda.manual_seed_all(attempt_seed)
        random.seed(attempt_seed)
        np.random.seed(attempt_seed)
        
        for batch_indices, test_batch, size_key in tqdm(
                eval_batches,
                desc=f"[GPU {rank}] Eval (attempt {attempt_idx}/{num_attempts}): ",
                disable=not accelerator.is_local_main_process,
                position=rank,  # 每张卡使用不同的 position，避免输出混乱
            ):
            input_pil_images = None
            if len(test_batch) == 4:  # MazeDataset
                prompts, metadatas, input_pil_images, ids = test_batch
            else:
                raise ValueError(f"Unexpected batch length: {len(test_batch)}")
            
            # 按尺寸分组后，同一batch内所有样本尺寸相同，可以安全地stack
            # 使用批量推理：一次性处理整个batch
            with autocast():
                with torch.no_grad():
                    # 批量推理：一次性处理整个batch
                    
                    output_dict = inferencer(
                        image=input_pil_images,  # 传入图像列表
                        text=prompts,  # 传入文本列表
                        noise_level=0, 
                        grpo_config=config, 
                        accelerator=accelerator, 
                        num_timesteps=config.sample.eval_num_steps,
                        **inference_hyper
                    )
                    
                    # 批量模式返回字典，包含 'images' 键（图像列表）
                    if 'images' in output_dict and output_dict['images'] is not None:
                        # 批量模式：返回图像列表
                        output_pil_images = output_dict['images']
                    else:
                        # 兼容单样本模式（不应该发生）
                        output_pil_images = [output_dict['image']]
                    
                    # 调试信息：检查返回的图像数量
                    if len(output_pil_images) != len(prompts):
                        print(f"[GPU {rank}] ⚠️ Warning: Batch inference returned {len(output_pil_images)} images, but expected {len(prompts)}")
                        print(f"[GPU {rank}] Output dict keys: {output_dict.keys()}")
                    
                    # 将PIL图像转换为torch tensor
                    images = []
                    for pil_img in output_pil_images:
                        if isinstance(pil_img, Image.Image):
                            # PIL Image -> numpy -> tensor
                            img_array = np.array(pil_img)
                            if len(img_array.shape) == 3:
                                # (H, W, C) -> (C, H, W)
                                img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float() / 255.0
                            else:
                                img_tensor = torch.from_numpy(img_array).float() / 255.0
                            images.append(img_tensor)
                        elif isinstance(pil_img, torch.Tensor):
                            # 如果已经是tensor，确保在CPU上（因为来自PIL转换的tensor在CPU上）
                            images.append(pil_img.cpu() if pil_img.is_cuda else pil_img)
                        else:
                            raise ValueError(f"Unexpected image type: {type(pil_img)}")
                    
                    # Stack成batch tensor，并移到GPU（reward计算需要GPU tensor）
                    images = torch.stack(images, dim=0)
                    images = images.to(accelerator.device)
                    
                    # 清理中间变量，释放内存
                    del output_dict, output_pil_images
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
     
            rewards_future = executor.submit(eval_reward_fn, images, prompts, metadatas, only_strict=False)
            time.sleep(0)
            scores, rewards = rewards_future.result()
            reward_dict = {}
            print(f"Rewards: {rewards}")
            # print(f"Reward Metadata: {reward_dict}")

            for key, value in rewards.items():
                all_rewards[key].append(value)
            
            for batch_idx, sample_id in enumerate(ids):
                sample_reward = {}
                for key, value in rewards.items():
                    # value 可能是数组或单个值
                    if isinstance(value, (np.ndarray, list, tuple)):
                        sample_reward[key] = float(value[batch_idx]) if len(value) > batch_idx else None
                    else:
                        sample_reward[key] = float(value) if value is not None else None
                
                # 从 metadata 中提取 maze_config 信息
                metadata = metadatas[batch_idx]
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
                
                # 如果上面没找到，尝试直接从 metadata 获取
                if not maze_config:
                    maze_config = metadata.get('maze_config', {})
                
                width = maze_config.get('width', 0)
                height = maze_config.get('height', 0)
                # size = f"{width}x{height}"
                
                sample_results.append({
                    "id": sample_id,
                    "attempt": attempt_idx,
                    "width": width,
                    "height": height,
                    "rewards": sample_reward
                })
            
         
            for batch_idx, (global_idx, metadata, input_img, sample_id) in enumerate(
                zip(batch_indices, metadatas, input_pil_images, ids)
            ):

                maze_config = {}
                if 'metadata' in metadata:
                    try:
                        if isinstance(metadata['metadata'], str):
                            parsed_meta = json.loads(metadata['metadata'])
                            maze_config = parsed_meta.get('maze_config', {})
                        elif isinstance(metadata['metadata'], dict):
                            maze_config = metadata['metadata'].get('maze_config', {})
                    except (json.JSONDecodeError, TypeError) as e:
                        print(f"Warning: Could not parse metadata: {e}")
                
                # 如果上面没找到，尝试直接从 metadata 获取
                if not maze_config:
                    maze_config = metadata.get('maze_config', {})
                
                # 圆形数据集使用 layers，其他数据集使用 width x height
                if is_circle_dataset:
                    layers = maze_config.get('layers', 0)
                    size = f"layers_{layers}"
                else:
                    width = maze_config.get('width', 0)
                    height = maze_config.get('height', 0)
                    size = f"{width}x{height}"
                shape = maze_config.get('shape', 'unknown')
                sample_num = sample_id
                
                # Create filename base: size_id
                filename_base = f"{size}_{sample_num}"
                
                # Save input image (only once, by main process on first attempt)
                # 使用 accelerator.is_main_process 确保只有主进程保存，避免多GPU重复保存
                if attempt_idx == 1 and accelerator.is_main_process and input_img is not None:
                    input_path = os.path.join(output_dir, f"{filename_base}_input.jpg")
                    if not os.path.exists(input_path):
                        input_img.save(input_path)
                        print(f"[GPU {rank}] 💾 Saved input image: {input_path}")
                
                # Save GT image (solution image) if available (only once, by main process on first attempt)
                if attempt_idx == 1 and accelerator.is_main_process:
                    sol_img = metadata.get('sol_img', None)
                    if sol_img is not None:
                        # sol_img 可能是 PIL Image（已解码）或 base64 字符串
                        if isinstance(sol_img, Image.Image):
                            gt_img = sol_img
                        elif isinstance(sol_img, str):
                            # 如果是 base64 字符串，需要解码
                            try:
                                gt_img = decode_base64_image(sol_img)
                                gt_img = gt_img.resize((config.sample.resolution, config.sample.resolution))
                            except Exception as e:
                                print(f"Warning: Could not decode sol_img for sample {sample_id}: {e}")
                                gt_img = None
                        else:
                            gt_img = None
                        
                        if gt_img is not None:
                            gt_path = os.path.join(output_dir, f"{filename_base}_gt.jpg")
                            if not os.path.exists(gt_path):
                                gt_img.save(gt_path)
                                print(f"[GPU {rank}] 💾 Saved GT image: {gt_path}")
                
                # Save generated image with attempt number (each GPU saves its own attempts)
                gen_image = images[batch_idx].cpu().numpy()
                gen_pil = Image.fromarray(
                    (gen_image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                # Use attempt naming: attemptXXX (3 digits, zero-padded)
                attempt_path = os.path.join(output_dir, f"{filename_base}_attempt{attempt_idx:03d}.jpg")
                gen_pil.save(attempt_path)
            
            # ==================== 内存清理：每个batch后清理 ====================
            # 将images移到CPU并删除，释放GPU内存
            del images
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # 等待所有CUDA操作完成
                torch.cuda.empty_cache()
            # 启用更激进的内存清理（批量推理时内存压力大）
            import gc
            gc.collect()

    # 处理所有 rewards（统一展平为数组）
    all_rewards_processed = {}
    for key, value in all_rewards.items():
        if len(value) == 0:
            all_rewards_processed[key] = np.array([])
        else:
            # 将所有列表展平为一个列表
            flattened = []
            for batch_rewards in value:
                if isinstance(batch_rewards, np.ndarray):
                    flattened.extend(batch_rewards.tolist())
                elif isinstance(batch_rewards, (list, tuple)):
                    flattened.extend(batch_rewards)
                else:
                    flattened.append(batch_rewards)
            all_rewards_processed[key] = np.array(flattened)

    all_rewards = all_rewards_processed
    
    # 汇总所有 GPU 的结果到一个 JSON 文件
    # 转换 numpy 类型为 Python 原生类型的辅助函数
    def convert_to_native(obj):
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
    
    # 每个 GPU 先保存临时文件
    if len(sample_results) > 0:
        temp_json_path = os.path.join(output_dir, f"sample_rewards_rank{rank}.json.tmp")
        sample_results_native = convert_to_native(sample_results)
        with open(temp_json_path, 'w', encoding='utf-8') as f:
            json.dump(sample_results_native, f, indent=2, ensure_ascii=False)
        print(f"[GPU {rank}] 💾 Saved {len(sample_results)} sample results to temp file")
    
    # 同步所有进程
    accelerator.wait_for_everyone()
    
    # 主进程合并所有临时文件
    if accelerator.is_main_process:
        all_sample_results = []
        print(f"\n{'='*60}")
        print(f"📦 开始合并临时文件 (主进程)")
        print(f"   预期进程数: {accelerator.num_processes}")
        print(f"{'='*60}\n")
        
        for r in range(accelerator.num_processes):
            temp_json_path = os.path.join(output_dir, f"sample_rewards_rank{r}.json.tmp")
            print(f"   检查 rank {r} 的临时文件: {temp_json_path}")
            if os.path.exists(temp_json_path):
                try:
                    with open(temp_json_path, 'r', encoding='utf-8') as f:
                        gpu_results = json.load(f)
                    if isinstance(gpu_results, list):
                        print(f"   ✅ Rank {r}: 找到 {len(gpu_results)} 个结果")
                        all_sample_results.extend(gpu_results)
                    else:
                        print(f"   ✅ Rank {r}: 找到 1 个结果 (非列表格式)")
                        all_sample_results.append(gpu_results)
                    # 删除临时文件
                    os.remove(temp_json_path)
                    print(f"   🗑️  Rank {r}: 已删除临时文件")
                except Exception as e:
                    print(f"   ❌ Warning: Could not read temp file from rank {r}: {e}")
            else:
                print(f"   ⚠️ Rank {r}: 临时文件不存在!")
        
        print(f"\n{'='*60}")
        print(f"📊 合并统计:")
        print(f"   总结果数: {len(all_sample_results)}")
        if len(all_sample_results) > 0:
            # 统计每个attempt的结果数
            attempts_count = {}
            for item in all_sample_results:
                att = item.get('attempt', -1)
                attempts_count[att] = attempts_count.get(att, 0) + 1
            print(f"   Attempts分布: {sorted(attempts_count.items())}")
        print(f"{'='*60}\n")
        
        # 保存汇总结果
        if len(all_sample_results) > 0:
            results_json_path = os.path.join(output_dir, "sample_rewards.json")
            with open(results_json_path, 'w', encoding='utf-8') as f:
                json.dump(all_sample_results, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved {len(all_sample_results)} sample results to {results_json_path}")
        else:
            print(f"⚠️ Warning: No sample results to save!")
    
    return all_rewards


def main(_):
    config = FLAGS.config

    # 根据 config.is_circle 设置 maze_metrics 的 IS_CIRCLE / IS_ISCIRCLE（与 infer_auto.sh / config/maze.py 一致）
    is_circle = getattr(config, 'is_circle', False)
    maze_metrics.IS_CIRCLE = bool(is_circle)

    if not config.run_name:
        # 如果没有设置run_name，使用时间戳
        unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        config.run_name = unique_id
    else:
        # 如果已经设置了run_name，添加时间戳
        unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        config.run_name += "_" + unique_id

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
    )
    
    # 同步run_name：使用主进程的run_name，确保所有进程使用相同的输出目录
    if accelerator.num_processes > 1:
        # 使用固定的同步文件名（不包含PID）
        # 确保logdir目录存在
        os.makedirs(config.logdir, exist_ok=True)
        sync_file = os.path.join(config.logdir, ".run_name_sync.tmp")
        if accelerator.is_main_process:
            # 主进程：将自己的run_name写入文件
            master_run_name = config.run_name
            with open(sync_file, 'w') as f:
                f.write(master_run_name)
        else:
            # 非主进程：等待主进程写入文件，然后读取
            import time
            max_wait = 10  # 最多等待10秒
            waited = 0
            while not os.path.exists(sync_file) and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1
            
            if os.path.exists(sync_file):
                with open(sync_file, 'r') as f:
                    master_run_name = f.read().strip()
                config.run_name = master_run_name
            else:
                print(f"[GPU {accelerator.process_index}] ⚠️ Warning: Could not sync run_name, using local one")
        
        # 同步所有进程，确保所有进程都读取了run_name
        accelerator.wait_for_everyone()
        
        # 清理临时文件（只在主进程中）
        if accelerator.is_main_process and os.path.exists(sync_file):
            try:
                os.remove(sync_file)
            except:
                pass
    accelerator.state.fsdp_plugin.transformer_cls_names_to_wrap = ['Qwen2MoTDecoderLayer']

    logger.info(f"\n{config}")

    set_seed(config.seed, device_specific=True)

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    model_path = config.pretrained.model
    if not os.path.exists(model_path):
        model_local_dir = snapshot_download(repo_id=model_path)
    else:
        model_local_dir = model_path

    # LLM config
    llm_config = Qwen2Config.from_json_file(os.path.join(model_local_dir, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # ViT config
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_local_dir, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers -1

    # VAE loading
    vae_model, vae_config = load_ae(local_path=os.path.join(model_local_dir, "ae.safetensors"))

    # Bagel config
    bagel_config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, bagel_config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # Tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(model_local_dir)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # Image Transform
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    print('***************accelerator.process_index**********', accelerator.process_index)

    # ================== 模型加载逻辑 ==================

    
    # Step 1: 加载 Base Model (Bagel 原版权重，包含 VAE、ViT、LLM 等)
    base_checkpoint = None
    candidates = ["model.safetensors", "ema.safetensors"]
    for c in candidates:
        p = os.path.join(model_local_dir, c)
        if os.path.exists(p):
            base_checkpoint = p
            break
            
    if base_checkpoint is None:
        raise FileNotFoundError(f"Could not find model weights in {model_local_dir}")

    print(f"🚀 [1/3] Loading Base Model (VAE+LLM) from: {base_checkpoint}")
    
    load_checkpoint_in_model(
        model,
        checkpoint=base_checkpoint,
        device_map={"": "cpu"},
        dtype=inference_dtype,
        offload_folder="/tmp/offload"
    )

    # Step 0: 应用 LoRA 配置（如果使用 LoRA，必须在加载 checkpoint 之前应用）
    use_lora = config.use_lora
    if use_lora:
        print("🔧 Applying LoRA configuration to model...")
        from peft import LoraConfig, get_peft_model
        target_modules = [
            "self_attn.q_proj_moe_gen",
            "self_attn.k_proj_moe_gen",
            "self_attn.v_proj_moe_gen",
            "self_attn.o_proj_moe_gen",
            # "self_attn.q_proj",
            # "self_attn.k_proj",
            # "self_attn.v_proj",
            # "self_attn.o_proj",
            "mlp_moe_gen.gate_proj",
            "mlp_moe_gen.up_proj",
            "mlp_moe_gen.down_proj",
        ]
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        model.language_model = get_peft_model(model.language_model, lora_config)
        print("✅ LoRA configuration applied successfully!")

    # Step 2: 加载 FSDP Checkpoint (来自 sft.py 的完整模型权重)
    # checkpoint_path 应该指向 checkpoint 目录，例如 /path/to/save_dir/0002000/
    # 注意：sft.py 保存的 model.safetensors 包含完整模型（language_model + vit_model + connectors等）

    if config.pretrained.checkpoint_path:
        ckpt_dir = config.pretrained.checkpoint_path
        
        if os.path.isdir(ckpt_dir):
            # 情况 A: checkpoint_path 是目录，查找 FSDP checkpoint 文件
            # sft.py 使用 FSDPCheckpoint.fsdp_save_ckpt 保存，文件名为 model.safetensors
            fsdp_checkpoint = None
            fsdp_candidates = [
                "model.safetensors",
                "ema.safetensors",  # 如果有 EMA 模型
                "pytorch_model.bin", 
                "model.pt",
                "consolidated.safetensors",
            ]
            for c in fsdp_candidates:
                p = os.path.join(ckpt_dir, c)
                if os.path.exists(p):
                    fsdp_checkpoint = p
                    break
            
            if fsdp_checkpoint:
                print(f"🚀 [2/3] Loading FSDP Checkpoint (完整模型) from: {ckpt_dir}")
                if fsdp_checkpoint.endswith('.safetensors'):
                    fsdp_state_dict = load_file(fsdp_checkpoint)
                else:
                    fsdp_state_dict = torch.load(fsdp_checkpoint, map_location='cpu')
                
                
                
                new_state_dict = {}
                key_fixed = False
                lora_keys_found = False
                
                msg = model.load_state_dict(fsdp_state_dict, strict=False)

                
                print(f"    -> FSDP Load result: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
                if len(msg.missing_keys) > 0:
                    print(f"    -> Missing keys (前20个): {msg.missing_keys[:20]}")
                if len(msg.unexpected_keys) > 0:
                    print(f"    -> Unexpected keys (前20个): {msg.unexpected_keys[:20]}")
            else:
                print(f"⚠️ Warning: No FSDP checkpoint found in {ckpt_dir}")
            
            # Step 3: 可选加载单独的 Connectors 文件（如果存在）
            # 注意：sft.py 不单独保存 connectors.pt，connectors 已经在 model.safetensors 中
            # 这个步骤是为了兼容其他训练脚本（如 train_bagel_w_llm.py）可能单独保存 connectors.pt
            connectors_path = os.path.join(ckpt_dir, "connectors.pt")
            if os.path.exists(connectors_path):
                print(f"🚀 [3/3] 检测到单独的 Connectors 文件，加载中: {connectors_path}")
                
                connector_state_dict = torch.load(connectors_path, map_location='cpu')
                
                # 加载 connector 权重到模型（会覆盖 model.safetensors 中的 connectors）
                msg = model.load_state_dict(connector_state_dict, strict=False)
                print(f"    -> Connectors Load result: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
                
                # 验证加载了哪些 connector 参数
                loaded_connectors = [k for k in connector_state_dict.keys()]
                print(f"    -> Loaded connector params: {loaded_connectors[:10]}..." if len(loaded_connectors) > 10 else f"    -> Loaded connector params: {loaded_connectors}")
            else:
                print(f"ℹ️  [3/3] 未找到单独的 connectors.pt（这是正常的，sft.py 不单独保存 connectors）")
        else:
            print(f"⚠️ Warning: Configured checkpoint_path not found: {ckpt_dir}")
    
    # ================== 模型加载完成 ==================

    # 检查并初始化仍然在 meta device 上的参数
    # 这些参数可能是因为 checkpoint 中缺失而没有被加载
    from accelerate.utils import set_module_tensor_to_device
    
    print("🔍 检查并初始化仍然在 meta device 上的参数...")
    meta_params = []
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            meta_params.append(name)
    
    if len(meta_params) > 0:
        print(f"⚠️  发现 {len(meta_params)} 个参数仍在 meta device 上，正在初始化...")
        # 对于缺失的参数，使用 base_checkpoint 中的值（如果存在）
        # 或者使用随机初始化
        base_state_dict = None
        if base_checkpoint:
            try:
                if base_checkpoint.endswith('.safetensors'):
                    # from safetensors.torch import load_file
                    base_state_dict = load_file(base_checkpoint)
                else:
                    base_state_dict = torch.load(base_checkpoint, map_location='cpu')
            except Exception as e:
                print(f"⚠️  无法加载 base_checkpoint 来初始化缺失参数: {e}")
        
        for name in meta_params:
            # 尝试从 base_checkpoint 中获取
            if base_state_dict and name in base_state_dict:
                value = base_state_dict[name]
                print(f"  -> 从 base_checkpoint 初始化: {name}")
            else:
                # 使用随机初始化（使用正确的 dtype）
                param = dict(model.named_parameters())[name]
                value = torch.randn(param.shape, dtype=inference_dtype)
                print(f"  -> 随机初始化: {name}")
            
            # 使用 set_module_tensor_to_device 安全地替换 meta tensor
            set_module_tensor_to_device(
                model,
                name,
                "cpu",  # 先放在 CPU 上
                value=value
            )
        print(f"✅ 已初始化 {len(meta_params)} 个 meta 参数")
    else:
        print("✅ 所有参数都已正确加载，没有 meta tensor")


    model = model.eval()

    # Freeze all parameters
    model.requires_grad_(False)

    inference_hyper = dict(
        cfg_text_scale=1.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=config.train.timestep_shift,
        cfg_renorm_min=0.0,
        cfg_renorm_type=None,
    )

    inferencer = InterleaveInferencer(
        model=model, 
        vae_model=vae_model, 
        tokenizer=tokenizer, 
        vae_transform=vae_transform, 
        vit_transform=vit_transform, 
        new_token_ids=new_token_ids
    )

    # Move models to device
    vae_model.to(accelerator.device, dtype=torch.float32)
    vit_model.to(accelerator.device, dtype=inference_dtype)
    model.to(dtype=inference_dtype)
    
    print(f"[Rank {accelerator.process_index}] Moving models to correct devices...")

    vae_model.to(accelerator.device, dtype=torch.float32)

    for name, module in model.named_children():
        if name != "language_model":
            remove_hook_from_module(module, recurse=True)
            if name == "time_embedder":
                module.to(accelerator.device, dtype=torch.float32)
            else:
                module.to(accelerator.device, dtype=inference_dtype)

    if hasattr(model, "vit_model") and model.vit_model is not None:
        print(f"[Rank {accelerator.process_index}] 🔧 Deep forcing ViT to {accelerator.device}...")
        model.vit_model.to(accelerator.device, dtype=inference_dtype)
        
        try:
            embeddings = model.vit_model.vision_model.embeddings
            if hasattr(embeddings, "patch_embedding"):
                print(f"[Rank {accelerator.process_index}]   -> Fixing patch_embedding layer...")
                embeddings.patch_embedding.to(accelerator.device, dtype=inference_dtype)
                if hasattr(embeddings.patch_embedding, "weight"):
                    embeddings.patch_embedding.weight.data = embeddings.patch_embedding.weight.data.to(accelerator.device)
                if hasattr(embeddings.patch_embedding, "bias") and embeddings.patch_embedding.bias is not None:
                    embeddings.patch_embedding.bias.data = embeddings.patch_embedding.bias.data.to(accelerator.device)
        except AttributeError as e:
            print(f"Warning: Could not access patch_embedding directly: {e}")

        for param in model.vit_model.parameters():
            if param.device.type == 'cpu':
                param.data = param.data.to(accelerator.device)

        try:
            check_layer = model.vit_model.vision_model.embeddings.patch_embedding
            print(f"[Rank {accelerator.process_index}] ✅ Check patch_embedding device: {check_layer.weight.device}")
            if check_layer.weight.device.type == 'cpu':
                raise RuntimeError("CRITICAL: patch_embedding is STILL on CPU!")
        except:
            pass

    if hasattr(model, "time_embedder"):
        def auto_gpu_hook(module, args):
            return tuple(a.to(accelerator.device) if isinstance(a, torch.Tensor) else a for a in args)
        model.time_embedder.register_forward_pre_hook(auto_gpu_hook)

    for name, param in model.named_parameters(recurse=False):
        setattr(model, name, torch.nn.Parameter(param.data.to(accelerator.device, dtype=inference_dtype)))
    
    # 彻底清理 GPU 内存，为 language_model 移动做准备
    import gc
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    # Move language model to device
    # model.language_model.to(accelerator.device, dtype=inference_dtype)
    
    transformer = model.language_model
    # 使用 no_grad 和更安全的方式移动，避免创建临时副本
    with torch.no_grad():
        transformer = transformer.to(accelerator.device, dtype=inference_dtype)
    transformer = accelerator.prepare(transformer)
    model.language_model = transformer

    # model.language_model = accelerator.prepare(model.language_model)

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.prompt_fn == "maze":
        # Get filter_size_min, filter_size_max, samples_per_size, and filter_shape from config
        filter_size_min = getattr(config.sample, 'filter_size_min', None)
        filter_size_max = getattr(config.sample, 'filter_size_max', None)
        samples_per_size = getattr(config.sample, 'samples_per_size', None)
        filter_shape = getattr(config.sample, 'filter_shape', None)
        test_dataset = MazePromptImageDataset(
            config.dataset, 
            config.dataset_split,
            filter_size_min=filter_size_min,
            filter_size_max=filter_size_max,
            samples_per_size=samples_per_size,
            filter_shape=filter_shape,
            resolution=config.sample.resolution
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=MazePromptImageDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    else:
        raise NotImplementedError(f"Unsupported prompt_fn: {config.prompt_fn}")

    autocast = accelerator.autocast
    eval_reward_fn = maze_metrics.maze_metric(accelerator.device)
    executor = futures.ThreadPoolExecutor(max_workers=8)

    test_dataloader = accelerator.prepare(test_dataloader)

    # Create output directory for saving images
    output_dir = os.path.join(config.logdir, f"{config.run_name}_{config.sample.eval_num_steps}", "generated_images")
    
    logger.info("***** Running Evaluation *****")
    logger.info(f"  Test dataset size = {len(test_dataset)}")
    test_batch_size_per_size = getattr(config.sample, 'test_batch_size_per_size', None)
    if test_batch_size_per_size is not None:
        logger.info(f"  Test batch size per size = {test_batch_size_per_size} (用于每个尺寸内的分批，避免OOM)")
        logger.info(f"  Test batch size (默认) = {config.sample.test_batch_size} (未使用)")
    else:
        logger.info(f"  Test batch size = {config.sample.test_batch_size}")
    if filter_shape is not None:
        logger.info(f"  Filter shape = {filter_shape}")
    else:
        logger.info(f"  Filter shape = All shapes")
    if filter_size_min is not None or filter_size_max is not None:
        filter_str = f"[{filter_size_min if filter_size_min else 'any'}, {filter_size_max if filter_size_max else 'any'}]"
        logger.info(f"  Filter size range = {filter_str}")
    else:
        logger.info(f"  Filter size = All sizes")
    if samples_per_size is not None:
        logger.info(f"  Samples per size = {samples_per_size}")
    else:
        logger.info(f"  Samples per size = All samples")
    logger.info(f"  Num attempts per sample = {config.sample.num_attempts}")
    logger.info(f"  Output directory = {output_dir}")

    # Run evaluation
    all_rewards = eval(
        inferencer, 
        inference_hyper, 
        test_dataloader, 
        tokenizer, 
        config, 
        accelerator, 
        eval_reward_fn, 
        executor, 
        autocast,
        output_dir=output_dir,
        num_attempts=config.sample.num_attempts  # Can be changed for multiple attempts
    )

    # Print final results
    if accelerator.is_main_process:
        print("\n" + "="*80)
        print("📊 Evaluation Results:")
        print("="*80)
        for key, value in all_rewards.items():
            valid_values = value[value != -10]
            if len(valid_values) > 0:
                print(f"  {key}: {np.mean(valid_values):.4f}")
        print("="*80)
        print(f"\n✅ Images saved to: {output_dir}")


if __name__ == "__main__":
    app.run(main)
