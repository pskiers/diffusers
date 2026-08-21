"""
CUDA_VISIBLE_DEVICES=0 python infer_new.py \
    --checkpoint_path /root/private_data/janus_outputs/janus_train_hexagon/hexagon_1/checkpoint-8-1703/tfmr \
    --data_path /root/private_data/hexagon/maze-dataset \
    --split test \
    --output_dir ./inference_results \
    --batch_size 16 \
    --temperature 1.0 \
    --num_attempts 5 \
    --filter_size_min 5 \
    --filter_size_max 16 \
    --samples_per_size 50 \
    --filter_shape hexagon
"""

import os
import json
import torch
import argparse
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

import numpy as np
import PIL.Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM

from janus.models import VLChatProcessor
from data.maze_dataset import MazeDataset
from maze_rewards import create_maze_reward_function

logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

@dataclass
class VLChatProcessorOutput():
    sft_format: str
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    num_image_tokens: torch.IntTensor

    def __len__(self):
        return len(self.input_ids)


def process_image_for_vq(processor, images):
    """
    处理图像用于VQ编码，与sft.py中的process_image逻辑一致
    """
    if not images:
        return None
    
    # 确保是列表
    if not isinstance(images, list):
        images = [images]
        
    # 加载图像
    loaded_images = []
    for img in images:
        if isinstance(img, str):
            img = PIL.Image.open(img).convert("RGB")
        elif isinstance(img, PIL.Image.Image):
            img = img.convert("RGB")
        loaded_images.append(img)
        
    # 使用 processor 处理 (Resize + Normalize)
    # 这会返回标准的 pixel_values
    images_outputs = processor.image_processor(loaded_images, return_tensors="pt")
    return images_outputs['pixel_values']


@torch.inference_mode()
def generate_image_batch(
    model,
    processor: VLChatProcessor,
    prompts: List[str],
    input_images: List[PIL.Image.Image] = None,
    temperature: float = 1.0,
    image_token_num_per_image: int = None,
    img_size: int = 384,
    patch_size: int = 16,
    device: torch.device = None,
):
    """
    批量生成图像 (VQ 输入模式)
    
    Args:
        model: 模型
        processor: VLChatProcessor
        prompts: 文本提示列表
        input_images: 输入图像列表（可选，每个样本一张图像，可以是 None）
        temperature: 生成温度
        image_token_num_per_image: 每张图像的token数量
        img_size: 图像尺寸
        patch_size: patch大小
        device: 设备
    
    Returns:
        dec: numpy array, shape (batch_size, H, W, C)，值范围 [0, 255]
    """
    if device is None:
        device = next(model.parameters()).device
    
    batch_size = len(prompts)
    
    # 使用 processor 的 num_image_tokens，确保与训练时一致 (384x384 = 576 tokens)
    if image_token_num_per_image is None:
        image_token_num_per_image = processor.num_image_tokens
    
    # 处理 input_images（如果为 None，创建 None 列表）
    if input_images is None:
        input_images = [None] * batch_size
    assert len(input_images) == batch_size, f"input_images length ({len(input_images)}) != prompts length ({batch_size})"
    
    # =====================================================
    # 1. 准备 Input Image 的 VQ Embeddings (Condition)
    # =====================================================
    input_image_embeds_list = []
    has_input_images = any(img is not None for img in input_images)
    
    if has_input_images:
        # 收集所有非 None 的图像及其索引
        valid_images = []
        valid_indices = []
        for i, img in enumerate(input_images):
            if img is not None:
                valid_images.append(img)
                valid_indices.append(i)
        
        if valid_images:
            # 预处理图像 (B_valid, C, H, W)
            pixel_values = process_image_for_vq(processor, valid_images)
            pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)
            
            # 使用 VQ Model 编码得到离散 Tokens
            _, _, info = model.gen_vision_model.encode(pixel_values)
            input_image_tokens = info[2].detach().reshape(pixel_values.shape[0], -1)
            
            # 将 Tokens 转换为 Embeddings
            input_image_embeds_valid = model.prepare_gen_img_embeds(input_image_tokens)  # (B_valid, num_image_tokens, Dim)
            
            # 为所有样本创建 embeddings 列表（None 的用 None 占位）
            input_image_embeds_list = [None] * batch_size
            for idx, valid_idx in enumerate(valid_indices):
                input_image_embeds_list[valid_idx] = input_image_embeds_valid[idx]

    # =====================================================
    # 2. 构建 Prompts (与 SftDataset.collate_fn 对齐)
    # =====================================================
    # 格式: User: <image_start><pad*num_image_tokens><image_end>\n{prompt} Assistant: <image_start>
    
    # 定义 Token 字符串部分
    image_token_str = processor.image_start_tag + \
                      processor.pad_tag * image_token_num_per_image + \
                      processor.image_end_tag
    
    # 为每个样本构建 input_ids
    pre_data = []
    for i, prompt in enumerate(prompts):
        # 如果有输入图像，添加图片占位符
        if input_images[i] is not None:
            user_content = image_token_str + "\n" + prompt
        else:
            user_content = prompt
        
        conversation = [
            {"role": "<|User|>", "content": user_content},
            {"role": "<|Assistant|>", "content": ""}
        ]
        
        # 应用对话模板
        sft_format = processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=processor.sft_format,
            system_prompt="",
        )
        
        # 拼接生成开始符（推理时只需要 start tag）
        sft_format = sft_format + processor.image_start_tag
        
        # Tokenize
        input_ids = torch.LongTensor(processor.tokenizer.encode(sft_format))
        
        pre_data.append(VLChatProcessorOutput(
            sft_format=sft_format,
            pixel_values=None,  # 不使用 pixel_values，我们手动处理
            input_ids=input_ids,
            num_image_tokens=torch.IntTensor([])  # 手动处理，这里不设置
        ))
    
    # 使用 processor.batchify 进行批量处理（处理 padding）
    prepare_inputs = processor.batchify(pre_data)
    
    # =====================================================
    # 3. 构建并注入 Embeddings
    # =====================================================
    # 移动到设备
    input_ids = prepare_inputs.input_ids.to(device)  # (B, max_seq_len)
    attention_mask = prepare_inputs.attention_mask.to(device)  # (B, max_seq_len)
    
    # 获取基础文本 Embeddings
    inputs_embeds = model.language_model.get_input_embeddings()(input_ids)  # (B, max_seq_len, Dim)
    
    # 如果有输入图像，注入 VQ Embeddings
    if has_input_images and input_image_embeds_list:
        image_start_id = processor.image_start_id
        
        for i in range(batch_size):
            if input_image_embeds_list[i] is not None:
                # 找到当前样本的 <image_start> 位置
                sample_input_ids = input_ids[i]
                start_indices = (sample_input_ids == image_start_id).nonzero(as_tuple=True)[0]
                
                if len(start_indices) >= 1:
                    input_start = start_indices[0].item() + 1
                    input_end = input_start + image_token_num_per_image
                    
                    # 检查长度是否越界
                    if input_end <= inputs_embeds.shape[1]:
                        # 注入 Input Image VQ Embeddings
                        inputs_embeds[i, input_start:input_end, :] = input_image_embeds_list[i]
                    else:
                        logger.warning(f"Sample {i}: Prompt length mismatch, cannot inject input image embeddings.")

    # =====================================================
    # 4. 批量生成循环 (Autoregressive Generation)
    # =====================================================
    generated_tokens = torch.zeros((batch_size, image_token_num_per_image), dtype=torch.int).to(device)
    
    outputs = None
    
    for step in range(image_token_num_per_image):
        # 使用 past_key_values 时，第一次传入完整序列，后续只传入新生成的 token embedding
        if step == 0:
            # 第一次：传入完整的 inputs_embeds
            step_inputs_embeds = inputs_embeds
            step_attention_mask = attention_mask
        else:
            # 后续步骤：只传入新生成的 token embedding
            step_inputs_embeds = img_embeds  # (B, 1, Dim)
            step_attention_mask = torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
        
        outputs = model.language_model.model(
            inputs_embeds=step_inputs_embeds,
            use_cache=True,
            attention_mask=step_attention_mask,
            past_key_values=outputs.past_key_values if step > 0 else None
        )
        hidden_states = outputs.last_hidden_state  # (B, seq_len, Dim)，第一次是完整序列，后续是1
        
        # 使用最后一个位置的 hidden state 进行预测
        logits = model.gen_head(hidden_states[:, -1, :])  # (B, vocab_size)
        probs = torch.softmax(logits / temperature, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)
        generated_tokens[:, step] = next_tokens
        
        # 将生成的 tokens 转换为 embeddings 作为下一步的输入
        img_embeds = model.prepare_gen_img_embeds(next_tokens)  # (B, Dim)
        img_embeds = img_embeds.unsqueeze(1)  # (B, 1, Dim)

    # =====================================================
    # 5. 批量解码生成的 Tokens
    # =====================================================
    dec = model.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[batch_size, 8, img_size // patch_size, img_size // patch_size]
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
    
    return dec


# 保持向后兼容的单个样本接口
@torch.inference_mode()
def generate_image(
    model,
    processor: VLChatProcessor,
    prompt: str,
    input_images: List[PIL.Image.Image] = None,
    temperature: float = 1.0,
    parallel_size: int = 1,
    cfg_weight: float = 5.0,
    image_token_num_per_image: int = None,
    img_size: int = 384,
    patch_size: int = 16,
    device: torch.device = None,
):
    """
    生成图像 (VQ 输入模式) - 单个样本接口（向后兼容）
    """
    # 转换为列表，调用批量接口
    prompts = [prompt]
    if input_images is None:
        input_images_batch = None
    elif isinstance(input_images, list):
        input_images_batch = input_images if len(input_images) > 0 else [None]
    else:
        input_images_batch = [input_images]
    
    results = generate_image_batch(
        model=model,
        processor=processor,
        prompts=prompts,
        input_images=input_images_batch,
        temperature=temperature,
        image_token_num_per_image=image_token_num_per_image,
        img_size=img_size,
        patch_size=patch_size,
        device=device,
    )
    
    return results  # 返回 (1, H, W, C)


class InferenceDataset(Dataset):
    """推理数据集"""
    def __init__(
        self, 
        data_path: str,
        split: str = 'train',
        filter_size_min: int = None,
        filter_size_max: int = None,
        samples_per_size: int = None,
        filter_shape: str = None
    ):
        """
        Args:
            data_path: 数据集路径
            split: 'train' 或 'test'
            filter_size_min: 可选，最小迷宫尺寸（例如：5 表示 5x5）
            filter_size_max: 可选，最大迷宫尺寸（例如：10 表示 10x10）
            samples_per_size: 可选，每个尺寸选择的样本数（例如：3 表示每个尺寸选择3个样本）
            filter_shape: 可选，迷宫形状类型过滤（例如：'triangle', 'square', 'hexagon'）
        """
        # 兼容文件路径或目录路径
        if os.path.isfile(data_path):
            data_dir = os.path.dirname(data_path)
        else:
            data_dir = data_path
            
        self.maze_dataset = MazeDataset(data_dir, split=split)
        self.filter_size_min = filter_size_min
        self.filter_size_max = filter_size_max
        self.samples_per_size = samples_per_size
        self.filter_shape = filter_shape
        
        # 检查是否是圆形数据集（按layers分组）
        self.is_circle_dataset = data_path and 'circle/maze-dataset' in data_path
        
        # 构建过滤后的索引列表
        if filter_size_min is not None or filter_size_max is not None or samples_per_size is not None or filter_shape is not None:
            if filter_shape is not None:
                logger.info(f"🔍 Filtering dataset by shape: {filter_shape}")
            if filter_size_min is not None and filter_size_max is not None:
                logger.info(f"🔍 Filtering dataset by size range: [{filter_size_min}, {filter_size_max}]")
            elif filter_size_min is not None:
                logger.info(f"🔍 Filtering dataset by minimum size: >= {filter_size_min}")
            elif filter_size_max is not None:
                logger.info(f"🔍 Filtering dataset by maximum size: <= {filter_size_max}")
            if samples_per_size is not None:
                logger.info(f"🔍 Selecting {samples_per_size} samples per size")
            self.filtered_indices = self._build_filtered_indices()
            logger.info(f"   Found {len(self.filtered_indices)} samples matching filter criteria")
        else:
            self.filtered_indices = list(range(len(self.maze_dataset)))
        
        logger.info(f'Dataset loaded from {data_dir}, total size: {len(self.maze_dataset)}, filtered size: {len(self.filtered_indices)}')
    
    def _build_filtered_indices(self):
        """构建符合过滤条件的索引列表"""
        from collections import defaultdict
        
        # 第一遍：按尺寸分类样本
        samples_by_size = {}
        
        for idx in range(len(self.maze_dataset)):
            maze_item = self.maze_dataset[idx]
            maze_config = self._extract_maze_config(maze_item)
            shape = maze_config.get('shape', '')
            
            # 检查形状是否匹配 filter_shape（如果指定）
            if self.filter_shape is not None:
                if shape != self.filter_shape:
                    continue
            
            # 圆形数据集按layers分组，其他数据集按width/height分组
            if self.is_circle_dataset:
                layers = maze_config.get('layers', 0)
                # 对于圆形数据集，可以按layers过滤（如果指定了filter_size_min/max，将其视为layers范围）
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
                # 非圆形数据集：按width/height分组
                width = maze_config.get('width', 0)
                height = maze_config.get('height', 0)
                
                # 只包含正方形迷宫（width == height）
                if width != height:
                    continue
                
                # 检查尺寸是否在指定范围内
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
        
        # 第二遍：选择样本（全部或每个尺寸限制数量）
        filtered = []
        if self.samples_per_size is not None:
            # 每个尺寸选择前N个样本
            for size_key in sorted(samples_by_size.keys()):
                available = samples_by_size[size_key]
                selected = available[:self.samples_per_size]
                filtered.extend(selected)
                logger.info(f"   {size_key}: selected {len(selected)}/{len(available)} samples")
        else:
            # 选择所有样本
            for size_key in sorted(samples_by_size.keys()):
                filtered.extend(samples_by_size[size_key])
        
        return filtered
    
    def _extract_maze_config(self, maze_item):
        """从 maze_item 的 metadata 中提取 maze_config"""
        maze_config = {}
        metadata = maze_item.get("metadata", {})
        
        # 尝试解析嵌套的 metadata JSON 字符串
        if 'metadata' in metadata:
            try:
                if isinstance(metadata['metadata'], str):
                    parsed_meta = json.loads(metadata['metadata'])
                    maze_config = parsed_meta.get('maze_config', {})
                elif isinstance(metadata['metadata'], dict):
                    maze_config = metadata['metadata'].get('maze_config', {})
            except (json.JSONDecodeError, TypeError):
                pass
        
        # 回退：尝试直接从 metadata 获取
        if not maze_config:
            maze_config = metadata.get('maze_config', {})
        
        return maze_config
    
    def __len__(self) -> int:
        return len(self.filtered_indices)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        # 将过滤后的索引映射到原始数据集索引
        original_idx = self.filtered_indices[index]
        maze_item = self.maze_dataset[original_idx]
        return {
            'id': maze_item['id'],
            'prompt': maze_item['prompt'],
            'input_image': maze_item.get('m_original_img'),  # 使用带标记的迷宫作为输入
            'ground_truth': maze_item.get('sol_img'), 
            'metadata': maze_item.get('metadata', {}),
            # 为reward计算保留完整的图像信息
            'original_img': maze_item.get('original_img'),
            'sol_img': maze_item.get('sol_img'),
            'm_original_img': maze_item.get('m_original_img'),
            'mask_img': maze_item.get('mask_img'),
            'cell_map': maze_item.get('cell_map'),
        }


def inference_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """处理包含PIL.Image对象的batch"""
    if len(batch) == 1:
        return batch[0]
    
    collated = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        # 保持 PIL Image 为列表，不转换为 Tensor，因为 process_image_for_vq 需要列表
        collated[key] = values
    return collated


def main(args: argparse.Namespace):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # 加载模型
    checkpoint_path = args.checkpoint_path
    if not os.path.exists(checkpoint_path):
        raise ValueError(f'Checkpoint path does not exist: {checkpoint_path}')
    
    logger.info(f'Loading model from: {checkpoint_path}')
    processor = VLChatProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    
    # 图像分辨率：384x384
    # processor 的 num_image_tokens 应该已经是 576 (384/16 = 24, 24*24 = 576)
    # generate_image 函数会自动使用 processor.num_image_tokens
    logger.info(f'Using processor.num_image_tokens: {processor.num_image_tokens} (should be 576 for 384x384 images)')
    
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    model = model.to(device).eval()
    logger.info('Model loaded successfully')
    
    # 初始化 Reward Function
    try:
        reward_fn = create_maze_reward_function()
        logger.info('Reward function initialized')
    except Exception as e:
        logger.warning(f"Could not initialize reward function: {e}. Rewards will be 0.")
        reward_fn = None

    # 加载数据
    dataset = InferenceDataset(
        args.data_path,
        split=args.split,
        filter_size_min=args.filter_size_min,
        filter_size_max=args.filter_size_max,
        samples_per_size=args.samples_per_size,
        filter_shape=args.filter_shape
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    
    logger.info(f'Starting inference on {len(dataset)} samples...')
    
    # ========== 按尺寸分组，然后创建batch ==========
    from collections import defaultdict
    
    # 使用dataset的is_circle_dataset属性（与InferenceDataset中的检测逻辑一致）
    is_circle_dataset = dataset.is_circle_dataset
    
    # 先按尺寸分组样本
    samples_by_size = defaultdict(list)
    indices = list(range(len(dataset)))
    
    for idx in indices:
        sample = dataset[idx]
        metadata = sample.get("metadata", {})
        maze_config = {}
        
        # 解析 maze_config
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
            if layers == 0:
                # 如果没有layers信息，记录警告但继续处理
                print("layers=0，继续处理")
                logger.warning(f"Sample {idx} (id: {sample.get('id', 'unknown')}): No layers found in maze_config, using layers_0")
            size_key = f"layers_{layers}"
        else:
            width = maze_config.get('width', 0)
            height = maze_config.get('height', 0)
            if width == 0 or height == 0:
                # 如果没有width/height信息，记录警告但继续处理
                logger.warning(f"Sample {idx} (id: {sample.get('id', 'unknown')}): No width/height found in maze_config, using 0x0")
            size_key = f"{width}x{height}"
        samples_by_size[size_key].append(idx)
    
    logger.info(f"\n{'='*60}")
    if is_circle_dataset:
        logger.info(f"📊 按layers分组结果（圆形数据集）:")
    else:
        logger.info(f"📊 按尺寸分组结果:")
    total_samples_in_groups = 0
    for size_key in sorted(samples_by_size.keys()):
        count = len(samples_by_size[size_key])
        total_samples_in_groups += count
        logger.info(f"   {size_key}: {count} 个样本")
    logger.info(f"   总计: {total_samples_in_groups} 个样本")
    if total_samples_in_groups == 0:
        logger.warning(f"⚠️  警告：没有找到任何样本！请检查数据集路径和过滤条件。")
    logger.info(f"{'='*60}\n")
    
    # 对每个尺寸组分别创建batch
    eval_batches = []
    # 如果设置了 batch_size_per_size，使用它；否则使用 batch_size
    bs = args.batch_size_per_size if args.batch_size_per_size is not None else args.batch_size
    logger.info(f"📦 每个尺寸内的batch大小: {bs} (batch_size_per_size={args.batch_size_per_size}, batch_size={args.batch_size})")
    
    for size_key in sorted(samples_by_size.keys()):
        size_indices = samples_by_size[size_key]
        for i in range(0, len(size_indices), bs):
            batch_indices = size_indices[i : i + bs]
            batch_samples = [dataset[idx] for idx in batch_indices]
            eval_batches.append((batch_indices, batch_samples, size_key))
    # ====================================================
    
    for batch_idx, (batch_indices, batch_samples, size_key) in enumerate(tqdm(eval_batches, desc='Inference')):
        # batch_samples 已经是样本列表
        batch_list = batch_samples
        batch_size = len(batch_list)
        
        # 解析所有样本的 metadata 和准备数据
        sample_ids = []
        prompts = []
        input_images_list = []
        metadata_list = []
        maze_configs = []  # (width, height)
        
        for sample in batch_list:
            sample_ids.append(sample['id'])
            prompts.append(sample['prompt'])
            input_images_list.append(sample.get('input_image'))
            
            # 解析 metadata 获取尺寸
            metadata = sample.get('metadata', {})
            maze_width, maze_height = 3, 3  # 默认值
            try:
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                elif isinstance(metadata, dict) and 'metadata' in metadata and isinstance(metadata['metadata'], str):
                    metadata = json.loads(metadata['metadata'])
                
                if isinstance(metadata, dict) and 'maze_config' in metadata:
                    maze_width = int(metadata['maze_config'].get('width', 3))
                    maze_height = int(metadata['maze_config'].get('height', 3))
            except Exception:
                pass
            
            maze_configs.append((maze_width, maze_height))
            metadata_list.append(metadata)
        
        # 使用 size_key 作为文件名前缀（圆形数据集使用 layers，其他使用 width×height）
        if is_circle_dataset:
            # 从 size_key 提取 layers
            filename_prefix = size_key  # "layers_X"
        else:
            # 从 size_key 提取 width 和 height
            filename_prefix = size_key.replace('x', '×')  # "width×height"
        
        # 保存 Ground Truth 和输入图像（每个样本只保存一次）
        for i, sample in enumerate(batch_list):
            sample_id = sample['id']
            # 保存 Ground Truth
            gt_path = os.path.join(args.output_dir, f"{filename_prefix}_{sample_id}_gt.png")
            if sample.get('ground_truth') and not os.path.exists(gt_path):
                sample['ground_truth'].save(gt_path)
            
            # 保存输入图像
            input_path = os.path.join(args.output_dir, f"{filename_prefix}_{sample_id}_input.png")
            if sample.get('input_image') and not os.path.exists(input_path):
                sample['input_image'].save(input_path)
        
        # 多次尝试生成（每次批量生成）
        for attempt in range(args.num_attempts):
            try:
                # 批量生成图像
                generated_images_np = generate_image_batch(
                    model=model,
                    processor=processor,
                    prompts=prompts,
                    input_images=input_images_list,
                    temperature=args.temperature,
                    device=device
                )
                # generated_images_np: (batch_size, H, W, C)
                
                # 处理每个样本的结果
                for i in range(batch_size):
                    sample_id = sample_ids[i]
                    prompt = prompts[i]
                    maze_width, maze_height = maze_configs[i]
                    metadata = metadata_list[i]
                    sample = batch_list[i]
                    
                    # 提取生成的图像
                    gen_img = PIL.Image.fromarray(generated_images_np[i])
                    
                    # 保存生成图（使用 filename_prefix）
                    output_filename = f"{filename_prefix}_{sample_id}_attempt{attempt+1:03d}.png"
                    output_path = os.path.join(args.output_dir, output_filename)
                    gen_img.save(output_path)
                    
                    # 计算 Reward
                    rewards_dict = {}
                    if reward_fn:
                        # 转换 image 为 Tensor (C, H, W) [0,1]
                        gen_tensor = torch.from_numpy(generated_images_np[i].transpose(2, 0, 1)).float() / 255.0
                        
                        reward_metadata = {
                            'original_img': sample.get('original_img'),
                            'm_original_img': sample.get('m_original_img'),
                            'sol_img': sample.get('sol_img'),
                            'mask_img': sample.get('mask_img'),
                            'cell_map': sample.get('cell_map'),
                            'metadata': metadata
                        }
                        
                        try:

                            rewards_dict_batch, reward_metadata_batch = reward_fn(
                                images=gen_tensor.unsqueeze(0).cpu(),
                                prompts=[prompt],
                                metadata=[reward_metadata],
                                only_strict=False
                            )
                            
                            # 提取单个样本的reward值（因为是batch，取第一个）
                            rewards_dict = {
                                'mse_inside': float(rewards_dict_batch['mse_inside'][0]) if 'mse_inside' in rewards_dict_batch else 0.0,
                                'mse_outside': float(rewards_dict_batch['mse_outside'][0]) if 'mse_outside' in rewards_dict_batch else 0.0,
                                'mse_solution': float(rewards_dict_batch['mse_solution'][0]) if 'mse_solution' in rewards_dict_batch else 0.0,
                                'path_validity': float(rewards_dict_batch['path_validity'][0]) if 'path_validity' in rewards_dict_batch else 0.0,
                                'gt_cell_coverage': float(rewards_dict_batch['gt_cell_coverage'][0]) if 'gt_cell_coverage' in rewards_dict_batch else 0.0,
                                'background_violation': float(rewards_dict_batch['background_violation'][0]) if 'background_violation' in rewards_dict_batch else 0.0,
                                'avg': float(rewards_dict_batch['avg'][0]) if 'avg' in rewards_dict_batch else 0.0,
                            }
                            
                            # 计算maze_reward（根据用户示例，使用avg）
                            rewards_dict['maze_reward'] = rewards_dict['gt_cell_coverage']-rewards_dict['background_violation']
                            
                        except Exception as e:
                            logger.error(f'Error computing rewards for sample {sample_id}, attempt {attempt+1}: {str(e)}')
                            import traceback
                            traceback.print_exc()
                            # 使用默认值
                            rewards_dict = {
                                'mse_inside': 0.0,
                                'mse_outside': 0.0,
                                'mse_solution': 0.0,
                                'path_validity': 0.0,
                                'gt_cell_coverage': 0.0,
                                'background_violation': 0.0,
                                'maze_reward': 0.0,
                                'avg': 0.0,
                            }
                        if is_circle_dataset:
                            layers = filename_prefix.split('_')[1]
                        # 保存结果信息（按照用户要求的格式）
                            result = {
                                'id': sample_id,
                                'attempt': attempt + 1,
                                'layers': layers,
                                'rewards': rewards_dict
                            }
                        else:
                            result = {
                                'id': sample_id,
                                'attempt': attempt + 1,
                                'width': maze_width,
                                'height': maze_height,
                                'rewards': rewards_dict
                            }
            #                 r_dict, _ = reward_fn(
            #                     images=gen_tensor.unsqueeze(0).cpu(),
            #                     prompts=[prompt],
            #                     metadata=[reward_metadata],
            #                     only_strict=False
            #                 )
            #                 # 提取第一个样本的指标
            #                 rewards_dict = {k: float(v[0]) for k, v in r_dict.items() if isinstance(v, (list, torch.Tensor))}
            #                 rewards_dict['maze_reward'] = rewards_dict.get('avg', 0.0)
            #             except Exception as e:
            #                 logger.error(f"Reward calculation failed for {sample_id}: {e}")
                    
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx} attempt {attempt}: {e}")
                import traceback
                traceback.print_exc()
            
            torch.cuda.empty_cache()

    # 保存 JSON 结果
    with open(os.path.join(args.output_dir, 'inference_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./inference_results')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'], help='数据集split')
    parser.add_argument('--batch_size', type=int, default=1, help='默认batch大小')
    parser.add_argument('--batch_size_per_size', type=int, default=None, help='每个尺寸内的batch大小（如果设置，会覆盖batch_size）')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--num_attempts', type=int, default=1, help='每个样本的生成尝试次数')
    # 尺寸过滤参数
    parser.add_argument('--filter_size_min', type=int, default=None, help='最小迷宫尺寸（例如：5 表示 5x5）')
    parser.add_argument('--filter_size_max', type=int, default=None, help='最大迷宫尺寸（例如：10 表示 10x10）')
    parser.add_argument('--samples_per_size', type=int, default=None, help='每个尺寸选择的样本数（例如：3 表示每个尺寸选择3个样本）')
    parser.add_argument('--filter_shape', type=str, default=None, help='迷宫形状类型过滤（例如：triangle, square, hexagon）')
    
    args = parser.parse_args()
    main(args)