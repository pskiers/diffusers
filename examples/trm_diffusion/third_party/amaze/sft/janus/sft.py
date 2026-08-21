"""
accelerate launch sft.py \
    --model_path /mnt/data/zhaoyanpeng/model/huggingface/hub/models--deepseek-ai--Janus-Pro-7B/snapshots/5c3eb3fb2a3b61094328465ba61fcd4272090d67 \
    --data_path /home/zhaoyanpeng/Documents/maze_dataset/triangle/maze-dataset_train \
    --output_dir ./outputs/triangle \
    --experiment_name janus_train_triangle \
    --run_name triangle_1 \
    --train_bsz_per_gpu 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --n_epochs 200 \
    --warmup_rates 0.05 \
    --min_lr_ratio 0.01 \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --max_ckpts 10 \
    --log_dir ./train_logs \
    --seed 42

"""

import os
import json
import torch
import logging
import argparse
import random
import shutil
from typing import List, Dict, Any
from dataclasses import dataclass

import wandb
from tqdm import tqdm
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from transformers import (
    set_seed,
)

from janus.models import VLChatProcessor
from transformers import AutoModelForCausalLM
import PIL.Image
from data.maze_dataset import MazeDataset

from torch.optim.lr_scheduler import LambdaLR
import math

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

def get_custom_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    min_lr_ratio=0.0, 
    num_cycles=0.5
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * 2 * num_cycles * progress))
        scaled_factor = (1 - min_lr_ratio) * cosine_factor + min_lr_ratio
        return scaled_factor

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)

def get_learning_rate(step, initial_lr, num_warmup_steps, num_training_steps, min_lr_ratio, num_cycles=0.5):
    if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps)) * initial_lr
    progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * 2 * num_cycles * progress))
    scaled_factor = (1 - min_lr_ratio) * cosine_factor + min_lr_ratio
    return scaled_factor * initial_lr

class TrainingMetrics:
    def __init__(self, device):
        self.n_step = 0
        self.right = torch.Tensor([0]).to(device=device)
        self.total = torch.Tensor([0]).to(device=device)
        self.total_loss = torch.Tensor([0]).to(device=device)
        self.total_mse = torch.Tensor([0]).to(device=device)
        self.world_size = dist.get_world_size()

    def __call__(self, logits, labels, loss, mse=None):
        return self.update(logits, labels, loss, mse)

    def update(self, logits, labels, loss, mse=None):
        self.n_step += 1
        with torch.no_grad():
            preds = logits.argmax(dim=-1) # [B, 576]
            # 直接对比，因为输入进来的已经是完全对齐的 576 个位置
            self.right += (preds == labels).masked_fill(labels.eq(-100), 0).sum().item()
            self.total += (labels != -100).sum().item()
            self.total_loss += loss.item()
            if mse is not None:
                self.total_mse += mse.item()

    def get_metric(self, reset=True):
        dist.all_reduce(self.right, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total_loss, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total_mse, op=torch.distributed.ReduceOp.SUM)

        acc = (self.right / self.total).item()
        loss = self.total_loss.item() / (self.world_size * self.n_step)
        mse = self.total_mse.item() / (self.world_size * self.n_step)

        if reset:
            self.n_step = 0
            self.right.fill_(0)
            self.total.fill_(0)
            self.total_loss.fill_(0)
            self.total_mse.fill_(0)
        return acc, loss, mse

class SftDataset(Dataset):
    def __init__(self, config, processor,accelerator, model):
        self.config = config
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.accelerator = accelerator
        
        # 使用 MazeDataset 加载数据
        # data_path 应该是包含 maze_dataset_train.parquet 的目录路径
        dataset_path = config.data_path
        # 如果是文件路径，使用其目录
        if os.path.isfile(dataset_path):
            dataset_path = os.path.dirname(dataset_path)
        split = getattr(config, 'split', 'train')
        self.maze_dataset = MazeDataset(dataset_path, split=split)
        accelerator.print(f'Total data amount: {len(self.maze_dataset)}')
        
        # 动态检测真实的图像token数量（只在主进程执行，避免分布式环境下的问题）
        self.image_len = 576  # 默认值
        # if len(self.maze_dataset) > 0 and accelerator.is_main_process:
        #     try:
        #         sample_item = self.maze_dataset[0]
        #         sample_image = sample_item['sol_img']
        #         if isinstance(sample_image, PIL.Image.Image):
        #             # 处理单张图像（内联process_image的逻辑）
        #             images = [sample_image.convert("RGB") if isinstance(sample_image, PIL.Image.Image) else sample_image]
        #             images_outputs = self.processor.image_processor(images, return_tensors="pt")
        #             pixel_values = images_outputs['pixel_values']
        #             # 转换为bfloat16并移动到正确的设备（与训练时保持一致）
        #             device = next(model.parameters()).device
        #             pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)
        #             with torch.no_grad():
        #                 quant, emb_loss, info = model.gen_vision_model.encode(pixel_values)
        #                 sample_tokens = info[2].detach().reshape(pixel_values.shape[0], -1)
        #                 self.image_len = sample_tokens.shape[1]  # 真实的token数量
        #                 accelerator.print(f'Detected image token count: {self.image_len}')
        #     except Exception as e:
        #         accelerator.print(f'Warning: Failed to detect image token count, using default 576. Error: {e}')
        
        # 在分布式环境中同步image_len（所有进程使用相同的值）
        # if dist.is_initialized() and dist.get_world_size() > 1:
        #     try:
        #         # 将image_len转换为tensor进行broadcast
        #         device = next(model.parameters()).device if hasattr(model, 'parameters') and len(list(model.parameters())) > 0 else torch.device('cuda:0')
        #         image_len_tensor = torch.tensor([self.image_len], dtype=torch.int).to(device)
        #         dist.broadcast(image_len_tensor, src=0)
        #         self.image_len = image_len_tensor.item()
        #         if not accelerator.is_main_process:
        #             accelerator.print(f'Received image token count from main process: {self.image_len}')
        #     except Exception as e:
        #         accelerator.print(f'Warning: Failed to broadcast image token count, using local value {self.image_len}. Error: {e}')

  
    def __len__(self) -> int:
        return len(self.maze_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # 从 MazeDataset 获取数据
        maze_item = self.maze_dataset[index]
        
        # 转换为 SFT 训练需要的格式
        # sol_img 作为 output_image（目标生成的图像）
        # m_original_img 或 original_img 作为 input_image（输入图像）
        # prompt 作为 input_prompt
        item = {
            'output_image': maze_item['sol_img'],  # 解答图像作为输出目标
            'input_prompt': maze_item['prompt'],
        }
        
        # 如果有输入图像，使用 m_original_img（带标记的迷宫）或 original_img
        input_images = []
        if maze_item['m_original_img'] is not None:
            input_images.append(maze_item['m_original_img'])
        # elif maze_item['original_img'] is not None:
        #     input_images.append(maze_item['original_img'])
        
        if len(input_images) > 0:
            item['input_image'] = input_images
        
        return item
    
    def get_code_book(self, images):
        # images 现在是 PIL Image 对象列表，不是路径列表
        if images and isinstance(images[0], str):
            images = [PIL.Image.open(img_path).convert("RGB") for img_path in images]
        elif images and not isinstance(images[0], PIL.Image.Image):
            images = [PIL.Image.fromarray(img).convert("RGB") if hasattr(img, 'shape') else img for img in images]
        images_outputs = self.processor.image_processor(images, return_tensors="pt")
        return images_outputs['pixel_values'].to(torch.bfloat16)
    
    def process_image(self, images):
        # images 可以是 PIL Image 对象列表或路径列表
        if not images:
            return torch.tensor([])
        # 如果是字符串路径列表，需要加载
        if isinstance(images[0], str):
            images = [PIL.Image.open(img_path).convert("RGB") for img_path in images]
        # 确保所有图像都是 PIL Image 并转换为 RGB
        images = [img.convert("RGB") if isinstance(img, PIL.Image.Image) else img for img in images]
        images_outputs = self.processor.image_processor(images, return_tensors="pt")
        return images_outputs['pixel_values']

# 在 SftDataset 中修改 collate_fn
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1. 准备图片数据
        gen_images = [x['output_image'] for x in batch]
        input_images = [x['input_image'][0] for x in batch] # 假设每个样本只有一张输入图
        
        # 2. 处理 Pixel Values (都要转成 bfloat16 给 VQ Model 用)
        # 输出图（Target）
        pixel_values_output = self.process_image(gen_images).to(torch.bfloat16)
        # 输入图（Condition），注意这里也用 process_image，而不是 encoder 的处理逻辑
        pixel_values_input = self.process_image(input_images).to(torch.bfloat16)
        
        # 3. 构造 Prompt 字符串
        # 我们需要手动构造： <vision_start><pad>*576<vision_end> + Prompt + <vision_start><pad>*576
        
        # 定义 Token 字符串
        image_token_str = self.processor.image_start_tag + \
                          self.processor.pad_tag * self.processor.num_image_tokens + \
                          self.processor.image_end_tag
                          
        # 仅用于输出的开头（生成部分通常不需要 end tag，看具体训练习惯，Janus通常只要 start）
        output_image_start = self.processor.image_start_tag + \
                             self.processor.pad_tag * self.processor.num_image_tokens

        pre_data = []
        for x in batch:
            # 构造输入部分：Input Image Tokens + Text Prompt
            # 注意：这里把输入图片当成了文本的一部分塞进去
            user_content = image_token_str + "\n" + x['input_prompt']
            
            conversation = [
                {"role": "<|User|>", "content": user_content},
                {"role": "<|Assistant|>", "content": ""}
            ]
            
            # 应用模板
            sft_format = self.processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self.processor.sft_format,
                system_prompt="",
            )
            
            # 拼接输出部分的占位符
            sft_format = sft_format + output_image_start
            
            input_ids = torch.LongTensor(self.processor.tokenizer.encode(sft_format))
            
            # 此时 input_ids 里有两段 pad_tokens：
            # 第一段是输入图，第二段是输出图。我们在 train 里通过位置来区分。
            
            pre_data.append(VLChatProcessorOutput(
                sft_format=sft_format,
                pixel_values=None, # 这里不再需要 encoder_pixel_values
                input_ids=input_ids,
                num_image_tokens=[] # 这里的逻辑可以忽略，因为我们手动处理了
            ))
            
        if len(pre_data) > 0:
            prepare_inputs = self.processor.batchify(pre_data)
            
        return {
            "input_ids": prepare_inputs.input_ids,
            "attention_mask": prepare_inputs.attention_mask,
            "pixel_values": pixel_values_output,      # 用于计算 Loss 的目标图
            "input_pixel_values": pixel_values_input, # 用于注入输入的条件图
            "images_seq_mask": prepare_inputs['images_seq_mask'],
            "images_emb_mask": prepare_inputs['images_emb_mask']
        }

def load_training_state(checkpoint_dir: str) -> Dict[str, Any]:
    """Load training state from checkpoint directory."""
    training_state_path = os.path.join(checkpoint_dir, 'training_state.json')
    if os.path.exists(training_state_path):
        with open(training_state_path, 'r') as f:
            return json.load(f)
    return None

def save_checkpoint(
    model,
    processor,
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch: int,
    step: int,
    global_step: int,
    is_last: bool = False
) -> None:

    save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")
    
    if accelerator.is_main_process:
        # Manage checkpoint numbers
        checkpoint_files = [f for f in os.listdir(args.output_dir) if f.startswith("checkpoint-")]
        if args.max_ckpts > 0 and len(checkpoint_files) >= args.max_ckpts:
            oldest_ckpt = min(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(args.output_dir, x)))
            shutil.rmtree(os.path.join(args.output_dir, oldest_ckpt))

        os.makedirs(save_dir, exist_ok=True)
        output_dir = os.path.join(save_dir, 'tfmr')

        model.save_pretrained(output_dir, state_dict=accelerator.get_state_dict(model))
        processor.save_pretrained(output_dir)
        
        # Save training metadata
        training_state = {
            'epoch': epoch,
            'step': step,
            'global_step': global_step,
            'n_epochs': args.n_epochs,
            'learning_rate': args.learning_rate,
        }
        training_state_path = os.path.join(save_dir, 'training_state.json')
        with open(training_state_path, 'w') as f:
            json.dump(training_state, f, indent=2)

    accelerator.wait_for_everyone()
    logger.info(f'Checkpoint {epoch}-{global_step} saved successfully')

def train(args: argparse.Namespace) -> None:

    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    if accelerator.is_main_process:
        wandb.init(
            project=args.experiment_name,
            name=args.run_name,
            config=args,
            dir=args.log_dir,
            mode="offline"
        )

    # Set batch size
    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_bsz_per_gpu
    accelerator.state.deepspeed_plugin.deepspeed_config['train_batch_size'] = (
        args.train_bsz_per_gpu * 
        dist.get_world_size() * 
        accelerator.gradient_accumulation_steps
    )

    # Determine model path (from checkpoint or original)
    resume_from_checkpoint = getattr(args, 'resume_from_checkpoint', None)
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        model_path = os.path.join(resume_from_checkpoint, 'tfmr')
        if accelerator.is_main_process:
            logger.info(f'Resuming training from checkpoint: {resume_from_checkpoint}')
            training_state = load_training_state(resume_from_checkpoint)
            if training_state:
                logger.info(f'Loaded training state: epoch={training_state.get("epoch")}, global_step={training_state.get("global_step")}')
    else:
        model_path = args.model_path
        if accelerator.is_main_process:
            logger.info(f'Starting training from model: {model_path}')

    # Load model and tokenizer
    processor = VLChatProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    model_config = model.config
    
    # Enable gradient checkpointing to save memory (critical for OOM)
    if hasattr(model.language_model.model, 'gradient_checkpointing_enable'):
        model.language_model.model.gradient_checkpointing_enable()
        if accelerator.is_main_process:
            logger.info('Gradient checkpointing enabled to save memory')

    # 检测真实的图像token数量（已移除，使用独立脚本detect_image_tokens.py进行检测）
    # 运行: CUDA_VISIBLE_DEVICES=0 python detect_image_tokens.py --model_path <模型路径>

    # Configure optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare data loader
    train_dataset = SftDataset(args, processor,accelerator,model)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_bsz_per_gpu,
        shuffle=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=1
    )

    # Set learning rate scheduler
    num_training_steps = int(len(train_dataloader) * args.n_epochs) // accelerator.gradient_accumulation_steps // dist.get_world_size()

    # Use custom scheduler instead of original call
    lr_scheduler = get_custom_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_rates * num_training_steps),
        num_training_steps=num_training_steps,
        min_lr_ratio=args.min_lr_ratio  # Pass minimum learning rate ratio directly
    )

    # Prepare training
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    # Load checkpoint state if resuming
    resume_from_checkpoint = getattr(args, 'resume_from_checkpoint', None)
    start_epoch = 0
    global_step = 0
    training_state = None
    
    if resume_from_checkpoint:
        if not os.path.exists(resume_from_checkpoint):
            raise ValueError(f'Checkpoint directory does not exist: {resume_from_checkpoint}')
        
        # Verify checkpoint structure
        tfmr_dir = os.path.join(resume_from_checkpoint, 'tfmr')
        if not os.path.exists(tfmr_dir):
            raise ValueError(f'Checkpoint directory missing tfmr subdirectory: {resume_from_checkpoint}')
        
        if accelerator.is_main_process:
            logger.info(f'Loading training state from {resume_from_checkpoint}')
        training_state = load_training_state(resume_from_checkpoint)
        if training_state:
            start_epoch = training_state.get('epoch', 0) + 1  # Start from next epoch
            global_step = training_state.get('global_step', 0)
            if accelerator.is_main_process:
                logger.info(f'Resumed from epoch {training_state.get("epoch")}, global_step {global_step}')
        else:
            if accelerator.is_main_process:
                logger.warning(f'Could not load training_state.json from checkpoint, starting from epoch 0')
        accelerator.wait_for_everyone()

    metric = TrainingMetrics(device=torch.cuda.current_device())
    model.train()
    
    # Early stopping variables
    best_loss = None  # Will be initialized when early stopping starts
    patience_counter = 0
    should_stop = False
    early_stopping_initialized = False

    for epoch in range(start_epoch, args.n_epochs):
        if should_stop:
            if accelerator.is_main_process:
                logger.info(f'Early stopping triggered at epoch {epoch}, step {global_step}')
            break
        
        # Initialize early stopping when reaching the specified epoch
        if epoch == args.early_stopping_start_epoch and not early_stopping_initialized:
            if accelerator.is_main_process:
                logger.info(f'Early stopping will be enabled starting from epoch {epoch}')
        
        train_iter = tqdm(train_dataloader, total=len(train_dataloader)) if accelerator.is_main_process else train_dataloader

        for batch in train_iter:
            # 1. 处理输出图像（生成路径）：将输出图像编码为 VQ tokens 作为 labels
            with torch.no_grad():
                quant, emb_loss, info = model.gen_vision_model.encode(batch['pixel_values'])
                target_image_tokens = info[2].detach().reshape(batch['pixel_values'].shape[0], -1)
            del quant, emb_loss, info
            
            # 生成 Target 的 Embeddings (用于 Teacher Forcing)
            target_image_embeds = model.prepare_gen_img_embeds(target_image_tokens)

            # =================================================================
            # 2. 编码 Input Image (作为 Condition) - 新增部分
            # =================================================================
            with torch.no_grad():
                # 同样的 VQ 编码器
                quant_in, emb_loss_in, info_in = model.gen_vision_model.encode(batch['input_pixel_values'])
                input_image_tokens = info_in[2].detach().reshape(batch['input_pixel_values'].shape[0], -1)
            del quant_in, emb_loss_in, info_in

            # 2. 处理输入图像（理解路径）：使用 prepare_inputs_embeds，它会自动通过 SigLIP 编码器处理输入图像
            # encoder_pixel_values 已经通过 batchify 正确处理，prepare_inputs_embeds 会使用 SigLIP 编码器处理它们
            # inputs_embeds = model.prepare_inputs_embeds(
            #     input_ids=batch['input_ids'],
            #     pixel_values=batch['encoder_pixel_values'],
            #     images_emb_mask=batch['images_emb_mask'],
            #     images_seq_mask=batch['images_seq_mask']
            # )
            # 这一行被你遗漏了！
            input_image_embeds = model.prepare_gen_img_embeds(input_image_tokens)
            # =================================================================
            # 3. 准备基础 Embeddings
            # =================================================================
            # 注意：这里不再使用 model.prepare_inputs_embeds 里的 ViT 逻辑
            # 直接把 input_ids 转成 text embeddings
            inputs_embeds = model.language_model.get_input_embeddings()(batch['input_ids'])
            # =================================================================
            # 4. 注入 Embeddings (关键步骤！)
            # =================================================================
            
            # 这里的逻辑假设：每个样本的结构都是 [Input_Img] ... [Output_Img]
            # 我们需要找到 input_ids 中 <vision_start> 的位置
            
            image_start_id = processor.image_start_id # 确保你有这个 ID，通常在 processor.tokenizer 里
            image_token_len = 576
            # 遍历 Batch 中的每一个样本
            batch_size = inputs_embeds.shape[0]
            for i in range(batch_size):
                # 找到所有 <vision_start> 的索引
                # (seq_len,) -> indices
                start_indices = (batch['input_ids'][i] == image_start_id).nonzero(as_tuple=True)[0]
                
                # 理论上应该有两个 start tag (一个输入，一个输出)
                # 如果有时候只有输出（比如有的样本没输入），要做判断。
                # 假设都有：
                if len(start_indices) >= 2:
                    # --- 处理输入图像 (第一个位置) ---
                    input_start = start_indices[0] + 1 # start tag 后面紧接着就是 image tokens
                    input_end = input_start + image_token_len
                    # 直接替换：Input Image 不需要 Shift！因为它完全是已知的条件
                    inputs_embeds[i, input_start:input_end, :] = input_image_embeds[i]

                    # --- 处理输出图像 (第二个位置) ---
                    # 保持你原本正确的 Teacher Forcing 逻辑 (Shift)
                    output_start = start_indices[-1] + 1
                    output_end = output_start + image_token_len
                    
                    # 目标区间：inputs_embeds[output_start : output_end]
                    # 我们填入：target_image_embeds[:-1] (前575个) 到前 575 个位置
                    # 最后一个位置保持原样（或者是 pad，无所谓，因为不预测它之后的东西）
                    
                    # 你的旧逻辑复用：
                    # inputs_embeds[i, output_start : output_end-1, :] = target_image_embeds[i, :-1, :]
                    
                    # 更稳健的写法：
                    inputs_embeds[i, output_start : output_end-1, :] = target_image_embeds[i, :-1, :]

            # forward and calculate loss
            outputs = model.language_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=batch['attention_mask'],
                return_dict=True,
                use_cache=False
            )
            hidden_states = outputs.last_hidden_state

            image_embeds_shape = 576
            full_len = inputs_embeds.shape[1]
            start_pos = full_len - image_embeds_shape # 假设输出图在最末尾
            
            # 提取 logits: [start_pos-1 : full_len-1]
            # start_pos-1 是 <vision_start>，预测 Token_0
            # full_len-1 是 Token_574，预测 Token_575
            logits = model.gen_head(hidden_states[:, start_pos - 1 : full_len - 1, :])
            # Calculate MSE: predicted embeddings vs true embeddings
            with torch.no_grad():
                # Get predicted tokens from logits (argmax)
                pred_tokens = logits.detach().argmax(dim=-1)  # [batch, seq_len]
                # Convert predicted tokens to embeddings (same way as true embeddings)
                # prepare_gen_img_embeds expects [batch, seq_len] and returns [batch, seq_len, embed_dim]
                pred_image_embeds = model.prepare_gen_img_embeds(pred_tokens)
                # Calculate MSE between predicted and true embeddings
                mse = F.mse_loss(pred_image_embeds, target_image_embeds, reduction='mean')
            # loss = model.language_model.loss_function(logits=logits, labels=image_tokens,vocab_size=model_config.gen_vision_config.params.image_token_size)
            logits_flat = logits.view(-1, logits.size(-1)) # [B*576, 16384]
            labels_flat = target_image_tokens.view(-1)            # [B*576]
            loss = F.cross_entropy(logits_flat, labels_flat)
            
            # Calculate MSE: predicted embeddings vs true embeddings
            # with torch.no_grad():
            #     # Get predicted tokens from logits (argmax)
            #     pred_tokens = logits.detach().argmax(dim=-1)  # [batch, seq_len]
            #     # Convert predicted tokens to embeddings (same way as true embeddings)
            #     # prepare_gen_img_embeds expects [batch, seq_len] and returns [batch, seq_len, embed_dim]
            #     pred_image_embeds = model.prepare_gen_img_embeds(pred_tokens)
            #     # Calculate MSE between predicted and true embeddings
            #     mse = F.mse_loss(pred_image_embeds, target_image_tokens, reduction='mean')
            
            # Calculate metrics before backward (use detached copies to save memory)
            metric(logits.detach(), target_image_tokens.detach(), loss.detach(), mse.detach())
            # Free computed tensors
            del pred_image_embeds, target_image_tokens
            # Free image_tokens (labels don't need gradients)
            # del target_image_tokens
            torch.cuda.empty_cache()

            # Backpropagation
            accelerator.backward(loss)
            # Free loss and logits after backward (gradients are computed)
            del loss, logits
            # torch.cuda.empty_cache()
            
            # Free inputs_embeds after backward (no longer needed)
            del inputs_embeds
            # torch.cuda.empty_cache()

            # Add gradient clipping
            if args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (global_step + 1) % accelerator.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # Clear cache after optimizer step
                torch.cuda.empty_cache()
            else:
                # Clear cache during gradient accumulation
                torch.cuda.empty_cache()

            
            global_step += 1

            if (global_step + 1) % accelerator.gradient_accumulation_steps == 0:
                # Calculate metrics
                acc, train_loss, train_mse = metric.get_metric()
                
                # Early stopping check (only after reaching the specified epoch)
                if epoch >= args.early_stopping_start_epoch:
                    # Initialize best_loss on first check
                    if not early_stopping_initialized:
                        best_loss = train_loss
                        patience_counter = 0
                        early_stopping_initialized = True
                        if accelerator.is_main_process:
                            logger.info(f'Early stopping enabled at epoch {epoch}, step {global_step}. Initial loss: {train_loss:.4f}')
                    
                    if train_loss < best_loss:
                        best_loss = train_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= args.early_stopping_patience:
                            should_stop = True
                            if accelerator.is_main_process:
                                logger.info(f'Early stopping: loss has not improved for {args.early_stopping_patience} steps. Best loss: {best_loss:.4f}, Current loss: {train_loss:.4f}')
                
                if accelerator.is_main_process:
                    postfix_dict = {
                        'epoch': epoch,
                        'step': global_step-1,
                        'total_steps': len(train_dataloader),
                        'skip': accelerator.optimizer_step_was_skipped,
                        'length': len(batch["input_ids"][0]),
                        'loss': f"{train_loss:.3f}",
                        'acc': f"{acc:.3f}",
                        'mse': f"{train_mse:.4f}",
                        'lr': f"{lr_scheduler.get_last_lr()[0]:.2e}"
                    }
                    if early_stopping_initialized:
                        postfix_dict['patience'] = f"{patience_counter}/{args.early_stopping_patience}"
                    
                    train_iter.set_postfix(**postfix_dict)
                    
                    log_dict = {
                        'loss': train_loss,
                        'acc': acc,
                        'mse': train_mse,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }
                    if early_stopping_initialized:
                        log_dict['best_loss'] = best_loss
                        log_dict['patience_counter'] = patience_counter
                    
                    wandb.log(log_dict, step=global_step)
                
                if should_stop:
                    # 早停时保存最新的权重
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        logger.info(f'Saving checkpoint before early stopping at epoch {epoch}, step {global_step}')
                    save_checkpoint(
                        model=model,
                        processor=processor, 
                        accelerator=accelerator,
                        args=args,
                        epoch=epoch,
                        step=global_step-1,
                        global_step=global_step,
                        is_last=True
                    )
                    break

        if should_stop:
            break
            
        accelerator.wait_for_everyone()
        save_checkpoint(
            model=model,
            processor=processor, 
            accelerator=accelerator,
            args=args,
            epoch=epoch,
            step=global_step-1,
            global_step=global_step,
            is_last=True
        )
    
    if should_stop and accelerator.is_main_process:
        best_loss_str = f"{best_loss:.4f}" if best_loss is not None else "N/A"
        logger.info(f'Training stopped early. Total epochs completed: {epoch}, Total steps: {global_step}, Best loss: {best_loss_str}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-training parameter configuration')
    
    # Experiment settings
    parser.add_argument('--experiment_name', type=str, default='janus_train', help='Experiment name')
    parser.add_argument('--run_name', type=str, default='run_1', help='Run name')
    parser.add_argument('--model_path', type=str, default='', help='Pre-trained model path')

    # Data related
    parser.add_argument('--data_path', type=str, required=True, help='Training data directory path containing maze_dataset_train.parquet')
    parser.add_argument('--output_dir', type=str, default='./', help='Model save path')
    parser.add_argument('--max_ckpts', type=int, default=5, help='Maximum number of checkpoints to save')
    parser.add_argument('--log_dir', type=str, default='./train_logs', help='Log save path')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint directory to resume training from')

    # Training related
    parser.add_argument('--max_seq_len', type=int, default=4096, help='Maximum sequence length')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping threshold, set to 0 for no clipping')
    parser.add_argument('--train_bsz_per_gpu', type=int, default=1, help='Batch size per GPU')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--min_lr_ratio', type=float, default=0.15, help='Minimum learning rate ratio to peak learning rate')
    parser.add_argument('--warmup_rates', type=float, default=0.05, help='Warmup ratio')
    parser.add_argument('--n_epochs', type=int, default=8, help='Number of training epochs')
    parser.add_argument('--early_stopping_start_epoch', type=int, default=200, help='Start early stopping check after this many epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Early stopping patience: stop training if loss does not improve for this many steps')

    # Others
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    
    # Set paths
    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    if args.run_name:
        args.output_dir = os.path.join(args.output_dir, args.run_name)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Set random seed
    set_seed(args.seed)

    # Start training
    train(args)     
