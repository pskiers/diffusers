"""
accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 infer_qwen.py \
    --dataset_path /root/private_data/circle/maze-dataset \
    --model_path /root/private_data/model/huggingface/hub/models--Qwen--Qwen-Image-Edit/snapshots/ac7f9318f633fc4b5778c59367c8128225f1e3de \
    --output_dir ./results/qwen_result/circle_5 \
    --split test \
    --save_images \
    --num_inference_steps 10 \
    --max_samples 700 \
    --num_attempts 5 \
    --max_samples_per_size 5 
"""
import os
import json
import torch
import numpy as np
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
import argparse
from tqdm import tqdm
import sys
import hashlib
from collections import defaultdict
from torch.utils.data import DataLoader
from accelerate import Accelerator
from functools import partial

# 添加Janus目录到路径，以便导入maze_dataset和maze_rewards
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Janus'))
from data.maze_dataset import MazeDataset
IS_CIRCLE=False
from infer.maze_metrics import maze_metric

tqdm = partial(tqdm, dynamic_ncols=True)


def main(args):
    # 初始化 Accelerator（多GPU支持）
    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        print("="*70)
        print("Multi-GPU Evaluation Setup")
        print("="*70)
        print(f"Number of GPUs: {accelerator.num_processes}")
        print(f"Current GPU rank: {accelerator.process_index}")
        print("="*70)
    
    # 加载pipeline（每个GPU加载自己的pipeline）
    if accelerator.is_local_main_process:
        print("Loading QwenImageEditPlusPipeline...")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16
    )
    if accelerator.is_local_main_process:
        print("Pipeline loaded")
    
    # 将pipeline移动到当前GPU
    device = accelerator.device
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=None)
    
    # 加载数据集
    if accelerator.is_main_process:
        print(f"Loading dataset from {args.dataset_path}...")
    dataset = MazeDataset(args.dataset_path, split=args.split)
    if accelerator.is_main_process:
        print(f"Dataset loaded: {len(dataset)} samples")
    
    # 创建reward函数
    if accelerator.is_main_process:
        print("Creating maze reward function...")
    reward_fn = maze_metric()
    if accelerator.is_main_process:
        print("Reward function created")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    
    # ========== 按尺寸分组，然后创建batch ==========
    # 限制样本数量
    num_samples = min(args.max_samples, len(dataset)) if args.max_samples else len(dataset)
    indices = list(range(num_samples))
    
    # 先按尺寸分组样本
    samples_by_size = defaultdict(list)
    for idx in indices:
        sample = dataset[idx]
        metadata = sample.get("metadata", {})
        maze_config = {}
        
        # 提取maze_config
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
        
        # 按width/height分组
        if IS_CIRCLE: 
            width=maze_config.get('layers', 0)
            height=maze_config.get('layers',0)
        else:
            width = maze_config.get('width', 0)
            height = maze_config.get('height', 0)
        size_key = f"{width}x{height}"
        samples_by_size[size_key].append(idx)
    
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"📊 按尺寸分组结果:")
        for size_key in sorted(samples_by_size.keys()):
            print(f"   {size_key}: {len(samples_by_size[size_key])} 个样本")
        print(f"{'='*60}\n")
    
    # 限制每个尺寸组的样本数量
    max_samples_per_size = getattr(args, 'max_samples_per_size', None)
    if max_samples_per_size is not None:
        for size_key in samples_by_size.keys():
            if len(samples_by_size[size_key]) > max_samples_per_size:
                samples_by_size[size_key] = samples_by_size[size_key][:max_samples_per_size]
        if accelerator.is_main_process:
            print(f"\n{'='*60}")
            print(f"📊 限制每个尺寸最多 {max_samples_per_size} 个样本后的结果:")
            for size_key in sorted(samples_by_size.keys()):
                print(f"   {size_key}: {len(samples_by_size[size_key])} 个样本")
            print(f"{'='*60}\n")
    
    # 对每个尺寸组分别创建batch
    batch_size = getattr(args, 'batch_size', 1)
    eval_batches = []
    for size_key in sorted(samples_by_size.keys()):
        size_indices = samples_by_size[size_key]
        for i in range(0, len(size_indices), batch_size):
            batch_indices = size_indices[i : i + batch_size]
            batch_samples = [dataset[idx] for idx in batch_indices]
            batch_data = MazeDataset.collate_fn(batch_samples)
            eval_batches.append((batch_indices, batch_data, size_key))
    
    if accelerator.is_main_process:
        print(f"📦 每个尺寸内的batch大小: {batch_size}")
        print(f"📦 总共创建了 {len(eval_batches)} 个batch")
        print(f"📦 将分配给 {world_size} 个GPU处理\n")
    
    # 将batch分配给不同的GPU（轮询分配）
    # 每个GPU处理 batch_idx % world_size == rank 的batch
    my_batches = []
    for batch_idx, (batch_indices, batch_data, size_key) in enumerate(eval_batches):
        if batch_idx % world_size == rank:
            my_batches.append((batch_idx, batch_indices, batch_data, size_key))
    
    if accelerator.is_local_main_process:
        print(f"[GPU {rank}] 分配到 {len(my_batches)} 个batch (总共 {len(eval_batches)} 个batch)")
    # ====================================================
    
    # ========== 多卡并行：每张卡负责不同的 attempts ==========
    num_attempts = getattr(args, 'num_attempts', 1)
    
    # 改进的分配逻辑：每个GPU按轮询方式分配attempts
    # 例如：num_attempts=5, world_size=2 -> rank0: [1,3,5], rank1: [2,4]
# ========== 修改后的逻辑：每个GPU都运行所有attempts，通过batch切分实现并行 ==========
    # 原来的逻辑会导致如果 attempt 数量少于显卡数量，部分显卡直接空闲等待导致超时
    
    # 也就是让每个 rank 都拥有完整的 attempts 列表
    my_attempts = list(range(1, num_attempts + 1))
    
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"🚀 Multi-GPU Strategy:")
        print(f"   Each GPU will process specific batches for ALL {num_attempts} attempts.")
        print(f"{'='*60}\n")
    # ====================================================
    
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"🚀 Multi-GPU Attempt Distribution:")
        print(f"   Total attempts per sample: {num_attempts}")
        print(f"   Number of GPUs: {world_size}")
        for r in range(world_size):
            r_attempts = [a for a in range(1, num_attempts + 1) if (a - 1) % world_size == r]
            print(f"   GPU {r}: attempts {r_attempts}")
        print(f"{'='*60}\n")
    
    if accelerator.is_local_main_process:
        print(f"[GPU {rank}] Will process attempts: {my_attempts}")
    
    # 检查是否有任务分配
    if len(my_attempts) == 0:
        if accelerator.is_local_main_process:
            print(f"[GPU {rank}] ⚠️ Warning: No attempts assigned to this GPU. Will skip processing.")
    if len(my_batches) == 0:
        if accelerator.is_local_main_process:
            print(f"[GPU {rank}] ⚠️ Warning: No batches assigned to this GPU. Will skip processing.")
    # ====================================================
    
    # 评估结果存储（每个GPU维护自己的结果）
    all_rewards = {
        'mse_inside': [],
        'mse_outside': [],
        'mse_solution': [],
        'path_validity': [],
        'gt_cell_coverage': [],
        'background_violation': [],
        'avg': []
    }
    results = []  # 存储每个样本的详细结果
    
    with torch.inference_mode():
        # 外层循环：遍历attempts（每个GPU处理分配给它的attempts）
        # 如果my_attempts为空，跳过循环但确保所有进程都能到达同步点
        if len(my_attempts) == 0 or len(my_batches) == 0:
            if accelerator.is_local_main_process:
                print(f"[GPU {rank}] ⏭️ Skipping processing (no tasks assigned)")
        else:
            for attempt_idx in my_attempts:
                if accelerator.is_local_main_process:
                    print(f"[GPU {rank}] 🎲 Processing Attempt {attempt_idx}/{num_attempts}")
                
                # 设置不同的随机种子用于每个attempt
                attempt_seed = args.seed + attempt_idx * 1000
                torch.manual_seed(attempt_seed)
                torch.cuda.manual_seed_all(attempt_seed)
                np.random.seed(attempt_seed)
                
                # 遍历分配给当前GPU的batch
                processed_count = 0
                for local_batch_idx, (global_batch_idx, batch_indices, batch_data, size_key) in enumerate(tqdm(
                    my_batches, 
                    desc=f"[GPU {rank}] Eval (attempt {attempt_idx}/{num_attempts})",
                    disable=not accelerator.is_local_main_process
                )):
                # batch_data 是 (prompts, metadatas) 格式（来自 MazeDataset.collate_fn）
                    prompts, metadatas = batch_data
                
                    # 处理batch中的每个样本（QwenImageEditPlusPipeline不支持batch处理）
                    for sample_idx in range(len(prompts)):
                        try:
                            prompt = prompts[sample_idx]
                            metadata = metadatas[sample_idx]
                            
                            # 从metadata中提取必要信息
                            sample_id = metadata.get('id', f"sample_{global_batch_idx}_{sample_idx}")
                            m_original_img = metadata.get('m_original_img')
                            sol_img = metadata.get('sol_img')
                            original_img = metadata.get('original_img')
                            mask_img = metadata.get('mask_img')
                            cell_map = metadata.get('cell_map')
                            
                            # 检查必要的图像是否存在
                            if m_original_img is None:
                                if accelerator.is_local_main_process:
                                    print(f"[GPU {rank}] Warning: Sample {sample_id} has no m_original_img, skipping")
                                continue
                            
                            if sol_img is None:
                                if accelerator.is_local_main_process:
                                    print(f"[GPU {rank}] Warning: Sample {sample_id} has no sol_img, skipping")
                                continue
                            
                            # 准备pipeline输入
                            # 使用sample_id和attempt_idx来设置随机种子，确保可重复性
                            sample_seed = args.seed + attempt_idx * 1000 + int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16) % 1000000
                            inputs = {
                                "image": [m_original_img],
                                "prompt": prompt,
                                "generator": torch.manual_seed(sample_seed),
                                "true_cfg_scale": args.true_cfg_scale,
                                "negative_prompt": args.negative_prompt,
                                "num_inference_steps": args.num_inference_steps,
                                "guidance_scale": args.guidance_scale,
                                "num_images_per_prompt": 1,
                            }
                            
                            # 生成图像
                            output = pipeline(**inputs)
                            generated_image = output.images[0]
                            
                            # 将生成的PIL图像转换为tensor格式 (B, C, H, W)，值范围[0, 1]
                            gen_img_array = np.array(generated_image.convert('RGB'))  # (H, W, 3), uint8
                            gen_img_tensor = torch.from_numpy(gen_img_array.transpose(2, 0, 1)).float() / 255.0  # (C, H, W), [0, 1]
                            gen_img_batch = gen_img_tensor.unsqueeze(0)  # (1, C, H, W)
                            gen_img_batch = gen_img_batch.cpu()  # 确保在CPU上
                            
                            # 处理metadata（可能需要解析JSON字符串）
                            metadata_content = metadata.get('metadata', {})
                            if metadata_content and isinstance(metadata_content, str):
                                try:
                                    metadata_content = json.loads(metadata_content)
                                except json.JSONDecodeError:
                                    if accelerator.is_local_main_process:
                                        print(f'[GPU {rank}] Warning: Failed to parse metadata JSON for sample {sample_id}')
                                    metadata_content = {}
                            elif not isinstance(metadata_content, dict):
                                metadata_content = {}
                            
                            # 从metadata中读取maze_config的width和height
                            maze_width = None
                            maze_height = None
                            if metadata_content and isinstance(metadata_content, dict):
                                maze_config = metadata_content.get('maze_config', {})
                                if isinstance(maze_config, dict):
                                    maze_width = maze_config.get('width')
                                    maze_height = maze_config.get('height')
                            
                            # 如果找不到width和height，使用默认值
                            if maze_width is None or maze_height is None:
                                maze_width = 3
                                maze_height = 3
                            else:
                                # 确保width和height是整数
                                try:
                                    maze_width = int(maze_width)
                                    maze_height = int(maze_height)
                                except (ValueError, TypeError):
                                    maze_width = 3
                                    maze_height = 3
                            
                            # 准备metadata用于reward计算
                            reward_metadata = {
                                'original_img': original_img,
                                'm_original_img': m_original_img,
                                'sol_img': sol_img,
                                'mask_img': mask_img,
                                'cell_map': cell_map,
                                'metadata': metadata_content  # 包含maze_config, path_cell_ids等
                            }
                            
                            # 调用reward函数
                            try:
                                rewards_dict_batch, reward_metadata_batch = reward_fn(
                                    images=gen_img_batch,
                                    prompts=[prompt],
                                    metadata=[reward_metadata],
                                    only_strict=False
                                )
                                
                                # 提取单个样本的reward值（因为是batch，取第一个）
                                sample_rewards = {
                                    'mse_inside': float(rewards_dict_batch['mse_inside'][0]) if 'mse_inside' in rewards_dict_batch else 0.0,
                                    'mse_outside': float(rewards_dict_batch['mse_outside'][0]) if 'mse_outside' in rewards_dict_batch else 0.0,
                                    'mse_solution': float(rewards_dict_batch['mse_solution'][0]) if 'mse_solution' in rewards_dict_batch else 0.0,
                                    'path_validity': float(rewards_dict_batch['path_validity'][0]) if 'path_validity' in rewards_dict_batch else 0.0,
                                    'gt_cell_coverage': float(rewards_dict_batch['gt_cell_coverage'][0]) if 'gt_cell_coverage' in rewards_dict_batch else 0.0,
                                    'background_violation': float(rewards_dict_batch['background_violation'][0]) if 'background_violation' in rewards_dict_batch else 0.0,
                                    'avg': float(rewards_dict_batch['avg'][0]) if 'avg' in rewards_dict_batch else 0.0,
                                }
                            except Exception as e:
                                if accelerator.is_local_main_process:
                                    print(f'[GPU {rank}] Error computing rewards for sample {sample_id}: {str(e)}')
                                    import traceback
                                    traceback.print_exc()
                                # 使用默认值
                                sample_rewards = {
                                    'mse_inside': 0.0,
                                    'mse_outside': 0.0,
                                    'mse_solution': 0.0,
                                    'path_validity': 0.0,
                                    'gt_cell_coverage': 0.0,
                                    'background_violation': 0.0,
                                    'avg': 0.0,
                                }
                            
                            # 存储所有指标
                            for key in all_rewards.keys():
                                all_rewards[key].append(sample_rewards[key])
                            
                            # 创建文件名基础部分
                            filename_base = f"{maze_width}×{maze_height}_{sample_id}"
                            
                            # 保存图像（可选）
                            if args.save_images:
                                # 保存生成的图像，使用attempt编号
                                attempt_filename = os.path.join(
                                    args.output_dir,
                                    f"{filename_base}_attempt{attempt_idx:03d}.png"
                                )
                                generated_image.save(attempt_filename)
                                
                                # 保存输入图像 (m_original_img) - 只在主进程的第一个attempt保存，避免重复
                                if attempt_idx == 1 and accelerator.is_main_process and m_original_img is not None:
                                    input_filename = os.path.join(
                                        args.output_dir,
                                        f"{filename_base}_input.png"
                                    )
                                    if not os.path.exists(input_filename):
                                        # 确保是PIL Image
                                        if isinstance(m_original_img, Image.Image):
                                            m_original_img.save(input_filename)
                                        else:
                                            # 如果是其他格式，尝试转换
                                            Image.fromarray(np.array(m_original_img)).save(input_filename)
                                
                                # 保存GT图像 (sol_img) - 只在主进程的第一个attempt保存，避免重复
                                if attempt_idx == 1 and accelerator.is_main_process and sol_img is not None:
                                    gt_filename = os.path.join(
                                        args.output_dir,
                                        f"{filename_base}_gt.png"
                                    )
                                    if not os.path.exists(gt_filename):
                                        # 确保是PIL Image
                                        if isinstance(sol_img, Image.Image):
                                            sol_img.save(gt_filename)
                                        else:
                                            # 如果是其他格式，尝试转换
                                            Image.fromarray(np.array(sol_img)).save(gt_filename)
                            
                            # 保存结果（按照inference_sft.py的格式，包含attempt信息）
                                result_entry = {
                                    'id': sample_id,
                                    'attempt': attempt_idx,
                                    'width': maze_width,
                                    'height': maze_height,
                                    'rewards': sample_rewards
                                }
                                results.append(result_entry)
                            
                            processed_count += 1
                            if processed_count % args.log_interval == 0:
                                current_avg_reward = np.mean(all_rewards['avg']) if len(all_rewards['avg']) > 0 else 0.0
                                current_mse_solution = np.mean(all_rewards['mse_solution']) if len(all_rewards['mse_solution']) > 0 else 0.0
                                if accelerator.is_local_main_process:
                                    print(f"[GPU {rank}] Processed {processed_count} samples (attempt {attempt_idx}, batch {local_batch_idx+1}/{len(my_batches)}, size: {size_key}), "
                                          f"current avg reward: {current_avg_reward:.6f}, "
                                          f"current mse_solution: {current_mse_solution:.6f}")
                
                        except Exception as e:
                            if accelerator.is_local_main_process:
                                print(f"[GPU {rank}] Error processing sample: {e}")
                                import traceback
                                traceback.print_exc()
                            continue
    
    # 同步所有进程（确保所有GPU都完成处理）
    if accelerator.is_local_main_process:
        print(f"[GPU {rank}] ✅ Finished processing, waiting for all GPUs to synchronize...")
    accelerator.wait_for_everyone()
    
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
    if len(results) > 0:
        temp_json_path = os.path.join(args.output_dir, f"evaluation_results_rank{rank}.json.tmp")
        results_native = convert_to_native(results)
        with open(temp_json_path, 'w', encoding='utf-8') as f:
            json.dump(results_native, f, indent=2, ensure_ascii=False)
        if accelerator.is_local_main_process:
            print(f"[GPU {rank}] 💾 Saved {len(results)} sample results to temp file")
    
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
            temp_json_path = os.path.join(args.output_dir, f"evaluation_results_rank{r}.json.tmp")
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
        print(f"{'='*60}\n")
        
        # 保存汇总结果
        if len(all_sample_results) > 0:
            results_json_path = os.path.join(args.output_dir, 'evaluation_results.json')
            with open(results_json_path, 'w', encoding='utf-8') as f:
                json.dump(all_sample_results, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved {len(all_sample_results)} sample results to {results_json_path}")
        else:
            print(f"⚠️ Warning: No sample results to save!")
    
    # 计算统计信息并打印（主进程汇总所有结果）
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # 从合并后的结果文件读取，计算统计信息
        results_json_path = os.path.join(args.output_dir, 'evaluation_results.json')
        if os.path.exists(results_json_path):
            with open(results_json_path, 'r', encoding='utf-8') as f:
                all_sample_results = json.load(f)
            
            # 汇总所有指标
            aggregated_rewards = {
                'mse_inside': [],
                'mse_outside': [],
                'mse_solution': [],
                'path_validity': [],
                'gt_cell_coverage': [],
                'background_violation': [],
                'avg': []
            }
            
            for result in all_sample_results:
                if 'rewards' in result:
                    rewards = result['rewards']
                    for key in aggregated_rewards.keys():
                        if key in rewards:
                            aggregated_rewards[key].append(rewards[key])
            
            if len(aggregated_rewards['avg']) > 0:
                print("\n" + "="*70)
                print("Evaluation Results (All GPUs Combined):")
                print("="*70)
                print(f"Total samples evaluated: {len(aggregated_rewards['avg'])}")
                print("\nReward Metrics:")
                print("-"*70)
                for metric_name in ['avg', 'mse_inside', 'mse_outside', 'mse_solution', 
                                   'path_validity', 'gt_cell_coverage', 'background_violation']:
                    values = aggregated_rewards[metric_name]
                    if len(values) > 0:
                        print(f"{metric_name:25s}: Mean={np.mean(values):.6f}, "
                              f"Std={np.std(values):.6f}, "
                              f"Min={np.min(values):.6f}, "
                              f"Max={np.max(values):.6f}")
                print("="*70)
                print(f"\nResults saved to: {results_json_path}")
                print(f"Total samples processed: {len(all_sample_results)}")
            else:
                print("No samples were successfully evaluated!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Qwen Image Edit model on maze dataset using maze rewards")
    
    # 数据集相关参数
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the maze dataset directory")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test"],
                        help="Dataset split to use")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (None for all)")
    parser.add_argument("--max_samples_per_size", type=int, default=None,
                        help="Maximum number of samples per size group to evaluate (None for all)")
    
    # 模型相关参数
    parser.add_argument("--model_path", type=str,
                        default="Qwen/Qwen-Image-Edit-2511",
                        help="Path to the Qwen Image Edit model")
    
    # Pipeline参数
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--true_cfg_scale", type=float, default=4.0,
                        help="True CFG scale")
    parser.add_argument("--negative_prompt", type=str, default=" ",
                        help="Negative prompt")
    parser.add_argument("--num_inference_steps", type=int, default=40,
                        help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="Guidance scale")
    
    # 输出相关参数
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--save_images", action="store_true",
                        help="Whether to save generated images")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log interval for progress updates")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for DataLoader (QwenImageEditPlusPipeline processes one at a time, but this controls DataLoader batching)")
    parser.add_argument("--num_attempts", type=int, default=1,
                        help="Number of generation attempts per sample (different random seeds)")
    
    args = parser.parse_args()
    main(args)