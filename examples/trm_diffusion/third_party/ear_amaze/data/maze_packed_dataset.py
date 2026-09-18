import os
import random
import torch
import pandas as pd
import base64
import io
from PIL import Image
from typing import Optional, List
import sys
script_path = os.path.abspath(__file__)
    

script_dir = os.path.dirname(script_path)

if script_dir not in sys.path:
    sys.path.append(script_dir)
from .dataset_base import PackedDataset
from .maze_dataset import MazeDataset
from .data_utils import pil_img2rgb
from .transforms import ImageTransform


class MazePackedDataset(PackedDataset):
    """
    A packed dataset for maze editing tasks that integrates with the SFT training framework.

    This class adapts the MazeDataset to work with the existing PackedDataset infrastructure,
    enabling efficient sequence packing and multi-modal training.
    """

    def __init__(
        self,
        dataset_name,
        tokenizer,
        local_rank=0,
        world_size=1,
        num_workers=1,
        data_status=None,
        maze_dataset_path=None,
        split='train',
        data_dir_list=None,  # For compatibility with PackedDataset interface
        transform=None,
        vit_transform=None,
        **kwargs
    ):
        """
        Initialize the maze packed dataset.

        Args:
            dataset_name: Name of the dataset group
            tokenizer: Tokenizer for text processing
            local_rank: Current process rank
            world_size: Total number of processes
            num_workers: Number of worker processes
            data_status: Status for resuming data loading
            maze_dataset_path: Path to the maze dataset directory
            split: 'train' or 'test'
            data_dir_list: For compatibility (not used)
            transform: Image transform for VAE processing
            vit_transform: Image transform for ViT processing
            **kwargs: Additional arguments
        """
        self.dataset_name = dataset_name
        self.split = split

        # Get maze dataset path from various sources
        if maze_dataset_path is None:
            # Try to get from data_config if available
            data_config = kwargs.get('data_config')
            if data_config and hasattr(data_config, 'maze_dataset_path'):
                maze_dataset_path = data_config.maze_dataset_path
            else:
                maze_dataset_path = "maze/"  # default fallback

        self.maze_dataset_path = maze_dataset_path

        # Initialize basic parameters
        self.expected_num_tokens = kwargs.get('expected_num_tokens', 32768)
        self.max_num_tokens_per_sample = kwargs.get('max_num_tokens_per_sample', 16384)
        self.prefer_buffer_before = kwargs.get('prefer_buffer_before', 16384)
        self.max_num_tokens = kwargs.get('max_num_tokens', 36864)
        self.max_buffer_size = kwargs.get('max_buffer_size', 50)
        self.tokenizer = tokenizer
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.use_flex = kwargs.get('use_flex', False)
        self.interpolate_pos = kwargs.get('interpolate_pos', False)

        # Set special tokens with defaults
        special_tokens = kwargs.get('special_tokens', {})
        self.bos_token_id = special_tokens.get('bos_token_id', 1)
        self.eos_token_id = special_tokens.get('eos_token_id', 2)
        self.start_of_image = special_tokens.get('start_of_image', 151857)
        self.end_of_image = special_tokens.get('end_of_image', 151858)
        self.start_of_latent = special_tokens.get('start_of_latent', 151859)
        self.end_of_latent = special_tokens.get('end_of_latent', 151860)

        # Set other attributes from special_tokens
        for k, v in special_tokens.items():
            if not hasattr(self, k):
                setattr(self, k, v)

        # Initialize position ID functions
        if self.interpolate_pos:
            from .data_utils import get_flattened_position_ids_interpolate
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            from .data_utils import get_flattened_position_ids_extrapolate
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        # Set transforms and ensure they're properly initialized
        self.vae_transform = transform
        self.vit_transform = vit_transform

        # Set transform attributes for later use
        if hasattr(self.vae_transform, 'stride'):
            self.vae_stride = self.vae_transform.stride
        else:
            self.vae_stride = 16  # default VAE stride

        if hasattr(self.vit_transform, 'stride'):
            self.vit_stride = self.vit_transform.stride
        else:
            self.vit_stride = 14  # default ViT stride

        # Set other important attributes
        self.max_num_patch_per_side = kwargs.get('max_num_patch_per_side', 70)
        self.max_latent_size = kwargs.get('max_latent_size', 32)
        self.text_cond_dropout_prob = kwargs.get('text_cond_dropout_prob', 0.0)
        self.vae_cond_dropout_prob = kwargs.get('vae_cond_dropout_prob', 0.0)
        self.vit_cond_dropout_prob = kwargs.get('vit_cond_dropout_prob', 0.0)

        # Load maze dataset
        if maze_dataset_path is None:
            raise ValueError("maze_dataset_path must be provided")

        self.maze_dataset = MazeDataset(maze_dataset_path, split=split)
        self.overfit_sample = self.maze_dataset[0] 
        self.maze_iterator = None
        self.reset_iterator()

    def reset_iterator(self):
        """Reset the maze dataset iterator."""
        indices = list(range(len(self.maze_dataset)))

        # Distributed sampling
        indices_per_rank = len(indices) // self.world_size
        start_idx = self.local_rank * indices_per_rank
        end_idx = start_idx + indices_per_rank
        if self.local_rank == self.world_size - 1:
            end_idx = len(indices)

        self.local_indices = indices[start_idx:end_idx]
        self.current_idx = 0

    def get_next_maze_sample(self):
        """Get next sample from maze dataset."""
        return self.overfit_sample
        if self.current_idx >= len(self.local_indices):
            # Shuffle and reset
            random.shuffle(self.local_indices)
            self.current_idx = 0

        idx = self.local_indices[self.current_idx]
        self.current_idx += 1

        return self.maze_dataset[idx]

    def maze_sample_to_sequence_plan(self, maze_sample):
        """
        Convert a maze sample to the sequence plan format expected by PackedDataset.
        For image editing task: uses m_original_img (marked original image) as input.

        Args:
            maze_sample: Sample from MazeDataset

        Returns:
            Dict with sequence plan for training
        """
        m_original_img = maze_sample['m_original_img']
        sol_img = maze_sample['sol_img']
        mask_img = maze_sample['mask_img']
        prompt = maze_sample['prompt']

        if m_original_img is None or sol_img is None:
            return None

        # Convert to RGB
        m_original_img = pil_img2rgb(m_original_img)
        sol_img = pil_img2rgb(sol_img)

        # Process mask image
        mask_tensor = None
        if mask_img is not None:
            mask_img = pil_img2rgb(mask_img)
            if self.vae_transform is not None:
                mask_tensor = self.vae_transform(mask_img)
                mask_tensor = (mask_tensor.mean(dim=0, keepdim=True) > 0.5).float()  # [1, H, W]

        else:
            print(f"[Dataset Debug] mask_img is None!")

        if mask_tensor is None:
            print(f"[Dataset Debug] mask_tensor is None after processing!")

        # Create sequence plan
        sequence_plan = []
        image_tensor_list = []
        text_ids_list = []

        if self.vit_transform is not None:
            vit_image_tensor = self.vit_transform(m_original_img)
            image_tensor_list.append(vit_image_tensor)
            sequence_plan.append({
                'type': 'vit_image',
                'enable_cfg': 1,
                'loss': 0,
                'special_token_loss': 0,
                'special_token_label': None,
            })

        if self.vae_transform is not None:
            vae_image_tensor = self.vae_transform(m_original_img)
            image_tensor_list.append(vae_image_tensor)
            sequence_plan.append({
                'type': 'vae_image',
                'enable_cfg': 1,
                'loss': 0,
                'special_token_loss': 0,
                'special_token_label': None,
            })

        # 3. Add instruction text
        # Use instruction from dataset, no fallback needed since dataset provides instructions
        prompt = maze_sample['prompt']
        text_ids = self.tokenizer.encode(prompt)
        text_ids_list.append(text_ids)
        sequence_plan.append({
            'type': 'text',
            'enable_cfg': 1,
            'loss': 1,  # Set to 1 to compute CE loss for language modeling
            'special_token_loss': 0,
            'special_token_label': None,
        })

        # 4. Add target solution image
        if self.vae_transform is not None:
            target_image_tensor = self.vae_transform(sol_img)
            image_tensor_list.append(target_image_tensor)
            sequence_plan.append({
                'type': 'vae_image',
                'enable_cfg': 0,
                'loss': 1,
                'special_token_loss': 0,
                'special_token_label': None,
            })

        # Calculate total tokens
        num_tokens = len(text_ids)

        # Add special tokens for image sections
        vit_token_count = 0
        vae_token_count = 0

        for i, (img_tensor, plan_item) in enumerate(zip(image_tensor_list, [item for item in sequence_plan if item['type'] in ['vit_image', 'vae_image']])):
            height, width = img_tensor.shape[1:]

            if plan_item['type'] == 'vit_image':
                # ViT tokens: 2 special tokens + image patches
                stride = self.vit_stride
                num_patches = (height // stride) * (width // stride)
                vit_token_count = 2 + num_patches  # start_of_image + patches + end_of_image
                num_tokens += vit_token_count
            elif plan_item['type'] == 'vae_image':
                # VAE tokens: only count if it's a loss target
                if plan_item['loss'] == 1:
                    stride = self.vae_stride
                    num_patches = (height // stride) * (width // stride)
                    vae_token_count = num_patches  # VAE images don't need special tokens in current implementation
                    num_tokens += vae_token_count

        return {
            'sequence_plan': sequence_plan,
            'text_ids_list': text_ids_list,
            'image_tensor_list': image_tensor_list,
            'mask_tensor': mask_tensor,
            'num_tokens': num_tokens,
        }

    def set_sequence_status(self):
        """Initialize sequence status dictionary for packing."""
        sequence_status = dict(
            curr                        = 0,
            sample_lens                 = list(),
            packed_position_ids         = list(),
            nested_attention_masks      = list(),
            split_lens                  = list(),
            attn_modes                  = list(),
            packed_text_ids             = list(),
            packed_text_indexes         = list(),
            packed_label_ids            = list(),
            ce_loss_indexes             = list(),
            ce_loss_weights             = list(),
            vae_image_tensors           = list(),
            packed_latent_position_ids  = list(),
            vae_latent_shapes           = list(),
            packed_vae_token_indexes    = list(),
            packed_timesteps            = list(),
            mse_loss_indexes            = list(),
            mask_tensors                = list(),
            packed_vit_tokens           = list(),
            vit_token_seqlens           = list(),
            packed_vit_position_ids     = list(),
            packed_vit_token_indexes    = list(),
        )
        return sequence_status

    def to_tensor(self, sequence_status):
        """Convert sequence status to tensor format for training."""
        data = dict(
            sequence_length=sum(sequence_status['sample_lens']),
            sample_lens=sequence_status['sample_lens'],
            packed_text_ids=torch.tensor(sequence_status['packed_text_ids']),
            packed_text_indexes=torch.tensor(sequence_status['packed_text_indexes']),
            packed_position_ids=torch.tensor(sequence_status['packed_position_ids']),
        )
        if not self.use_flex:
            data['nested_attention_masks'] = sequence_status['nested_attention_masks']
        else:
            sequence_len = data['sequence_length']
            pad_len = self.max_num_tokens - sequence_len
            data['split_lens'] = sequence_status['split_lens'] + [pad_len]
            data['attn_modes'] = sequence_status['attn_modes'] + ['causal']
            data['sample_lens'] += [pad_len]

        # Handle VAE images
        if len(sequence_status['vae_image_tensors']) > 0:
            image_tensors = sequence_status.pop('vae_image_tensors')
            image_sizes = [item.shape for item in image_tensors]
            max_image_size = [max(item) for item in list(zip(*image_sizes))]
            padded_images = torch.zeros(size=(len(image_tensors), *max_image_size))
            for idx, image in enumerate(image_tensors):
                padded_images[idx, :image.size(0), :image.size(1), :image.size(2)] = image
            data['padded_images'] = padded_images
            data['patchified_vae_latent_shapes'] = sequence_status['vae_latent_shapes']
            data['packed_latent_position_ids'] = torch.cat(sequence_status['packed_latent_position_ids'], dim=0)
            data['packed_vae_token_indexes'] = torch.tensor(sequence_status['packed_vae_token_indexes'])

            # Handle mask tensors - should have same length as image_tensors
            if len(sequence_status['mask_tensors']) > 0:
                mask_tensors = sequence_status['mask_tensors']
                # Ensure mask_tensors and image_tensors have same length
                assert len(mask_tensors) == len(image_tensors), \
                    f"Mask count {len(mask_tensors)} != image count {len(image_tensors)}"

                # Pad masks to match image sizes
                padded_masks = torch.zeros(size=(len(mask_tensors), 1, max_image_size[1], max_image_size[2]))
                for idx, mask in enumerate(mask_tensors):
                    if mask is not None:
                        padded_masks[idx, :, :mask.size(1), :mask.size(2)] = mask
                    # If mask is None, keep it as zeros (will be ignored in model)
                data['padded_masks'] = padded_masks
              #  print(f"[Dataset Debug] Added padded_masks: shape={padded_masks.shape}, "
               #       f"sum={padded_masks.sum().item():.1f}, "
                #      f"num_images={len(image_tensors)}, num_masks={len(mask_tensors)}")
            else:
                print(f"[Dataset Debug] mask_tensors list is EMPTY! "
                      f"num_images={len(image_tensors)}")

            # Add other VAE-related data
            if sequence_status['mse_loss_indexes']:
                data['mse_loss_indexes'] = torch.tensor(sequence_status['mse_loss_indexes'])
                data['packed_timesteps'] = torch.tensor(sequence_status['packed_timesteps'])

        # Handle ViT tokens
        if len(sequence_status['packed_vit_tokens']) > 0:
            data['packed_vit_tokens'] = torch.cat(sequence_status['packed_vit_tokens'], dim=0)
            data['vit_token_seqlens'] = torch.tensor(sequence_status['vit_token_seqlens'])
            data['packed_vit_position_ids'] = torch.cat(sequence_status['packed_vit_position_ids'], dim=0)
            data['packed_vit_token_indexes'] = torch.tensor(sequence_status['packed_vit_token_indexes'])

        # Handle loss indexes
        # Use len check like dataset_base.py to ensure consistency
        if len(sequence_status['packed_label_ids']) > 0:
            data['packed_label_ids'] = torch.tensor(sequence_status['packed_label_ids'])
            data['ce_loss_indexes'] = torch.tensor(sequence_status['ce_loss_indexes'])
            data['ce_loss_weights'] = torch.tensor(sequence_status['ce_loss_weights'])

        return data

    def pack_sequence(self, sample, sequence_status):
        """Pack a single sample into the sequence status."""
        from .data_utils import patchify, len2weight, prepare_attention_mask_per_sample

        image_tensor_list = sample['image_tensor_list']
        text_ids_list = sample['text_ids_list']
        sequence_plan = sample['sequence_plan']
        mask_tensor = sample.get('mask_tensor', None)

        split_lens, attn_modes = list(), list()
        curr = sequence_status['curr']
        curr_rope_id = 0
        sample_lens = 0
        vae_image_idx = 0  # Track VAE image index for mask alignment

        for item in sequence_plan:
            split_start = item.get('split_start', True)
            if split_start:
                curr_split_len = 0

            if item['type'] == 'text':
                text_ids = text_ids_list.pop(0)
                # Apply text dropout if configured
                if hasattr(self, 'text_cond_dropout_prob') and item['enable_cfg'] == 1:
                    if random.random() < getattr(self, 'text_cond_dropout_prob', 0.0):
                        continue

                shifted_text_ids = [self.bos_token_id] + text_ids
                sequence_status['packed_text_ids'].extend(shifted_text_ids)
                sequence_status['packed_text_indexes'].extend(range(curr, curr + len(shifted_text_ids)))

                if item['loss'] == 1:
                    sequence_status['ce_loss_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                    sequence_status['ce_loss_weights'].extend([len2weight(len(shifted_text_ids))] * len(shifted_text_ids))
                    sequence_status['packed_label_ids'].extend(text_ids + [self.eos_token_id])

                curr += len(shifted_text_ids)
                curr_split_len += len(shifted_text_ids)

                # Add end token
                sequence_status['packed_text_ids'].append(self.eos_token_id)
                sequence_status['packed_text_indexes'].append(curr)
                if item['special_token_loss'] == 1:
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                # Update sequence status
                attn_modes.append("causal")
                sequence_status['packed_position_ids'].extend(range(curr_rope_id, curr_rope_id + curr_split_len))
                curr_rope_id += curr_split_len

            elif item['type'] == 'vit_image':
                image_tensor = image_tensor_list.pop(0)
                # Apply VIT dropout if configured
                if hasattr(self, 'vit_cond_dropout_prob') and item['enable_cfg'] == 1:
                    if random.random() < getattr(self, 'vit_cond_dropout_prob', 0.0):
                        curr_rope_id += 1
                        continue

                # Add start token
                sequence_status['packed_text_ids'].append(self.start_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                # Process image
                if hasattr(self, 'vit_transform') and self.vit_transform:
                    patch_size = self.vit_transform.stride
                else:
                    patch_size = 14  # default

                vit_tokens = patchify(image_tensor, patch_size)
                num_img_tokens = vit_tokens.shape[0]
                sequence_status['packed_vit_token_indexes'].extend(range(curr, curr + num_img_tokens))
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                sequence_status['packed_vit_tokens'].append(vit_tokens)
                sequence_status['vit_token_seqlens'].append(num_img_tokens)

                # Get position IDs
                max_patches = getattr(self, 'max_num_patch_per_side', 70)
                sequence_status['packed_vit_position_ids'].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1), image_tensor.size(2),
                        patch_size, max_num_patches_per_side=max_patches
                    )
                )

                # Add end token
                sequence_status['packed_text_ids'].append(self.end_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                if item['special_token_loss'] == 1:
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                # Update sequence status
                attn_modes.append("full")
                sequence_status['packed_position_ids'].extend([curr_rope_id] * curr_split_len)
                curr_rope_id += 1

            elif item['type'] == 'vae_image':
                image_tensor = image_tensor_list.pop(0)
                # Apply VAE dropout if configured
                if hasattr(self, 'vae_cond_dropout_prob') and item['enable_cfg'] == 1:
                    if random.random() < getattr(self, 'vae_cond_dropout_prob', 0.0):
                        curr_rope_id += 1
                        vae_image_idx += 1
                        # Still need to add None mask to maintain alignment
                        sequence_status['mask_tensors'].append(None)

                        # Don't add mask since image is also not added (skip this image entirely)
                        continue

                # Add a <|startofimage|> token
                sequence_status['packed_text_ids'].append(self.start_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                # Preprocess image
                sequence_status['vae_image_tensors'].append(image_tensor)

                # Get patch size for VAE
                if hasattr(self, 'vae_transform') and self.vae_transform:
                    patch_size = self.vae_transform.stride
                else:
                    patch_size = 16  # default

                # Add packed_latent_position_ids and vae_latent_shapes
                sequence_status['packed_latent_position_ids'].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1), image_tensor.size(2),
                        patch_size,
                        max_num_patches_per_side=self.max_latent_size
                    )
                )
                H, W = image_tensor.shape[1:]
                h = H // patch_size
                w = W // patch_size
                sequence_status['vae_latent_shapes'].append((h, w))

                num_img_tokens = w * h
                sequence_status['packed_vae_token_indexes'].extend(range(curr, curr + num_img_tokens))

                # Generate timesteps for all VAE images
                # For loss computation (target image), use random timesteps
                # For conditioning images (loss=0), use -inf to indicate clean image
                if item['loss'] == 1:
                    sequence_status['mse_loss_indexes'].extend(range(curr, curr + num_img_tokens))
                    # Generate random timesteps for diffusion
                    timesteps = torch.randint(0, 1000, (num_img_tokens,))
                    sequence_status['packed_timesteps'].extend(timesteps.tolist())

                    # Add mask for target image (only target images have meaningful masks)
                    # This should be the solution image based on sequence_plan
                    sequence_status['mask_tensors'].append(mask_tensor if mask_tensor is not None else None)
                else:
                    # For conditioning images, use -inf to indicate clean image (no noise)
                    sequence_status['packed_timesteps'].extend([float('-inf')] * num_img_tokens)
                    # Add None mask for conditioning images to maintain alignment
                    sequence_status['mask_tensors'].append(None)

                vae_image_idx += 1

                curr += num_img_tokens
                curr_split_len += num_img_tokens

                # Add a <|endofimage|> token
                sequence_status['packed_text_ids'].append(self.end_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                if item.get('special_token_loss', 0) == 1:
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item.get('special_token_label', self.end_of_image))
                curr += 1
                curr_split_len += 1

                # Update sequence status
                attn_modes.append("full")
                sequence_status['packed_position_ids'].extend([curr_rope_id] * curr_split_len)
                curr_rope_id += 1

            split_end = item.get('split_end', True)
            if split_end:
                split_lens.append(curr_split_len)
                sample_lens += curr_split_len

        sequence_status['curr'] = curr
        sequence_status['sample_lens'].append(sample_lens)
        # prepare attention mask
        if not self.use_flex:
            sequence_status['nested_attention_masks'].append(
                prepare_attention_mask_per_sample(split_lens, attn_modes)
            )
        else:
            sequence_status['split_lens'].extend(split_lens)
            sequence_status['attn_modes'].extend(attn_modes)

    def set_epoch(self, seed):
        """Set epoch for reproducible shuffling."""
        random.seed(seed + self.local_rank)
        self.reset_iterator()

    def __iter__(self):
        """Iterate over packed sequences."""
        # Buffer for accumulating samples
        buffer = []
        sequence_status = self.set_sequence_status()
        batch_data_indexes = []

        while True:
            # Try to get a sample from buffer first (if buffer is not full)
            sample = None
            if len(buffer) > 0 and sequence_status['curr'] < self.prefer_buffer_before:
                sample = buffer.pop(0)
            else:
                # Get new sample from maze dataset
                maze_sample = self.get_next_maze_sample()
                sample = self.maze_sample_to_sequence_plan(maze_sample)

                if sample is None:
                    continue

                 # 添加token数量打印
                # if not hasattr(self, '_iter_debug_count'):
                #     self._iter_debug_count = 0
                # self._iter_debug_count += 1
                # if self._iter_debug_count <= 20:  # 打印前20个样本
                #     print(f"[Iter Debug] Sample {self._iter_debug_count}: "
                #           f"num_tokens={sample['num_tokens']}, "
                #           f"current_batch_tokens={sequence_status['curr']}")

                if sample['num_tokens'] > self.max_num_tokens_per_sample:
                    if len(buffer) < self.max_buffer_size:
                        buffer.append(sample)
                    continue

                # If adding this sample would exceed max tokens, yield current batch
                if sequence_status['curr'] + sample['num_tokens'] > self.max_num_tokens:
                    if sequence_status['curr'] > 0:
                        data = self.to_tensor(sequence_status)
                        data['batch_data_indexes'] = batch_data_indexes
                        yield data
                        sequence_status = self.set_sequence_status()
                        batch_data_indexes = []

                    # Add to buffer if still too long
                    if sample['num_tokens'] > self.max_num_tokens:
                        if len(buffer) < self.max_buffer_size:
                            buffer.append(sample)
                        continue

            # Pack the sample
            if sample is not None:
                self.pack_sequence(sample, sequence_status)
                batch_data_indexes.append({
                    'dataset_name': 'maze_edit',
                    'worker_id': self.local_rank,
                    'data_indexes': [self.current_idx - 1],
                })

                # Yield if we have enough tokens
                if sequence_status['curr'] >= self.expected_num_tokens:
                    data = self.to_tensor(sequence_status)
                    data['batch_data_indexes'] = batch_data_indexes
                    yield data
                    sequence_status = self.set_sequence_status()
                    batch_data_indexes = []
    def __len__(self):
        """Return approximate length of the dataset."""
        return len(self.maze_dataset) // self.world_size

    def __getitem__(self, idx):
        """Get item by index (for compatibility)."""
        return self.maze_dataset[idx]

