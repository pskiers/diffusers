# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import List, Dict, Optional, Union, Any

from PIL import Image
import torch

from bagel.data.data_utils import pil_img2rgb
from bagel.modeling.bagel.qwen2_navit import NaiveCache



VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''


class InterleaveInferencer:
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        
    def init_gen_context(self): 
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        # used for interleave data, currently only support 1 data inference, 

        # Context准备阶段使用eval模式（特征提取，不需要训练）
        was_training = self.model.training
        self.model.eval()
        
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=[text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )

        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        # 恢复原始训练状态
        if was_training:
            self.model.train()
        
        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True):
        # used for interleave data, currently only support 1 data inference, 

        assert vae or vit
        
        # Context准备阶段使用eval模式（特征提取，不需要训练）
        was_training = self.model.training
        self.model.eval()
        
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']
        
        # 【新增 1】获取 VAE 模型当前的精度 (现在是 float32 了)
        vae_dtype = next(self.vae_model.parameters()).dtype
        # 获取模型所在的设备
        model_device = next(self.model.parameters()).device

        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vae_transform, 
                new_token_ids=self.new_token_ids,
            )
            
            # 【新增 2】关键修复：遍历输入，把图片 Tensor 强转为 VAE 的精度
            # 这样无论外面是 bf16 还是 fp16，进 VAE 之前都会变成 float32
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    # 'padded_images' 是实际的像素数据，必须转 float32
                    if k == 'padded_images':
                        generation_input[k] = v.to(model_device, dtype=vae_dtype)
                    else:
                        # 其他索引类的 tensor 只需移动设备
                        generation_input[k] = v.to(model_device)

            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
        
        if vit:
            ## update vit
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vit_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        # 恢复原始训练状态
        if was_training:
            self.model.train()
        
        return gen_context

    @torch.no_grad()
    def update_context_text_batch(self, texts: List[str], gen_context):
        """
        批量更新文本context
        """
        was_training = self.model.training
        self.model.eval()
        
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        
        # 直接传入文本列表，底层模型已支持批量
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=texts,  # 传入列表
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )
        
        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        if was_training:
            self.model.train()
        
        return gen_context

    @torch.no_grad()
    def update_context_image_batch(self, images: List[Image.Image], gen_context, vae=True, vit=True):
        """
        批量更新图像context
        """
        assert vae or vit
        
        was_training = self.model.training
        self.model.eval()
        
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        
        vae_dtype = next(self.vae_model.parameters()).dtype
        model_device = next(self.model.parameters()).device
        
        if vae:
            # 直接传入图像列表，底层模型已支持批量
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=images,  # 传入列表
                transforms=self.vae_transform, 
                new_token_ids=self.new_token_ids,
            )
            
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    if k == 'padded_images':
                        generation_input[k] = v.to(model_device, dtype=vae_dtype)
                    else:
                        generation_input[k] = v.to(model_device)
            
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
        
        if vit:
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=images,  # 传入列表
                transforms=self.vit_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)
        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        if was_training:
            self.model.train()
        
        return gen_context

    def gen_image(
        self, 
        image_shape, 
        gen_context, 
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,

        cfg_text_precontext=None, 
        cfg_img_precontext=None, 
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        
        num_timesteps=50, 
        timestep_shift=3.0,

        # for grpo learn
        learn=False,
        sample=None,
        grpo_config=None,
        accelerator=None,
        optimizer=None,
        transformer=None,
        noise_level=0.7,
        generators=None,
    ):
        # Do not set the initial latent to be the same for the same prompt in eval mode
        if noise_level==0:
            generators=None
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            image_sizes=[image_shape], 
            new_token_ids=self.new_token_ids,
            generators=generators
        ) 
        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )
        if learn:
            # =============== 修改：只对connector设置train模式，LLM保持eval ===============
            # 确保 vae2llm 和 llm2vae 处于训练模式
            # if hasattr(self.model, 'vae2llm'):
            #     self.model.vae2llm.train()
            # if hasattr(self.model, 'llm2vae'):
            #     self.model.llm2vae.train()

            # # 确保 language_model 保持 eval 模式（使用 forward_inference）
            # if hasattr(self.model, 'language_model'):
            #     self.model.language_model.eval()
            # ===========================================================================

            clipfrac, clipfrac_gt_one, clipfrac_lt_one, policy_loss, kl_loss, loss, mse_loss, grad_norm = self.model.generate_image_learn(
                sample=sample,
                grpo_config=grpo_config,
                accelerator=accelerator,
                optimizer=optimizer,
                transformer=transformer,
                past_key_values=past_key_values,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_img_past_key_values=cfg_img_past_key_values,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                timestep_shift=timestep_shift,
                **generation_input,
                cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
                cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
                cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
                cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
                cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
                cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
                cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
                cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
                noise_level=noise_level,
            )
            return {
                "clipfrac": clipfrac, 
                "clipfrac_gt_one": clipfrac_gt_one,
                "clipfrac_lt_one": clipfrac_lt_one,
                "policy_loss": policy_loss, 
                "kl_loss": kl_loss,
                "loss": loss,
                "mse_loss": mse_loss,  # 用于监控 latent 预测质量
                "grad_norm": grad_norm,  # gradient norm
            }
        else:
            unpacked_latent, all_latents, all_log_probs, timesteps = self.model.generate_image(
                past_key_values=past_key_values,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_img_past_key_values=cfg_img_past_key_values,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                timestep_shift=timestep_shift,
                **generation_input,
                cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
                cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
                cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
                cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
                cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
                cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
                cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
                cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
                noise_level=noise_level,
                sample_sde_window_size=grpo_config.sample.sde_window_size,
                sample_sde_window_range=grpo_config.sample.sde_window_range,
                process_index=getattr(accelerator, 'process_index', 0),
                device=getattr(accelerator, 'device', 'cuda'),
            )
            image = self.decode_image(unpacked_latent[0].float(), image_shape)
            return {
                "image": image,
                "all_latents": all_latents,
                "all_log_probs": all_log_probs,
                "timesteps": timesteps
            }

    def gen_image_batch(
        self, 
        image_shapes: List[tuple],  # 图像尺寸列表
        gen_context, 
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_text_precontext=None, 
        cfg_img_precontext=None, 
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        num_timesteps=50, 
        timestep_shift=3.0,
        learn=False,
        sample=None,
        grpo_config=None,
        accelerator=None,
        optimizer=None,
        transformer=None,
        noise_level=0.7,
        generators=None,
        is_intermediate_save=False,
    ):
    
        """
        批量生成图像
        """
        # 关键修复：即使 noise_level=0，也要为每个样本创建独立的 generator
        # 确保批量推理时每个样本使用独立的随机状态，避免样本间的相关性
        # 
        # 问题分析：
        # - 串行推理：每个样本独立调用 gen_image，使用不同的随机状态，初始噪声是独立的
        # - 批量推理（修复前）：所有样本连续使用全局随机状态生成初始噪声，导致样本间存在相关性
        # 
        # 解决方案：为批量推理中的每个样本创建独立的 generator，确保初始噪声的独立性
        batch_size = len(image_shapes)
        if generators is None:
            # 为每个样本创建独立的 generator
            generators = []
            # 使用一个随机数作为基础种子，然后为每个样本使用 base_seed + index
            # 这样既保证了每个样本的独立性，又确保了不同 batch 之间的随机性
            base_seed = torch.randint(0, 2**31, (1,)).item()
            for i in range(batch_size):
                g = torch.Generator()
                g.manual_seed(base_seed + i)
                generators.append(g)
        
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        
        # 传入图像尺寸列表
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            image_sizes=image_shapes,  # 传入列表
            new_token_ids=self.new_token_ids,
            generators=generators
        )
        
        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=image_shapes,  # 传入列表
        )
        
        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=image_shapes,  # 传入列表
        )
        
        if learn:
            # learn模式暂不支持批量
            raise NotImplementedError("gen_image_batch does not support learn mode yet")
        else:
            unpacked_latent, all_latents, all_log_probs, timesteps = self.model.generate_image(
                past_key_values=past_key_values,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_img_past_key_values=cfg_img_past_key_values,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                timestep_shift=timestep_shift,
                **generation_input,
                cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
                cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
                cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
                cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
                cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
                cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
                cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
                cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
                noise_level=noise_level,
                sample_sde_window_size=grpo_config.sample.sde_window_size,
                sample_sde_window_range=grpo_config.sample.sde_window_range,
                process_index=getattr(accelerator, 'process_index', 0),
                device=getattr(accelerator, 'device', 'cuda'),
            )
            
            # unpacked_latent 是列表，包含多个latent
            # 需要为每个latent解码图像
            images = []
            for i, latent in enumerate(unpacked_latent):
                image = self.decode_image(latent.float(), image_shapes[i])
                images.append(image)
            
            # 解包 all_latents：将每个步骤的 packed latent 解包为每个样本的 latent
            # all_latents 是列表，每个元素是 packed 格式的 tensor
            # 需要根据 packed_seqlens 解包
            unpacked_all_latents = []
            if len(all_latents) > 0 and 'packed_seqlens' in generation_input:
                packed_seqlens = generation_input['packed_seqlens']
                for step_latent in all_latents:
                    # 每个步骤的 latent 也是 packed 格式，需要解包
                    step_unpacked = step_latent.split((packed_seqlens - 2).tolist())
                    unpacked_all_latents.append(step_unpacked)
            else:
                # 如果没有 packed_seqlens 或 all_latents 为空，保持原样
                unpacked_all_latents = all_latents

            # ================= [新增] 解码中间步骤图像 START =================
            intermediate_images = []
            if unpacked_all_latents and is_intermediate_save:
                # unpacked_all_latents 结构: List[step] -> List[batch_sample_tensor]
                for step_idx, step_latents in enumerate(unpacked_all_latents):
                    step_imgs = []
                    # 遍历当前步的每个样本
                    for batch_idx, latent in enumerate(step_latents):
                        # 使用现有的 decode_image 方法，注意转 float()
                        # image_shapes[batch_idx] 对应当前样本的尺寸
                        img = self.decode_image(latent.float(), image_shapes[batch_idx])
                        step_imgs.append(img)
                    intermediate_images.append(step_imgs)
            # ================= [新增] 解码中间步骤图像 END ===================
            
            return {
                "images": images,  # 返回图像列表
                "all_latents": unpacked_all_latents,  # 返回解包后的 latent 列表（每个步骤是样本列表）
                "all_log_probs": all_log_probs,
                "timesteps": timesteps,
                "intermediate_images": intermediate_images  # <--- [新增] 返回中间图像
            }

        
    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample
        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].float()
        return image

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0, accelerator=None):
        # 文本生成使用eval模式
        was_training = self.model.training
        self.model.eval()
        
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        
        # [修改] 确保所有 tensor 都在正确的设备上
        # 获取 language_model 所在的设备（因为 embedding 层在这里）
        target_model = self.model.language_model
        if accelerator is not None:
            # 如果提供了 accelerator，使用 accelerator 的设备
            model_device = accelerator.device
        else:
            # 否则，尝试从 target_model 获取设备
            try:
                model_device = next(target_model.parameters()).device
            except:
                # 如果失败，使用默认的模型设备
                model_device = next(self.model.parameters()).device
        
        for key, value in generation_input.items():
            if torch.is_tensor(value):
                generation_input[key] = value.to(model_device)
        
        # [修改] 定义上下文管理器
        # 如果传入了 accelerator，且 language_model 被 FSDP 包装了，就临时解包
        # 注意：这里我们解包 self.model.language_model，因为在主脚本中是它被 prepare 的
        from contextlib import nullcontext
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        
        # 检查 language_model 是否被 FSDP 包装
        target_model = self.model.language_model
        context = nullcontext()
        
        if accelerator is not None:
            # 检查是否是 FSDP 模块
            if isinstance(target_model, FSDP):
                # 使用 FSDP 的 summon_full_params 上下文管理器
                # 这会临时 gather 所有分片的参数，执行完后自动释放
                context = FSDP.summon_full_params(target_model, writeback=False, recurse=True)
            else:
                # 如果不是 FSDP，尝试使用 accelerator.unwrap_model
                # 注意：unwrap_model 返回的是解包后的模型，不是上下文管理器
                # 但我们可以通过检查来确保模型可用
                try:
                    unwrapped = accelerator.unwrap_model(target_model)
                    # 如果 unwrap 成功，使用 nullcontext（因为不需要特殊处理）
                    context = nullcontext()
                except:
                    context = nullcontext()
        
        with context:
            # 在这个 context 内部，参数会被临时拼凑成完整的 2D 形状
            # 执行完后会自动释放，恢复成切分状态节省显存
            unpacked_latent = self.model.generate_text(
                past_key_values=past_key_values,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=self.new_token_ids['eos_token_id'],
                **generation_input,
            )
        
        output = self.tokenizer.decode(unpacked_latent[:,0])
        # 防止 output 为空或格式不对的简单处理
        try:
            output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        except:
            pass  # 保持原样或者做其他处理
        
        # 恢复原始训练状态
        if was_training:
            self.model.train()
        
        return output
        
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,

        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        learn=False,
        sample=None,
        grpo_config=None,
        accelerator=None,
        optimizer=None,
        transformer=None,
        noise_level=0.7,
        generators=None,
    ) -> List[Union[str, Image.Image]]:

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT 
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output, vit=True)

                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n, accelerator=accelerator)
                output_list.append(gen_text)

            else:
                if think:
                    gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n, accelerator=accelerator)
                    gen_context = self.update_context_text(gen_text, gen_context)
                    output_list.append(gen_text)

                img = self.gen_image(
                    image_shapes, 
                    gen_context, 
                    cfg_text_precontext=cfg_text_context, 
                    cfg_img_precontext=cfg_img_context,

                    cfg_text_scale=cfg_text_scale, 
                    cfg_img_scale=cfg_img_scale, 
                    cfg_interval=cfg_interval, 
                    timestep_shift=timestep_shift, 
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,

                    # for grpo learn
                    learn=learn,
                    sample=sample,
                    grpo_config=grpo_config,
                    accelerator=accelerator,
                    optimizer=optimizer,
                    transformer=transformer,
                    noise_level=noise_level,
                    generators=generators,
                )

                output_list.append(img)

        return output_list
    
    @torch.no_grad()
    def batch_image_edit(
        self,
        images: List[Image.Image],
        texts: List[str],
        noise_level=0.7,
        grpo_config=None,
        accelerator=None,
        num_timesteps=50,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        generators=None,
        is_intermediate_save=False,
        think=False,  # <--- [新增] 开启思考模式
        max_think_tokens=1024,  # <--- [新增] 思考的最大长度
    ) -> Dict[str, Any]:  # 返回类型改为 Dict 以包含 thoughts
        """
        批量图像编辑：支持先思考后生成
        
        Args:
            images: 输入图像列表
            texts: 文本提示列表
            think: 是否开启思考模式
            max_think_tokens: 思考的最大token数
            其他参数同 gen_image
        
        Returns:
            包含生成图像和思考文本的字典
        """
        assert len(images) == len(texts), f"images and texts must have the same length: {len(images)} vs {len(texts)}"
        
        batch_size = len(images)
        
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            # 1. 初始化 Context
            gen_context = {
                'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
                'kv_lens': [0] * batch_size,
                'ropes': [0] * batch_size
            }
            
            # [关键步骤 1] 如果开启思考，先注入 System Prompt
            if think:
                # 批量注入 System Prompt
                system_prompts = [GEN_THINK_SYSTEM_PROMPT] * batch_size
                gen_context = self.update_context_text_batch(system_prompts, gen_context)

            # cfg_img_context (用于 Classifier-Free Guidance 的图像条件分支) 
            # 通常不需要 System Prompt 或者保持与 gen_context 一致，这里我们深拷贝基础状态
            cfg_img_context = deepcopy(gen_context)

            # 2. 处理图像输入 (Image Input)
            # 预处理图像
            from flow_grpo.bagel.data.data_utils import pil_img2rgb
            processed_images = []
            image_shapes = []
            for img in images:
                processed_img = self.vae_transform.resize_transform(pil_img2rgb(img))
                processed_images.append(processed_img)
                image_shapes.append(processed_img.size[::-1])  # (H, W)
            
            # 更新 Context (Image)
            gen_context = self.update_context_image_batch(processed_images, gen_context, vae=True, vit=True)
            
            # cfg_text_context (CFG 的文本条件分支): 只包含图像，不包含后续的用户文本
            cfg_text_context = deepcopy(gen_context)
            
            # 3. 处理用户文本输入 (User Prompt)
            gen_context = self.update_context_text_batch(texts, gen_context)
            # cfg_img_context 也需要包含用户文本
            cfg_img_context = self.update_context_text_batch(texts, cfg_img_context)

            thoughts = []
            # [关键步骤 2] 生成思考过程 (Text Generation)
            if think:
                # 临时切换到 eval 模式进行文本生成
                was_training = self.model.training
                self.model.eval()
                
                # 准备生成参数
                past_key_values = gen_context['past_key_values']
                kv_lens = gen_context['kv_lens']
                ropes = gen_context['ropes']
                
                # generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
                
                # 注意：generate_text 只支持 batch=1，需要逐个样本生成
                batch_thoughts = []
                for batch_idx in range(batch_size):
                    # 为每个样本单独准备 context
                    single_kv_lens = [kv_lens[batch_idx]]
                    single_ropes = [ropes[batch_idx]]
                    
                    # 创建单个样本的 context（需要从 batch context 中提取）
                    # 注意：past_key_values 是共享的，但我们需要为每个样本单独处理
                    single_context = {
                        'past_key_values': past_key_values,  # 共享的 cache
                        'kv_lens': single_kv_lens,
                        'ropes': single_ropes
                    }
                    
                    # 使用 gen_text 方法（单样本版本）
                    # [修改] 传递 accelerator 参数，用于临时解包 FSDP 模型
                    thought_text = self.gen_text(
                        single_context,
                        max_length=max_think_tokens,
                        do_sample=True,
                        temperature=0.7,
                        accelerator=accelerator
                    )
                    batch_thoughts.append(thought_text)
                
                thoughts = batch_thoughts
                
                # [关键步骤 3] 将思考结果回填进 Context
                # 这样接下来的图像生成就会基于这些思考
                gen_context = self.update_context_text_batch(batch_thoughts, gen_context)
                
                if was_training:
                    self.model.train()

            # 4. 生成图像 (Image Generation)
            # 此时的 gen_context 已经包含了: SystemPrompt + Image + UserPrompt + Thought
            result = self.gen_image_batch(
                image_shapes=image_shapes,
                gen_context=gen_context,
                cfg_text_precontext=cfg_text_context,
                cfg_img_precontext=cfg_img_context,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                learn=False,
                sample=None,
                grpo_config=grpo_config,
                accelerator=accelerator,
                optimizer=None,
                transformer=None,
                noise_level=noise_level,
                generators=generators,
                is_intermediate_save=is_intermediate_save,
            )
            
            # 将 thoughts 加入返回结果
            if isinstance(result, dict):
                result['thoughts'] = thoughts
            else:
                # 如果返回的不是字典，转换为字典
                result = {
                    'images': result if isinstance(result, list) else [result],
                    'thoughts': thoughts
                }
                
            return result
    
    def __call__(
        self, 
        image: Optional[Union[Image.Image, List[Image.Image]]] = None, 
        text: Optional[Union[str, List[str]]] = None, 
        **kargs
    ) -> Dict[str, Any]:
        output_dict = {'image': None, 'text': None, 'images': None}

        if image is None and text is None:
            print('Please provide at least one input: either an image or text.')
            return output_dict

        # 批量模式：如果image或text是列表
        if isinstance(image, list) or isinstance(text, list):
            if not isinstance(image, list):
                image = [image] * len(text) if isinstance(text, list) else [image]
            if not isinstance(text, list):
                text = [text] * len(image) if isinstance(image, list) else [text]
            
            if len(image) != len(text):
                raise ValueError(f"image and text lists must have the same length: {len(image)} vs {len(text)}")
            
            # 使用批量编辑方法
            result = self.batch_image_edit(
                images=image,
                texts=text,
                **kargs
            )
            # result 现在包含完整的字典（images, all_latents, all_log_probs, timesteps）
            if isinstance(result, dict):
                output_dict.update(result)
                output_dict['image'] = result.get('images', [None])[0] if result.get('images') else None
            else:
                # 兼容旧版本（如果返回的是列表）
                output_dict['images'] = result
                output_dict['image'] = result[0] if result else None
            return output_dict

        # 单样本模式：保持原有逻辑
        input_list = []
        if image is not None:
            input_list.append(image)
        if text is not None:
            input_list.append(text)

        output_list = self.interleave_inference(input_list, **kargs)
        return output_list[0]

