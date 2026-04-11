import math
import torch
import wandb
import numpy as np
import logging
from tqdm.auto import tqdm
from torchvision.utils import make_grid
from diffusers.utils import is_accelerate_version
from diffusers import DDPMPipeline
from trm_utils import get_model_output
from sokoban.sokoban_utils import SokobanSampler


def generate_image_batch(
    unet,
    scheduler,
    vae,
    vae_scaling_factor,
    args,
    bsz,
    generator,
    device,
    weight_dtype=torch.float32,
    show_progress=True,
    single_scene=True,
    early_stopping_threshold=None,
    cond_images=None,
    class_labels=None
):
    """The unified core engine for generating a batch of images from latents."""
    sample_size = args.dataset.resolution if vae is None else args.dataset.resolution // 8

    # 1. Base Noise
    latents = torch.randn(
        (bsz, args.dataset.input_channels, sample_size, sample_size),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )

    # 2. Build Conditions
    conds, masks, unconds = None, None, None
    metadata = []

    # Safely check condition_mode (handling both old direct models and new wrapped models)
    model_config = getattr(args, "model", {})

    # If it's a Ratatouille model, the conditions are consumed by the thinker_model
    if hasattr(model_config, "thinker_model"):
        condition_target_config = model_config.thinker_model
    else:
        # If using our old wrappers, we inspect the inner core_model
        condition_target_config = getattr(model_config, "core_model", model_config)

    cond_mode = getattr(condition_target_config, "condition_mode", None)

    is_unified_class = cond_mode in ["class", "class_adaln"]
    is_unified_sequence = cond_mode == "sequence"
    is_unified_spatial = cond_mode == "spatial_concat"

    target_str = str(getattr(condition_target_config, "_target_", ""))
    is_standard_conditional = ("UNet2DModel" in target_str or "UNet2DConditionModel" in target_str) and getattr(
        args.dataset, "num_classes", None
    )

    do_cfg = args.guidance_scale > 1.0 and (is_unified_class or is_standard_conditional)

    if class_labels is not None:
        conds = class_labels
        metadata = [{"class_label": int(c.item())} for c in conds]
        if do_cfg and hasattr(args.dataset, "num_classes"):
            unconds = torch.full_like(conds, args.dataset.num_classes)

    elif is_unified_class or is_standard_conditional:
        conds = torch.randint(0, args.dataset.num_classes, [bsz], generator=generator, device=device)
        metadata = [{"class_label": int(c.item())} for c in conds]
        if do_cfg:
            unconds = torch.full_like(conds, args.dataset.num_classes)

    elif is_unified_sequence:
        from clevr_dataset import sample_random_scene, make_tensor_from_scene

        c_list, m_list = [], []
        scene = sample_random_scene(num_objects=None, mode=args.dataset.dataset_mode)
        for _ in range(bsz):
            if not single_scene:
                scene = sample_random_scene(num_objects=None, mode=args.dataset.dataset_mode)
            c, m = make_tensor_from_scene(scene)
            c_list.append(c)
            m_list.append(m)
            metadata.append(scene)
        conds = torch.cat(c_list, dim=0).to(device)
        masks = torch.cat(m_list, dim=0).to(device)

    elif is_unified_spatial:
        from clevr_dataset import sample_random_scene, make_mask_from_scene

        mask_size = sample_size  # latent resolution (already computed above)
        scene_ref = None
        mask_list = []
        for _ in range(bsz):
            if single_scene:
                if scene_ref is None:
                    scene_ref = sample_random_scene(num_objects=None, mode="relative")
                scene = scene_ref
            else:
                scene = sample_random_scene(num_objects=None, mode="relative")
            mask_list.append(make_mask_from_scene(scene, mask_size))
            metadata.append(scene)
        conds = torch.stack(mask_list).to(device)  # (B, MASK_CHANNELS, H, W)

    else:
        # Unconditional fallback
        metadata = [{"class_label": "unconditional"} for _ in range(bsz)]
        do_cfg = False

    # 3. The Denoising Loop
    do_early_stop = (
        early_stopping_threshold is not None
        and early_stopping_threshold > 0.0
        and hasattr(unet, "reasoning_step")
    )

    for t in tqdm(scheduler.timesteps, desc="Sampling", disable=not show_progress):
        if cond_images is not None:
            current_latents = torch.cat([latents, cond_images], dim=1) # latents: [B, C_noise, H, W], cond_images: [B, C_cond, H, W]
        else:
            current_latents = latents

        latent_model_input = torch.cat([current_latents] * 2) if do_cfg else current_latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        latent_model_input_cast = latent_model_input.to(weight_dtype)

        class_input = torch.cat([conds, unconds]) if do_cfg and conds is not None else conds
        mask_input = torch.cat([masks, masks]) if (do_cfg and masks is not None) else masks

        cfg_applied = False
        with torch.no_grad():
            if do_early_stop:
                # Per-sample early stopping over n_sup iterations.
                # CFG is applied inside the loop so convergence is measured on
                # the post-guidance prediction.
                t_idx = t.long().cpu().item()
                alpha_bar = scheduler.alphas_cumprod[t_idx].float().to(device)
                scale = torch.sqrt((1.0 - alpha_bar) / alpha_bar.clamp(min=1e-8))

                y, z = unet.get_initial_states(latent_model_input_cast.shape[0])
                final_output = None
                prev_output  = None
                converged    = torch.zeros(bsz, dtype=torch.bool, device=device)

                for _ in range(unet.n_sup):
                    raw, y, z = unet.reasoning_step(
                        latent_model_input_cast, y, z, t, class_input, mask_input
                    )
                    raw = raw.float()
                    if do_cfg:
                        cond_p, uncond_p = raw.chunk(2)
                        merged = uncond_p + args.guidance_scale * (cond_p - uncond_p)
                    else:
                        merged = raw

                    not_done = ~converged
                    if final_output is None:
                        final_output = merged.clone()
                    else:
                        final_output = torch.where(
                            not_done.view(bsz, 1, 1, 1).expand_as(merged),
                            merged, final_output,
                        )

                    if prev_output is not None:
                        diff = (merged - prev_output).view(bsz, -1).norm(dim=-1)
                        converged |= (scale * diff < early_stopping_threshold) & not_done
                        if converged.all():
                            break

                    prev_output = merged.detach()

                noise_pred = final_output
                cfg_applied = True  # already merged above

            elif hasattr(unet, "reasoning_step"):
                noise_pred = unet(
                    latent_model_input_cast,
                    t,
                    class_labels=class_input,
                    encoder_hidden_states=class_input,
                    attention_mask=mask_input,
                ).sample

            elif args.use_small_loop:
                from trm_utils import deep_recursion

                y = unet.y_init.expand(latent_model_input.shape[0], -1, -1, -1).to(device)
                z = unet.z_init.expand(latent_model_input.shape[0], -1, -1, -1).to(device)

                for _ in range(args.N_supervision):
                    noise_pred, y, z = deep_recursion(
                        unet, latent_model_input_cast, y, z, t, class_input, mask_input, args.n, args.T
                    )
            else:
                noise_pred = get_model_output(unet, latent_model_input_cast, t, class_input, mask_input)

        noise_pred = noise_pred.to(torch.float32)
        if do_cfg and not cfg_applied:
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # 4. VAE Decoding & Clamping
    if vae is not None:
        latents = latents / vae_scaling_factor
        vae_bsz = getattr(args, "vae_batch_size", bsz)
        decoded_images = []

        # Spoon-feed the latents to the VAE in chunks
        for i in range(0, latents.shape[0], vae_bsz):
            latent_chunk = latents[i : i + vae_bsz].to(vae.dtype)
            decoded_chunk = vae.decode(latent_chunk).sample
            decoded_images.append(decoded_chunk)

        images = torch.cat(decoded_images, dim=0)
    else:
        images = latents

    return (images / 2 + 0.5).clamp(0, 1), metadata


@torch.no_grad()
def evaluate_and_save(
    model,
    ema_model,
    noise_scheduler,
    args,
    accelerator,
    epoch,
    global_step,
    vae=None,
    vae_scaling_factor=1.0,
    weight_dtype=torch.float32,
):
    """Wrapper that calls the generator and pushes the outputs to W&B/Tensorboard."""
    unet = accelerator.unwrap_model(model)
    if args.use_ema:
        ema_target = unet.core_model if hasattr(unet, "get_trainable_modules") else unet
        ema_model.store(ema_target.parameters())
        ema_model.copy_to(ema_target.parameters())

    unet.eval()
    noise_scheduler.set_timesteps(args.ddpm_num_inference_steps)
    generator = torch.Generator(device=unet.device).manual_seed(0)

    is_sokoban = getattr(args.dataset, 'dataset_type', None) == 'sokoban'
    sokoban_sampler = None
    if is_sokoban:
        sokoban_sampler = SokobanSampler(args)

    current_bsz = args.eval_batch_size
    cond_images_prompt, class_labels = None, None
    if sokoban_sampler:
        cond_images_prompt, class_labels, current_bsz = sokoban_sampler.prepare_for_conditioning(
            accelerator, weight_dtype, current_bsz
        )

    # --- Use the Shared Engine ---
    images, _ = generate_image_batch(
        unet=unet,
        scheduler=noise_scheduler,
        vae=vae,
        vae_scaling_factor=vae_scaling_factor,
        args=args,
        bsz=current_bsz,
        generator=generator,
        device=unet.device,
        weight_dtype=weight_dtype,
        show_progress=accelerator.is_local_main_process,
        cond_images=cond_images_prompt,
        class_labels=class_labels
    )

    images = images.cpu().float()

    if sokoban_sampler:
        # Formatuje do (H, W, C) i przetwarza ewaluację
        images_np = images.permute(0, 2, 3, 1).numpy()
        images_np = (images_np * 255).round().astype(np.uint8)
        sokoban_sampler.register_generated(images_np)

        eval_logger = logging.getLogger(__name__)
        sokoban_sampler.sampling_evaluation(
            logger=eval_logger,
            accelerator=accelerator,
            global_step=global_step,
            log_with=args.logger
        )

        images = sokoban_sampler.render_boards()

    # Log Images
    images_processed = (images * 255).round().numpy().astype("uint8")
    if args.logger == "tensorboard":
        tracker = (
            accelerator.get_tracker("tensorboard", unwrap=True)
            if is_accelerate_version(">=", "0.17.0.dev0")
            else accelerator.get_tracker("tensorboard")
        )
        tracker.add_images("test_samples", images_processed, epoch)
    elif args.logger == "wandb":
        n_cols = math.ceil(math.sqrt(current_bsz) * 1.5)
        image_grid = make_grid(images, nrow=n_cols, padding=2, normalize=(is_sokoban))  # no normalization for sokoban, render already does that
        accelerator.get_tracker("wandb").log(
            {"test_samples": wandb.Image(image_grid), "epoch": epoch}, step=global_step
        )

    # Save Pipeline
    if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
        pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)
        pipeline.save_pretrained(args.output_dir)

    if args.use_ema:
        ema_model.restore(ema_target.parameters())
