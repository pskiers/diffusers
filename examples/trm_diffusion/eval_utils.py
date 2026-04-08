import math
import torch
import wandb
from tqdm.auto import tqdm
from torchvision.utils import make_grid
from diffusers.utils import is_accelerate_version
from diffusers import DDPMPipeline
from trm_utils import get_model_output


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

    target_str = str(getattr(condition_target_config, "_target_", ""))
    is_standard_conditional = ("UNet2DModel" in target_str or "UNet2DConditionModel" in target_str) and getattr(
        args.dataset, "num_classes", None
    )

    do_cfg = args.guidance_scale > 1.0 and (is_unified_class or is_standard_conditional)

    if is_unified_class or is_standard_conditional:
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
    else:
        # Unconditional fallback
        metadata = [{"class_label": "unconditional"} for _ in range(bsz)]

    # 3. The Denoising Loop
    for t in tqdm(scheduler.timesteps, desc="Sampling", disable=not show_progress):
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        latent_model_input_cast = latent_model_input.to(weight_dtype)

        class_input = torch.cat([conds, unconds]) if do_cfg else conds
        mask_input = torch.cat([masks, masks]) if (do_cfg and masks is not None) else masks

        with torch.no_grad():
            if hasattr(unet, "reasoning_step"):
                noise_pred = unet(
                    latent_model_input_cast,
                    t,
                    class_labels=class_input,
                    encoder_hidden_states=class_input,
                    attention_mask=mask_input,
                ).sample

            elif args.use_small_loop:
                # 2. Old procedural logic (Maintains 100% backward compatibility)
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
        if do_cfg:
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

    # --- Use the Shared Engine ---
    images, _ = generate_image_batch(
        unet=unet,
        scheduler=noise_scheduler,
        vae=vae,
        vae_scaling_factor=vae_scaling_factor,
        args=args,
        bsz=args.eval_batch_size,
        generator=generator,
        device=unet.device,
        weight_dtype=weight_dtype,
        show_progress=accelerator.is_local_main_process,
    )

    images = images.cpu().float()

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
        n_cols = math.ceil(math.sqrt(args.eval_batch_size) * 1.5)
        image_grid = make_grid(images, nrow=n_cols, padding=2, normalize=True)
        accelerator.get_tracker("wandb").log(
            {"test_samples": wandb.Image(image_grid), "epoch": epoch}, step=global_step
        )

    # Save Pipeline
    if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
        pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)
        pipeline.save_pretrained(args.output_dir)

    if args.use_ema:
        ema_model.restore(ema_target.parameters())
