import math
import torch
import wandb
from tqdm.auto import tqdm
from torchvision.utils import make_grid
from diffusers.utils import is_accelerate_version
from diffusers import DDPMPipeline
from trm_utils import get_model_output, deep_recursion


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
    unet = accelerator.unwrap_model(model)
    if args.use_ema:
        ema_model.store(unet.parameters())
        ema_model.copy_to(unet.parameters())

    unet.eval()
    device = unet.device
    generator = torch.Generator(device=device).manual_seed(0)

    bsz = args.eval_batch_size
    sample_size = args.dataset.resolution if vae is None else args.dataset.resolution // 8

    # 1. Setup Base Latents
    latents = torch.randn(
        (bsz, args.dataset.input_channels, sample_size, sample_size),
        generator=generator,
        device=device,
        dtype=weight_dtype,
    )

    # 2. Route Conditions (Unconditional vs Class vs Sequence)
    conds, masks, unconds = None, None, None
    is_unified_class = getattr(args.model, "condition_mode", None) == "class"
    is_unified_sequence = getattr(args.model, "condition_mode", None) == "sequence"
    is_standard_conditional = "UNet2DModel" in args.model._target_ and args.dataset.num_classes

    do_cfg = args.guidance_scale > 1.0 and is_unified_class

    if is_unified_class or is_standard_conditional:
        conds = torch.randint(0, args.dataset.num_classes, [bsz], generator=generator, device=device)
        if do_cfg:
            unconds = torch.full_like(conds, args.dataset.num_classes)

    elif is_unified_sequence:
        from clevr_dataset import sample_random_scene, make_tensor_from_scene

        c_list, m_list = [], []
        for _ in range(bsz):
            scene = sample_random_scene(num_objects=4, mode=args.dataset.dataset_mode)
            c, m = make_tensor_from_scene(scene)
            c_list.append(c)
            m_list.append(m)
        conds = torch.cat(c_list, dim=0).to(device)
        masks = torch.cat(m_list, dim=0).to(device)

    noise_scheduler.set_timesteps(args.ddpm_num_inference_steps)

    # 3. Custom Sampling Loop
    for t in tqdm(noise_scheduler.timesteps, desc="Sampling", disable=not accelerator.is_local_main_process):
        # Handle double-batching for Classifier-Free Guidance
        if do_cfg:
            latent_input = torch.cat([latents] * 2)
            class_input = torch.cat([conds, unconds])
            mask_input = None
        else:
            latent_input = latents
            class_input = conds
            mask_input = masks

        # Route standard vs small loop
        if args.use_small_loop:
            y = torch.cat([unet.y_init for _ in range(latent_input.shape[0])], dim=0).to(device)
            z = torch.cat([unet.z_init for _ in range(latent_input.shape[0])], dim=0).to(device)
            noise_pred, _, _ = deep_recursion(unet, latent_input, y, z, t, class_input, mask_input, args.n, args.T)
        else:
            noise_pred = get_model_output(unet, latent_input, t, class_input, mask_input)

        if do_cfg:
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    # 4. Decode via VAE if necessary
    if vae is not None:
        latents = latents / vae_scaling_factor
        images = vae.decode(latents.to(torch.float32)).sample
    else:
        images = latents

    images = (images / 2 + 0.5).clamp(0, 1).cpu().float()

    # 5. Log Images
    images_processed = (images * 255).round().numpy().astype("uint8")
    if args.logger == "tensorboard":
        tracker = (
            accelerator.get_tracker("tensorboard", unwrap=True)
            if is_accelerate_version(">=", "0.17.0.dev0")
            else accelerator.get_tracker("tensorboard")
        )
        tracker.add_images("test_samples", images_processed, epoch)
    elif args.logger == "wandb":
        n_cols = math.ceil(math.sqrt(bsz) * 1.5)
        image_grid = make_grid(images, nrow=n_cols, padding=2, normalize=True)
        accelerator.get_tracker("wandb").log(
            {"test_samples": wandb.Image(image_grid), "epoch": epoch}, step=global_step
        )

    # 6. Save Pipeline
    if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
        pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)
        pipeline.save_pretrained(args.output_dir)

    if args.use_ema:
        ema_model.restore(unet.parameters())
