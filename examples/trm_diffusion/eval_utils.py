import math
import torch
import wandb
import logging
import numpy as np
from tqdm.auto import tqdm
from torchvision.utils import make_grid
from diffusers.utils import is_accelerate_version
from diffusers import DDPMPipeline
import matplotlib.pyplot as plt

import sokoban.utils as sokoban_utils
from trm_utils import get_model_output
from sokoban.evaluate_sokoban import boards_from_bit_images, boards_from_normalized_tensor, compute_sokoban_metrics

logger = logging.getLogger(__name__)


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
    prompt=None,        # sokoban
    class_labels=None   # sokoban
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
    is_concat_conditional = prompt is not None

    do_cfg = args.guidance_scale > 1.0 and (is_unified_class or is_standard_conditional or is_concat_conditional)

    if is_concat_conditional:
        metadata = [{"type": "concat_conditional"} for _ in range(bsz)]
        if class_labels is not None:
            conds = class_labels.to(device)
            if do_cfg:
                unconds = torch.full_like(conds, args.dataset.num_classes)

    elif is_unified_class or is_standard_conditional:
        conds = torch.randint(0, args.dataset.num_classes, [bsz], generator=generator, device=device)
        metadata = [{"class_label": int(c.item())} for c in conds]
        if do_cfg:
            unconds = torch.full_like(conds, args.dataset.num_classes)

    elif is_unified_sequence:
        from clevr_dataset import sample_random_scene, make_tensor_from_scene

        c_list, m_list = [], []
        for _ in range(bsz):
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

        if is_concat_conditional:
            prompt_input = torch.cat([prompt] * 2) if do_cfg else prompt
            latent_model_input = torch.cat([prompt_input, latent_model_input], dim=1)

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

        # Sokoban custom DDPMScheduler
        scheduler_output = scheduler.step(noise_pred, t, latents, generator=generator)
        latents = scheduler_output.prev_sample

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
    eval_dataloader=None,
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

    if is_sokoban:
        _evaluate_sokoban(
            unet, noise_scheduler, args, accelerator, epoch, global_step, generator, weight_dtype, eval_dataloader
        )
    else:
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


@torch.no_grad()
def _evaluate_sokoban(unet, noise_scheduler, args, accelerator, epoch, global_step, generator, weight_dtype, eval_dataloader):
    if eval_dataloader is None:
        raise AttributeError("Evaluation dataset required for sokoban validation during training")

    n_images_to_eval = getattr(args.dataset, 'n_images_to_eval', 20)
    n_images_per_cond = getattr(args.dataset, 'n_images_per_conditioning', 16)
    is_conditional = getattr(args.dataset, 'concat_conditioning', False)
    is_class_conditional = getattr(args.dataset, 'class_conditional', False)
    num_boxes = getattr(args.dataset, 'num_boxes', 4)

    all_gen_boards = []
    all_cond_boards = []
    all_target_boards = []
    log_images = []

    pbar = tqdm(total=n_images_to_eval, desc="Sokoban evaluation", disable=not accelerator.is_local_main_process)

    eval_iter = iter(eval_dataloader)
    for i in range(n_images_to_eval):
        try:
            batch = next(eval_iter)
        except StopIteration:
            break

        target_img = batch["images"][:1] if is_conditional else None
        cond_img = batch["conditions"][:1].to(unet.device, dtype=weight_dtype) if is_conditional else None
        class_label = batch["class_labels"][:1].to(unet.device, dtype=torch.long) if (is_conditional and is_class_conditional) else None

        prompt = cond_img.repeat(n_images_per_cond, 1, 1, 1) if cond_img is not None else None
        class_labels = class_label.repeat(n_images_per_cond) if class_label is not None else None

        images, _ = generate_image_batch(
            unet=unet,
            scheduler=noise_scheduler,
            vae=None,
            vae_scaling_factor=1.0,
            args=args,
            bsz=n_images_per_cond,
            generator=generator,
            device=unet.device,
            weight_dtype=weight_dtype,
            show_progress=False,
            prompt=prompt,
            class_labels=class_labels
        )

        gen_boards = boards_from_bit_images(images.cpu())
        all_gen_boards.append(gen_boards)

        if is_conditional:
            cond_board = boards_from_normalized_tensor(cond_img.cpu().float())[0]
            target_board = boards_from_normalized_tensor(target_img.cpu().float())[0]
            log_images.append((cond_board, gen_boards[0]))

            all_cond_boards.extend([cond_board] * n_images_per_cond)
            all_target_boards.extend([target_board] * n_images_per_cond)
        else:
            log_images.append((None, gen_boards[0]))

        pbar.update(1)

    pbar.close()

    if not all_gen_boards:
        return

    all_gen_boards = np.concatenate(all_gen_boards, axis=0)
    all_cond_boards = np.array(all_cond_boards) if all_cond_boards else None
    all_target_boards = np.array(all_target_boards) if all_target_boards else None

    metrics = compute_sokoban_metrics(
        all_gen_boards,
        conditioning_boards=all_cond_boards,
        target_boards=all_target_boards,
        n_images_per_conditioning=n_images_per_cond,
        num_boxes=num_boxes,
    )

    log_dict = {**metrics, "epoch": epoch}
    accelerator.log(log_dict, step=global_step)
    logger.info(f"Sokoban metrics at step {global_step}: {metrics}")

    if args.logger == "wandb" and log_images:
        wandb_images = []
        for i, (cond_board, gen_board) in enumerate(log_images):
            fig, axs = plt.subplots(1, 2 if cond_board is not None else 1, figsize=(8, 4))
            if cond_board is not None:
                axs[0].imshow(sokoban_utils.render(cond_board).astype(np.uint8))
                axs[0].set_title("Input")
                axs[0].axis("off")
                axs[1].imshow(sokoban_utils.render(gen_board).astype(np.uint8))
                axs[1].set_title("Generated")
                axs[1].axis("off")
            else:
                if not isinstance(axs, np.ndarray):
                    axs = [axs]
                axs[0].imshow(sokoban_utils.render(gen_board).astype(np.uint8))
                axs[0].set_title("Generated")
                axs[0].axis("off")
            plt.tight_layout()
            wandb_images.append(wandb.Image(fig))
            plt.close(fig)

        accelerator.get_tracker("wandb").log(
            {"sokoban/visual_samples": wandb_images}, step=global_step
        )
