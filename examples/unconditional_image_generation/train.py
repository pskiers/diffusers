import os
import math
import shutil
import logging
from datetime import timedelta
from pathlib import Path
from packaging import version

import datasets
import torch
import torch.nn.functional as F
import accelerate
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from huggingface_hub import create_repo, upload_folder

import diffusers
from diffusers import DDPMScheduler, AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import (
    check_min_version,
    is_tensorboard_available,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm

# Local abstracted modules
from config import parse_args
from data_factory import get_dataloaders
from model_factory import build_model
from model_utils import _extract_into_tensor
from trm_utils import get_model_output, deep_recursion
from eval_utils import evaluate_and_save
from data_utils import SafeIterator

# Will error if the minimal version of diffusers is not installed.
check_min_version("0.34.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def compute_loss(model_output, noise, clean_images, timesteps, noise_scheduler, args):
    if args.prediction_type == "epsilon":
        return F.mse_loss(model_output.float(), noise.float())
    elif args.prediction_type == "sample":
        alpha_t = _extract_into_tensor(noise_scheduler.alphas_cumprod, timesteps, clean_images.shape)
        snr_weights = alpha_t / (1 - alpha_t)
        loss = snr_weights * F.mse_loss(model_output.float(), clean_images.float(), reduction="none")
        return loss.mean()
    raise ValueError(f"Unsupported prediction type: {args.prediction_type}")


def main():
    args = parse_args()

    # ---------------------------------------------------------
    # 1. Setup Accelerator & Loggers
    # ---------------------------------------------------------
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging.")
    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # ---------------------------------------------------------
    # 2. Initialize VAE, Data, and Model
    # ---------------------------------------------------------
    vae, vae_scaling_factor = None, 1.0
    if args.vae_name is not None:
        vae = AutoencoderKL.from_pretrained(args.vae_name, cache_dir=args.cache_dir).to(accelerator.device, dtype=torch.float32)
        vae.requires_grad_(False)
        vae.eval()
        vae_scaling_factor = vae.config.scaling_factor

    train_dl, eval_dl = get_dataloaders(args)
    model = build_model(args)
    model_cls = type(model)

    # Enable xformers
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            if version.parse(xformers.__version__) == version.parse("0.0.16"):
                logger.warning("xFormers 0.0.16 cannot be used for training in some GPUs. Please update to >=0.0.17.")
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of parameters: {total_params}")
    logger.info(f"Number of trainable parameters: {trainable_params}")

    # ---------------------------------------------------------
    # 3. EMA & Custom Checkpoint Hooks
    # ---------------------------------------------------------
    ema_model = None
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(), decay=args.ema_max_decay, use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma, power=args.ema_power,
            model_cls=model_cls, model_config=model.config
        )

    # `accelerate` custom saving & loading hooks
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))
                for i, m in enumerate(models):
                    m.save_pretrained(os.path.join(output_dir, "unet"))
                    weights.pop() # pop weight so it's not saved twice

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), model_cls)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                m = models.pop()
                load_model = model_cls.from_pretrained(input_dir, subfolder="unet")
                m.register_to_config(**load_model.config)
                m.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Save initial reasoning tokens for small loop
    if args.use_small_loop:
        y_path = os.path.join(args.output_dir, "y_init.pt")
        z_path = os.path.join(args.output_dir, "z_init.pt")

        # If the files exist (e.g., resuming), load them and overwrite the factory's random ones
        if os.path.exists(y_path) and os.path.exists(z_path):
            logger.info("Loading existing small loop anchor tokens (y_init.pt, z_init.pt)")
            model.y_init = torch.load(y_path, map_location="cpu")
            model.z_init = torch.load(z_path, map_location="cpu")
        # Otherwise, save the factory's freshly generated ones for future resumes
        elif accelerator.is_main_process:
            logger.info("Saving new small loop anchor tokens (y_init.pt, z_init.pt)")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.y_init, y_path)
            torch.save(model.z_init, z_path)

    # ---------------------------------------------------------
    # 4. Optimizers, Schedulers, and Trackers
    # ---------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon
    )
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule, prediction_type=args.prediction_type)

    mult = args.N_supervision if args.use_small_loop else 1
    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps * mult,
        num_training_steps=len(train_dl) * args.num_epochs * mult
    )

    model, optimizer, train_dl, eval_dl, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dl, eval_dl, lr_scheduler
    )
    if args.use_ema:
        ema_model.to(accelerator.device)

    if accelerator.is_main_process and args.logger == "wandb":
        tracker_config = vars(args)
        tracker_config["total_params"] = total_params
        tracker_config["trainable_params"] = trainable_params
        accelerator.init_trackers(
            project_name="small-llm-diffusion",
            config=tracker_config,
            init_kwargs={"wandb": {"name": args.output_dir}} if args.logger == "wandb" else {}
        )
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16": weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16": weight_dtype = torch.bfloat16

    # ---------------------------------------------------------
    # 5. Resume from Checkpoint Logic
    # ---------------------------------------------------------
    num_update_steps_per_epoch = math.ceil(len(train_dl) / args.gradient_accumulation_steps)
    global_step, first_epoch, resume_step = 0, 0, 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new run.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.num_epochs * num_update_steps_per_epoch}")

    # ---------------------------------------------------------
    # MAIN LOOP
    # ---------------------------------------------------------
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in SafeIterator(enumerate(train_dl), logger=logger):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            clean_images = batch["images"]
            cond = batch["conditions"].to(model.device) if batch["conditions"] is not None else None
            mask = batch["masks"].to(model.device) if batch["masks"] is not None else None

            # CFG Label Dropout
            if cond is not None and args.model_type == "unified_class" and args.cfg_drop_rate > 0:
                drop_mask = torch.rand(cond.shape, device=cond.device) < args.cfg_drop_rate
                cond = torch.where(drop_mask, torch.tensor(args.num_classes, device=cond.device), cond)

            # VAE Encoding & Noise Addition
            if vae is not None:
                clean_images = clean_images.to(device=accelerator.device, dtype=vae.dtype)
                with torch.no_grad():
                    clean_images = vae.encode(clean_images).latent_dist.sample() * vae_scaling_factor

            clean_images = clean_images.to(device=accelerator.device, dtype=weight_dtype)
            noise = torch.randn_like(clean_images)
            bsz = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Forward & Backward Pass
            with accelerator.accumulate(model):
                if args.use_small_loop:
                    base_model = accelerator.unwrap_model(model)
                    y = torch.cat([base_model.y_init for _ in range(bsz)], dim=0).to(model.device)
                    z = torch.cat([base_model.z_init for _ in range(bsz)], dim=0).to(model.device)

                    for _ in range(args.N_supervision):
                        model_output, y, z = deep_recursion(model, noisy_images, y, z, timesteps, cond, mask, args.n, args.T)
                        loss = compute_loss(model_output, noise, clean_images, timesteps, noise_scheduler, args)

                        accelerator.backward(loss)
                        if accelerator.sync_gradients: accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                else:
                    model_output = get_model_output(model, noisy_images, timesteps, cond, mask)
                    loss = compute_loss(model_output, noise, clean_images, timesteps, noise_scheduler, args)

                    accelerator.backward(loss)
                    if accelerator.sync_gradients: accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # Sync & Checkpoint
            if accelerator.sync_gradients:
                if args.use_ema: ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    # Clean up old checkpoints
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]
                            logger.info(f"{len(checkpoints)} checkpoints exist, removing {len(removing_checkpoints)}.")
                            for r_chk in removing_checkpoints:
                                shutil.rmtree(os.path.join(args.output_dir, r_chk))

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            logs = {"train/loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.use_ema: logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        progress_bar.close()
        accelerator.wait_for_everyone()

        # ---------------------------------------------------------
        # VALIDATION & GENERATION
        # ---------------------------------------------------------
        model.eval()
        val_pbar = tqdm(total=len(eval_dl), disable=not accelerator.is_local_main_process)
        val_pbar.set_description(f"Eval epoch {epoch}")

        for step, batch in SafeIterator(enumerate(eval_dl), logger=logger):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    val_pbar.update(1)
                continue

            clean_images = batch["images"].to(device=accelerator.device, dtype=weight_dtype)
            cond = batch["conditions"].to(model.device) if batch["conditions"] is not None else None
            mask = batch["masks"].to(model.device) if batch["masks"] is not None else None

            if vae is not None:
                clean_images = clean_images.to(dtype=vae.dtype)
                with torch.no_grad():
                    clean_images = vae.encode(clean_images).latent_dist.sample() * vae_scaling_factor
                clean_images = clean_images.to(dtype=weight_dtype)

            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (clean_images.shape[0],), device=clean_images.device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with torch.no_grad():
                if args.use_small_loop:
                    base_model = accelerator.unwrap_model(model)
                    y = torch.cat([base_model.y_init for _ in range(clean_images.shape[0])], dim=0).to(model.device)
                    z = torch.cat([base_model.z_init for _ in range(clean_images.shape[0])], dim=0).to(model.device)
                    model_output, _, _ = deep_recursion(model, noisy_images, y, z, timesteps, cond, mask, args.n, args.T)
                else:
                    model_output = get_model_output(model, noisy_images, timesteps, cond, mask)

                val_loss = compute_loss(model_output, noise, clean_images, timesteps, noise_scheduler, args)

            val_pbar.update(1)
            val_pbar.set_postfix({"val/loss": val_loss.detach().item()})
            accelerator.log({"val/loss": val_loss.detach().item()}, step=global_step)

        val_pbar.close()

        if accelerator.is_main_process and (epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1):
            evaluate_and_save(model, ema_model, noise_scheduler, args, accelerator, epoch, global_step, vae, vae_scaling_factor, weight_dtype)

        if accelerator.is_main_process and (epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1):
            # Save standard diffusers pipeline to output_dir
            unet = accelerator.unwrap_model(model)
            if args.use_ema:
                ema_model.store(unet.parameters())
                ema_model.copy_to(unet.parameters())

            pipeline = diffusers.DDPMPipeline(unet=unet, scheduler=noise_scheduler)
            pipeline.save_pretrained(args.output_dir)

            if args.use_ema:
                ema_model.restore(unet.parameters())

            if args.push_to_hub:
                upload_folder(
                    repo_id=repo_id, folder_path=args.output_dir, commit_message=f"Epoch {epoch}", ignore_patterns=["step_*", "epoch_*"]
                )

    accelerator.end_training()

if __name__ == "__main__":
    main()
