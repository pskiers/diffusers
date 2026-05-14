import sys
from xml.parsers.expat import model
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
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
from safetensors.torch import load_file
from tqdm.auto import tqdm

# Local abstracted modules
from data_factory import get_dataloaders
from model_utils import extract_into_tensor, load_with_backward_compatibility
from trm_utils import get_model_output, deep_recursion
from eval_utils import evaluate_and_save
from data_utils import SafeIterator

# Will error if the minimal version of diffusers is not installed.
check_min_version("0.34.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def get_n_sup_phase(step, phases, n_sup_phases, default):
    if not len(phases):
        return default
    for phase, n_sup in zip(phases, n_sup_phases):
        if step < phase:
            return n_sup


def compute_loss(model_output, noise, clean_images, timesteps, noise_scheduler, args):
    if args.prediction_type == "epsilon":
        return F.mse_loss(model_output.float(), noise.float())
    elif args.prediction_type == "sample":
        alpha_t = extract_into_tensor(noise_scheduler.alphas_cumprod, timesteps, clean_images.shape)
        snr_weights = alpha_t / (1 - alpha_t)
        loss = snr_weights * F.mse_loss(model_output.float(), clean_images.float(), reduction="none")
        return loss.mean()
    raise ValueError(f"Unsupported prediction type: {args.prediction_type}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(args: DictConfig):
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
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

    import debugpy
    #debugpy.listen(("0.0.0.0", 5679))
    logger.info("⏳ Oczekiwanie na podłączenie debuggera z VS Code (port 5679)...")

    #debugpy.wait_for_client()
    logger.info("✅ Debugger podłączony, ruszamy dalej!")

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
    if args.dataset.vae_name is not None:
        vae = AutoencoderKL.from_pretrained(args.dataset.vae_name, cache_dir=args.cache_dir).to(
            accelerator.device, dtype=torch.float32
        )
        vae.requires_grad_(False)
        vae.eval()
        vae_scaling_factor = vae.config.scaling_factor

    train_dl, eval_dl = get_dataloaders(args)

    model = instantiate(args.model, _convert_="all")
    if hasattr(model, "core_model"):
        model.core_model = torch.compile(model.core_model, mode="reduce-overhead")

    # Enable xformers
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            if version.parse(xformers.__version__) == version.parse("0.0.16"):
                logger.warning("xFormers 0.0.16 cannot be used for training in some GPUs. Please update to >=0.0.17.")
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if hasattr(model, "get_trainable_modules"):
        total_params = sum(p.numel() for m in model.get_trainable_modules().values() for p in m.parameters())
        trainable_params = sum(
            p.numel() for m in model.get_trainable_modules().values() for p in m.parameters() if p.requires_grad
        )
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of parameters: {total_params}")
    logger.info(f"Number of trainable parameters: {trainable_params}")

    # ---------------------------------------------------------
    # 3. EMA & Custom Checkpoint Hooks
    # ---------------------------------------------------------
    ema_model = None

    # Safely target the core model if it's wrapped
    unet_for_ema = model.core_model if hasattr(model, "core_model") else model
    model_cls_for_ema = type(unet_for_ema)

    if args.use_ema:
        ema_model = EMAModel(
            unet_for_ema.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=model_cls_for_ema,
            model_config=unet_for_ema.config,
        )

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

                # ROUTE A: New Strategy
                if hasattr(model, "get_trainable_modules"):
                    model.save(output_dir)
                    while len(weights) > 0:
                        weights.pop()  # Prevent duplicate saving
                # ROUTE B: Legacy Modules
                else:
                    for i, m in enumerate(models):
                        m_to_save = m.core_model if hasattr(m, "core_model") else m
                        m_to_save.save_pretrained(os.path.join(output_dir, "unet"))
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                ema_dir = os.path.join(input_dir, "unet_ema")
                sf_path = os.path.join(ema_dir, "diffusion_pytorch_model.safetensors")
                bin_path = os.path.join(ema_dir, "diffusion_pytorch_model.bin")

                if os.path.exists(sf_path):
                    ema_state_dict = load_file(sf_path)
                elif os.path.exists(bin_path):
                    ema_state_dict = torch.load(bin_path, map_location="cpu")
                else:
                    raise FileNotFoundError(f"Could not find EMA weights in {ema_dir}")

                if hasattr(model, "get_trainable_modules"):
                    m_to_load = model.core_model.module if hasattr(model.core_model, "module") else model.core_model
                else:
                    m = models[0]
                    m_to_load = m.core_model if hasattr(m, "core_model") else m

                load_with_backward_compatibility(m_to_load, ema_state_dict)
                new_ema = EMAModel(m_to_load.parameters(), model_cls=type(m_to_load), model_config=m_to_load.config)
                ema_model.load_state_dict(new_ema.state_dict())
                ema_model.to(accelerator.device)
                del new_ema

            if hasattr(model, "get_trainable_modules"):
                model.load(input_dir)
                while len(models) > 0:
                    models.pop()
            else:
                for _ in range(len(models)):
                    m = models.pop()
                    m_to_load = m.core_model if hasattr(m, "core_model") else m
                    unet_dir = os.path.join(input_dir, "unet")
                    sf_path = os.path.join(unet_dir, "diffusion_pytorch_model.safetensors")
                    bin_path = os.path.join(unet_dir, "diffusion_pytorch_model.bin")

                    if os.path.exists(sf_path):
                        state_dict = load_file(sf_path)
                    elif os.path.exists(bin_path):
                        state_dict = torch.load(bin_path, map_location="cpu")
                    else:
                        raise FileNotFoundError(f"Could not find model weights in {unet_dir}")

                    load_with_backward_compatibility(m_to_load, state_dict)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Note: The manual anchor generation block that was here has been completely deleted!

    # ---------------------------------------------------------
    # 4. Optimizers, Schedulers, and Trackers
    # ---------------------------------------------------------
    if hasattr(model, "get_trainable_modules"):
        params = []
        for m in model.get_trainable_modules().values():
            params.extend(m.parameters())
    else:
        params = model.parameters()

    optimizer = instantiate(args.optimizer, params=params)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.ddpm_num_steps,
        beta_schedule=args.ddpm_beta_schedule,
        prediction_type=args.prediction_type,
    )
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(accelerator.device)

    # mult = getattr(model, "n_sup", 1) if not hasattr(args.model, "n_sup") else getattr(args.model, "n_sup", 1)

    total_optimization_steps = (len(train_dl) // args.gradient_accumulation_steps) * args.num_epochs

    lr_scheduler = get_scheduler(
        args.lr_scheduler.name,
        optimizer=optimizer,
        # num_warmup_steps=args.lr_scheduler.warmup_steps * args.gradient_accumulation_steps * mult,
        # num_training_steps=len(train_dl) * args.num_epochs * mult,
        num_warmup_steps=args.lr_scheduler.warmup_steps,
        num_training_steps=total_optimization_steps,
    )

    if hasattr(model, "get_trainable_modules"):
        optimizer, train_dl, eval_dl, lr_scheduler = accelerator.prepare(optimizer, train_dl, eval_dl, lr_scheduler)
        prepared_modules = {name: accelerator.prepare(m) for name, m in model.get_trainable_modules().items()}
        model.update_modules(prepared_modules)
    else:
        model, optimizer, train_dl, eval_dl, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dl, eval_dl, lr_scheduler
        )
    if args.use_ema:
        ema_model.to(accelerator.device)

    if accelerator.is_main_process and args.logger == "wandb":
        tracker_config = OmegaConf.to_container(args, resolve=True)
        tracker_config["total_params"] = total_params
        tracker_config["trainable_params"] = trainable_params
        accelerator.init_trackers(
            project_name="TRM-Diffusion",
            config=tracker_config,
            init_kwargs={"wandb": {"name": args.output_dir}} if args.logger == "wandb" else {},
        )
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

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
    logger.info(
        f"  Total train batch size = {args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}"
    )
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

            batch = {k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            clean_images = batch["images"]
            cond = batch["conditions"].to(model.device) if batch["conditions"] is not None else None
            mask = batch["masks"].to(model.device) if batch["masks"] is not None else None
            class_labels = batch["class_labels"].to(model.device) if batch["class_labels"] is not None else None

            is_concat = getattr(args.dataset, 'concat_conditioning', False)

            # CFG Label Dropout
            if args.cfg_drop_rate > 0:
                if is_concat:   # sokoban: 'cond' has cond img, 'class_labels' has label
                    drop_mask = torch.rand(clean_images.shape[0], device=clean_images.device) < args.cfg_drop_rate
                    if class_labels is not None:
                        class_labels = torch.where(drop_mask, torch.tensor(args.dataset.num_classes, device=class_labels.device), class_labels)
                        class_labels = class_labels.long()
                    if cond is not None:
                        cond = torch.where(drop_mask.view(-1, 1, 1, 1), torch.zeros_like(cond), cond)
                else:   # STANDARD (CIFAR/CLEVR): 'cond' has labels
                    if cond is not None:
                        drop_mask = torch.rand(cond.shape, device=cond.device) < args.cfg_drop_rate
                        cond = torch.where(drop_mask, torch.tensor(args.dataset.num_classes, device=cond.device), cond)

            # VAE Encoding & Noise Addition
            if vae is not None:
                clean_images = clean_images.to(device=accelerator.device, dtype=vae.dtype)
                with torch.no_grad():
                    clean_images = vae.encode(clean_images).latent_dist.sample() * vae_scaling_factor

            clean_images = clean_images.to(device=accelerator.device, dtype=torch.float32)
            noise = torch.randn_like(clean_images)
            bsz = clean_images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps).to(weight_dtype)

            if is_concat and cond is not None:  # sokoban with conditioning image
                latent_model_input = torch.cat([noisy_images, cond], dim=1)
                model_cond = class_labels
            else:   # standard, cond is labels
                latent_model_input = noisy_images
                model_cond = cond

            # --- FORWARD & BACKWARD PASS ROUTING ---
            with accelerator.accumulate(model):
                base_model = accelerator.unwrap_model(model)

                # ROUTE 1: New Object-Oriented TRM Models
                if hasattr(base_model, "reasoning_step"):
                    y, z = base_model.get_initial_states(bsz)
                    y, z = y.to(model.device), z.to(model.device)

                    loss_full = None

                    (x_high, enc_hs, ts, embedded_ts,
                    class_labels, enc_mask, autocast_ctx) = base_model.encode_features(
                        latent_model_input, timesteps, model_cond, mask
                    )

                    n_sup_current = get_n_sup_phase(global_step, args.phases, args.n_sup_phases, base_model.n_sup)
                    for n_step in range(n_sup_current):
                        # with accelerator.autocast():
                        y_final_high, y, z = base_model.reasoning_core(
                            x_high, y, z, enc_hs, ts, class_labels, enc_mask, autocast_ctx
                        )
                        model_output = base_model.decode_features(
                            y_final_high, ts, class_labels, embedded_ts, autocast_ctx
                        )

                        loss = compute_loss(model_output, noise, clean_images, timesteps, noise_scheduler, args)
                        loss = loss if not args.trm_loss_nsup_decay else loss * (args.trm_loss_nsup_decay**n_step)

                        if args.grad_every_n_sup:
                            accelerator.backward(loss)
                            if accelerator.sync_gradients:
                                if hasattr(base_model, "get_trainable_modules"):
                                    params_to_clip = []
                                    for m in base_model.get_trainable_modules().values():
                                        params_to_clip.extend(m.parameters())
                                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                                else:
                                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            lr_scheduler.step()
                            optimizer.zero_grad()
                        else:
                            loss_full = loss_full + loss if loss_full is not None else loss

                    if not args.grad_every_n_sup:
                        final_loss = loss_full / n_sup_current
                        accelerator.backward(final_loss)
                        if accelerator.sync_gradients:
                            if hasattr(base_model, "get_trainable_modules"):
                                params_to_clip = []
                                for m in base_model.get_trainable_modules().values():
                                    params_to_clip.extend(m.parameters())
                                accelerator.clip_grad_norm_(params_to_clip, 1.0)
                            else:
                                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                # ROUTE 2: Old Procedural TRM Logic (Backward Compatibility)
                elif args.use_small_loop:
                    y = torch.cat([base_model.y_init for _ in range(bsz)], dim=0).to(model.device)
                    z = torch.cat([base_model.z_init for _ in range(bsz)], dim=0).to(model.device)

                    for _ in range(args.N_supervision):
                        model_output, y, z = deep_recursion(
                            base_model, latent_model_input, y, z, timesteps, model_cond, mask, args.n, args.T
                        )
                        loss = compute_loss(model_output, noise, clean_images, timesteps, noise_scheduler, args)

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                # ROUTE 3: Standard Non-Recursive Models
                else:
                    model_output = get_model_output(model, latent_model_input, timesteps, model_cond, mask)
                    loss = compute_loss(model_output, noise, clean_images, timesteps, noise_scheduler, args)

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # Sync & Checkpoint
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_target = model.core_model if hasattr(model, "get_trainable_modules") else model
                    ema_model.step(ema_target.parameters())
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
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
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
            class_labels = batch["class_labels"].to(model.device) if batch["class_labels"] is not None else None

            is_concat = getattr(args.dataset, 'concat_conditioning', False)

            if vae is not None:
                clean_images = clean_images.to(dtype=vae.dtype)
                with torch.no_grad():
                    clean_images = vae.encode(clean_images).latent_dist.sample() * vae_scaling_factor
                clean_images = clean_images.to(dtype=weight_dtype)
            clean_images = clean_images.to(device=accelerator.device, dtype=torch.float32)

            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (clean_images.shape[0],), device=clean_images.device
            ).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps).to(weight_dtype)

            if is_concat and cond is not None:
                latent_model_input = torch.cat([noisy_images, cond], dim=1)
                model_cond = class_labels
            else:
                latent_model_input = noisy_images
                model_cond = cond

            # --- VALIDATION PASS ROUTING ---
            with torch.no_grad():
                base_model = accelerator.unwrap_model(model)

                # ROUTE 1: New Object-Oriented TRM Models
                if hasattr(base_model, "reasoning_step"):
                    model_output = model(latent_model_input, timesteps, class_labels=model_cond, attention_mask=mask).sample

                # ROUTE 2: Old Procedural TRM Logic
                elif args.use_small_loop:
                    y = torch.cat([base_model.y_init for _ in range(clean_images.shape[0])], dim=0).to(model.device)
                    z = torch.cat([base_model.z_init for _ in range(clean_images.shape[0])], dim=0).to(model.device)
                    model_output, _, _ = deep_recursion(
                        base_model, latent_model_input, y, z, timesteps, model_cond, mask, args.n, args.T
                    )

                # ROUTE 3: Standard Non-Recursive Models
                else:
                    model_output = get_model_output(model, latent_model_input, timesteps, model_cond, mask)

                val_loss = compute_loss(model_output, noise, clean_images, timesteps, noise_scheduler, args)

            val_pbar.update(1)
            val_pbar.set_postfix({"val/loss": val_loss.detach().item()})
            accelerator.log({"val/loss": val_loss.detach().item()}, step=global_step)

        val_pbar.close()

        if accelerator.is_main_process and (epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1):
            evaluate_and_save(
                model,
                ema_model,
                noise_scheduler,
                args,
                accelerator,
                epoch,
                global_step,
                vae,
                vae_scaling_factor,
                weight_dtype,
            )

        if accelerator.is_main_process and (epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1):
            unet = accelerator.unwrap_model(model)
            unet_to_save = unet.core_model if hasattr(unet, "core_model") else unet

            if args.use_ema:
                ema_model.store(unet_to_save.parameters())
                ema_model.copy_to(unet_to_save.parameters())

            pipeline = diffusers.DDPMPipeline(unet=unet_to_save, scheduler=noise_scheduler)
            pipeline.save_pretrained(args.output_dir)

            if args.use_ema:
                ema_model.restore(unet_to_save.parameters())

            if args.push_to_hub:
                upload_folder(
                    repo_id=repo_id,
                    folder_path=args.output_dir,
                    commit_message=f"Epoch {epoch}",
                    ignore_patterns=["step_*", "epoch_*"],
                )

    accelerator.end_training()


if __name__ == "__main__":
    # Accelerate passes --local_rank via sys.argv. Hydra hates this.
    # This strips all dashes so Hydra only processes key=value pairs.
    sys.argv = [a for a in sys.argv if not a.startswith("--")]
    main()
