import sys
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import os
import math
import logging
from datetime import timedelta
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from tqdm.auto import tqdm
import numpy as np
import wandb
from torchvision.utils import make_grid

from data_factory import get_dataloaders
from data_utils import SafeIterator
from trm_wrappers import OriginalTRMRatatouilleV0Tok, OriginalTRMRatatouilleV1, OriginalTRMRatatouilleV2
from sokoban.sokoban_utils import SokobanSampler
from sokoban.sokoban_utils import SokobanEvaluator


check_min_version("0.34.0.dev0")
logger = get_logger(__name__, log_level="INFO")

FLOOR_ID = 2


def set_requires_grad(module: torch.nn.Module, requires_grad: bool):
    for param in module.parameters():
        param.requires_grad = requires_grad


def bits_to_tokens(bits_tensor: torch.Tensor, clip_sample_range: float = 1.0) -> torch.Tensor:
    """Translates diffusion vectors [-1, 1] to flat tokens [0-7] for TRM. Assumes bits_tensor shape is (B, num_bits, H, W).."""
    bits_01 = ((bits_tensor / clip_sample_range) + 1.0) / 2.0
    bits_01 = bits_01.round().long()
    B, num_bits, H, W = bits_01.shape
    tokens = torch.zeros((B, H, W), device=bits_tensor.device, dtype=torch.long)
    for i in range(num_bits):
        tokens += bits_01[:, i, :, :] * (2 ** i)
    return tokens.view(B, -1)


def bits_to_boards(tensor, clip_range: float = 1.0) -> np.ndarray:
        images_01 = (tensor / clip_range / 2 + 0.5).clamp(0, 1)
        bits = (images_01 > 0.5).float()
        bits = bits.permute(0, 2, 3, 1).to(torch.uint8)
        powers = 2 ** torch.arange(bits.shape[-1], device=bits.device)
        return torch.sum(bits * powers, dim=-1).cpu().numpy()

@torch.no_grad()
def _run_sokoban_evaluation(
    model, noise_scheduler, args, accelerator, epoch, global_step,
    weight_dtype, is_v0tok,
):
    base_model = accelerator.unwrap_model(model)
    base_model.eval()

    evaluator = SokobanEvaluator(args.dataset.num_boxes)

    clip_range = getattr(args, "clip_sample_range", 1.0)
    num_samples = getattr(args, "num_samples", 64)
    resolution = args.dataset.resolution
    n_bits = args.dataset.input_channels
    device = accelerator.device

    noise_scheduler.set_timesteps(args.ddpm_num_inference_steps)
    generator = torch.Generator(device=device).manual_seed(0)

    if is_v0tok:
        seq_len = resolution * resolution
        blank_tokens = torch.full((num_samples, seq_len), FLOOR_ID, device=device, dtype=torch.long)
    else:
        blank_condition = torch.zeros(
            (num_samples, n_bits, resolution, resolution), device=device, dtype=weight_dtype,
        )

    latents = torch.randn(
        (num_samples, n_bits, resolution, resolution),
        generator=generator, device=device, dtype=torch.float32,
    )

    for t in tqdm(noise_scheduler.timesteps, desc="Sampling", disable=not accelerator.is_local_main_process):
        latent_input = noise_scheduler.scale_model_input(latents.to(weight_dtype), t)

        if is_v0tok:
            noise_pred, _ = base_model(latent_input, t, blank_tokens)
        else:
            noise_pred, _ = base_model(latent_input, t, blank_condition)

        noise_pred = noise_pred.to(torch.float32)
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    gen_boards = bits_to_boards(latents, clip_range=clip_range)

    sokoban_metrics = evaluator.generate_metrics(
        generated_boards=gen_boards,
        conditioning_boards=None,
        target_boards=None,
        k_values=None,
        n_images_per_conditioning=1,
    )

    logger.info(f"Epoch {epoch} sokoban metrics: {sokoban_metrics}")
    accelerator.log(sokoban_metrics, step=global_step)

    if args.logger == "wandb":
        sampler = SokobanSampler(args)
        sampler.all_gen_boards_list = [gen_boards]
        rendered = sampler.render_boards()
        n_cols = math.ceil(math.sqrt(num_samples) * 1.5)
        grid = make_grid(rendered, nrow=n_cols, padding=2)
        accelerator.get_tracker("wandb").log(
            {"sokoban_samples": wandb.Image(grid), "epoch": epoch}, step=global_step,
        )


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args: DictConfig):
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

    # Optimization tricks
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    train_dl, eval_dl = get_dataloaders(args)
    model = instantiate(args.model, _convert_="all")

    # EXPERIMENT TYPE
    is_v0tok = isinstance(model, OriginalTRMRatatouilleV0Tok) and not isinstance(model, OriginalTRMRatatouilleV1)
    is_v1 = isinstance(model, OriginalTRMRatatouilleV1) and not isinstance(model, OriginalTRMRatatouilleV2)
    is_v2 = isinstance(model, OriginalTRMRatatouilleV2)

    # V0TOK: Loading and freezing Thinker
    if is_v0tok and getattr(args, "pretrained_thinker_path", None):
        logger.info(f"Loading pretrained Thinker from: {args.pretrained_thinker_path}")
        state_dict = torch.load(args.pretrained_thinker_path, map_location="cpu")
        clean_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
        model.thinker.load_state_dict(clean_state_dict, strict=True)
        set_requires_grad(model.thinker, False)

    # 3. Optimizers and Schedulers
    total_train_steps = len(train_dl) * args.num_epochs
    warmup_steps = args.lr_scheduler.warmup_steps * args.gradient_accumulation_steps

    # Thinker optimizer: LLM-style (used in V1/V2)
    def _make_thinker_optimizer(params):
        t_cfg = args.thinker_optimizer
        return torch.optim.AdamW(
            params,
            lr=t_cfg.lr,
            betas=tuple(t_cfg.betas),
            weight_decay=t_cfg.weight_decay,
        )

    if is_v2:
        opt_thinker = _make_thinker_optimizer(model.get_thinker_params())
        opt_all = instantiate(args.optimizer, params=model.parameters())

        sched_thinker = get_scheduler(args.lr_scheduler.name, optimizer=opt_thinker, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps)
        sched_all = get_scheduler(args.lr_scheduler.name, optimizer=opt_all, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps)

        model, opt_thinker, opt_all, train_dl, eval_dl, sched_thinker, sched_all = accelerator.prepare(
            model, opt_thinker, opt_all, train_dl, eval_dl, sched_thinker, sched_all
        )
    elif is_v1:
        # V1: separate optimizers for thinker (LLM-style) and painter (diffusion-style)
        opt_thinker = _make_thinker_optimizer(model.get_thinker_params())
        opt_painter = instantiate(args.optimizer, params=model.get_painter_params())

        sched_thinker = get_scheduler(args.lr_scheduler.name, optimizer=opt_thinker, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps)
        sched_painter = get_scheduler(args.lr_scheduler.name, optimizer=opt_painter, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps)

        model, opt_thinker, opt_painter, train_dl, eval_dl, sched_thinker, sched_painter = accelerator.prepare(
            model, opt_thinker, opt_painter, train_dl, eval_dl, sched_thinker, sched_painter
        )
    else:
        # V0Tok: only painter is trained
        params = model.get_painter_params()
        optimizer = instantiate(args.optimizer, params=params)
        lr_scheduler = get_scheduler(args.lr_scheduler.name, optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps)

        model, optimizer, train_dl, eval_dl, lr_scheduler = accelerator.prepare(model, optimizer, train_dl, eval_dl, lr_scheduler)

    noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule, prediction_type=args.prediction_type)

    # torch.compile (opt-in)
    if getattr(args, "compile", False):
        base = accelerator.unwrap_model(model)
        base.thinker.inner.L_level = torch.compile(base.thinker.inner.L_level, fullgraph=False)
        base.painter = torch.compile(base.painter)
        if hasattr(base, "bridge"):
            base.bridge = torch.compile(base.bridge)
        logger.info("torch.compile applied to thinker.inner.L_level, painter, bridge")

    if accelerator.is_main_process and args.logger == "wandb":
        run_name = getattr(args, "run_name", f"pt_run_{os.environ.get('SLURM_JOB_ID', 'local')}")

        accelerator.init_trackers(
            "sokoban-painter-thinker",
            config=OmegaConf.to_container(args, resolve=True),
            init_kwargs={"wandb": {"name": run_name}}
        )

    weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32
    num_update_steps_per_epoch = math.ceil(len(train_dl) / args.gradient_accumulation_steps)
    global_step = 0

    logger.info("***** Start Training *****")
    logger.info(f"Variant: {type(accelerator.unwrap_model(model)).__name__}")

    # TRAINING LOOP
    for epoch in range(args.num_epochs):
        model.train()
        if is_v0tok:
            accelerator.unwrap_model(model).thinker.eval()  # Frozen thinker stays in eval mode

        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in SafeIterator(enumerate(train_dl), logger=logger):
            clean_images = batch["images"].to(accelerator.device, dtype=weight_dtype)

            clean_images = clean_images * 2.0 - 1.0

            noise = torch.randn_like(clean_images)
            bsz = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps).to(weight_dtype)

            with accelerator.accumulate(model):
                base_model = accelerator.unwrap_model(model)
                log_metrics = {}

                # =========================================================
                # V0TOK LOGIC  (unconditional: blank floor tokens)
                # =========================================================
                if is_v0tok:
                    seq_len = args.dataset.resolution ** 2
                    blank_tokens = torch.full((bsz, seq_len), FLOOR_ID, device=clean_images.device, dtype=torch.long)
                    x0_pred, _ = base_model(noisy_images, timesteps, blank_tokens)
                    loss = F.mse_loss(x0_pred.float(), clean_images.float())

                    accelerator.backward(loss)
                    if getattr(args, 'max_grad_norm', 0) > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    log_metrics["loss_v0tok_mse"] = loss.detach().item()

                # =========================================================
                # V1 LOGIC (Teacher Forcing with MSE and CE)
                # =========================================================
                elif is_v1:
                    solution_tokens = bits_to_tokens(clean_images, getattr(args, 'clip_sample_range', 1.0))
                    V = base_model.thinker.vocab_size
                    G = base_model._grid

                    # Unconditional: zeros as condition input
                    blank_condition = torch.zeros_like(noisy_images)
                    enc_emb = base_model._get_enc_emb(blank_condition, noisy_images, timesteps)
                    z_H, z_L = base_model.get_initial_states(bsz)
                    z_H, z_L = z_H.to(accelerator.device), z_L.to(accelerator.device)

                    for _ in range(base_model.n_sup):
                        logits, z_H, z_L = base_model.thinker.reasoning_step(enc_emb, z_H, z_L)

                    loss_ce = F.cross_entropy(logits.view(-1, V), solution_tokens.view(-1))

                    # Painter with hard mask (Teacher Forcing)
                    true_onehot = F.one_hot(solution_tokens, num_classes=V).float()
                    true_spatial_cond = true_onehot.transpose(1, 2).reshape(bsz, V, G, G)
                    x0_pred = base_model._run_painter(noisy_images, true_spatial_cond, timesteps)
                    loss_mse = F.mse_loss(x0_pred.float(), clean_images.float())

                    loss = loss_ce + loss_mse
                    accelerator.backward(loss)
                    if getattr(args, 'max_grad_norm', 0) > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    opt_thinker.step()
                    sched_thinker.step()
                    opt_thinker.zero_grad()

                    opt_painter.step()
                    sched_painter.step()
                    opt_painter.zero_grad()

                    log_metrics.update({"loss_v1_ce": loss_ce.item(), "loss_v1_mse": loss_mse.item()})

                # =========================================================
                # V2 LOGIC (Two Stage, End-to-End Latent)
                # =========================================================
                elif is_v2:
                    # Stage 1: Thinker-only bottleneck (unconditional: zeros)
                    set_requires_grad(base_model.painter, False)
                    set_requires_grad(base_model.bridge, False)
                    set_requires_grad(base_model.thinker, True)

                    blank_condition = torch.zeros_like(noisy_images)
                    x0_pred_stage1, _ = base_model(noisy_images, timesteps, blank_condition)
                    loss_stage1 = F.mse_loss(x0_pred_stage1.float(), clean_images.float())
                    accelerator.backward(loss_stage1)

                    if getattr(args, 'max_grad_norm', 0) > 0:
                        accelerator.clip_grad_norm_(base_model.thinker.parameters(), args.max_grad_norm)

                    opt_thinker.step()
                    sched_thinker.step()
                    opt_thinker.zero_grad()
                    opt_all.zero_grad()  # Clear stale thinker grads before Stage 2

                    # Stage 2: Całość
                    set_requires_grad(base_model.painter, True)
                    set_requires_grad(base_model.bridge, True)
                    set_requires_grad(base_model.thinker, True)

                    x0_pred_stage2, _ = base_model(noisy_images, timesteps, clean_images)
                    loss_stage2 = F.mse_loss(x0_pred_stage2.float(), clean_images.float())
                    accelerator.backward(loss_stage2)
                    if getattr(args, 'max_grad_norm', 0) > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    opt_all.step()
                    sched_all.step()
                    opt_all.zero_grad()

                    log_metrics.update({"loss_v2_thinker": loss_stage1.item(), "loss_v2_all": loss_stage2.item()})

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log(log_metrics, step=global_step)

        progress_bar.close()

        # ---------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------
        model.eval()
        val_losses = []
        for val_step, batch in SafeIterator(enumerate(eval_dl), logger=logger):
            clean_images = batch["images"].to(accelerator.device, dtype=weight_dtype)
            clean_images = clean_images * 2.0 - 1.0

            noise = torch.randn_like(clean_images)
            bsz = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps).to(weight_dtype)

            with torch.no_grad():
                base_model = accelerator.unwrap_model(model)
                if is_v0tok:
                    seq_len = args.dataset.resolution ** 2
                    blank_tokens = torch.full((bsz, seq_len), FLOOR_ID, device=clean_images.device, dtype=torch.long)
                    x0_pred, _ = base_model(noisy_images, timesteps, blank_tokens)
                else:
                    blank_condition = torch.zeros_like(noisy_images)
                    x0_pred, _ = base_model(noisy_images, timesteps, blank_condition)
                val_loss = F.mse_loss(x0_pred.float(), clean_images.float())
                val_losses.append(val_loss.item())

        if val_losses:
            avg_val_loss = sum(val_losses) / len(val_losses)
            accelerator.log({"val/loss": avg_val_loss}, step=global_step)
            logger.info(f"Epoch {epoch} - val_loss: {avg_val_loss:.6f}")

        # ---------------------------------------------------------
        # SOKOBAN SAMPLING & EVALUATION
        # ---------------------------------------------------------
        if accelerator.is_main_process and (epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1):
            _run_sokoban_evaluation(
                model, noise_scheduler, args, accelerator, epoch, global_step,
                weight_dtype, is_v0tok,
            )

        if accelerator.is_main_process and (epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1):
            unet = accelerator.unwrap_model(model)
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            os.makedirs(save_path, exist_ok=True)

            if is_v0tok:
                # V0Tok zapisuje tylko Paintera i Most
                state = {k: v for k, v in unet.state_dict().items() if "thinker" not in k}
                torch.save(state, os.path.join(save_path, "model.pt"))
            else:
                torch.save(unet.state_dict(), os.path.join(save_path, "model.pt"))
            logger.info(f"Zapisano model w {save_path}")

    accelerator.end_training()

if __name__ == "__main__":
    sys.argv = [a for a in sys.argv if not a.startswith("--")]
    main()
