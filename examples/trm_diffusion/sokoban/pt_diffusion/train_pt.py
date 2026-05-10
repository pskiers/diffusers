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

# Importy z Twojego projektu
from data_factory import get_dataloaders
from data_utils import SafeIterator
from trm_wrappers import OriginalTRMRatatouilleV0Tok, OriginalTRMRatatouilleV1, OriginalTRMRatatouilleV2

check_min_version("0.34.0.dev0")
logger = get_logger(__name__, log_level="INFO")


def set_requires_grad(module: torch.nn.Module, requires_grad: bool):
    for param in module.parameters():
        param.requires_grad = requires_grad


def bits_to_tokens(bits_tensor: torch.Tensor, clip_sample_range: float = 1.0) -> torch.Tensor:
    """Tłumaczy tensory dyfuzyjne [-1, 1] na płaskie tokeny [0-7] dla TRM."""
    bits_01 = ((bits_tensor / clip_sample_range) + 1.0) / 2.0
    bits_01 = bits_01.round().long()
    B, num_bits, H, W = bits_01.shape
    tokens = torch.zeros((B, H, W), device=bits_tensor.device, dtype=torch.long)
    for i in range(num_bits):
        tokens += bits_01[:, i, :, :] * (2 ** i)
    return tokens.view(B, -1)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(args: DictConfig):
    # 1. Konfiguracja Accelerate
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

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # 2. Inicjalizacja Dataloaderów i Modelu
    train_dl, eval_dl = get_dataloaders(args)
    model = instantiate(args.model, _convert_="all")

    # Rozpoznanie wariantu eksperymentu
    is_v0tok = isinstance(model, OriginalTRMRatatouilleV0Tok) and not isinstance(model, OriginalTRMRatatouilleV1)
    is_v1 = isinstance(model, OriginalTRMRatatouilleV1) and not isinstance(model, OriginalTRMRatatouilleV2)
    is_v2 = isinstance(model, OriginalTRMRatatouilleV2)

    # V0TOK: Ładowanie wag Mózgu i zamrożenie
    if is_v0tok and getattr(args, "pretrained_thinker_path", None):
        logger.info(f"Ładowanie wyuczonego Mózgu z: {args.pretrained_thinker_path}")
        state_dict = torch.load(args.pretrained_thinker_path, map_location="cpu")
        clean_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
        model.thinker.load_state_dict(clean_state_dict, strict=True)
        set_requires_grad(model.thinker, False)

    # 3. Optymalizatory i Schedulery
    total_train_steps = len(train_dl) * args.num_epochs
    warmup_steps = args.lr_scheduler.warmup_steps * args.gradient_accumulation_steps

    if is_v2:
        # V2: Dwa osobne optymalizatory dla Stage 1 i Stage 2
        opt_thinker = instantiate(args.optimizer, params=model.get_thinker_params())
        opt_all = instantiate(args.optimizer, params=model.parameters())
        sched_thinker = get_scheduler(args.lr_scheduler.name, optimizer=opt_thinker, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps)
        sched_all = get_scheduler(args.lr_scheduler.name, optimizer=opt_all, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps)

        model, opt_thinker, opt_all, train_dl, eval_dl, sched_thinker, sched_all = accelerator.prepare(
            model, opt_thinker, opt_all, train_dl, eval_dl, sched_thinker, sched_all
        )
    else:
        # V0Tok / V1
        params = model.get_painter_params() if is_v0tok else model.parameters()
        optimizer = instantiate(args.optimizer, params=params)
        lr_scheduler = get_scheduler(args.lr_scheduler.name, optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps)

        model, optimizer, train_dl, eval_dl, lr_scheduler = accelerator.prepare(model, optimizer, train_dl, eval_dl, lr_scheduler)

    noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule, prediction_type=args.prediction_type)

    if accelerator.is_main_process and args.logger == "wandb":
        accelerator.init_trackers("sokoban-painter-thinker", config=OmegaConf.to_container(args, resolve=True))

    weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32
    num_update_steps_per_epoch = math.ceil(len(train_dl) / args.gradient_accumulation_steps)
    global_step = 0

    logger.info("***** Start Treningu *****")
    logger.info(f"Wariant: {type(accelerator.unwrap_model(model)).__name__}")

    # 4. GŁÓWNA PĘTLA TRENINGOWA
    for epoch in range(args.num_epochs):
        model.train()
        accelerator.unwrap_model(model).thinker.eval() # TRM musi myśleć deterministycznie

        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoka {epoch}")

        for step, batch in SafeIterator(enumerate(train_dl), logger=logger):
            clean_images = batch["images"].to(accelerator.device, dtype=weight_dtype)

            # W SokobanBitDataset solved to 'images', a unsolved to 'conditions'
            unsolved_conditions = batch["conditions"].to(accelerator.device, dtype=weight_dtype) if "conditions" in batch and batch["conditions"] is not None else None

            noise = torch.randn_like(clean_images)
            bsz = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps).to(weight_dtype)

            with accelerator.accumulate(model):
                base_model = accelerator.unwrap_model(model)
                log_metrics = {}

                # =========================================================
                # LOGIKA V0TOK
                # =========================================================
                if is_v0tok:
                    cond_tokens = bits_to_tokens(unsolved_conditions, getattr(args, 'clip_sample_range', 1.0))
                    noise_pred, _ = base_model(noisy_images, timesteps, cond_tokens)
                    loss = F.mse_loss(noise_pred.float(), noise.float())

                    accelerator.backward(loss)
                    optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()
                    log_metrics["loss_v0tok_mse"] = loss.detach().item()

                # =========================================================
                # LOGIKA V1 (Teacher Forcing z MSE i CE)
                # =========================================================
                elif is_v1:
                    solution_tokens = bits_to_tokens(clean_images, getattr(args, 'clip_sample_range', 1.0))

                    # Logika TRM
                    enc_emb = base_model._get_enc_emb(unsolved_conditions, noisy_images, timesteps)
                    z_H, z_L = base_model.get_initial_states(bsz)
                    z_H, z_L = z_H.to(accelerator.device), z_L.to(accelerator.device)

                    for _ in range(base_model.n_sup):
                        logits, z_H, z_L = base_model.thinker.reasoning_step(enc_emb, z_H, z_L)

                    loss_ce = F.cross_entropy(logits.view(-1, 8), solution_tokens.view(-1))

                    # Painter z twardą maską (Teacher Forcing)
                    true_onehot = F.one_hot(solution_tokens, num_classes=8).float()
                    true_spatial_cond = true_onehot.transpose(1, 2).reshape(bsz, 8, 12, 12)
                    noise_pred = base_model._run_painter(noisy_images, true_spatial_cond, timesteps)
                    loss_mse = F.mse_loss(noise_pred.float(), noise.float())

                    loss = loss_ce + loss_mse
                    accelerator.backward(loss)
                    optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()
                    log_metrics.update({"loss_v1_ce": loss_ce.item(), "loss_v1_mse": loss_mse.item()})

                # =========================================================
                # LOGIKA V2 (Dwa Stage, End-to-End Latent)
                # =========================================================
                elif is_v2:
                    # Stage 1: Thinker-only bottleneck
                    set_requires_grad(base_model.painter, False)
                    set_requires_grad(base_model.bridge, False)
                    set_requires_grad(base_model.thinker, True)

                    noise_pred_stage1, _ = base_model(noisy_images, timesteps, unsolved_conditions)
                    loss_stage1 = F.mse_loss(noise_pred_stage1.float(), noise.float())
                    accelerator.backward(loss_stage1)
                    opt_thinker.step(); sched_thinker.step(); opt_thinker.zero_grad()

                    # Stage 2: Całość
                    set_requires_grad(base_model.painter, True)
                    set_requires_grad(base_model.bridge, True)
                    set_requires_grad(base_model.thinker, True)

                    noise_pred_stage2, _ = base_model(noisy_images, timesteps, clean_images)
                    loss_stage2 = F.mse_loss(noise_pred_stage2.float(), noise.float())
                    accelerator.backward(loss_stage2)
                    opt_all.step(); sched_all.step(); opt_all.zero_grad()

                    log_metrics.update({"loss_v2_thinker": loss_stage1.item(), "loss_v2_all": loss_stage2.item()})

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log(log_metrics, step=global_step)

        progress_bar.close()

        # 5. Zapis Modelu
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
