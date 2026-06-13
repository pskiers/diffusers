"""
Sampling script for Bit-Diffusion Sokoban (standard and TRM).

Loads the best checkpoint, generates boards via lit_model.evaluate(),
prints metrics, and saves rendered images.

Usage (standard diffusion):
  python sample.py --config-name=standard_diffusion run_name=my_run checkpoint_path=best

Usage (TRM diffusion):
  python sample.py --config-name=trm_diffusion run_name=my_run checkpoint_path=best
"""
import json
import sys
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from tqdm.auto import tqdm

from diffusers import DDPMScheduler, Transformer2DModel

from sokoban.bit_diffusion.train_std import SokobanBitDataModule, SokobanBitDiffusion
from sokoban.bit_diffusion.train_trm import SokobanTRMBitDiffusion, TRMDiT


def find_best_checkpoint(output_dir: str, run_name: str, is_trm: bool) -> str:
    """Szuka najlepszego checkpointu z uwzględnieniem struktury katalogów."""
    base_dir = Path(output_dir)
    # W train_std.py zapisywałaś do output_dir / run_name / "std_checkpoints"
    # W train_trm.py zapisywałaś do output_dir / "checkpoints"
    # (warto w train_trm.py ujednolicić ścieżkę do: Path(cfg.output_dir) / run_name / "checkpoints")

    if is_trm:
        ckpt_dir = base_dir / "checkpoints" # Dostosuj, jeśli dodałaś run_name do train_trm.py
    else:
        ckpt_dir = base_dir / run_name / "std_checkpoints"

    candidates = list(ckpt_dir.glob("best-*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"Nie znaleziono pliku best-*.ckpt w katalogu {ckpt_dir}")

    # Pobiera najnowszy plik, jeśli jest ich więcej
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


@hydra.main(version_base=None, config_path="config", config_name="trm_diffusion")
def main(cfg: DictConfig):
    # Inicjalizacja ziarna dla powtarzalności wyników
    seed = cfg.get("seed", 42)
    L.seed_everything(seed, workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_trm = cfg.get("trm") is not None

    run_name = cfg.get("run_name", None)
    if not run_name and not is_trm:
        raise ValueError("Musisz podać nazwę uruchomienia (run_name) do samplowania standardowej dyfuzji.")

    # Rozwiązywanie ścieżki checkpointu
    ckpt_path = cfg.get("checkpoint_path", "best")
    if ckpt_path == "best":
        ckpt_path = find_best_checkpoint(cfg.output_dir, run_name or "", is_trm)
    elif not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Nie znaleziono podanego checkpointu: {ckpt_path}")

    print(f"Ładowanie checkpointu: {ckpt_path}")

    # Logika dla dynamicznego num_classes (zależna od conditioning)
    num_bits = cfg.num_bits
    k_values = cfg.dataset.get("k_values", [1, 3, 5, 8, 10])
    use_self_cond = cfg.get("self_cond", False)

    if cfg.conditioning == "k_steps":
        in_channels = num_bits * (2 if use_self_cond else 1) + num_bits
        num_classes = len(k_values)
    else:
        in_channels = num_bits * (2 if use_self_cond else 1)
        num_classes = cfg.num_classes

    # Budowa rdzenia modelu Transformera
    core_model = Transformer2DModel(
        sample_size=cfg.resolution,
        in_channels=in_channels,
        out_channels=num_bits,
        num_layers=cfg.model.num_layers,
        patch_size=cfg.model.patch_size,
        attention_head_dim=cfg.model.attention_head_dim,
        num_attention_heads=cfg.model.num_attention_heads,
        cross_attention_dim=cfg.model.get("cross_attention_dim", None),
        activation_fn=cfg.model.get("activation_fn", "gelu-approximate"),
        dropout=cfg.model.get("dropout", 0.0),
        num_embeds_ada_norm=num_classes + 1,
        norm_type="ada_norm_zero",
    )

    # Rekonstrukcja harmonogramu szumu (nie polegamy na obiekcie zapiklowanym w .ckpt)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.get("ddpm_num_train_timesteps", 1000),
        beta_schedule=cfg.get("beta_schedule", "squaredcos_cap_v2"),
        prediction_type=cfg.get("prediction_type", "sample"),
        rescale_betas_zero_snr=cfg.get("rescale_betas_zero_snr", True),
        clip_sample=True,
        clip_sample_range=1.0,
    )

    # Inicjalizacja modelu PyTorch Lightning (Wagi EMA ładują się natywnie z .ckpt)
    if is_trm:
        model = TRMDiT(
            core_model=core_model,
            resolution=cfg.resolution,
            n=cfg.trm.n,
            T=cfg.trm.T,
            n_sup_max=cfg.n_sup_max,
            use_grid_pos_embed=cfg.trm.get("use_grid_pos_embed", True),
        )
        lit_model = SokobanTRMBitDiffusion.load_from_checkpoint(
            ckpt_path, model=model, noise_scheduler=noise_scheduler, map_location=device,
        )
        print("Wczytano model TRM (wagi EMA wbudowane w plik .ckpt).")
    else:
        model = core_model
        lit_model = SokobanBitDiffusion.load_from_checkpoint(
            ckpt_path, model=model, noise_scheduler=noise_scheduler, map_location=device,
        )
        print("Wczytano model Standardowej Dyfuzji (wagi EMA wbudowane w plik .ckpt).")

    lit_model.eval()
    lit_model.to(device)

    # Przekazanie parametru time_shift_xi ze struktury konfiguracyjnej
    lit_model.time_shift_xi = cfg.get("time_shift_xi", 0.0)

    # Inicjalizacja Modułu Danych
    num_samples = cfg.get("num_samples", 200)
    batch_size = cfg.get("sample_batch_size", 50)

    data_module = SokobanBitDataModule(
        data_path=cfg.dataset.get("test_path", cfg.dataset.data_path),
        conditioning=cfg.conditioning,
        total_train_size=cfg.dataset.total_train_size,
        total_eval_size=cfg.dataset.total_eval_size,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        num_bits=num_bits,
        k_values=k_values,
    )
    data_module.setup()

    # Generowanie i walidacja plansz Sokobana
    print(f"Generowanie {num_samples} plansz...")
    metrics, gen_bits = lit_model.evaluate(data_module.val_dataloader(), num_samples=num_samples)

    print("\n--- Sokoban Metrics ---")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")

    # Struktura zapisu wyników (rozdzielenie TRM od standard)
    shift_str = f"_shift_{lit_model.time_shift_xi}"
    prefix = "trm" if is_trm else "std"
    run_dir = run_name if run_name else "trm_default"

    sample_dir = Path(cfg.output_dir) / run_dir / f"{prefix}_samples{shift_str}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Zapis wyrenderowanych obrazów
    print(f"\nZapisywanie wyrenderowanych plansz do {sample_dir}...")
    val_ds = data_module.val_ds
    for i in tqdm(range(min(num_samples, 100)), desc="Rendering"):
        img = val_ds.render_bit_boards(gen_bits[i])
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img.save(sample_dir / f"board_{i:04d}.png")

    # Zapis metryk
    metrics_path = sample_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metryki poprawnie zapisane w {metrics_path}")


if __name__ == "__main__":
    # Strip launcher-injected flags (e.g. accelerate) but keep Hydra's --config-name/--config-path
    sys.argv = [a for a in sys.argv if not a.startswith("--") or a.startswith("--config")]
    main()
