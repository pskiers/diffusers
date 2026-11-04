"""experiments/run_interpretability.py — run one interpretability ablation (experiments/interpretability/).

Usage (from examples/trm_diffusion, with PYTHONPATH=$PWD):
    python experiments/run_interpretability.py experiment=amaze_thinker_v2_controlnet \
      painter.checkpoint=runs/pt_maze_final_painter/checkpoint_final.pt \
      +interprebability_config.ablation=z_representation_ablation \
      +interprebability_config.checkpoint=runs/pt_maze_final_thinker/checkpoint_final.pt \
      +interprebability_config.data_parquet=data/amaze/maze/square/n8_test.parquet \
      +interprebability_config.num_inference_steps=20 \
      "+interprebability_config.record_at=[70,45,20]"

Other interprebability_config keys: num_samples (N metric boards, 128), template_samples (K template boards, 32),
batch_size (16), seed (0), output_dir, t_start (70) and prefix_frac (0.5) for z_representation_ablation,
keep_L_indices (None = every z_L update).
"""
import hydra
from omegaconf import DictConfig

from experiments.interpretability import launch_interpretability_ablation


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    ablation_name = cfg.interprebability_config.get("ablation", "z_changes_ablation")
    launch_interpretability_ablation(ablation_name)().run(cfg)


if __name__ == "__main__":
    main()
