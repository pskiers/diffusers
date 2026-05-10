import os
import torch
import numpy as np
from PIL import Image
import logging
import hydra
import pydantic
from omegaconf import DictConfig, OmegaConf

from trm_paper_utils.utils.functions import load_model_class
from sokoban.fields_states import FieldStates
from sokoban.sokoban_utils import SokobanSampler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str

class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig

class SampleConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='ignore')

    arch: ArchConfig
    checkpoint_path: str
    output_dir: str = "../outputs/samples"
    num_samples: int = 5
    batch_size: int = 1

    vocab_size: int = 8
    seq_len: int = 144
    num_puzzle_identifiers: int = 1
    seed: int = 42

class DummyDatasetArgs:
    num_boxes = 4
    concat_conditioning = False

class DummyArgs:
    dataset = DummyDatasetArgs()


@torch.inference_mode()
def sample_impl(config: SampleConfig):
    os.makedirs(config.output_dir, exist_ok=True)
    torch.random.manual_seed(config.seed)
    np.random.seed(config.seed)

    model_cfg = dict(
        **config.arch.__pydantic_extra__, # type: ignore
        batch_size=config.batch_size,
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        num_puzzle_identifiers=config.num_puzzle_identifiers,
        causal=False
    )

    model_cls = load_model_class(config.arch.name)
    model = model_cls(model_cfg)
    model.cuda()
    model.eval()

    logger.info(f"Loading from checkpoint: {config.checkpoint_path}")
    state_dict = torch.load(config.checkpoint_path, map_location="cuda")

    # Skipping loss func's wrapper used in pretrain
    clean_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
    model.load_state_dict(clean_state_dict, strict=False)

    sampler = SokobanSampler(DummyArgs())

    for sample_idx in range(config.num_samples):
        logger.info(f"Generating sample {sample_idx + 1}/{config.num_samples}...")

        # Inicjalizacja pustej planszy (same podłogi przesunięte o +1)
        empty_board = np.full((config.batch_size, config.seq_len), fill_value=FieldStates.FLOOR.id, dtype=np.int32)
        batch = {
            "inputs": torch.tensor(empty_board).cuda(),
            "puzzle_identifiers": torch.tensor([[0]] * config.batch_size).cuda()
        }

        carry = model.initial_carry(batch)
        steps_taken = 0

        while True:
            carry, outputs = model(carry, batch)
            steps_taken += 1

            if carry.halted.all() or steps_taken >= model_cfg.get("halt_max_steps", 16):
                break

        logger.info(f"Wnioskowanie ACT zakończone po {steps_taken} iteracjach.")

        logits = outputs["logits"] # (1, 144, 8)
        predictions_1d = logits.argmax(dim=-1).squeeze(0).cpu().numpy() # (144,)
        board_2d = predictions_1d.reshape(12, 12)

        rendered_np = sampler._render(board_2d)

        img = Image.fromarray(rendered_np.astype(np.uint8))
        save_path = os.path.join(config.output_dir, f"trm_sample_{sample_idx + 1:04d}.png")
        img.save(save_path)
        logger.info(f"Saved in: {save_path}")


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def main(hydra_config: DictConfig):
    config_dict = OmegaConf.to_container(hydra_config, resolve=True)

    sample_config = SampleConfig(
        arch=config_dict.get("arch"),
        checkpoint_path=config_dict.get("checkpoint_path", ""),
        output_dir=config_dict.get("output_dir", "../outputs/trm_sokoban/samples"),
        num_samples=config_dict.get("num_samples", 10),
        seed=config_dict.get("seed", 42)
    )

    if not sample_config.checkpoint_path:
        raise ValueError("Checkpoint path is requiered.")

    sample_impl(sample_config)


if __name__ == "__main__":
    main()
