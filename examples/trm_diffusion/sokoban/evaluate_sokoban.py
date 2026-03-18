from sokoban.utils import (
    validality_metrics, conditional_is_valid, are_same_instance, accumulate_metrics, is_solvable
)
import numpy as np
import torch
import logging


logger = logging.getLogger(__name__)


def boards_from_bit_images(images):
    bits = (images > 0.5).float()
    bits = bits.permute(0, 2, 3, 1)  # (B, H, W, num_bits)
    bits = bits.to(torch.uint8)
    boards = torch.sum(bits * (2 ** torch.arange(bits.shape[-1]).to("cpu")), -1).numpy()
    return boards


def boards_from_normalized_tensor(tensor):
    images_01 = (tensor / 2 + 0.5).clamp(0, 1)
    return boards_from_bit_images(images_01)


def compute_sokoban_metrics(
    generated_boards,
    conditioning_boards=None,
    target_boards=None,
    n_images_per_conditioning=1,
    num_boxes=4,
):
    """
    Compute Sokoban-specific evaluation metrics, matching the sokoban/ pipeline.

    Metrics computed:
      - is_valid (one_player, box_count, is_connected, targets_num)
      - conditional_is_valid (walls_correct, correct_targets, increase_boxes_on_targets)
      - same_instance_ratio
      - diversity (unique fraction per conditioning)
      - target_in_generated accuracy

    Returns:
        dict: metric_name -> float value
    """
    metrics = {}

    if len(generated_boards) == 0:
        return metrics

    # Validity metrics
    valid_results = [validality_metrics(board, num_boxes=num_boxes) for board in generated_boards]
    valid_agg = accumulate_metrics(valid_results)
    for k, v in valid_agg.items():
        metrics[f"sokoban/{k}"] = v


    # Conditional validity (if conditioning boards provided)
    if conditioning_boards is not None and len(conditioning_boards) == len(generated_boards):
        cond_results = [conditional_is_valid(cond, gen) for cond, gen in zip(conditioning_boards, generated_boards)]
        cond_agg = accumulate_metrics(cond_results)
        for k, v in cond_agg.items():
            metrics[f"sokoban/cond_{k}"] = v

        same_count = sum(are_same_instance(gen, cond)for gen, cond in zip(generated_boards, conditioning_boards))
        metrics["sokoban/matching_walls_and_targets_ratio"] = same_count / len(generated_boards)

        if target_boards is not None:
            target_in_generated = []
            for i in range(0, len(generated_boards), n_images_per_conditioning):
                gen_chunk = generated_boards[i : i + n_images_per_conditioning]
                target = target_boards[i]
                found = any(np.array_equal(target, gen) for gen in gen_chunk)
                target_in_generated.append(found)
            if target_in_generated:
                metrics["sokoban/target_in_generated"] = sum(target_in_generated) / len(target_in_generated)

    # Diversity (uniqueness fraction per conditioning group)
    if n_images_per_conditioning > 1 and len(generated_boards) >= n_images_per_conditioning:
        unique_fracs = []
        for i in range(0, len(generated_boards), n_images_per_conditioning):
            chunk = generated_boards[i : i + n_images_per_conditioning]
            if len(chunk) < 2:
                continue
            flat = chunk.reshape(len(chunk), -1)
            unique_count = np.unique(flat, axis=0).shape[0]
            unique_fracs.append(unique_count / len(chunk))
        if unique_fracs:
            metrics["sokoban/diversity"] = sum(unique_fracs) / len(unique_fracs)

    # Solvable
    solvable_results = []
    for board, is_val in zip(generated_boards, valid_results):
        if is_val['is_valid']:
            solvable = is_solvable(board, max_states=5000)
            solvable_results.append(1.0 if solvable else 0.0)
        else:
            solvable_results.append(0.0)

    if solvable_results:
        metrics["sokoban/is_solvable"] = sum(solvable_results) / len(solvable_results)

    return metrics
