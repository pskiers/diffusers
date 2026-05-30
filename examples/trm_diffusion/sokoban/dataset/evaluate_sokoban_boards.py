""" Sokoban board evaluation functions. Provides metrics for generated boards: validity, solvability, conditional metrics.
"""
import heapq
from collections import defaultdict, deque
from typing import List, Optional

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from sokoban.dataset.fields_states import FieldStates


def _accumulate_metrics(metrics: List[dict]) -> dict:
    result = defaultdict(list)
    results_acc = {}

    for metric in metrics:
        for k, v in metric.items():
            if v is not None:
                result[k].append(v)

    for k, v in result.items():
        results_acc[k] = sum(v) / len(v) if v else 0.0

    return results_acc


def generate_metrics(
    generated_boards: np.ndarray,
    num_boxes_labels: Optional[np.ndarray] = None,
    conditioning_boards: Optional[np.ndarray] = None,
    target_boards: Optional[np.ndarray] = None,
    k_values: Optional[List[int]] = None,
    n_images_per_conditioning: int = 1,
) -> dict:
    """Compute Sokoban generation metrics.

    Args:
        generated_boards: (N, H, W) integer boards, values 0-6.
        num_boxes_labels: (N,) expected box count per board. If None, derives from each board.
        conditioning_boards: (N_cond, H, W) conditioning boards for k_steps mode.
        target_boards: (N_cond, H, W) ground-truth target boards.
        k_values: per-condition k step values.
        n_images_per_conditioning: how many generated boards per condition.
    """
    metrics = {}
    if len(generated_boards) == 0:
        return metrics

    # Per-board num_boxes (either from labels or derived from each board)
    if num_boxes_labels is not None:
        boxes_per_board = np.asarray(num_boxes_labels)
    else:
        boxes_per_board = np.array([
            int(np.sum((b == FieldStates.BOX.id) | (b == FieldStates.BOX_ON_TARGET.id)))
            for b in generated_boards
        ])

    # Validity
    valid_results = [is_board_valid(board, int(nb)) for board, nb in zip(generated_boards, boxes_per_board)]
    valid_agg = _accumulate_metrics(valid_results)
    for k_metric, v in valid_agg.items():
        metrics[f"sokoban/validity_{k_metric}_ratio"] = v

    # Solvability
    solvable_results = Parallel(n_jobs=4, backend="loky")(
        delayed(is_board_solvable)(board, is_val)
        for board, is_val in zip(generated_boards, valid_results)
    )
    metrics["sokoban/solvable_in_all_percentage"] = sum(solvable_results) / len(solvable_results) if solvable_results else 0.0  # type: ignore
    valid_count = sum(1 for r in valid_results if r['is_valid'])
    metrics["sokoban/solvable_in_valid_percentage"] = sum(solvable_results) / valid_count if valid_count > 0 else 0.0 # type: ignore

    # Spatial-conditional metrics (k_steps)
    if conditioning_boards is not None and target_boards is not None:
        num_conditions = len(generated_boards) // n_images_per_conditioning

        static_results = []
        for gen_idx, gen in enumerate(generated_boards):
            c_idx = gen_idx // n_images_per_conditioning
            cond = conditioning_boards[c_idx]
            static_results.append(cond_board_structure_retention(cond, gen))
        metrics["sokoban/cond_structure_match_percentage"] = sum(static_results) / len(generated_boards)

        target_in_generated = []
        in_correct_k_distance = []

        for c_idx in range(num_conditions):
            start_idx = c_idx * n_images_per_conditioning
            end_idx = start_idx + n_images_per_conditioning
            gen_chunk = generated_boards[start_idx:end_idx]

            target = target_boards[c_idx]
            cond = conditioning_boards[c_idx]

            found_exact = any(np.array_equal(target, gen) for gen in gen_chunk)
            target_in_generated.append(found_exact)

            if k_values is not None:
                k = k_values[c_idx]
                if found_exact:
                    in_correct_k_distance.append(True)
                else:
                    k_ok = [check_k_step_dist_validity(cond, gen, k) for gen in gen_chunk]
                    in_correct_k_distance.append(any(k_ok))

        metrics["sokoban/target_in_generated_percentage"] = (
            sum(target_in_generated) / len(target_in_generated) if target_in_generated else 0.0
        )
        if k_values is not None:
            metrics["sokoban/in_correct_k_distance_percentage"] = (
                sum(in_correct_k_distance) / len(in_correct_k_distance) if in_correct_k_distance else 0.0
            )

        if n_images_per_conditioning > 1:
            unique_fracs = []
            for c_idx in range(num_conditions):
                start_idx = c_idx * n_images_per_conditioning
                end_idx = start_idx + n_images_per_conditioning
                chunk = generated_boards[start_idx:end_idx]
                if len(chunk) < 2:
                    continue
                flat = chunk.reshape(len(chunk), -1)
                unique_count = np.unique(flat, axis=0).shape[0]
                unique_fracs.append(unique_count / len(chunk))
            if unique_fracs:
                metrics["sokoban/diversity_per_conditioning_board_ratio"] = sum(unique_fracs) / len(unique_fracs)

    return {k: round(v, 2) for k, v in metrics.items()}


def is_board_valid(board: np.ndarray, num_boxes: int) -> dict:
    """Check board validity with a specific expected box count."""
    is_board_correct = board.ndim == 2 and np.all(board >= 0) and np.all(board < 8)
    is_one_player = np.sum((board == FieldStates.PLAYER.id) | (board == FieldStates.PLAYER_ON_TARGET.id)) == 1
    box_count_match = np.sum((board == FieldStates.BOX.id) | (board == FieldStates.BOX_ON_TARGET.id)) == num_boxes
    targets_num_match = np.sum(
        (board == FieldStates.BOX_TARGET.id) | (board == FieldStates.BOX_ON_TARGET.id) | (board == FieldStates.PLAYER_ON_TARGET.id)
    ) == num_boxes

    _, num_components = label(board != FieldStates.WALL.id)   # type: ignore
    is_board_connected = num_components == 1

    return {
        'one_player': is_one_player,
        'desired_boxes_number': box_count_match,
        'boxes_eq_targets': targets_num_match,
        'board_connected': is_board_connected,
        'is_valid': is_board_correct & is_one_player & box_count_match & targets_num_match & is_board_connected,
    }


def is_board_solvable(board: np.ndarray, is_val: dict, max_states: int = 100000) -> float:
    """A* search to check if a Sokoban board is solvable. Returns 1.0 if solvable, else 0.0."""
    if not is_val['is_valid']:
        return 0.0

    walls = set(zip(*np.where(board == FieldStates.WALL.id)))

    target_ids = [FieldStates.BOX_TARGET.id, FieldStates.BOX_ON_TARGET.id, FieldStates.PLAYER_ON_TARGET.id]
    targets = set(zip(*np.where(np.isin(board, target_ids))))

    box_ids = [FieldStates.BOX.id, FieldStates.BOX_ON_TARGET.id]
    boxes = tuple(zip(*np.where(np.isin(board, box_ids))))

    player_ids = [FieldStates.PLAYER.id, FieldStates.PLAYER_ON_TARGET.id]
    player_loc = np.where(np.isin(board, player_ids))

    if len(player_loc[0]) != 1:
        return 0.0

    start_player = (player_loc[0][0], player_loc[1][0])

    targets_array = np.array(list(targets))

    def heuristic(current_boxes):
        if not current_boxes:
            return 0
        boxes_array = np.array(current_boxes)
        cost_matrix = cdist(boxes_array, targets_array, metric='cityblock')
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return cost_matrix[row_ind, col_ind].sum()

    # Deadlock corners
    corners = set()
    for r in range(board.shape[0]):
        for c in range(board.shape[1]):
            if (r, c) in walls or (r, c) in targets:
                continue
            up = (r - 1, c) in walls
            down = (r + 1, c) in walls
            left = (r, c - 1) in walls
            right = (r, c + 1) in walls
            if (up or down) and (left or right):
                corners.add((r, c))

    def reachable_positions(player_start, current_boxes_set):
        visited_pos = set()
        queue = deque([player_start])
        visited_pos.add(player_start)
        max_r, max_c = board.shape

        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= max_r or nc < 0 or nc >= max_c:
                    continue
                if (nr, nc) in walls or (nr, nc) in current_boxes_set:
                    continue
                if (nr, nc) not in visited_pos:
                    visited_pos.add((nr, nc))
                    queue.append((nr, nc))
        return visited_pos

    # A* search
    start_boxes = tuple(sorted(boxes))
    queue = [(heuristic(start_boxes), 0, start_player, start_boxes)]
    visited = {}
    states_explored = 0
    MAX_QUEUE_SIZE = 100000

    while queue and states_explored < max_states:
        if len(queue) > MAX_QUEUE_SIZE:
            return 0.0

        f, g, player, current_boxes = heapq.heappop(queue)
        boxes_set = set(current_boxes)
        reachable = reachable_positions(player, boxes_set)

        normalized_player = min(reachable)
        state = (normalized_player, current_boxes)

        if state in visited and visited[state] <= g:
            continue
        visited[state] = g
        states_explored += 1

        if boxes_set <= targets:
            return 1.0

        for bx, by in current_boxes:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                player_pos = (bx - dr, by - dc)
                if player_pos not in reachable:
                    continue

                new_box = (bx + dr, by + dc)
                if new_box in walls or new_box in boxes_set or new_box in corners:
                    continue

                new_boxes = list(current_boxes)
                new_boxes.remove((bx, by))
                new_boxes.append(new_box)
                new_boxes = tuple(sorted(new_boxes))

                new_player = (bx, by)
                new_g = g + 1
                new_f = new_g + heuristic(new_boxes)
                heapq.heappush(queue, (new_f, new_g, new_player, new_boxes))

    return 0.0


# Conditional metrics
def cond_board_structure_retention(cond: np.ndarray, gen: np.ndarray) -> float:
    cond_walls = (cond == FieldStates.WALL.id)
    gen_walls = (gen == FieldStates.WALL.id)

    cond_targets = (cond == FieldStates.BOX_TARGET.id) | (cond == FieldStates.BOX_ON_TARGET.id) | (cond == FieldStates.PLAYER_ON_TARGET.id)
    gen_targets = (gen == FieldStates.BOX_TARGET.id) | (gen == FieldStates.BOX_ON_TARGET.id) | (gen == FieldStates.PLAYER_ON_TARGET.id)

    walls_match = np.array_equal(cond_walls, gen_walls)
    targets_match = np.array_equal(cond_targets, gen_targets)
    return 1.0 if (walls_match and targets_match) else 0.0


def check_k_step_dist_validity(cond: np.ndarray, gen: np.ndarray, k: int) -> bool:
    if k == 0:
        return np.array_equal(cond, gen)

    player_ids = [FieldStates.PLAYER.id, FieldStates.PLAYER_ON_TARGET.id]

    cond_p_loc = np.where(np.isin(cond, player_ids))
    gen_p_loc = np.where(np.isin(gen, player_ids))

    if len(cond_p_loc[0]) != 1 or len(gen_p_loc[0]) != 1:
        return False

    cond_y, cond_x = cond_p_loc[0][0], cond_p_loc[1][0]
    gen_y, gen_x = gen_p_loc[0][0], gen_p_loc[1][0]

    manhattan_distance = abs(cond_x - gen_x) + abs(cond_y - gen_y)
    if manhattan_distance > k:
        return False

    # For k=1 the player must have moved; for k>=2 backtracking is possible. But at least one box must have changed position (otherwise nothing happened)
    box_ids = [FieldStates.BOX.id, FieldStates.BOX_ON_TARGET.id]
    cond_boxes = set(zip(*np.where(np.isin(cond, box_ids))))
    gen_boxes = set(zip(*np.where(np.isin(gen, box_ids))))

    if manhattan_distance == 0 and cond_boxes == gen_boxes:
        return False

    # Structure unchanged
    if not np.array_equal((cond == FieldStates.WALL.id), (gen == FieldStates.WALL.id)):
        return False

    cond_targets = (cond == FieldStates.BOX_TARGET.id) | (cond == FieldStates.BOX_ON_TARGET.id) | (cond == FieldStates.PLAYER_ON_TARGET.id)
    gen_targets = (gen == FieldStates.BOX_TARGET.id) | (gen == FieldStates.BOX_ON_TARGET.id) | (gen == FieldStates.PLAYER_ON_TARGET.id)
    if not np.array_equal(cond_targets, gen_targets):
        return False

    # Max boxes in different positions
    boxes_diff = cond_boxes.symmetric_difference(gen_boxes)
    if len(boxes_diff) > 2 * k:
        return False

    return True
