import numpy as np
from typing import Optional


def generate_metrics(
    generated_boards: np.ndarray,
    conditioning_boards: Optional[np.ndarray] = None,
    target_boards: Optional[np.ndarray] = None,
    k_values: Optional[list[int]] = None,
    n_images_per_conditioning: Optional[int] = None
):
    if n_images_per_conditioning is not None:
        return unconditional_generation_metrics(generated_boards)
    else:
        return conditional_generation_metrics(
            generated_boards, conditioning_boards, target_boards, k_values, n_images_per_conditioning
        )


def conditional_generation_metrics(
    generated_boards: np.ndarray,
    conditioning_boards: Optional[np.ndarray] = None,
    target_boards: Optional[np.ndarray] = None,
    k_values: Optional[list[int]] = None,
    n_images_per_conditioning: Optional[int] = None
):
    base_uncond_metrics = unconditional_generation_metrics(generated_boards)



def unconditional_generation_metrics(generated_boards: np.ndarray):
    ...


def _is_board_valid(board):
    ...


from joblib import Parallel, delayed
from collections import defaultdict
from enum import Enum
from typing import Tuple, List, Optional
import pkg_resources
from PIL import Image
from scipy.ndimage import label
from omegaconf import ListConfig
import numpy as np
import heapq
import torch
import json
from collections import deque
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from data_factory import get_dataloaders
from sokoban.fields_states import FieldStates
from diffusers.utils import is_accelerate_version


class SokobanSampler:
    def __init__(self, args) -> None:
        self.all_cond_boards_list = []
        self.all_k_values_list = []
        self.all_target_boards_list = []
        self.all_gen_boards_list = []
        self._surface_cache = {}
        self._gen_rendered_cache = None

        self.num_boxes = args.dataset.num_boxes
        self.evaluator = SokobanEvaluator(self.num_boxes)
        is_concat_cond = getattr(args.dataset, 'concat_conditioning', False)
        self.n_images_per_cond = getattr(args.dataset, 'n_images_per_conditioning', 1)
        self.n_images_to_eval = getattr(args.dataset, 'n_images_to_eval', 1)

        self.prompt_iterator = None
        if is_concat_cond:
            _, self.eval_dl = get_dataloaders(args)
            self.prompt_iterator = iter(self.eval_dl)

        self.k_list = getattr(args.dataset, 'k', [0])
        if isinstance(self.k_list, (int, float)):
            self.k_list = [self.k_list]
        elif isinstance(self.k_list, ListConfig):
            self.k_list = list(self.k_list)

    def render_boards(self):
        if self._gen_rendered_cache is not None:
            return self._gen_rendered_cache

        boards = np.concatenate(self.all_gen_boards_list, axis=0)

        rendered_images = []
        for board in boards:
            rendered_np = self._render(board) # np.array [H, W, 3] 0-255
            rendered_tensor = torch.from_numpy(rendered_np).permute(2, 0, 1).float() / 255.0
            rendered_images.append(rendered_tensor)

        self._gen_rendered_cache = torch.stack(rendered_images)
        return self._gen_rendered_cache

    def sampling_evaluation(self, logger, process_index=None, output_dir=None, accelerator=None, global_step=None, log_with=None):
        all_boards = np.concatenate(self.all_gen_boards_list, axis=0)
        cond_boards_final = np.concatenate(self.all_cond_boards_list, axis=0) if self.all_cond_boards_list else None
        target_boards_final = np.concatenate(self.all_target_boards_list, axis=0) if self.all_target_boards_list else None

        sokoban_metrics = self.evaluator.generate_metrics(
            generated_boards=all_boards,
            conditioning_boards=cond_boards_final,
            target_boards=target_boards_final,
            k_values=self.all_k_values_list if self.all_k_values_list else None,
            n_images_per_conditioning=self.n_images_per_cond
        )

        if log_with and accelerator and global_step:    # training
            if log_with == "wandb":
                import wandb
                accelerator.get_tracker("wandb").log(sokoban_metrics, step=global_step)

                if cond_boards_final is not None:
                    cond_rendered = []
                    for board in cond_boards_final:
                        r_tensor = torch.from_numpy(self._render(board)).permute(2, 0, 1).float() / 255.0
                        cond_rendered.append(r_tensor)
                    cond_rendered = torch.stack(cond_rendered)

                    gen_rendered_tensor = self.render_boards()

                    n_per_cond = self.n_images_per_cond
                    N_conds = len(cond_rendered)
                    gen_reshaped = gen_rendered_tensor.view(N_conds, n_per_cond, 3, gen_rendered_tensor.shape[-2], gen_rendered_tensor.shape[-1])

                    columns = ["k_distance", "Condition"] + [f"Generated {i+1}" for i in range(n_per_cond)]
                    table = wandb.Table(columns=columns)

                    for i in range(N_conds):
                        k_val = self.all_k_values_list[i] if self.all_k_values_list else "N/A"
                        row = [k_val, wandb.Image(cond_rendered[i])]
                        for j in range(n_per_cond):
                            row.append(wandb.Image(gen_reshaped[i, j]))
                        table.add_data(*row)

                    accelerator.get_tracker("wandb").log({"Sokoban_Details": table}, step=global_step)

            elif log_with == "tensorboard":
                tracker = accelerator.get_tracker("tensorboard", unwrap=True) if is_accelerate_version(">=", "0.17.0.dev0") else accelerator.get_tracker("tensorboard")
                for k, v in sokoban_metrics.items():
                    tracker.add_scalar(k, v, global_step)

        elif output_dir:  # sampling
            logger.info(f"Sokoban sampling metrics (rank {process_index}): {sokoban_metrics}")

            metrics_path = output_dir / f"sokoban_metrics_rank{process_index}.json"
            with open(metrics_path, "w") as f:
                json.dump({k: float(v) for k, v in sokoban_metrics.items()}, f, indent=2)

            logger.info(f"Sokoban metrics saved to {metrics_path}")

        else:
            raise ValueError('Provide output_dir or (log_with and accelerator and global_step)')

    def register_generated(self, generated_boards_np): # generated_boards_np: numpy [B, H, W, C] w zakresie 0-255.
        bits = (generated_boards_np > 127).astype(np.uint8)
        powers = 2 ** np.arange(bits.shape[-1])
        boards = np.sum(bits * powers, axis=-1)
        self.all_gen_boards_list.append(boards)
        self._gen_rendered_cache = None

    def prepare_for_conditioning(self, accelerator, weight_dtype, current_bsz):
        cond_images_prompt = None
        class_labels_prompt = None

        if self.prompt_iterator is not None:
            num_unique_needed = current_bsz // self.n_images_per_cond

            conds_collected = []
            targets_collected = []
            labels_collected = []
            collected_count = 0

            while collected_count < num_unique_needed:
                try:
                    cond_batch = next(self.prompt_iterator)
                except StopIteration:
                    self.prompt_iterator = iter(self.eval_dl)
                    cond_batch = next(self.prompt_iterator)

                labels = cond_batch.get("class_labels")

                take = min(len(cond_batch["conditions"]), num_unique_needed - collected_count)
                conds_collected.append(cond_batch["conditions"][:take])
                targets_collected.append(cond_batch["images"][:take])
                if labels is not None:
                    labels_collected.append(labels[:take])

                collected_count += take

            cond_slice = torch.cat(conds_collected, dim=0).to(accelerator.device, dtype=weight_dtype)
            target_slice = torch.cat(targets_collected, dim=0).to(accelerator.device, dtype=weight_dtype)

            cond_images_prompt = cond_slice.repeat_interleave(self.n_images_per_cond, dim=0)

            if labels_collected and labels_collected[0] is not None:
                label_slice = torch.cat(labels_collected, dim=0).to(accelerator.device)
                class_labels_prompt = label_slice.repeat_interleave(self.n_images_per_cond, dim=0)
                for lbl in label_slice.tolist():
                    self.all_k_values_list.append(self.k_list[lbl])
            else:
                self.all_k_values_list.extend([self.k_list[0]] * num_unique_needed)

            self.all_cond_boards_list.append(self._boards_from_normalized_tensor(cond_slice))
            self.all_target_boards_list.append(self._boards_from_normalized_tensor(target_slice))

            current_bsz = cond_images_prompt.shape[0]

        return cond_images_prompt, class_labels_prompt, current_bsz

    def _boards_from_normalized_tensor(self, tensor):
        images_01 = (tensor / 2 + 0.5).clamp(0, 1)
        bits = (images_01 > 0.5).float()
        bits = bits.permute(0, 2, 3, 1).to(torch.uint8)

        powers = 2 ** torch.arange(bits.shape[-1], device=bits.device)
        boards = torch.sum(bits * powers, dim=-1).cpu().numpy()
        return boards

    def _load_surface(self, shape: Tuple[int, int]):
        if shape in self._surface_cache:
            return self._surface_cache[shape]

        asset_file_names = [field_state.asset_file_name for field_state in FieldStates]
        resource_package = __name__
        surface = []
        for asset_file_name in asset_file_names:
            asset_path = pkg_resources.resource_filename(resource_package, "/".join(("surface", asset_file_name)))
            asset_np_array = np.array(Image.open(asset_path).convert("RGB").resize(shape))
            surface.append(asset_np_array)

        self._surface_cache[shape] = np.stack(surface)
        return self._surface_cache[shape]

    def _render(self, x: np.ndarray) -> np.ndarray:
        w, h = x.shape
        render_surface = self._load_surface(shape=(w, h))
        res = np.empty((w**2, h**2, 3))
        for i in range(w):
            for j in range(h):
                res[i * w : (i + 1) * w, j * h : (j + 1) * h] = render_surface[x[i, j] % len(render_surface)]
        return res


class SokobanEvaluator:
    def __init__(self, num_boxes: int = 4) -> None:
        self.num_boxes = num_boxes

    def generate_metrics(
            self,
            generated_boards: np.ndarray,
            conditioning_boards: Optional[np.ndarray] = None,
            target_boards: Optional[np.ndarray] = None,
            k_values: Optional[List[int]] = None,
            n_images_per_conditioning: int = 1
        ) -> dict:
        metrics = {}
        if len(generated_boards) == 0:
            return metrics

        valid_results = [self._is_board_valid(board) for board in generated_boards]
        valid_agg = self._accumulate_metrics(valid_results)
        for k_metric, v in valid_agg.items():
            metrics[f"sokoban/validity_{k_metric}_ratio"] = v

        solvable_results = Parallel(n_jobs=8, backend="loky")(
            delayed(self._is_board_solvable)(board, is_val)
            for board, is_val in zip(generated_boards, valid_results)
        )

        metrics["sokoban/solvable_in_all_percentage"] = sum(solvable_results) / len(solvable_results) if solvable_results else 0.0

        valid_count = sum(1 for is_val in valid_results if is_val['is_valid'])
        metrics["sokoban/solvable_in_valid_percentage"] = sum(solvable_results) / valid_count if valid_count > 0 else 0.0

        # Metrics for conditional generation
        if conditioning_boards is not None and target_boards is not None:
            num_conditions = len(generated_boards) // n_images_per_conditioning

            static_results = []
            for gen_idx, gen in enumerate(generated_boards):
                c_idx = gen_idx // n_images_per_conditioning
                cond = conditioning_boards[c_idx]
                static_results.append(self._cond_board_structure_retention(cond, gen))
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
                        k_distances_correctness = [self._check_k_step_dist_validity(cond, gen, k) for gen in gen_chunk]
                        in_correct_k_distance.append(any(k_distances_correctness))

            metrics["sokoban/target_in_generated_percentage"] = sum(target_in_generated) / len(target_in_generated) if target_in_generated else 0.0
            if k_values is not None:
                metrics["sokoban/in_correct_k_distance_percentage"] = sum(in_correct_k_distance) / len(in_correct_k_distance) if in_correct_k_distance else 0.0

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

        return metrics

    def _is_board_valid(self, board: np.ndarray) -> dict:
        is_board_correct = board.ndim == 2 and np.all(board >= 0) and np.all(board < 8)
        is_one_player = np.sum((board == FieldStates.PLAYER.id) | (board == FieldStates.PLAYER_ON_TARGET.id)) == 1
        box_count_match = np.sum((board == FieldStates.BOX.id) | (board == FieldStates.BOX_ON_TARGET.id)) == self.num_boxes
        targets_num_match = np.sum((board == FieldStates.BOX_TARGET.id) | (board == FieldStates.BOX_ON_TARGET.id) | (board == FieldStates.PLAYER_ON_TARGET.id)) == self.num_boxes

        _, num_components = label(board != 0)
        is_board_connected = num_components == 1

        return {
            'one_player': is_one_player,
            'desired_boxes_number': box_count_match,
            'boxes_eq_targets': targets_num_match,
            'board_connected': is_board_connected,
            'is_valid': is_board_correct & is_one_player & box_count_match & targets_num_match & is_board_connected
        }

    def _cond_board_structure_retention(self, cond: np.ndarray, gen: np.ndarray) -> float:
        cond_walls = (cond == FieldStates.WALL.id)
        gen_walls = (gen == FieldStates.WALL.id)

        cond_targets = (cond == FieldStates.BOX_TARGET.id) | (cond == FieldStates.BOX_ON_TARGET.id) | (cond == FieldStates.PLAYER_ON_TARGET.id)
        gen_targets = (gen == FieldStates.BOX_TARGET.id) | (gen == FieldStates.BOX_ON_TARGET.id) | (gen == FieldStates.PLAYER_ON_TARGET.id)

        walls_match = np.array_equal(cond_walls, gen_walls)
        targets_match = np.array_equal(cond_targets, gen_targets)

        return 1.0 if (walls_match and targets_match) else 0.0

    def _check_k_step_dist_validity(self, cond: np.ndarray, gen: np.ndarray, k: int) -> float:
        if k == 0:
            return 1.0 if np.array_equal(cond, gen) else 0.0

        player_ids = [FieldStates.PLAYER.id, FieldStates.PLAYER_ON_TARGET.id]

        cond_p_loc = np.where(np.isin(cond, player_ids))
        gen_p_loc = np.where(np.isin(gen, player_ids))

        if len(cond_p_loc[0]) != 1 or len(gen_p_loc[0]) != 1:   # one player
            return 0.0

        cond_y, cond_x = cond_p_loc[0][0], cond_p_loc[1][0]
        gen_y, gen_x = gen_p_loc[0][0], gen_p_loc[1][0]

        manhattan_distance = abs(cond_x - gen_x) + abs(cond_y - gen_y)

        if manhattan_distance == 0 or manhattan_distance > k:
            return 0.0

        cond_walls = (cond == FieldStates.WALL.id)  # board structure unchanged
        gen_walls = (gen == FieldStates.WALL.id)
        if not np.array_equal(cond_walls, gen_walls):
            return 0.0

        cond_targets = (cond == FieldStates.BOX_TARGET.id) | (cond == FieldStates.BOX_ON_TARGET.id) | (cond == FieldStates.PLAYER_ON_TARGET.id)
        gen_targets = (gen == FieldStates.BOX_TARGET.id) | (gen == FieldStates.BOX_ON_TARGET.id) | (gen == FieldStates.PLAYER_ON_TARGET.id)
        if not np.array_equal(cond_targets, gen_targets):
            return 0.0

        box_ids = [FieldStates.BOX.id, FieldStates.BOX_ON_TARGET.id]    # max boxes in different positions
        cond_boxes = set(zip(*np.where(np.isin(cond, box_ids))))
        gen_boxes = set(zip(*np.where(np.isin(gen, box_ids))))

        boxes_diff = cond_boxes.symmetric_difference(gen_boxes)
        if len(boxes_diff) > 2 * k: # one push = two different elements in sets
            return 0.0

        return 1.0

    def _is_board_solvable(self, board, is_val):
        def is_solvable(board: np.ndarray, max_states: int = 5000) -> bool:
            walls = set(zip(*np.where(board == FieldStates.WALL.id)))

            target_ids = [FieldStates.BOX_TARGET.id, FieldStates.BOX_ON_TARGET.id, FieldStates.PLAYER_ON_TARGET.id]
            targets = set(zip(*np.where(np.isin(board, target_ids))))

            box_ids = [FieldStates.BOX.id, FieldStates.BOX_ON_TARGET.id]
            boxes = tuple(zip(*np.where(np.isin(board, box_ids))))

            player_ids = [FieldStates.PLAYER.id, FieldStates.PLAYER_ON_TARGET.id]
            player_loc = np.where(np.isin(board, player_ids))

            if len(player_loc[0]) != 1:
                return False

            start_player = (player_loc[0][0], player_loc[1][0])

            # (Min-Cost Bipartite Matching)
            targets_array = np.array(list(targets))
            def heuristic(current_boxes):
                if not current_boxes:
                    return 0

                boxes_array = np.array(current_boxes)
                cost_matrix = cdist(boxes_array, targets_array, metric='cityblock')
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                return cost_matrix[row_ind, col_ind].sum()

            # Deadlock detection
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

            # Reachability for player
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

            # A* Search
            start_boxes = tuple(sorted(boxes))

            queue = [(heuristic(start_boxes), 0, start_player, start_boxes)] # (f_score, g_score, player_pos, boxes)
            visited = {}

            states_explored = 0
            MAX_QUEUE_SIZE = 20000

            while queue and states_explored < max_states:
                if len(queue) > MAX_QUEUE_SIZE:
                    return False

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
                    return True

                # Actions
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

            return False

        if is_val['is_valid']:
            return 1.0 if is_solvable(board, max_states=5000) else 0.0
        return 0.0

    def _accumulate_metrics(self, metrics):
        result = defaultdict(list)
        results_acc = {}

        for metric in metrics:
            for k, v in metric.items():
                if v is not None:
                    result[k].append(v)

        for k, v in result.items():
            if len(v) > 0:
                results_acc[k] = sum(v) / len(v)
            else:
                results_acc[k] = 0.0

        return results_acc
