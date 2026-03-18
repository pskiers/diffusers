from collections import defaultdict
from enum import Enum
from typing import Tuple

import numpy as np
import pkg_resources
from PIL import Image
from scipy.ndimage import label
import heapq
from collections import deque
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


class FieldStates(Enum):
    """Field states."""
    WALL = (0, "wall.png")
    FLOOR = (1, "floor.png")
    BOX_TARGET = (2, "box_target.png")
    BOX_ON_TARGET = (3, "box_on_target.png")
    BOX = (4, "box.png")
    PLAYER = (5, "player.png")
    PLAYER_ON_TARGET = (6, "player_on_target.png")

    def __init__(self, id: int, asset_file_name: str):
        self.id = id
        self.asset_file_name = asset_file_name


def load_surface(shape: Tuple[int, int]):
    """Load the surface assets."""
    asset_file_names = [field_state.asset_file_name for field_state in FieldStates]
    resource_package = __name__
    surface = []
    for asset_file_name in asset_file_names:
        asset_path = pkg_resources.resource_filename(resource_package, "/".join(("surface", asset_file_name)))
        asset_np_array = np.array(Image.open(asset_path).resize(shape))
        surface.append(asset_np_array)

    return np.stack(surface)


def render(x: np.ndarray) -> np.ndarray:
    w, h = x.shape
    render_surface = load_surface(shape=(w, h))
    res = np.empty((w**2, h**2, 3))
    for i in range(w):
        for j in range(h):
            res[i * w : (i + 1) * w, j * h : (j + 1) * h] = render_surface[x[i, j] % len(render_surface)]
    return res

def accumulate_metrics(metrics):
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


def _num_connected_components(board: np.ndarray) -> int:
    _, num_components = label(board != 0)
    return num_components


def validality_metrics(board: np.ndarray, num_boxes: int = 4) -> dict:
    metrics = {}
    """Verify whether board is valid."""
    is_board_correct = board.ndim == 2 and np.all(board >= 0) and np.all(board < 7)
    is_one_player = np.sum((board == FieldStates.PLAYER.id) | (board == FieldStates.PLAYER_ON_TARGET.id)) == 1
    box_count_match = np.sum((board == FieldStates.BOX.id) | (board == FieldStates.BOX_ON_TARGET.id)) == num_boxes
    targets_num_equal_boxes_num = np.sum((board == FieldStates.BOX_TARGET.id) | (board == FieldStates.BOX_ON_TARGET.id) | (board == FieldStates.PLAYER_ON_TARGET.id)) == num_boxes
    is_board_connected = _num_connected_components(board) == 1

    metrics = {
        'is_one_player': is_one_player,
        'box_count_match': box_count_match,
        'targets_num_equal_boxes_num': targets_num_equal_boxes_num,
        'is_board_connected': is_board_connected,
        'is_valid': is_board_correct & is_one_player & box_count_match & targets_num_equal_boxes_num & is_board_connected}

    return metrics

def conditional_is_valid(conditioning: np.ndarray, board: np.ndarray) -> dict:
    """Verify whether board is valid."""
    metrics = {}

    should_be_walls = conditioning == FieldStates.WALL.id
    walls_matching = (board[should_be_walls] == FieldStates.WALL.id).all() and (board[~should_be_walls] != FieldStates.WALL.id).all()

    should_be_targets = np.logical_or(np.logical_or(conditioning == FieldStates.BOX_TARGET.id, conditioning == FieldStates.BOX_ON_TARGET.id), conditioning == FieldStates.PLAYER_ON_TARGET.id)
    targets_matching = np.logical_or(np.logical_or(board[should_be_targets] == FieldStates.BOX_TARGET.id, board[should_be_targets] == FieldStates.BOX_ON_TARGET.id),
                                    board[should_be_targets] == FieldStates.PLAYER_ON_TARGET.id).all() and np.logical_and(
                                        np.logical_and(board[~should_be_targets] != FieldStates.BOX_TARGET.id, board[~should_be_targets] != FieldStates.BOX_ON_TARGET.id), board[~should_be_targets] != FieldStates.PLAYER_ON_TARGET.id).all()
    metrics = {
        "walls_matching": walls_matching,
        "targets_matching": targets_matching,
    }

    return metrics


def are_same_instance(board1: np.ndarray, board2: np.ndarray) -> bool:
    """Verify if two boards are the same instance."""
    if np.all((board1 == FieldStates.WALL.id) == (board2 == FieldStates.WALL.id)) and np.all(
        ((board1 == FieldStates.BOX_TARGET.id) | (board1 == FieldStates.BOX_ON_TARGET.id) | (board1 == FieldStates.PLAYER_ON_TARGET.id))
        == ((board2 == FieldStates.BOX_TARGET.id) | (board2 == FieldStates.BOX_ON_TARGET.id) | (board2 == FieldStates.PLAYER_ON_TARGET.id))
    ):
        return True
    return False


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

        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
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

    while queue and states_explored < max_states:
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
