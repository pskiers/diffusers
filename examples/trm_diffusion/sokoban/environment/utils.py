from collections import defaultdict
from enum import Enum
from typing import Tuple

import numpy as np
import pkg_resources
from PIL import Image
from scipy.ndimage import label


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
    for metric in metrics:
        for k, v in metric.items():
            if v is not None:
                result[k].append(v)

    for k, v in result.items():
        result[k] = sum(result[k])/len(result[k])

    return result





def _num_connected_components(board: np.ndarray) -> int:
    _, num_components = label(board != 0)
    return num_components


def is_valid(board: np.ndarray, num_boxes: int = 4) -> bool:
    metrics = {}
    """Verify whether board is valid."""
    is_board_correct = board.ndim == 2 and np.all(board >= 0) and np.all(board < 7)
    one_player = np.sum((board == FieldStates.PLAYER.id) | (board == FieldStates.PLAYER_ON_TARGET.id)) == 1
    box_count = np.sum((board == FieldStates.BOX.id) | (board == FieldStates.BOX_ON_TARGET.id)) == num_boxes
    is_connected = _num_connected_components(board) == 1
    targets_num = np.sum((board == FieldStates.BOX_TARGET.id) | (board == FieldStates.BOX_ON_TARGET.id) | (board == FieldStates.PLAYER_ON_TARGET.id)) == num_boxes

    metrics = {'one_player': one_player, 'box_count': box_count, 'is_connected': is_connected, 'targets_num': targets_num,
               'is_valid': is_board_correct & one_player & box_count & is_connected & targets_num}

    return metrics

def conditional_is_valid(conditioning: np.ndarray, board: np.ndarray) -> dict:
    """Verify whether board is valid."""
    metrics = {}

    should_be_walls = conditioning == FieldStates.WALL.id
    walls_correct = (board[should_be_walls] == FieldStates.WALL.id).all() and (board[~should_be_walls] != FieldStates.WALL.id).all()

    should_be_targets = np.logical_or(np.logical_or(conditioning == FieldStates.BOX_TARGET.id, conditioning == FieldStates.BOX_ON_TARGET.id), conditioning == FieldStates.PLAYER_ON_TARGET)
    correct_targets = np.logical_or(np.logical_or(board[should_be_targets] == FieldStates.BOX_TARGET.id, board[should_be_targets] == FieldStates.BOX_ON_TARGET.id),
                                    board[should_be_targets] == FieldStates.PLAYER_ON_TARGET.id).all() and np.logical_and(
                                        np.logical_and(board[~should_be_targets] != FieldStates.BOX_TARGET.id, board[~should_be_targets] != FieldStates.BOX_ON_TARGET.id), board[~should_be_targets] != FieldStates.PLAYER_ON_TARGET.id).all()
    correct_boxes = ((board == FieldStates.BOX.id).sum() + (board == FieldStates.BOX_ON_TARGET.id).sum()) == ((conditioning == FieldStates.BOX.id).sum() + (conditioning == FieldStates.BOX_ON_TARGET.id).sum())
    increase_boxes = None
    if correct_targets and correct_boxes:
        num_boxes_cond = (conditioning == FieldStates.BOX_ON_TARGET.id).sum()
        num_boxes = (board == FieldStates.BOX_ON_TARGET.id).sum()
        increase_boxes = num_boxes - num_boxes_cond

    metrics = {"walls_correct": walls_correct, "correct_targets": correct_targets, "increase_boxes_on_targets": increase_boxes}

    return metrics


def is_solved(board: np.ndarray, num_boxes: int = 4) -> bool:
    return np.sum(board == FieldStates.BOX_ON_TARGET.id) == num_boxes


def are_same_instance(board1: np.ndarray, board2: np.ndarray) -> bool:
    """Verify if two boards are the same instance."""
    if np.all((board1 == FieldStates.WALL.id) == (board2 == FieldStates.WALL.id)) and np.all(
        ((board1 == FieldStates.BOX_TARGET.id) | (board1 == FieldStates.BOX_ON_TARGET.id) | (board1 == FieldStates.PLAYER_ON_TARGET.id))
        == ((board2 == FieldStates.BOX_TARGET.id) | (board2 == FieldStates.BOX_ON_TARGET.id) | (board2 == FieldStates.PLAYER_ON_TARGET.id))
    ):
        return True
    return False
