from enum import Enum


class FieldStates(Enum):
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
