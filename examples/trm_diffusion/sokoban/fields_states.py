from enum import Enum


class FieldStates(Enum):
    WALL = (1, "wall.png")
    FLOOR = (2, "floor.png")
    BOX_TARGET = (3, "box_target.png")
    BOX_ON_TARGET = (4, "box_on_target.png")
    BOX = (5, "box.png")
    PLAYER = (6, "player.png")
    PLAYER_ON_TARGET = (7, "player_on_target.png")

    def __init__(self, id: int, asset_file_name: str):
        self.id = id
        self.asset_file_name = asset_file_name
