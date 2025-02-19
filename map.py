import sys
import numpy as np
from enum import Enum

from utils import _NDArray, Shape
from observation import Observation

class Landscape(Enum):
    """地形类型"""
    UNKNOWN = -1
    EMPTY = 0
    NEBULA = 1
    ASTEROID = 2

class Map:
    """地图类, 处理寻路、障碍移动估计等"""

    obstacle_map: _NDArray[Shape["(map_height, map_width)"], np.int8]

    nebula_drift_estimated: bool = False
    nebula_drift_speed: float = -1
    nebula_drift_direction: int = -1

    def __init__(self, obs: Observation) -> None:
        """初始化地图"""
        self.obstacle_map = obs.map_tile_type.copy()

    def update_map(self, obs: Observation) -> None:
        """更新地图状态"""

        last_visible_mask = (self.obstacle_map != Landscape.UNKNOWN.value)
        current_visible_mask = (obs.map_tile_type != Landscape.UNKNOWN.value)
        both_visible_mask = np.logical_and(last_visible_mask, current_visible_mask)

        unmoved: bool = np.array_equal(self.obstacle_map[both_visible_mask], obs.map_tile_type[both_visible_mask])
        if unmoved:
            self.obstacle_map = np.maximum(self.obstacle_map, obs.map_tile_type)  # type: ignore
            return

        # 障碍移动估计
        if not self.nebula_drift_estimated:
            hypo_map = np.roll(self.obstacle_map, (1, -1), (0, 1))
            both_visible_mask = np.logical_and(hypo_map != Landscape.UNKNOWN.value, current_visible_mask)
            if np.array_equal(hypo_map[both_visible_mask], obs.map_tile_type[both_visible_mask]):
                self.nebula_drift_direction = 1
            else:
                self.nebula_drift_direction = -1
                # Assert
                hypo_map = np.roll(self.obstacle_map, (-1, 1), (0, 1))
                both_visible_mask = np.logical_and(hypo_map != Landscape.UNKNOWN.value, current_visible_mask)
                assert np.array_equal(hypo_map[both_visible_mask], obs.map_tile_type[both_visible_mask])

            self.nebula_drift_speed = 1 / (obs.step - 1)
            if self.nebula_drift_speed > 0.1:
                self.nebula_drift_speed = 0.15  # 此时移动间隔不是定值
            self.nebula_drift_estimated = True
            print(f"Nebula drift estimated, direction: {self.nebula_drift_direction}, speed: {self.nebula_drift_speed}", file=sys.stderr)
        else:
            rolled_map = np.roll(self.obstacle_map, np.array((1, -1)) * self.nebula_drift_direction, (0, 1))
            self.obstacle_map = np.maximum(rolled_map, obs.map_tile_type)  # type: ignore
