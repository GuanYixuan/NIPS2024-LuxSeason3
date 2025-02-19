import sys
import heapq
import numpy as np
from enum import Enum

from utils import _NDArray, Shape
from observation import Observation

from typing import Tuple

class Landscape(Enum):
    """地形类型"""
    UNKNOWN = -1
    EMPTY = 0
    NEBULA = 1
    ASTEROID = 2

class Map:
    """地图类, 处理寻路、障碍移动估计等"""

    MAP_SIZE: int = 24
    obstacle_map: _NDArray[Shape["(MAP_SIZE, MAP_SIZE)"], np.int8]
    energy_map: _NDArray[Shape["(MAP_SIZE, MAP_SIZE)"], np.int8]

    nebula_drift_estimated: bool = False
    nebula_drift_speed: float = -1
    nebula_drift_direction: int = -1

    def __init__(self, obs: Observation) -> None:
        """初始化地图"""
        self.obstacle_map = obs.map_tile_type.copy()

        assert self.MAP_SIZE == obs.map_tile_type.shape[0] == obs.map_tile_type.shape[1]

    def update_map(self, obs: Observation) -> None:
        """更新地图状态"""

        # 更新障碍地图
        last_visible_mask = (self.obstacle_map != Landscape.UNKNOWN.value)
        both_visible_mask = np.logical_and(last_visible_mask, obs.sensor_mask)

        unmoved: bool = np.array_equal(self.obstacle_map[both_visible_mask], obs.map_tile_type[both_visible_mask])
        if unmoved:
            self.obstacle_map = np.maximum(self.obstacle_map, obs.map_tile_type)  # type: ignore
        else:
            # 障碍移动估计
            if not self.nebula_drift_estimated:
                hypo_map = np.roll(self.obstacle_map, (1, -1), (0, 1))
                both_visible_mask = np.logical_and(hypo_map != Landscape.UNKNOWN.value, obs.sensor_mask)
                if np.array_equal(hypo_map[both_visible_mask], obs.map_tile_type[both_visible_mask]):
                    self.nebula_drift_direction = 1
                else:
                    self.nebula_drift_direction = -1
                    # Assert
                    hypo_map = np.roll(self.obstacle_map, (-1, 1), (0, 1))
                    both_visible_mask = np.logical_and(hypo_map != Landscape.UNKNOWN.value, obs.sensor_mask)
                    assert np.array_equal(hypo_map[both_visible_mask], obs.map_tile_type[both_visible_mask])

                self.nebula_drift_speed = 1 / (obs.step - 1)
                if self.nebula_drift_speed > 0.1:
                    self.nebula_drift_speed = 0.15  # 此时移动间隔不是定值
                self.nebula_drift_estimated = True
                print(f"Nebula drift estimated, direction: {self.nebula_drift_direction}, speed: {self.nebula_drift_speed}", file=sys.stderr)
            else:
                rolled_map = np.roll(self.obstacle_map, np.array((1, -1)) * self.nebula_drift_direction, (0, 1))
                self.obstacle_map = np.maximum(rolled_map, obs.map_tile_type)  # type: ignore

        # TODO: 更新能量地图

    def direction_to(self, src: np.ndarray, dst: np.ndarray) -> int:
        """利用A*算法计算从src到dst的下一步方向"""
        if np.array_equal(src, dst):
            return 0

        # 启发函数
        def heuristic(pos: np.ndarray) -> float:
            return np.abs(pos[0] - dst[0]) + np.abs(pos[1] - dst[1])

        # 定义方向数组：中心、上、右、下、左
        DIRECTIONS = np.array([
            [0, 0],   # 中心
            [0, -1],  # 上
            [1, 0],   # 右
            [0, 1],   # 下
            [-1, 0],  # 左
        ])

        # 初始化数组
        closed_array = np.zeros((self.MAP_SIZE, self.MAP_SIZE), dtype=bool)
        g_scores = np.full((self.MAP_SIZE, self.MAP_SIZE), np.inf)
        g_scores[src[0], src[1]] = 0

        start: Tuple[int, int] = tuple(src)
        open_queue = [(heuristic(src), 0, start, start)]  # (f_score, g_score, current, parent)
        heapq.heapify(open_queue)
        came_from = {}

        while open_queue:
            # 获取f_score最小的节点
            f_score, g_score, current, parent = heapq.heappop(open_queue)
            current_pos = np.array(current)

            # 使用数组索引检查节点是否已访问
            if closed_array[current_pos[0], current_pos[1]]:
                continue

            came_from[current] = parent

            if np.array_equal(current_pos, dst):
                # 回溯找到第一步
                while tuple(src) != current:
                    prev = came_from[current]
                    if tuple(src) == prev:
                        # 返回对应的方向索引
                        diff = np.array(current) - src
                        for i, d in enumerate(DIRECTIONS):
                            if np.array_equal(diff, d):
                                return i
                    current = prev
                return 0

            # 标记当前节点为已访问
            closed_array[current_pos[0], current_pos[1]] = True

            # 检查所有可能的移动方向
            for i, d in enumerate(DIRECTIONS):
                neighbor = current_pos + d

                # 检查是否越界
                if (neighbor[0] < 0 or neighbor[0] >= self.MAP_SIZE or
                    neighbor[1] < 0 or neighbor[1] >= self.MAP_SIZE):
                    continue

                # 检查是否是障碍物
                if self.obstacle_map[neighbor[0], neighbor[1]] == Landscape.ASTEROID.value:
                    continue

                # 使用数组索引检查节点是否已访问
                if closed_array[neighbor[0], neighbor[1]]:
                    continue

                tentative_g_score = g_score + 1

                if tentative_g_score < g_scores[neighbor[0], neighbor[1]]:
                    neighbor_tuple = tuple(neighbor)
                    g_scores[neighbor[0], neighbor[1]] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor)
                    heapq.heappush(open_queue, (f_score, tentative_g_score, neighbor_tuple, current))

        # 如果没有找到路径，返回朝向目标的简单方向
        dx = dst[0] - src[0]
        dy = dst[1] - src[1]
        if abs(dx) > abs(dy):
            return 2 if dx > 0 else 4
        else:
            return 3 if dy > 0 else 1
