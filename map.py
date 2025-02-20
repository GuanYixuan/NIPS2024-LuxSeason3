import sys
import heapq
import numpy as np
from enum import Enum

from utils import Constants as C
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

    obstacle_map: _NDArray[Shape["(MAP_SIZE, MAP_SIZE)"], np.int8]
    """障碍地图, 数值依照Landscape枚举设置"""
    energy_map: _NDArray[Shape["(MAP_SIZE, MAP_SIZE)"], np.int8]
    """能量地图, 未知值设为0"""

    move_cost: int = 5
    """移动代价, 默认取最大可能值"""
    move_cost_estimated: bool = False

    nebula_cost: int = 5
    """星云能量损耗, 默认取一个较大的可能值"""
    nebula_cost_estimated: bool = False

    nebula_drift_estimated: bool = False
    nebula_drift_speed: float = -1
    nebula_drift_direction: int = -1

    def __init__(self, obs: Observation) -> None:
        """初始化地图"""
        self.obstacle_map = obs.map_tile_type.copy()
        self.energy_map = obs.map_energy.copy()

        assert C.MAP_SIZE == obs.map_tile_type.shape[0] == obs.map_tile_type.shape[1]

    def update_map(self, obs: Observation) -> None:
        """更新地图状态"""

        # 更新障碍地图
        last_visible_mask = (self.obstacle_map != Landscape.UNKNOWN.value)
        both_visible_mask = np.logical_and(last_visible_mask, obs.sym_sensor_mask)

        unmoved: bool = np.array_equal(self.obstacle_map[both_visible_mask], obs.map_tile_type[both_visible_mask])
        if unmoved:
            self.obstacle_map = np.maximum(self.obstacle_map, obs.map_tile_type)  # type: ignore
        else:
            # 障碍移动估计
            if not self.nebula_drift_estimated:
                hypo_map = np.roll(self.obstacle_map, (1, -1), (0, 1))
                both_visible_mask = np.logical_and(hypo_map != Landscape.UNKNOWN.value, obs.sym_sensor_mask)
                if np.array_equal(hypo_map[both_visible_mask], obs.map_tile_type[both_visible_mask]):
                    self.nebula_drift_direction = 1
                else:
                    self.nebula_drift_direction = -1
                    # Assert
                    hypo_map = np.roll(self.obstacle_map, (-1, 1), (0, 1))
                    both_visible_mask = np.logical_and(hypo_map != Landscape.UNKNOWN.value, obs.sym_sensor_mask)
                    assert np.array_equal(hypo_map[both_visible_mask], obs.map_tile_type[both_visible_mask])

                self.nebula_drift_speed = 1 / (obs.step - 1)
                if self.nebula_drift_speed > 0.1:
                    self.nebula_drift_speed = 0.15  # 此时移动间隔不是定值
                self.nebula_drift_estimated = True
                print(f"Nebula drift estimated, direction: {self.nebula_drift_direction}, speed: {self.nebula_drift_speed}", file=sys.stderr)
            else:
                rolled_map = np.roll(self.obstacle_map, np.array((1, -1)) * self.nebula_drift_direction, (0, 1))
                self.obstacle_map = np.maximum(rolled_map, obs.map_tile_type)  # type: ignore

        # 更新能量地图
        last_visible_mask = (self.energy_map != C.DEFAULT_ENERGY_VALUE)
        both_visible_mask = np.logical_and(last_visible_mask, obs.sym_sensor_mask)
        unmoved = np.array_equal(self.energy_map[both_visible_mask], obs.map_energy[both_visible_mask])
        if unmoved:
            self.energy_map[obs.sym_sensor_mask] = obs.map_energy[obs.sym_sensor_mask]
        else:
            # print(f"Energy map updated at step {obs.step}:\n", self.energy_map.T, file=sys.stderr)
            self.energy_map = obs.map_energy.copy()  # 目前直接更新能量地图

    def direction_to(self, src: np.ndarray, dst: np.ndarray, energy_weight: float) -> int:
        """利用A*算法计算从src到dst的下一步方向"""
        if np.array_equal(src, dst):
            return 0

        # 启发函数
        def heuristic(pos: np.ndarray) -> float:
            return np.abs(pos[0] - dst[0]) + np.abs(pos[1] - dst[1])

        # 注意：方向的索引比游戏中的方向值小1，因为去掉了中心方向
        DIRECTIONS = np.array([
            [0, -1],  # 上  (游戏方向值1)
            [1, 0],   # 右  (游戏方向值2)
            [0, 1],   # 下  (游戏方向值3)
            [-1, 0],  # 左  (游戏方向值4)
        ])

        # 初始化数组
        closed_array = np.zeros((C.MAP_SIZE, C.MAP_SIZE), dtype=bool)
        g_scores = np.full((C.MAP_SIZE, C.MAP_SIZE), np.inf)
        g_scores[src[0], src[1]] = 0

        # 使用-1初始化came_from数组，表示未访问
        came_from = np.full((C.MAP_SIZE, C.MAP_SIZE, 2), -1, dtype=np.int16)
        came_from[src[0], src[1]] = src  # 起点的父节点是自己

        start: Tuple[int, int] = tuple(src)
        open_queue = [(heuristic(src), 0.0, start)]  # (f_score, g_score, current)
        heapq.heapify(open_queue)

        while open_queue:
            # 获取f_score最小的节点
            f_score, g_score, current = heapq.heappop(open_queue)
            current_pos = np.array(current)

            # 使用数组索引检查节点是否已访问
            if closed_array[current_pos[0], current_pos[1]]:
                continue

            if np.array_equal(current_pos, dst):
                # 回溯找到第一步
                current_pos = dst
                while not np.array_equal(current_pos, src):
                    parent_pos = came_from[current_pos[0], current_pos[1]]
                    if np.array_equal(parent_pos, src):
                        # 返回对应的方向索引
                        diff = current_pos - src
                        for i, d in enumerate(DIRECTIONS):
                            if np.array_equal(diff, d):
                                return i + 1  # 加1转换为游戏中的方向值
                    current_pos = parent_pos
                return 0

            # 标记当前节点为已访问
            closed_array[current_pos[0], current_pos[1]] = True

            # 检查所有可能的移动方向
            for i, d in enumerate(DIRECTIONS):
                neighbor = current_pos + d
                next_pos = tuple(neighbor)

                # 检查是否越界
                if (neighbor[0] < 0 or neighbor[0] >= C.MAP_SIZE or
                    neighbor[1] < 0 or neighbor[1] >= C.MAP_SIZE):
                    continue

                # 检查是否是障碍物
                landscape = self.obstacle_map[next_pos]
                if landscape == Landscape.ASTEROID.value:
                    continue

                # 使用数组索引检查节点是否已访问
                if closed_array[next_pos]:
                    continue

                move_cost = 1 + (self.nebula_cost if landscape == Landscape.NEBULA.value else 0) / self.move_cost
                tentative_g_score: float = g_score + max(0, move_cost - self.energy_map[next_pos] * energy_weight)  # type: ignore

                if tentative_g_score < g_scores[next_pos]:
                    g_scores[next_pos] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor) * 0.3  # heuristic量纲为1
                    came_from[next_pos] = current_pos
                    heapq.heappush(open_queue, (f_score, tentative_g_score, next_pos))

        # 如果没有找到路径，返回朝向目标的简单方向
        dx = dst[0] - src[0]
        dy = dst[1] - src[1]
        if abs(dx) > abs(dy):
            return 2 if dx > 0 else 4
        else:
            return 3 if dy > 0 else 1
