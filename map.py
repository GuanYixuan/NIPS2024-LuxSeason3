import heapq
import numpy as np
from enum import Enum

import utils
from logger import Logger
from utils import Constants as C
from observation import Observation

from typing import Tuple, Callable, Optional

class Landscape(Enum):
    """地形类型"""
    UNKNOWN = -1
    EMPTY = 0
    NEBULA = 1
    ASTEROID = 2

class Map:
    """地图类, 处理寻路、障碍移动估计等"""

    obstacle_map: np.ndarray
    """障碍地图, 数值依照Landscape枚举设置, 形状为(MAP_SIZE, MAP_SIZE)"""
    energy_map: np.ndarray
    """能量地图, 未知值设为0, 形状为(MAP_SIZE, MAP_SIZE)"""
    energy_map_mask: np.ndarray
    """能量地图掩码, 未知值设为False, 形状与energy_map相同"""
    full_energy_map: np.ndarray
    """完整能量地图, 考虑了障碍的影响, 形状与energy_map相同"""

    move_cost: int = 5
    """移动代价, 默认取最大可能值"""
    move_cost_estimated: bool = False

    nebula_cost: int = 5
    """星云能量损耗, 默认取一个较大的可能值"""
    nebula_cost_estimated: bool = False

    nebula_drift_estimated: bool = False
    nebula_drift_speed: float = -1
    nebula_drift_direction: int = -1

    energy_drift_estimated: bool = False
    energy_drift_speed: float = -1

    logger: Logger = Logger()

    def __init__(self, obs: Observation) -> None:
        """初始化地图"""
        self.obstacle_map = obs.map_tile_type.copy()
        self.energy_map = obs.map_energy.copy()
        self.energy_map_mask = np.zeros_like(self.energy_map, dtype=bool)

        assert C.MAP_SIZE == obs.map_tile_type.shape[0] == obs.map_tile_type.shape[1]

    def update_map(self, obs: Observation) -> Tuple[bool, bool]:
        """更新地图状态, 返回这一回合内obstacle_map和energy_map是否发生变化"""

        # 更新障碍地图
        last_visible_mask = (self.obstacle_map != Landscape.UNKNOWN.value)
        both_visible_mask = np.logical_and(last_visible_mask, obs.sym_sensor_mask)

        obstacle_unmoved: bool = np.array_equal(self.obstacle_map[both_visible_mask], obs.map_tile_type[both_visible_mask])
        if obstacle_unmoved:
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
                self.logger.info("Nebula drift estimated, direction: %d, speed: %.3f" %
                                 (self.nebula_drift_direction, self.nebula_drift_speed))
            else:
                rolled_map = np.roll(self.obstacle_map, np.array((1, -1)) * self.nebula_drift_direction, (0, 1))
                # Assert
                both_visible_mask = np.logical_and(rolled_map != Landscape.UNKNOWN.value, obs.sym_sensor_mask)
                if not np.array_equal(rolled_map[both_visible_mask], obs.map_tile_type[both_visible_mask]):
                    self.logger.warning("Nebula drift estimation failed! Resetting...")
                    self.obstacle_map = obs.map_tile_type.copy()
                    self.nebula_drift_estimated = False
                    self.nebula_drift_speed = -1  # 重置速度（TODO: 虽然第二次估计时这会是错的）
                    self.nebula_drift_direction = -1
                else:
                    self.obstacle_map = np.maximum(rolled_map, obs.map_tile_type)  # type: ignore

        # 更新能量地图
        last_visible_mask = (self.energy_map != C.DEFAULT_ENERGY_VALUE)
        both_visible_mask = np.logical_and(last_visible_mask, obs.sym_sensor_mask)
        energy_unmoved = np.array_equal(self.energy_map[both_visible_mask], obs.map_energy[both_visible_mask])
        if energy_unmoved:
            self.energy_map[obs.sym_sensor_mask] = obs.map_energy[obs.sym_sensor_mask]
            self.energy_map_mask[obs.sym_sensor_mask] = True
        else:
            if not self.energy_drift_estimated and obs.step > 2:
                self.energy_drift_speed = 1 / (obs.step - 2)
                self.energy_drift_estimated = True
                self.logger.info("Energy drift estimated, speed: %.3f" % self.energy_drift_speed)
            self.energy_map = obs.map_energy.copy()  # 目前直接更新能量地图
            self.energy_map_mask = np.zeros_like(self.energy_map, dtype=bool)
            self.energy_map_mask[obs.sym_sensor_mask] = True

        # 更新完整能量地图
        self.full_energy_map = self.energy_map.copy()
        self.full_energy_map[self.obstacle_map == Landscape.ASTEROID.value] = -50
        self.full_energy_map[self.obstacle_map == Landscape.NEBULA.value] -= self.nebula_cost

        return not obstacle_unmoved, not energy_unmoved

    @staticmethod
    def __l1_heuristic(src: np.ndarray, dst: np.ndarray) -> float:
        """L1距离启发函数"""
        return utils.l1_dist(src, dst).sum() * 0.3

    def direction_to(self, src: np.ndarray, dst: np.ndarray, energy_weight: float,
                     _heuristic: Callable[[np.ndarray, np.ndarray], float] = __l1_heuristic,
                     extra_cost: Optional[np.ndarray] = None,
                     collision_info: Optional[np.ndarray] = None) -> int:
        """利用A*算法计算从src到dst的下一步方向"""
        src = np.array(src)
        dst = np.array(dst)
        if np.array_equal(src, dst):
            return 0

        # 如果目标是障碍物, 寻找最近的可达邻格作为新目标
        if self.obstacle_map[tuple(dst)] == Landscape.ASTEROID.value:
            # 找出相邻格
            neighbors = dst + C.DIRECTIONS
            valid_mask = np.all((neighbors >= 0) & (neighbors < C.MAP_SIZE), axis=1)
            valid_neighbors = neighbors[valid_mask]

            # 找出可达邻格
            reachable_mask = np.array([
                self.obstacle_map[tuple(n)] != Landscape.ASTEROID.value
                for n in valid_neighbors
            ])
            # 计算到src的曼哈顿距离
            distances = np.abs(valid_neighbors - src).sum(axis=1)

            if reachable_mask.any():
                # 在可达邻格中选择距离最近的作为新目标
                reachable_neighbors = valid_neighbors[reachable_mask]
                reachable_distances = distances[reachable_mask]

                self.logger.debug(f"Original target {dst} redirected to {reachable_neighbors[np.argmin(reachable_distances)]}")
                dst = reachable_neighbors[np.argmin(reachable_distances)]

        if np.array_equal(src, dst):
            return 0

        heuristic: Callable[[np.ndarray], float] = lambda x: _heuristic(x, dst)

        # 初始化数组
        closed_array = np.zeros((C.MAP_SIZE, C.MAP_SIZE), dtype=bool)
        g_scores = np.full((C.MAP_SIZE, C.MAP_SIZE), np.inf)
        g_scores[src[0], src[1]] = 0

        # 使用-1初始化came_from数组，表示未访问
        came_from = np.full((C.MAP_SIZE, C.MAP_SIZE, 2), -1, dtype=np.int16)
        came_from[src[0], src[1]] = src  # 起点的父节点是自己

        start: Tuple[int, int] = tuple(src)
        # 在队列中添加步数信息：
        open_queue = [(heuristic(src), 0.0, 0, start)]  # (f_score, g_score, steps, current)
        heapq.heapify(open_queue)

        while open_queue:
            # 获取f_score最小的节点
            f_score, g_score, steps, current = heapq.heappop(open_queue)
            current_pos = np.array(current)

            # 使用数组索引检查节点是否已访问
            if closed_array[current_pos[0], current_pos[1]]:
                continue

            if np.array_equal(current_pos, dst):
                # 回溯找到第一步
                while not np.array_equal(current_pos, src):
                    parent_pos = came_from[current_pos[0], current_pos[1]]
                    if np.array_equal(parent_pos, src):
                        # 返回对应的方向索引
                        diff = current_pos - src
                        for i, d in enumerate(C.FULL_DIRECTIONS):
                            if np.array_equal(diff, d):
                                return i
                    current_pos = parent_pos
                return 0

            # 标记当前节点为已访问
            closed_array[current_pos[0], current_pos[1]] = True

            # 检查所有可能的移动方向（此处方向顺序随机）
            for d in C.FULL_DIRECTIONS:
                neighbor = current_pos + d
                next_pos = tuple(neighbor)

                # 检查是否越界
                if np.any(neighbor < 0) or np.any(neighbor >= C.MAP_SIZE):
                    continue

                # 检查是否是障碍物
                landscape = self.obstacle_map[next_pos]
                if landscape == Landscape.ASTEROID.value:
                    continue

                # 使用数组索引检查节点是否已访问
                if closed_array[next_pos]:
                    continue

                # 根据步数决定是否考虑能量图
                energy_factor = self.full_energy_map[next_pos] * energy_weight if steps <= 8 else 0
                tentative_g_score: float = g_score + max(0.1, 1 - energy_factor)  # type: ignore
                if extra_cost is not None:
                    tentative_g_score += extra_cost[next_pos]  # type: ignore
                if collision_info is not None and utils.l1_dist(neighbor, src) == 1:
                    tentative_g_score += float(collision_info[next_pos])

                if tentative_g_score < g_scores[next_pos]:
                    g_scores[next_pos] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor)
                    came_from[next_pos] = current_pos
                    heapq.heappush(open_queue, (f_score, tentative_g_score, steps + 1, next_pos))

        # 如果没有找到路径，返回朝向目标的简单方向
        self.logger.warning(f"No path found from {src} to {dst}")
        dx = dst[0] - src[0]
        dy = dst[1] - src[1]
        if abs(dx) > abs(dy):
            return 2 if dx > 0 else 4
        else:
            return 3 if dy > 0 else 1
