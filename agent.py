import numpy as np
from enum import Enum

import utils
from logger import Logger
from map import Map, Landscape
from utils import Constants as C
from observation import Observation

from typing import Set, Dict, List, Tuple, Any

class RelicInfo(Enum):
    """遗迹得分点推断情况"""
    UNKNOWN = -1
    FAKE = 0
    REAL = 1

class UnitTaskType(Enum):
    """单位任务类型"""
    DEAD = -1
    NOT_ALLOCATED = 0
    CAPTURE_RELIC = 1
    """在指定的遗迹得分点上保持不动得分"""
    INVESTIGATE = 2
    """协助推断尚未确定的遗迹得分点"""
    EXPLORE = 3
    """探索未知的遗迹中心点"""
    ATTACK = 4
    DEFEND = 5

class UnitTask:
    """单位任务"""

    type: UnitTaskType
    target_pos: np.ndarray
    start_step: int
    data: Dict[str, Any]

    def __init__(self, type: UnitTaskType, target_pos: np.ndarray, start_step: int) -> None:
        self.type = type
        self.target_pos = target_pos
        self.start_step = start_step
        self.data = {}

    def clear(self) -> None:
        self.type = UnitTaskType.NOT_ALLOCATED
        self.target_pos = np.zeros(2, dtype=np.int8)
        self.start_step = 0
        self.data.clear()

class Agent():

    player: str
    opp_player: str
    team_id: int
    opp_team_id: int
    env_cfg: Dict[str, Any]

    sap_cost: int
    """攻击开销"""
    sap_range: int
    """攻击范围"""


    base_pos: np.ndarray
    """基地位置"""

    game_map: Map
    """游戏地图"""
    relic_center: np.ndarray
    """可见的遗迹中心点位置, 形状(N, 2)"""
    relic_nodes: np.ndarray
    """遗迹得分点推断情况, 数值依照`RelicInfo`枚举设置, 注意平移了RELIC_SPREAD距离
       形状(N, RELIC_SPREAD*2+1, RELIC_SPREAD*2+1)"""
    relic_map: np.ndarray
    """遗迹得分点地图, 在`relic_nodes`更新后更新, 形状(MAP_SIZE, MAP_SIZE)"""

    history: List[Observation]
    """历史观测结果, 保存至**上一个回合**"""
    obs: Observation
    """当前观测结果"""

    task_list: List[UnitTask]
    """单位任务列表"""

    logger: Logger = Logger()

    def __init__(self, player: str, env_cfg: Dict[str, Any], obs: Observation) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.base_pos = np.full(2, 0 if self.team_id == 0 else C.MAP_SIZE-1, dtype=np.int8)

        np.random.seed(0)
        self.env_cfg = env_cfg
        self.sap_cost = self.env_cfg["unit_sap_cost"]
        self.sap_range = self.env_cfg["unit_sap_range"]

        self.relic_center = np.zeros((0, 2), dtype=np.int8)
        self.relic_nodes = np.zeros((0, C.RELIC_SPREAD*2+1, C.RELIC_SPREAD*2+1), dtype=np.int8)

        self.task_list = [UnitTask(UnitTaskType.DEAD, np.zeros(2, dtype=np.int8), 0) for _ in range(C.MAX_UNITS)]
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_locations = dict()

        self.game_map = Map(obs)
        self.history = []

    def act(self, step: int, obs: Observation, remainingOverageTime: int = 60) -> np.ndarray:
        """step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1"""

        self.obs = obs
        self.logger.set_step(step)

        # 更新地图并进行参数估计
        self.game_map.update_map(obs)
        self.estimate_params()

        if step == 0:
            return np.zeros((C.MAX_UNITS, 3), dtype=int)

        # 添加新发现的遗迹中心点位置
        self.update_relic_center()
        # 估计遗迹得分点位置
        relic_updated = self.estimate_relic_positions()
        self.update_relic_map()
        if relic_updated:
            self.logger.info(f"Map updated: \n{self.relic_map.T}")

        # 根据到基地距离排序各得分点
        MAX_POINT_DIST = C.MAP_SIZE + 10
        real_points = np.vstack(np.where(self.relic_map == RelicInfo.REAL.value)).T  # shape (N, 2)
        dists = np.sum(np.abs(real_points - self.base_pos), axis=1)
        real_points = real_points[dists <= MAX_POINT_DIST]
        real_points = real_points[np.argsort(dists[dists <= MAX_POINT_DIST])]  # 按到基地距离排序

        # 根据到基地距离排序各未知点
        unknown_points = np.vstack(np.where(self.relic_map == RelicInfo.UNKNOWN.value)).T  # shape (N, 2)
        dists = np.sum(np.abs(unknown_points - self.base_pos), axis=1)
        unknown_points = unknown_points[dists <= C.MAP_SIZE]  # 只需要排查我方半区的未知点
        unknown_points = unknown_points[np.argsort(dists[dists <= C.MAP_SIZE])]  # 按到基地距离排序

        # 主要策略
        # 0. 处理单位的出现与消失
        for uid in range(C.MAX_UNITS):
            task = self.task_list[uid]
            u_selector = (self.team_id, uid)
            if not obs.units_mask[u_selector] and task.type != UnitTaskType.DEAD:  # 死亡单位不做任何动作
                self.logger.info(f"Unit {uid} is dead")
                self.task_list[uid] = UnitTask(UnitTaskType.DEAD, np.zeros(2, dtype=np.int8), step)
                continue
            elif obs.units_mask[u_selector] and self.task_list[uid].type == UnitTaskType.DEAD:  # 新出现的单位等待重新分配任务
                self.task_list[uid] = UnitTask(UnitTaskType.NOT_ALLOCATED, np.zeros(2, dtype=np.int8), step)

        # 1. 从各个遗迹得分点出发, 寻找最近的空闲单位并分配为CAPTURE_RELIC任务
        actions = np.zeros((C.MAX_UNITS, 3), dtype=int)
        allocated_real_positions: Set[Tuple[int, int]] = \
            set([(t.target_pos[0], t.target_pos[1]) for t in self.task_list if t.type == UnitTaskType.CAPTURE_RELIC])
        allocated_unknown_positions: Set[Tuple[int, int]] = \
            set([(t.target_pos[0], t.target_pos[1]) for t in self.task_list if t.type == UnitTaskType.INVESTIGATE])
        for real_pos in real_points:
            pos_tuple = (real_pos[0], real_pos[1])
            if pos_tuple in allocated_real_positions:
                continue

            dists = np.sum(np.abs(obs.my_units_pos - real_pos), axis=1)
            free_mask = np.array([t.type in (UnitTaskType.NOT_ALLOCATED, UnitTaskType.INVESTIGATE, UnitTaskType.EXPLORE) for t in self.task_list])
            if not np.any(free_mask):
                break  # 所有单位都有更高优先级任务

            dists[~free_mask] = 10000
            closest_uid = np.argmin(dists)
            self.task_list[closest_uid] = UnitTask(UnitTaskType.CAPTURE_RELIC, real_pos, step)
            self.logger.info(f"Unit {closest_uid} -> relic {real_pos}")

        # 2. 处理未知遗迹点, 寻找距离最近的空闲单位并分配为INVESTIGATE任务
        for unknown_pos in unknown_points:
            pos_tuple = (unknown_pos[0], unknown_pos[1])
            if pos_tuple in allocated_unknown_positions:
                continue

            dists = np.sum(np.abs(obs.my_units_pos - unknown_pos), axis=1)
            free_mask = np.array([t.type == UnitTaskType.NOT_ALLOCATED for t in self.task_list])
            if not np.any(free_mask):
                break  # 所有单位都有更高优先级任务

            dists[~free_mask] = 10000
            closest_uid = np.argmin(dists)
            self.task_list[closest_uid] = UnitTask(UnitTaskType.INVESTIGATE, unknown_pos, step)
            self.logger.info(f"Unit {closest_uid} -> unknown {unknown_pos}")

        # 3. 最后未分配的单位执行EXPLORE任务
        for uid in range(C.MAX_UNITS):
            if self.task_list[uid].type == UnitTaskType.NOT_ALLOCATED:
                t_pos = np.array([np.random.randint(0, C.MAP_SIZE), np.random.randint(0, C.MAP_SIZE)])
                if np.sum(np.abs(t_pos - self.base_pos)) > C.MAP_SIZE:
                    t_pos = utils.flip_coord(t_pos)  # 确保探索点在我方半区

                self.task_list[uid] = UnitTask(UnitTaskType.EXPLORE, t_pos, step)
                self.logger.info(f"Unit {uid} -> explore {t_pos}")

        # -1. 各单位执行各自的任务
        for uid in range(C.MAX_UNITS):
            task = self.task_list[uid]
            u_selector = (self.team_id, uid)
            u_pos = obs.units_position[u_selector]
            u_energy = obs.units_energy[u_selector]
            energy_weight = self.energy_weight_fn(u_energy, self.game_map.move_cost)

            # CAPTURE_RELIC任务: 直接走向对应目标
            if task.type == UnitTaskType.CAPTURE_RELIC:
                actions[uid] = [self.game_map.direction_to(u_pos, task.target_pos, energy_weight), 0, 0]

            # INVESTIGATE任务: 在未知点上来回走动
            elif task.type == UnitTaskType.INVESTIGATE:
                first_arrival: int = task.data.get("first_arrival", 10000)
                if np.array_equal(u_pos, task.target_pos):
                    actions[uid] = [np.random.randint(0, 5), 0, 0]
                    task.data["first_arrival"] = min(first_arrival, step)
                else:
                    actions[uid] = [self.game_map.direction_to(u_pos, task.target_pos, energy_weight), 0, 0]

                if step > first_arrival + 20:  # 任务自动结束
                    self.logger.info(f"Unit {uid} ends INVESTIGATE task")
                    task.clear()

            # EXPLORE任务: 移动到指定点
            elif task.type == UnitTaskType.EXPLORE:
                if np.array_equal(u_pos, task.target_pos):
                    task.clear()
                else:
                    actions[uid] = [self.game_map.direction_to(u_pos, task.target_pos, energy_weight), 0, 0]

        # 保存历史观测结果
        self.history.append(obs)
        return actions

    def estimate_params(self) -> None:
        """估计移动成本、星云能量损耗等参数"""
        if not len(self.history):
            return

        gmap = self.game_map
        last_obs = self.history[-1]
        # 估计移动开销
        if not gmap.move_cost_estimated:
            # 任意一个单位移动了, 但没有移动到星云上
            for i in range(C.MAX_UNITS):
                unit_sel = (self.team_id, i)
                if last_obs.units_mask[unit_sel] and self.obs.units_mask[unit_sel]:
                    last_pos = last_obs.units_position[unit_sel]
                    curr_pos = self.obs.units_position[unit_sel]
                    if not np.array_equal(last_pos, curr_pos) and gmap.obstacle_map[tuple(curr_pos)] != Landscape.NEBULA.value:
                        gmap.move_cost = last_obs.units_energy[unit_sel] - self.obs.units_energy[unit_sel] \
                            + gmap.energy_map[tuple(curr_pos)]
                        gmap.move_cost_estimated = True
                        self.logger.info(f"Move cost estimated: {gmap.move_cost}")

        # 估计移动开销后, 估计星云能量损耗
        elif not gmap.nebula_cost_estimated:
            # 任意一个单位移动到了星云上
            for i in range(C.MAX_UNITS):
                unit_sel = (self.team_id, i)
                if last_obs.units_mask[unit_sel] and self.obs.units_mask[unit_sel]:
                    last_pos = last_obs.units_position[unit_sel]
                    curr_pos = self.obs.units_position[unit_sel]
                    if not np.array_equal(last_pos, curr_pos) and gmap.obstacle_map[tuple(curr_pos)] == Landscape.NEBULA.value:
                        gmap.nebula_cost = last_obs.units_energy[unit_sel] - self.obs.units_energy[unit_sel] \
                            + gmap.energy_map[tuple(curr_pos)] - gmap.move_cost
                        gmap.nebula_cost_estimated = True
                        self.logger.info(f"Nebula cost estimated: {gmap.nebula_cost}")

    def update_relic_center(self) -> None:
        """添加新发现的遗迹中心点位置
           """
        obs = self.obs
        for r_center in obs.relic_nodes[obs.relic_nodes_mask]:
            if not np.any(np.all(self.relic_center == r_center, axis=1)):
                self.logger.info(f"New relic center: {r_center}")
                # 新的遗迹范围若与旧的重合, 则重合部分的FAKE应该重新设为UNKNOWN
                # 预计出现次数不多, 故使用原始实现
                for old_idx, old_center in enumerate(self.relic_center):
                    for dx in range(-C.RELIC_SPREAD, C.RELIC_SPREAD+1):
                        for dy in range(-C.RELIC_SPREAD, C.RELIC_SPREAD+1):
                            pos = old_center + np.array([dx, dy])
                            if utils.in_map(pos) and np.sum(np.abs(pos - r_center)) <= C.RELIC_SPREAD:
                                if self.relic_nodes[old_idx, dx+C.RELIC_SPREAD, dy+C.RELIC_SPREAD] == RelicInfo.FAKE.value:
                                    self.relic_nodes[old_idx, dx+C.RELIC_SPREAD, dy+C.RELIC_SPREAD] = RelicInfo.UNKNOWN.value

                # 添加新的遗迹中心点
                self.relic_center = np.vstack((self.relic_center, r_center, utils.flip_coord(r_center)))  # 对称地添加
                self.relic_nodes = np.vstack((self.relic_nodes,
                                              np.full((2, C.RELIC_SPREAD*2+1, C.RELIC_SPREAD*2+1), RelicInfo.UNKNOWN.value)))

    def estimate_relic_positions(self) -> bool:
        """估计遗迹得分点的位置, 返回是否有更新"""
        if not len(self.history) or not len(self.relic_center):
            return False

        delta_pts = self.obs.team_points[self.team_id] - self.history[-1].team_points[self.team_id]
        if self.obs.match_steps == 0:
            return False

        accounted_delta: int = 0
        processed_pos: Set[Tuple[int, int]] = set()
        inferring_unit_info: List[Tuple[int, int, int]] = []  # (relic_idx, r_pos_x, r_pos_y)
        for u_pos in self.obs.units_position[self.team_id, self.obs.units_mask[self.team_id]]:
            # 去除重复坐标
            if (u_pos[0], u_pos[1]) in processed_pos:
                continue
            processed_pos.add((u_pos[0], u_pos[1]))

            # 确定unit周围的遗迹中心点
            relic_dist_valid = np.max(np.abs(self.relic_center - u_pos), axis=1) <= C.RELIC_SPREAD
            if not np.any(relic_dist_valid):
                continue
            relic_idx = int(np.argmax(relic_dist_valid))  # 有多个中心点时, 数据记录在下标最小处

            # 在TRUE点上的unit记录accounted_delta
            relative_pos = u_pos - self.relic_center[relic_idx] + C.RELIC_SPREAD
            grid_info = self.relic_nodes[relic_idx, relative_pos[0], relative_pos[1]]
            if grid_info == RelicInfo.REAL.value:
                accounted_delta += 1
            elif grid_info == RelicInfo.UNKNOWN.value:
                inferring_unit_info.append((relic_idx, relative_pos[0], relative_pos[1]))

        if not len(inferring_unit_info):
            return False

        # 对inferring_unit_info中的点进行推断
        result = RelicInfo.UNKNOWN.value
        if delta_pts - accounted_delta == 0:
            # 所有REAL点都已确定, 其他位置的unit都推断为FAKE
            result = RelicInfo.FAKE.value
        elif delta_pts - accounted_delta == len(inferring_unit_info):
            # 所有unit都在REAL点上
            result = RelicInfo.REAL.value

        # 标记推断结果
        if result == RelicInfo.UNKNOWN.value:
            return False
        for relic_idx, r_pos_x, r_pos_y in inferring_unit_info:
            self.relic_nodes[relic_idx, r_pos_x, r_pos_y] = result
            flipped_pos = utils.flip_coord((r_pos_x, r_pos_y), size=C.RELIC_SPREAD*2+1)
            self.relic_nodes[relic_idx ^ 1, flipped_pos[0], flipped_pos[1]] = result  # 对称地标记

        self.logger.info(f"Relic nodes: \n{self.relic_nodes[0].T}")
        return True

    def update_relic_map(self) -> None:
        """更新遗迹得分点地图"""
        self.relic_map = np.zeros((C.MAP_SIZE, C.MAP_SIZE), dtype=np.int8)
        for idx in range(self.relic_center.shape[0] - 1, -1, -1):
            r_pos = self.relic_center[idx]
            big_map_x_slice = slice(max(0, r_pos[0] - C.RELIC_SPREAD), min(C.MAP_SIZE, r_pos[0] + C.RELIC_SPREAD + 1))
            big_map_y_slice = slice(max(0, r_pos[1] - C.RELIC_SPREAD), min(C.MAP_SIZE, r_pos[1] + C.RELIC_SPREAD + 1))
            small_map_x_slice = slice(max(0, C.RELIC_SPREAD - r_pos[0]),
                                      C.RELIC_SPREAD*2+1 - max(0, r_pos[0] + C.RELIC_SPREAD + 1 - C.MAP_SIZE))
            small_map_y_slice = slice(max(0, C.RELIC_SPREAD - r_pos[1]),
                                      C.RELIC_SPREAD*2+1 - max(0, r_pos[1] + C.RELIC_SPREAD + 1 - C.MAP_SIZE))
            self.relic_map[big_map_x_slice, big_map_y_slice] = self.relic_nodes[idx, small_map_x_slice, small_map_y_slice]

        # 对称化地图
        self.relic_map = np.maximum(self.relic_map, utils.flip_matrix(self.relic_map))

    @staticmethod
    def energy_weight_fn(energy: int, move_cost: int) -> float:
        steps = energy // move_cost

        if energy < 100 or steps < 20:
            return 0.2
        elif energy < 250:
            return 0.15
        elif energy < 350:
            return 0.10
        else:
            return 0.05
