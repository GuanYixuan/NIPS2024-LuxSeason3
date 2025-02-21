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

class Agent():

    player: str
    opp_player: str
    team_id: int
    opp_team_id: int
    env_cfg: Dict[str, Any]

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

    logger: Logger = Logger()

    def __init__(self, player: str, env_cfg: Dict[str, Any], obs: Observation) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg

        self.relic_center = np.zeros((0, 2), dtype=np.int8)
        self.relic_nodes = np.zeros((0, C.RELIC_SPREAD*2+1, C.RELIC_SPREAD*2+1), dtype=np.int8)

        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_locations = dict()

        self.game_map = Map(obs)
        self.history = []

    def act(self, step: int, obs: Observation, remainingOverageTime: int = 60) -> np.ndarray:
        """implement this function to decide what actions to send to each available unit.

        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """

        self.obs = obs
        self.logger.set_step(step)

        # 更新地图并进行参数估计
        self.game_map.update_map(obs)
        self.estimate_params()

        # 添加新发现的遗迹中心点位置
        # TODO: 新的遗迹范围若与旧的重合, 则重合部分的FAKE应该重新设为UNKNOWN
        for r_center in obs.relic_nodes[obs.relic_nodes_mask]:
            if not np.any(np.all(self.relic_center == r_center, axis=1)):
                self.logger.info(f"New relic center: {r_center}")
                self.relic_center = np.vstack((self.relic_center, r_center, utils.flip_coord(r_center)))  # 对称地添加
                self.relic_nodes = np.vstack((self.relic_nodes,
                                              np.full((2, C.RELIC_SPREAD*2+1, C.RELIC_SPREAD*2+1), RelicInfo.UNKNOWN.value)))

        # 估计遗迹得分点的位置
        relic_updated = self.estimate_relic_positions()

        unit_mask = obs.units_mask[self.team_id]  # shape (max_units, )
        unit_positions = obs.units_position[self.team_id]  # shape (max_units, 2)
        unit_energys = obs.units_energy[self.team_id]  # shape (max_units, 1)
        observed_relic_node_positions = obs.relic_nodes  # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = obs.relic_nodes_mask  # shape (max_relic_nodes, )
        team_points = obs.team_points  # points of each team, team_points[self.team_id] is the points of the your team

        # ids of units you can control at this timestep
        available_unit_ids = np.where(unit_mask)[0]
        # visible relic nodes
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)


        # basic strategy here is simply to have some units randomly explore and some units collecting as much energy as possible
        # and once a relic node is found, we send all units to move randomly around the first relic node to gain points
        # and information about where relic nodes are found are saved for the next match

        # save any new relic nodes that we discover for the rest of the game.
        for id in visible_relic_node_ids:
            if id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(id)
                self.relic_node_positions.append(observed_relic_node_positions[id])


        # unit ids range from 0 to max_units - 1
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            unit_energy = unit_energys[unit_id]

            energy_weight = self.energy_weight_fn(unit_energy, self.game_map.move_cost)

            if len(self.relic_node_positions) > 0:
                nearest_relic_node_position = self.relic_node_positions[0]
                manhattan_distance = abs(unit_pos[0] - nearest_relic_node_position[0]) + abs(unit_pos[1] - nearest_relic_node_position[1])

                # if close to the relic node we want to hover around it and hope to gain points
                if manhattan_distance <= 4:
                    random_direction = np.random.randint(0, 5)
                    actions[unit_id] = [random_direction, 0, 0]
                else:
                    # otherwise we want to move towards the relic node
                    actions[unit_id] = [self.game_map.direction_to(unit_pos, nearest_relic_node_position, energy_weight), 0, 0]
            else:
                # randomly explore by picking a random location on the map and moving there for about 20 steps
                if step % 20 == 0 or unit_id not in self.unit_explore_locations:
                    rand_loc = (np.random.randint(0, self.env_cfg["map_width"]), np.random.randint(0, self.env_cfg["map_height"]))
                    self.unit_explore_locations[unit_id] = rand_loc
                actions[unit_id] = [self.game_map.direction_to(unit_pos, self.unit_explore_locations[unit_id], energy_weight), 0, 0]

        # save the current observation for the next step
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
            relic_dist = np.max(np.abs(self.relic_center - u_pos), axis=1)
            relic_idx = int(np.argmin(relic_dist))  # 有多个最近中心点时, 数据记录在下标最小处
            if relic_dist[relic_idx] > C.RELIC_SPREAD:
                continue

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

    @staticmethod
    def energy_weight_fn(energy: int, move_cost: int) -> float:
        steps = energy // move_cost

        if energy < 100 or steps < 20:
            return 0.15
        elif energy < 250:
            return 0.1
        elif energy < 350:
            return 0.075
        else:
            return 0.03
