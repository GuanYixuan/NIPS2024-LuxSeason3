import numpy as np
from enum import Enum
from scipy.signal import convolve2d

import utils
from logger import Logger
from map import Map, Landscape
from utils import Constants as C
from observation import Observation

from typing import Set, Dict, List, Tuple, Optional, Any

class RelicInfo(Enum):
    """遗迹得分点推断情况"""
    UNKNOWN = -1
    FAKE = 0
    REAL = 1

class UnitTaskType(Enum):
    """单位任务类型"""
    DEAD = -1
    IDLE = 0
    """就近寻找能量高点等待"""
    CAPTURE_RELIC = 1
    """在指定的遗迹得分点上保持不动得分"""
    INVESTIGATE = 2
    """协助推断尚未确定的遗迹得分点"""
    EXPLORE = 3
    """探索未知的遗迹中心点"""
    ATTACK = 4
    """激进地尝试占领指定遗迹得分点"""
    DEFEND = 5

class UnitTask:
    """单位任务"""

    type: UnitTaskType
    target_pos: np.ndarray
    start_step: int
    priority: float
    data: Dict[str, Any]

    def __init__(self, type: UnitTaskType, target_pos: np.ndarray, start_step: int, priority: float = 0.0) -> None:
        self.type = type
        self.target_pos = target_pos
        self.start_step = start_step
        self.priority = priority
        self.data = {}

    def __repr__(self) -> str:
        return f"({self.type.name}, {self.target_pos})"

    def __str__(self) -> str:
        return f"({self.type.name}, {self.target_pos})"

    def clear(self) -> None:
        self.type = UnitTaskType.IDLE
        self.target_pos = np.zeros(2, dtype=np.int8)
        self.start_step = 0
        self.priority = 0.0
        self.data.clear()

class SapOrder:
    """攻击需求项"""

    target_pos: np.ndarray
    """目标位置"""
    priority: float
    """目标优先级"""
    need_hit_count: float
    """需要的命中数"""
    fulfilled_count: float
    """已分配的命中数"""

    def __init__(self, target_pos: np.ndarray, priority: float, need_hit_count: float) -> None:
        self.target_pos = target_pos
        self.priority = priority
        self.need_hit_count = need_hit_count
        self.fulfilled_count = 0

    def __lt__(self, other: "SapOrder") -> bool:
        if self.priority == other.priority:
            return self.need_hit_count > other.need_hit_count
        return self.priority < other.priority

    def __repr__(self) -> str:
        return "(pri=%.2f, %s, %d hits)" % (self.priority, self.target_pos, self.need_hit_count)

    def __str__(self) -> str:
        return f"({self.priority}, {self.target_pos}, {self.need_hit_count} hits)"

    def satisfied(self) -> bool:
        return self.fulfilled_count >= self.need_hit_count

class Agent():

    player: str
    opp_player: str
    team_id: int
    opp_team_id: int
    env_cfg: Dict[str, Any]

    sap_cost: int
    """攻击开销, 也等于直接命中时的伤害"""
    sap_range: int
    """攻击范围"""
    sap_dropoff: float = 0.25
    """攻击偏离时的伤害倍率"""
    sap_dropoff_estimated: bool = False

    sensor_range: int = 1
    """感知范围"""
    sensor_range_estimated: bool = False

    base_pos: np.ndarray
    """基地位置"""
    frontline_indicator: float
    """前线位置指示器"""
    capture_weight: float
    """CAPTURE_RELIC任务的权重"""
    full_attack: bool
    """全面进攻模式"""
    danger_map: np.ndarray
    """危险地图, 形状(MAP_SIZE, MAP_SIZE)"""

    game_map: Map
    """游戏地图"""
    map_moved: Tuple[bool, bool]
    """这一回合内obstacle_map和energy_map是否发生移动"""
    relic_center: np.ndarray
    """可见的遗迹中心点位置, 形状(N, 2)"""
    relic_nodes: np.ndarray
    """遗迹得分点推断情况, 数值依照`RelicInfo`枚举设置, 注意平移了RELIC_SPREAD距离
       形状(N, RELIC_SPREAD*2+1, RELIC_SPREAD*2+1)"""
    relic_map: np.ndarray
    """遗迹得分点地图, 在`relic_nodes`更新后更新, 形状(MAP_SIZE, MAP_SIZE)"""
    last_relic_observed: int = -1
    """最后一次发现新遗迹中心点的时间步"""
    explore_map: np.ndarray
    """探索地图, 其中值表示对应格有多少回合未出现在视野内"""
    unclustering_cost: np.ndarray
    """为避免聚集给寻路添加的额外开销"""

    history: List[Observation]
    """历史观测结果, 保存至**上一个回合**"""
    last_action: np.ndarray
    """上一个动作"""
    obs: Observation
    """当前观测结果"""
    opp_menory: np.ndarray
    """敌方单位的历史观测结果, 形状(MAX_UNITS, 5), 第二维表示(x, y, energy, latency, used_count)"""

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
        self.explore_map = np.zeros((C.MAP_SIZE, C.MAP_SIZE), dtype=np.int16)

        self.task_list = [UnitTask(UnitTaskType.DEAD, np.zeros(2, dtype=np.int8), 0) for _ in range(C.MAX_UNITS)]
        self.sap_orders = []

        self.game_map = Map(obs)
        self.history = []

    def act(self, step: int, obs: Observation, remainingOverageTime: int = 60) -> np.ndarray:
        """step: [0, max_steps_in_match * match_count_per_episode)"""

        # -------------------- 初始化项 --------------------
        if obs.is_match_start:
            self.frontline_indicator = C.MAP_SIZE
            self.capture_weight = 1.0
            self.full_attack = False
            self.danger_map = np.zeros((C.MAP_SIZE, C.MAP_SIZE), dtype=np.float32)
            self.opp_menory = np.zeros((C.MAX_UNITS, 5), dtype=np.int16)
            self.hit_intensity_map = np.zeros((C.MAP_SIZE, C.MAP_SIZE), dtype=np.float32)

        # -------------------- 处理观测结果 --------------------
        self.obs = obs
        self.logger.set_step(step)

        # 更新地图并进行参数估计
        self.map_moved = self.game_map.update_map(obs)
        self.estimate_params()

        if step == 0:
            self.history.append(obs)
            return np.zeros((C.MAX_UNITS, 3), dtype=int)

        # 添加新发现的遗迹中心点位置
        self.update_relic_center()
        # 估计遗迹得分点位置
        relic_updated = self.estimate_relic_positions()
        self.update_relic_map()
        # if np.any(self.explore_map > 5):
        #     self.logger.info("Explore map: \n" + str(self.explore_map.T))
        # if relic_updated:
        #     self.logger.info(f"Map updated: \n{self.relic_map.T}")

        # 计算frontline
        FRONTLINE_DECAY: float = 0.5 ** (1 / 20)
        FRT_DEATH_WEIGHT = 0.25
        FRT_OPP_DEATH_WEIGHT = 0.16
        self.frontline_indicator = self.frontline_indicator * FRONTLINE_DECAY + (1 - FRONTLINE_DECAY) * C.MAP_SIZE
        # 根据死亡单位位置更新frontline_indicator
        for uid in range(C.MAX_UNITS):
            task = self.task_list[uid]
            if not obs.my_units_mask[uid] and task.type != UnitTaskType.DEAD and not obs.is_match_start:
                dist = int(utils.l1_dist(self.history[-1].my_units_pos[uid], self.base_pos))
                self.frontline_indicator = self.frontline_indicator * (1 - FRT_DEATH_WEIGHT) + FRT_DEATH_WEIGHT * dist

            if obs.opp_units_mask[uid] and obs.opp_units_energy[uid] < 0:
                dist = int(utils.l1_dist(obs.opp_units_pos[uid], self.base_pos))
                self.frontline_indicator = self.frontline_indicator * (1 - FRT_OPP_DEATH_WEIGHT) + FRT_OPP_DEATH_WEIGHT * dist

        # 计算capture_weight
        CAPTURE_WEIGHT_DECAY: float = 0.5 ** (1 / 20)
        CPT_DEATH_WEIGHT = 0.2
        self.capture_weight = (1 - CAPTURE_WEIGHT_DECAY) * 1.0 + CAPTURE_WEIGHT_DECAY * self.capture_weight
        for uid in range(C.MAX_UNITS):
            task = self.task_list[uid]
            if not obs.my_units_mask[uid] and task.type == UnitTaskType.CAPTURE_RELIC and not obs.is_match_start:
                self.capture_weight *= (1 - CPT_DEATH_WEIGHT)

        self.logger.info("Frontline indicator: %.2f, Capture weight: %.2f" % \
                         (self.frontline_indicator, self.capture_weight))

        # 计算danger_map
        DANGER_MAP_DECAY: float = 0.5 ** (1 / 7)
        DANGER_DEATH_WEIGHT = (3.0, 2.0, 1.0)
        self.danger_map *= DANGER_MAP_DECAY
        for uid in range(C.MAX_UNITS):
            task = self.task_list[uid]
            if not obs.my_units_mask[uid] and task.type != UnitTaskType.DEAD and not obs.is_match_start:
                last_pos = self.history[-1].my_units_pos[uid]
                coord_list = utils.get_coord_list()
                sqr_dists = utils.square_dist(last_pos, coord_list)
                self.danger_map[tuple(coord_list[sqr_dists <= 1.0].T)] += DANGER_DEATH_WEIGHT[0]
                self.danger_map[tuple(coord_list[sqr_dists == 2.0].T)] += DANGER_DEATH_WEIGHT[1]
                self.danger_map[tuple(coord_list[sqr_dists == 3.0].T)] += DANGER_DEATH_WEIGHT[2]

        # 计算unclustering_cost
        self.unclustering_cost = np.zeros((C.MAP_SIZE, C.MAP_SIZE), dtype=np.float32)
        # for uid in range(C.MAX_UNITS):
        #     if not obs.my_units_mask[uid]:
        #         continue
        #     if obs.my_units_energy[uid] < 0:
        #         continue

        #     u_pos = obs.my_units_pos[uid]
        #     bounds = np.array([u_pos - 1, u_pos + 1])
        #     bounds = np.clip(bounds, 0, C.MAP_SIZE-1)
        #     self.unclustering_cost[bounds[0, 0]:bounds[1, 0]+1, bounds[0, 1]:bounds[1, 1]+1] += 0.1
        # self.logger.info(f"Unclustering cost: \n{self.unclustering_cost.T}")

        # 更新opp_menory
        self.opp_menory[:, 3] += 1
        for uid in range(C.MAX_UNITS):
            if not obs.opp_units_mask[uid]:
                continue
            if obs.opp_units_energy[uid] < 0:
                self.opp_menory[uid] = (-1, -1, -1, 0, 0)
            else:
                self.opp_menory[uid] = (obs.opp_units_pos[uid, 0], obs.opp_units_pos[uid, 1], obs.opp_units_energy[uid], 0, 0)

        # 更新full_attack
        if obs.match_steps >= 60 and obs.team_points[self.team_id] < obs.team_points[self.opp_team_id]:
            # defend任务随机分配到敌方基地
            if not self.full_attack:
                self.full_attack = True
                for uid in range(C.MAX_UNITS):
                    if task.type == UnitTaskType.DEFEND:
                        task.clear()

                self.__try_allocate_defend(99)
                self.logger.info(f"Enter full attack mode")

        # -------------------- 任务分配预处理 --------------------

        # 根据到基地距离排序各未知点
        INVEST_ENG_WEIGHT = 1
        unknown_points = np.vstack(np.where(
            (self.relic_map == RelicInfo.UNKNOWN.value) & (self.game_map.obstacle_map != Landscape.ASTEROID.value)
        )).T  # shape (N, 2)
        dists = np.sum(np.abs(unknown_points - self.base_pos), axis=1)
        argsort = np.argsort(dists - INVEST_ENG_WEIGHT * self.game_map.energy_map[unknown_points[:, 0], unknown_points[:, 1]])
        dists = dists[argsort]
        unknown_points = unknown_points[argsort]  # 按到基地距离和能量排序
        unknown_points = unknown_points[dists <= C.MAP_SIZE]  # 只需要排查我方半区的未知点

        self.generate_sap_order()

        # 选出值得探索的目标点
        K_SIZE = 5
        explore_values: np.ndarray = convolve2d(self.explore_map, np.ones((K_SIZE, K_SIZE)), mode='same')
        coords = np.meshgrid(np.arange(C.MAP_SIZE), np.arange(C.MAP_SIZE))
        explore_points = np.column_stack((coords[0].flatten(), coords[1].flatten(), explore_values.flatten()))  # shape (N, 3)
        explore_points = explore_points[np.sum(np.abs(explore_points[:, :2] - self.base_pos), axis=1) < C.MAP_SIZE]
        explore_points = explore_points[
            self.game_map.obstacle_map[tuple(explore_points[:, :2].astype(int).T)] != Landscape.ASTEROID.value
        ]
        explore_pri = explore_points[:, 2] + 0.01 * np.sum(np.abs(explore_points[:, :2] - self.base_pos), axis=1)
        explore_points = explore_points[np.argsort(-explore_pri)]  # 按优先级排序

        # 根据sensor_range去重
        next_idx: int = 1
        while next_idx < explore_points.shape[0]:
            dup_mask = np.any(np.abs(explore_points[next_idx:, :2] - explore_points[next_idx-1, :2]) <= self.sensor_range, axis=1)
            explore_points = np.concatenate((explore_points[:next_idx], explore_points[next_idx:][~dup_mask]))
            next_idx += 1

        self.generate_watch_points()

        # -------------------- 任务分配 --------------------

        # 主要策略
        # 0. 处理单位的出现与消失
        explore_disabled = np.all(self.explore_map == 0)
        for uid in range(C.MAX_UNITS):
            task = self.task_list[uid]
            u_selector = (self.team_id, uid)
            if not obs.units_mask[u_selector] and task.type != UnitTaskType.DEAD:  # 死亡单位不做任何动作
                self.logger.info(f"U{uid} is dead")
                self.task_list[uid] = UnitTask(UnitTaskType.DEAD, np.zeros(2, dtype=np.int8), step)
                continue
            elif obs.units_mask[u_selector] and self.task_list[uid].type == UnitTaskType.DEAD:  # 新出现的单位等待重新分配任务
                self.task_list[uid].clear()

            if explore_disabled and self.task_list[uid].type == UnitTaskType.EXPLORE:  # 探索已完成时, 中断当前的探索任务
                self.task_list[uid].clear()

        allocated_unknown_positions: Set[Tuple[int, int]] = \
            set([(t.target_pos[0], t.target_pos[1]) for t in self.task_list if t.type == UnitTaskType.INVESTIGATE])
        allocated_explore_positions: Set[Tuple[int, int]] = \
            set([(t.target_pos[0], t.target_pos[1]) for t in self.task_list if t.type == UnitTaskType.EXPLORE])

        actions = np.zeros((C.MAX_UNITS, 3), dtype=int)
        self.task_swapped = np.zeros(C.MAX_UNITS, dtype=bool)

        # 0 高能量单位进攻
        relic_center_opp = self.relic_center[utils.l1_dist(self.relic_center, self.base_pos) > C.MAP_SIZE]
        if relic_center_opp.shape[0] > 0:
            for uid in range(C.MAX_UNITS):
                if self.task_list[uid].type == UnitTaskType.ATTACK or self.task_list[uid].type == UnitTaskType.CAPTURE_RELIC:
                    continue
                if obs.my_units_energy[uid] < 250:
                    continue

                tgt = relic_center_opp[np.random.choice(relic_center_opp.shape[0])]
                self.task_list[uid] = UnitTask(UnitTaskType.ATTACK, tgt, step, 2)  # 优先级较高

        # 1. 处理未知遗迹点, 寻找距离最近的空闲单位并分配为INVESTIGATE任务
        for unknown_pos in unknown_points:
            if len(allocated_unknown_positions) >= 4:  # 限制INVESTIGATE数量，避免难以推断
                break
            pos_tuple = (unknown_pos[0], unknown_pos[1])
            if pos_tuple in allocated_unknown_positions:
                continue

            dists = np.sum(np.abs(obs.my_units_pos - unknown_pos), axis=1)
            free_mask = np.array([t.type == UnitTaskType.IDLE for t in self.task_list])
            if not np.any(free_mask):
                break  # 所有单位都有更高优先级任务

            dists[~free_mask] = 10000
            closest_uid = np.argmin(dists)
            allocated_unknown_positions.add(pos_tuple)
            self.task_list[closest_uid] = UnitTask(UnitTaskType.INVESTIGATE, unknown_pos, step)

        # 2. 交替分配DEFEND和CAPTURE_RELIC任务
        prob_list = np.array([self.capture_weight, 1.0])
        prob_list /= np.sum(prob_list)
        while True:
            choice = np.random.choice(len(prob_list), p=prob_list)
            try_alloc = [self.__try_allocate_relic, self.__try_allocate_defend][choice]
            success = try_alloc(1)
            if not success:
                prob_list[choice] = 0.0
                if np.sum(prob_list) == 0:
                    break
                prob_list /= np.sum(prob_list)

        # 3. 未分配的单位执行EXPLORE任务
        MIN_PRIO = 250 if self.last_relic_observed >= 0 else 40
        for uid in range(C.MAX_UNITS):
            if len(allocated_explore_positions) >= 5 and len(self.relic_center):  # 限制EXPLORE数量
                break

            if self.task_list[uid].type != UnitTaskType.IDLE:
                continue

            for i in range(explore_points.shape[0]):
                t_pos = explore_points[i, :2].copy().astype(int)
                prio = float(explore_points[i, 2])
                if prio < MIN_PRIO:
                    break  # 优先级太低的点不考虑
                if (int(t_pos[0]), int(t_pos[1])) in allocated_explore_positions:
                    continue

                np.delete(explore_points, i, axis=0)
                self.task_list[uid] = UnitTask(UnitTaskType.EXPLORE, t_pos, step)
                break

        # 4. 空闲单位走向较远的遗迹得分点, 不再去重
        self.__try_allocate_attack(99)

        # 进行任务交换：相邻的两个单位, 能量较高者执行距离基地更远的任务
        for u1 in range(C.MAX_UNITS):
            u1_task = self.task_list[u1]
            u1_pos = obs.my_units_pos[u1]
            u1_energy = obs.my_units_energy[u1]
            if u1_task.type == UnitTaskType.DEAD or self.task_swapped[u1]:
                continue

            for u2 in range(u1+1, C.MAX_UNITS):
                u2_task = self.task_list[u2]
                u2_pos = obs.my_units_pos[u2]
                u2_energy = obs.my_units_energy[u2]
                if u2_task.type == UnitTaskType.DEAD or self.task_swapped[u2]:
                    continue

                if abs(u1_energy - u2_energy) <= 40:
                    continue  # 能量差距过小
                if utils.l1_dist(u1_pos, u2_pos) > 2:
                    continue

                # 防止交换后无法到达指定地点
                u1_on_spot = (u1_task.type == UnitTaskType.CAPTURE_RELIC and np.array_equal(u1_task.target_pos, u1_pos))
                u2_on_spot = (u2_task.type == UnitTaskType.CAPTURE_RELIC and np.array_equal(u2_task.target_pos, u2_pos))
                if u1_on_spot and (self.game_map.obstacle_map[tuple(u1_pos)] == Landscape.ASTEROID.value):
                    continue
                if u2_on_spot and (self.game_map.obstacle_map[tuple(u2_pos)] == Landscape.ASTEROID.value):
                    continue
                if (u1_on_spot or u2_on_spot) and (utils.l1_dist(u1_pos, u2_pos) > 1):
                    continue  # 避免干扰生产

                delta_tgt_dist = utils.l1_dist(self.base_pos, u1_task.target_pos) - \
                                 utils.l1_dist(self.base_pos, u2_task.target_pos)
                if abs(delta_tgt_dist) >= 3 and np.sign(delta_tgt_dist) != np.sign(u1_energy - u2_energy):
                    swp = u1_task, u2_task
                    self.task_list[u2], self.task_list[u1] = swp
                    self.task_swapped[u1] = True
                    self.task_swapped[u2] = True
                    self.logger.info(f"Task swap: {swp[0]} <-> {swp[1]} at {u1_pos}")
                    break

        # ------------------ 各单位执行各自的任务 --------------------
        MIN_SAP_PRIORITY: Dict[UnitTaskType, float] = {
            UnitTaskType.CAPTURE_RELIC: 6.5,
            UnitTaskType.INVESTIGATE: 7.0,
            UnitTaskType.DEFEND: 5.5,
            UnitTaskType.ATTACK: 15.0,
            UnitTaskType.EXPLORE: 6.0,
            UnitTaskType.IDLE: 5.0,
        }
        for uid in range(C.MAX_UNITS):
            task = self.task_list[uid]
            u_selector = (self.team_id, uid)
            u_pos = obs.units_position[u_selector]
            u_energy = obs.units_energy[u_selector]
            if task.type == UnitTaskType.DEAD:
                continue

            self.logger.info(f"U{uid} ({u_energy}) -> {task.type.name} {task.target_pos} {task.data}")

            # 若相邻格有比自己能量低的敌方单位, 则直接走向敌方
            action_decided = False
            curr_pos_units_mask = np.all(obs.my_units_pos == u_pos, axis=1) & obs.my_units_mask
            curr_pos_energy = np.sum(obs.my_units_energy[curr_pos_units_mask])
            for i, delta in enumerate(C.DIRECTIONS):
                enemy_pos = u_pos + delta
                enemy_mask = np.all(obs.opp_units_pos == enemy_pos, axis=1) & obs.opp_units_mask
                if not np.any(enemy_mask):
                    continue
                total_energy = np.sum(obs.opp_units_energy[enemy_mask])
                if total_energy < curr_pos_energy - np.count_nonzero(curr_pos_units_mask) * self.game_map.move_cost:
                    actions[uid] = [i+1, 0, 0]
                    self.logger.info(f"U{uid} -> crash {enemy_pos}")
                else:
                    actions[uid] = [(i^2)+1, 0, 0]
                    self.logger.info(f"U{uid} -> avoid {enemy_pos}")
                action_decided = True
                break
            if action_decided:
                continue

            # 判断是否进行Sap攻击
            if u_energy >= self.sap_cost + 10:
                min_prio = MIN_SAP_PRIORITY[task.type] * float(np.interp(u_energy,
                           [self.sap_cost + 10, 1.5*self.sap_cost, 4*self.sap_cost, 8*self.sap_cost],
                           [1.7, 1.0, 1.0, 0.8]
                ))
                saps_in_range_mask = np.array([
                    utils.square_dist(u_pos - sap.target_pos) <= self.sap_range and not sap.satisfied()
                    for sap in self.sap_orders
                ])
                if np.any(saps_in_range_mask):
                    selected_sap = np.argmax(saps_in_range_mask)
                    sap_priority = self.sap_orders[selected_sap].priority
                    sap_target = self.sap_orders[selected_sap].target_pos
                    if sap_priority >= min_prio:
                        actions[uid] = [5, sap_target[0]-u_pos[0], sap_target[1]-u_pos[1]]

                        # 更新Sap需求状态
                        for sap in self.sap_orders:
                            sq_dist = utils.square_dist(sap_target, sap.target_pos)
                            if sq_dist == 0:
                                sap.fulfilled_count += 1
                            elif sq_dist == 1:
                                sap.fulfilled_count += self.sap_dropoff
                        self.logger.info(f"U{uid} -> sap {sap_target}")
                        continue

            # CAPTURE_RELIC任务: 直接走向对应目标
            if task.type == UnitTaskType.CAPTURE_RELIC:
                if not np.array_equal(u_pos, task.target_pos) and \
                   self.game_map.obstacle_map[tuple(task.target_pos)] == Landscape.ASTEROID.value:
                    task.clear()
                actions[uid] = [self.__find_path_for(uid, task.target_pos, False), 0, 0]

            # INVESTIGATE任务: 在未知点上来回走动
            elif task.type == UnitTaskType.INVESTIGATE:
                first_arrival: int = task.data.get("first_arrival", 10000)
                if np.array_equal(u_pos, task.target_pos):
                    actions[uid] = [np.random.randint(0, 5), 0, 0]
                    task.data["first_arrival"] = min(first_arrival, step)
                elif self.game_map.obstacle_map[tuple(task.target_pos)] == Landscape.ASTEROID.value:
                    task.clear()
                else:
                    actions[uid] = [self.__find_path_for(uid, task.target_pos), 0, 0]

                if self.relic_map[tuple(task.target_pos)] != RelicInfo.UNKNOWN.value:
                    self.logger.info(f"U{uid} complete INVESTIGATE")
                    task.clear()
                if step > first_arrival + 20:  # 任务自动结束
                    self.logger.info(f"U{uid} ends INVESTIGATE")
                    task.clear()

            # EXPLORE任务: 移动到指定点
            elif task.type == UnitTaskType.EXPLORE:
                if np.array_equal(u_pos, task.target_pos):
                    task.clear()
                elif self.game_map.obstacle_map[tuple(task.target_pos)] == Landscape.ASTEROID.value:
                    task.clear()
                else:
                    actions[uid] = [self.__find_path_for(uid, task.target_pos), 0, 0]

            # ATTACK任务
            elif task.type == UnitTaskType.ATTACK:
                actions[uid] = self.__conduct_attack(uid)

            # DEFEND任务: 移动到指定点并防御
            elif task.type == UnitTaskType.DEFEND:
                best_target = None
                max_eng = -1
                for eid in range(C.MAX_UNITS):
                    if not obs.opp_units_mask[eid]:
                        continue
                    e_pos = obs.opp_units_pos[eid]
                    e_energy = obs.opp_units_energy[eid]
                    if utils.l1_dist(u_pos, e_pos) <= 5 and u_energy > e_energy + 30:
                        if e_energy > max_eng:
                            max_eng = e_energy
                            best_target = e_pos
                if best_target is not None:
                    actions[uid] = [self.__find_path_for(uid, best_target), 0, 0]
                    self.logger.info(f"U{uid} -> chasing {best_target}")

                if np.array_equal(u_pos, task.target_pos):
                    # TODO: 随机移动
                    pass
                else:
                    actions[uid] = [self.__find_path_for(uid, task.target_pos), 0, 0]

                if np.all(utils.square_dist(task.target_pos, self.watch_points[:, :2]) > 1):
                    self.logger.info(f"U{uid}'s watch {task.target_pos} invalid")
                    task.clear()
                    continue
                if self.game_map.full_energy_map[tuple(task.target_pos)] <= 3:
                    self.logger.info(f"U{uid}'s watch {task.target_pos} invalid")
                    task.clear()
                    continue

            # IDLE任务: 在周围10格寻找能量较高点并移动过去
            elif task.type == UnitTaskType.IDLE:
                MAX_IDLE_DIST = 10
                curr_eng = self.game_map.full_energy_map[tuple(u_pos)]
                high_eng_mask = self.game_map.full_energy_map >= curr_eng + 3
                high_eng_pts = np.column_stack(np.where(high_eng_mask) + (self.game_map.full_energy_map[high_eng_mask].flatten(),))
                dists = np.sum(np.abs(high_eng_pts[:, :2] - u_pos), axis=1)
                high_eng_pts = high_eng_pts[dists <= MAX_IDLE_DIST]
                dists = dists[dists <= MAX_IDLE_DIST]

                if high_eng_pts.shape[0]:
                    target_pos = high_eng_pts[np.argmax(high_eng_pts[:, 2])][:2]
                    actions[uid] = [self.__find_path_for(uid, target_pos), 0, 0]

        # 保存历史观测结果
        self.history.append(obs)
        return actions

    def estimate_params(self) -> None:
        """估计移动成本、星云能量损耗等参数"""

        # 估计sensor_range
        if not self.sensor_range_estimated:
            visible_pos = np.vstack(np.where(self.obs.sensor_mask)).T
            if visible_pos.shape[0]:
                self.sensor_range = max(self.sensor_range, np.max(np.abs(visible_pos - self.base_pos)))
                self.logger.info(f"Sensor range estimated: {self.sensor_range}")
                # TODO: 假如所有视野都被星云遮挡?
                self.sensor_range_estimated = True

        if not len(self.history):
            return

        obs = self.obs
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
            if not self.map_moved[0]:  # 本回合内地图自身移动不能算
                # 任意一个单位移动到了星云上
                for i in range(C.MAX_UNITS):
                    unit_sel = (self.team_id, i)
                    if last_obs.units_mask[unit_sel] and self.obs.units_mask[unit_sel]:
                        last_pos = last_obs.units_position[unit_sel]
                        curr_pos = self.obs.units_position[unit_sel]
                        if not np.array_equal(last_pos, curr_pos) and \
                           gmap.obstacle_map[tuple(curr_pos)] == Landscape.NEBULA.value:
                            gmap.nebula_cost = last_obs.units_energy[unit_sel] - self.obs.units_energy[unit_sel] \
                                + gmap.energy_map[tuple(curr_pos)] - gmap.move_cost
                            gmap.nebula_cost_estimated = True
                            self.logger.info(f"Nebula cost estimated: {gmap.nebula_cost}")

        # 此后估计sap_dropoff
        elif not self.sap_dropoff_estimated:
            # 提取上回合我方单位发动的sap
            sap_mask = self.last_action[:, 0] == 5
            sap_pos = last_obs.my_units_pos[sap_mask] + self.last_action[sap_mask, 1:]
            # 提取两回合都能看见的敌方单位
            enemy_mask = last_obs.units_mask[self.opp_team_id] & obs.units_mask[self.opp_team_id]
            # 提取我方可见单位
            available_unit_pos = obs.my_units_pos[obs.units_mask[self.team_id]]  # 稍微放宽一点, 不加上能量>0的限制

            # 推测sap_dropoff
            if sap_pos.shape[0] and np.any(enemy_mask):
                self.logger.info(f"sap pos: {sap_pos}")

                energy_deltas = obs.opp_units_energy[enemy_mask] - last_obs.opp_units_energy[enemy_mask]
                enemy_moved = np.any(last_obs.opp_units_pos[enemy_mask] != obs.opp_units_pos[enemy_mask], axis=1)
                for e_pos, e_moved, e_delta in zip(obs.opp_units_pos[enemy_mask], enemy_moved, energy_deltas):  # type: ignore
                    # 若对方附近有我方单位, 则涉及unit_energy_void_factor参数, 故跳过
                    if np.any(np.sum(np.abs(e_pos - available_unit_pos), axis=1) <= 1):
                        continue

                    # 考虑各种已知的能量变化因素
                    accounted_delta = gmap.energy_map[tuple(e_pos)]  # 地形能量
                    if gmap.obstacle_map[tuple(e_pos)] == Landscape.NEBULA.value:  # 星云能量损耗
                        accounted_delta -= gmap.nebula_cost
                    if e_moved:  # 移动消耗
                        accounted_delta -= gmap.move_cost

                    self.logger.info(f"Sap dropoff inference: {e_pos} {e_delta} {e_moved}, accounted_delta: {accounted_delta}")

                    # 考虑我方sap的影响
                    indirect_hits = 0
                    for s_pos in sap_pos:
                        if np.array_equal(s_pos, e_pos):  # 直接命中
                            accounted_delta -= self.sap_cost
                        elif np.max(np.abs(s_pos - e_pos)) <= 1:
                            indirect_hits += 1
                    if indirect_hits == 0:  # 没有间接命中, 无法推测
                        continue

                    dropoff_estimate = (e_delta - accounted_delta) / (-self.sap_cost) / indirect_hits
                    self.logger.info(f"Sap dropoff estimated: {dropoff_estimate} from {indirect_hits} hits")

                    # 取最近值更新sap_dropoff
                    error = np.abs(C.POSSIBLE_SAP_DROPOFF - dropoff_estimate)
                    if np.min(error) > 0.05:  # 误差过大, 跳过
                        continue

                    self.sap_dropoff = C.POSSIBLE_SAP_DROPOFF[np.argmin(error)]
                    self.sap_dropoff_estimated = True
                    break

        # TODO: 估计unit_energy_void_factor

    def update_relic_center(self) -> None:
        """添加新发现的遗迹中心点位置, 并更新explore_map"""
        obs = self.obs

        # 检查是否有新的遗迹中心点
        for r_center in obs.relic_nodes[obs.relic_nodes_mask]:
            if not np.any(np.all(self.relic_center == r_center, axis=1)):
                # 发现新的遗迹中心点
                self.last_relic_observed = obs.step
                self.explore_map = np.zeros((C.MAP_SIZE, C.MAP_SIZE), dtype=np.int16)
                self.logger.info(f"New relic center: {r_center}")
                # 新的遗迹范围若与旧的重合, 则重合部分的FAKE应该重新设为UNKNOWN
                # 预计出现次数不多, 故使用原始实现
                for new_center in (r_center, utils.flip_coord(r_center)):
                    for old_idx, old_center in enumerate(self.relic_center):
                        for dx in range(-C.RELIC_SPREAD, C.RELIC_SPREAD+1):
                            for dy in range(-C.RELIC_SPREAD, C.RELIC_SPREAD+1):
                                pos = old_center + np.array([dx, dy])
                                if utils.in_map(pos) and np.max(np.abs(pos - new_center)) <= C.RELIC_SPREAD:
                                    if self.relic_nodes[old_idx, dx+C.RELIC_SPREAD, dy+C.RELIC_SPREAD] == RelicInfo.FAKE.value:
                                        self.relic_nodes[old_idx, dx+C.RELIC_SPREAD, dy+C.RELIC_SPREAD] = RelicInfo.UNKNOWN.value

                # 添加新的遗迹中心点
                self.relic_center = np.vstack((self.relic_center, r_center, utils.flip_coord(r_center)))  # 对称地添加
                self.relic_nodes = np.vstack((self.relic_nodes,
                                              np.full((2, C.RELIC_SPREAD*2+1, C.RELIC_SPREAD*2+1), RelicInfo.UNKNOWN.value)))

        # 更新explore_map
        if obs.curr_match >= 4:  # 关闭遗迹探索
            self.explore_map = np.zeros((C.MAP_SIZE, C.MAP_SIZE), dtype=np.int16)
            return
        elif self.relic_center.shape[0] >= obs.curr_match * 2:  # 所有遗迹已知
            self.explore_map = np.zeros((C.MAP_SIZE, C.MAP_SIZE), dtype=np.int16)
            return
        # 若上一个match中没有发现遗迹且视野范围较大, 关闭探索
        elif (obs.curr_match - 1 > self.last_relic_observed // (C.MAX_STEPS_IN_MATCH + 1) + 1) and (self.sensor_range > 1) and \
             (self.relic_center.shape[0] > 0):
            self.explore_map = np.zeros((C.MAP_SIZE, C.MAP_SIZE), dtype=np.int16)
            return

        # 本match内还没发现遗迹, 则出现概率累加
        if obs.match_steps <= 50 and self.last_relic_observed < (C.MAX_STEPS_IN_MATCH + 1) * (obs.curr_match - 1):
            self.explore_map += 1

        # 清除视野内的未探索标记
        self.explore_map[self.obs.sym_sensor_mask] = 0

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
        unit_mask = self.obs.units_mask[self.team_id] & (self.obs.my_units_energy >= 0)
        for u_pos in self.obs.units_position[self.team_id, unit_mask]:
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
        return True

    def update_relic_map(self) -> None:
        """更新遗迹得分点地图, 同时计算遗迹视野mask"""
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

    hit_intensity_map: np.ndarray
    """记录地图各个位置的火力密度"""
    sap_orders: List[SapOrder]
    """攻击需求列表"""
    attack_sap_orders: List[SapOrder]
    """专为攻击而设的sap_orders"""
    def generate_sap_order(self) -> None:
        """生成攻击需求列表"""
        obs = self.obs
        self.sap_orders = []
        self.attack_sap_orders = []

        # 更新hit_intensity_map
        HIT_INTENSITY_DECAY = 0.5 ** (1/10)  # 半衰期10回合
        self.hit_intensity_map *= HIT_INTENSITY_DECAY
        # 将上回合发动的Sap计入
        for action in self.last_action:
            if action[0] != 5:  # Sap
                continue
            pos = action[1:]
            # 中心+1, 周围1格内+sap_dropoff倍
            bounds = np.clip(np.array([pos - 1, pos + 1]), 0, C.MAP_SIZE-1)
            self.hit_intensity_map[bounds[0, 0]:bounds[1, 0]+1, bounds[0, 1]:bounds[1, 1]+1] += self.sap_dropoff
            self.hit_intensity_map[pos[0], pos[1]] += (1 - self.sap_dropoff)
        # 若本回合观测到敌方单位, 则其位置的hit_intensity_map值清零
        e_pos = obs.opp_units_pos[obs.opp_units_mask]
        self.hit_intensity_map[e_pos[:, 0], e_pos[:, 1]] = 0

        real_points = np.vstack(np.where(self.relic_map == RelicInfo.REAL.value)).T  # shape (N, 2)
        dists = np.sum(np.abs(real_points - self.base_pos), axis=1)
        real_points_myhalf = real_points[dists <= C.MAP_SIZE]  # 我方半区得分点
        # 针对可见敌人生成SapOrder
        ON_RELIC_WEIGHT = 3.5
        visible_enemy_on_relic: Set[Tuple[int, int]] = set()
        for eid, e_pos, e_energy in zip(range(C.MAX_UNITS), obs.opp_units_pos, obs.opp_units_energy):
            enemy_can_move = (e_energy >= self.game_map.move_cost)
            if not obs.units_mask[self.team_id, eid] or e_energy < 0:
                continue

            priority: float = 1.0

            # 按条件累加优先级
            # 1. 在我方半区得分点附近
            if real_points_myhalf.shape[0] > 0:
                min_dist = np.min(np.abs(real_points_myhalf - e_pos))
                if min_dist <= self.sap_range:
                    priority += min(self.sap_range - min_dist, 3.0) + 2.0

            # 2. 在得分点上
            on_relic = (self.relic_map[tuple(e_pos)] == RelicInfo.REAL.value)
            if on_relic:
                priority += ON_RELIC_WEIGHT
                visible_enemy_on_relic.add((e_pos[0], e_pos[1]))
            elif e_energy == 0:
                continue  # 无能量且不在得分点上的敌人可放置以拖延时间

            # 3. 能量较高
            priority += float(np.interp(e_energy, [100, 150, 250, 300], [0.0, 1.0, 2.0, 2.5]))

            # 打提前量
            if not on_relic and enemy_can_move:
                e_pos += C.ADJACENT_DELTAS[np.argmax(self.__pred_enemy_pos(eid))]

            need_hit_count = int(e_energy / self.sap_cost) + 1
            safe_hit_count = need_hit_count + (0 if (self.sap_dropoff == 1 or not enemy_can_move) else 1)
            self.sap_orders.append(SapOrder(e_pos, priority, min(safe_hit_count, max(1, int(priority / 3.5)), 4)))

        # 针对不在视野内的得分点生成SapOrder
        MAX_LATENCY = 4
        possibility: Optional[float] = None
        unknown_points = np.vstack(np.where(self.relic_map == RelicInfo.UNKNOWN.value)).T
        unknown_points_invisible = unknown_points[obs.sensor_mask[unknown_points[:, 0], unknown_points[:, 1]] == 0]
        real_points_invisible = real_points[obs.sensor_mask[real_points[:, 0], real_points[:, 1]] == 0]
        if real_points_invisible.shape[0] > 0:
            possibility = (obs.team_points[self.opp_team_id] - self.history[-1].team_points[self.opp_team_id])
            assert possibility is not None
            possibility -= len(visible_enemy_on_relic)
            denominator = (real_points_invisible.shape[0] + 0.4 * unknown_points_invisible.shape[0])
            possibility = min(1.0, possibility / denominator)
            self.logger.debug("Possibility of invisible relics: {} / {} = {}".format(
                possibility * denominator, denominator, possibility))

            for r_pos in real_points_invisible:
                # 生成普通SapOrder
                normal_prio = ON_RELIC_WEIGHT * possibility * \
                    float(np.interp(self.hit_intensity_map[r_pos[0], r_pos[1]], [0.5, 1.5], [1.0, 0.0])) * \
                    float(np.interp(self.game_map.energy_map[r_pos[0], r_pos[1]], [-6, 0], [1.5, 1.0]))
                self.sap_orders.append(SapOrder(r_pos, normal_prio, 2))

        for r_pos in real_points[utils.l1_dist(real_points, self.base_pos) <= C.MAP_SIZE]:
            pos_tup = tuple(r_pos)
            visible = obs.sensor_mask[pos_tup]
            # 生成攻击SapOrder
            attack_prio = possibility if (not visible and possibility is not None) else 0.0
            if not visible:
                attack_prio += float(np.interp(self.game_map.energy_map[pos_tup], [-6, 0, 6], [2.0, 0, -2.0]))  # 能量奖励
                attack_prio += float(np.interp(self.hit_intensity_map[pos_tup], [0.3, 2.0], [2.0, 0]))  # 低密度区奖励
            close_enemy = self.opp_menory[
                (self.opp_menory[:, 3] <= MAX_LATENCY) & (utils.square_dist(r_pos, self.opp_menory[:, :2]) <= 1)
            ]
            if visible and not np.any(utils.square_dist(r_pos, obs.opp_units_pos[obs.opp_units_mask]) == 0):
                continue
            vis_enemy_hp = -1
            for e_data in close_enemy:
                e_pos: np.ndarray = e_data[:2]
                e_energy: int = e_data[2]
                e_last_seen: int = e_data[3]
                e_used: int = e_data[4]

                delta_prio = 0.0
                delta_prio += float(np.interp(e_energy,
                    [self.sap_cost, 2 * self.sap_cost, 4 * self.sap_cost],
                    [1.5, 0.7, 0.25]
                ))

                if np.array_equal(e_pos, r_pos):
                    vis_enemy_hp = max(vis_enemy_hp, e_energy)
                else:
                    delta_prio *= self.sap_dropoff

                over_hit = e_used - ((e_energy // self.sap_cost) + 1)
                delta_prio *= float(np.interp(over_hit, [0, 2], [1.0, 0.25]))

                if e_last_seen <= 1:  # 目前可见
                    delta_prio += 2.0

                attack_prio += delta_prio

                # TODO: 在SapOrder中记录used的增量
            if not visible:
                attack_prio *= float(np.interp(self.hit_intensity_map[pos_tup], [2.5, 6.0], [1.0, 0.3]))  # 高密度区惩罚
            self.attack_sap_orders.append(
                SapOrder(r_pos, attack_prio, min(2.0, (vis_enemy_hp / self.sap_cost) if (vis_enemy_hp >= 0) else 100))
            )

        # 累加处于同一位置上的SapOrder的优先级
        self.sap_orders.sort(key=lambda x: (x.target_pos[0], x.target_pos[1], x.priority), reverse=True)  # 位置相同时按优先级降序排列
        curr_idx = 0
        while curr_idx < len(self.sap_orders) - 1:
            curr_order = self.sap_orders[curr_idx]
            if np.array_equal(curr_order.target_pos, self.sap_orders[curr_idx+1].target_pos):
                curr_order.priority += self.sap_orders[curr_idx+1].priority
                curr_order.need_hit_count = max(curr_order.need_hit_count, self.sap_orders[curr_idx+1].need_hit_count)
                self.sap_orders.pop(curr_idx+1)
            else:
                curr_idx += 1

        # 为每个SapOrder在四周创建空的SapOrder，方便后续的聚合
        def __create_empty_order(orders: List[SapOrder]) -> None:
            new_orders = []
            curr_order_pos = np.array([ord.target_pos for ord in orders])
            curr_order_hits = np.array([ord.need_hit_count for ord in orders])
            for order in orders:
                for delta in C.DIRECTIONS:
                    new_pos = order.target_pos + delta
                    if not np.all(new_pos >= 0) or not np.all(new_pos < C.MAP_SIZE):
                        continue
                    if np.any(np.all(new_pos == curr_order_pos, axis=1)):
                        continue
                    adjacent_order_mask = utils.square_dist(new_pos, curr_order_pos) == 1
                    new_orders.append(SapOrder(new_pos, 0.0, np.max(curr_order_hits[adjacent_order_mask])))
            orders.extend(new_orders)
        __create_empty_order(self.sap_orders)
        __create_empty_order(self.attack_sap_orders)

        def __cumulate_priority(orders: List[SapOrder]) -> None:
            """每个SapOrder向相邻的SapOrder加上self.sap_dropoff倍的优先级"""
            delta_priority = np.zeros(len(orders))
            for i in range(len(orders)):
                for j in range(i+1, len(orders)):
                    if utils.square_dist(orders[i].target_pos, orders[j].target_pos) == 1:
                        i_full = orders[i].need_hit_count <= self.sap_dropoff
                        j_full = orders[j].need_hit_count <= self.sap_dropoff
                        delta_priority[i] += (1 if j_full else self.sap_dropoff) * orders[j].priority
                        delta_priority[j] += (1 if i_full else self.sap_dropoff) * orders[i].priority
            for i in range(len(orders)):
                orders[i].priority += delta_priority[i]
        __cumulate_priority(self.sap_orders)
        __cumulate_priority(self.attack_sap_orders)

        self.sap_orders.sort(reverse=True)
        self.attack_sap_orders.sort(reverse=True)
        print_orders = [order for order in self.sap_orders if order.priority >= 3.0]
        print_atk_orders = [order for order in self.attack_sap_orders if order.priority >= 2.0]
        if len(print_orders) > 0: self.logger.info(f"Sap orders: {print_orders}")
        if len(print_atk_orders) > 0: self.logger.info(f"Attack orders: {print_atk_orders}")

    def __pred_enemy_pos(self, eid: int) -> np.ndarray:
        """预测指定id的敌方单位下回合的移动, 下标同C.ADJACENT_DELTAS"""

        curr_pos: np.ndarray = self.obs.opp_units_pos[eid]
        dir_scores = np.zeros(C.ADJACENT_DELTAS.shape[0])
        dir_scores[0] = 0.5  # 默认倾向于不动

        # 移动判据：观察历史2~3步移动方向
        # TODO: 长期站着不动需要特判
        MAX_HISTORY = 4
        has_history = False
        pos_delta = curr_pos.copy()
        for backward_time in range(MAX_HISTORY, 0, -1):
            if backward_time > len(self.history):
                continue
            if self.history[-backward_time].opp_units_mask[eid]:
                has_history = True
                pos_delta -= self.history[-backward_time].opp_units_pos[eid]
                break

        MOVE_CRIT_WEIGHT = max(1.0, 0.3 * backward_time)
        if has_history:
            if np.sum(np.abs(pos_delta)) == 0:
                dir_scores[0] += MOVE_CRIT_WEIGHT
            else:  # 投影到各个方向
                hist_dir = pos_delta / np.linalg.norm(pos_delta)
                dots = np.dot(C.ADJACENT_DELTAS, hist_dir)
                dir_scores += dots * (dots > 0) * MOVE_CRIT_WEIGHT
            self.logger.debug(f"Predict enemy {eid} direction after move_crit: {dir_scores}")

        # 得分点判据: 所有距离较近的我方半区得分点基本在同一半平面内
        MAX_DIST = 7
        real_points_myhalf = np.vstack(np.where(self.relic_map == RelicInfo.REAL.value)).T
        real_points_myhalf = real_points_myhalf[np.sum(np.abs(real_points_myhalf - self.base_pos), axis=1) <= C.MAP_SIZE]
        if real_points_myhalf.shape[0] > 0:
            real_point_deltas = real_points_myhalf - curr_pos
            dists = np.sum(np.abs(real_point_deltas), axis=1)
            real_point_deltas = (real_point_deltas[(dists <= MAX_DIST) & (dists > 0)]).astype(np.float32)
            real_point_deltas /= np.linalg.norm(real_point_deltas, axis=1, keepdims=True)

            best_dir: Optional[np.ndarray] = None
            best_score = 0.0
            for angle in np.linspace(0, 2*np.pi, 20, endpoint=False):
                proj_deltas = np.dot(real_point_deltas, np.array([np.cos(angle), np.sin(angle)]))
                score = np.sum(proj_deltas >= 0) - np.sum(proj_deltas < 0) * 3 + np.mean(proj_deltas) * 0.5
                if score > best_score:
                    best_dir = np.array([np.cos(angle), np.sin(angle)])
                    best_score = score

            RELIC_CRIT_WEIGHT = 1.0
            if best_dir is not None:
                # 投影到各个方向
                dots = np.dot(C.ADJACENT_DELTAS, best_dir)
                dir_scores += dots * (dots > 0) * RELIC_CRIT_WEIGHT
                self.logger.debug(f"Predict enemy {eid} direction after relic_crit: {dir_scores}, best_dir: {best_dir}")

        # TODO: 单位在得分点上

        # 地形调整(不可达格设置为0)
        for i in range(C.ADJACENT_DELTAS.shape[0]):
            pos = curr_pos + C.ADJACENT_DELTAS[i]
            if not utils.in_map(pos):
                dir_scores[i] = 0.0
            elif self.game_map.obstacle_map[tuple(pos)] == Landscape.ASTEROID.value and i != 0:
                dir_scores[i] = 0.0

        # 若不符合任何判据, 返回原点
        if np.sum(dir_scores) == 0:
            return np.array([1.0, 0.0, 0.0, 0.0, 0.0])

        # 处理冲突
        if np.all(dir_scores[1:] > 0):
            dir_scores[1:] -= np.min(dir_scores[1:])
        if np.all(dir_scores[[1, 2]] > 0):
            minv = np.min(dir_scores[[1, 2]])
            dir_scores[[1, 2]] -= minv
            dir_scores[0] += minv
        if np.all(dir_scores[[3, 4]] > 0):
            minv = np.min(dir_scores[[3, 4]])
            dir_scores[[3, 4]] -= minv
            dir_scores[0] += minv

        # 结合能量进行调整
        max_eng_dir = -1
        max_eng_score = -np.inf
        for i in range(1, C.ADJACENT_DELTAS.shape[0]):
            if dir_scores[i] == 0:
                continue

            pos = curr_pos + C.ADJACENT_DELTAS[i]
            eng_score = self.game_map.energy_map[tuple(pos)]
            if self.game_map.obstacle_map[tuple(pos)] == Landscape.NEBULA.value:
                eng_score -= self.game_map.nebula_cost

            if eng_score > max_eng_score:
                max_eng_dir = i
                max_eng_score = eng_score

        if max_eng_dir != -1:
            dir_scores[max_eng_dir] *= 2
            self.logger.debug(f"Predict enemy {eid} direction after energy modifications: {dir_scores}")

        dir_scores /= np.sum(dir_scores)  # 归一化
        self.logger.debug(f"Predict enemy {eid} direction: {dir_scores}")
        return dir_scores

    watch_points: np.ndarray  # 前哨点列表, shape (N, 3)
    def generate_watch_points(self) -> None:
        """生成前哨点列表"""
        self.watch_points = np.zeros((0, 3))

        # 选取我方半区内得分点最多的relic_center作为中心
        relic_center_mask = utils.l1_dist(self.base_pos, self.relic_center) <= C.MAP_SIZE
        relic_center_score = np.sum(self.relic_nodes == 1, axis=(1, 2)) + utils.l1_dist(self.base_pos, self.relic_center)
        relic_center_score[~relic_center_mask] = 0
        if relic_center_mask.shape[0] == 0:
            return
        base_point: np.ndarray = self.relic_center[np.argmax(relic_center_score)]
        self.logger.info(f"Base point: {base_point}")

        # 创建备选点列表
        x, y = np.meshgrid(np.arange(C.MAP_SIZE), np.arange(C.MAP_SIZE))
        watch_points = np.column_stack((x.flatten(), y.flatten()))  # shape (N, 2)

        # 计算角度score
        def angle_score(angle: float) -> float:
            if angle <= np.pi/6:
                return 0.8
            elif angle <= np.pi/3:
                return 1.0
            elif angle <= np.pi/2:
                return 0.8
            return 0.6
        forward_dir = np.full((2,), 1/np.sqrt(2)) * (-1 if self.team_id else 1)
        to_base_dir = (utils.flip_coord(self.base_pos) - base_point).astype(np.float32)
        to_base_dir /= np.linalg.norm(to_base_dir)
        to_base_dir_weight = float(np.interp(utils.l1_dist(self.base_pos, base_point), [0, 20], [0.2, 1.0]))
        forward_dir = forward_dir * (1 - to_base_dir_weight) + to_base_dir * to_base_dir_weight
        self.logger.info(f"Forward direction: {forward_dir}")

        deltas = watch_points - base_point
        delta_dirs = deltas / np.linalg.norm(deltas, axis=1, keepdims=True)
        angles = np.arccos(np.clip(delta_dirs @ forward_dir, -1.0, 1.0))
        watch_points = np.column_stack((watch_points, angles.flatten()))  # shape (N, 3)

        # 根据距离筛选
        dist_thresh = np.zeros(watch_points.shape[0])
        dist_thresh = np.interp(angles,
            [0, np.pi/6, np.pi/2, np.pi],
            [24, C.RELIC_SPREAD + self.sap_range + self.sensor_range * 2, max(self.sap_range, 4.0), C.RELIC_SPREAD]
        )
        watch_points = watch_points[utils.square_dist(deltas) <= dist_thresh]

        # 根据能量筛选
        engval = self.game_map.full_energy_map[tuple(watch_points[:, :2].astype(int).T)]
        watch_points = watch_points[
            (engval == 0) | (engval >= 5)
        ]

        # 筛除与得分点相邻的点
        real_points_myhalf = np.column_stack(np.where(self.relic_map == RelicInfo.REAL.value))
        real_points_myhalf = real_points_myhalf[utils.l1_dist(real_points_myhalf, base_point) <= C.MAP_SIZE]
        if real_points_myhalf.shape[0] > 0:
            distances = np.max(np.abs(watch_points[:, :2].reshape(-1, 1, 2) - real_points_myhalf.reshape(1, -1, 2)), axis=2)
            watch_points = watch_points[np.min(distances, axis=1) > 1]

        # 计算能量score
        ENERGY_WEIGHT = 1.0
        energy_vals = self.game_map.full_energy_map[tuple(watch_points[:, :2].astype(int).T)] * ENERGY_WEIGHT
        final_scores = np.array(list(map(angle_score, watch_points[:, 2]))) * energy_vals

        # 排序并选择前哨点
        watch_points[:, 2] = final_scores
        watch_points = watch_points[np.argsort(-final_scores)]
        next_undedup_idx = 1
        while next_undedup_idx < watch_points.shape[0]:
            dists = np.max(np.abs(
                watch_points[next_undedup_idx:, :2].reshape(-1, 1, 2) - watch_points[:next_undedup_idx, :2].reshape(1, -1, 2)
            ), axis=2)
            valid_mask = np.min(dists, axis=1) > 2
            watch_points = np.vstack((watch_points[:next_undedup_idx], watch_points[next_undedup_idx:][valid_mask]))
            next_undedup_idx += 1

        self.watch_points = watch_points

    def __try_allocate_defend(self, count: int) -> int:
        """尝试分配count个防御任务"""
        obs = self.obs
        allocated_count = 0

        if self.full_attack:
            real_points = np.column_stack(np.where(
                (self.relic_map == RelicInfo.REAL.value) & (self.game_map.obstacle_map != Landscape.ASTEROID.value)
            ))
            real_points = real_points[utils.l1_dist(real_points, self.base_pos) >= C.MAP_SIZE]
            if real_points.shape[0] == 0:
                return allocated_count

            for uid in range(C.MAX_UNITS):
                task = self.task_list[uid]
                if task.type == UnitTaskType.IDLE:
                    self.task_list[uid] = UnitTask(UnitTaskType.DEFEND,
                                                   real_points[np.random.randint(real_points.shape[0])],
                                                   obs.step, 1)
                    allocated_count += 1
                    if allocated_count >= count:
                        return allocated_count
            return allocated_count

        # 根据到基地距离排序前哨点
        PREFERRED_DISTANCE = C.MAP_SIZE if (obs.match_steps <= 30) else self.frontline_indicator - 5
        if self.watch_points.shape[0] > 0:
            dists = np.abs(utils.l1_dist(self.watch_points[:, :2], self.base_pos) - PREFERRED_DISTANCE)
            self.watch_points = self.watch_points[np.argsort(dists)]


        allocated_defend_positions: List[Tuple[int, int]] = \
            list([(t.target_pos[0], t.target_pos[1]) for t in self.task_list if t.type == UnitTaskType.DEFEND])
        for i in range(self.watch_points.shape[0]):
            if len(allocated_defend_positions) >= 12:  # 限制DEFEND数量
                break

            pos = self.watch_points[i, :2].copy().astype(int)
            if len(allocated_defend_positions) and np.any(utils.square_dist(pos, np.array(allocated_defend_positions)) <= 1):
                continue

            dists = np.sum(np.abs(obs.my_units_pos - pos), axis=1)
            free_mask = np.array([t.type == UnitTaskType.IDLE for t in self.task_list])
            if not np.any(free_mask):
                break  # 所有单位都有更高优先级任务

            dists[~free_mask] = 10000
            closest_uid = int(np.argmin(dists))
            self.task_list[closest_uid] = UnitTask(UnitTaskType.DEFEND, pos, obs.step, 1 if (obs.match_steps <= 30) else 0)
            self.__alloc_swap(closest_uid, 10, 75)

            allocated_count += 1
            allocated_defend_positions.append((pos[0], pos[1]))
            if allocated_count >= count:
                return allocated_count

        return allocated_count

    def __try_allocate_relic(self, count: int) -> int:
        """尝试分配count个占领得分点任务"""
        obs = self.obs

        # 根据到基地距离排序各得分点
        PREFERRED_DISTANCE = 20 if (obs.step % (C.MAX_STEPS_IN_MATCH + 1) <= 30) else 0
        MAX_POINT_DIST = self.frontline_indicator + 5
        CAPT_RELIC_ENG_WEIGHT = 0.5
        real_points = np.column_stack(np.where(
            (self.relic_map == RelicInfo.REAL.value) & (self.game_map.obstacle_map != Landscape.ASTEROID.value)
        ))
        real_points = real_points[self.danger_map[tuple(real_points.T)] <= 8.0]

        dists = np.sum(np.abs(real_points - self.base_pos), axis=1)
        near_score = np.where(dists <= 14, 100, 0)
        argsort = np.argsort(np.abs(dists - PREFERRED_DISTANCE) -
                             CAPT_RELIC_ENG_WEIGHT * self.game_map.energy_map[real_points[:, 0], real_points[:, 1]] -
                             near_score)
        dists = dists[argsort]
        real_points = real_points[argsort]  # 按到基地距离和能量排序
        real_points_near = real_points[dists <= MAX_POINT_DIST]

        allocated_count = 0
        allocated_real_positions: Set[Tuple[int, int]] = \
            set([(t.target_pos[0], t.target_pos[1]) for t in self.task_list if t.type == UnitTaskType.CAPTURE_RELIC])
        for real_pos in real_points_near:
            pos_tuple = (real_pos[0], real_pos[1])
            if pos_tuple in allocated_real_positions:
                continue

            dists = np.sum(np.abs(obs.my_units_pos - real_pos), axis=1)
            free_mask = np.array([
                t.type in (UnitTaskType.IDLE, UnitTaskType.DEFEND, UnitTaskType.EXPLORE, UnitTaskType.ATTACK) and
                ((dists[i] <= 8 and self.danger_map[pos_tuple] <= 6.0) or t.type == UnitTaskType.IDLE) and
                t.priority == 0 for i, t in enumerate(self.task_list)])
            if not np.any(free_mask):
                break  # 所有单位都有更高优先级任务

            dists[~free_mask] = 10000
            closest_uid = int(np.argmin(dists))
            if dists[closest_uid] > 8 and self.task_list[closest_uid].type != UnitTaskType.IDLE:
                continue
            self.task_list[closest_uid] = UnitTask(UnitTaskType.CAPTURE_RELIC, real_pos, obs.step)
            if self.danger_map[pos_tuple] <= 3.0:
                self.__alloc_swap(closest_uid, 3, 50)

            allocated_count += 1
            allocated_real_positions.add(pos_tuple)
            if allocated_count >= count:
                return allocated_count

        return allocated_count

    def __try_allocate_attack(self, count: int) -> int:
        """尝试分配count个攻击任务"""
        obs = self.obs
        relic_center_opp = self.relic_center[utils.l1_dist(self.relic_center, self.base_pos) > C.MAP_SIZE]

        allocated_count = 0
        if relic_center_opp.shape[0] > 0:
            for uid in range(C.MAX_UNITS):
                if self.task_list[uid].type not in (UnitTaskType.IDLE, UnitTaskType.DEFEND):
                    continue
                if obs.my_units_energy[uid] < 175:
                    continue

                tgt = relic_center_opp[np.random.choice(relic_center_opp.shape[0])]
                self.task_list[uid] = UnitTask(UnitTaskType.ATTACK, tgt, obs.step)
                self.__alloc_swap(uid, 10, 150)

                allocated_count += 1
                if allocated_count >= count:
                    return allocated_count

        return allocated_count

    def __alloc_swap(self, uid: int, max_dist: int, min_energy: int) -> None:
        """为uid根据能量等进行任务置换"""
        obs = self.obs
        task = self.task_list[uid]
        unit_task_targets = np.array([t.target_pos for t in self.task_list])

        target_front_dist = utils.l1_dist(task.target_pos, self.base_pos)
        potential_target_mask = obs.my_units_mask & (obs.my_units_energy >= min_energy) & (~self.task_swapped)
        potential_target_mask &= np.array([t.type == task.type or t.type == UnitTaskType.IDLE for t in self.task_list])
        potential_target_mask &= utils.l1_dist(unit_task_targets, self.base_pos) < target_front_dist
        potential_target_mask &= (utils.l1_dist(obs.my_units_pos, task.target_pos) <= max_dist)

        if task.type == UnitTaskType.CAPTURE_RELIC:
            potential_target_mask &= (np.any(unit_task_targets != obs.my_units_pos, axis=-1) |
                                      (self.game_map.obstacle_map[tuple(obs.my_units_pos.T)] != Landscape.ASTEROID.value))

        if not np.any(potential_target_mask):
            return

        ids = np.where(potential_target_mask)[0]
        potential_target = ids[np.argmax(obs.my_units_energy[ids])]
        swp = task, self.task_list[potential_target]
        self.task_list[potential_target], self.task_list[uid] = swp
        self.task_swapped[potential_target] = True
        self.logger.info(f"Swap task {uid} with {potential_target}: {swp}")

        self.__alloc_swap(uid, max_dist, min_energy)

    def __conduct_attack(self, uid: int) -> List[int]:
        """执行攻击任务"""
        TARGET_KEY = "specfic_target"

        obs = self.obs
        gmap = self.game_map
        allocated_pos = [t.data[TARGET_KEY] for t in self.task_list if t.type == UnitTaskType.ATTACK and TARGET_KEY in t.data]

        task = self.task_list[uid]
        u_pos = obs.my_units_pos[uid]
        u_energy = obs.my_units_energy[uid]
        rough_target = task.target_pos.copy()

        # 计算需要避开的视野mask
        avoid_mask = np.zeros((C.MAP_SIZE, C.MAP_SIZE), dtype=bool)
        real_points = np.column_stack(np.where(self.relic_map == RelicInfo.REAL.value))
        real_points = real_points[utils.square_dist(real_points, rough_target) <= 4]
        if real_points.shape[0] > 0:
            avoid_points = utils.get_coord_list()
            avoid_points = avoid_points[np.min(
                np.max(np.abs(avoid_points.reshape(-1, 1, 2) - real_points.reshape(1, -1, 2)), axis=-1), axis=-1
            ) <= self.sensor_range]
            avoid_mask[tuple(avoid_points.T)] = True

        # 若未设定具体目标或目标不符合标准, 重新选择目标
        target_reset = False
        if task.data.get(TARGET_KEY, None) is None:
            target_reset = True
        else:
            target_pos: np.ndarray = task.data[TARGET_KEY]
            t_tuple = (target_pos[0], target_pos[1])
            if gmap.obstacle_map[t_tuple] == Landscape.ASTEROID.value: target_reset = True
            elif gmap.energy_map_mask[t_tuple] and gmap.full_energy_map[t_tuple] < 3: target_reset = True
        if target_reset and real_points.shape[0] == 0:
            task.clear()
            return [0, 0, 0]
        elif target_reset and real_points.shape[0] > 0:
            # 选出所有“与real_points中所有点距离大于等于sensor_range, 但至少与一点距离等于sensor_range”的点
            valid_points = utils.get_coord_list()
            dists = np.max(np.abs(valid_points.reshape(-1, 1, 2) - real_points.reshape(1, -1, 2)), axis=-1)
            valid_points = valid_points[np.min(dists, axis=-1) == max(self.sensor_range, self.sap_range)]
            # 与现有点对比去除临近点
            if len(allocated_pos) > 0:
                dists = np.max(np.abs(valid_points.reshape(-1, 1, 2) - np.array(allocated_pos).reshape(1, -1, 2)), axis=-1)
                valid_points = valid_points[np.min(dists, axis=-1) >= 2]
            # 根据能量进行筛选
            MIN_FULL_ENG = 1
            eng_val = gmap.full_energy_map[tuple(valid_points.T)]
            mask = eng_val >= MIN_FULL_ENG
            valid_points, eng_val = valid_points[mask], eng_val[mask]
            # 根据到敌方基地-rough_target线段的距离进行筛选
            MIN_PATH_DIST = 1.0
            dist1 = utils.dist_to_segment(utils.flip_coord(self.base_pos), rough_target, valid_points)
            mask = dist1 >= MIN_PATH_DIST
            valid_points, dist1, eng_val = valid_points[mask], dist1[mask], eng_val[mask]
            # 计算到rough_target与其对称点连线的距离
            dist2 = utils.dist_to_segment(utils.flip_coord(rough_target), rough_target, valid_points)
            # 根据分数排序
            dist_to_upos = utils.l1_dist(valid_points, u_pos)
            scores = eng_val * np.minimum(dist1, dist2) / np.clip(dist_to_upos, 3, 100)
            if len(allocated_pos):
                dists = np.max(np.abs(valid_points.reshape(-1, 1, 2) - np.array(allocated_pos).reshape(1, -1, 2)), axis=-1)
                scores *= np.where(np.min(dists, axis=-1) <= 2, 0.5, 1.0)
            self.logger.info(f"valid_points: {np.column_stack((valid_points, scores, dist1, dist2, eng_val))}")
            valid_points = valid_points[np.argsort(-scores)]
            # 选择最优点作为目标
            if valid_points.shape[0] > 0:
                task.data[TARGET_KEY] = valid_points[0]
            else:
                self.logger.info(f"U{uid} ({obs.my_units_energy[uid]}) has no valid target")
                task.clear()
                return [0, 0, 0]

        specfic_target: np.ndarray = task.data[TARGET_KEY]

        # 若未曾到达目标点附近, 继续行进
        last_arrival = task.data.get("last_arrival", -1)
        if utils.square_dist(u_pos, specfic_target) <= 2:
            last_arrival = obs.step
            task.data["last_arrival"] = last_arrival
        if last_arrival < 0:
            return [
                self.__find_path_for(uid, specfic_target, extra_cost=avoid_mask.astype(float) * 10), 0, 0
            ]

        # 到达过目标点附近, 判断自身是否在得分点附近的敌方视野内
        enemy_in_sight = obs.opp_units_pos[obs.opp_units_mask]
        enemy_in_sight = enemy_in_sight[utils.square_dist(enemy_in_sight, u_pos) <= min(self.sensor_range, self.sap_range)]
        enemy_in_sight_relic = enemy_in_sight[utils.square_dist(enemy_in_sight, rough_target) <= 3]

        # 若是, 尝试后退离开视野
        if enemy_in_sight_relic.shape[0] > 0 and u_energy < 250:
            clostest_enemy = enemy_in_sight_relic[np.argmin(utils.square_dist(enemy_in_sight_relic, u_pos))]
            move_dir = u_pos - clostest_enemy
            normalized_dir = C.DIRECTIONS[np.argmax(C.DIRECTIONS @ move_dir)]
            return [self.__find_path_for(uid, np.clip(u_pos + 2 * normalized_dir, 0, C.MAP_SIZE - 1)), 0, 0]

        # 若否, 考虑攻击目标点、前进一格或保持不动吸收能量
        elif u_energy >= self.sap_cost:
            MIN_PRIO = float(np.interp(u_energy,
                                       [self.sap_cost + 10, self.sap_cost * 2, self.sap_cost * 4, self.sap_cost * 6],
                                       [7.0, 4.0, 2.5, 2.0]))
            self.logger.info(f"U{uid} ({obs.my_units_energy[uid]}) finding sap with priority {MIN_PRIO}")
            for order in self.attack_sap_orders:
                if order.priority < MIN_PRIO:
                    break
                if order.satisfied():
                    continue
                delta_pos = order.target_pos - u_pos
                if utils.square_dist(delta_pos) <= self.sap_range:
                    order.fulfilled_count += 1
                    return [5, delta_pos[0], delta_pos[1]]
        # 能量较低, 保持不动
        else:
            return [0, 0, 0]
        # 否则前进一格
        if self.sap_dropoff < 1 or np.random.rand() < 0.2:
            return [self.__find_path_for(uid, rough_target), 0, 0]
        return [0, 0, 0]

    @staticmethod
    def energy_weight_fn(energy: int, move_cost: int) -> float:
        steps = energy // move_cost

        ret = float(np.interp(energy, [25, 30, 100, 250, 350], [5, 0.3, 0.20, 0.15, 0.05]))
        ret = max(ret, 3.0 if steps <= 6 else 0)
        return ret

    def __find_path_for(self, uid: int, target_pos: np.ndarray, danger_weight: float = 0.4,
                        extra_cost: Optional[np.ndarray] = None) -> int:
        """寻找从当前位置到target_pos的路径, 并返回下一步方向"""
        u_energy = self.obs.my_units_energy[uid]
        u_pos = self.obs.my_units_pos[uid]
        ex_cost = extra_cost if extra_cost is not None else np.zeros((C.MAP_SIZE, C.MAP_SIZE), dtype=np.float32)
        ex_cost += self.danger_map * danger_weight
        return self.game_map.direction_to(u_pos, target_pos,
                                          self.energy_weight_fn(u_energy, self.game_map.move_cost),
                                          extra_cost=ex_cost, collision_info=self.unclustering_cost)
