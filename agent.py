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
    data: Dict[str, Any]

    def __init__(self, type: UnitTaskType, target_pos: np.ndarray, start_step: int) -> None:
        self.type = type
        self.target_pos = target_pos
        self.start_step = start_step
        self.data = {}

    def __repr__(self) -> str:
        return f"UnitTask(type={self.type.name}, target_pos={self.target_pos})"

    def __str__(self) -> str:
        return f"({self.type.name}, {self.target_pos})"

    def clear(self) -> None:
        self.type = UnitTaskType.IDLE
        self.target_pos = np.zeros(2, dtype=np.int8)
        self.start_step = 0
        self.data.clear()

class SapOrder:
    """攻击需求项"""

    target_pos: np.ndarray
    """目标位置"""
    priority: float
    """目标优先级"""
    need_hit_count: int
    """需要的命中数"""
    fulfilled_count: int
    """已分配的命中数"""

    def __init__(self, target_pos: np.ndarray, priority: float, need_hit_count: int) -> None:
        self.target_pos = target_pos
        self.priority = priority
        self.need_hit_count = need_hit_count
        self.fulfilled_count = 0

    def __lt__(self, other: "SapOrder") -> bool:
        if self.priority == other.priority:
            return self.need_hit_count > other.need_hit_count
        return self.priority < other.priority

    def __repr__(self) -> str:
        return f"SapOrder(pri={self.priority}, {self.target_pos}, need {self.need_hit_count} hits)"

    def __str__(self) -> str:
        return f"({self.priority}, {self.target_pos}, need {self.need_hit_count} hits)"

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

    history: List[Observation]
    """历史观测结果, 保存至**上一个回合**"""
    last_action: np.ndarray
    """上一个动作"""
    obs: Observation
    """当前观测结果"""

    task_list: List[UnitTask]
    """单位任务列表"""
    sap_orders: List[SapOrder]
    """攻击需求列表"""

    logger: Logger = Logger()

    def __init__(self, player: str, env_cfg: Dict[str, Any], obs: Observation) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.base_pos = np.full(2, 0 if self.team_id == 0 else C.MAP_SIZE-1, dtype=np.int8)

        # np.random.seed(0)
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
        self.logger.info(f"obstacle_map: \n{self.game_map.obstacle_map.T}")
        # if np.any(self.explore_map > 5):
        #     self.logger.info("Explore map: \n" + str(self.explore_map.T))
        # if relic_updated:
        #     self.logger.info(f"Map updated: \n{self.relic_map.T}")

        # -------------------- 任务分配预处理 --------------------

        # 根据到基地距离排序各得分点
        MAX_POINT_DIST = C.MAP_SIZE + 3
        CAPT_RELIC_ENG_WEIGHT = 0.5
        real_points = np.vstack(np.where(
            (self.relic_map == RelicInfo.REAL.value) & (self.game_map.obstacle_map != Landscape.ASTEROID.value)
        )).T  # shape (N, 2)
        dists = np.sum(np.abs(real_points - self.base_pos), axis=1)
        argsort = np.argsort(dists - CAPT_RELIC_ENG_WEIGHT * self.game_map.energy_map[real_points[:, 0], real_points[:, 1]])
        dists = dists[argsort]
        real_points = real_points[argsort]  # 按到基地距离和能量排序
        real_points_near = real_points[dists <= MAX_POINT_DIST]

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
        # if explore_points.shape[0] > 0 and explore_points[0, 2] > 0:
        #     self.logger.info(f"Explore points: \n{explore_points}")

        # -------------------- 任务分配 --------------------

        # 主要策略
        # 0. 处理单位的出现与消失
        explore_disabled = np.all(self.explore_map == 0)
        for uid in range(C.MAX_UNITS):
            task = self.task_list[uid]
            u_selector = (self.team_id, uid)
            if not obs.units_mask[u_selector] and task.type != UnitTaskType.DEAD:  # 死亡单位不做任何动作
                self.logger.info(f"Unit {uid} is dead")
                self.task_list[uid] = UnitTask(UnitTaskType.DEAD, np.zeros(2, dtype=np.int8), step)
                continue
            elif obs.units_mask[u_selector] and self.task_list[uid].type == UnitTaskType.DEAD:  # 新出现的单位等待重新分配任务
                self.task_list[uid].clear()

            if explore_disabled and self.task_list[uid].type == UnitTaskType.EXPLORE:  # 探索已完成时, 中断当前的探索任务
                self.task_list[uid].clear()

        allocated_real_positions: Set[Tuple[int, int]] = \
            set([(t.target_pos[0], t.target_pos[1]) for t in self.task_list if t.type == UnitTaskType.CAPTURE_RELIC])
        allocated_unknown_positions: Set[Tuple[int, int]] = \
            set([(t.target_pos[0], t.target_pos[1]) for t in self.task_list if t.type == UnitTaskType.INVESTIGATE])
        allocated_explore_positions: Set[Tuple[int, int]] = \
            set([(t.target_pos[0], t.target_pos[1]) for t in self.task_list if t.type == UnitTaskType.EXPLORE])

        # 1. 从各个较近的遗迹得分点出发, 寻找最近的空闲单位并分配为CAPTURE_RELIC任务
        actions = np.zeros((C.MAX_UNITS, 3), dtype=int)
        for real_pos in real_points_near:
            pos_tuple = (real_pos[0], real_pos[1])
            if pos_tuple in allocated_real_positions:
                continue

            dists = np.sum(np.abs(obs.my_units_pos - real_pos), axis=1)
            free_mask = np.array([
                t.type in (UnitTaskType.IDLE, UnitTaskType.INVESTIGATE,
                           UnitTaskType.EXPLORE, UnitTaskType.ATTACK)
                for t in self.task_list])
            if not np.any(free_mask):
                break  # 所有单位都有更高优先级任务

            dists[~free_mask] = 10000
            closest_uid = np.argmin(dists)
            if dists[closest_uid] > 8 and self.task_list[closest_uid].type == UnitTaskType.ATTACK:
                continue
            self.task_list[closest_uid] = UnitTask(UnitTaskType.CAPTURE_RELIC, real_pos, step)
            self.logger.info(f"Unit {closest_uid} -> relic {real_pos}")

        # 2. 处理未知遗迹点, 寻找距离最近的空闲单位并分配为INVESTIGATE任务
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
            self.logger.info(f"Unit {closest_uid} -> unknown {unknown_pos}")

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
                self.logger.info(f"Unit {uid} -> explore {t_pos}")
                break

        # 4. 空闲单位走向较远的遗迹得分点, 不再去重
        real_points = np.column_stack(np.where(self.relic_map == RelicInfo.REAL.value))
        real_points_far = real_points[np.sum(np.abs(real_points - self.base_pos), axis=1) >= C.MAP_SIZE]
        if real_points_far.shape[0] > 0:
            for uid in range(C.MAX_UNITS):
                if self.task_list[uid].type != UnitTaskType.IDLE:
                    continue

                tgt = real_points_far[np.random.choice(real_points_far.shape[0])]
                self.task_list[uid] = UnitTask(UnitTaskType.ATTACK, tgt, step)
                self.logger.info(f"Unit {uid} -> relic far {tgt}")

        # 输出统计信息
        # self.logger.info("Task list: \n%s" % "\n".join(["%d: %s" % (i, str(t)) for i, t in enumerate(self.task_list)]))

        # ------------------ 各单位执行各自的任务 --------------------
        MIN_SAP_PRIORITY: Dict[UnitTaskType, float] = {
            UnitTaskType.CAPTURE_RELIC: 4.5,
            UnitTaskType.INVESTIGATE: 3.0,
            UnitTaskType.ATTACK: 2.0,
            UnitTaskType.EXPLORE: 2.0,
            UnitTaskType.IDLE: 1.0,
        }
        attack_idle_pts: Set[Tuple[int, int]] = set()
        assert np.all(obs.opp_units_pos[obs.opp_units_mask] >= 0)
        for uid in range(C.MAX_UNITS):
            task = self.task_list[uid]
            u_selector = (self.team_id, uid)
            u_pos = obs.units_position[u_selector]
            u_energy = obs.units_energy[u_selector]
            energy_weight = self.energy_weight_fn(u_energy, self.game_map.move_cost)

            # 若相邻格有比自己能量低的敌方单位, 则直接走向敌方
            action_decided = False
            for i, delta in enumerate(C.DIRECTIONS):
                enemy_pos = u_pos + delta
                enemy_mask = np.all(obs.opp_units_pos == enemy_pos, axis=1) & obs.opp_units_mask
                if not np.any(enemy_mask):
                    continue
                total_energy = np.sum(obs.opp_units_energy[enemy_mask])
                if total_energy <= u_energy:
                    actions[uid] = [i+1, 0, 0]
                    self.logger.info(f"Unit {uid} -> crash {enemy_pos}")
                else:
                    actions[uid] = [(i^2)+1, 0, 0]
                    self.logger.info(f"Unit {uid} -> avoid {enemy_pos}")
                action_decided = True
                break
            if action_decided:
                continue

            # 判断是否进行Sap攻击
            if u_energy >= 1.5 * self.sap_cost:
                saps_in_range_mask = np.array([
                    np.max(np.abs(u_pos - sap.target_pos)) <= self.sap_range and not sap.satisfied()
                    for sap in self.sap_orders
                ])
                if np.any(saps_in_range_mask):
                    selected_sap = np.argmax(saps_in_range_mask)
                    sap_priority = self.sap_orders[selected_sap].priority
                    sap_target = self.sap_orders[selected_sap].target_pos
                    if sap_priority >= MIN_SAP_PRIORITY[task.type]:
                        actions[uid] = [5, sap_target[0]-u_pos[0], sap_target[1]-u_pos[1]]
                        self.sap_orders[selected_sap].fulfilled_count += 1
                        self.logger.info(f"Unit {uid} -> sap {sap_target}")
                        continue

            # CAPTURE_RELIC任务: 直接走向对应目标
            if task.type == UnitTaskType.CAPTURE_RELIC:
                if not np.array_equal(u_pos, task.target_pos) and \
                   self.game_map.obstacle_map[tuple(task.target_pos)] == Landscape.ASTEROID.value:
                    task.clear()
                actions[uid] = [self.game_map.direction_to(u_pos, task.target_pos, energy_weight), 0, 0]

            # INVESTIGATE任务: 在未知点上来回走动
            elif task.type == UnitTaskType.INVESTIGATE:
                first_arrival: int = task.data.get("first_arrival", 10000)
                if np.array_equal(u_pos, task.target_pos):
                    actions[uid] = [np.random.randint(0, 5), 0, 0]
                    task.data["first_arrival"] = min(first_arrival, step)
                elif self.game_map.obstacle_map[tuple(task.target_pos)] == Landscape.ASTEROID.value:
                    task.clear()
                else:
                    actions[uid] = [self.game_map.direction_to(u_pos, task.target_pos, energy_weight), 0, 0]

                if self.relic_map[tuple(task.target_pos)] != RelicInfo.UNKNOWN.value:
                    self.logger.info(f"Unit {uid} complete INVESTIGATE task")
                    task.clear()
                if step > first_arrival + 20:  # 任务自动结束
                    self.logger.info(f"Unit {uid} ends INVESTIGATE task")
                    task.clear()

            # EXPLORE任务: 移动到指定点
            elif task.type == UnitTaskType.EXPLORE:
                if np.array_equal(u_pos, task.target_pos):
                    task.clear()
                elif self.game_map.obstacle_map[tuple(task.target_pos)] == Landscape.ASTEROID.value:
                    task.clear()
                else:
                    actions[uid] = [self.game_map.direction_to(u_pos, task.target_pos, energy_weight), 0, 0]

            # ATTACK任务: 在遗迹附近寻找能量较高点
            elif task.type == UnitTaskType.ATTACK:
                MIN_ATTACK_DIST = 1
                MAX_ATTACK_DIST = self.sap_range

                if np.max(np.abs(u_pos - task.target_pos)) >= MAX_ATTACK_DIST + 2 or u_energy >= 350:
                    actions[uid] = [self.game_map.direction_to(u_pos, task.target_pos, energy_weight), 0, 0]
                    continue

                high_eng_mask = np.ones_like(self.game_map.full_energy_map, dtype=bool)  # 暂时不设mask
                high_eng_pts = np.column_stack(np.where(high_eng_mask) + (self.game_map.full_energy_map[high_eng_mask].flatten(),))
                dists = np.max(np.abs(high_eng_pts[:, :2] - task.target_pos), axis=1)
                mask = (MIN_ATTACK_DIST <= dists) & (dists <= MAX_ATTACK_DIST)
                high_eng_pts = high_eng_pts[mask]
                dists = dists[mask]
                high_eng_pts = high_eng_pts[np.argsort(high_eng_pts[:, 2])]

                for i in range(high_eng_pts.shape[0]):
                    target_pos = high_eng_pts[i, :2]
                    if (target_pos[0], target_pos[1]) in attack_idle_pts:
                        continue
                    actions[uid] = [self.game_map.direction_to(u_pos, target_pos, energy_weight), 0, 0]
                    attack_idle_pts.add((target_pos[0], target_pos[1]))
                    break

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
                    actions[uid] = [self.game_map.direction_to(u_pos, target_pos, energy_weight), 0, 0]

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

    def generate_sap_order(self) -> None:
        """生成攻击需求列表"""
        obs = self.obs
        self.sap_orders = []

        real_points = np.vstack(np.where(self.relic_map == RelicInfo.REAL.value)).T  # shape (N, 2)
        dists = np.sum(np.abs(real_points - self.base_pos), axis=1)
        real_points_myhalf = real_points[dists <= C.MAP_SIZE]  # 我方半区得分点
        # 针对可见敌人生成SapOrder
        ON_RELIC_WEIGHT = 3.0
        visible_enemy_on_relic: Set[Tuple[int, int]] = set()
        for eid, e_pos, e_energy in zip(range(C.MAX_UNITS), obs.opp_units_pos, obs.opp_units_energy):
            enemy_can_move = (e_energy >= self.game_map.move_cost)
            if not obs.units_mask[self.team_id, eid] or e_energy < 0:
                continue

            priority = 0.0

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

            # 3. 能量较低
            need_hit_count = int(e_energy / self.sap_cost) + 1
            if need_hit_count == 1:
                priority += 2.0
            elif need_hit_count == 2:
                priority += 1.0
            elif need_hit_count == 3:
                priority += 0.5

            # 打提前量
            if not on_relic and enemy_can_move:
                e_pos += C.ADJACENT_DELTAS[np.argmax(self.__pred_enemy_pos(eid))]

            # TODO: 融合多个相邻的SapOrder
            safe_hit_count = need_hit_count + 0 if (self.sap_dropoff == 1 or not enemy_can_move) else 1
            self.sap_orders.append(SapOrder(e_pos, priority, safe_hit_count))

        # 针对不在视野内的得分点生成SapOrder
        unknown_points = np.vstack(np.where(self.relic_map == RelicInfo.UNKNOWN.value)).T
        unknown_points_invisible = unknown_points[obs.sensor_mask[unknown_points[:, 0], unknown_points[:, 1]] == 0]
        real_points_invisible = real_points[obs.sensor_mask[real_points[:, 0], real_points[:, 1]] == 0]
        if real_points_invisible.shape[0] > 0:
            possibility: float = (obs.team_points[self.opp_team_id] - self.history[-1].team_points[self.opp_team_id])
            possibility -= len(visible_enemy_on_relic)
            possibility = min(1.0, possibility / (real_points_invisible.shape[0] + 0.4 * unknown_points_invisible.shape[0]))
            self.logger.info("Possibility of invisible relics: {} / {} = {}".format(
                possibility * (real_points_invisible.shape[0] + 0.4 * unknown_points_invisible.shape[0]),
                (real_points_invisible.shape[0] + 0.4 * unknown_points_invisible.shape[0]), possibility))
            for r_pos in real_points_invisible:
                self.sap_orders.append(SapOrder(r_pos, ON_RELIC_WEIGHT * possibility, 2))

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

        # 每个SapOrder向相邻的SapOrder加上self.sap_dropoff倍的优先级
        delta_priority = np.zeros(len(self.sap_orders))
        for i in range(len(self.sap_orders)):
            for j in range(i+1, len(self.sap_orders)):
                if np.sum(np.abs(self.sap_orders[i].target_pos - self.sap_orders[j].target_pos)) == 1:
                    delta_priority[i] += self.sap_dropoff * self.sap_orders[j].priority
                    delta_priority[j] += self.sap_dropoff * self.sap_orders[i].priority
        for i in range(len(self.sap_orders)):
            self.sap_orders[i].priority += delta_priority[i]

        self.sap_orders.sort(reverse=True)
        if len(self.sap_orders) > 0: self.logger.info(f"Sap orders: {self.sap_orders}")

    def __pred_enemy_pos(self, eid: int) -> np.ndarray:
        """预测指定id的敌方单位下回合的移动, 下标同C.ADJACENT_DELTAS"""

        curr_pos: np.ndarray = self.obs.opp_units_pos[eid]
        dir_scores = np.zeros(C.ADJACENT_DELTAS.shape[0])

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
            self.logger.info(f"Predict enemy {eid} direction after move_crit: {dir_scores}")

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
                self.logger.info(f"Predict enemy {eid} direction after relic_crit: {dir_scores}, best_dir: {best_dir}")

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
            self.logger.info(f"Predict enemy {eid} direction after energy modifications: {dir_scores}")

        dir_scores /= np.sum(dir_scores)  # 归一化
        self.logger.info(f"Predict enemy {eid} direction: {dir_scores}")
        return dir_scores

    @staticmethod
    def energy_weight_fn(energy: int, move_cost: int) -> float:
        steps = energy // move_cost

        if energy < 25:
            return 10
        elif energy < 100 or steps < 20:
            return 0.3
        elif energy < 250:
            return 0.2
        elif energy < 350:
            return 0.10
        else:
            return 0.05
