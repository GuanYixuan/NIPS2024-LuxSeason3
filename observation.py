from typing import Dict, Any
import numpy as np
from dataclasses import dataclass

"""
Explanations for obs dictionary:
// T is the number of teams (default is 2)
// N is the max number of units per team
// W, H are the width and height of the map
// R is the max number of relic nodes
{
    "units": {
        "position": Array(T, N, 2),
        "energy": Array(T, N, 1)
    },
    // whether the unit exists and is visible to you. units_mask[t][i] is whether team t's unit i can be seen and exists.
    "units_mask": Array(T, N),
    // whether the tile is visible to the unit for that team
    "sensor_mask": Array(W, H),
    "map_features": {
        // amount of energy on the tile
        "energy": Array(W, H),
        // type of the tile. 0 is empty, 1 is a nebula tile, 2 is asteroid
        "tile_type": Array(W, H)
    },
    // whether the relic node exists and is visible to you.
    "relic_nodes_mask": Array(R),
    // position of the relic nodes.
    "relic_nodes": Array(R, 2),
    // points scored by each team in the current match
    "team_points": Array(T),
    // number of wins each team has in the current game/episode
    "team_wins": Array(T),
    // number of steps taken in the current game/episode
    "steps": int,
    // number of steps taken in the current match
    "match_steps": int
}
"""

@dataclass
class Observation:
    """游戏观察状态类

    属性说明：
    - units_position: 单位位置，shape (T, N, 2)，T为队伍数，N为每队最大单位数
    - units_energy: 单位能量，shape (T, N, 1)
    - units_mask: 单位掩码，shape (T, N)，表示单位是否存在且可见
    - sensor_mask: 视野掩码，shape (W, H)，表示地图上的可见区域
    - map_energy: 地图上的能量分布，shape (W, H)
    - map_tile_type: 地图格子类型，shape (W, H)，0为空，1为星云，2为小行星
    - relic_nodes: 遗迹节点位置，shape (R, 2)，R为最大遗迹数
    - relic_nodes_mask: 遗迹节点掩码，shape (R,)，表示遗迹是否存在且可见
    - team_points: 队伍得分，shape (T,)
    - team_wins: 队伍获胜次数，shape (T,)
    - steps: 当前游戏/回合的步数
    - match_steps: 当前比赛的步数
    """

    step: int  # 当前时间步，从0开始
    player: str  # 玩家ID，"player_0" 或 "player_1"
    remainingOverageTime: int  # 剩余的超时时间
    info: Dict[str, Any]  # 环境配置信息

    # 从obs中提取的字段
    units_position: np.ndarray
    units_energy: np.ndarray
    units_mask: np.ndarray
    sensor_mask: np.ndarray
    map_energy: np.ndarray
    map_tile_type: np.ndarray
    relic_nodes: np.ndarray
    relic_nodes_mask: np.ndarray
    team_points: np.ndarray
    team_wins: np.ndarray
    steps: int
    match_steps: int

    def __init__(self, step: int, player: str, remainingOverageTime: int,
                 obs: Dict[str, Any], info: Dict[str, Any]):
        """从原始数据初始化观察状态

        Args:
            step: 当前时间步
            player: 玩家ID
            remainingOverageTime: 剩余超时时间
            obs: 观察状态字典
            info: 环境配置信息
        """
        self.step = step
        self.player = player
        self.remainingOverageTime = remainingOverageTime
        self.info = info

        # 提取obs中的字段
        self.units_position = np.array(obs["units"]["position"])
        self.units_energy = np.array(obs["units"]["energy"])
        self.units_mask = np.array(obs["units_mask"])
        self.sensor_mask = np.array(obs["sensor_mask"])
        self.map_energy = np.array(obs["map_features"]["energy"])
        self.map_tile_type = np.array(obs["map_features"]["tile_type"])
        self.relic_nodes = np.array(obs["relic_nodes"])
        self.relic_nodes_mask = np.array(obs["relic_nodes_mask"])
        self.team_points = np.array(obs["team_points"])
        self.team_wins = np.array(obs["team_wins"])
        self.steps = obs["steps"]
        self.match_steps = obs["match_steps"]
