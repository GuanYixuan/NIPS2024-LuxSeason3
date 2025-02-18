from typing import Dict, Any
import numpy as np
from dataclasses import dataclass
import json

@dataclass
class Observation:
    step: int  # 当前时间步，从0开始
    player: str  # 玩家ID，"player_0" 或 "player_1"
    remainingOverageTime: int  # 剩余的超时时间
    obs: Dict[str, Any]  # 游戏观察状态
    info: Dict[str, Any]  # 环境配置信息

    @property
    def units_mask(self) -> np.ndarray:
        """单位掩码，shape (max_units, )"""
        return np.array(self.obs["units_mask"])

    @property
    def units_positions(self) -> np.ndarray:
        """单位位置，shape (max_units, 2)"""
        return np.array(self.obs["units"]["position"])

    @property
    def units_energy(self) -> np.ndarray:
        """单位能量，shape (max_units, 1)"""
        return np.array(self.obs["units"]["energy"])

    @property
    def relic_nodes(self) -> np.ndarray:
        """遗迹节点位置，shape (max_relic_nodes, 2)"""
        return np.array(self.obs["relic_nodes"])

    @property
    def relic_nodes_mask(self) -> np.ndarray:
        """遗迹节点掩码，shape (max_relic_nodes, )"""
        return np.array(self.obs["relic_nodes_mask"])

    @property
    def team_points(self) -> np.ndarray:
        """队伍得分，每个队伍的得分"""
        return np.array(self.obs["team_points"])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Observation':
        """从字典创建 Observation 实例"""
        return cls(
            step=data["step"],
            player=data["player"],
            remainingOverageTime=data["remainingOverageTime"],
            obs=data["obs"] if isinstance(data["obs"], dict) else json.loads(data["obs"]),
            info=data["info"]
        )
