import numpy as np

from typing import Union, Tuple, TypeVar

class Constants:
    """游戏常量"""

    MAP_SIZE = 24
    """游戏地图大小"""
    MAX_UNITS = 16
    """最大单位数量"""
    MAX_STEPS_IN_MATCH = 100
    """每场比赛的最大步数"""
    RELIC_SPREAD = 2
    """遗迹得分点相对遗迹中心点的最大单轴距离"""

    FULL_DIRECTIONS = np.array([
        [0, 0],
        [0, -1],  # 上  (游戏方向值1)
        [1, 0],   # 右  (游戏方向值2)
        [0, 1],   # 下  (游戏方向值3)
        [-1, 0],  # 左  (游戏方向值4)
    ])

    DIRECTIONS = np.array([
        [0, -1],  # 上  (游戏方向值1)
        [1, 0],   # 右  (游戏方向值2)
        [0, 1],   # 下  (游戏方向值3)
        [-1, 0],  # 左  (游戏方向值4)
    ])
    """四个运动方向, 注意此处索引比游戏中的方向值小1"""

    ADJACENT_DELTAS = np.array([
        [0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]
    ])
    """相邻坐标"""

    DEFAULT_ENERGY_VALUE = 0
    """默认的能量未知值"""

    INIT_UNIT_ENERGY = 100
    """单位的初始能量"""

    POSSIBLE_SAP_DROPOFF = np.array([0.25, 0.5, 1])

def get_coord_list() -> np.ndarray:
    """获取游戏地图的所有坐标"""
    return np.stack(np.meshgrid(np.arange(Constants.MAP_SIZE), np.arange(Constants.MAP_SIZE)), axis=-1).reshape(-1, 2)

def get_dir_index(delta: np.ndarray) -> int:
    """根据相邻坐标delta返回游戏方向值"""
    return np.where(np.all(Constants.DIRECTIONS == delta, axis=1))[0][0] + 1

def in_map(pos: np.ndarray) -> Union[np.bool_, np.ndarray]:
    """判断输入的坐标是否在游戏地图内, pos形状如(..., 2)"""
    return np.all(np.logical_and(pos >= 0, pos < Constants.MAP_SIZE), axis=-1)

def l1_dist(pos1: np.ndarray, pos2: np.ndarray = np.zeros(2)) -> np.ndarray:
    """计算两个坐标之间的L1距离, pos1和pos2其一形状如(..., 2)"""
    return np.sum(np.abs(pos1 - pos2), axis=-1)

def square_dist(pos1: np.ndarray, pos2: np.ndarray = np.zeros(2)) -> np.ndarray:
    """计算两个坐标之间的单轴最大距离, pos1和pos2其一形状如(..., 2)"""
    return np.max(np.abs(pos1 - pos2), axis=-1)

def dist_to_segment(seg_p1: np.ndarray, seg_p2: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """计算点pos到线段seg_p1-seg_p2的距离, seg_p1和seg_p2形状为(2,), pos形状为(..., 2), 结果形状为(...,)"""
    line_vec = seg_p2 - seg_p1
    point_vec = pos - seg_p1
    line_len_sq = np.sum(line_vec * line_vec)

    if line_len_sq == 0:
        return np.sqrt(np.sum(point_vec * point_vec, axis=-1))

    # 投影比例 t
    t = np.sum(point_vec * line_vec, axis=-1) / line_len_sq
    t = np.clip(t, 0, 1)

    # 投影点坐标
    projection = seg_p1 + np.expand_dims(t, axis=-1) * line_vec
    return np.sqrt(np.sum((pos - projection) ** 2, axis=-1))

def flip_matrix(arr: np.ndarray) -> np.ndarray:
    """根据游戏的对称方式, 对输入的方阵进行翻转"""
    assert arr.shape[0] == arr.shape[1] and arr.ndim == 2, "arr must be a square matrix"
    return np.rot90(np.rot90(arr).T, -1)

flip_type = TypeVar("flip_type", Tuple[int, int], np.ndarray)
def flip_coord(coord: flip_type, size: int = Constants.MAP_SIZE) -> flip_type:
    """根据游戏的对称方式, 对输入的坐标进行翻转"""
    if isinstance(coord, tuple):
        x, y = coord
        return (size - 1) - y, (size - 1) - x
    else:
        x = coord[..., 0]
        y = coord[..., 1]
        return np.stack([size - 1 - y, size - 1 - x], axis=-1)
