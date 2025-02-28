import numpy as np
from numpy import dtype

from typing import Literal, Union, Tuple
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    Size = TypeVar("Size", bound=str, covariant=True)
    ScalarType = TypeVar("ScalarType", bound=np.generic, covariant=True)
    Shape = Literal
    _NDArray = np.ndarray[Size, dtype[ScalarType]]  # type: ignore
else:
    Shape = Literal
    _NDArray = Tuple

class Constants:
    """游戏常量"""

    MAP_SIZE = 24
    """游戏地图大小"""
    MAX_UNITS = 16
    """最大单位数量"""
    RELIC_SPREAD = 2
    """遗迹得分点相对遗迹中心点的最大单轴距离"""

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

def in_map(pos: np.ndarray) -> Union[np.bool_, np.ndarray]:
    """判断输入的坐标是否在游戏地图内, pos形状如(..., 2)"""
    return np.all(np.logical_and(pos >= 0, pos < Constants.MAP_SIZE), axis=-1)

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
