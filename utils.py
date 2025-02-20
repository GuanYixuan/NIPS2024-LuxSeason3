import numpy as np
from numpy import dtype

from typing import Literal, Tuple
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    Size = TypeVar("Size", bound=str, covariant=True)
    ScalarType = TypeVar("ScalarType", bound=np.generic, covariant=True)
    Shape = Literal
    _NDArray = np.ndarray[Size, dtype[ScalarType]]  # type: ignore
else:
    Shape = Literal
    _NDArray = Tuple

def flip_matrix(arr: np.ndarray) -> np.ndarray:
    """根据游戏的对称方式, 对输入的方阵进行翻转"""
    assert arr.shape[0] == arr.shape[1] and arr.ndim == 2, "arr must be a square matrix"
    return np.rot90(np.rot90(arr).T, -1)

class Constants:
    """游戏常量"""

    MAP_SIZE = 24
    """游戏地图大小"""

    DEFAULT_ENERGY_VALUE = 0
    """默认的能量未知值"""
