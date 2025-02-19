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
