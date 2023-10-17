import typing as Tp
import numpy as np

class ArrayEntry(Tp.TypedDict):
    uid: str
    data: np.array
    extra: Tp.Optional[dict]
