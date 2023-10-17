import numpy as np
from typing import Sequence
from pipeline import P

from .statistics import quick_statistics_binary, fpr, tpr, fnr


def get_roc_binary(predictions: Sequence[P.ArrayEntry], pos_label, steps=100) -> np.array:
    result = np.zeros((2, steps + 2))
    for step in range(0, steps + 1):
        thresh = step / steps
        results = quick_statistics_binary(predictions, pos_label, thresh)
        result[0, step] = fpr(*results)
        result[1, step] = tpr(*results)
    return result


def get_det_binary(predictions: Sequence[P.ArrayEntry], pos_label, steps=100) -> np.array:
    result = np.zeros((2, steps + 1))
    for step in range(0, steps + 1):
        thresh = step / steps
        results = quick_statistics_binary(predictions, pos_label, thresh)
        result[0, step] = fpr(*results)
        result[1, step] = fnr(*results)
    return result
