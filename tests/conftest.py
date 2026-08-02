from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.calib_observability.lie_se2 import se2_exp
from src.calib_observability.lie_se3 import se3_exp


def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def small_xi6(seed: int = 0) -> np.ndarray:
    r = np.random.default_rng(seed)
    return r.normal(0.0, [0.2, 0.2, 0.2, 0.5, 0.5, 0.5])


def small_xi3(seed: int = 0) -> np.ndarray:
    r = np.random.default_rng(seed)
    return r.normal(0.0, [0.2, 0.4, 0.4])


def T6(seed: int = 0) -> np.ndarray:
    return se3_exp(small_xi6(seed))


def T3(seed: int = 0) -> np.ndarray:
    return se2_exp(small_xi3(seed))
