# Copyright (C) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import decimal
import time
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def sort_values(*vectors: Sequence) -> Tuple[NDArray]:
    """
    Sort two or more sequences of objects as if each sequence was a column in a
    table and the table was being sorted row-by-row based on values from all columns.

    Parameters
    ----------
    vectors: two or more sequences to be sorted

    Returns
    -------
    Tuple[NDArray]
        the sorted sequences as numpy.ndarray
    """
    if len(vectors) < 2:
        raise ValueError("please specify at least two sequences")

    for v in vectors[1:]:
        assert len(vectors[0]) == len(v)

    if len(vectors[0]) == 0:
        return tuple((np.array(v) for v in vectors))  # noqa

    df = pd.DataFrame({i: v for i, v in enumerate(vectors)})
    df.sort_values(by=df.columns.tolist(), inplace=True, kind="stable")

    return tuple(df[col].to_numpy() for col in df.columns)  # noqa


def pretty_format_elapsed_time(t0: float, t1: Optional[float] = None) -> str:
    if t1 is None:
        t1 = time.time()

    microsecond = 1.0e-6
    millisecond = 1.0e-3
    second = 1.0
    minute = 60.0
    hour = 3600.0

    delta = t1 - t0

    if delta < microsecond:
        return f"{delta * 1.0e9:.0f}ns"

    if delta < millisecond:
        return f"{delta * 1.0e6:.3f}us"

    if delta < second:
        return f"{delta * 1000:.3f}ms"

    if delta < minute:
        return f"{delta:.3f}s"

    if delta < hour:
        minutes = delta // 60
        seconds = delta - (minutes * 60)
        return f"{minutes:.0f}m:{seconds:.3f}s"

    hours = delta // 3600
    minutes = (delta - (hours * 3600)) // 60
    seconds = delta - (hours * 3600) - (minutes * 60)
    return f"{hours:.0f}h:{minutes:.0f}m:{seconds:.3f}s"


def truncate_np(v: NDArray[float], places: int) -> NDArray[float]:
    """
    Truncate a numpy array to the given number of decimal places.
    Implementation based on https://stackoverflow.com/a/28323804

    Parameters
    ----------
    v: NDArray[float]
        the numpy array to be truncated
    places: int
        the number of decimal places to truncate to

    Returns
    -------
    NDArray[float]
        numpy array with truncated values
    """
    assert places >= 0

    if places == 0:
        return v.round(0)

    with decimal.localcontext() as context:
        context.rounding = decimal.ROUND_DOWN
        exponent = decimal.Decimal(str(10**-places))
        return np.array([float(decimal.Decimal(str(n)).quantize(exponent)) for n in v], dtype=float)


def _import_matplotlib():
    """
    Helper function to import matplotlib.
    """
    try:
        import matplotlib

        return matplotlib
    except ImportError as e:
        raise ImportError(
            "unable to import matplotlib: to enable matplotlib support, please install StripePy with: pip install 'stripepy-hic[all]'"
        ) from e


def _import_pyplot():
    """
    Helper function to import matplotlib.pyplot.
    """
    _import_matplotlib()
    import matplotlib.pyplot as plt

    return plt


class _DummyPyplot(object):
    """
    class to mock common types from matplotlib.pyplot.
    """

    def __init__(self):
        self.Figure = None
        self.Axes = None
