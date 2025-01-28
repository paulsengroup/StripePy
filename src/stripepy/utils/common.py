# Copyright (C) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import decimal
import time
from typing import Optional, Sequence

import numpy as np
import scipy.sparse as ss
from numpy.typing import NDArray


def pretty_format_elapsed_time(
    t0: float,
    t1: Optional[float] = None,
) -> str:
    """
    Format elapsed time between t1 and t0 as a human-readable string.

    Examples:
        123ns
        1.234us
        1.23ms
        1.23s
        1m:2.345s
        1h:2m:3.456s

    Parameters
    ----------
    t0: float
        start time in seconds.
    t1: Optional[float]
        end time in seconds.
        When not provided, use the current time.

    Returns
    -------
    str
        a human-friendly string representation of the elapsed time.
    """
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


def truncate_np(
    v: NDArray[float],
    places: int,
) -> NDArray[float]:
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


def zero_rows(
    matrix: ss.csr_matrix,
    rows: Sequence[int],
) -> ss.csr_matrix:
    """
    Set the given rows of the CSR matrix to zero.
    Original implementation from https://stackoverflow.com/a/43114513

    Parameters
    ----------
    matrix: ss.csr_matrix
        the CSR matrix to be zeroed
    rows: Sequence[int]
        the rows of the CSR matrix to be zeroed

    Returns
    -------
    ss.csr_matrix
        a copy of the input CSR matrix with the relevant rows set to zero.
    """
    diag = ss.eye(matrix.shape[0]).tolil()
    for i in rows:
        diag[i, i] = 0
    return diag.dot(matrix).tocsr()


def zero_columns(
    matrix: ss.csc_matrix,
    columns: Sequence[int],
) -> ss.csc_matrix:
    """
    Set the given columns of the CSC matrix to zero.
    Original implementation from https://stackoverflow.com/a/43114513

    Parameters
    ----------
    matrix: ss.csc_matrix
        the CSC matrix to be zeroed
    columns: Sequence[int]
        the columns of the CSC matrix to be zeroed

    Returns
    -------
    ss.csc_matrix
        a copy of the input CSC matrix with the relevant columns set to zero.
    """
    diag = ss.eye(matrix.shape[1]).tolil()
    for i in columns:
        diag[i, i] = 0
    return matrix.dot(diag).tocsc()


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
    _import_matplotlib()  # this will deal with import errors
    import matplotlib.pyplot as plt

    return plt


class _DummyPyplot(object):
    """
    class to mock common types from matplotlib.pyplot.
    """

    def __init__(self):
        self.Figure = None
        self.Axes = None
