import math

import numpy as np


def colorbar(matrix, scale_type):
    """
    Specifies the plot's colorbar.

    Returns
    -------
    colorbar:
        a JSON parsable dictionary containing the parameters specifying the colorbar
    """
    find_normal_values = matrix[np.isfinite(matrix)]
    max_float = np.nanmax(find_normal_values)
    min_float = np.nanmin(find_normal_values)
    if scale_type == "log scale":
        return _colorbar(max_float, min_float)
    else:
        return _colorbar_normal_scale(max_float)


def _colorbar(max_float, min_float):
    max_int = round(max_float)
    tickvals_array = np.array([math.log(val, max_int) for val in range(1, max_int)]) * max_float
    np.append(tickvals_array, max_float)
    tickvals_set = set(map(int, tickvals_array))
    tickvals = list(tickvals_set)

    ticktext = [str(int(np.e**val)) for val in tickvals]
    ticktext.append(np.e**max_float)

    return dict(
        title="Counts (log)",
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        separatethousands=True,
    )


def _colorbar_normal_scale(max_float):
    max_int = round(max_float)
    tickvals_array = np.linspace(1, max_int, num=10)
    tickvals_set = set(map(int, tickvals_array))
    tickvals = list(tickvals_set)

    return dict(
        title="Counts",
        tickmode="array",
        tickvals=tickvals,
        ticktext=[str(int(val)) for val in tickvals],
        separatethousands=True,
    )
