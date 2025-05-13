import math

import numpy as np


def colorbar(matrix):
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
    return _colorbar(max_float, min_float)


def _colorbar(max_float, min_float):
    max_int = round(max_float)
    tickvals_array = np.array([math.log(val, max_int) for val in range(1, max_int)]) * max_float
    tickvals_set = set(map(int, tickvals_array))
    tickvals = list(tickvals_set)

    return dict(
        title="Counts (log)",
        exponentformat="e",
        tickmode="array",
        tickvals=tickvals,
        ticktext=[str(int(np.e**val)) for val in tickvals],
        separatethousands=True,
    )
