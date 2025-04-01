import numpy as np


def colorbar(matrix):
    find_normal_values = matrix[np.isfinite(matrix)]
    max_float = np.nanmax(find_normal_values)
    min_float = np.nanmin(find_normal_values)
    return _colorbar(max_float, min_float)


def _colorbar(max_float, min_float):
    tickvals = [np.e ** (max_float * num / 20) for num in range(20, 0, -3)]

    return dict(
        title="Counts (log)",
        tickmode="array",
        exponentformat="e",
        tickvals=np.log(tickvals),
        ticktext=[str(int(val)) for val in tickvals],
        separatethousands=True,
    )
