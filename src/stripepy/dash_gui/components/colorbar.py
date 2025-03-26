import numpy as np


def colorbar(matrix):
    max_value = np.nanmax(matrix)
    colorbar_max = round(max_value) + 1
    return _colorbar(colorbar_max)


def _colorbar(n):
    return dict(
        tick0=0,
        title="Counts (log)",
        tickmode="array",
        tickvals=np.linspace(1, n, n + 1),
        ticktext=np.linspace(1, n, n + 1),
    )
