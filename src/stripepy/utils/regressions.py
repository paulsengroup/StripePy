# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import numpy as np
import numpy.typing as npt


def _compute_wQISA_predictions(Y: npt.NDArray, k: int) -> npt.NDArray[float]:
    assert k >= 1
    # Control points of a weighted quasi-interpolant spline approximation with a k-NN weight function:
    kernel = np.full(k, 1 / k)
    radius = (k - 1) // 2
    Y_ext = np.pad(Y, (radius, radius), mode="edge")
    predictions = np.convolve(Y_ext, kernel, "valid")

    return predictions
