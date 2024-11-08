import numpy as np


def _compute_wQISA_predictions(Y, k):

    # Control points of a weighted quasi-interpolant spline approximation with a k-NN weight function:
    kernel = np.full((k,), 1 / k)
    radius = (k - 1) // 2
    Y_ext = np.pad(Y.tolist(), (radius, radius), mode="edge")
    predictions = np.convolve(Y_ext, kernel, "valid")

    return predictions
