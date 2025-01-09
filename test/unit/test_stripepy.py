# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
import scipy.sparse as ss

from stripepy.stripepy import _log_transform
from stripepy.stripepy import _band_extraction


@pytest.mark.unit
class TestLogTransform:

    def test_empty(self):
        I = ss.csr_matrix((0, 0), dtype=float)
        Iproc = _log_transform(I)

        assert Iproc.shape == (0, 0)
        assert np.issubdtype(I.dtype, np.floating)

    def test_all_finite(self):
        I = ss.rand(100, 100, density=0.5, format="csr")
        Iproc = _log_transform(I)

        assert I.size == Iproc.size
        assert I.shape == Iproc.shape
        assert np.isfinite(Iproc.data).all()

    def test_with_nans(self):
        I = ss.rand(100, 100, density=0.5, format="csr")
        size_I = I.size
        mean_I = I.mean()
        num_nan_values = (I.data >= mean_I).sum()
        I.data[I.data >= mean_I] = np.nan

        Iproc = _log_transform(I)

        assert np.isfinite(Iproc.data).all()
        assert Iproc.size == size_I - num_nan_values

@pytest.mark.unit
class TestBandExtraction:

    def test_empty_matrix(self):
        I = ss.rand(10, 10, density=0, format="csr")
        LT_I, _ = _band_extraction(I, 1, 1)

        assert LT_I.size == 0

    def test_non_symmetric(self):
        data = [*range(1,5)]
        offsets = [-1]
        I = ss.dia_matrix((data, offsets), shape=(5,5))
        LT_I, UT_I = _band_extraction(I, 1, 4)

        assert LT_I.size == 4

    def test_non_diagonal(self):
        data = [[*range(1,5)], [*range(0, 4)]]
        offsets = [-1, 1]
        I = ss.dia_matrix((data, offsets), shape=(5,5))
        LT_I, _ = _band_extraction(I, 1, 4)

        assert LT_I.size == 4

    def test_matrix_belt_oob(self):
        data = [*range(1,5)]
        offsets = [0]
        I = ss.dia_matrix((data, offsets), shape=(4,4))
        LT_I, _ = _band_extraction(I, 0.0000001, 3)

        assert LT_I.size == 4


    def test_is_correct_triangle(self):
        data = [[*range(1,5)], [*range(4, 0, -1)]]
        offsets = [-1, 1]
        I = ss.dia_matrix((data, offsets), shape=(5,5))
        LT_I, UT_I = _band_extraction(I, 1, 2)

        assert (LT_I.diagonal(0) == np.array([0, 0, 0, 0, 0])).all()
        assert (LT_I.diagonal(-1) == np.array([1, 2, 3, 4])).all()
        assert (UT_I.diagonal(1) == np.array([3, 2, 1, 0])).all()
        pass

