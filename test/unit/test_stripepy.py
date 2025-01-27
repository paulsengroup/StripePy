# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
import scipy.sparse as ss

from stripepy.stripepy import (
    _band_extraction,
    _check_neighborhood,
    _compute_global_pseudodistribution,
    _extract_RoIs,
    _log_transform,
    _scale_Iproc,
)


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
        row1 = np.array([0, 0, 0, 0, 0])
        row2 = np.array([0, 0, 0, 0, 0])
        row3 = np.array([0, 0, 0, 0, 0])
        row4 = np.array([0, 0, 0, 0, 0])
        row5 = np.array([0, 0, 0, 0, 0])
        matrix = np.array([row1, row2, row3, row4, row5])
        I = ss.csr_matrix(matrix)

        LT_I, UT_I = _band_extraction(I, 1, 4)

        assert np.array_equal(LT_I.toarray(), UT_I.toarray())

    def test_non_symmetric(self):
        """
        |   0   0   0   0   0   |
        |   1   0   0   0   0   |
        |   0   2   0   0   0   |
        |   0   0   3   0   0   |
        |   0   0   0   4   0   |
        """
        row1 = np.array([0, 0, 0, 0, 0])
        row2 = np.array([1, 0, 0, 0, 0])
        row3 = np.array([0, 2, 0, 0, 0])
        row4 = np.array([0, 0, 3, 0, 0])
        row5 = np.array([0, 0, 0, 4, 0])
        matrix = np.array([row1, row2, row3, row4, row5])
        I = ss.csr_matrix(matrix)

        LT_I, UT_I = _band_extraction(I, 1, 4)

        assert np.array_equal(LT_I.toarray(), matrix)
        assert UT_I.size == 0

    def test_non_diagonal(self):
        """
        |   0   2   0   0   0   |
        |   1   0   3   0   0   |
        |   0   2   0   4   0   |
        |   0   0   3   0   5   |
        |   0   0   0   4   0   |
        """
        row1 = np.array([0, 2, 0, 0, 0])
        row2 = np.array([1, 0, 3, 0, 0])
        row3 = np.array([0, 2, 0, 4, 0])
        row4 = np.array([0, 0, 3, 0, 5])
        row5 = np.array([0, 0, 0, 4, 0])
        matrix = np.array([row1, row2, row3, row4, row5])
        I = ss.csr_matrix(matrix)

        LT_I, UT_I = _band_extraction(I, 1, 4)

        verify_LT = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 3, 0, 0], [0, 0, 0, 4, 0]])
        verify_UT = np.array([[0, 2, 0, 0, 0], [0, 0, 3, 0, 0], [0, 0, 0, 4, 0], [0, 0, 0, 0, 5], [0, 0, 0, 0, 0]])
        assert np.array_equal(LT_I.toarray(), verify_LT)
        assert np.array_equal(UT_I.toarray(), verify_UT)

    def test_matrix_belt_oob(self):
        """
        |   1   0   0   0   0   |
        |   0   2   0   0   0   |
        |   0   0   3   0   0   |
        |   0   0   0   4   0   |
        |   0   0   0   0   5   |
        """
        row1 = np.array([1, 0, 0, 0, 0])
        row2 = np.array([0, 2, 0, 0, 0])
        row3 = np.array([0, 0, 3, 0, 0])
        row4 = np.array([0, 0, 0, 4, 0])
        row5 = np.array([0, 0, 0, 0, 5])
        matrix = np.array([row1, row2, row3, row4, row5])
        I = ss.csr_matrix(matrix)

        LT_I, UT_I = _band_extraction(I, 1, 3 * 10**3)
        verify_array = np.array([[1, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 3, 0, 0], [0, 0, 0, 4, 0], [0, 0, 0, 0, 5]])

        assert np.array_equal(LT_I.toarray(), verify_array)
        assert np.array_equal(UT_I.toarray(), verify_array)


@pytest.mark.unit
class TestScaleIProc:
    def test_divide_by_one(self):
        """
        |   1   0   0   0   |
        |   0   1   0   0   |
        |   0   0   1   0   |
        |   0   0   0   1   |
        """
        row1 = np.array([1, 0, 0, 0])
        row2 = np.array([0, 1, 0, 0])
        row3 = np.array([0, 0, 1, 0])
        row4 = np.array([0, 0, 0, 1])
        matrix = [row1, row2, row3, row4]
        I = ss.csr_matrix(matrix)
        LT_I, UT_I = _band_extraction(I, 1, 5)
        IScaled, LT_IScaled, UT_IScaled = _scale_Iproc(I, LT_I, UT_I)

        assert np.array_equal(IScaled.toarray(), LT_IScaled.toarray())
        assert np.array_equal(LT_IScaled.toarray(), UT_IScaled.toarray())
        assert np.array_equal(UT_IScaled.toarray(), matrix)

    def test_halve(self):
        """
        |   1   0   0   0   |
        |   0   2   0   0   |
        |   0   0   1   0   |
        |   0   0   0   1   |
        """
        row1 = np.array([1, 0, 0, 0])
        row2 = np.array([0, 2, 0, 0])
        row3 = np.array([0, 0, 1, 0])
        row4 = np.array([0, 0, 0, 1])
        matrix = [row1, row2, row3, row4]
        I = ss.csr_matrix(matrix)
        LT_I, UT_I = _band_extraction(I, 1, 5)
        IScaled, LT_IScaled, UT_IScaled = _scale_Iproc(I, LT_I, UT_I)
        assert (IScaled.diagonal(0) == np.array([0.5, 1, 0.5, 0.5])).all()


@pytest.mark.unit
class TestExtractRoIs:
    def test_is_NDarray(self):
        I = ss.rand(10, 10, density=0.5, format="csr")
        I_RoI = _extract_RoIs(I, {"matrix": [2, 5]})

        assert isinstance(I_RoI, np.ndarray)

    def test_is_square(self):
        I = ss.rand(10, 10, density=0.5, format="csr")
        I_RoI = _extract_RoIs(I, {"matrix": [2, 5]})

        assert I.shape[0] == I.shape[1]


"""
"""
"""
STEP 2
"""
"""
"""


@pytest.mark.unit
class TestComputeGlobalPseudodistribution:
    # TODO: implement generator function for matrix

    """
    The input matrix looks like
    |   9   0   0   |
    |   0   2   0   |
    |   0   4   0   |
    """

    @pytest.mark.skip(reason="Outgoing matrix is not necessarily sparse")
    def test_is_sparse(self):
        I = ss.rand(10, 10, density=0.5, format="csr")
        I_RoI = _compute_global_pseudodistribution(I, {"matrix": [2, 5]})

        assert ss.issparse(I_RoI)

    def test_is_marginalized(self):
        row = np.array([0, 1, 2])
        col = np.array([0, 1, 1])
        data = np.array([9.0, 2.0, 4.0])
        I = ss.csr_matrix((data, (row, col)), shape=(3, 3))
        I_RoI = _compute_global_pseudodistribution(I, {"matrix": [3, 3]})

        assert I_RoI.shape == (3,)

    def test_is_scaled(self):
        row = np.array([0, 1, 2])
        col = np.array([0, 1, 1])
        data = np.array([9.0, 2.0, 4.0])
        I = ss.csr_matrix((data, (row, col)), shape=(3, 3))
        I_RoI = _compute_global_pseudodistribution(I, {"matrix": [3, 3]})

        assert I_RoI.max() >= 0.9

    def test_is_smoothed(self):
        row = np.array([0, 1, 2])
        col = np.array([0, 1, 1])
        data = np.array([9.0, 2.0, 4.0])
        I = ss.csr_matrix((data, (row, col)), shape=(3, 3))
        I_RoI = _compute_global_pseudodistribution(I, {"matrix": [3, 3]})

        assert I_RoI.max() <= 1.0


@pytest.mark.unit
class TestCheckNeighborhood:
    @pytest.fixture(scope="function")
    def correct_mask(self):
        yield [0] * 20 + [1] * 70 + [0] * 10

    @pytest.fixture(scope="function")
    def generic_setup(self):
        yield (np.linspace(0.0, 29.0, 30), 5.0, 4, 0.5)

    # TODO: The vector can be a lot smaller than this. Just calculate the resulting values, and you get the correct result.
    @pytest.fixture(scope="class")
    def values(self):
        yield np.linspace(0.0, 99.0, 100)

    @pytest.fixture(scope="class")
    def min_value(self):
        yield 20.0

    @pytest.fixture(scope="class")
    def neighborhood_size(self):
        yield 10

    @pytest.fixture(scope="class")
    def threshold_percentage(self):
        yield 0.5

    # TODO: test_min_value_too_low_for_params and test_threshold_percentage_too_low modifies to the same list. Change this, so that every output is unique
    # TODO: Verify neighborhood size shouldn't be more than half the size of the values array
    def test_correct_output(self, values, min_value, neighborhood_size, threshold_percentage, correct_mask):

        mask = _check_neighborhood(values, min_value, neighborhood_size, threshold_percentage)

        assert mask == correct_mask

    def test_min_value_too_low_to_run(self, values, min_value, neighborhood_size, threshold_percentage, correct_mask):

        with pytest.raises(AssertionError):
            mask = _check_neighborhood(
                values=values,
                min_value=-1.0,
                neighborhood_size=neighborhood_size,
                threshold_percentage=threshold_percentage,
            )

        # assert mask == correct_mask

    def test_min_value_too_low_for_params(
        self, values, min_value, neighborhood_size, threshold_percentage, correct_mask
    ):

        mask = _check_neighborhood(
            values=values,
            min_value=19.0,
            neighborhood_size=neighborhood_size,
            threshold_percentage=threshold_percentage,
        )
        correct_mask[19] = 1

        assert mask == correct_mask

    def test_min_value_too_large_for_params(
        self, values, min_value, neighborhood_size, threshold_percentage, correct_mask
    ):

        mask = _check_neighborhood(
            values=values,
            min_value=21.0,
            neighborhood_size=neighborhood_size,
            threshold_percentage=threshold_percentage,
        )
        correct_mask[20] = 0

        assert mask == correct_mask

    def test_neighborhood_size_too_low(self, values, min_value, neighborhood_size, threshold_percentage, correct_mask):

        mask = _check_neighborhood(
            values=values, min_value=min_value, neighborhood_size=9, threshold_percentage=threshold_percentage
        )
        correct_mask[90] = 1

        assert mask == correct_mask

    def test_neighborhood_size_too_large(
        self, values, min_value, neighborhood_size, threshold_percentage, correct_mask
    ):

        mask = _check_neighborhood(
            values=values, min_value=min_value, neighborhood_size=11, threshold_percentage=threshold_percentage
        )
        correct_mask[89] = 0

        assert mask == correct_mask

    def test_threshold_percentage_too_low(
        self, values, min_value, neighborhood_size, threshold_percentage, correct_mask
    ):

        mask = _check_neighborhood(
            values=values, min_value=min_value, neighborhood_size=neighborhood_size, threshold_percentage=0.4
        )
        correct_mask[18] = 1
        correct_mask[19] = 1

        assert mask == correct_mask

    def test_threshold_percentage_too_large(
        self, values, min_value, neighborhood_size, threshold_percentage, correct_mask
    ):

        mask = _check_neighborhood(
            values=values, min_value=min_value, neighborhood_size=neighborhood_size, threshold_percentage=0.6
        )
        correct_mask[20] = 0
        correct_mask[21] = 0

        assert mask == correct_mask
