# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import pytest
import numpy as np
import scipy.sparse as ss

from stripepy.stripepy import _log_transform, _band_extraction, _scale_Iproc, _extract_RoIs, _compute_global_pseudodistribution, _check_neighborhood


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

@pytest.mark.unit
class TestScaleIProc:
    def test_divide_by_one(self):
        indices = [0, 1, 2, 3]
        data = [1]*len(indices)
        I = ss.csr_matrix((data, (indices, indices)))
        LT_I, UT_I = _band_extraction(I, 1, 5)
        I, LT_I, UT_I = _scale_Iproc(I, LT_I, UT_I)

        assert (I.diagonal(0) == np.array([1, 1, 1, 1])).all()
    
    def test_halve(self):
        indices = [0, 1, 2, 3]
        data = [1, 2, 1, 1]
        I = ss.csr_matrix((data, (indices, indices)))
        LT_I, UT_I = _band_extraction(I, 1, 5)
        I, LT_I, UT_I = _scale_Iproc(I, LT_I, UT_I)
        assert (I.diagonal(0) == np.array([0.5, 1, 0.5, 0.5])).all()
        
@pytest.mark.unit
class TestExtractRoIs:
    def test_is_NDarray(self):
        I = ss.rand(10, 10, density=0.5, format="csr")
        I_RoI = _extract_RoIs(I, {"matrix": [2,5]})

        assert isinstance(I_RoI, np.ndarray)
    
    def test_is_square(self):
        I = ss.rand(10, 10, density=0.5, format="csr")
        I_RoI = _extract_RoIs(I, {"matrix": [2,5]})

        assert (I_RoI.size ** 0.5).is_integer()

@pytest.mark.unit
class TestComputeGlobalPseudodistribution:
    #TODO: implement generator function for matrix
    """
    The input matrix looks like
    |   9   0   0   |
    |   0   2   0   |
    |   0   4   0   |
    """
    @pytest.mark.skip(reason="Outgoing matrix is not necessarily sparse")
    def test_is_sparse(self):
        I = ss.rand(10, 10, density=0.5, format="csr")
        I_RoI = _compute_global_pseudodistribution(I, {"matrix": [2,5]})

        assert ss.issparse(I_RoI)

    def test_is_marginalized(self):
        row = np.array([0, 1, 2])
        col = np.array([0, 1, 1])
        data = np.array([9.0, 2.0, 4.0])
        I = ss.csr_matrix((data, (row, col)), shape=(3, 3))
        I_RoI = _compute_global_pseudodistribution(I, {"matrix": [3,3]})

        assert I_RoI.shape == (3,)

    def test_is_scaled(self):
        row = np.array([0, 1, 2])
        col = np.array([0, 1, 1])
        data = np.array([9.0, 2.0, 4.0])
        I = ss.csr_matrix((data, (row, col)), shape=(3, 3))
        I_RoI = _compute_global_pseudodistribution(I, {"matrix": [3,3]})

        assert I_RoI.max() >= 0.9

    def test_is_smoothed(self):
        row = np.array([0, 1, 2])
        col = np.array([0, 1, 1])
        data = np.array([9.0, 2.0, 4.0])
        I = ss.csr_matrix((data, (row, col)), shape=(3, 3))
        I_RoI = _compute_global_pseudodistribution(I, {"matrix": [3,3]})

        assert I_RoI.max() <= 1.0

@pytest.mark.unit
class TestCheckNeighborhood:
    @pytest.fixture(scope="function")
    def correct_mask(self):
        yield [0] * 20 + [1] * 70 + [0] * 10
    
    @pytest.fixture(scope="function")
    def generic_setup(self):
        yield (
            np.linspace(0.0, 29.0, 30), 
            5.0, 
            4, 
            0.5
            )
    
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
    
    #TODO: test_min_value_too_low_for_params and test_threshold_percentage_too_low modifies to the same list. Change this, so that every output is unique
    def test_correct_output(self, values, min_value, neighborhood_size, threshold_percentage, correct_mask):

        mask = _check_neighborhood(values, min_value, neighborhood_size, threshold_percentage)

        assert mask == correct_mask
    
    def test_min_value_too_low_to_run(self, values, min_value, neighborhood_size, threshold_percentage, correct_mask):

        with pytest.raises(AssertionError):
            mask = _check_neighborhood(
                values=values, 
                min_value=-1.0, 
                neighborhood_size=neighborhood_size, 
                threshold_percentage=threshold_percentage)

        
        #assert mask == correct_mask

    def test_min_value_too_low_for_params(self, values, min_value, neighborhood_size, threshold_percentage, correct_mask):
        
        mask = _check_neighborhood(
            values=values, 
            min_value=19.0, 
            neighborhood_size=neighborhood_size, 
            threshold_percentage=threshold_percentage)
        correct_mask[19] = 1

        assert mask == correct_mask

    def test_min_value_too_large_for_params(self, values, min_value, neighborhood_size, threshold_percentage, correct_mask):
        
        mask = _check_neighborhood(
            values=values, 
            min_value=21.0, 
            neighborhood_size=neighborhood_size, 
            threshold_percentage=threshold_percentage)
        correct_mask[20] = 0

        assert mask == correct_mask

    def test_neighborhood_size_too_low(self, values, min_value, neighborhood_size, threshold_percentage, correct_mask):

        mask = _check_neighborhood(
            values=values, 
            min_value=min_value, 
            neighborhood_size=9, 
            threshold_percentage=threshold_percentage)
        correct_mask[90] = 1

        assert mask == correct_mask

    def test_neighborhood_size_too_large(self, values, min_value, neighborhood_size, threshold_percentage, correct_mask):

        mask = _check_neighborhood(
            values=values, 
            min_value=min_value, 
            neighborhood_size=11, 
            threshold_percentage=threshold_percentage)
        correct_mask[89] = 0

        assert mask == correct_mask

    def test_threshold_percentage_too_low(self, values, min_value, neighborhood_size, threshold_percentage, correct_mask):

        mask = _check_neighborhood(
            values=values, 
            min_value=min_value, 
            neighborhood_size=neighborhood_size, 
            threshold_percentage=0.4)
        correct_mask[18] = 1
        correct_mask[19] = 1

        assert mask == correct_mask


    def test_threshold_percentage_too_large(self, values, min_value, neighborhood_size, threshold_percentage, correct_mask):

        mask = _check_neighborhood(
            values=values, 
            min_value=min_value, 
            neighborhood_size=neighborhood_size, 
            threshold_percentage=0.6)
        correct_mask[20] = 0
        correct_mask[21] = 0

        assert mask == correct_mask

class TestFilterExtremaBySparseness:
    pass