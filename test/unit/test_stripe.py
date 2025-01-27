import math
import re

import numpy as np
import pytest
import scipy.sparse as ss

from stripepy.utils.stripe import Stripe


@pytest.fixture(scope="function")
def U_stripe():
    stripe = Stripe(
        seed=5,
        top_pers=5.0,
        horizontal_bounds=(4, 6),
        vertical_bounds=(1, 4),
        where="upper_triangular",
    )
    return stripe


@pytest.fixture(scope="function")
def L_stripe():
    stripe = Stripe(
        seed=5,
        top_pers=5.0,
        horizontal_bounds=(4, 6),
        vertical_bounds=(4, 10),
        where="lower_triangular",
    )
    return stripe


@pytest.mark.unit
class TestObjectInitialization:
    # TODO: Try to pass None-values and other types of values
    def test_constructor(self):
        stripe = Stripe(
            seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(1, 4), where="upper_triangular"
        )
        assert stripe.seed == 5
        assert np.isclose(stripe.top_persistence, 5.0)
        assert not stripe.lower_triangular
        assert stripe.upper_triangular
        assert stripe.left_bound == 4
        assert stripe.right_bound == 6
        assert stripe.top_bound == 1
        assert stripe.bottom_bound == 4


@pytest.mark.unit
class TestPropertyBoundaries:
    class TestSeed:
        def test_seed_at_matrix_border(self):
            stripe = Stripe(seed=0, top_pers=None, horizontal_bounds=None, vertical_bounds=None, where=None)

        def test_seed_outside_matrix(self):
            with pytest.raises(ValueError, match="seed must be a non-negative integral number"):
                stripe = Stripe(seed=-1, top_pers=None, horizontal_bounds=None, vertical_bounds=None, where=None)

        def test_seed_inside_matrix(self):
            stripe = Stripe(seed=5, top_pers=None, horizontal_bounds=None, vertical_bounds=None, where=None)

        def test_seed_none(self):
            with pytest.raises(
                TypeError, match=re.escape(r"'<' not supported between instances of 'NoneType' and 'int'")
            ):
                stripe = Stripe(
                    seed=None, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(1, 4), where="upper_triangular"
                )

    class TestTopPersistence:
        def test_top_persistence_1(self):
            stripe = Stripe(seed=0, top_pers=1.0, horizontal_bounds=None, vertical_bounds=None, where=None)

        def test_top_persistence_0(self):
            stripe = Stripe(seed=0, top_pers=0.0, horizontal_bounds=None, vertical_bounds=None, where=None)

        def test_top_persistence_negative(self):
            with pytest.raises(ValueError, match="when not None, top_pers must be a positive number"):
                stripe = Stripe(seed=0, top_pers=-1, horizontal_bounds=None, vertical_bounds=None, where=None)

        def top_persistence_higher_value(self):
            stripe = Stripe(seed=0, top_pers=100.0, horizontal_bounds=None, vertical_bounds=None, where=None)

    class TestSetHorizontalRelativeToSeed:
        def test_left_at_seed(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(5, 8), vertical_bounds=(1, 5), where="upper_triangular"
            )

        def test_right_at_seed(self):
            stripe = Stripe(
                seed=6, top_pers=5.0, horizontal_bounds=(3, 6), vertical_bounds=(1, 5), where="upper_triangular"
            )

        def test_left_and_right_at_seed(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(5, 5), vertical_bounds=(1, 5), where="upper_triangular"
            )

        def test_seed_adjacent_to_left(self):
            stripe = Stripe(
                seed=6, top_pers=5.0, horizontal_bounds=(5, 8), vertical_bounds=(1, 6), where="upper_triangular"
            )

        def test_seed_adjacent_to_right(self):
            stripe = Stripe(
                seed=6, top_pers=5.0, horizontal_bounds=(5, 7), vertical_bounds=(1, 5), where="upper_triangular"
            )

        def test_seed_adjacent_to_left_and_right(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(1, 4), where="upper_triangular"
            )

        def test_seed_to_left_of_boundary(self):
            with pytest.raises(
                ValueError,
                match="horizontal bounds must enclose the seed position: seed=5, left_bound=6, right_bound=6",
            ) as e:
                stripe = Stripe(
                    seed=5, top_pers=5.0, horizontal_bounds=(6, 6), vertical_bounds=(1, 4), where="upper_triangular"
                )

        def test_seed_to_right_of_boundary(self):
            with pytest.raises(
                ValueError,
                match="horizontal bounds must enclose the seed position: seed=5, left_bound=4, right_bound=4",
            ) as e:
                u_stripe = Stripe(
                    seed=5, top_pers=5.0, horizontal_bounds=(4, 4), vertical_bounds=(1, 4), where="upper_triangular"
                )

    class SetHorizontalRelativeToSelf:
        def test_horizontal_size_0(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(5, 5), vertical_bounds=(1, 4), where="upper_triangular"
            )

        def test_horizontal_size_1(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(5, 6), vertical_bounds=(1, 4), where="upper_triangular"
            )

        def test_left_and_right_cross_themselves(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(6, 5), vertical_bounds=(1, 4), where="upper_triangular"
            )

    class TestSetHorizontalRelativeToMatrixEdges:
        def test_left_at_matrix_edge(self):
            stripe = Stripe(
                seed=1, top_pers=5.0, horizontal_bounds=(0, 1), vertical_bounds=(1, 10), where="lower_triangular"
            )

        def test_left_and_right_at_matrix_edge(self):
            stripe = Stripe(
                seed=0, top_pers=5.0, horizontal_bounds=(0, 0), vertical_bounds=(0, 10), where="lower_triangular"
            )

        def test_left_over_matrix_edge(self):
            with pytest.raises(ValueError, match="stripe bounds must be positive integers"):
                stripe = Stripe(
                    seed=1, top_pers=5.0, horizontal_bounds=(-1, 1), vertical_bounds=(1, 10), where="lower_triangular"
                )

    class TestSetHorizontalWhenAlreadySet:
        def test_horizontal_bounds_already_set(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(1, 4), where="upper_triangular"
            )

            with pytest.raises(RuntimeError, match="horizontal stripe bounds have already been set"):
                stripe.set_horizontal_bounds(5, 7)

    class TestSetVerticalRelativeToSelf:
        def test_vertical_size_0(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(5, 5), where="upper_triangular"
            )

        def test_vertical_size_1(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(4, 5), where="upper_triangular"
            )

        def test_top_and_bottom_cross_themselves(self):
            with pytest.raises(
                ValueError,
                match="the lower vertical bound must be greater than the upper vertical bound: top_bound=5, bottom_bound=4",
            ):
                stripe = Stripe(
                    seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(5, 4), where="upper_triangular"
                )

    class TestSetVerticalRelativeToMatrixEdges:
        def test_top_at_matrix_edge(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(0, 5), where="upper_triangular"
            )

        def test_top_and_bottom_at_matrix_edge(self):
            stripe = Stripe(
                seed=1, top_pers=5.0, horizontal_bounds=(0, 3), vertical_bounds=(0, 0), where="upper_triangular"
            )

        def test_top_over_matrix_edge(self):
            with pytest.raises(ValueError, match="stripe bounds must be positive integers"):
                stripe = Stripe(
                    seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(-1, 5), where="upper_triangular"
                )

        def test_top_and_bottom_over_matrix_edge(self):
            with pytest.raises(ValueError, match="stripe bounds must be positive integers"):
                stripe = Stripe(
                    seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(-1, -1), where="upper_triangular"
                )

    class TestSetVerticalWhenAlreadySet:
        def test_vertical_bounds_already_set(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(1, 4), where="upper_triangular"
            )

            with pytest.raises(RuntimeError, match="vertical stripe bounds have already been set"):
                stripe.set_vertical_bounds(2, 6)

    class TestWhere:
        def test_where_upper(self):
            u_stripe = Stripe(
                seed=5, top_pers=None, horizontal_bounds=None, vertical_bounds=(1, 4), where="upper_triangular"
            )

        def test_where_lower(self):
            l_stripe = Stripe(
                seed=5, top_pers=None, horizontal_bounds=None, vertical_bounds=(4, 10), where="lower_triangular"
            )

        def test_where_none_in_upper(self):
            u_stripe = Stripe(seed=5, top_pers=None, horizontal_bounds=None, vertical_bounds=(1, 4), where=None)

            assert u_stripe.upper_triangular

        def test_where_none_in_lower(self):
            l_stripe = Stripe(seed=5, top_pers=None, horizontal_bounds=None, vertical_bounds=(4, 10), where=None)

            assert l_stripe.lower_triangular

        def test_where_invalid_input(self):
            with pytest.raises(
                ValueError,
                match=re.escape(r"when specified, where must be one of ('upper_triangular', 'lower_triangular')"),
            ) as e:
                u_stripe = Stripe(
                    seed=5, top_pers=None, horizontal_bounds=None, vertical_bounds=(1, 4), where="invalid_triangular"
                )

            with pytest.raises(
                ValueError,
                match=re.escape(r"when specified, where must be one of ('upper_triangular', 'lower_triangular')"),
            ) as e:
                l_stripe = Stripe(
                    seed=5, top_pers=None, horizontal_bounds=None, vertical_bounds=(4, 10), where="invalid_triangular"
                )

    class TestSetterMethods:
        def test_set_horizontal(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=None, vertical_bounds=(1, 4), where="upper_triangular"
            )
            stripe.set_horizontal_bounds(3, 7)

            assert stripe.left_bound == 3
            assert stripe.right_bound == 7

        def test_set_vertical(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=None, where="upper_triangular"
            )
            stripe.set_vertical_bounds(2, 5)

            assert stripe.top_bound == 2
            assert stripe.bottom_bound == 5


@pytest.mark.unit
class TestComputeBiodescriptors:
    #####
    ### Confirmation tests
    #####
    def test_compute_statistics(self):
        stripe = Stripe(
            seed=5,
            top_pers=5.0,
            horizontal_bounds=(4, 6),
            vertical_bounds=(1, 4),
            where="upper_triangular",
        )
        matrix = ss.csr_matrix(
            np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 2, 0, 0, 1, 2, 0, 0, 0, 0, 0],
                    [0, 0, 3, 0, 3, 4, 0, 0, 0, 0, 0],
                    [0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
                ]
            )
        )
        stripe.compute_biodescriptors(matrix)
        """
        horizontal_bound = (4,6)
        vertical_bounds = (1,4)
        convex_comp = 4
        rows = slice(1, 4)
        cols = slice(4, 6)
        restI =     |   1   2   |
                    |   3   4   |
                    |   0   0   |
        restI.size = 6
        _compute_inner_descriptors(restrI) = (array([0.0, 0.25, 1.5, 2.75, 4.0]), 1.6666666666666667, 1.4907119849998596)
        five_number = array([0.0, 0.25, 1.5, 2.75, 4.0]
        inner_mean = 1.6666666666666667
        inner_std = 1.4907119849998596
        submatrix = I[1:3, 1:4] =   |   2   0   0   |
                                    |   0   3   0   |
                                    |   0   0   4   |
        outer_lmean = 1.0
        outer_rmean = 0.0
        """

        assert np.allclose(stripe._five_number, np.array([0.0, 0.25, 1.5, 2.75, 4.0]))
        assert np.isclose(stripe._inner_mean, 1.6666666666666667, atol=1e-16)
        assert np.isclose(stripe._inner_std, 1.4907119849998596, atol=1e-16)
        assert np.isclose(stripe._outer_lmean, 1.0)
        assert np.isclose(stripe._outer_rmean, 0.0)

    def test_compute_stripe_size_0(self):
        stripe = Stripe(
            seed=0,
            top_pers=1.0,
            horizontal_bounds=(0, 0),
            vertical_bounds=(0, 0),
            where="upper_triangular",
        )
        matrix = ss.csr_matrix(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0],
                    [0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0],
                    [0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
                ]
            )
        )
        stripe.compute_biodescriptors(matrix)
        assert np.allclose(stripe._five_number, np.array([-1.0, -1.0, -1.0, -1.0, -1.0]))
        assert np.isclose(stripe._inner_mean, -1.0, atol=1e-5)
        assert np.isclose(stripe._inner_std, -1.0, atol=1e-5)
        assert np.isclose(stripe._outer_lmean, -1.0, atol=1e-5)
        assert np.isclose(stripe._outer_rmean, -1.0, atol=1e-5)

    #####
    ### Boundary tests
    #####
    class TestComputeAtLeftEdge:
        def test_compute_empty_stripe(self):
            stripe = Stripe(
                seed=1,
                top_pers=1.0,
                horizontal_bounds=(0, 3),
                vertical_bounds=(1, 4),
                where="lower_triangular",
            )
            matrix = ss.csr_matrix(
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0],
                        [0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0],
                        [0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
                    ]
                )
            )
            stripe.compute_biodescriptors(matrix)
            assert np.allclose(stripe._five_number, np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
            assert np.isclose(stripe._inner_mean, 0.0, atol=1e-5)
            assert np.isclose(stripe._inner_std, 0.0, atol=1e-5)
            assert math.isnan(stripe._outer_lmean)
            assert np.isclose(stripe._outer_rmean, 3.0, atol=1e-5)

        def test_compute_with_empty_neighbourhoods(self):
            stripe = Stripe(
                seed=1,
                top_pers=4.0,
                horizontal_bounds=(0, 3),
                vertical_bounds=(1, 4),
                where="lower_triangular",
            )
            matrix = ss.csr_matrix(
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0],
                        [2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0],
                        [2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
                    ]
                )
            )
            stripe.compute_biodescriptors(matrix)
            assert np.allclose(stripe._five_number, np.array([2.0, 2.0, 3.0, 4.0, 4.0]))
            assert np.isclose(stripe._inner_mean, 3.0, atol=1e-5)
            assert np.isclose(stripe._inner_std, 0.816496580927726, atol=1e-15)
            assert math.isnan(stripe._outer_lmean)
            assert np.isclose(stripe._outer_rmean, 0.0, atol=1e-5)

        def test_compute_empty_stripe_with_empty_neighbourhoods(self):
            stripe = Stripe(
                seed=1,
                top_pers=4.0,
                horizontal_bounds=(0, 3),
                vertical_bounds=(1, 4),
                where="lower_triangular",
            )
            matrix = ss.csr_matrix(
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
                    ]
                )
            )
            stripe.compute_biodescriptors(matrix)
            assert np.allclose(stripe._five_number, np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
            assert np.isclose(stripe._inner_mean, 0.0, atol=1e-5)
            assert np.isclose(stripe._inner_std, 0.0, atol=1e-5)
            assert math.isnan(stripe._outer_lmean)
            assert np.isclose(stripe._outer_rmean, 0.0, atol=1e-5)

        def test_compute_stripe_with_denser_neighbourhoods(self):
            stripe = Stripe(
                seed=1,
                top_pers=4.0,
                horizontal_bounds=(0, 3),
                vertical_bounds=(1, 4),
                where="lower_triangular",
            )
            matrix = ss.csr_matrix(
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [2, 3, 2, 5, 5, 5, 0, 0, 0, 0, 0],
                        [2, 3, 2, 5, 5, 5, 0, 0, 0, 0, 0],
                        [1, 2, 1, 5, 5, 5, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
                    ]
                )
            )
            stripe.compute_biodescriptors(matrix)
            assert np.allclose(stripe._five_number, np.array([1.0, 2.0, 2.0, 2.0, 3.0]))
            assert np.isclose(stripe._inner_mean, 2.0, atol=1e-5)
            assert np.isclose(stripe._inner_std, 0.6666666666666666, atol=1e-16)
            assert math.isnan(stripe._outer_lmean)
            assert np.isclose(stripe._outer_rmean, 5.0, atol=1e-5)

        def test_compute_stripe_with_uniform_neighbourhoods(self):
            stripe = Stripe(
                seed=1,
                top_pers=4.0,
                horizontal_bounds=(0, 3),
                vertical_bounds=(1, 4),
                where="lower_triangular",
            )
            matrix = ss.csr_matrix(
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0],
                        [5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0],
                        [5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
                    ]
                )
            )
            stripe.compute_biodescriptors(matrix)
            assert np.allclose(stripe._five_number, np.array([5.0, 5.0, 5.0, 5.0, 5.0]))
            assert np.isclose(stripe._inner_mean, 5.0, atol=1e-5)
            assert np.isclose(stripe._inner_std, 0.0, atol=1e-5)
            assert math.isnan(stripe._outer_lmean)
            assert np.isclose(stripe._outer_rmean, 5.0, atol=1e-5)

    class TestComputeAtRightEdge:
        def test_compute_empty_stripe(self):
            stripe = Stripe(
                seed=10,
                top_pers=4.0,
                horizontal_bounds=(8, 11),
                vertical_bounds=(6, 10),
                where="upper_triangular",
            )
            matrix = ss.csr_matrix(
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 6, 0, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0, 6, 6, 6, 0, 0, 0],
                        [0, 0, 0, 0, 0, 6, 6, 6, 0, 0, 0],
                        [0, 0, 0, 0, 0, 6, 6, 6, 0, 0, 0],
                        [0, 0, 0, 0, 0, 6, 6, 6, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    ]
                )
            )
            stripe.compute_biodescriptors(matrix)
            assert np.allclose(stripe._five_number, np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
            assert np.isclose(stripe._inner_mean, 0.0, atol=1e-5)
            assert np.isclose(stripe._inner_std, 0.0, atol=1e-5)
            assert np.isclose(stripe._outer_lmean, 6.0, atol=1e-5)
            assert math.isnan(stripe._outer_rmean)

        def test_compute_with_empty_neighbourhoods(self):
            stripe = Stripe(
                seed=10,
                top_pers=4.0,
                horizontal_bounds=(8, 11),
                vertical_bounds=(6, 10),
                where="upper_triangular",
            )
            matrix = ss.csr_matrix(
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 6, 0, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 5, 4, 5],
                        [0, 0, 0, 0, 0, 0, 0, 0, 6, 5, 6],
                        [0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 7],
                        [0, 0, 0, 0, 0, 0, 0, 0, 8, 7, 8],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    ]
                )
            )
            stripe.compute_biodescriptors(matrix)
            assert np.allclose(stripe._five_number, np.array([4.0, 5.0, 6.0, 7.0, 8.0]))
            assert np.isclose(stripe._inner_mean, 6.166666666666667, atol=1e-5)
            assert np.isclose(stripe._inner_std, 1.2133516482134197, atol=1e-5)
            assert np.isclose(stripe._outer_lmean, 0.0, atol=1e-5)
            assert math.isnan(stripe._outer_rmean)

        def test_compute_empty_stripe_with_empty_neighbourhoods(self):
            stripe = Stripe(
                seed=10,
                top_pers=4.0,
                horizontal_bounds=(8, 11),
                vertical_bounds=(6, 10),
                where="upper_triangular",
            )
            matrix = ss.csr_matrix(
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 6, 0, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    ]
                )
            )
            stripe.compute_biodescriptors(matrix)
            assert np.allclose(stripe._five_number, np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
            assert np.isclose(stripe._inner_mean, 0.0, atol=1e-5)
            assert np.isclose(stripe._inner_std, 0.0, atol=1e-5)
            assert np.isclose(stripe._outer_lmean, 0.0, atol=1e-5)
            assert math.isnan(stripe._outer_rmean)

        def test_compute_stripe_with_denser_neighbourhoods(self):
            stripe = Stripe(
                seed=10,
                top_pers=4.0,
                horizontal_bounds=(8, 11),
                vertical_bounds=(6, 10),
                where="upper_triangular",
            )
            matrix = ss.csr_matrix(
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 6, 0, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 6, 5, 4],
                        [0, 0, 0, 0, 0, 0, 0, 0, 6, 5, 4],
                        [0, 0, 0, 0, 0, 0, 0, 0, 6, 5, 4],
                        [0, 0, 0, 0, 0, 0, 0, 0, 6, 5, 4],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    ]
                )
            )
            stripe.compute_biodescriptors(matrix)
            assert np.allclose(stripe._five_number, np.array([4.0, 4.0, 5.0, 6.0, 6.0]))
            assert np.isclose(stripe._inner_mean, 5.0, atol=1e-5)
            assert np.isclose(stripe._inner_std, 0.816496580927726, atol=1e-5)
            assert np.isclose(stripe._outer_lmean, 0.0, atol=1e-5)
            assert math.isnan(stripe._outer_rmean)

        def test_compute_stripe_with_uniform_neighbourhoods(self):
            stripe = Stripe(
                seed=10,
                top_pers=4.0,
                horizontal_bounds=(8, 11),
                vertical_bounds=(6, 10),
                where="upper_triangular",
            )
            matrix = ss.csr_matrix(
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 6, 0, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6],
                        [0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6],
                        [0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6],
                        [0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    ]
                )
            )
            stripe.compute_biodescriptors(matrix)
            assert np.allclose(stripe._five_number, np.array([6.0, 6.0, 6.0, 6.0, 6.0]))
            assert np.isclose(stripe._inner_mean, 6.0, atol=1e-5)
            assert np.isclose(stripe._inner_std, 0.0, atol=1e-5)
            assert np.isclose(stripe._outer_lmean, 6.0, atol=1e-5)
            assert math.isnan(stripe._outer_rmean)

    class TestCopmuteAtMiddleUpperTriangle:
        def test_compute_empty_stripe(self):
            stripe = Stripe(
                seed=5,
                top_pers=5.0,
                horizontal_bounds=(4, 7),
                vertical_bounds=(2, 5),
                where="upper_triangular",
            )
            matrix = ss.csr_matrix(
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 2, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 5, 6, 7, 0, 0, 0, 7, 6, 5, 0],
                        [0, 4, 5, 6, 0, 0, 0, 6, 5, 4, 0],
                        [0, 3, 4, 5, 0, 0, 0, 5, 4, 3, 0],
                        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
                    ]
                )
            )
            stripe.compute_biodescriptors(matrix)
            assert np.allclose(stripe._five_number, np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
            assert np.isclose(stripe._inner_mean, 0.0, atol=1e-5)
            assert np.isclose(stripe._inner_std, 0.0, atol=1e-5)
            assert np.isclose(stripe._outer_lmean, 5.0, atol=1e-5)
            assert np.isclose(stripe._outer_rmean, 5.0, atol=1e-5)

        def test_compute_with_empty_neighbourhoods(self):
            stripe = Stripe(
                seed=5,
                top_pers=5.0,
                horizontal_bounds=(4, 7),
                vertical_bounds=(2, 5),
                where="upper_triangular",
            )
            matrix = ss.csr_matrix(
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 2, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 7, 8, 7, 0, 0, 0, 0],
                        [0, 0, 0, 0, 8, 9, 8, 0, 0, 0, 0],
                        [0, 0, 0, 0, 7, 8, 7, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
                    ]
                )
            )
            stripe.compute_biodescriptors(matrix)
            assert np.allclose(stripe._five_number, np.array([7.0, 7.0, 8.0, 8.0, 9.0]))
            assert np.isclose(stripe._inner_mean, 7.666666666666667, atol=1e-15)
            assert np.isclose(stripe._inner_std, 0.6666666666666666, atol=1e-15)
            assert np.isclose(stripe._outer_lmean, 0.0, atol=1e-5)
            assert np.isclose(stripe._outer_rmean, 0.0, atol=1e-5)

        def test_compute_empty_stripe_with_empty_neighbourhoods(self):
            stripe = Stripe(
                seed=5,
                top_pers=5.0,
                horizontal_bounds=(4, 7),
                vertical_bounds=(2, 5),
                where="upper_triangular",
            )
            matrix = ss.csr_matrix(
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 2, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
                    ]
                )
            )
            stripe.compute_biodescriptors(matrix)
            assert np.allclose(stripe._five_number, np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
            assert np.isclose(stripe._inner_mean, 0.0, atol=1e-5)
            assert np.isclose(stripe._inner_std, 0.0, atol=1e-5)
            assert np.isclose(stripe._outer_lmean, 0.0, atol=1e-5)
            assert np.isclose(stripe._outer_rmean, 0.0, atol=1e-5)

        def test_compute_stripe_with_denser_neighbourhoods(self):
            stripe = Stripe(
                seed=5,
                top_pers=5.0,
                horizontal_bounds=(4, 7),
                vertical_bounds=(2, 5),
                where="upper_triangular",
            )
            matrix = ss.csr_matrix(
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 2, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 7, 8, 7, 4, 5, 4, 7, 8, 7, 0],
                        [0, 8, 9, 8, 5, 6, 5, 8, 9, 8, 0],
                        [0, 7, 8, 7, 4, 5, 4, 7, 8, 7, 0],
                        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
                    ]
                )
            )
            stripe.compute_biodescriptors(matrix)
            assert np.allclose(stripe._five_number, np.array([4.0, 4.0, 5.0, 5.0, 6.0]))
            assert np.isclose(stripe._inner_mean, 4.666666666666667, atol=1e-15)
            assert np.isclose(stripe._inner_std, 0.6666666666666666, atol=1e-15)
            assert np.isclose(stripe._outer_lmean, 7.666666666666667, atol=1e-15)
            assert np.isclose(stripe._outer_rmean, 7.666666666666667, atol=1e-15)

        def test_compute_stripe_with_uniform_neighbourhoods(self):
            stripe = Stripe(
                seed=5,
                top_pers=5.0,
                horizontal_bounds=(4, 7),
                vertical_bounds=(2, 5),
                where="upper_triangular",
            )
            matrix = ss.csr_matrix(
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 2, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
                        [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
                        [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
                        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
                    ]
                )
            )
            stripe.compute_biodescriptors(matrix)
            assert np.allclose(stripe._five_number, np.array([5.0, 5.0, 5.0, 5.0, 5.0]))
            assert np.isclose(stripe._inner_mean, 5.0, atol=1e-5)
            assert np.isclose(stripe._inner_std, 0.0, atol=1e-15)
            assert np.isclose(stripe._outer_lmean, 5.0, atol=1e-15)
            assert np.isclose(stripe._outer_rmean, 5.0, atol=1e-15)

    class TestComputeAtMiddleLowerTriangle:
        def test_compute_empty_stripe(self):
            stripe = Stripe(
                seed=4,
                top_pers=5.0,
                horizontal_bounds=(4, 7),
                vertical_bounds=(4, 7),
                where="lower_triangular",
            )
            matrix = ss.csr_matrix(
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 4, 1, 1, 1, 0, 0, 0, 0],
                        [0, 4, 5, 4, 0, 0, 0, 4, 5, 4, 0],
                        [0, 5, 6, 5, 0, 0, 0, 5, 6, 5, 0],
                        [0, 4, 5, 4, 0, 0, 0, 4, 5, 4, 0],
                        [0, 0, 0, 0, 1, 1, 1, 8, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
                    ]
                )
            )
            stripe.compute_biodescriptors(matrix)
            assert np.allclose(stripe._five_number, np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
            assert np.isclose(stripe._inner_mean, 0.0, atol=1e-5)
            assert np.isclose(stripe._inner_std, 0.0, atol=1e-5)
            assert np.isclose(stripe._outer_lmean, 4.666666666666667, atol=1e-15)
            assert np.isclose(stripe._outer_rmean, 4.666666666666667, atol=1e-15)

        def test_compute_with_empty_neighbourhoods(self):
            stripe = Stripe(
                seed=4,
                top_pers=5.0,
                horizontal_bounds=(4, 7),
                vertical_bounds=(4, 7),
                where="lower_triangular",
            )
            matrix = ss.csr_matrix(
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 4, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0],
                        [0, 0, 0, 0, 8, 9, 8, 0, 0, 0, 0],
                        [0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 1, 8, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
                    ]
                )
            )
            stripe.compute_biodescriptors(matrix)
            assert np.allclose(stripe._five_number, np.array([7.0, 7.0, 7.0, 8.0, 9.0]))
            assert np.isclose(stripe._inner_mean, 7.444444444444445, atol=1e-15)
            assert np.isclose(stripe._inner_std, 0.6849348892187751, atol=1e-15)
            assert np.isclose(stripe._outer_lmean, 0.0, atol=1e-5)
            assert np.isclose(stripe._outer_rmean, 0.0, atol=1e-5)

        def test_compute_empty_stripe_with_empty_neighbourhoods(self):
            stripe = Stripe(
                seed=4,
                top_pers=5.0,
                horizontal_bounds=(4, 7),
                vertical_bounds=(4, 7),
                where="lower_triangular",
            )
            matrix = ss.csr_matrix(
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 4, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 1, 8, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
                    ]
                )
            )
            stripe.compute_biodescriptors(matrix)
            assert np.allclose(stripe._five_number, np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
            assert np.isclose(stripe._inner_mean, 0.0, atol=1e-5)
            assert np.isclose(stripe._inner_std, 0.0, atol=1e-5)
            assert np.isclose(stripe._outer_lmean, 0.0, atol=1e-5)
            assert np.isclose(stripe._outer_rmean, 0.0, atol=1e-5)

        def test_compute_stripe_with_denser_neighbourhoods(self):
            stripe = Stripe(
                seed=4,
                top_pers=5.0,
                horizontal_bounds=(4, 7),
                vertical_bounds=(4, 7),
                where="lower_triangular",
            )
            matrix = ss.csr_matrix(
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 4, 1, 1, 1, 0, 0, 0, 0],
                        [0, 4, 5, 4, 2, 1, 2, 4, 5, 4, 0],
                        [0, 5, 6, 5, 1, 3, 1, 5, 6, 5, 0],
                        [0, 4, 5, 4, 2, 1, 2, 4, 5, 4, 0],
                        [0, 0, 0, 0, 1, 1, 1, 8, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
                    ]
                )
            )
            stripe.compute_biodescriptors(matrix)
            assert np.allclose(stripe._five_number, np.array([1.0, 1.0, 2.0, 2.0, 3.0]))
            assert np.isclose(stripe._inner_mean, 1.6666666666666667, atol=1e-15)
            assert np.isclose(stripe._inner_std, 0.6666666666666666, atol=1e-15)
            assert np.isclose(stripe._outer_lmean, 4.666666666666667, atol=1e-15)
            assert np.isclose(stripe._outer_rmean, 4.666666666666667, atol=1e-15)

        def test_compute_stripe_with_uniform_neighbourhoods(self):
            stripe = Stripe(
                seed=4,
                top_pers=5.0,
                horizontal_bounds=(4, 7),
                vertical_bounds=(4, 7),
                where="lower_triangular",
            )
            matrix = ss.csr_matrix(
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 4, 1, 1, 1, 0, 0, 0, 0],
                        [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
                        [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
                        [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
                        [0, 0, 0, 0, 1, 1, 1, 8, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
                    ]
                )
            )
            stripe.compute_biodescriptors(matrix)
            assert np.allclose(stripe._five_number, np.array([5.0, 5.0, 5.0, 5.0, 5.0]))
            assert np.isclose(stripe._inner_mean, 5.0, atol=1e-5)
            assert np.isclose(stripe._inner_std, 0.0, atol=1e-5)
            assert np.isclose(stripe._outer_lmean, 5.0, atol=1e-5)
            assert np.isclose(stripe._outer_rmean, 5.0, atol=1e-5)

    #####
    ### Error messages
    #####
    class TestComputeBiodescriptorErrors:
        def test_compute_horizontal_bounds_not_set(self):
            stripe = Stripe(
                seed=4,
                top_pers=5.0,
                horizontal_bounds=None,
                vertical_bounds=(4, 7),
                where="lower_triangular",
            )
            matrix = ss.csr_matrix(
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 4, 1, 1, 1, 0, 0, 0, 0],
                        [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
                        [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
                        [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
                        [0, 0, 0, 0, 1, 1, 1, 8, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
                    ]
                )
            )
            with pytest.raises(
                RuntimeError, match=re.escape(r"compute_biodescriptors() was called on a bound-less stripe")
            ):
                stripe.compute_biodescriptors(matrix)
            with pytest.raises(
                RuntimeError,
                match=re.escape(
                    r"caught an attempt to access five_number property before compute_biodescriptors() was called"
                ),
            ):
                assert stripe.five_number == None

        def test_compute_vertical_bounds_not_set(self):
            stripe = Stripe(
                seed=4,
                top_pers=5.0,
                horizontal_bounds=(4, 7),
                vertical_bounds=None,
                where="lower_triangular",
            )
            matrix = ss.csr_matrix(
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 4, 1, 1, 1, 0, 0, 0, 0],
                        [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
                        [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
                        [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
                        [0, 0, 0, 0, 1, 1, 1, 8, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
                    ]
                )
            )
            with pytest.raises(
                RuntimeError, match=re.escape(r"compute_biodescriptors() was called on a bound-less stripe")
            ):
                stripe.compute_biodescriptors(matrix)
            with pytest.raises(
                RuntimeError,
                match=re.escape(
                    r"caught an attempt to access five_number property before compute_biodescriptors() was called"
                ),
            ):
                assert stripe.five_number == None

        def test_compute_no_bounds_set(self):
            stripe = Stripe(
                seed=4,
                top_pers=5.0,
                horizontal_bounds=None,
                vertical_bounds=None,
                where="lower_triangular",
            )
            matrix = ss.csr_matrix(
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 4, 1, 1, 1, 0, 0, 0, 0],
                        [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
                        [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
                        [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
                        [0, 0, 0, 0, 1, 1, 1, 8, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
                    ]
                )
            )
            with pytest.raises(
                RuntimeError, match=re.escape(r"compute_biodescriptors() was called on a bound-less stripe")
            ):
                stripe.compute_biodescriptors(matrix)
            with pytest.raises(
                RuntimeError,
                match=re.escape(
                    r"caught an attempt to access five_number property before compute_biodescriptors() was called"
                ),
            ):
                assert stripe.five_number == None

        def test_window_negative(self):
            stripe = Stripe(
                seed=4,
                top_pers=5.0,
                horizontal_bounds=(4, 7),
                vertical_bounds=(4, 7),
                where="lower_triangular",
            )
            matrix = ss.csr_matrix(
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 4, 1, 1, 1, 0, 0, 0, 0],
                        [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
                        [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
                        [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
                        [0, 0, 0, 0, 1, 1, 1, 8, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
                    ]
                )
            )
            with pytest.raises(ValueError, match="window cannot be negative"):
                stripe.compute_biodescriptors(matrix, window=-1)
            with pytest.raises(
                RuntimeError,
                match=re.escape(
                    r"caught an attempt to access five_number property before compute_biodescriptors() was called"
                ),
            ):
                assert stripe.five_number == None
