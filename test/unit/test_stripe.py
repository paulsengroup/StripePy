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


@pytest.fixture(scope="function")
def matrix():
    row1 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    row2 = np.array([0, 2, 0, 0, 1, 2, 0, 0, 0, 0, 0])
    row3 = np.array([0, 0, 3, 0, 3, 4, 0, 0, 0, 0, 0])
    row4 = np.array([0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0])
    row5 = np.array([0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0])
    row6 = np.array([0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0])
    row7 = np.array([0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0])
    row8 = np.array([0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0])
    row9 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0])
    row10 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0])
    row11 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11])
    matrix = np.array([row1, row2, row3, row4, row5, row6, row7, row8, row9, row10, row11])
    return ss.csr_matrix(matrix)


@pytest.mark.unit
class TestObjectInitialization:
    # TODO: Try to pass None-values and other types of values
    def test_constructor(self, U_stripe, L_stripe):
        assert U_stripe.seed == 5
        assert np.isclose(U_stripe.top_persistence, 5.0)
        assert not U_stripe.lower_triangular
        assert U_stripe.upper_triangular
        assert U_stripe.left_bound == 4
        assert U_stripe.right_bound == 6
        assert U_stripe.top_bound == 1
        assert U_stripe.bottom_bound == 4

        assert L_stripe.seed == 5
        assert np.isclose(L_stripe.top_persistence, 5.0)
        assert L_stripe.lower_triangular
        assert not L_stripe.upper_triangular
        assert L_stripe.left_bound == 4
        assert L_stripe.right_bound == 6
        assert L_stripe.top_bound == 4
        assert L_stripe.bottom_bound == 10

    def test_seed_lower_valid(self):
        u_stripe = Stripe(
            seed=0, top_pers=0.1, horizontal_bounds=(0, 2), vertical_bounds=(0, 0), where="upper_triangular"
        )

        l_stripe = Stripe(
            seed=0, top_pers=0.1, horizontal_bounds=(0, 2), vertical_bounds=(0, 2), where="lower_triangular"
        )

    # TODO: Add test_seed_upper_valid and maybe test_seed_too_high. Also test middle values

    def test_seed_too_low(self):
        with pytest.raises(ValueError, match="seed must be a non-negative integral number") as e:
            u_stripe = Stripe(
                seed=-1, top_pers=0.1, horizontal_bounds=(4, 6), vertical_bounds=(1, 4), where="upper_triangular"
            )

        with pytest.raises(ValueError, match="seed must be a non-negative integral number") as e:
            l_stripe = Stripe(
                seed=-1, top_pers=0.1, horizontal_bounds=(4, 6), vertical_bounds=(4, 10), where="lower_triangular"
            )

    # TODO: add test_top_pers_lower_valid and test_top_pers_upper_valid. Also test middle values
    def test_top_pers_too_low(self):
        with pytest.raises(ValueError, match="when not None, top_pers must be a positive number") as e:
            u_stripe = Stripe(
                seed=5, top_pers=-5.0, horizontal_bounds=(4, 6), vertical_bounds=(1, 4), where="upper_triangular"
            )

    # TODO: Add test_where_upper_valid and test_where_lower_valid. Maybe failing values as well
    def test_where_is_opposite_computed_upper(self):
        with pytest.raises(
            RuntimeError,
            match="computed location does not match the provided stripe location: computed=upper_triangular, expected=lower_triangular",
        ) as e:
            u_stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(1, 4), where="lower_triangular"
            )

    def test_where_is_opposite_computed_lower(self):
        with pytest.raises(
            RuntimeError,
            match="computed location does not match the provided stripe location: computed=lower_triangular, expected=upper_triangular",
        ) as e:
            l_stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(4, 10), where="upper_triangular"
            )

    def test_invalid_location(self):
        with pytest.raises(
            ValueError,
            match=re.escape(r"when specified, where must be one of ('upper_triangular', 'lower_triangular')"),
        ) as e:
            u_stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(1, 4), where="invalid_triangular"
            )

        with pytest.raises(
            ValueError,
            match=re.escape(r"when specified, where must be one of ('upper_triangular', 'lower_triangular')"),
        ) as e:
            l_stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(4, 10), where="invalid_triangular"
            )

    # TODO: add test_left_bound_over_right_seed_between and test_left_bound_over_right_seed_outside. Same for right, top and bottom boundary.
    ## These tests do not really add much in the way of testing different scenarios. Go over it again, and make a more comprehensive test suite.
    ## Also, test for negative integers.
    def test_left_bound_over_right(self):
        with pytest.raises(
            ValueError, match="horizontal bounds must enclose the seed position: seed=5, left_bound=7, right_bound=6"
        ) as e:
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(7, 6), vertical_bounds=(1, 4), where="upper_triangular"
            )

    def test_right_bound_over_left(self):
        with pytest.raises(
            ValueError, match="horizontal bounds must enclose the seed position: seed=5, left_bound=4, right_bound=3"
        ) as e:
            u_stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 3), vertical_bounds=(1, 4), where="upper_triangular"
            )

    def test_top_bound_under_bottom(self):
        with pytest.raises(
            ValueError,
            match="the lower vertical bound must be greater than the upper vertical bound: top_bound=1, bottom_bound=0",
        ) as e:
            u_stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(1, 0), where="upper_triangular"
            )

    def test_bottom_bound_over_top(self):
        with pytest.raises(
            ValueError,
            match="the lower vertical bound must be greater than the upper vertical bound: top_bound=5, bottom_bound=4",
        ):
            l_stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(5, 4), where="upper_triangular"
            )


@pytest.mark.unit
class TestSetBoundaryProperties:
    class TestSetHorizontalRelativeToSeed:
        def test_left_at_seed(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(5, 8), vertical_bounds=(1, 3), where="upper_triangular"
            )

        def test_right_at_seed(self):
            stripe = Stripe(
                seed=6, top_pers=5.0, horizontal_bounds=(3, 6), vertical_bounds=(1, 3), where="upper_triangular"
            )

        def test_left_and_right_at_seed(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(5, 5), vertical_bounds=(1, 3), where="upper_triangular"
            )

        def test_seed_adjacent_to_left(self):
            stripe = Stripe(
                seed=6, top_pers=5.0, horizontal_bounds=(5, 8), vertical_bounds=(1, 4), where="upper_triangular"
            )

        def test_seed_adjacent_to_right(self):
            stripe = Stripe(
                seed=7, top_pers=5.0, horizontal_bounds=(5, 8), vertical_bounds=(1, 4), where="upper_triangular"
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

    class TestSetHorizontalRelativeToDiagonal:
        def test_left_under_diagonal_upper_triangle(self):
            with pytest.raises(
                ValueError,
                match="top bound is not enclosed between the left and right bounds: left_bound=7, right_bound=4, top_bound=1",
            ):
                stripe = Stripe(
                    seed=5, top_pers=5.0, horizontal_bounds=(1, 6), vertical_bounds=(1, 4), where="upper_triangular"
                )

        def test_right_over_diagonal_lower_triangle(self):
            with pytest.raises(
                ValueError,
                match="top bound is not enclosed between the left and right bounds: left_bound=7, right_bound=4, top_bound=1",
            ):
                stripe = Stripe(
                    seed=5, top_pers=5.0, horizontal_bounds=(2, 7), vertical_bounds=(4, 10), where="lower_triangular"
                )

        def test_horizontal_field_under_diagonal_upper_triangle(self):
            with pytest.raises(
                ValueError,
                match="top bound is not enclosed between the left and right bounds: left_bound=7, right_bound=4, top_bound=1",
            ):
                stripe = Stripe(
                    seed=5, top_pers=5.0, horizontal_bounds=(1, 2), vertical_bounds=(3, 4), where="upper_triangular"
                )

        def test_horizontal_field_over_diagonal_lower_triangle(self):
            with pytest.raises(
                ValueError,
                match="top bound is not enclosed between the left and right bounds: left_bound=7, right_bound=4, top_bound=1",
            ):
                stripe = Stripe(
                    seed=5, top_pers=5.0, horizontal_bounds=(10, 12), vertical_bounds=(6, 8), where="lower_triangular"
                )

    class TestSetHorizontalRelativeToMatrixEdges:
        def test_left_at_matrix_edge(self):
            stripe = Stripe(
                seed=1, top_pers=5.0, horizontal_bounds=(0, 1), vertical_bounds=(1, 10), where="lower_triangular"
            )

        def test_left_and_right_at_matrix_edge(self):
            stripe = Stripe(
                seed=0, top_pers=5.0, horizontal_bounds=(0, 0), vertical_bounds=(1, 10), where="lower_triangular"
            )

        def test_left_over_matrix_edge(self):
            stripe = Stripe(
                seed=1, top_pers=5.0, horizontal_bounds=(-1, 1), vertical_bounds=(1, 10), where="lower_triangular"
            )

        def test_left_and_right_over_matrix_edge(self):
            stripe = Stripe(
                seed=1, top_pers=5.0, horizontal_bounds=(-1, -1), vertical_bounds=(1, 10), where="lower_triangular"
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
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(5, 5), where="lower_triangular"
            )

        def test_vertical_size_1(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(5, 6), where="lower_triangular"
            )

        def test_top_and_bottom_cross_themselves(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(5, 4), where="lower_triangular"
            )

    class TestSetVerticalRelativeToDiagonal:
        def test_bottom_under_diagonal_upper_triangle(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(1, 7), where="upper_triangular"
            )

        def test_vertical_field_under_diagonal_upper_triangle(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(6, 7), where="upper_triangular"
            )

        def test_top_over_diagonal_lower_triangle(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(1, 7), where="lower_triangular"
            )

        def test_vertical_field_over_diagonal_lower_triangle(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(1, 2), where="lower_triangular"
            )

    class TestSetVerticalRelativeToMatrixEdges:
        def test_top_at_matrix_edge(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(0, 5), where="upper_triangular"
            )

        def test_top_and_bottom_at_matrix_edge(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(0, 0), where="upper_triangular"
            )

        def test_top_over_matrix_edge(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(-1, 5), where="upper_triangular"
            )

        def test_top_and_bottom_over_matrix_edge(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(-1, -1), where="upper_triangular"
            )

    class TestSetVerticalWhenAlreadySet:
        def test_horizontal_bounds_already_set(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(1, 4), where="upper_triangular"
            )

            with pytest.raises(RuntimeError, match="horizontal stripe bounds have already been set"):
                stripe.set_vertical_bounds(2, 6)

    class TestSetMethods:
        #####
        ### Left boundary
        #####
        def test_left_bound_to_left(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=None, vertical_bounds=(1, 4), where="upper_triangular"
            )
            stripe.set_horizontal_bounds(3, 7)
            assert stripe.left_bound == 3

        def test_left_bound_to_right(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=None, vertical_bounds=(1, 4), where="upper_triangular"
            )
            stripe.set_horizontal_bounds(5, 6)

            assert stripe.left_bound == 5

        def test_left_bound_over_seed(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=None, vertical_bounds=(1, 4), where="upper_triangular"
            )
            with pytest.raises(
                ValueError,
                match="horizontal bounds must enclose the seed position: seed=5, left_bound=6, right_bound=6",
            ) as e:
                stripe.set_horizontal_bounds(6, 6)

        def test_left_bound_over_right(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=None, vertical_bounds=(1, 4), where="upper_triangular"
            )
            with pytest.raises(
                ValueError,
                match="horizontal bounds must enclose the seed position: seed=5, left_bound=7, right_bound=6",
            ) as e:
                stripe.set_horizontal_bounds(7, 6)

        #####
        ### Right boundary
        #####
        def test_right_bound_to_right(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=None, vertical_bounds=(1, 4), where="upper_triangular"
            )
            stripe.set_horizontal_bounds(4, 7)

            assert stripe.right_bound == 7

        def test_right_bound_to_left(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=None, vertical_bounds=(1, 4), where="upper_triangular"
            )
            stripe.set_horizontal_bounds(4, 5)

            assert stripe.right_bound == 5

        def test_right_bound_over_seed(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=None, vertical_bounds=(1, 4), where="upper_triangular"
            )
            with pytest.raises(
                ValueError,
                match="horizontal bounds must enclose the seed position: seed=5, left_bound=4, right_bound=4",
            ) as e:
                stripe.set_horizontal_bounds(4, 4)

        def test_right_bound_over_left(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=None, vertical_bounds=(1, 4), where="upper_triangular"
            )
            with pytest.raises(
                ValueError,
                match="horizontal bounds must enclose the seed position: seed=5, left_bound=4, right_bound=3",
            ) as e:
                stripe.set_horizontal_bounds(4, 3)

        #####
        ### Top boundary
        #####
        def test_top_bound_over_matrix_top(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=None, where="upper_triangular"
            )
            stripe.set_vertical_bounds(-1, 4)

            assert stripe.top_bound == -1

        def test_top_bound_under_diagonal(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=None, where="upper_triangular"
            )
            with pytest.raises(
                ValueError,
                match="the lower vertical bound must be greater than the upper vertical bound: top_bound=7, bottom_bound=4",
            ) as e:
                stripe.set_vertical_bounds(7, 4)

        def test_top_bound_under_bottom(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=None, where="upper_triangular"
            )
            with pytest.raises(
                ValueError,
                match="the lower vertical bound must be greater than the upper vertical bound: top_bound=5, bottom_bound=4",
            ) as e:
                stripe.set_vertical_bounds(5, 4)

        #####
        ### Bottom boundary
        #####
        def test_bottom_bound_over_top(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=None, where="upper_triangular"
            )
            with pytest.raises(
                ValueError,
                match="the lower vertical bound must be greater than the upper vertical bound: top_bound=4, bottom_bound=1",
            ) as e:
                stripe.set_vertical_bounds(4, 1)

        def test_bottom_bound_under_diagonal(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=None, where="lower_triangular"
            )
            stripe.set_vertical_bounds(1, 6)

            assert stripe.bottom_bound > stripe.seed

        def test_bottom_bound_under_matrix_bottom(self):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=None, where="lower_triangular"
            )
            stripe.set_vertical_bounds(1, 12)

            assert stripe.bottom_bound > 10


def statistical_stripe(
    inner_mean=1.0,
    inner_std=1.0,
    five_number=[1.0] * 5,
    outer_lmean=1.0,
    outer_rmean=1.0,
    outer_mean=None,
    rel_change=None,
):
    if outer_mean == None:
        outer_mean = (outer_lmean + outer_rmean) / 2
    if rel_change == None:  # 1
        rel_change = abs(inner_mean - outer_mean) / outer_mean * 100  # 0

    stripe = Stripe(
        seed=5,
        top_pers=5.0,
        horizontal_bounds=(4, 6),
        vertical_bounds=(1, 4),
        where="upper_triangular",
    )
    stripe._inner_mean = inner_mean
    stripe._inner_std = inner_std
    stripe._five_number = five_number
    stripe._outer_lmean = outer_lmean
    stripe._outer_rmean = outer_rmean
    stripe._outer_mean = outer_mean
    stripe._rel_change = rel_change
    return stripe


@pytest.mark.unit
class TestStatistics:
    def test_initialize_with_biodescriptors(self):
        stripe = statistical_stripe()

        assert np.isclose(stripe.inner_mean, 1.0)
        assert np.isclose(stripe.inner_std, 1.0)
        assert np.isclose(stripe.five_number, [1.0] * 5)
        assert np.isclose(stripe.outer_lmean, 1.0)
        assert np.isclose(stripe.outer_rmean, 1.0)
        assert np.isclose(stripe.outer_mean, 1.0)
        assert np.isclose(stripe.rel_change, 0.0)

    def test_compute_statistics(self, matrix, U_stripe):
        U_stripe.compute_biodescriptors(matrix)
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

        assert np.isclose(U_stripe._five_number, np.array([0.0, 0.25, 1.5, 2.75, 4.0]))
        assert np.isclose(U_stripe._inner_mean, 1.6666666666666667, atol=1e-16)
        assert np.isclose(U_stripe._inner_std, 1.4907119849998596, atol=1e-16)
        assert np.isclose(U_stripe._outer_lmean, 1.0)
        assert np.isclose(U_stripe._outer_rmean, 0.0)

    # TODO: Add tests with lower and upper boundary for all statistical values. Also middle values.


@pytest.mark.unit
class TestInferLocation:
    # TODO: If the tests are useful, move them to class TestSetBoundaryProperties. If not, delete.
    def test_within_upper_triangle(self, U_stripe):
        UP = U_stripe._infer_location(U_stripe._seed, U_stripe._top_bound, U_stripe._bottom_bound)

        assert UP == "upper_triangular"

    def test_within_lower_triangular(self, L_stripe):
        LO = L_stripe._infer_location(L_stripe._seed, L_stripe._top_bound, L_stripe._bottom_bound)

        assert LO == "lower_triangular"


@pytest.mark.unit
class TestComputeConvexComp:
    # TODO: Delete.
    def test_in_upper(self, U_stripe):
        comp = U_stripe._compute_convex_comp()

        assert comp == 4

    def test_in_lower(self, L_stripe):
        comp = L_stripe._compute_convex_comp()

        assert comp == 4
