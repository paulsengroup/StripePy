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


def custom_stripe(seed="NaN", top_pers="NaN", horizontal_bounds="NaN", vertical_bounds="NaN", where="NaN"):
    stripe = Stripe(
        seed=(seed if seed != "NaN" else 5),
        top_pers=(top_pers if top_pers != "NaN" else 5.0),
        horizontal_bounds=(horizontal_bounds if horizontal_bounds != "NaN" else (4, 6)),
        vertical_bounds=(vertical_bounds if vertical_bounds != "NaN" else (1, 4)),
        where=(where if where != "NaN" else "upper_triangular"),
    )
    return stripe


@pytest.mark.unit
class TestInit:
    def test_all_values_okay(self, U_stripe):
        stripe = U_stripe

        assert stripe.seed == 5
        assert stripe.top_persistence == 5.0
        assert not stripe.lower_triangular
        assert stripe.upper_triangular
        assert stripe.left_bound == 4
        assert stripe.right_bound == 6
        assert stripe.top_bound == 1
        assert stripe.bottom_bound == 4
        # assert stripe._five_number == None
        # assert stripe._inner_mean == None
        # assert stripe._inner_std == None
        # assert stripe._outer_lmean == None
        # assert stripe._outer_rmean == None

    def test_seed_lower_valid(self):
        with pytest.raises(RuntimeError) as e:
            stripe = custom_stripe(seed=0, horizontal_bounds=(0, 2))
        assert (
            str(e.value)
            == "computed location does not match the provided stripe location: computed=lower_triangular, expected=upper_triangular"
        )

    def test_seed_too_low(self):
        with pytest.raises(ValueError) as e:
            stripe = custom_stripe(seed=-1)
        assert e

    def test_top_pers_too_low(self):
        with pytest.raises(ValueError) as e:
            stripe = custom_stripe(top_pers=-1.0)
        assert e

    def test_where_is_opposite(self):
        with pytest.raises(RuntimeError) as e:
            stripe = custom_stripe(seed=5, vertical_bounds=(1, 6), where="upper_triangular")

            assert stripe.lower_triangular()
        assert e

    def test_invalid_location(self):
        with pytest.raises(ValueError) as e:
            stripe = custom_stripe(where="invalid_location")
        assert e

    def test_left_bound_over_right(self):
        with pytest.raises(ValueError) as e:
            stripe = custom_stripe(horizontal_bounds=(7, 6))
        assert e

    def test_left_bound_over_seed(self):
        with pytest.raises(ValueError) as e:
            stripe = custom_stripe(horizontal_bounds=(6, 6))
        assert e

    def test_right_bound_over_left(self):
        with pytest.raises(ValueError) as e:
            stripe = custom_stripe(horizontal_bounds=(4, 3))
        assert e

    def test_right_bound_over_seed(self):
        with pytest.raises(ValueError) as e:
            stripe = custom_stripe(horizontal_bounds=(4, 4))
        assert e

    def test_top_bound_under_bottom(self):
        with pytest.raises(ValueError) as e:
            stripe = custom_stripe(vertical_bounds=(1, 0))
        assert e

    def test_bottom_bound_over_top(self):
        with pytest.raises(ValueError):
            stripe = custom_stripe(vertical_bounds=(5, 4))


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
class TestBoundaryProperties:
    #####
    ### Left boundary
    #####
    def test_left_bound_to_left(self):
        stripe = custom_stripe(horizontal_bounds=None)
        stripe.set_horizontal_bounds(3, 6)
        assert stripe.left_bound == 3

    def test_left_bound_to_right(self):
        stripe = custom_stripe(horizontal_bounds=None)
        stripe.set_horizontal_bounds(5, 6)

        assert stripe.left_bound == 5

    def test_left_bound_over_seed(self):
        stripe = custom_stripe(horizontal_bounds=None)
        with pytest.raises(ValueError) as e:
            stripe.set_horizontal_bounds(6, 6)
        assert str(e.value) == "horizontal bounds must enclose the seed position: seed=5, left_bound=6, right_bound=6"

    def test_left_bound_over_right(self):
        stripe = custom_stripe(horizontal_bounds=None)
        with pytest.raises(ValueError) as e:
            stripe.set_horizontal_bounds(7, 6)
        assert str(e.value) == "horizontal bounds must enclose the seed position: seed=5, left_bound=7, right_bound=6"

    #####
    ### Right boundary
    #####
    def test_right_bound_to_right(self):
        stripe = custom_stripe(horizontal_bounds=None)
        stripe.set_horizontal_bounds(4, 7)

        assert stripe.right_bound == 7

    def test_right_bound_to_left(self):
        stripe = custom_stripe(horizontal_bounds=None)
        stripe.set_horizontal_bounds(4, 5)

        assert stripe.right_bound == 5

    def test_right_bound_over_seed(self):
        stripe = custom_stripe(horizontal_bounds=None)
        with pytest.raises(ValueError) as e:
            stripe.set_horizontal_bounds(4, 4)

        assert str(e.value) == "horizontal bounds must enclose the seed position: seed=5, left_bound=4, right_bound=4"

    def test_right_bound_over_left(self):
        stripe = custom_stripe(horizontal_bounds=None)
        with pytest.raises(ValueError) as e:
            stripe.set_horizontal_bounds(4, 3)

        assert str(e.value) == "horizontal bounds must enclose the seed position: seed=5, left_bound=4, right_bound=3"

    #####
    ### Top boundary
    #####
    def test_top_bound_over_matrix_top(self):
        stripe = custom_stripe(vertical_bounds=None)
        stripe.set_vertical_bounds(-1, 4)

        assert stripe.top_bound == -1

    def test_top_bound_under_diagonal(self):
        stripe = custom_stripe(vertical_bounds=None)
        with pytest.raises(ValueError) as e:
            stripe.set_vertical_bounds(7, 4)

        assert (
            str(e.value)
            == "the lower vertical bound must be greater than the upper vertical bound: top_bound=7, bottom_bound=4"
        )

    def test_top_bound_under_bottom(self):
        stripe = custom_stripe(vertical_bounds=None)
        with pytest.raises(ValueError) as e:
            stripe.set_vertical_bounds(5, 4)

        assert (
            str(e.value)
            == "the lower vertical bound must be greater than the upper vertical bound: top_bound=5, bottom_bound=4"
        )

    #####
    ### Bottom boundary
    #####
    def test_bottom_bound_over_top(self):
        stripe = custom_stripe(vertical_bounds=None)
        with pytest.raises(ValueError) as e:
            stripe.set_vertical_bounds(4, 1)

        assert (
            str(e.value)
            == "the lower vertical bound must be greater than the upper vertical bound: top_bound=4, bottom_bound=1"
        )

    def test_bottom_bound_under_diagonal(self):
        stripe = custom_stripe(where="lower_triangular", vertical_bounds=None)
        stripe.set_vertical_bounds(1, 6)

        assert stripe.bottom_bound > stripe.seed

    def test_bottom_bound_under_matrix_bottom(self):
        stripe = custom_stripe(where="lower_triangular", vertical_bounds=None)
        stripe.set_vertical_bounds(1, 12)

        assert stripe.bottom_bound > 10


@pytest.fixture(scope="function")
def matrix():
    row1 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    row2 = np.array([0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    row3 = np.array([0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0])
    row4 = np.array([0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0])
    row5 = np.array([0, 0, 0, 0, 5, 0, 0, 0, 0, 0])
    row6 = np.array([0, 0, 0, 0, 0, 6, 0, 0, 0, 0])
    row7 = np.array([0, 0, 0, 0, 0, 0, 7, 0, 0, 0])
    row8 = np.array([0, 0, 0, 0, 0, 0, 0, 8, 0, 0])
    row9 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 9, 0])
    row10 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 10])
    matrix = np(row1, row2, row3, row4, row5, row6, row7, row8, row9, row10)
    return matrix


@pytest.mark.unit
class TestStatistics:
    def test_statistical_manual(self):
        stripe = statistical_stripe()

        assert stripe.inner_mean == 1.0
        assert stripe.inner_std == 1.0
        assert stripe.five_number == [1.0] * 5
        assert stripe.outer_lmean == 1.0
        assert stripe.outer_rmean == 1.0
        assert stripe.outer_mean == 1.0
        assert stripe.rel_change == 0.0


@pytest.mark.unit
class TestInferLocation:
    def test_within_upper_triangle(self, U_stripe):
        UP = U_stripe._infer_location(U_stripe._seed, U_stripe._top_bound, U_stripe._bottom_bound)

        assert UP == "upper_triangular"

    def test_within_lower_triangular(self, L_stripe):
        LO = L_stripe._infer_location(L_stripe._seed, L_stripe._top_bound, L_stripe._bottom_bound)

        assert LO == "lower_triangular"


@pytest.mark.unit
class TestComputeConvexComp:
    def test_in_upper(self, U_stripe):
        comp = U_stripe._compute_convex_comp()

        assert comp == 4

    def test_in_lower(self, L_stripe):
        comp = L_stripe._compute_convex_comp()

        assert comp == 4
