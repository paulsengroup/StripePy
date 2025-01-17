import pytest

from stripepy.utils.stripe import Stripe


@pytest.fixture(scope="function")
def U_stripe():
    stripe = Stripe(seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(0, 4), where="upper_triangular")
    yield stripe


@pytest.fixture(scope="function")
def L_stripe():
    stripe = Stripe(seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(4, 10), where="lower_triangular")
    yield stripe


@pytest.mark.unit
class TestInit:
    def test_all_values_okay(self, U_stripe):
        stripe = U_stripe

        assert stripe._seed == 5
        assert stripe._persistence == 5.0
        assert stripe._where == "upper_triangular"
        assert stripe._left_bound == 4
        assert stripe._right_bound == 6
        assert stripe._top_bound == 0
        assert stripe._bottom_bound == 4
        assert stripe._five_number == None
        assert stripe._inner_mean == None
        assert stripe._inner_std == None
        assert stripe._outer_lmean == None
        assert stripe._outer_rmean == None

    def test_seed_too_low(self):
        with pytest.raises(ValueError):
            stripe = Stripe(
                seed=-1, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(0, 4), where="invalid location"
            )

    def test_top_pers_too_low(self):
        with pytest.raises(ValueError):
            stripe = Stripe(
                seed=5, top_pers=-1.0, horizontal_bounds=(4, 6), vertical_bounds=(0, 4), where="upper_triangular"
            )

    def test_invalid_location(self):
        with pytest.raises(ValueError):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(0, 4), where="upper_triangular"
            )

    def test_left_bound_over_right(self):
        with pytest.raises(ValueError):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(7, 6), vertical_bounds=(0, 4), where="upper_triangular"
            )

    def test_left_bound_over_seed(self):
        with pytest.raises(ValueError):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(6, 6), vertical_bounds=(0, 4), where="upper_triangular"
            )

    def test_right_bound_over_left(self):
        with pytest.raises(ValueError):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 3), vertical_bounds=(0, 4), where="upper_triangular"
            )

    def test_right_bound_over_seed(self):
        with pytest.raises(ValueError):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 4), vertical_bounds=(0, 4), where="upper_triangular"
            )

    def test_top_bound_over_bottom(self):
        with pytest.raises(ValueError):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 4), vertical_bounds=(1, 0), where="upper_triangular"
            )

    def test_bottom_bound_over_top(self):
        with pytest.raises(ValueError):
            stripe = Stripe(
                seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(5, 4), where="upper_triangular"
            )


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
