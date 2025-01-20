import pytest

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
def L_stripe(seed=None, top_pers=None, horizontal_bounds=None, vertical_bounds=None, where=None):
    stripe = Stripe(
        seed=(seed or 5),
        top_pers=(top_pers or 5.0),
        horizontal_bounds=(horizontal_bounds or (4, 6)),
        vertical_bounds=(vertical_bounds or (4, 10)),
        where=(where or "lower_triangular"),
    )
    return stripe


def custom_stripe(seed=None, top_pers=None, horizontal_bounds=None, vertical_bounds=None, where=None):
    stripe = Stripe(
        seed=(seed or 5),
        top_pers=(top_pers or 5.0),
        horizontal_bounds=(horizontal_bounds or (4, 6)),
        vertical_bounds=(vertical_bounds or (1, 4)),
        where=(where or "upper_triangular"),
    )
    return stripe


@pytest.mark.unit
class TestInit:
    def test_all_values_okay(self, U_stripe):
        stripe = U_stripe

        assert stripe._seed == 5
        assert stripe._persistence == 5.0
        assert stripe._where == "upper_triangular"
        assert stripe._left_bound == 4
        assert stripe._right_bound == 6
        assert stripe._top_bound == 1
        assert stripe._bottom_bound == 4
        assert stripe._five_number == None
        assert stripe._inner_mean == None
        assert stripe._inner_std == None
        assert stripe._outer_lmean == None
        assert stripe._outer_rmean == None

    def test_seed_lower_valid(self):
        stripe = custom_stripe(seed=0)

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


@pytest.mark.unit
class TestProperties:
    def test_seed(self, U_stripe):
        assert U_stripe.seed

    def test_top_persistence(self, U_stripe):
        assert U_stripe.top_persistence

    def test_lower_triangular(self, U_stripe):
        assert not U_stripe.lower_triangular

    def test_upper_triangular(self, U_stripe):
        assert U_stripe.upper_triangular

    def test_left_bound(self, U_stripe):
        assert U_stripe.left_bound

    def test_right_bound(self, U_stripe):
        assert U_stripe.right_bound

    def test_top_bound(self, U_stripe):
        assert U_stripe.top_bound

    def test_bottom_bound(self, U_stripe):
        assert U_stripe.bottom_bound

    """
    #### These have not been set ####
    """

    # TODO: give these pointers a value before moving on
    @pytest.mark.skip(reason="Immature test")
    def test_inner_mean(self, U_stripe):
        assert U_stripe.inner_mean

    @pytest.mark.skip(reason="Immature test")
    def test_inner_std(self, U_stripe):
        assert U_stripe.inner_std

    @pytest.mark.skip(reason="Immature test")
    def test_five_number(self, U_stripe):
        assert U_stripe.five_number

    @pytest.mark.skip(reason="Immature test")
    def test_outer_lmean(self, U_stripe):
        assert U_stripe.outer_lmean

    @pytest.mark.skip(reason="Immature test")
    def test_outer_rmean(self, U_stripe):
        assert U_stripe.outer_rmean

    @pytest.mark.skip(reason="Immature test")
    def test_outer_mean(self, U_stripe):
        assert U_stripe.outer_mean

    @pytest.mark.skip(reason="Immature test")
    def test_rel_change(self, U_stripe):
        assert U_stripe.rel_change


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
