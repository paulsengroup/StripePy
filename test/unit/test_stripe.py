import pytest

from stripepy.utils.stripe import Stripe

@pytest.fixture(scope="function")
def U_stripe():
    stripe = Stripe(
        seed=5, 
        top_pers=5.0, 
        horizontal_bounds=(4,6), 
        vertical_bounds=(0, 4), 
        where="upper_triangular")
    yield stripe

@pytest.fixture(scope="function")
def L_stripe():
    stripe = Stripe(
        seed = 5,
        top_pers=5.0
        horizontal_bounds=(4,6)
        vertical_bounds=(4,10)
        where="lower_triangular")
    yield stripe

@pytest.mark.unit
class TestInferLocation:
    @pytest.mark.skip(reason="Method not poperly implemented")
    def test_within_upper_triangle(self, U_stripe):
        UP = U_stripe._infer_location(U_stripe._seed, U_stripe._top_bound, U_stripe._bottom_bound)

        assert UP == "upper_triangular"

    def test_within_lower_triangular(self, L_stripe):
        LO = L_stripe._infer_location(L_stripe._seed, L_stripe._top_bound, L_stripe._bottom_bound)

        assert LO == "lower_triangular"

    @pytest.mark.skip(reason="Method not poperly implemented")
    def test_equal_coordinate(self, U_stripe):
        with pytest.raises(Exception):
            EQ = U_stripe._infer_location(10, 100, 100)

@pytest.mark.unit
class TestComputeConvexComp:
    def test_in_upper(self, U_stripe):
        comp = U_stripe._compute_convex_comp()
        
        assert comp == 4
    
    def test_in_lower(self, L_stripe):
        comp = L_stripe._compute_convex_comp()

        assert comp == 4

