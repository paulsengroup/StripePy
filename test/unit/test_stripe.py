import pytest

from stripepy.utils.stripe import Stripe

@pytest.fixture(scope="function")
def stripe():
    stripe = Stripe(10, 5.0, (4,6), (2,8), "upper_triangular")
    yield stripe

@pytest.mark.unit
class TestInferLocation:
    @pytest.mark.skip(reason="Method not poperly implemented")
    def test_infer_location_upper(self, stripe):
        UP = stripe._infer_location(stripe._seed, stripe._top_bound, stripe._bottom_bound)

        assert UP == "upper_triangular"

    def test_infer_location_lower(self, stripe):
        LO = stripe._infer_location(10, 102, 100)

        assert LO == "lower_triangular"

    @pytest.mark.skip(reason="Method not poperly implemented")
    def test_infer_location_equal(self):
        with pytest.raises(Exception):
            EQ = stripe._infer_location(10, 100, 100)
