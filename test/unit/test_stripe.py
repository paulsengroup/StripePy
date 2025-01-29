import re

import numpy as np
import pytest

from stripepy.utils.stripe import Stripe


@pytest.mark.unit
class TestObjectInitialization:
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

    def test_none_to_constructor(self):
        stripe = Stripe(seed=5, top_pers=None, horizontal_bounds=None, vertical_bounds=None, where=None)

        assert stripe.seed == 5


@pytest.mark.unit
class TestSeed:
    def test_seed_at_matrix_border(self):
        stripe = Stripe(seed=0, top_pers=None, horizontal_bounds=None, vertical_bounds=None, where=None)

        assert stripe.seed == 0

    def test_seed_outside_matrix(self):
        with pytest.raises(ValueError, match="seed must be a non-negative integral number"):
            stripe = Stripe(seed=-1, top_pers=None, horizontal_bounds=None, vertical_bounds=None, where=None)

    def test_seed_none(self):
        with pytest.raises(TypeError, match=re.escape(r"'<' not supported between instances of 'NoneType' and 'int'")):
            stripe = Stripe(
                seed=None, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(1, 4), where="upper_triangular"
            )

    def test_seed_higher_value(self):
        stripe = Stripe(seed=100, top_pers=None, horizontal_bounds=None, vertical_bounds=None, where=None)
