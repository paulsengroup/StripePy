# Copyright (C) 2025 Bendik Berg <bendber@ifi.uio.no>
#
# SPDX-License-Identifier: MIT

import math

import numpy as np
import pytest
import scipy.sparse as ss

from stripepy.utils.stripe import Stripe


@pytest.mark.unit
class TestObjectInitialization:
    def test_seed_outside_matrix(self):
        with pytest.raises(ValueError, match="seed must be a non-negative integral number"):
            stripe = Stripe(seed=-1, top_pers=None, horizontal_bounds=None, vertical_bounds=None, where=None)

    def test_top_persistence_negative(self):
        with pytest.raises(ValueError, match="when not None, top_pers must be a positive number"):
            stripe = Stripe(seed=0, top_pers=-1.0, horizontal_bounds=None, vertical_bounds=None, where=None)

    def test_where_invalid_input(self):
        with pytest.raises(ValueError, match="when specified, where must be one of (.*upper.*|.*lower.*){2}"):
            stripe = Stripe(
                seed=5, top_pers=None, horizontal_bounds=None, vertical_bounds=None, where="invalid_triangular"
            )


@pytest.mark.unit
class TestBoundaryGetters:
    def test_valid_access(self):
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

        stripe = Stripe(
            seed=5, top_pers=5.0, horizontal_bounds=(4, 6), vertical_bounds=(4, 10), where="lower_triangular"
        )

        assert stripe.lower_triangular
        assert not stripe.upper_triangular

    def test_not_valid_access(self):
        stripe = Stripe(seed=5, top_pers=None, horizontal_bounds=None, vertical_bounds=None, where=None)

        assert stripe.seed == 5
        assert stripe.top_persistence is None
        with pytest.raises(RuntimeError, match="left_bound has not been set"):
            assert stripe.left_bound is None
        with pytest.raises(RuntimeError, match="right_bound has not been set"):
            assert stripe.right_bound is None
        with pytest.raises(RuntimeError, match="top_bound has not been set"):
            assert stripe.top_bound is None
        with pytest.raises(RuntimeError, match="bottom_bound has not been set"):
            assert stripe.bottom_bound is None
        assert not stripe.upper_triangular
        assert not stripe.lower_triangular


@pytest.mark.unit
class TestBoundarySetters:
    def test_left_and_right_at_seed(self):
        stripe = Stripe(seed=5, top_pers=None, horizontal_bounds=None, vertical_bounds=None, where=None)

        stripe.set_horizontal_bounds(5, 5)

        assert stripe.left_bound == 5
        assert stripe.right_bound == 5

    def test_seed_outside_horizontal_bounds(self):
        stripe = Stripe(seed=5, top_pers=None, horizontal_bounds=None, vertical_bounds=None, where=None)
        with pytest.raises(
            ValueError, match="horizontal bounds must enclose the seed position: seed=5, left_bound=6, right_bound=6"
        ):
            stripe.set_horizontal_bounds(6, 6)
        with pytest.raises(
            ValueError,
            match="horizontal bounds must enclose the seed position: seed=5, left_bound=4, right_bound=4",
        ):
            stripe.set_horizontal_bounds(4, 4)

    def test_empty_horizontal_domains(self):
        stripe = Stripe(seed=5, top_pers=None, horizontal_bounds=None, vertical_bounds=None, where=None)
        with pytest.raises(
            ValueError, match="horizontal bounds must enclose the seed position: seed=5, left_bound=6, right_bound=5"
        ):
            stripe.set_horizontal_bounds(6, 5)

    def test_negative_horizontal_bounds(self):
        stripe = Stripe(seed=1, top_pers=None, horizontal_bounds=None, vertical_bounds=None, where=None)
        with pytest.raises(ValueError, match="stripe bounds must be positive integers"):
            stripe.set_horizontal_bounds(-1, 1)

    def test_horizontal_bounds_already_set(self):
        stripe = Stripe(seed=5, top_pers=None, horizontal_bounds=(4, 6), vertical_bounds=None, where=None)

        with pytest.raises(RuntimeError, match="horizontal stripe bounds have already been set"):
            stripe.set_horizontal_bounds(5, 7)

    def test_vertical_size_0(self):
        stripe = Stripe(seed=5, top_pers=None, horizontal_bounds=None, vertical_bounds=None, where=None)

        stripe.set_vertical_bounds(5, 5)

        assert stripe.top_bound == 5
        assert stripe.bottom_bound == 5

    def test_empty_vertical_domains(self):
        stripe = Stripe(seed=5, top_pers=None, horizontal_bounds=None, vertical_bounds=None, where=None)

        with pytest.raises(
            ValueError,
            match="the lower vertical bound must be greater than the upper vertical bound: top_bound=5, bottom_bound=4",
        ):
            stripe.set_vertical_bounds(5, 4)

    def test_negative_vertical_bounds(self):
        stripe = Stripe(seed=5, top_pers=None, horizontal_bounds=None, vertical_bounds=None, where=None)
        with pytest.raises(ValueError, match="stripe bounds must be positive integers"):
            stripe.set_vertical_bounds(-1, -1)

    def test_vertical_bounds_already_set(self):
        stripe = Stripe(seed=5, top_pers=None, horizontal_bounds=None, vertical_bounds=(1, 4), where=None)

        with pytest.raises(RuntimeError, match="vertical stripe bounds have already been set"):
            stripe.set_vertical_bounds(2, 6)


@pytest.mark.unit
class TestComputeBiodescriptors:
    def test_stripe_in_upper_left_corner(self):
        stripe = Stripe(
            seed=0,
            top_pers=None,
            horizontal_bounds=(0, 2),
            vertical_bounds=(0, 2),
            where="lower_triangular",
        )
        matrix = ss.csr_matrix(
            [
                [5, 5, 5, 2, 0, 0, 0],
                [5, 5, 5, 3, 0, 0, 0],
                [5, 5, 5, 4, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        )
        stripe.compute_biodescriptors(matrix, window=1)

        assert np.allclose(stripe._five_number, np.full(5, 5.0))
        assert np.isclose(stripe._inner_mean, 5.0)
        assert np.isclose(stripe._inner_std, 0.0)
        # TODO re-enable after bug fix
        # assert math.isnan(stripe._outer_lmean)
        # assert np.isclose(stripe._outer_rmean, 3.0, atol=1e-5)

    def test_stripe_in_middle(self):
        stripe = Stripe(
            seed=4,
            top_pers=None,
            horizontal_bounds=(3, 4),
            vertical_bounds=(3, 6),
            where="lower_triangular",
        )
        matrix = ss.csr_matrix(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 4, 4, 0, 0],
                [0, 0, 0, 4, 4, 0, 0],
                [0, 0, 0, 4, 4, 0, 0],
                [0, 0, 0, 4, 4, 0, 0],
            ]
        )
        stripe.compute_biodescriptors(matrix)

        assert np.allclose(stripe._five_number, np.full(5, 4.0))
        assert np.isclose(stripe._inner_mean, 4.0)
        assert np.isclose(stripe._inner_std, 0.0)
        # TODO re-enable after bug fix
        # assert np.isclose(stripe._outer_lmean, 0.0)
        # assert np.isclose(stripe._outer_rmean, 1.625, atol=1e-5)


@pytest.mark.unit
class TestComputeBiodescriptorErrors:
    def test_compute_no_bounds_set(self):
        stripe = Stripe(
            seed=2,
            top_pers=None,
            horizontal_bounds=None,
            vertical_bounds=None,
            where="lower_triangular",
        )
        matrix = ss.csr_matrix([5, 5])
        with pytest.raises(RuntimeError, match=r"compute_biodescriptors\(\) was called on a bound-less stripe"):
            stripe.compute_biodescriptors(matrix)

    def test_window_negative(self):
        stripe = Stripe(
            seed=2,
            top_pers=None,
            horizontal_bounds=(0, 3),
            vertical_bounds=(1, 3),
            where="lower_triangular",
        )
        matrix = ss.csr_matrix([5, 5])
        with pytest.raises(ValueError, match="window cannot be negative"):
            stripe.compute_biodescriptors(matrix, window=-1)
