# Copyright (C) 2025 Bendik Berg <bendber@ifi.uio.no>
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
import scipy.sparse as ss

from stripepy.utils.stripe import Stripe


@pytest.mark.unit
class TestObjectInitialization:
    def test_negative_seed(self):
        with pytest.raises(ValueError, match="seed must be a non-negative integral number"):
            Stripe(
                seed=-1,
                top_pers=None,
                horizontal_bounds=None,
                vertical_bounds=None,
                where=None,
            )

    def test_negative_top_persistence(self):
        with pytest.raises(ValueError, match="when not None, top_pers must be a positive number"):
            Stripe(
                seed=0,
                top_pers=-1.0,
                horizontal_bounds=None,
                vertical_bounds=None,
                where=None,
            )

    def test_invalid_location(self):
        with pytest.raises(ValueError, match="when specified, where must be one of (.*upper.*|.*lower.*){2}"):
            Stripe(
                seed=5,
                top_pers=None,
                horizontal_bounds=None,
                vertical_bounds=None,
                where="invalid_triangular",
            )

    def test_invalid_vertical_bounds(self):
        with pytest.raises(
            ValueError, match=r"At least one of top_bound=\d+ and bottom_bound=\d+ must be equal to seed=\d+"
        ):
            Stripe(
                seed=5,
                top_pers=None,
                horizontal_bounds=None,
                vertical_bounds=(1, 6),
                where=None,
            )


@pytest.mark.unit
class TestGetters:
    def test_upper_triangular(self):
        stripe = Stripe(
            seed=5,
            top_pers=5.0,
            horizontal_bounds=(4, 6),
            vertical_bounds=(1, 5),
            where="upper_triangular",
        )
        assert stripe.seed == 5
        assert np.isclose(stripe.top_persistence, 5.0)
        assert not stripe.lower_triangular
        assert stripe.upper_triangular
        assert stripe.left_bound == 4
        assert stripe.right_bound == 6
        assert stripe.top_bound == 1
        assert stripe.bottom_bound == 5

    def test_lower_triangular(self):
        stripe = Stripe(
            seed=5,
            top_pers=5.0,
            horizontal_bounds=(4, 6),
            vertical_bounds=(5, 10),
            where="lower_triangular",
        )

        assert stripe.lower_triangular
        assert not stripe.upper_triangular

    def test_access_before_set(self):
        stripe = Stripe(
            seed=5,
            top_pers=None,
            horizontal_bounds=None,
            vertical_bounds=None,
            where=None,
        )

        assert stripe.seed == 5
        assert stripe.top_persistence is None

        with pytest.raises(RuntimeError, match="left_bound has not been set"):
            _ = stripe.left_bound

        with pytest.raises(RuntimeError, match="right_bound has not been set"):
            _ = stripe.right_bound

        with pytest.raises(RuntimeError, match="top_bound has not been set"):
            _ = stripe.top_bound

        with pytest.raises(RuntimeError, match="bottom_bound has not been set"):
            _ = stripe.bottom_bound

        assert not stripe.upper_triangular
        assert not stripe.lower_triangular


@pytest.mark.unit
class TestBoundarySetters:
    def test_set_horizontal_bounds(self):
        stripe = Stripe(
            seed=5,
            top_pers=None,
            horizontal_bounds=None,
            vertical_bounds=None,
            where=None,
        )

        stripe.set_horizontal_bounds(5, 5)

        assert stripe.left_bound == 5
        assert stripe.right_bound == 5

    def test_set_invalid_horizontal_bounds(self):
        stripe = Stripe(
            seed=5,
            top_pers=None,
            horizontal_bounds=None,
            vertical_bounds=None,
            where=None,
        )
        with pytest.raises(
            ValueError,
            match=r"horizontal bounds must enclose the seed position: seed=\d+, left_bound=\d+, right_bound=\d+",
        ):
            stripe.set_horizontal_bounds(6, 6)
        with pytest.raises(
            ValueError,
            match=r"horizontal bounds must enclose the seed position: seed=\d+, left_bound=\d+, right_bound=\d+",
        ):
            stripe.set_horizontal_bounds(4, 4)

    def test_empty_horizontal_domain(self):
        stripe = Stripe(
            seed=5,
            top_pers=None,
            horizontal_bounds=None,
            vertical_bounds=None,
            where=None,
        )
        with pytest.raises(
            ValueError,
            match=r"horizontal bounds must enclose the seed position: seed=\d+, left_bound=\d+, right_bound=\d+",
        ):
            stripe.set_horizontal_bounds(6, 5)

    def test_negative_horizontal_bounds(self):
        stripe = Stripe(
            seed=1,
            top_pers=None,
            horizontal_bounds=None,
            vertical_bounds=None,
            where=None,
        )
        with pytest.raises(ValueError, match="stripe bounds must be positive integers"):
            stripe.set_horizontal_bounds(-1, 1)

    def test_horizontal_bounds_already_set(self):
        stripe = Stripe(
            seed=5,
            top_pers=None,
            horizontal_bounds=(4, 6),
            vertical_bounds=None,
            where=None,
        )

        with pytest.raises(RuntimeError, match="horizontal stripe bounds have already been set"):
            stripe.set_horizontal_bounds(5, 7)

    def test_empty_vertical_domain(self):
        stripe = Stripe(
            seed=5,
            top_pers=None,
            horizontal_bounds=None,
            vertical_bounds=None,
            where=None,
        )

        with pytest.raises(
            ValueError,
            match=r"the lower vertical bound must be greater than the upper vertical bound: top_bound=\d+, bottom_bound=\d+",
        ):
            stripe.set_vertical_bounds(5, 4)

    def test_negative_vertical_bounds(self):
        stripe = Stripe(
            seed=5,
            top_pers=None,
            horizontal_bounds=None,
            vertical_bounds=None,
            where=None,
        )
        with pytest.raises(ValueError, match="stripe bounds must be positive integers"):
            stripe.set_vertical_bounds(-1, 5)

    def test_vertical_bounds_already_set(self):
        stripe = Stripe(
            seed=5,
            top_pers=None,
            horizontal_bounds=None,
            vertical_bounds=(1, 5),
            where=None,
        )

        with pytest.raises(RuntimeError, match="vertical stripe bounds have already been set"):
            stripe.set_vertical_bounds(2, 5)

    def test_invalid_vertical_bounds(self):
        stripe = Stripe(
            seed=1,
            top_pers=None,
            horizontal_bounds=None,
            vertical_bounds=None,
            where="lower_triangular",
        )

        with pytest.raises(
            ValueError, match=r"At least one of top_bound=\d+ and bottom_bound=\d+ must be equal to seed=\d+"
        ):
            stripe.set_vertical_bounds(0, 3)

    def test_location_mismatch(self):
        stripe = Stripe(
            seed=1,
            top_pers=None,
            horizontal_bounds=None,
            vertical_bounds=None,
            where="lower_triangular",
        )

        with pytest.raises(
            RuntimeError,
            match="computed location does not match the provided stripe location: computed=upper_triangular, expected=lower_triangular",
        ):
            stripe.set_vertical_bounds(0, 1)


@pytest.mark.unit
class TestComputeBiodescriptors:
    def test_stripe_in_upper_left_corner_with_empty_lneighborhood(self):
        stripe = Stripe(
            seed=0,
            top_pers=None,
            horizontal_bounds=(0, 2),
            vertical_bounds=(0, 2),
            where="lower_triangular",
        )
        matrix = ss.csr_matrix(
            [
                [5, 0, 0, 2, 0, 0, 0],
                [5, 5, 0, 3, 0, 0, 0],
                [5, 5, 5, 4, 0, 0, 0],
                [0, 5, 5, 1, 0, 0, 0],
                [0, 0, 5, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        )
        stripe.compute_biodescriptors(matrix, window=1)

        assert np.isclose(stripe.inner_mean, 5.0)
        assert np.isclose(stripe.inner_std, 0.0)
        assert np.allclose(stripe.five_number, 5.0)

        assert np.isclose(stripe.outer_lsum, 0.0)
        assert np.isclose(stripe.outer_rsum, 3.0)
        assert stripe.outer_lsize == 0
        assert stripe.outer_rsize == 3
        assert np.isnan(stripe.outer_lmean)
        assert np.isclose(stripe.outer_rmean, 1.0)
        assert np.isclose(stripe.outer_mean, 1.0)
        assert np.isclose(stripe.rel_change, 400.0)

    def test_stripe_in_lower_middle(self):
        stripe = Stripe(
            seed=3,
            top_pers=None,
            horizontal_bounds=(3, 4),
            vertical_bounds=(3, 4),
            where="lower_triangular",
        )
        matrix = ss.csr_matrix(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 4, 0, 0, 0, 0],
                [0, 0, 4, 4, 0, 0, 0],
                [0, 0, 0, 4, 4, 0, 0],
                [0, 0, 0, 0, 4, 1, 2],
                [0, 0, 0, 0, 0, 1, 1],
            ]
        )

        stripe.compute_biodescriptors(matrix, window=1)

        assert np.isclose(stripe.inner_mean, 4.0)
        assert np.isclose(stripe.inner_std, 0.0)
        assert np.allclose(stripe.five_number, 4.0)

        assert np.isclose(stripe.outer_lsum, 8.0)
        assert np.isclose(stripe.outer_rsum, 2.0)
        assert stripe.outer_lsize == 2
        assert stripe.outer_rsize == 2
        assert np.isclose(stripe.outer_mean, 2.5)
        assert np.isclose(stripe.rel_change, 60.0)

    def test_stripe_in_lower_right_corner(self):
        stripe = Stripe(
            seed=4,
            top_pers=None,
            horizontal_bounds=(4, 5),
            vertical_bounds=(4, 5),
            where="lower_triangular",
        )
        matrix = ss.csr_matrix(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 3, 0, 0],
                [0, 0, 0, 0, 3, 3, 0],
                [0, 0, 0, 0, 0, 3, 5],
            ]
        )
        stripe.compute_biodescriptors(matrix)

        assert np.isclose(stripe.inner_mean, 3.0)
        assert np.isclose(stripe.inner_std, 0.0)
        assert np.allclose(stripe.five_number, 3.0)

        assert np.isclose(stripe.outer_lsum, 2.0)
        assert np.isclose(stripe.outer_rsum, 5.0)
        assert stripe.outer_lsize == 6
        assert stripe.outer_rsize == 1
        assert np.isclose(stripe.outer_mean, 1.0)
        assert np.isclose(stripe.rel_change, 200.0)

    def test_stripe_in_lower_right_corner_with_empty_rneighborhood(self):
        stripe = Stripe(
            seed=4,
            top_pers=None,
            horizontal_bounds=(4, 6),
            vertical_bounds=(4, 6),
            where="lower_triangular",
        )
        matrix = ss.csr_matrix(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 2, 0, 0, 0],
                [0, 0, 1, 2, 3, 0, 0],
                [0, 0, 0, 2, 3, 3, 0],
                [0, 0, 0, 0, 3, 3, 3],
            ]
        )
        stripe.compute_biodescriptors(matrix)

        assert np.isclose(stripe.outer_rsum, 0.0)
        assert stripe.outer_rsize == 0
        assert np.isclose(stripe.outer_lmean, 1.0)
        assert np.isnan(stripe.outer_rmean)
        assert np.isclose(stripe.outer_mean, 1.0)

    def test_stripe_in_upper_middle(self):
        stripe = Stripe(
            seed=3,
            top_pers=None,
            horizontal_bounds=(3, 4),
            vertical_bounds=(2, 3),
            where="upper_triangular",
        )
        matrix = ss.csc_matrix(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 4, 0, 0, 0, 0],
                [0, 0, 4, 4, 0, 0, 0],
                [0, 0, 0, 4, 4, 0, 0],
                [0, 0, 0, 0, 4, 1, 1],
                [0, 0, 0, 0, 0, 1, 2],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )

        with pytest.raises(
            RuntimeError,
            match=r"caught an attempt to access inner_mean property before compute_biodescriptors\(\) was called",
        ):
            _ = stripe.inner_mean

        with pytest.raises(
            RuntimeError,
            match=r"caught an attempt to access inner_std property before compute_biodescriptors\(\) was called",
        ):
            _ = stripe.inner_std

        with pytest.raises(
            RuntimeError,
            match=r"caught an attempt to access five_number property before compute_biodescriptors\(\) was called",
        ):
            _ = stripe.five_number

        with pytest.raises(
            RuntimeError,
            match=r"caught an attempt to access outer_lsum property before compute_biodescriptors\(\) was called",
        ):
            _ = stripe.outer_lsum

        with pytest.raises(
            RuntimeError,
            match=r"caught an attempt to access outer_rsum property before compute_biodescriptors\(\) was called",
        ):
            _ = stripe.outer_rsum

        with pytest.raises(
            RuntimeError,
            match=r"caught an attempt to access outer_lsize property before compute_biodescriptors\(\) was called",
        ):
            _ = stripe.outer_lsize

        with pytest.raises(
            RuntimeError,
            match=r"caught an attempt to access outer_rsize property before compute_biodescriptors\(\) was called",
        ):
            _ = stripe.outer_rsize

        with pytest.raises(
            RuntimeError,
            match=r"caught an attempt to access outer_lmean property before compute_biodescriptors\(\) was called",
        ):
            _ = stripe.outer_lmean

        with pytest.raises(
            RuntimeError,
            match=r"caught an attempt to access outer_rmean property before compute_biodescriptors\(\) was called",
        ):
            _ = stripe.outer_rmean

        with pytest.raises(
            RuntimeError,
            match=r"caught an attempt to access outer_lmean property before compute_biodescriptors\(\) was called",
        ):
            _ = stripe.outer_lmean

        with pytest.raises(
            RuntimeError,
            match=r"caught an attempt to access outer_rmean property before compute_biodescriptors\(\) was called",
        ):
            _ = stripe.outer_rmean

        stripe.compute_biodescriptors(matrix, window=1)

        assert np.isclose(stripe.inner_mean, 4.0)
        assert np.isclose(stripe.inner_std, 0.0)
        assert np.allclose(stripe.five_number, 4.0)

        assert np.isclose(stripe.outer_lsum, 8.0)
        assert np.isclose(stripe.outer_rsum, 2.0)
        assert stripe.outer_lsize == 2
        assert stripe.outer_rsize == 2
        assert np.isclose(stripe.outer_lmean, 4.0)
        assert np.isclose(stripe.outer_rmean, 1.0)
        assert np.isclose(stripe.outer_mean, 2.5)
        assert np.isclose(stripe.rel_change, 60.0)


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
            vertical_bounds=(2, 3),
            where="lower_triangular",
        )
        matrix = ss.csr_matrix([5, 5])
        with pytest.raises(ValueError, match="window cannot be negative"):
            stripe.compute_biodescriptors(matrix, window=-1)


@pytest.mark.unit
class TestSetBioescriptors:

    def test_call_biodescriptors_boundless_stripe(self):
        stripe = Stripe(
            seed=1,
            top_pers=None,
            horizontal_bounds=None,
            vertical_bounds=None,
            where="lower_triangular",
        )

        toy_inner_mean = 0.0
        toy_inner_std = 1.0
        toy_outer_lsum = 3.0
        toy_lsize = 3
        toy_outer_rsum = 3.0
        toy_rsize = 3
        toy_five_number = np.full(5, 5.0)

        with pytest.raises(RuntimeError, match=r"set_biodescriptors\(\) was called on a bound-less stripe"):
            stripe.set_biodescriptors(
                toy_inner_mean,
                toy_inner_std,
                toy_outer_lsum,
                toy_lsize,
                toy_outer_rsum,
                toy_rsize,
                toy_five_number,
            )

    def test_set_biodescriptors(self):
        stripe = Stripe(
            seed=1,
            top_pers=None,
            horizontal_bounds=None,
            vertical_bounds=None,
            where="lower_triangular",
        )

        stripe.set_vertical_bounds(1, 3)
        stripe.set_horizontal_bounds(1, 2)

        toy_inner_mean = 0.0
        toy_inner_std = 1.0
        toy_outer_lsum = 3.0
        toy_lsize = 3
        toy_outer_rsum = 3.0
        toy_rsize = 3
        toy_five_number = np.full(5, 5.0)
        stripe.set_biodescriptors(
            toy_inner_mean,
            toy_inner_std,
            toy_outer_lsum,
            toy_lsize,
            toy_outer_rsum,
            toy_rsize,
            toy_five_number,
        )

        assert np.isclose(stripe.inner_mean, toy_inner_mean)
        assert np.isclose(stripe.inner_std, toy_inner_std)
        assert np.isclose(stripe.outer_lsum, toy_outer_lsum)
        assert np.isclose(stripe.outer_rsum, toy_outer_rsum)
        assert stripe.outer_lsize == toy_lsize
        assert stripe.outer_rsize == toy_rsize
        assert np.allclose(stripe.five_number, toy_five_number)
