# Copyright (C) 2024 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import warnings
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as ss
from numpy.typing import NDArray

from stripepy.utils.shared_sparse_matrix import SparseMatrix


class Stripe(object):
    """
    A class used to represent architectural stripes.
    This class takes care of validating stripe coordinates and computing several descriptive statistics.

    This is how this class should be used:

        * Initialize the class by providing at least the seed position
        * At a later time, set the vertical and horizontal boundaries by calling `set_horizontal_bounds()` and `set_vertical_bounds()`
        * Finally, call `compute_biodescriptors()` to compute and store the descriptive statistics

    The stripe properties and statistics can now be accessed through the attributes listed below.

    Attributes representing the descriptive statistics return negative values to signal that it was not
    possible to compute the statistics for the current Stripe instance.

    Attributes
    ----------

    seed: int
        the stripe seed
    top_persistence: Union[float, None]
        the topological persistence
    lower_triangular: bool
        true when the stripe extends in the lower-triangular portion of the matrix
    upper_triangular: bool
        true when the stripe extends in the upper-triangular portion of the matrix
    left_bound: int
        the left bound of the stripe
    right_bound: int
        the right bound of the stripe
    top_bound: int
        the top bound of the stripe
    bottom_bound: int
        the bottom bound of the stripe
    inner_mean: float
        the average number of interactions within the stripe
    inner_std: float
        the standard deviation of the number of interactions within the stripe
    five_number: NDArray[float]
        a vector of five numbers corresponding to the 0, 25, 50, 75, and 100 percentiles of the number of within-stripe interactions
    outer_lsum: float
        the sum of interactions in the band to the left of the stripe
    outer_rsum: float
        the sum of interactions in the band to the right of the stripe
    outer_lsize float
        the number of entries in the band to the left of the stripe
    outer_rsize float
        the number of entries in the band to the right of the stripe
    outer_lmean: float
        the average number of interactions in the band to the left of the stripe
    outer_rmean: float
        the average number of interactions in the band to the right of the stripe
    outer_mean: float
        the average number of interactions in the bands to the left and right of the stripe
    rel_change: float
        the ratio of the average number of interactions within the stripe and in the neighborhood outside of the stripe
    """

    def __init__(
        self,
        seed: int,
        top_pers: Optional[float],
        horizontal_bounds: Optional[Tuple[int, int]] = None,
        vertical_bounds: Optional[Tuple[int, int]] = None,
        where: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        seed: int
            the stripe seed position
        top_pers: Optional[float]
            the topological persistence of the seed
        horizontal_bounds: Optional[Tuple[int, int]]
            the horizontal bounds of the stripe
        vertical_bounds: Optional[Tuple[int, int]]
            the_vertical bounds of the stripe
        where: Optional[str]
            the location of the stripe: should be "upper_triangular" or "lower_triangular".
            When provided, this is used validate the coordinates set when calling `set_horizontal_bounds()` and `set_vertical_bounds()`.
        """
        if seed < 0:
            raise ValueError("seed must be a non-negative integral number")
        self._seed = seed

        if top_pers is not None and top_pers < 0:
            raise ValueError("when not None, top_pers must be a positive number")
        self._persistence = top_pers

        valid_locations = {"upper_triangular", "lower_triangular"}
        if where is not None and where not in valid_locations:
            raise ValueError(f"when specified, where must be one of {tuple(valid_locations)}")
        self._where = where

        self._left_bound = None
        self._right_bound = None
        self._bottom_bound = None
        self._top_bound = None
        if horizontal_bounds is not None:
            self.set_horizontal_bounds(*horizontal_bounds)

        self._bottom_bound = None
        self._top_bound = None
        if vertical_bounds is not None:
            self.set_vertical_bounds(*vertical_bounds)

        self._five_number = None
        self._inner_mean = None
        self._inner_std = None
        self._outer_lsum = None
        self._outer_rsum = None
        self._outer_lsize = None
        self._outer_rsize = None

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def top_persistence(self) -> Optional[float]:
        return self._persistence

    @property
    def lower_triangular(self) -> bool:
        return self._where == "lower_triangular"

    @property
    def upper_triangular(self) -> bool:
        return self._where == "upper_triangular"

    @property
    def left_bound(self) -> int:
        if self._left_bound is None:
            raise RuntimeError("left_bound has not been set")
        return self._left_bound

    @property
    def right_bound(self) -> int:
        if self._right_bound is None:
            raise RuntimeError("right_bound has not been set")
        return self._right_bound

    @property
    def top_bound(self) -> int:
        if self._top_bound is None:
            raise RuntimeError("top_bound has not been set")
        return self._top_bound

    @property
    def bottom_bound(self) -> int:
        if self._bottom_bound is None:
            raise RuntimeError("bottom_bound has not been set")
        return self._bottom_bound

    @property
    def inner_mean(self) -> float:
        if self._inner_mean is None:
            raise RuntimeError(
                "caught an attempt to access inner_mean property before compute_biodescriptors() was called"
            )

        return self._inner_mean

    @property
    def inner_std(self) -> float:
        if self._inner_std is None:
            raise RuntimeError(
                "caught an attempt to access inner_std property before compute_biodescriptors() was called"
            )

        return self._inner_std

    @property
    def five_number(self) -> NDArray[float]:
        if self._five_number is None:
            raise RuntimeError(
                "caught an attempt to access five_number property before compute_biodescriptors() was called"
            )

        return self._five_number

    @property
    def outer_lsum(self) -> float:
        if self._outer_lsum is None:
            raise RuntimeError(
                "caught an attempt to access outer_lsum property before compute_biodescriptors() was called"
            )

        return self._outer_lsum

    @property
    def outer_rsum(self) -> float:
        if self._outer_rsum is None:
            raise RuntimeError(
                "caught an attempt to access outer_rsum property before compute_biodescriptors() was called"
            )

        return self._outer_rsum

    @property
    def outer_lsize(self) -> float:
        if self._outer_lsize is None:
            raise RuntimeError(
                "caught an attempt to access outer_lsize property before compute_biodescriptors() was called"
            )

        return self._outer_lsize

    @property
    def outer_rsize(self) -> float:
        if self._outer_rsize is None:
            raise RuntimeError(
                "caught an attempt to access outer_rsize property before compute_biodescriptors() was called"
            )

        return self._outer_rsize

    @property
    def outer_lmean(self) -> float:
        if self._outer_lsum is None or self._outer_lsize is None:
            raise RuntimeError(
                "caught an attempt to access outer_lmean property before compute_biodescriptors() was called"
            )

        # Suppress divide-by-zero warning:
        with warnings.catch_warnings():
            warnings.filterwarnings(category=RuntimeWarning, action="ignore")
            return self._outer_lsum / self._outer_lsize

    @property
    def outer_rmean(self) -> float:
        if self._outer_rsum is None or self._outer_rsize is None:
            raise RuntimeError(
                "caught an attempt to access outer_rmean property before compute_biodescriptors() was called"
            )

        # Suppress divide-by-zero warning:
        with warnings.catch_warnings():
            warnings.filterwarnings(category=RuntimeWarning, action="ignore")
            return self._outer_rsum / self._outer_rsize

    @property
    def outer_mean(self) -> float:
        # Suppress divide-by-zero warning:
        with warnings.catch_warnings():
            warnings.filterwarnings(category=RuntimeWarning, action="ignore")
            return (self._outer_lsum + self._outer_rsum) / (self._outer_lsize + self._outer_rsize)

    @property
    def rel_change(self) -> float:
        outer_mean = self.outer_mean

        # Suppress divide-by-zero warning:
        with warnings.catch_warnings():
            warnings.filterwarnings(category=RuntimeWarning, action="ignore")
            return abs(self.inner_mean - outer_mean) / outer_mean * 100

    def set_horizontal_bounds(self, left_bound: int, right_bound: int):
        """
        Set the horizontal bounds for the stripe.
        This function raises an exception when the coordinates have already been set or when the
        given coordinates are incompatible with the seed position.

        Parameters
        ----------
        left_bound: int
        right_bound: int
        """
        if self._left_bound is not None:
            assert self._right_bound is not None
            raise RuntimeError("horizontal stripe bounds have already been set")

        if left_bound < 0 or right_bound < 0:
            raise ValueError("stripe bounds must be positive integers")

        if not left_bound <= self._seed <= right_bound:
            raise ValueError(
                f"horizontal bounds must enclose the seed position: seed={self._seed}, {left_bound=}, {right_bound=}"
            )

        self._left_bound = left_bound
        self._right_bound = right_bound

    def set_vertical_bounds(self, top_bound: int, bottom_bound: int):
        """
        Set the vertical bounds for the stripe.
        This function raises an exception when the coordinates have already been set or when the
        given coordinates are incompatible with the seed position and/or the where location.

        Parameters
        ----------
        top_bound: int
        bottom_bound: int
        """
        if self._bottom_bound is not None:
            assert self._top_bound is not None
            raise RuntimeError("vertical stripe bounds have already been set")

        if top_bound < 0 or bottom_bound < 0:
            raise ValueError("stripe bounds must be positive integers")

        if top_bound > bottom_bound:
            raise ValueError(
                f"the lower vertical bound must be greater than the upper vertical bound: {top_bound=}, {bottom_bound=}"
            )

        computed_where = self._infer_location(self._seed, top_bound, bottom_bound)

        if self._where is not None and computed_where != self._where:
            raise RuntimeError(
                f"computed location does not match the provided stripe location: computed={computed_where}, expected={self._where}"
            )

        self._top_bound = top_bound
        self._bottom_bound = bottom_bound

        self._where = computed_where

    def compute_biodescriptors(self, matrix: SparseMatrix, window: int = 3):
        """
        Use the sparse matrix to compute various descriptive statistics.
        Statistics are stored in the current Stripe instance.
        This function raises an exception when it is called before the stripe bounds have been set.

        Parameters
        ----------
        matrix: SparseMatrix
            the sparse matrix from which the stripe originated
        window: int
            window size used to compute statistics to the left and right of the stripe
        """
        if not self._all_bounds_set():
            raise RuntimeError("compute_biodescriptors() was called on a bound-less stripe")

        if window < 0:
            raise ValueError("window cannot be negative")

        left_submatrix, stripe_submatrix, right_submatrix = self._slice_matrix(matrix, window)

        # Suppress divide-by-zero warning:
        with warnings.catch_warnings():
            warnings.filterwarnings(category=RuntimeWarning, action="ignore")
            self._five_number, self._inner_mean, self._inner_std = self._compute_inner_descriptors(stripe_submatrix)

            # Compute outer descriptors
            self._outer_lsum, self._outer_lsize = self._compute_outer_descriptors(left_submatrix)
            self._outer_rsum, self._outer_rsize = self._compute_outer_descriptors(right_submatrix)

    def set_biodescriptors(
        self,
        inner_mean: float,
        inner_std: float,
        outer_lsum: float,
        outer_lsize: int,
        outer_rsum: float,
        outer_rsize: int,
        five_number: NDArray[float],
    ):
        """
        Set the stripe biodescriptors based on pre-computed statistics.

        inner_mean: float

        inner_std: float

        outer_lsum: float

        outer_lsize: int

        outer_rsum: float

        outer_rsize: int

        five_number: NDArray[float]
        """
        if not self._all_bounds_set():
            raise RuntimeError("set_biodescriptors() was called on a bound-less stripe")

        self._inner_mean = inner_mean
        self._inner_std = inner_std
        self._outer_lsum = outer_lsum
        self._outer_lsize = outer_lsize
        self._outer_rsum = outer_rsum
        self._outer_rsize = outer_rsize
        self._five_number = five_number

    def _all_bounds_set(self) -> bool:
        return self._horizontal_bounds_set() and self._vertical_bounds_set()

    def _horizontal_bounds_set(self) -> bool:
        return self._left_bound is not None and self._right_bound is not None

    def _vertical_bounds_set(self) -> bool:
        return self._top_bound is not None and self._bottom_bound is not None

    @staticmethod
    def _infer_location(seed: int, top_bound: int, bottom_bound: int) -> str:
        # TODO is it ok that when bottom_bound==seed==top_bound the stripe is considered as upper_triangular?

        if bottom_bound == seed:
            return "upper_triangular"
        if top_bound == seed:
            return "lower_triangular"

        raise ValueError(f"At least one of {top_bound=} and {bottom_bound=} must be equal to {seed=}")

    def _compute_convex_comp(self) -> int:
        cfx1 = 0.99
        cfx2 = 0.01

        if self.upper_triangular:
            cfx1, cfx2 = cfx2, cfx1

        return int(round(cfx1 * self._top_bound + cfx2 * self._bottom_bound))

    def _pad_horizontal_domain(self, matrix: SparseMatrix, padding: int) -> Tuple[int, int]:
        j0 = max(0, self._left_bound - padding)
        j1 = min(self._right_bound + padding + 1, matrix.shape[1])
        return j0, j1

    def _pad_vertical_domain(self, matrix: SparseMatrix, j0: int, j1: int) -> Tuple[int, int]:
        convex_comb = self._compute_convex_comp()
        if self.lower_triangular:
            # convex_comb replaces _top_bound
            convex_comb = self._top_bound
            i0 = j0
            i1 = min(j1 + (self._bottom_bound - convex_comb), matrix.shape[0])
        else:
            # convex_comb replaces _bottom_bound
            convex_comb = self._bottom_bound
            i0 = max(0, j0 - (convex_comb - self._top_bound))
            i1 = min(j1, matrix.shape[0])
        return i0, i1

    def _slice_matrix(self, matrix: SparseMatrix, padding: int) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Extract the minimum-bounding matrix containing the sub-matrices corresponding to: the current stripe, as well
        as its left and right neighbors; values outside the sub-matrices are initialized to np.nan. These sub-matrices
        are computed as k diagonals around the main diagonal (either in the lower or the upper triangular part), where
        k is the height of the stripe.

        Parameters
        ----------
        matrix: SparseMatrix
            matrix to be sliced
        padding: int
            width of the left and right neighborhoods

        Returns
        -------
        NDArray
            The matrix left neighborhood
        NDArray
            The matrix stripe
        NDArray
            The matrix right neighborhood
        """
        stripe_height = self._bottom_bound - self._top_bound + 1

        # Compute the indices for the minimum-bounding matrix
        j0, j1 = self._pad_horizontal_domain(matrix, padding)
        i0, i1 = self._pad_vertical_domain(matrix, j0, j1)

        # The sub-matrix we here obtained is the smallest rectangular matrix enclosing the desired matrices
        if isinstance(matrix, ss.csr_matrix):
            submatrix = matrix[i0:i1, :].tocsc()[:, j0:j1].toarray()
        else:
            submatrix = matrix[:, j0:j1].tocsr()[i0:i1, :].toarray()

        # Compute the padded matrix shape
        expected_top_padding = self._seed - self._left_bound + padding
        expected_bottom_padding = self._right_bound - self._seed + padding
        expected_width = self._right_bound - self._left_bound + 1 + 2 * padding
        expected_height = expected_top_padding + stripe_height + expected_bottom_padding

        # Initialized the padded sub-matrix to nan. Note that the previously sliced sub-matrix will fit into this
        # matrix.
        padded_submatrix = np.full((expected_height, expected_width), np.nan)

        # Compute the number of rows/columns that we can expand through (around the stripe) in the original matrix.
        # This can be less than the expected padding when we are at the left/right ends of the chromosome matrix.
        effective_top_padding = self._top_bound - i0
        effective_left_padding = self._left_bound - j0
        effective_right_padding = j1 - self._right_bound - 1
        effective_bottom_padding = i1 - self._bottom_bound - 1

        # Compute the offsets used to assign the sliced sub-matrix to the padded sub-matrix
        offset_top_rows = expected_top_padding - effective_top_padding
        offset_left_cols = padding - effective_left_padding
        offset_right_cols = effective_right_padding - padding + padded_submatrix.shape[1]
        offset_bottom_rows = effective_bottom_padding - expected_bottom_padding + padded_submatrix.shape[0]

        assert offset_bottom_rows >= offset_top_rows
        assert offset_right_cols >= offset_left_cols

        # Fill the padded sub-matrix
        padded_submatrix[offset_top_rows:offset_bottom_rows, offset_left_cols:offset_right_cols] = submatrix

        # Mask values outside the diagonal window, i.e., the stripe height
        idx1, idx2 = np.triu_indices(
            n=padded_submatrix.shape[0],
            m=padded_submatrix.shape[1],
            k=1,
        )
        idx3, idx4 = np.tril_indices(
            n=padded_submatrix.shape[0],
            m=padded_submatrix.shape[1],
            k=self._top_bound - self._bottom_bound - 1,
        )
        padded_submatrix[idx1, idx2] = np.nan
        padded_submatrix[idx3, idx4] = np.nan

        # Extract the three sub-matrices
        left_submatrix = padded_submatrix[:, :padding]
        stripe_submatrix = padded_submatrix[:, padding:-padding]
        right_submatrix = padded_submatrix[:, -padding:]

        return left_submatrix, stripe_submatrix, right_submatrix

    @staticmethod
    def _compute_inner_descriptors(matrix: NDArray) -> Tuple[NDArray[float], float, float]:
        assert matrix.size > 0

        return np.nanpercentile(matrix, [0, 25, 50, 75, 100]), np.nanmean(matrix), np.nanstd(matrix)  # noqa

    @staticmethod
    def _compute_outer_descriptors(matrix: NDArray) -> Tuple[float, int]:
        """
        Compute the sum and number of entries in the outer-left neighborhood
        """

        assert matrix.size > 0

        return np.nansum(matrix), np.isfinite(matrix).sum()
