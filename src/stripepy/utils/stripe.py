import math
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy.sparse as ss


class Stripe(object):
    """
    TODO document
    """

    def __init__(
        self,
        seed: int,
        top_pers: Union[float, None],
        horizontal_bounds: Union[Tuple[int, int], None] = None,
        vertical_bounds: Union[Tuple[int, int], None] = None,
        where: Union[str, None] = None,
    ):
        """
        TODO document
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
        if horizontal_bounds is not None:
            self.set_horizontal_bounds(*horizontal_bounds)

        self._bottom_bound = None
        self._top_bound = None
        if vertical_bounds is not None:
            self.set_vertical_bounds(*vertical_bounds)

        self._five_number = None
        self._inner_mean = None
        self._inner_std = None
        self._outer_lmean = None
        self._outer_rmean = None

    @staticmethod
    def _infer_location(seed: int, top_bound: int, bottom_bound: int) -> str:
        # TODO this check is temporarily disabled as it fails when processing stripes from chromosomes that are mostly empty
        # if bottom_bound == top_bound:
        #    raise ValueError(f"unable to infer stripe location: stripe bottom and top bounds are identical ({top_bound})")

        if bottom_bound > seed:
            return "lower_triangular"
        # TODO the equal check should be removed as is not correct
        if top_bound <= seed:
            return "upper_triangular"

        NotImplementedError

    def _compute_convex_comp(self) -> int:
        cfx1 = 0.99
        cfx2 = 0.01

        if self.upper_triangular:
            cfx1, cfx2 = cfx2, cfx1

        return int(round(cfx1 * self._top_bound + cfx2 * self._bottom_bound))

    def _slice_matrix(self, I: ss.csr_matrix) -> npt.NDArray:
        convex_comb = self._compute_convex_comp()

        if self.lower_triangular:
            rows = slice(convex_comb, self._bottom_bound)
            cols = slice(self._left_bound, self._right_bound)
            return I[rows, :].tocsc()[:, cols].toarray()

        rows = slice(self._top_bound, convex_comb)
        cols = slice(self._left_bound, self._right_bound)
        return I[rows, :].tocsc()[:, cols].toarray()

    @staticmethod
    def _compute_inner_descriptors(I: npt.NDArray) -> Tuple[npt.NDArray[float], float, float]:
        return np.percentile(I, [0, 25, 50, 75, 100]), np.mean(I), np.std(I)

    def _compute_lmean(self, I: ss.csr_matrix, window: int) -> float:
        """
        Compute the mean intensity for the left neighborhood
        """
        assert window >= 0

        new_bound = max(0, self._left_bound - window)
        if new_bound == self._left_bound:
            return math.nan

        convex_comb = self._compute_convex_comp()
        if self.lower_triangular:
            submatrix = I[convex_comb : self._bottom_bound, new_bound : self._left_bound]
        else:
            submatrix = I[self._top_bound : convex_comb, new_bound : self._left_bound]

        return submatrix.mean()

    def _compute_rmean(self, I: ss.csr_matrix, window: int) -> float:
        """
        Compute the mean intensity for the right neighborhood
        """
        assert window >= 0

        new_bound = min(I.shape[1], self._right_bound + window)

        if new_bound == self._right_bound:
            return math.nan

        convex_comb = self._compute_convex_comp()
        if self.lower_triangular:
            submatrix = I[convex_comb : self._bottom_bound, self._right_bound : new_bound]
        else:
            submatrix = I[self._top_bound : convex_comb, self._right_bound : new_bound]

        return submatrix.mean()

    def _all_bounds_set(self) -> bool:
        return all((x is not None for x in [self._left_bound, self._right_bound, self._bottom_bound, self._top_bound]))

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
    def five_number(self) -> npt.NDArray[float]:  # TODO find a better name
        if self._five_number is None:
            raise RuntimeError(
                "caught an attempt to access five_number property before compute_biodescriptors() was called"
            )

        return self._five_number

    @property
    def outer_lmean(self) -> float:
        if self._outer_lmean is None:
            raise RuntimeError(
                "caught an attempt to access outer_lmean property before compute_biodescriptors() was called"
            )

        return self._outer_lmean

    @property
    def outer_rmean(self) -> float:
        if self._outer_rmean is None:
            raise RuntimeError(
                "caught an attempt to access outer_rmean property before compute_biodescriptors() was called"
            )

        return self._outer_rmean

    @property
    def outer_mean(self) -> float:
        if self.outer_rmean == -1:
            assert self.outer_lmean == -1
            return -1.0

        if math.isnan(self.outer_rmean):
            return self.outer_lmean

        if math.isnan(self.outer_lmean):
            return self.outer_rmean

        return (self.outer_lmean + self.outer_rmean) / 2

    @property
    def rel_change(self) -> float:
        if self.outer_mean <= 0:
            return -1.0

        return abs(self.inner_mean - self.outer_mean) / self.outer_mean * 100

    def set_horizontal_bounds(self, left_bound: int, right_bound: int):
        if self._left_bound is not None:
            assert self._right_bound is not None
            raise RuntimeError("horizontal stripe bounds have already been set")

        if not left_bound <= self._seed <= right_bound:
            raise ValueError(
                f"horizontal bounds must enclose the seed position: seed={self._seed}, {left_bound=}, {right_bound=}"
            )

        self._left_bound = left_bound
        self._right_bound = right_bound

    def set_vertical_bounds(self, top_bound: int, bottom_bound: int):
        if self._bottom_bound is not None:
            assert self._top_bound is not None
            raise RuntimeError("vertical stripe bounds have already been set")

        if top_bound > bottom_bound:
            raise ValueError(
                f"the lower vertical bound must be greater than the upper vertical bound: {top_bound=}, {bottom_bound=}"
            )

        self._top_bound = top_bound
        self._bottom_bound = bottom_bound

        computed_where = self._infer_location(self._seed, self._top_bound, self._bottom_bound)

        if self._where is not None and computed_where != self._where:
            raise RuntimeError(
                f"computed location does not match the provided stripe location: computed={computed_where}, expected={self._where}"
            )

        self._where = computed_where

    def compute_biodescriptors(self, I: ss.csr_matrix, window: int = 3):
        if not self._all_bounds_set():
            raise RuntimeError("compute_biodescriptors() was called on a bound-less stripe")

        if window < 0:
            raise ValueError("window cannot be negative")

        restrI = self._slice_matrix(I)

        # ATT This can avoid empty stripes, which can occur e.g. when the column has (approximately) constant entries
        if np.prod(restrI.size) == 0:
            self._five_number = np.array([-1.0] * 5)
            self._inner_mean = -1.0
            self._inner_std = -1.0
            self._outer_lmean = -1.0
            self._outer_rmean = -1.0
            return

        self._five_number, self._inner_mean, self._inner_std = self._compute_inner_descriptors(restrI)

        # Mean intensity - left and right neighborhoods:
        self._outer_lmean = self._compute_lmean(I, window)
        self._outer_rmean = self._compute_rmean(I, window)
