# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import functools
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from stripepy.utils.unionfind import UnionFind


class Persistence1DTable(object):
    """
    Class to compute and represent a table of 1D persistence values.

    A typical use of this class is as follows:

    persistence = Persistence1DTable.calculate_persistence(...)
    persistence.filter(1, "greater")  # drop points associated with persistence values > 1
    persistence.sort("persistence")  # sort based on persistence values

    persistence.min  # access the sorted and filtered minimum points
    persistence.max  # access the sorted and filtered maximum points

    Attributes
    ----------
    global_minimum: int
        the global minimum of the persistence values
    global_maximum: int
        the global maximum of the persistence values
    min: pd.Series
        persistence values for the minimum points.
        Returns a pd.Series with points as index and persistence as values.
    max: pd.Series
        persistence values for the maximum points
        Returns a pd.Series with points as index and persistence as values.
    level_sets: str
        level sets used to compute the persistence values
    """

    def __init__(
        self,
        pers_of_min_points: pd.Series,
        pers_of_max_points: pd.Series,
        level_sets: str,
    ):
        assert level_sets in {"lower", "upper"}

        self._min = pers_of_min_points.copy()
        self._max = pers_of_max_points.copy()
        self._level_sets = level_sets

        self._min.name = "persistence"
        self._max.name = "persistence"

    @classmethod
    def calculate_persistence(
        cls,
        data: npt.NDArray,
        level_sets: str = "upper",
        min_persistence: Optional[float] = None,
        sort_by: Optional[str] = None,
    ):
        """
        Construct an instance of Persistence1DTable by calculating persistence values for the given data.

        Parameters
        ----------
        data : npt.NDArray
            a 1D numeric vector
        level_sets : str
            should be either "lower" or "upper"
        min_persistence : Optional[float]
            minimum persistence value.
            Values below this cutoff are dropped before constructing the table.
        sort_by : str
            sort values in the table by the given key.

        Returns
        -------
        Persistence1DTable
            table with the calculated persistence values.
        """
        if level_sets not in {"lower", "upper"}:
            raise ValueError("level_sets should be 'lower' or 'upper'")

        pers_of_min_points, pers_of_max_points = Persistence1DTable._compute_persistence(data, level_sets)
        table = cls(
            pers_of_min_points=pers_of_min_points,
            pers_of_max_points=pers_of_max_points,
            level_sets=level_sets,
        )

        if min_persistence is not None:
            table.filter(min_persistence, method="greater")

        if sort_by is not None:
            table.sort(by=sort_by)

        return table

    @functools.cached_property
    def global_minimum(self) -> int:
        if self._level_sets == "lower":
            return self._min.max()
        return self._max.min()

    @functools.cached_property
    def global_maximum(self) -> int:
        if self._level_sets == "lower":
            return self._max.max()
        return self._min.min()

    @property
    def min(self) -> pd.Series:
        return self._min

    @property
    def max(self) -> pd.Series:
        return self._max

    @property
    def level_sets(self) -> str:
        return self._level_sets

    def copy(self):
        """
        Return a copy of the current instance.
        """
        return Persistence1DTable(
            pers_of_min_points=self._min,
            pers_of_max_points=self._max,
            level_sets=self._level_sets,
        )

    def filter(self, persistence: float, method: str = "greater"):
        """
        Remove persistence values from the current table based on the given parameters.

        Parameters
        ----------
        persistence : float
            persistence cutoff
        method : str
            filtering method. Should be one of "greater", "greater_equal", "smaller", "smaller_equal"
        """
        if method == "greater":
            self._min = self._min[self._min > persistence]
            self._max = self._max[self._max > persistence]
        elif method == "greater_equal":
            self._min = self._min[self._min >= persistence]
            self._max = self._max[self._max >= persistence]
        elif method == "smaller":
            self._min = self._min[self._min < persistence]
            self._max = self._max[self._max < persistence]
        elif method == "smaller_equal":
            self._min = self._min[self._min <= persistence]
            self._max = self._max[self._max <= persistence]
        else:
            raise ValueError("method should be one of: greater, greater_equal, smaller, smaller_equal")

    def sort(self, by: str, ascending: bool = True):
        """
        In-place sort the table based on the given key

        Parameters
        ----------
        by : str
            sorting key.
            Should be one of "persistence", "min", "max", or "position".
        ascending : bool
            controls sorting order
        """
        if by == "persistence":
            self._min.sort_values(ascending=ascending, inplace=True, kind="stable")
            self._max.sort_values(ascending=ascending, inplace=True, kind="stable")
        elif by == "min":
            self._min.sort_index(ascending=ascending, inplace=True, kind="stable")
        elif by == "max":
            self._max.sort_index(ascending=ascending, inplace=True, kind="stable")
        elif by == "position":
            self.sort(by="min", ascending=ascending)
            self.sort(by="max", ascending=ascending)
        else:
            raise ValueError("unknown sorting key. Valid values are: min, max, persistence, or position.")

    # Implementation adapted from the library by Tino Weinkauf downloadable at:
    # https://www.csc.kth.se/~weinkauf/notes/persistence1d.html

    # ATT: in its original implementation, the terms minima/maxima/extrema were sometimes used in place of
    # minimum/maximum/extremum points. This notation is here fixed.
    # For more details, consult: https://en.wikipedia.org/wiki/Maximum_and_minimum

    @staticmethod
    def _compute_persistence(data: npt.NDArray, level_sets: str) -> Tuple[pd.Series, pd.Series]:
        """
        This function finds local extrema and their persistence in one-dimensional data w.r.t. lower (default,
        level_sets="lower") or upper (level_sets="upper") level sets.

        Local minima and local maxima are extracted, paired, and returned together with their persistence.
        For level_sets="lower", the global minimum is extracted as well.
        For level_sets="upper", the global maximum is extracted as well.

        We assume a connected one-dimensional domain.

        Short explanation for the case of level_sets=="lower" (the case of level_sets="upper" is analogous).

        This function returns a Persistence1DTable object.

        Original implementation by Tino Weinkauf
        Modified implementation by Andrea Raffo and Roberto Rossini
        """
        assert level_sets in {"lower", "upper"}

        # Number of data to break ties (leftmost index comes first):
        num_elements = len(data)
        sorted_data_idx = np.argsort(data, stable=True)
        if level_sets == "upper":
            sorted_data_idx = sorted_data_idx[::-1]

        # Get a union find data structure:
        uf = UnionFind(num_elements)

        # Extrema paired with topological persistence:
        min_points = np.empty(num_elements + 1, dtype=int)
        max_points = np.empty(num_elements + 1, dtype=int)
        persistence_values = np.empty(num_elements + 1, dtype=float)

        # Watershed:
        value_idx = 0
        for data_idx in sorted_data_idx:

            # Get neighborhood indices:
            left_idx = max(data_idx - 1, 0)
            right_idx = min(data_idx + 1, num_elements - 1)

            # Count number of components in neighborhood:
            neighbor_components = [
                uf.Find(neighbor) for neighbor in [left_idx, right_idx] if uf.Find(neighbor) != UnionFind.NOSET
            ]
            num_neighbor_components = len(neighbor_components)

            if num_neighbor_components == 0:
                # Create a new component:
                uf.MakeSet(data_idx)
                continue
            if num_neighbor_components == 1:
                # Extend the one and only component in the neighborhood
                # Note that NeighborComponents[0] holds the root of a component, since we called Find() earlier to retrieve
                # it!
                uf.ExtendSetByID(neighbor_components[0], data_idx)
                continue

            if level_sets == "lower":
                # Merge the two components on either side of the current point:
                idx_lowest_minimum = neighbor_components[np.argmin(data[neighbor_components])]
                idx_highest_minimum = [comp for comp in neighbor_components if comp != idx_lowest_minimum][0]
                uf.ExtendSetByID(idx_lowest_minimum, data_idx)
                uf.Union(idx_highest_minimum, idx_lowest_minimum)

                # Record the two paired extrema: index of minimum, index of maximum, persistence value:
                persistence = data[data_idx] - data[idx_highest_minimum]
                min_points[value_idx] = idx_highest_minimum
                max_points[value_idx] = data_idx
                persistence_values[value_idx] = persistence
            else:
                # Merge the two components on either side of the current point:
                idx_highest_maximum = neighbor_components[np.argmax(data[neighbor_components])]
                idx_lowest_maximum = [comp for comp in neighbor_components if comp != idx_highest_maximum][0]
                uf.ExtendSetByID(idx_highest_maximum, data_idx)
                uf.Union(idx_lowest_maximum, idx_highest_maximum)

                # Record the two paired extrema: index of minimum, index of maximum, persistence value:
                persistence = data[idx_lowest_maximum] - data[data_idx]
                min_points[value_idx] = data_idx
                max_points[value_idx] = idx_lowest_maximum
                persistence_values[value_idx] = persistence

            value_idx += 1

        persistence_values[value_idx] = np.inf
        if level_sets == "lower":
            min_points[value_idx] = uf.Find(0)
            value_idx += 1
            return (
                pd.Series(
                    index=min_points[:value_idx],
                    data=persistence_values[:value_idx],
                    name="persistence",
                ),
                pd.Series(
                    index=max_points[: value_idx - 1],
                    data=persistence_values[: value_idx - 1],
                    name="persistence",
                ),
            )

        max_points[value_idx] = uf.Find(0)
        value_idx += 1
        return (
            pd.Series(
                index=min_points[: value_idx - 1],
                data=persistence_values[: value_idx - 1],
                name="persistence",
            ),
            pd.Series(
                index=max_points[:value_idx],
                data=persistence_values[:value_idx],
                name="persistence",
            ),
        )
