# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import functools
import itertools
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as ss
import structlog

from stripepy.algorithms.regressions import compute_wQISA_predictions
from stripepy.data_structures import Persistence1DTable, SparseMatrix, get_shared_state
from stripepy.utils import pretty_format_elapsed_time


def _shift_vector_left(
    v: npt.NDArray[float],
    padding: float,
) -> npt.NDArray[float]:
    """
    Shift elements in the given vector to the left by one position and pad the remaining values.

    Parameters
    ----------
    v: npt.NDArray[float]
        vector to be shifted
    padding: float
        padding value

    Returns
    -------
    npt.NDArray[float]
        shifted and padded vector
    """
    # assertion disabled for performance reasons
    # assert len(v) != 0

    vs = np.empty_like(v)
    vs[:-1] = v[1:]
    vs[-1] = padding

    return vs


def _find_horizontal_domain(
    profile: npt.NDArray[float],
    params: Tuple[int, int, int, float],
    max_width: int = int(1e9),
) -> Tuple[int, int]:
    """
    Find the horizontal domains for the given profile (i.e. pseudo-distribution).

    Parameters
    ----------
    profile : npt.NDArray[float]
        pseudo-distribution vector
    params: Tuple[int, int, int, float]
        tuple consisting of:
        1) maxima
        2) minima to the left of 1)
        3) minima to the right of 1)
        4) global maxima for the given profile
    max_width : int
        maximum width allowed

    Returns
    -------
    int
        left boundary of the domain
    int
        right boundary of the domain
    """

    # Unpacking:
    maxima, left_minima, right_minima, max_profile_value = params

    # assertions disabled for performance reasons
    # assert left_minima >= 0
    # assert right_minima < len(profile)
    # assert left_minima <= maxima <= right_minima

    # Left side of the candidate
    left_interval = np.flip(profile[left_minima : maxima + 1])
    left_interval_shifted = _shift_vector_left(left_interval, padding=max_profile_value + 1)

    # Find the left bound
    left_bound = np.argmax(left_interval - left_interval_shifted < 0) + 1
    left_bound = min(left_bound, max_width)

    # Right side of the candidate
    right_interval = profile[maxima : right_minima + 1]
    right_interval_shifted = _shift_vector_left(right_interval, padding=max_profile_value + 1)

    # Find the right bound
    right_bound = np.argmax(right_interval - right_interval_shifted < 0) + 1
    right_bound = min(right_bound, max_width)

    return max(maxima - left_bound, 0), min(maxima + right_bound, len(profile))


def _extract_standardized_local_1d_pseudodistribution(
    matrix: SparseMatrix,
    seed_site: int,
    left_bound: int,
    right_bound: int,
    max_height: int,
    location: str,
) -> npt.NDArray[float]:
    """
    Extract the standardized local 1D pseudo-distribution from the given matrix as efficiently as possible.

    Parameters
    ----------
    matrix : SparseMatrix
        matrix to be processed
    seed_site : int
        seed site position
    left_bound : int
        left boundary of the domain
    right_bound : int
        right boundary of the domain
    max_height : int
        maximum height allowed
    location: str
        matrix location (should be "lower" or "upper")

    Returns
    -------
    npt.NDArray[float]
        the standardized local 1D pseudo-distribution for the given seed site.
    """
    # assertions disabled for performance reasons
    # assert left_bound >= 0
    # assert right_bound < matrix.shape[0]
    # assert left_bound <= seed_site <= right_bound
    # assert max_height > 0

    if location == "lower":
        i0, i1 = seed_site, min(seed_site + max_height, matrix.shape[0])
    else:
        i0, i1 = max(seed_site - max_height, 0), seed_site

    stripe_height = abs(i1 - i0)

    j0, j1 = left_bound, right_bound

    # When stripes are very tall, it is always better to slice by columns first (even if the matrix is in CSR format)
    if isinstance(matrix, ss.csc_matrix) or stripe_height > 1000:
        submatrix = matrix[:, j0:j1].tocsr()[i0:i1, :].tocsc()
    elif isinstance(matrix, ss.csr_matrix):
        submatrix = matrix[i0:i1, :].tocsc()[:, j0:j1]
    else:
        raise NotImplementedError

    y = submatrix.sum(axis=1)
    if isinstance(y, np.matrix):
        y = y.A1

    y /= y.max()
    if location == "lower":
        return y

    # assertion disabled for performance reasons
    # assert location == "upper"
    return np.flip(y)


def _find_v_domain_helper(
    profile: npt.NDArray[float],
    threshold_cut: float,
    min_persistence: Optional[float],
) -> Tuple[int, Optional[npt.NDArray[int]]]:
    """
    Helper function for the _find_*_v_domain() functions.
    It computes the bound of the profile using one of the two following criteria:
    - if min_persistence is provided, it finds persistent maximum points by thresholding
      topological persistence. If at least two persistent maxima are found, it returns
      as bound the rightmost persistent maximum point
    - if min_persistence is not provided or the previous condition does not hold,
      it finds the bound as the index from where the (smoothed) profile is identically
      below threshold_cut


    Parameters
    ----------
    profile : npt.NDArray[float]
        1D array representing a uniformly-sample scalar function works
    threshold_cut: float
        threshold used to determine the span of vertical domains.
        Higher values will lead to shorter domains.
    min_persistence: Optional[float]
        threshold used to find peaks in the profile associated with a horizontal domain.

    Returns
    -------
    Tuple[int, Optional[npt.NDArray[int]]]
        A tuple with the following entries:
        - the bound of the domain
        - the array of maximum points (optional, when min_persistence is provided)
    """
    if min_persistence is None:
        max_points = tuple()
    else:
        max_points = (
            Persistence1DTable.calculate_persistence(profile, min_persistence=min_persistence)
            .max.sort_values(kind="stable")
            .index.to_numpy()
        )

    if len(max_points) > 1:
        return max_points.max(), max_points[:-1]  # drop global maximum

    approximated_profile = compute_wQISA_predictions(profile, 5)
    # the type cast here is just to avoid returning numpy ints
    candida_bound = int(np.argmax(approximated_profile < threshold_cut))
    if approximated_profile[candida_bound] >= threshold_cut:
        candida_bound = len(profile) - 1

    return candida_bound, None


def _find_lower_v_domain(
    coords: Tuple[int, int, int],
    matrix: Optional[SparseMatrix],
    threshold_cut: float,
    max_height: int,
    min_persistence: float,
    return_maxima: bool,
) -> Dict[str, Any]:
    """
    Find vertical domains spanning the lower-triangle of the matrix.

    Parameters
    ----------
    coords : Tuple[int, int, int]
        1) seed_site
        2) left_bound
        3) right_bound
    matrix: Optional[SparseMatrix]
        matrix with the interactions to be processed.
        When set to None, the matrix will be fetched from the global shared state.
    threshold_cut: float
        threshold used to determine the span of vertical domains.
        Higher values will lead to shorter domains.
    max_height: int
        maximum height allowed
    min_persistence: float
        threshold used to find peaks in the profile associated with a horizontal domain.
    return_maxima: bool
        whether to return the local profile maxima associated with each domain

    Returns
    -------
    Dict[str, Any]
        A dictionary with the following keys:
        - top_bound: int
        - bottom_bound: int
        - local_maxima: npt.NDArray[int] (optional)
    """
    seed_site, left_bound, right_bound = coords
    # assertions disabled for performance reasons
    # assert left_bound >= 0
    # assert left_bound <= seed_site <= right_bound
    # assert 0 <= threshold_cut <= 1
    # assert max_height > 0
    # assert 0 <= min_persistence <= 1

    if matrix is None:
        matrix = get_shared_state("lower").get()
    # assert right_bound < matrix.shape[0]

    profile = _extract_standardized_local_1d_pseudodistribution(
        matrix,
        seed_site,
        left_bound,
        right_bound,
        max_height,
        "lower",
    )
    candidate_bound, local_maxima = _find_v_domain_helper(profile, threshold_cut, min_persistence)

    res = {"top_bound": seed_site, "bottom_bound": seed_site + candidate_bound}

    if not return_maxima or local_maxima is None:
        return res

    local_maxima += seed_site
    res["maxima"] = local_maxima
    return res


def _find_upper_v_domain(
    coords: Tuple[int, int, int],
    matrix: Optional[SparseMatrix],
    threshold_cut: float,
    max_height: int,
    min_persistence: float,
    return_maxima: bool,
) -> Dict[str, Any]:
    """
    Find vertical domains spanning the upper-triangle of the matrix.

    Parameters
    ----------
    coords : Tuple[int, int, int]
        1) seed_site
        2) left_bound
        3) right_bound
    matrix: Optional[SparseMatrix]
        matrix with the interactions to be processed.
        When set to None, the matrix will be fetched from the global shared state.
    threshold_cut: float
        threshold used to determine the span of vertical domains.
        Higher values will lead to shorter domains.
    max_height: int
        maximum height allowed
    min_persistence: float
        threshold used to find peaks in the profile associated with a horizontal domain.
    return_maxima: bool
        whether to return the local profile maxima associated with each domain

    Returns
    -------
    Dict[str, Any]
        A dictionary with the following keys:
        - top_bound: int
        - bottom_bound: int
        - local_maxima: npt.NDArray[int] (optional)
    """

    seed_site, left_bound, right_bound = coords
    # assertions disabled for performance reasons
    # assert left_bound >= 0
    # assert left_bound <= seed_site <= right_bound
    # assert 0 <= threshold_cut <= 1
    # assert max_height > 0
    # assert 0 <= min_persistence <= 1

    if matrix is None:
        matrix = get_shared_state("upper").get()
    # assert right_bound < matrix.shape[0]

    profile = _extract_standardized_local_1d_pseudodistribution(
        matrix,
        seed_site,
        left_bound,
        right_bound,
        max_height,
        "upper",
    )
    candidate_bound, local_maxima = _find_v_domain_helper(profile, threshold_cut, min_persistence)

    res = {"top_bound": seed_site - candidate_bound, "bottom_bound": seed_site}

    if not return_maxima or local_maxima is None:
        return res

    local_maxima *= -1
    local_maxima += seed_site

    res["maxima"] = local_maxima
    return res


def find_horizontal_intervals_of_interest(
    pseudodistribution: npt.NDArray[float],
    seed_sites: npt.NDArray[int],
    seed_site_bounds: npt.NDArray[int],
    max_width: int,
    logger=None,
) -> pd.DataFrame:
    """
    Find the left and right bounds for the given seed sites.

    Parameters
    ----------
    pseudodistribution: npt.NDArray[float]
        1D array representing a uniformly-sample scalar function works
    seed_sites: npt.NDArray[int]
        maximum values in the pseudo-distribution (i.e., genomic coordinates hosting linear patterns)
    seed_site_bounds: npt.NDArray[int]
        for the i-th entry of seed_sites:
        - seed_site_bounds[i] is the left boundary
        - seed_site_bounds[i+1] is the right boundary
    max_width: int
        maximum width allowed
    logger:
        logger

    Returns
    -------
    pd.DataFrame
        a DataFrame with the list of left and right boundaries associated with each seed site
    """
    assert len(seed_site_bounds) == len(seed_sites) + 1
    assert max_width > 0

    t0 = time.time()
    if logger is None:
        logger = structlog.get_logger()

    # Pre-compute the global max
    max_pseudodistribution_value = pseudodistribution.max()

    # Declare a generator for the params used to generate tasks
    params = (
        (
            seed_site,
            seed_site_bounds[maxima_idx],
            seed_site_bounds[maxima_idx + 1],
            max_pseudodistribution_value,
        )
        for maxima_idx, seed_site in enumerate(seed_sites)
    )

    # Set up tasks to find horizontal domains
    # It is not worth parallelizing this step
    tasks = map(
        functools.partial(
            _find_horizontal_domain,
            pseudodistribution,
            max_width=max_width,
        ),
        params,
    )

    # This efficiently constructs a 2D numpy with shape (N, 2) from a list of 2-element tuples, where N is the number of seed sites.
    # The first and second columns contains the left and right boundaries of the horizontal domains, respectively.
    bounds = np.fromiter(
        itertools.chain.from_iterable(tasks),
        count=2 * len(seed_sites),
        dtype=int,
    ).reshape(-1, 2)

    # Handle possible overlapping intervals by ensuring that the
    # left bound of interval i + 1 is always greater or equal than the right bound of interval i
    bounds[1:, 0] = np.maximum(bounds[1:, 0], bounds[:-1, 1])

    df = pd.DataFrame(data=bounds, columns=["left_bound", "right_bound"])

    logger.debug("find_horizontal_intervals_of_interest() took %s", pretty_format_elapsed_time(t0))

    return df


def find_vertical_intervals_of_interest(
    matrix: Optional[SparseMatrix],
    seed_sites: npt.NDArray[int],
    horizontal_domains: pd.DataFrame,
    max_height: int,
    threshold_cut: float,
    min_persistence: float,
    location: str,
    return_maxima: bool = False,
    map_=map,
    logger=None,
) -> pd.DataFrame:
    """
    Find the top and bottom bounds for the given seed sites.

    Parameters
    ----------
    matrix: Optional[SparseMatrix]
        matrix with the interactions to be processed.
        When set to None, the matrix will be fetched from the global shared state.
    seed_sites: npt.NDArray[int]
        maximum values in the pseudo-distribution (i.e., genomic coordinates hosting linear patterns)
    horizontal_domains: pd.DataFrame
        DataFrame with the horizontal domains identified by find_horizontal_intervals_of_interest()
    max_height: int
        maximum height allowed
    threshold_cut: float
        threshold used to determine the span of vertical domains.
        Higher values will lead to shorter domains.
    min_persistence: float
        threshold used to find peaks in the profile associated with a horizontal domain.
    location: str
        matrix location (should be "lower" or "upper")
    return_maxima: bool
        whether to return the local profile maxima associated with each domain
    map_: Callable
        a callable that behaves like the built-in map function
    logger:
        logger

    Returns
    -------
    pd.DataFrame
        a DataFrame with the list of top and bottom bounds associated with each seed site.
        If return_maxima is True, the DataFrame also has a column called "maxima" containing
        the maxima from the local profile associated with each domain.
    """
    assert len(seed_sites) > 0
    assert len(seed_sites) == len(horizontal_domains)

    t0 = time.time()
    if logger is None:
        logger = structlog.get_logger()

    # Pair each seed site with its left and right bounds
    df = horizontal_domains.copy()
    df["seed_site"] = seed_sites
    df = df[["seed_site", "left_bound", "right_bound"]]

    if location == "lower":
        finder = _find_lower_v_domain
    elif location == "upper":
        finder = _find_upper_v_domain
    else:
        raise ValueError("where should be lower or upper")

    # Declare a generator for the params used to generate tasks
    params = ((seed, lb, rb) for seed, lb, rb in df.itertuples(index=False))

    # Prepare tasks
    tasks = map_(
        functools.partial(
            finder,
            matrix=matrix,
            threshold_cut=threshold_cut,
            max_height=max_height,
            min_persistence=min_persistence,
            return_maxima=return_maxima,
        ),
        params,
    )

    # Collect results
    df = pd.DataFrame.from_records(
        data=tasks,
        nrows=len(seed_sites),
    )

    assert len(df) == len(seed_sites)

    logger.debug("find_vertical_intervals_of_interest() took %s", pretty_format_elapsed_time(t0))

    return df
