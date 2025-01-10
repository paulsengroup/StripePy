# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import itertools
import time
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as ss
import structlog

from . import TDA
from .common import pretty_format_elapsed_time
from .regressions import _compute_wQISA_predictions


def find_horizontal_domain(
    profile: npt.NDArray[float],
    coarse_h_domain: Tuple[int, int, int],
    max_width: int = 1e9,
) -> Tuple[int, int]:
    """
    Returns
    -------
    Tuple[int, int]
        the left and right coordinates of the horizontal domain
    """

    # Unpacking:
    MP, L_mP, R_mP = coarse_h_domain

    # Left and sides of candidate:
    L_interval = np.flip(profile[L_mP : MP + 1])
    R_interval = profile[MP : R_mP + 1]

    # LEFT INTERVAL
    L_interval_shifted = np.append(L_interval[1:], [max(profile) + 1], axis=0)
    L_bound = np.where(L_interval - L_interval_shifted < 0)[0][0] + 1
    # L_interval_restr = L_interval[:L_bound]
    # L_interval_shifted_restr = L_interval_shifted[:L_bound]
    # L_bound = np.argmax(L_interval_restr - L_interval_shifted_restr) + 1
    L_bound = np.minimum(L_bound, max_width)

    # RIGHT INTERVAL
    R_interval_shifted = np.append(R_interval[1:], [max(profile) + 1], axis=0)
    R_bound = np.where(R_interval - R_interval_shifted < 0)[0][0] + 1
    # R_interval_restr = R_interval[:R_bound]
    # R_interval_shifted_restr = R_interval_shifted[:R_bound]
    # R_bound = np.argmax(R_interval_restr - R_interval_shifted_restr) + 1
    R_bound = np.minimum(R_bound, max_width)

    return max(MP - L_bound, 0), min(MP + R_bound, len(profile))


def find_lower_v_domain(I, threshold_cut, max_height, min_persistence, it) -> Tuple[List, Optional[List]]:

    # For each slice (hosting a seed site):
    n, seed_site, HIoI = it

    # Standardized Local 1D pseudo-distribution for current HIoI:
    rows = slice(seed_site, min(seed_site + max_height, I.shape[0]))
    cols = slice(HIoI[0], HIoI[1])
    I_nb = I[rows, :].tocsc()[:, cols].toarray()

    Y = np.sum(I_nb, axis=1)
    Y /= max(Y)

    # Lower boundary:
    Y_hat = _compute_wQISA_predictions(Y, 5)  # Basically: average of a 2-"pixel" neighborhood

    # Peaks:
    if min_persistence is None:
        candida_bound = np.where(np.array(Y_hat) < threshold_cut)[0]
        if len(candida_bound) == 0:
            candida_bound = [len(Y_hat) - 1]

        # Vertical domain + no peak:
        return [seed_site, seed_site + candida_bound[0]], None

    _, _, loc_Maxima, loc_pers_of_Maxima = TDA.TDA(Y, min_persistence=min_persistence)
    candida_bound = [max(loc_Maxima)]

    # Consider as min_persistence is set to None:
    if len(loc_Maxima) < 2:
        candida_bound = np.where(np.array(Y_hat) < threshold_cut)[0]
        if len(candida_bound) == 0:
            candida_bound = [len(Y_hat) - 1]

    # Vertical domain + peaks:
    return [seed_site, seed_site + candida_bound[0]], list(np.array(loc_Maxima[:-1]) + seed_site)


def find_upper_v_domain(I, threshold_cut, max_height, min_persistence, it) -> Tuple[List, Optional[List]]:

    # For each slice (hosting a seed site):
    n, seed_site, HIoI = it

    # Standardized Local 1D pseudo-distribution for current HIoI:
    rows = slice(max(seed_site - max_height, 0), seed_site)
    cols = slice(HIoI[0], HIoI[1])
    I_nb = I[rows, :].tocsc()[:, cols].toarray()
    Y = np.flip(np.sum(I_nb, axis=1))
    Y /= max(Y)

    # Upper boundary:
    Y_hat = _compute_wQISA_predictions(Y, 5)

    # Peaks:
    if min_persistence is None:
        candida_bound = np.where(np.array(Y_hat) < threshold_cut)[0]
        if len(candida_bound) == 0:
            candida_bound = [len(Y_hat) - 1]

        # Vertical domain + no peak:
        return [seed_site - candida_bound[0], seed_site], None

    _, _, loc_Maxima, loc_pers_of_Maxima = TDA.TDA(Y, min_persistence=min_persistence)
    candida_bound = [max(loc_Maxima)]

    # Consider as min_persistence is set to None:
    if len(loc_Maxima) < 2:
        candida_bound = np.where(np.array(Y_hat) < threshold_cut)[0]
        if len(candida_bound) == 0:
            candida_bound = [len(Y_hat) - 1]

    # Vertical domain + peaks:
    return [seed_site - candida_bound[0], seed_site], list(seed_site - np.array(loc_Maxima[:-1]))


def find_HIoIs(
    pseudodistribution: npt.NDArray[float],
    seed_sites: npt.NDArray[int],
    seed_site_bounds: npt.NDArray[int],
    max_width: int,
    map_=map,
    logger=None,
) -> pd.DataFrame:
    """
    :param pseudodistribution:  1D array representing a uniformly-sample scalar function works
    :param seed_sites:          maximum values in the pseudo-distribution (i.e., genomic coordinates hosting linear
                                patterns)
    :param seed_site_bounds:    for the i-th entry of seed_sites:
                                (*) seed_site_bounds[i] is the left boundary
                                (*) seed_site_bounds[i+1] is the right boundary
    :param max_width:           maximum width allowed
    :param map_:                 alternative implementation of the built-in map function. Can be used to e.g. run this step in parallel by passing multiprocessing.Pool().map.
    :return:
    HIoIs                       a pd.DataFrame the list of left and right boundary for each seed site
    """
    assert len(seed_site_bounds) == len(seed_sites) + 1

    t0 = time.time()
    if logger is None:
        logger = structlog.get_logger()

    params = (
        (seed_site, seed_site_bounds[num_MP], seed_site_bounds[num_MP + 1])
        for num_MP, seed_site in enumerate(seed_sites)
    )

    tasks = map_(partial(find_horizontal_domain, pseudodistribution, max_width=max_width), params)
    # This efficiently constructs a 2D numpy with shape (N, 2) from a list of 2-element tuples, where N is the number of seed sites.
    # The first and second columns contains the left and right boundaries of the horizontal domains, respectively.
    HIoIs = np.fromiter(itertools.chain.from_iterable(tasks), count=2 * len(seed_sites), dtype=int).reshape(-1, 2)

    # Handle possible overlapping intervals by ensuring that the
    # left bound of interval i + 1 is always greater or equal than the right bound of interval i
    HIoIs[1:, 0] = np.maximum(HIoIs[1:, 0], HIoIs[:-1, 1])

    df = pd.DataFrame(data=HIoIs, columns=["left_bound", "right_bound"])

    logger.debug("find_HIoIs took %s", pretty_format_elapsed_time(t0))

    return df


def find_VIoIs(
    I,
    seed_sites,
    HIoIs,
    max_height,
    threshold_cut=0.1,
    min_persistence=0.20,
    where="lower",
    map=map,
):

    # Triplets to use in multiprocessing:
    iterable_input = [(n, seed_site, HIoI) for n, (seed_site, HIoI) in enumerate(zip(seed_sites, HIoIs))]

    # Lower-triangular part of the Hi-C matrix:
    if where == "lower":
        Vdomains_and_peaks = map(
            partial(find_lower_v_domain, I, threshold_cut, max_height, min_persistence),
            iterable_input,
        )

        VIoIs, peak_locs = list(zip(*Vdomains_and_peaks))

    # Upper-triangular part of the Hi-C matrix:
    elif where == "upper":
        # HIoIs = pool.map(partial(find_h_domain, pd), iterable_input)
        Vdomains_and_peaks = map(
            partial(find_upper_v_domain, I, threshold_cut, max_height, min_persistence),
            iterable_input,
        )

        VIoIs, peak_locs = list(zip(*Vdomains_and_peaks))

    return VIoIs, peak_locs
