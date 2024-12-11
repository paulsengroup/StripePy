# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import pathlib
import time
from typing import Dict, List, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as ss
import seaborn as sns
from numpy.typing import NDArray

from . import IO
from .utils import TDA, finders, regressions, stripe

be_verbose = True  # TODO consider safe removal of be_verbose


def _log_transform(I: ss.csr_matrix) -> ss.csr_matrix:
    """
    Apply a log-transform to a sparse matrix ignoring (i.e. dropping) NaNs.

    Parameters
    ----------
    I : ss.csr_matrix
        the sparse matrix to be transformed

    Returns
    -------
    ss.csr_matrix
        the log-transformed sparse matrix
    """

    I.data[np.isnan(I.data)] = 0
    I.eliminate_zeros()
    Iproc = I.log1p()
    return Iproc


def _band_extraction(I: ss.csr_matrix, resolution: int, genomic_belt: int) -> Tuple[ss.csr_matrix, ss.csr_matrix]:
    """
    Given a symmetric sparse matrix in CSR format, do the following:

      * Split the input matrix into a upper/lower-triangular matrices
      * Zero (i.e. drop) all values that lie outside the first genomic_belt // resolution diagonals

    Parameters
    ----------
    I : ss.csr_matrix
        the sparse matrix to be processed
    resolution : int
        the genomic resolution of the sparse matrix I
    genomic_belt: int
        the width of the genomic belt to be extracted

    Returns
    -------
    Tuple[ss.csr_matrix, ss.csr_matrix]
        2 elements tuple with the lower-triangular and upper-triangular matrix after band extraction
    """

    assert resolution > 0
    assert genomic_belt > 0

    matrix_belt = genomic_belt // resolution
    LT_I = ss.tril(I, k=0, format="csr") - ss.tril(I, k=-matrix_belt, format="csr")
    UT_I = ss.triu(I, k=0, format="csr") - ss.triu(I, k=matrix_belt, format="csr")
    return LT_I, UT_I


def _scale_Iproc(
    I: ss.csr_matrix, LT_I: ss.csr_matrix, UT_I: ss.csr_matrix
) -> Tuple[ss.csr_matrix, ss.csr_matrix, ss.csr_matrix]:
    """
    Rescale matrices LT_I and UT_I based on the maximum value found in matrix I

    Parameters
    ----------
    I : ss.csr_matrix
        the sparse matrix used to compute the scaling factor
    LT_I : ss.csr_matrix
        the lower-triangular sparse matrix to be rescaled
    UT_I : ss.csr_matrix
        the upper-triangular sparse matrix to be rescaled

    Returns
    -------
    Tuple[ss.csr_matrix, ss.csr_matrix]
        the rescaled lower and upper-triangular matrices
    """

    scaling_factor_Iproc = I.max()
    return tuple(J / scaling_factor_Iproc for J in [I, LT_I, UT_I])  # noqa


def _extract_RoIs(I: ss.csr_matrix, RoI: Dict[str, List[int]]) -> NDArray:
    """
    Extract a region of interest (ROI) from the sparse matrix I

    Parameters
    ----------
    I: ss.csr_matrix
        the sparse matrix to be processed
    RoI: Dict[str, List[int]]
        dictionary with the region of interest in matrix ('matrix') and genomic ('genomic') coordinates

    Returns
    -------
    NDArray
        dense matrix with the interactions for the regions of interest
    """

    rows = cols = slice(RoI["matrix"][0], RoI["matrix"][1])
    I_RoI = I[rows, cols].toarray()
    return I_RoI


def _plot_RoIs(
    I: ss.csr_matrix, Iproc: ss.csr_matrix, RoI: Union[NDArray, None], output_folder: Union[pathlib.Path, None]
) -> Union[NDArray, None]:
    """
    Helper function to plot a region of interest.
    This function does nothing when RoI is None.
    If output_folder is None (but RoI is not), then this function simply extracts the interactions
    that would've been used for plotting, but generates no plots.

    Parameters
    ----------
    I: ss.csr_matrix
        the unprocessed input sparse matrix
    Iproc: ss.csr_matrix
        the processed input sparse matrix
    RoI: Union[NDArray, None]
        the region of interest to be plotted in matrix ('matrix') and genomic ('genomic') coordinates
    output_folder: pathlib.Path
        folder where to save the plots

    Returns
    -------
    Union[NDArray, None]
        the dense matrix used for plotting or None when RoI is None
    """

    # TODO rea1991 Once there is better test coverage, rewrite this as suggested in in #16
    if RoI is not None:
        print("1.4) Extracting a Region of Interest (RoI) for plot purposes...")
        I_RoI = _extract_RoIs(I, RoI)
        Iproc_RoI = _extract_RoIs(Iproc, RoI)

        if output_folder is not None:
            # Plots:
            IO.HiC(
                I_RoI,
                RoI["genomic"],
                plot_in_bp=True,
                output_folder=output_folder,
                file_name=f"I_{RoI['genomic'][0]}_{RoI['genomic'][1]}.jpg",
                compactify=False,
            )
            IO.HiC(
                Iproc_RoI,
                RoI["genomic"],
                plot_in_bp=True,
                output_folder=output_folder,
                file_name=f"Iproc_{RoI['genomic'][0]}_{RoI['genomic'][1]}.jpg",
                compactify=False,
            )
    else:
        Iproc_RoI = None  # TODO handle this case in _extract_RoIs
    return Iproc_RoI


def _compute_global_pseudodistribution(T: ss.csr_matrix, smooth: bool = True) -> NDArray[float]:
    """
    Given a sparse matrix T, marginalize it, scale the marginal so that maximum is 1, and then smooth it.

    Parameters
    ----------
    T: ss.csr_matrix
        the sparse matrix to be processed
    smooth: bool
        if set to True, smoothing is applied to the pseudo-distribution (default value is True)

    Returns
    -------
    NDArray[np.float64]
        a vector with the re-scaled and smoothed marginals.
    """

    pseudo_dist = np.squeeze(np.asarray(np.sum(T, axis=0)))  # marginalization
    pseudo_dist /= np.max(pseudo_dist)  # scaling
    if smooth:
        pseudo_dist = np.maximum(regressions._compute_wQISA_predictions(pseudo_dist, 11), pseudo_dist)  # smoothing
    return pseudo_dist


def _sort_extrema_by_coordinate(ps_ePs: List[int], pers_of_ps_ePs: List[float]) -> Tuple[List[int], List[float]]:
    """
    Sort the two lists given as input in ascending order based on values from ps_ePs.

    Parameters
    ----------
    ps_ePs: List[int]
        the location of the local maximum points
    pers_of_ps_ePs: List[float]
        the values of topological persistence corresponding to the locations listed in ps_ePs

    Returns
    -------
    Tuple[List[int], List[float]]
        the two input lists sorted based on ps_ePs
    """
    # Some notation:
    # ps = persistence-sorted
    # ePs = extremum points
    # pers = topological persistence

    assert len(ps_ePs) == len(pers_of_ps_ePs)

    permutation_ps2cs_ePs = np.argsort(ps_ePs)

    # Maximum and minimum points sorted w.r.t. coordinates: actual application of permutations
    cs_ePs = np.array(ps_ePs)[permutation_ps2cs_ePs].tolist()
    cs_pers_of_ePs = np.array(pers_of_ps_ePs)[permutation_ps2cs_ePs].tolist()

    return cs_ePs, cs_pers_of_ePs


def _find_seeds_in_RoI(
    seeds: List[int], left_bound_RoI: int, right_bound_RoI: int
) -> Tuple[NDArray[np.int64], List[int]]:
    """
    Select seed coordinates that fall within the given left and right boundaries.

    Parameters
    ----------
    seeds: List[int]
        a list with the seed coordinates
    left_bound_RoI: int
        left bound of the region of interest
    right_bound_RoI: int
        right bound of the region of interest

    Returns
    -------
    Tuple[NDArray[np.int64], List[int]]
        a tuple consisting of:

         * the indices of seed coordinates falling within the given boundaries
         * the coordinates of the selected seeds
    """

    assert left_bound_RoI >= 0
    assert right_bound_RoI >= left_bound_RoI

    # Find sites within the range of interest -- lower-triangular:
    ids_seeds_in_RoI = np.where((left_bound_RoI <= np.array(seeds)) & (np.array(seeds) <= right_bound_RoI))[0]
    seeds_in_RoI = np.array(seeds)[ids_seeds_in_RoI].tolist()

    return ids_seeds_in_RoI, seeds_in_RoI


def _store_results(
    hf: h5py._hl.group.Group,
    pd: NDArray[np.float64],
    min_points: List[int],
    pers_of_min_points: List[float],
    max_points: List[int],
    pers_of_max_points: List[float],
    min_persistence: float,
):
    hf.create_dataset("pseudo-distribution", data=np.array(pd), compression="gzip", compression_opts=4, shuffle=True)
    hf.create_dataset(
        "minima_pts_and_persistence",
        data=np.array([min_points, pers_of_min_points]),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )
    hf.create_dataset(
        "maxima_pts_and_persistence",
        data=np.array([max_points, pers_of_max_points]),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )
    hf.parent.attrs["min_persistence_used"] = min_persistence


def _check_neighborhood(
    values: NDArray[np.float64], min_value: float = 0.1, neighborhood_size: int = 10, threshold_percentage: float = 0.85
) -> List[int]:
    # TODO rea1991 Change neighborhood size from "matrix" to "genomic" (eg, default of 1 Mb)
    assert 0 <= min_value
    assert 1 <= neighborhood_size <= len(values)
    assert 0 <= threshold_percentage <= 1

    mask = [0] * len(values)
    for i in range(neighborhood_size, len(values) - neighborhood_size):
        neighborhood = values[i - neighborhood_size : i + neighborhood_size + 1]
        ratio_above_min_value = sum(1 for value in neighborhood if value >= min_value) / len(neighborhood)

        if ratio_above_min_value >= threshold_percentage:
            mask[i] = 1
    return mask


def _filter_extrema_by_sparseness(
    ps_mPs: List[int], pers_of_ps_mPs: List[float], ps_MPs: List[int], pers_of_ps_MPs: List[float], mask: List[int]
) -> Tuple[List[int], List[float], List[int], List[float]]:
    ps_mPs_2, pers_of_ps_mPs_2, ps_MPs_2, pers_of_ps_MPs_2 = [], [], [], []

    for i in range(len(ps_MPs)):
        if mask[ps_MPs[i]] == 1:
            ps_MPs_2.append(ps_MPs[i])
            pers_of_ps_MPs_2.append(pers_of_ps_MPs[i])

            # Last maximum point is not paired with a minimum point by construction:
            if i < len(ps_MPs) - 1:
                ps_mPs_2.append(ps_mPs[i])
                pers_of_ps_mPs_2.append(pers_of_ps_mPs[i])

    # If last maximum point was discarded, we need to remove the minimum point before it:
    if len(ps_mPs_2) == len(ps_MPs_2) and len(ps_mPs_2) > 0:
        ps_mPs_2.pop()
        pers_of_ps_mPs_2.pop()

    return ps_mPs_2, pers_of_ps_mPs_2, ps_MPs_2, pers_of_ps_MPs_2


def step_1(I, genomic_belt, resolution, RoI=None, output_folder=None):
    print("1.1) Log-transformation...")
    Iproc = _log_transform(I)

    print("1.2) Focusing on a neighborhood of the main diagonal...")
    LT_Iproc, UT_Iproc = _band_extraction(Iproc, resolution, genomic_belt)

    print("1.3) Projection onto [0, 1]...")
    Iproc, LT_Iproc, UT_Iproc = _scale_Iproc(Iproc, LT_Iproc, UT_Iproc)

    Iproc_RoI = _plot_RoIs(I, Iproc, RoI, output_folder)

    return LT_Iproc, UT_Iproc, Iproc_RoI


def step_2(chrom: str, L, U, resolution, min_persistence, Iproc_RoI=None, RoI=None, output_folder=None) -> IO.Result:
    print("2.1) Global 1D pseudo-distributions...")
    LT_pd = _compute_global_pseudodistribution(L, smooth=True)
    UT_pd = _compute_global_pseudodistribution(U, smooth=True)

    result = IO.Result(chrom)
    if RoI is not None:
        result.set_roi(RoI)
    result.set_min_persistence(min_persistence)

    result.set("pseudodistribution", LT_pd, "LT")
    result.set("pseudodistribution", UT_pd, "UT")

    print("2.2) Detection of persistent maxima and corresponding minima for lower- and upper-triangular matrices...")

    print("2.2.0) All maxima and their persistence")
    # NOTATION: mPs = minimum points, MPs = maximum Points, ps = persistence-sorted
    # NB: MPs are the actual sites of interest, i.e., the sites hosting linear patterns

    # All local minimum and maximum points:
    all_LT_ps_mPs, all_pers_of_LT_ps_mPs, all_LT_ps_MPs, all_pers_of_LT_ps_MPs = TDA.TDA(LT_pd, min_persistence=0)
    all_UT_ps_mPs, all_pers_of_UT_ps_mPs, all_UT_ps_MPs, all_pers_of_UT_ps_MPs = TDA.TDA(UT_pd, min_persistence=0)

    result.set("all_minimum_points", all_LT_ps_mPs, "LT")
    result.set("all_maximum_points", all_LT_ps_MPs, "LT")
    result.set("persistence_of_all_minimum_points", all_pers_of_LT_ps_mPs, "LT")
    result.set("persistence_of_all_maximum_points", all_pers_of_LT_ps_MPs, "LT")
    result.set("all_minimum_points", all_UT_ps_mPs, "UT")
    result.set("all_maximum_points", all_UT_ps_MPs, "UT")
    result.set("persistence_of_all_minimum_points", all_pers_of_UT_ps_mPs, "UT")
    result.set("persistence_of_all_maximum_points", all_pers_of_UT_ps_MPs, "UT")

    print("2.2.1) Lower triangular part")
    LT_ps_mPs, pers_of_LT_ps_mPs, LT_ps_MPs, pers_of_LT_ps_MPs = TDA.TDA(LT_pd, min_persistence=min_persistence)

    print("2.2.2) Upper triangular part")
    # Here, LT_ps_mPs means that the lower-triangular minimum points are sorted w.r.t. persistence
    # (NOTATION: ps = persistence-sorted)
    UT_ps_mPs, pers_of_UT_ps_mPs, UT_ps_MPs, pers_of_UT_ps_MPs = TDA.TDA(UT_pd, min_persistence=min_persistence)
    # NB: Maxima are sorted w.r.t. their persistence... and this sorting is applied to minima too,
    # so that each maximum is still paired to its minimum.

    # Maximum and minimum points sorted w.r.t. coordinates (NOTATION: cs = coordinate-sorted):
    LT_mPs, LT_pers_of_mPs = _sort_extrema_by_coordinate(LT_ps_mPs, pers_of_LT_ps_mPs)
    LT_MPs, LT_pers_of_MPs = _sort_extrema_by_coordinate(LT_ps_MPs, pers_of_LT_ps_MPs)
    UT_mPs, UT_pers_of_mPs = _sort_extrema_by_coordinate(UT_ps_mPs, pers_of_UT_ps_mPs)
    UT_MPs, UT_pers_of_MPs = _sort_extrema_by_coordinate(UT_ps_MPs, pers_of_UT_ps_MPs)

    print("2.2.3) Filter out seeds in sparse regions")
    LT_mask = _check_neighborhood(_compute_global_pseudodistribution(L, smooth=False))
    UT_mask = _check_neighborhood(_compute_global_pseudodistribution(U, smooth=False))
    x = _filter_extrema_by_sparseness(LT_mPs, LT_pers_of_mPs, LT_MPs, LT_pers_of_MPs, LT_mask)
    LT_mPs, LT_pers_of_mPs, LT_MPs, LT_pers_of_MPs = x
    x = _filter_extrema_by_sparseness(UT_mPs, UT_pers_of_mPs, UT_MPs, UT_pers_of_MPs, UT_mask)
    UT_mPs, UT_pers_of_mPs, UT_MPs, UT_pers_of_MPs = x
    if len(LT_MPs) < len(LT_ps_MPs):
        print(f"Number of lower-triangular seed sites is reduced from {len(LT_ps_MPs)} to {len(LT_MPs)}")
    if len(UT_MPs) < len(UT_ps_MPs):
        print(f"Number of upper-triangular seed sites is reduced from {len(UT_ps_MPs)} to {len(UT_MPs)}")
    if len(LT_MPs) == len(LT_ps_MPs) and len(UT_MPs) == len(UT_ps_MPs):
        print("No change in number of seed sites")

    result.set("persistent_minimum_points", LT_mPs, "LT")
    result.set("persistent_maximum_points", LT_MPs, "LT")
    result.set("persistence_of_minimum_points", LT_pers_of_mPs, "LT")
    result.set("persistence_of_maximum_points", LT_pers_of_MPs, "LT")

    result.set("persistent_minimum_points", UT_mPs, "UT")
    result.set("persistent_maximum_points", UT_MPs, "UT")
    result.set("persistence_of_minimum_points", UT_pers_of_mPs, "UT")
    result.set("persistence_of_maximum_points", UT_pers_of_MPs, "UT")

    # If no candidates are found in the lower- or upper-triangular maps, exit:
    if len(LT_MPs) == 0 or len(UT_MPs) == 0:
        return result

    print("2.3) Storing into a list of Stripe objects...")
    stripes = [
        stripe.Stripe(seed=LT_MP, top_pers=LT_pers_of_MP, where="lower_triangular")
        for LT_MP, LT_pers_of_MP in zip(LT_MPs, LT_pers_of_MPs)
    ]
    result.set("stripes", stripes, "LT")

    stripes = [
        stripe.Stripe(seed=UT_MP, top_pers=UT_pers_of_MP, where="upper_triangular")
        for UT_MP, UT_pers_of_MP in zip(UT_MPs, UT_pers_of_MPs)
    ]
    result.set("stripes", stripes, "UT")

    if result.roi is not None:
        print("2.4) Finding sites inside the region selected above...")
        ids_LT_MPs_in_RoI, LT_MPs_in_RoI = _find_seeds_in_RoI(LT_MPs, result.roi["matrix"][0], result.roi["matrix"][1])
        ids_UT_MPs_in_RoI, UT_MPs_in_RoI = _find_seeds_in_RoI(UT_MPs, result.roi["matrix"][0], result.roi["matrix"][1])

        print("2.5) Plotting pseudo-distributions and sites for the region selected above...")
        if output_folder is not None:

            # Plot pseudo-distributions:
            IO.pseudodistrib(
                LT_pd, RoI["genomic"][0:2], resolution, output_folder=output_folder, file_name=f"LT_pseudo-distrib.jpg"
            )
            IO.pseudodistrib(
                UT_pd, RoI["genomic"][0:2], resolution, output_folder=output_folder, file_name=f"UT-pseudo-distrib.jpg"
            )

            # Plot pseudo-distributions and persistent maxima:
            IO.pseudodistrib(
                LT_pd,
                RoI["genomic"][0:2],
                resolution,
                coords2scatter=[LT_MPs_in_RoI],
                colors=["blue"],
                output_folder=output_folder,
                title=None,
                file_name=f"LT_pseudo-distrib_and_pers-maxima.jpg",
                display=False,
            )
            IO.pseudodistrib(
                UT_pd,
                RoI["genomic"][0:2],
                resolution,
                coords2scatter=[UT_MPs_in_RoI],
                colors=["blue"],
                output_folder=output_folder,
                title=None,
                file_name=f"UT_pseudo-distrib_and_pers-maxima.jpg",
                display=False,
            )

            # Plot the region of interest of Iproc with over-imposed vertical lines for seeds:
            if Iproc_RoI is not None:
                IO.HiC_and_sites(
                    Iproc_RoI,
                    LT_MPs_in_RoI,
                    RoI["genomic"],
                    resolution,
                    where="lower",
                    plot_in_bp=True,
                    output_folder=output_folder,
                    display=False,
                    file_name=f"LT_seeds.jpg",
                    title=None,
                )
                IO.HiC_and_sites(
                    Iproc_RoI,
                    UT_MPs_in_RoI,
                    RoI["genomic"],
                    resolution,
                    where="upper",
                    plot_in_bp=True,
                    output_folder=output_folder,
                    display=False,
                    file_name=f"UT_seeds.jpg",
                    title=None,
                )

    return result


def step_3(
    result: IO.Result,
    L,
    U,
    resolution,
    genomic_belt,
    max_width,
    constrain_height,
    loc_pers_min,
    loc_trend_min,
    Iproc_RoI=None,
    RoI=None,
    output_folder=None,
    map=map,
) -> IO.Result:
    if result.empty:
        print("3) No candidates found by step 2. Returning immediately!")
        return result

    # Retrieve data:
    LT_mPs = result.get("persistent_minimum_points", "LT")
    UT_mPs = result.get("persistent_minimum_points", "UT")
    LT_MPs = result.get("persistent_maximum_points", "LT")
    UT_MPs = result.get("persistent_maximum_points", "UT")
    LT_pseudo_distrib = result.get("pseudodistribution", "LT")
    UT_pseudo_distrib = result.get("pseudodistribution", "UT")

    start_time = time.time()

    print("3.1) Width estimation")
    print("3.1.1) Estimating widths (equiv. HIoIs, where HIoI stands for Horizontal Interval of Interest)...")

    # Complement mPs with:
    # the global minimum (if any) that is to the left of the leftmost persistent maximum
    # AND
    # the global minimum (if any) that is to the right of the rightmost persistent maximum
    LT_L_nb = np.arange(0, LT_MPs[0])
    LT_R_nb = np.arange(LT_MPs[-1], L.shape[0])
    UT_L_nb = np.arange(0, UT_MPs[0])
    UT_R_nb = np.arange(UT_MPs[-1], U.shape[0])
    LT_L_mP = np.argmin(LT_pseudo_distrib[LT_L_nb]) if len(LT_L_nb) > 0 else -1
    LT_R_mP = LT_MPs[-1] + np.argmin(LT_pseudo_distrib[LT_R_nb]) if len(LT_R_nb) > 0 else -1
    UT_L_mP = np.argmin(UT_pseudo_distrib[UT_L_nb]) if len(UT_L_nb) > 0 else -1
    UT_R_mP = UT_MPs[-1] + np.argmin(UT_pseudo_distrib[UT_R_nb]) if len(UT_R_nb) > 0 else -1

    LT_bounded_mPs = np.concatenate(([max(LT_L_mP, 0)], LT_mPs, [max(LT_R_mP, L.shape[0])]))
    UT_bounded_mPs = np.concatenate(([max(UT_L_mP, 0)], UT_mPs, [max(UT_R_mP, U.shape[0])]))

    # List of pairs (pair = left and right boundaries):
    # Choose the variable criterion between max_ascent and max_perc_descent
    # ---> When variable criterion is set to max_ascent, set the variable max_ascent
    # ---> When variable criterion is set to max_perc_descent, set the variable max_perc_descent
    LT_HIoIs = finders.find_HIoIs(
        LT_pseudo_distrib, LT_MPs, LT_bounded_mPs, int(max_width / (2 * resolution)) + 1, map=map
    )
    UT_HIoIs = finders.find_HIoIs(
        UT_pseudo_distrib, UT_MPs, UT_bounded_mPs, int(max_width / (2 * resolution)) + 1, map=map
    )

    # List of left or right boundaries:
    LT_L_bounds, LT_R_bounds = map(list, zip(*LT_HIoIs))
    UT_L_bounds, UT_R_bounds = map(list, zip(*UT_HIoIs))

    print("3.1.2) Updating list of Stripe objects with HIoIs...")
    stripes = result.get("stripes", "LT")
    for num_cand_stripe, (LT_L_bound, LT_R_bound) in enumerate(zip(LT_L_bounds, LT_R_bounds)):
        stripes[num_cand_stripe].set_horizontal_bounds(LT_L_bound, LT_R_bound)

    stripes = result.get("stripes", "UT")
    for num_cand_stripe, (UT_L_bound, UT_R_bound) in enumerate(zip(UT_L_bounds, UT_R_bounds)):
        stripes[num_cand_stripe].set_horizontal_bounds(UT_L_bound, UT_R_bound)

    if all([param is not None for param in [RoI, output_folder]]):

        print("3.1.3) Plots")
        # 3.1.3.1 "Finding HIoIs inside the region (RoI) selected above..."

        ids_LT_MPs_in_RoI, LT_MPs_in_RoI = _find_seeds_in_RoI(LT_MPs, result.roi["matrix"][0], result.roi["matrix"][1])
        ids_UT_MPs_in_RoI, UT_MPs_in_RoI = _find_seeds_in_RoI(UT_MPs, result.roi["matrix"][0], result.roi["matrix"][1])

        # Left and right boundaries in RoI:
        LT_L_bounds_in_RoI = np.array(LT_L_bounds)[ids_LT_MPs_in_RoI].tolist()
        LT_R_bounds_in_RoI = np.array(LT_R_bounds)[ids_LT_MPs_in_RoI].tolist()
        UT_L_bounds_in_RoI = np.array(UT_L_bounds)[ids_UT_MPs_in_RoI].tolist()
        UT_R_bounds_in_RoI = np.array(UT_R_bounds)[ids_UT_MPs_in_RoI].tolist()

        # 3.1.3.2 "Plotting pseudo-distributions and sites for the region selected above..."
        IoIs = [RoI["genomic"][0:2]] + [
            [LT_L_bound_in_RoI * resolution, (LT_R_bound_in_RoI + 1) * resolution]
            for (LT_L_bound_in_RoI, LT_R_bound_in_RoI) in zip(LT_L_bounds_in_RoI, LT_R_bounds_in_RoI)
        ]
        IO.pseudodistrib_and_HIoIs(
            LT_pseudo_distrib,
            IoIs,
            resolution,
            colors=["red"] + ["blue"] * len(LT_L_bounds_in_RoI),
            title=None,
            output_folder=output_folder,
            file_name=f"LT_pseudo-distrib_and_h-doms.jpg",
            display=False,
        )
        IoIs = [RoI["genomic"][0:2]] + [
            [UT_L_bound_in_RoI * resolution, (UT_R_bound_in_RoI + 1) * resolution]
            for (UT_L_bound_in_RoI, UT_R_bound_in_RoI) in zip(UT_L_bounds_in_RoI, UT_R_bounds_in_RoI)
        ]
        IO.pseudodistrib_and_HIoIs(
            UT_pseudo_distrib,
            IoIs,
            resolution,
            colors=["red"] + ["blue"] * len(UT_L_bounds_in_RoI),
            title=None,
            output_folder=output_folder,
            file_name=f"UT_pseudo-distrib_and_h-doms.jpg",
            display=False,
        )

        if Iproc_RoI is not None:

            # Projecting left and right boundaries onto the sub-intervals:
            LT_bounds_in_RoI_proj = [
                [max(0, LT_L_bound - RoI["matrix"][0]), min(LT_R_bound - RoI["matrix"][0], Iproc_RoI.shape[0] - 1)]
                for (LT_L_bound, LT_R_bound) in zip(LT_L_bounds_in_RoI, LT_R_bounds_in_RoI)
            ]
            UT_bounds_in_RoI_proj = [
                [max(0, UT_L_bound - RoI["matrix"][0]), min(UT_R_bound - RoI["matrix"][0], Iproc_RoI.shape[0] - 1)]
                for (UT_L_bound, UT_R_bound) in zip(UT_L_bounds_in_RoI, UT_R_bounds_in_RoI)
            ]
            LT_bounds_in_RoI_proj_gen_coord = [[a[0] * resolution, a[1] * resolution] for a in LT_bounds_in_RoI_proj]
            UT_bounds_in_RoI_proj_gen_coord = [[a[0] * resolution, a[1] * resolution] for a in UT_bounds_in_RoI_proj]

            # Slices, i.e., intervals determined by a pair of left & right boundaries:
            LT_slices2keep_proj = [
                list(range(LT_bound_in_RoI_proj[0], LT_bound_in_RoI_proj[1] + 1))
                for LT_bound_in_RoI_proj in LT_bounds_in_RoI_proj
            ]
            UT_slices2keep_proj = [
                list(range(UT_bound_in_RoI_proj[0], UT_bound_in_RoI_proj[1] + 1))
                for UT_bound_in_RoI_proj in UT_bounds_in_RoI_proj
            ]

            # 3.1.3.3 "Plotting RoI restricted to HIoIs..."
            # Setting rows/columns not included in proj_LT_ids_2_keep to zero:

            Iproc0_RoI_LT_sliced = np.triu(Iproc_RoI)
            Iproc0_RoI_UT_sliced = np.tril(Iproc_RoI)
            for num_slice, LT_slice2keep_proj in enumerate(LT_slices2keep_proj):

                Iproc0_RoI_LT_cur_sliced = np.triu(Iproc_RoI)
                for idx2keep in LT_slice2keep_proj:
                    Iproc0_RoI_LT_cur_sliced[idx2keep:, idx2keep] = Iproc_RoI[idx2keep:, idx2keep]
                    Iproc0_RoI_LT_sliced[idx2keep:, idx2keep] = Iproc_RoI[idx2keep:, idx2keep]

            IO.HiC_and_HIoIs(
                Iproc0_RoI_LT_sliced,
                LT_bounds_in_RoI_proj_gen_coord,
                RoI["genomic"],
                resolution,
                title=None,
                output_folder=output_folder,
                plot_in_bp=True,
                where="lower",
                file_name=f"LT_all_h-doms.jpg",
                display=False,
            )

            for num_slice, UT_slice2keep_proj in enumerate(UT_slices2keep_proj):

                Iproc0_RoI_UT_cur_sliced = np.tril(Iproc_RoI)
                for idx2keep in UT_slice2keep_proj:
                    Iproc0_RoI_UT_cur_sliced[: idx2keep + 1, idx2keep] = Iproc_RoI[: idx2keep + 1, idx2keep]
                    Iproc0_RoI_UT_sliced[: idx2keep + 1, idx2keep] = Iproc_RoI[: idx2keep + 1, idx2keep]

            IO.HiC_and_HIoIs(
                Iproc0_RoI_UT_sliced,
                UT_bounds_in_RoI_proj_gen_coord,
                RoI["genomic"],
                resolution,
                title=None,
                output_folder=output_folder,
                plot_in_bp=True,
                where="upper",
                file_name=f"UT_all_h-doms.jpg",
                display=False,
            )

    print(f"Execution time: {time.time() - start_time} seconds ---")

    print("3.2) Height estimation")

    start_time = time.time()

    print("3.2.1) Estimating heights (equiv. VIoIs, where VIoI stands for Vertical Interval of Interest)...")
    if be_verbose and all([param is not None for param in [RoI, output_folder]]):
        LT_VIoIs, LT_peaks_ids = finders.find_VIoIs(
            L,
            LT_MPs,
            LT_HIoIs,
            VIoIs2plot=ids_LT_MPs_in_RoI,
            max_height=int(genomic_belt / resolution),
            threshold_cut=loc_trend_min,
            min_persistence=loc_pers_min,
            where="lower",
            output_folder=f"{output_folder}local_pseudodistributions/",
            map=map,
        )
        UT_VIoIs, UT_peaks_ids = finders.find_VIoIs(
            U,
            UT_MPs,
            UT_HIoIs,
            VIoIs2plot=ids_UT_MPs_in_RoI,
            max_height=int(genomic_belt / resolution),
            threshold_cut=loc_trend_min,
            min_persistence=loc_pers_min,
            where="upper",
            output_folder=f"{output_folder}local_pseudodistributions/",
            map=map,
        )
    else:
        LT_VIoIs, LT_peaks_ids = finders.find_VIoIs(
            L,
            LT_MPs,
            LT_HIoIs,
            VIoIs2plot=None,
            max_height=int(genomic_belt / resolution),
            threshold_cut=loc_trend_min,
            min_persistence=loc_pers_min,
            where="lower",
            output_folder=None,
            map=map,
        )
        UT_VIoIs, UT_peaks_ids = finders.find_VIoIs(
            U,
            UT_MPs,
            UT_HIoIs,
            VIoIs2plot=None,
            max_height=int(genomic_belt / resolution),
            threshold_cut=loc_trend_min,
            min_persistence=loc_pers_min,
            where="upper",
            output_folder=None,
            map=map,
        )

    # List of left or right boundaries:
    LT_U_bounds, LT_D_bounds = map(list, zip(*LT_VIoIs))
    UT_U_bounds, UT_D_bounds = map(list, zip(*UT_VIoIs))

    print("3.2.2) Updating list of Stripe objects with VIoIs...")
    lt_stripes = result.get("stripes", "LT")
    for num_cand_stripe, (LT_U_bound, LT_D_bound) in enumerate(zip(LT_U_bounds, LT_D_bounds)):
        lt_stripes[num_cand_stripe].set_vertical_bounds(LT_U_bound, LT_D_bound)
    ut_stripes = result.get("stripes", "UT")
    for num_cand_stripe, (UT_U_bound, UT_D_bound) in enumerate(zip(UT_U_bounds, UT_D_bounds)):
        ut_stripes[num_cand_stripe].set_vertical_bounds(UT_U_bound, UT_D_bound)

    print(f"Execution time: {time.time() - start_time} seconds ---")

    if RoI is not None:

        print("3.3) Finding HIoIs and VIoIs inside the region (RoI) selected above...")

        # Restricting to the RoI:
        LT_HIoIs_in_RoI = np.array(LT_HIoIs)[ids_LT_MPs_in_RoI].tolist()
        UT_HIoIs_in_RoI = np.array(UT_HIoIs)[ids_UT_MPs_in_RoI].tolist()
        LT_VIoIs_in_RoI = np.array(LT_VIoIs)[ids_LT_MPs_in_RoI].tolist()
        UT_VIoIs_in_RoI = np.array(UT_VIoIs)[ids_UT_MPs_in_RoI].tolist()
        LT_HIoIs_in_RoI_proj = [
            [LT_HIoI_RoI[0] - RoI["matrix"][0], LT_HIoI_RoI[1] - RoI["matrix"][0]] for LT_HIoI_RoI in LT_HIoIs_in_RoI
        ]
        UT_HIoIs_in_RoI_proj = [
            [UT_HIoI_RoI[0] - RoI["matrix"][0], UT_HIoI_RoI[1] - RoI["matrix"][0]] for UT_HIoI_RoI in UT_HIoIs_in_RoI
        ]
        LT_VIoIs_in_RoI_proj = [
            [LT_VIoI_RoI[0] - RoI["matrix"][0], LT_VIoI_RoI[1] - RoI["matrix"][0]] for LT_VIoI_RoI in LT_VIoIs_in_RoI
        ]
        UT_VIoIs_in_RoI_proj = [
            [UT_VIoI_RoI[0] - RoI["matrix"][0], UT_VIoI_RoI[1] - RoI["matrix"][0]] for UT_VIoI_RoI in UT_VIoIs_in_RoI
        ]

        if constrain_height:
            LT_peaks_ids_RoI = [
                LT_peaks_ids_in_candida_in_RoI
                for n, LT_peaks_ids_in_candida_in_RoI in enumerate(LT_peaks_ids)
                if n in ids_LT_MPs_in_RoI
            ]
            LT_peaks_ids_RoI_proj = [
                [
                    LT_peak_idx_in_candida_in_RoI - RoI["matrix"][0]
                    for LT_peak_idx_in_candida_in_RoI in LT_peaks_ids_in_candida_in_RoI
                    if 0 < LT_peak_idx_in_candida_in_RoI - RoI["matrix"][0] < Iproc_RoI.shape[0]
                ]
                for LT_peaks_ids_in_candida_in_RoI in LT_peaks_ids_RoI
            ]
            UT_peaks_ids_RoI = [
                UT_peaks_ids_in_candida_in_RoI
                for n, UT_peaks_ids_in_candida_in_RoI in enumerate(UT_peaks_ids)
                if n in ids_UT_MPs_in_RoI
            ]
            UT_peaks_ids_RoI_proj = [
                [
                    UT_peak_idx_in_candida_in_RoI - RoI["matrix"][0]
                    for UT_peak_idx_in_candida_in_RoI in UT_peaks_ids_in_candida_in_RoI
                    if 0 < UT_peak_idx_in_candida_in_RoI - RoI["matrix"][0] < Iproc_RoI.shape[0]
                ]
                for UT_peaks_ids_in_candida_in_RoI in UT_peaks_ids_RoI
            ]

        print("3.4) Plotting candidate stripes restricted to HIoIs...")

        # Plot of the candidate stripes within the RoI:
        IO.plot_stripes(
            Iproc_RoI,
            LT_HIoIs_in_RoI_proj,
            LT_VIoIs_in_RoI_proj,
            [],
            [],
            RoI["genomic"],
            resolution,
            plot_in_bp=True,
            output_folder=output_folder,
            file_name=f"LT_all_candidates.jpg",
            title=None,
            display=False,
        )

        if constrain_height:
            IO.plot_stripes_and_peaks(
                Iproc_RoI,
                LT_HIoIs_in_RoI_proj,
                LT_VIoIs_in_RoI_proj,
                [],
                [],
                LT_peaks_ids_RoI_proj,
                [],
                RoI["genomic"],
                resolution,
                plot_in_bp=True,
                output_folder=output_folder,
                file_name=f"LT_all_candidates_and_peaks.jpg",
                title=None,
                display=False,
            )

        IO.plot_stripes(
            Iproc_RoI,
            [],
            [],
            UT_HIoIs_in_RoI_proj,
            UT_VIoIs_in_RoI_proj,
            RoI["genomic"],
            resolution,
            plot_in_bp=True,
            output_folder=output_folder,
            file_name=f"UT_all_candidates.jpg",
            title=None,
            display=False,
        )

        if constrain_height:
            IO.plot_stripes_and_peaks(
                Iproc_RoI,
                [],
                [],
                UT_HIoIs_in_RoI_proj,
                UT_VIoIs_in_RoI_proj,
                [],
                UT_peaks_ids_RoI_proj,
                RoI["genomic"],
                resolution,
                plot_in_bp=True,
                output_folder=output_folder,
                file_name=f"UT_all_candidates_and_peaks.jpg",
                title=None,
                display=False,
            )

    print("3.6) Bar plots of widths and heights...")
    LT_widths = [HIoI[1] - HIoI[0] for HIoI in LT_HIoIs]
    LT_heights = [VIoI[1] - VIoI[0] for VIoI in LT_VIoIs]
    UT_widths = [HIoI[1] - HIoI[0] for HIoI in UT_HIoIs]
    UT_heights = [VIoI[1] - VIoI[0] for VIoI in UT_VIoIs]
    if be_verbose and output_folder is not None:
        fig, ax = plt.subplots(1, 1)
        sns.histplot(
            data=pd.DataFrame(LT_widths),
            kde=False,
            legend=False,
            fill=True,
            discrete=True,
            color="#2F539B",
            edgecolor=None,
            alpha=1,
        )
        plt.xlim(0, max(max(LT_widths), max(UT_widths)) + 1)
        plt.title("Widths")
        plt.savefig(f"{output_folder}/LT_histogram_widths.jpg", bbox_inches="tight")
        plt.close()
        fig, ax = plt.subplots(1, 1)
        sns.histplot(
            data=pd.DataFrame(UT_widths),
            kde=False,
            legend=False,
            fill=True,
            discrete=True,
            color="#2F539B",
            edgecolor=None,
            alpha=1,
        )
        plt.xlim(0, max(max(LT_widths), max(UT_widths)) + 1)
        plt.title("Widths")
        plt.savefig(f"{output_folder}/UT_histogram_widths.jpg", bbox_inches="tight")
        plt.close()
        fig, ax = plt.subplots(1, 1)
        sns.histplot(
            data=pd.DataFrame(LT_heights),
            kde=False,
            legend=False,
            fill=True,
            discrete=True,
            color="#2F539B",
            edgecolor=None,
            alpha=1,
        )
        plt.xlim(0, max(max(LT_widths), max(UT_heights)) + 1)
        plt.title("Heights")
        plt.savefig(f"{output_folder}/LT_histogram_heights.jpg", bbox_inches="tight")
        plt.close()
        fig, ax = plt.subplots(1, 1)
        sns.histplot(
            data=pd.DataFrame(UT_heights),
            kde=False,
            legend=False,
            fill=True,
            discrete=True,
            color="#2F539B",
            edgecolor=None,
            alpha=1,
        )
        plt.xlim(0, max(max(LT_heights), max(UT_heights)) + 1)
        plt.title("Heights")
        plt.savefig(f"{output_folder}/UT_histogram_heights.jpg", bbox_inches="tight")
        plt.close()

    return result


def step_4(
    result: IO.Result,
    L,
    U,
    resolution=None,
    thresholds_relative_change=None,
    Iproc_RoI=None,
    RoI=None,
    output_folder=None,
):
    if result.empty:
        print("4) No candidates found by step 2. Returning immediately!")
        return result

    print("4.1) Computing and saving biological descriptors")
    for LT_candidate_stripe in result.get("stripes", "LT"):
        LT_candidate_stripe.compute_biodescriptors(L)
    for UT_candidate_stripe in result.get("stripes", "UT"):
        UT_candidate_stripe.compute_biodescriptors(U)

    if all(param is not None for param in [resolution, thresholds_relative_change, Iproc_RoI, RoI, output_folder]):

        print("4.2) Thresholding...")

        # Retrieve data:
        LT_MPs = [c_s.seed for c_s in result.get("stripes", "LT")]
        UT_MPs = [c_s.seed for c_s in result.get("stripes", "UT")]
        LT_HIoIs = [[c_s.left_bound, c_s.right_bound] for c_s in result.get("stripes", "LT")]
        LT_VIoIs = [[c_s.top_bound, c_s.bottom_bound] for c_s in result.get("stripes", "LT")]
        UT_HIoIs = [[c_s.left_bound, c_s.right_bound] for c_s in result.get("stripes", "UT")]
        UT_VIoIs = [[c_s.top_bound, c_s.bottom_bound] for c_s in result.get("stripes", "UT")]

        for threshold in thresholds_relative_change:

            # Filtration:
            LT_candidates2keep = [
                index
                for index, rel_change in enumerate(s.rel_change for s in result.get("stripes", "LT"))
                if rel_change >= threshold
            ]
            UT_candidates2keep = [
                index
                for index, rel_change in enumerate(s.rel_change for s in result.get("stripes", "UT"))
                if rel_change >= threshold
            ]

            LT_filt_MPs = [LT_MPs[num_cand] for num_cand in LT_candidates2keep]
            LT_filt_HIoIs = [LT_HIoIs[num_cand] for num_cand in LT_candidates2keep]
            LT_filt_VIoIs = [LT_VIoIs[num_cand] for num_cand in LT_candidates2keep]

            UT_filt_MPs = [UT_MPs[num_cand] for num_cand in UT_candidates2keep]
            UT_filt_HIoIs = [UT_HIoIs[num_cand] for num_cand in UT_candidates2keep]
            UT_filt_VIoIs = [UT_VIoIs[num_cand] for num_cand in UT_candidates2keep]

            # Plotting stripes in range:
            if RoI is not None:
                LT_candidates2keep_in_RoI = np.where(
                    (np.array(LT_filt_MPs) > RoI["matrix"][0]) & (np.array(LT_filt_MPs) < RoI["matrix"][1])
                )[0]
                LT_filt_MPs_in_RoI = np.array(LT_filt_MPs)[LT_candidates2keep_in_RoI].tolist()
                LT_filt_MPs_in_RoI_proj = [a - RoI["matrix"][0] for a in LT_filt_MPs_in_RoI]
                LT_filt_HIoIs_in_RoI = np.array(LT_filt_HIoIs)[LT_candidates2keep_in_RoI].tolist()
                LT_filt_HIoIs_in_RoI_proj = [
                    [a[0] - RoI["matrix"][0], a[1] - RoI["matrix"][2]] for a in LT_filt_HIoIs_in_RoI
                ]
                LT_filt_VIoIs_in_RoI = np.array(LT_filt_VIoIs)[LT_candidates2keep_in_RoI].tolist()
                LT_filt_VIoIs_in_RoI_proj = [
                    [a[0] - RoI["matrix"][0], a[1] - RoI["matrix"][2]] for a in LT_filt_VIoIs_in_RoI
                ]

                IO.plot_stripes(
                    Iproc_RoI,
                    LT_filt_HIoIs_in_RoI_proj,
                    LT_filt_VIoIs_in_RoI_proj,
                    [],
                    [],
                    RoI["genomic"],
                    resolution,
                    plot_in_bp=True,
                    output_folder=f"{output_folder}",
                    file_name=f"LT_{threshold:.2f}.jpg",
                    title=None,
                    display=False,
                )

                UT_candidates2keep_in_RoI = np.where(
                    (np.array(UT_filt_MPs) > RoI["matrix"][0]) & (np.array(UT_filt_MPs) < RoI["matrix"][1])
                )[0]
                UT_filt_MPs_in_RoI = np.array(UT_filt_MPs)[UT_candidates2keep_in_RoI].tolist()
                UT_filt_MPs_in_RoI_proj = [a - RoI["matrix"][0] for a in UT_filt_MPs_in_RoI]
                UT_filt_HIoIs_in_RoI = np.array(UT_filt_HIoIs)[UT_candidates2keep_in_RoI].tolist()
                UT_filt_HIoIs_in_RoI_proj = [
                    [a[0] - RoI["matrix"][0], a[1] - RoI["matrix"][2]] for a in UT_filt_HIoIs_in_RoI
                ]
                UT_filt_VIoIs_in_RoI = np.array(UT_filt_VIoIs)[UT_candidates2keep_in_RoI].tolist()
                UT_filt_VIoIs_in_RoI_proj = [
                    [a[0] - RoI["matrix"][0], a[1] - RoI["matrix"][2]] for a in UT_filt_VIoIs_in_RoI
                ]

                IO.plot_stripes(
                    Iproc_RoI,
                    [],
                    [],
                    UT_filt_HIoIs_in_RoI_proj,
                    UT_filt_VIoIs_in_RoI_proj,
                    RoI["genomic"],
                    resolution,
                    plot_in_bp=True,
                    output_folder=f"{output_folder}",
                    file_name=f"UT_{threshold:.2f}.jpg",
                    title=None,
                    display=False,
                )

    return result
