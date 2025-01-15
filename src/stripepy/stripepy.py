# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import itertools
import pathlib
import time
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as ss
import structlog
from numpy.typing import NDArray

from . import IO, plot
from .utils import TDA, common, finders, regressions, stripe


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


def _extract_RoIs(I: ss.csr_matrix, RoI: Dict[str, List[int]]) -> Optional[NDArray]:
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
    Optional[NDArray]
        dense matrix with the interactions for the regions of interest (or None in case RoI itself is None)
    """

    if RoI is None:
        return None

    rows = cols = slice(RoI["matrix"][0], RoI["matrix"][1])
    I_RoI = I[rows, cols].toarray()
    return I_RoI


def _compute_global_pseudodistribution(
    T: ss.csr_matrix, smooth: bool = True, decimal_places: int = 10
) -> NDArray[float]:
    """
    Given a sparse matrix T, marginalize it, scale the marginal so that maximum is 1, and then smooth it.

    Parameters
    ----------
    T: ss.csr_matrix
        the sparse matrix to be processed
    smooth: bool
        if set to True, smoothing is applied to the pseudo-distribution (default value is True)
    decimal_places: int
        the number of decimal places to truncate the pseudo-distribution to.
        Pass -1 to not truncate the pseudo-distribution values

    Returns
    -------
    NDArray[float]
        a vector with the re-scaled and smoothed marginals.
    """

    pseudo_dist = np.squeeze(np.asarray(np.sum(T, axis=0)))  # marginalization
    pseudo_dist /= np.max(pseudo_dist)  # scaling
    if smooth:
        pseudo_dist = np.maximum(regressions._compute_wQISA_predictions(pseudo_dist, 11), pseudo_dist)  # smoothing

    if decimal_places >= 0:
        # We need to truncate FP numbers to ensure that later steps generate consistent results
        # even in the presence to very minor numeric differences on different platforms.
        return common.truncate_np(pseudo_dist, decimal_places)

    return pseudo_dist


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
    ps_mPs: NDArray[int],
    pers_of_ps_mPs: NDArray[float],
    ps_MPs: NDArray[int],
    pers_of_ps_MPs: NDArray[float],
    mask: NDArray[int],
) -> Tuple[NDArray[int], NDArray[float], NDArray[int], NDArray[float]]:
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

    return np.array(ps_mPs_2), np.array(pers_of_ps_mPs_2), np.array(ps_MPs_2), np.array(pers_of_ps_MPs_2)


def step_1(I, genomic_belt, resolution, RoI=None, output_folder=None, logger=None):
    if logger is None:
        logger = structlog.get_logger()

    logger.bind(step=(1, 1)).info("applying log-transformation")
    Iproc = _log_transform(I)

    logger.bind(step=(1, 2)).info("focusing on a neighborhood of the main diagonal")
    LT_Iproc, UT_Iproc = _band_extraction(Iproc, resolution, genomic_belt)
    nnz0 = I.count_nonzero()
    nnz1 = LT_Iproc.count_nonzero()
    delta = nnz0 - nnz1
    logger.bind(step=(1, 2)).info("removed %.2f%% of the non-zero entries (%d/%d)", (delta / nnz0) * 100, delta, nnz0)

    logger.bind(step=(1, 3)).info("projecting interactions onto [1, 0]")
    Iproc, LT_Iproc, UT_Iproc = _scale_Iproc(Iproc, LT_Iproc, UT_Iproc)

    return LT_Iproc, UT_Iproc, _extract_RoIs(Iproc, RoI)


def step_2(
    chrom_name: str,
    chrom_size: int,
    L: ss.csr_matrix,
    U: ss.csr_matrix,
    min_persistence: float,
    logger=None,
) -> IO.Result:
    if logger is None:
        logger = structlog.get_logger()

    logger.bind(step=(2, 1, 0)).info("computing global 1D pseudo-distributions...")
    LT_pd = _compute_global_pseudodistribution(L, smooth=True)
    UT_pd = _compute_global_pseudodistribution(U, smooth=True)

    result = IO.Result(chrom_name, chrom_size)

    result.set_min_persistence(min_persistence)

    result.set("pseudodistribution", LT_pd, "LT")
    result.set("pseudodistribution", UT_pd, "UT")

    logger.bind(step=(2, 2, 0)).info(
        "detection of persistent maxima and corresponding minima for lower- and upper-triangular matrices..."
    )
    logger.bind(step=(2, 2, 0)).info("all maxima and their persistence")

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

    logger.bind(step=(2, 2, 1)).info("lower triangular part")
    LT_ps_mPs, pers_of_LT_ps_mPs, LT_ps_MPs, pers_of_LT_ps_MPs = TDA.TDA(LT_pd, min_persistence=min_persistence)

    logger.bind(step=(2, 2, 2)).info("upper triangular part")
    # Here, LT_ps_mPs means that the lower-triangular minimum points are sorted w.r.t. persistence
    # (NOTATION: ps = persistence-sorted)
    UT_ps_mPs, pers_of_UT_ps_mPs, UT_ps_MPs, pers_of_UT_ps_MPs = TDA.TDA(UT_pd, min_persistence=min_persistence)
    # NB: Maxima are sorted w.r.t. their persistence... and this sorting is applied to minima too,
    # so that each maximum is still paired to its minimum.

    # Maximum and minimum points sorted w.r.t. coordinates (NOTATION: cs = coordinate-sorted):
    LT_mPs, LT_pers_of_mPs = common.sort_values(LT_ps_mPs, pers_of_LT_ps_mPs)
    LT_MPs, LT_pers_of_MPs = common.sort_values(LT_ps_MPs, pers_of_LT_ps_MPs)
    UT_mPs, UT_pers_of_mPs = common.sort_values(UT_ps_mPs, pers_of_UT_ps_mPs)
    UT_MPs, UT_pers_of_MPs = common.sort_values(UT_ps_MPs, pers_of_UT_ps_MPs)

    logger.bind(step=(2, 2, 3)).info("removing seeds overlapping sparse regions")
    LT_mask = _check_neighborhood(_compute_global_pseudodistribution(L, smooth=False))
    UT_mask = _check_neighborhood(_compute_global_pseudodistribution(U, smooth=False))
    x = _filter_extrema_by_sparseness(LT_mPs, LT_pers_of_mPs, LT_MPs, LT_pers_of_MPs, LT_mask)
    LT_mPs, LT_pers_of_mPs, LT_MPs, LT_pers_of_MPs = x
    x = _filter_extrema_by_sparseness(UT_mPs, UT_pers_of_mPs, UT_MPs, UT_pers_of_MPs, UT_mask)
    UT_mPs, UT_pers_of_mPs, UT_MPs, UT_pers_of_MPs = x
    if len(LT_MPs) < len(LT_ps_MPs):
        logger.bind(step=(2, 2, 3)).info(
            "lower triangular part: number of seed sites reduced from %d to %d", len(LT_ps_MPs), len(LT_MPs)
        )
    if len(UT_MPs) < len(UT_ps_MPs):
        logger.bind(step=(2, 2, 3)).info(
            "upper triangular part: number of seed sites reduced from %d to %d", len(UT_ps_MPs), len(UT_MPs)
        )
    if len(LT_MPs) == len(LT_ps_MPs) and len(UT_MPs) == len(UT_ps_MPs):
        logger.bind(step=(2, 2, 3)).info("no change in the number of seed sites")

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

    logger.bind(step=(2, 3, 1)).info("lower-triangular part: generating list of candidate stripes...")
    stripes = [
        stripe.Stripe(seed=LT_MP, top_pers=LT_pers_of_MP, where="lower_triangular")
        for LT_MP, LT_pers_of_MP in zip(LT_MPs, LT_pers_of_MPs)
    ]
    logger.bind(step=(2, 3, 1)).info("lower-triangular part: generated %d candidate stripes", len(stripes))
    result.set("stripes", stripes, "LT")

    logger.bind(step=(2, 3, 2)).info("upper-triangular part: generating list of candidate stripes...")
    stripes = [
        stripe.Stripe(seed=UT_MP, top_pers=UT_pers_of_MP, where="upper_triangular")
        for UT_MP, UT_pers_of_MP in zip(UT_MPs, UT_pers_of_MPs)
    ]
    logger.bind(step=(2, 3, 2)).info("upper-triangular part: generated %d candidate stripes", len(stripes))
    result.set("stripes", stripes, "UT")

    return result


def step_3(
    result: IO.Result,
    L: ss.csr_matrix,
    U: ss.csr_matrix,
    resolution: int,
    genomic_belt: int,
    max_width: int,
    loc_pers_min: float,
    loc_trend_min: float,
    map=map,
    logger=None,
) -> IO.Result:
    if logger is None:
        logger = structlog.get_logger()

    if result.empty:
        logger.bind(step=(3,)).warning("no candidates found by step 2: returning immediately!")
        return result

    # Retrieve data:
    LT_mPs = result.get("persistent_minimum_points", "LT")
    UT_mPs = result.get("persistent_minimum_points", "UT")
    LT_MPs = result.get("persistent_maximum_points", "LT")
    UT_MPs = result.get("persistent_maximum_points", "UT")
    LT_pseudo_distrib = result.get("pseudodistribution", "LT")
    UT_pseudo_distrib = result.get("pseudodistribution", "UT")

    start_time = time.time()

    logger.bind(step=(3, 1)).info("width estimation")
    logger.bind(step=(3, 1, 1)).info("estimating candidate stripe widths")

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

    LT_bounded_mPs = [(max(LT_L_mP, 0),), (max(LT_R_mP, L.shape[0]),)]
    UT_bounded_mPs = [(max(UT_L_mP, 0),), (max(UT_R_mP, U.shape[0]),)]
    # We need to check that the list of minimum points are not empty, otherwise np.concatenate will create an array with dtype=float
    if len(LT_mPs) != 0:
        LT_bounded_mPs.insert(1, LT_mPs)
    if len(UT_mPs) != 0:
        UT_bounded_mPs.insert(1, UT_mPs)

    LT_bounded_mPs = np.concatenate(LT_bounded_mPs, dtype=int)
    UT_bounded_mPs = np.concatenate(UT_bounded_mPs, dtype=int)

    # DataFrame with the left and right boundaries for each seed site
    LT_HIoIs = finders.find_HIoIs(
        pseudodistribution=LT_pseudo_distrib,
        seed_sites=LT_MPs,
        seed_site_bounds=LT_bounded_mPs,
        max_width=int(max_width / (2 * resolution)) + 1,
        logger=logger,
    )
    UT_HIoIs = finders.find_HIoIs(
        pseudodistribution=UT_pseudo_distrib,
        seed_sites=UT_MPs,
        seed_site_bounds=UT_bounded_mPs,
        max_width=int(max_width / (2 * resolution)) + 1,
        logger=logger,
    )

    logger.bind(step=(3, 1, 2)).info("updating candidate stripes with width information")
    stripes = result.get("stripes", "LT")
    LT_HIoIs.apply(
        lambda seed: stripes[seed.name].set_horizontal_bounds(seed["left_bound"], seed["right_bound"]),
        axis="columns",
    )

    stripes = result.get("stripes", "UT")
    UT_HIoIs.apply(
        lambda seed: stripes[seed.name].set_horizontal_bounds(seed["left_bound"], seed["right_bound"]),
        axis="columns",
    )

    logger.bind(step=(3, 1)).info("width estimation took %s", common.pretty_format_elapsed_time(start_time))

    logger.bind(step=(3, 2)).info("height estimation")
    start_time = time.time()

    LT_HIoIs = LT_HIoIs.to_numpy()  # TODO remove
    UT_HIoIs = UT_HIoIs.to_numpy()  # TODO remove

    logger.bind(step=(3, 2, 1)).info("estimating candidate stripe heights")
    LT_VIoIs, LT_peaks_ids = finders.find_VIoIs(
        L,
        LT_MPs,
        LT_HIoIs,
        max_height=int(genomic_belt / resolution),
        threshold_cut=loc_trend_min,
        min_persistence=loc_pers_min,
        where="lower",
        map=map,
    )
    UT_VIoIs, UT_peaks_ids = finders.find_VIoIs(
        U,
        UT_MPs,
        UT_HIoIs,
        max_height=int(genomic_belt / resolution),
        threshold_cut=loc_trend_min,
        min_persistence=loc_pers_min,
        where="upper",
        map=map,
    )

    # List of left or right boundaries:
    LT_U_bounds, LT_D_bounds = map(list, zip(*LT_VIoIs))
    UT_U_bounds, UT_D_bounds = map(list, zip(*UT_VIoIs))

    logger.bind(step=(3, 1, 2)).info("updating candidate stripes with height information")
    lt_stripes = result.get("stripes", "LT")
    for num_cand_stripe, (LT_U_bound, LT_D_bound) in enumerate(zip(LT_U_bounds, LT_D_bounds)):
        lt_stripes[num_cand_stripe].set_vertical_bounds(LT_U_bound, LT_D_bound)
    ut_stripes = result.get("stripes", "UT")
    for num_cand_stripe, (UT_U_bound, UT_D_bound) in enumerate(zip(UT_U_bounds, UT_D_bounds)):
        ut_stripes[num_cand_stripe].set_vertical_bounds(UT_U_bound, UT_D_bound)

    logger.bind(step=(3, 2)).info("height estimation took %s", common.pretty_format_elapsed_time(start_time))

    return result


def step_4(
    result: IO.Result,
    L: ss.csr_matrix,
    U: ss.csr_matrix,
    logger=None,
):
    if logger is None:
        logger = structlog.get_logger()

    if result.empty:
        logger.bind(step=(4,)).warning("no candidates found by step 2: returning immediately!")
        return result

    logger.bind(step=(4, 1)).info("computing stripe biological descriptors")
    for LT_candidate_stripe in result.get("stripes", "LT"):
        LT_candidate_stripe.compute_biodescriptors(L)
    for UT_candidate_stripe in result.get("stripes", "UT"):
        UT_candidate_stripe.compute_biodescriptors(U)

    return result


def _plot_pseudodistribution(
    result: IO.Result,
    resolution: int,
    matrix: Optional[NDArray],
    output_folder: pathlib.Path,
    logger,
):
    plt = common._import_pyplot()
    assert result.roi is not None

    logger.bind(step=(5, 1, 1)).info("plotting pseudo-distributions")
    start, end = result.roi["genomic"]
    fig, _ = plot.plot(
        result,
        resolution,
        plot_type="pseudodistribution",
        start=start,
        end=end,
    )
    fig.savefig(output_folder / "pseudo_distribution.jpg", dpi=256)
    plt.close(fig)

    if matrix is None:
        return

    logger.bind(step=(5, 1, 2)).info("plotting processed matrix with highlighted seed(s)")
    # Plot the region of interest of Iproc with over-imposed vertical lines for seeds:
    fig, _ = plot.plot(
        result,
        resolution,
        plot_type="matrix_with_seeds",
        matrix=matrix,
        start=start,
        end=end,
        log_scale=False,
    )
    fig.savefig(output_folder / "matrix_with_seeds.jpg", dpi=256)
    plt.close(fig)


def _plot_hic_and_hois(
    result: IO.Result,
    resolution: int,
    matrix: Optional[NDArray],
    output_folder: pathlib.Path,
    logger,
):
    plt = common._import_pyplot()
    assert result.roi is not None

    if matrix is None:
        return

    logger.bind(step=(5, 2, 1)).info("plotting regions overlapping with candidate stripe(s)")
    start, end = result.roi["genomic"]
    fig, _ = plot.plot(
        result,
        resolution,
        "matrix_with_stripes_masked",
        start=start,
        end=end,
        matrix=matrix,
    )
    fig.savefig(output_folder / "all_domains.jpg", dpi=256)
    plt.close(fig)

    logger.bind(step=(5, 2, 1)).info("plotting processed matrix with candidate stripe(s) highlighted")
    fig, _ = plot.plot(
        result,
        resolution,
        "matrix_with_stripes",
        start=start,
        end=end,
        matrix=matrix,
    )
    fig.savefig(output_folder / "all_candidates.jpg", dpi=256)
    plt.close(fig)


def _plot_geo_descriptors(
    result: IO.Result,
    resolution: int,
    output_folder: pathlib.Path,
    logger,
):
    plt = common._import_pyplot()
    logger.bind(step=(5, 3, 1)).info("generating histograms for geo-descriptors")
    fig, _ = plot.plot(result, resolution, plot_type="geo_descriptors", start=0, end=result.chrom[1])
    fig.savefig(output_folder / "geo_descriptors.jpg", dpi=256)
    plt.close(fig)


def _marginalize_matrix_lt(
    matrix: ss.csr_matrix, seed: int, left_bound: int, right_bound: int, max_height: int
) -> NDArray:
    i1, i2 = seed, min(seed + max_height, matrix.shape[0])
    j1, j2 = left_bound, right_bound
    v = np.asarray(matrix[i1:i2, :].tocsc()[:, j1:j2].sum(axis=1)).flatten()
    v /= v.max()

    return v


def _marginalize_matrix_ut(
    matrix: ss.csr_matrix, seed: int, left_bound: int, right_bound: int, max_height: int
) -> NDArray:
    i1, i2 = max(seed - max_height, 0), seed
    j1, j2 = left_bound, right_bound
    y = np.flip(np.asarray(matrix[i1:i2, :].tocsc()[:, j1:j2].sum(axis=1)).flatten())
    y /= y.max()

    return y


def _plot_local_pseudodistributions_helper(args):
    # TODO matrix should be passed around using shared mem?
    (
        i,
        seed,
        left_bound,
        right_bound,
        matrix,
        min_persistence,
        max_height,
        loc_trend_min,
        output_folder,
        location,
        logger,
    ) = args
    plt = common._import_pyplot()
    logger.bind(step=(5, 4, i)).debug("plotting local profile for seed %d (%s)", seed, location)

    if location == "LT":
        y = _marginalize_matrix_lt(matrix, seed, left_bound, right_bound, max_height)
    else:
        assert location == "UT"
        y = _marginalize_matrix_ut(matrix, seed, left_bound, right_bound, max_height)

    x = np.arange(seed, seed + len(y))
    y_hat = finders._compute_wQISA_predictions(y, 5)  # Basically: average of a 2-"pixel" neighborhood

    _, _, loc_maxima, _ = TDA.TDA(y, min_persistence=min_persistence)
    candidate_bound = [max(loc_maxima)]

    if len(loc_maxima) < 2:
        candidate_bound = np.where(y_hat < loc_trend_min)[0]
        if len(candidate_bound) == 0:
            candidate_bound = [len(y_hat) - 1]

    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y, color="red", linewidth=0.5, linestyle="solid")
    ax.plot(x, y_hat, color="black", linewidth=0.5, linestyle="solid")
    ax.plot(
        [seed + a for a in loc_maxima[:-1]],
        y[loc_maxima[:-1]],
        color="blue",
        marker=".",
        linestyle="",
        markersize=8 * 1.5,
    )
    ax.vlines(
        seed + candidate_bound[0],
        0.0,
        1.0,
        color="blue",
        linewidth=1.0,
        linestyle="dashed",
    )
    ax.set(xlim=(x[0], x[-1]), ylim=(0.0, 1.0))
    fig.tight_layout()
    fig.savefig(output_folder / f"{seed}_{location}_local_pseudodistribution.jpg", dpi=256)
    plt.close(fig)


def _plot_local_pseudodistributions(
    result: IO.Result,
    matrix_lt: ss.csr_matrix,
    matrix_ut: ss.csr_matrix,
    resolution: int,
    genomic_belt: int,
    loc_pers_min: float,
    loc_trend_min: float,
    output_folder: pathlib.Path,
    map,
    logger,
):
    output_folder.mkdir()

    start, end = result.roi["genomic"]
    max_height = int(np.ceil(genomic_belt / resolution))

    df = result.get_stripe_geo_descriptors("LT")
    df = df[(df["left_bound"] * resolution >= start) & (df["right_bound"] * resolution <= end)]

    logger.bind(step=(5, 4, 0)).info("plotting local profiles for %d LT seed(s)", len(df))
    list(
        itertools.filterfalse(
            None,
            map(
                _plot_local_pseudodistributions_helper,
                zip(
                    itertools.count(1, 1),
                    df["seed"],
                    df["left_bound"],
                    df["right_bound"],
                    itertools.repeat(matrix_lt),
                    itertools.repeat(loc_pers_min),
                    itertools.repeat(max_height),
                    itertools.repeat(loc_trend_min),
                    itertools.repeat(output_folder),
                    itertools.repeat("LT"),
                    itertools.repeat(logger),
                ),
            ),
        )
    )

    offset = len(df) + 1
    df = result.get_stripe_geo_descriptors("UT")
    df = df[(df["left_bound"] * resolution >= start) & (df["right_bound"] * resolution <= end)]

    list(
        itertools.filterfalse(
            None,
            map(
                _plot_local_pseudodistributions_helper,
                zip(
                    itertools.count(offset, 1),
                    df["seed"],
                    df["left_bound"],
                    df["right_bound"],
                    itertools.repeat(matrix_ut),
                    itertools.repeat(loc_pers_min),
                    itertools.repeat(max_height),
                    itertools.repeat(loc_trend_min),
                    itertools.repeat(output_folder),
                    itertools.repeat("UT"),
                    itertools.repeat(logger),
                ),
            ),
        )
    )


def _plot_stripes_helper(args):
    plt = common._import_pyplot()
    matrix, result, resolution, start, end, cutoff, output_folder, logger = args
    logger.debug("plotting stripes with cutoff=%.2f", cutoff)

    fig, _ = plot.plot(
        result, resolution, "matrix_with_stripes", start=start, end=end, matrix=matrix, relative_change_threshold=cutoff
    )
    dest = output_folder / f"stripes_{cutoff:.2f}.jpg"
    fig.savefig(dest, dpi=256)
    plt.close(fig)


def _get_stripes(result: IO.Result, resolution: int) -> pd.DataFrame:
    start, end = result.roi["genomic"]

    descriptors_lt = result.get_stripe_geo_descriptors("LT")
    descriptors_ut = result.get_stripe_geo_descriptors("UT")

    for col in ("seed", "left_bound", "right_bound", "top_bound", "bottom_bound"):
        descriptors_lt[col] *= resolution
        descriptors_ut[col] *= resolution

    descriptors_lt["rel_change"] = result.get_stripe_bio_descriptors("LT")["rel_change"].iloc[descriptors_lt.index]
    descriptors_ut["rel_change"] = result.get_stripe_bio_descriptors("UT")["rel_change"].iloc[descriptors_ut.index]

    df = pd.concat([descriptors_lt, descriptors_ut])
    return df[df["left_bound"].between(start, end, inclusive="both")]


def _plot_stripes(
    result: IO.Result,
    resolution: int,
    matrix: Optional[NDArray],
    output_folder: pathlib.Path,
    map,
    logger,
):
    assert result.roi is not None

    if matrix is None:
        return

    df = _get_stripes(result, resolution)

    cutoffs = {}
    for cutoff in np.linspace(0, 15, 76):
        num_stripes = (df["rel_change"] >= cutoff).sum()
        if num_stripes not in cutoffs:
            cutoffs[num_stripes] = cutoff

    start, end = result.roi["genomic"]

    list(
        itertools.filterfalse(
            None,
            map(
                _plot_stripes_helper,
                zip(
                    itertools.repeat(matrix),
                    itertools.repeat(result),
                    itertools.repeat(resolution),
                    itertools.repeat(start),
                    itertools.repeat(end),
                    cutoffs.values(),
                    itertools.repeat(output_folder),
                    itertools.repeat(logger),
                ),
            ),
        ),
    )


def step_5(
    result: IO.Result,
    resolution: int,
    gw_matrix_proc_lt: ss.csr_matrix,
    gw_matrix_proc_ut: ss.csr_matrix,
    raw_matrix: NDArray,
    proc_matrix: Optional[NDArray],
    genomic_belt: int,
    loc_pers_min: float,
    loc_trend_min: float,
    output_folder: pathlib.Path,
    map=map,
    logger=None,
):
    assert result.roi is not None

    plt = common._import_pyplot()

    if logger is None:
        logger = structlog.get_logger()

    chrom_name, chrom_size = result.chrom
    start, end = result.roi["genomic"]

    for directory in ("1_preprocessing", "2_TDA", "3_shape_analysis", "4_biological_analysis"):
        (output_folder / chrom_name / directory).mkdir(parents=True, exist_ok=True)

    dummy_result = IO.Result(chrom_name, chrom_size)

    matrix_output_paths = (
        output_folder / chrom_name / "1_preprocessing" / f"raw_matrix_{start}_{end}.jpg",
        output_folder / chrom_name / "1_preprocessing" / f"proc_matrix_{start}_{end}.jpg",
    )

    matrices = (raw_matrix, proc_matrix)

    for dest, matrix in zip(matrix_output_paths, matrices):
        fig, _ = plot.plot(
            dummy_result,
            resolution=resolution,
            plot_type="matrix",
            start=start,
            end=end,
            matrix=matrix,
            log_scale=False,
        )

        fig.savefig(dest, dpi=256)
        plt.close(fig)

    _plot_pseudodistribution(
        result,
        resolution,
        proc_matrix,
        output_folder / chrom_name / "2_TDA",
        logger,
    )
    _plot_hic_and_hois(
        result,
        resolution,
        proc_matrix,
        output_folder / chrom_name / "3_shape_analysis",
        logger,
    )
    _plot_geo_descriptors(
        result,
        resolution,
        output_folder / chrom_name / "3_shape_analysis",
        logger,
    )
    _plot_local_pseudodistributions(
        result,
        gw_matrix_proc_lt,
        gw_matrix_proc_ut,
        resolution,
        genomic_belt,
        loc_pers_min,
        loc_trend_min,
        output_folder / chrom_name / "3_shape_analysis" / "local_pseudodistributions",
        map,
        logger,
    )
    _plot_stripes(
        result,
        resolution,
        proc_matrix,
        output_folder / chrom_name / "4_biological_analysis",
        map,
        logger,
    )
