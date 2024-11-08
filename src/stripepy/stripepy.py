import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as ss
import seaborn as sns

from . import IO
from .configs import be_verbose
from .utils import TDA, finders, regressions, stripe


def log_transform(I: ss.csr_matrix) -> ss.csr_matrix:
    I.data[np.isnan(I.data)] = 0
    I.eliminate_zeros()
    Iproc = I.log1p()
    return Iproc


def step_1(I, genomic_belt, resolution, RoI=None, output_folder=None):

    print("1.1) Log-transformation...")
    Iproc = log_transform(I)

    print("1.2) Focusing on a neighborhood of the main diagonal...")
    matrix_belt = int(genomic_belt / resolution)

    LT_Iproc = ss.tril(Iproc, k=0, format="csr") - ss.tril(Iproc, k=-matrix_belt, format="csr")
    UT_Iproc = ss.triu(Iproc, k=0, format="csr") - ss.triu(Iproc, k=matrix_belt, format="csr")

    # Scaling
    print("1.3) Projection onto [0, 1]...")
    scaling_factor_Iproc = Iproc.max()
    Iproc /= scaling_factor_Iproc
    LT_Iproc /= scaling_factor_Iproc
    UT_Iproc /= scaling_factor_Iproc

    if RoI is not None:
        print("1.4) Extracting a Region of Interest (RoI) for plot purposes...")
        rows = cols = slice(RoI["matrix"][0], RoI["matrix"][1])
        I_RoI = I[rows, cols].toarray()
        Iproc_RoI = Iproc[rows, cols].toarray()

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
        I_RoI = None
        Iproc_RoI = None

    return LT_Iproc, UT_Iproc, Iproc_RoI


def step_2(L, U, resolution, thresh_pers_type, thresh_pers_value, hf, Iproc_RoI=None, RoI=None, output_folder=None):

    print("2.1) Global 1D pseudo-distributions...")

    # Pseudo-distributions:
    LT_pd = np.squeeze(np.asarray(np.sum(L, axis=0)))
    UT_pd = np.squeeze(np.asarray(np.sum(U, axis=0)))

    # Scaling:
    LT_pd /= np.max(LT_pd)
    UT_pd /= np.max(UT_pd)

    # Smoothing:
    LT_pd = np.maximum(regressions.compute_wQISA_predictions(LT_pd, 11), LT_pd)
    UT_pd = np.maximum(regressions.compute_wQISA_predictions(UT_pd, 11), UT_pd)

    # Keep track of all maxima and persistence values:
    hf["LT/"].create_dataset("pseudo-distribution", data=np.array(LT_pd))
    hf["UT/"].create_dataset("pseudo-distribution", data=np.array(UT_pd))

    print("2.2) Detection of persistent maxima and corresponding minima for lower- and upper-triangular matrices...")

    print("2.2.0) All maxima and their persistence")
    # NOTATION: mPs = minimum points, MPs = maximum Points, ps = persistence-sorted
    # NB: MPs are the actual sites of interest, i.e., the sites hosting linear patterns
    LT_ps_mPs, pers_of_LT_ps_mPs, LT_ps_MPs, pers_of_LT_ps_MPs = TDA.TDA(LT_pd, min_persistence=0)
    UT_ps_mPs, pers_of_UT_ps_mPs, UT_ps_MPs, pers_of_UT_ps_MPs = TDA.TDA(UT_pd, min_persistence=0)

    # Store results:
    hf["LT/"].create_dataset("minima_pts_and_persistence", data=np.array([LT_ps_mPs, pers_of_LT_ps_mPs]))
    hf["LT/"].create_dataset("maxima_pts_and_persistence", data=np.array([LT_ps_MPs, pers_of_LT_ps_MPs]))
    hf["UT/"].create_dataset("minima_pts_and_persistence", data=np.array([UT_ps_mPs, pers_of_UT_ps_mPs]))
    hf["UT/"].create_dataset("maxima_pts_and_persistence", data=np.array([UT_ps_MPs, pers_of_UT_ps_MPs]))

    if thresh_pers_type == "constant":
        min_persistence = thresh_pers_value
    else:
        # min_persistence = (np.quantile(LT_pers_of_MPs, 0.75) +
        #                    1.5 * (np.quantile(LT_pers_of_MPs, 0.75) - np.quantile(LT_pers_of_MPs, 0.25)))
        min_persistence_LT = np.quantile(pers_of_LT_ps_MPs, thresh_pers_value)
        min_persistence_UT = np.quantile(pers_of_UT_ps_MPs, thresh_pers_value)
        min_persistence = np.max(min_persistence_LT, min_persistence_UT)
        print(f"This quantile is used: {thresh_pers_value}")
    hf.attrs["thresholding_type"] = thresh_pers_type
    hf.attrs["min_persistence_used"] = min_persistence

    print("2.2.1) Lower triangular part")
    LT_ps_mPs, pers_of_LT_ps_mPs, LT_ps_MPs, pers_of_LT_ps_MPs = TDA.TDA(LT_pd, min_persistence=min_persistence)

    print("2.2.2) Upper triangular part")
    UT_ps_mPs, pers_of_UT_ps_mPs, UT_ps_MPs, pers_of_UT_ps_MPs = TDA.TDA(UT_pd, min_persistence=min_persistence)

    # NB: Maxima are sorted w.r.t. their persistence... and this sorting is applied to minima too,
    # so that each maximum is still paired to its minimum!

    # Maximum and minimum points sorted w.r.t. coordinates: permutations and inverse permutations
    # NOTATION: cs = coordinate-sorted
    LT_permutation_ps2cs_mPs = np.argsort(LT_ps_mPs)
    LT_permutation_ps2cs_MPs = np.argsort(LT_ps_MPs)
    UT_permutation_ps2cs_mPs = np.argsort(UT_ps_mPs)
    UT_permutation_ps2cs_MPs = np.argsort(UT_ps_MPs)

    # Maximum and minimum points sorted w.r.t. coordinates: actual application of permutations
    LT_mPs = np.array(LT_ps_mPs)[LT_permutation_ps2cs_mPs].tolist()
    LT_MPs = np.array(LT_ps_MPs)[LT_permutation_ps2cs_MPs].tolist()
    UT_mPs = np.array(UT_ps_mPs)[UT_permutation_ps2cs_mPs].tolist()
    UT_MPs = np.array(UT_ps_MPs)[UT_permutation_ps2cs_MPs].tolist()
    LT_pers_of_MPs = np.array(pers_of_LT_ps_MPs)[LT_permutation_ps2cs_MPs].tolist()
    UT_pers_of_MPs = np.array(pers_of_UT_ps_MPs)[UT_permutation_ps2cs_MPs].tolist()

    print("2.3) Storing into a list of Stripe objects...")
    candidate_stripes = {
        "lower": [
            stripe.Stripe(seed=LT_MP, top_pers=LT_pers_of_MP, where="lower_triangular")
            for LT_MP, LT_pers_of_MP in zip(LT_MPs, LT_pers_of_MPs)
        ],
        "upper": [
            stripe.Stripe(seed=UT_MP, top_pers=UT_pers_of_MP, where="upper_triangular")
            for UT_MP, UT_pers_of_MP in zip(UT_MPs, UT_pers_of_MPs)
        ],
    }

    # Dictionary containing everything that should be returned
    pseudo_distributions = dict()
    pseudo_distributions["lower"] = dict()
    pseudo_distributions["upper"] = dict()
    pseudo_distributions["lower"]["pseudo-distribution"] = LT_pd
    pseudo_distributions["upper"]["pseudo-distribution"] = UT_pd
    pseudo_distributions["lower"]["persistent_minimum_points"] = LT_mPs
    pseudo_distributions["upper"]["persistent_minimum_points"] = UT_mPs
    pseudo_distributions["lower"]["persistent_maximum_points"] = LT_MPs
    pseudo_distributions["upper"]["persistent_maximum_points"] = UT_MPs
    pseudo_distributions["lower"]["persistence_of_maximum_points"] = LT_pers_of_MPs
    pseudo_distributions["upper"]["persistence_of_maximum_points"] = UT_pers_of_MPs

    if RoI is not None:

        print("2.4) Finding sites inside the region selected above...")
        # Find sites within the range of interest -- lower-triangular:
        ids_LT_MPs_in_RoI = np.where((RoI["matrix"][0] <= np.array(LT_MPs)) & (np.array(LT_MPs) <= RoI["matrix"][1]))[0]
        LT_MPs_in_RoI = np.array(LT_MPs)[ids_LT_MPs_in_RoI].tolist()

        # Find sites within the range of interest -- upper-triangular:
        ids_UT_MPs_in_RoI = np.where((RoI["matrix"][2] <= np.array(UT_MPs)) & (np.array(UT_MPs) <= RoI["matrix"][3]))[0]
        UT_MPs_in_RoI = np.array(UT_MPs)[ids_UT_MPs_in_RoI].tolist()

        # Store indices of persistence maxima points inside the RoI:
        pseudo_distributions["lower"]["indices_persistent_maximum_points_in_RoI"] = ids_LT_MPs_in_RoI
        pseudo_distributions["upper"]["indices_persistent_maximum_points_in_RoI"] = ids_UT_MPs_in_RoI

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

    return pseudo_distributions, candidate_stripes


def step_3(
    L,
    U,
    resolution,
    genomic_belt,
    max_width,
    constrain_height,
    loc_pers_min,
    loc_trend_min,
    pseudo_distributions,
    candidate_stripes,
    hf,
    Iproc_RoI=None,
    RoI=None,
    output_folder=None,
):

    # Retrieve data:
    LT_mPs = pseudo_distributions["lower"]["persistent_minimum_points"]
    UT_mPs = pseudo_distributions["upper"]["persistent_minimum_points"]
    LT_MPs = pseudo_distributions["lower"]["persistent_maximum_points"]
    UT_MPs = pseudo_distributions["upper"]["persistent_maximum_points"]
    LT_pseudo_distrib = pseudo_distributions["lower"]["pseudo-distribution"]
    UT_pseudo_distrib = pseudo_distributions["upper"]["pseudo-distribution"]

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

    LT_bounded_mPs = [max(LT_L_mP, 0)] + LT_mPs + [max(LT_R_mP, L.shape[0])]
    UT_bounded_mPs = [max(UT_L_mP, 0)] + UT_mPs + [max(UT_R_mP, U.shape[0])]

    # List of pairs (pair = left and right boundaries):
    # Choose the variable criterion between max_ascent and max_perc_descent
    # ---> When variable criterion is set to max_ascent, set the variable max_ascent
    # ---> When variable criterion is set to max_perc_descent, set the variable max_perc_descent
    LT_HIoIs = finders.find_HIoIs(LT_pseudo_distrib, LT_MPs, LT_bounded_mPs, int(max_width / (2 * resolution)) + 1)
    UT_HIoIs = finders.find_HIoIs(UT_pseudo_distrib, UT_MPs, UT_bounded_mPs, int(max_width / (2 * resolution)) + 1)

    # List of left or right boundaries:
    LT_L_bounds, LT_R_bounds = map(list, zip(*LT_HIoIs))
    UT_L_bounds, UT_R_bounds = map(list, zip(*UT_HIoIs))

    print("3.1.2) Updating list of Stripe objects with HIoIs...")
    for num_cand_stripe, (LT_L_bound, LT_R_bound) in enumerate(zip(LT_L_bounds, LT_R_bounds)):
        candidate_stripes["lower"][num_cand_stripe].set_horizontal_bounds(LT_L_bound, LT_R_bound)
    for num_cand_stripe, (UT_L_bound, UT_R_bound) in enumerate(zip(UT_L_bounds, UT_R_bounds)):
        candidate_stripes["upper"][num_cand_stripe].set_horizontal_bounds(UT_L_bound, UT_R_bound)

    if all([param is not None for param in [RoI, output_folder]]):

        print("3.1.3) Plots")
        # 3.1.3.1 "Finding HIoIs inside the region (RoI) selected above..."

        # Recover indices of persistent maximum points in RoI:
        ids_LT_MPs_in_RoI = pseudo_distributions["lower"]["indices_persistent_maximum_points_in_RoI"]
        ids_UT_MPs_in_RoI = pseudo_distributions["upper"]["indices_persistent_maximum_points_in_RoI"]

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
        )

    # List of left or right boundaries:
    LT_U_bounds, LT_D_bounds = map(list, zip(*LT_VIoIs))
    UT_U_bounds, UT_D_bounds = map(list, zip(*UT_VIoIs))

    print("3.2.2) Updating list of Stripe objects with VIoIs...")
    for num_cand_stripe, (LT_U_bound, LT_D_bound) in enumerate(zip(LT_U_bounds, LT_D_bounds)):
        candidate_stripes["lower"][num_cand_stripe].set_vertical_bounds(LT_U_bound, LT_D_bound)
    for num_cand_stripe, (UT_U_bound, UT_D_bound) in enumerate(zip(UT_U_bounds, UT_D_bounds)):
        candidate_stripes["upper"][num_cand_stripe].set_vertical_bounds(UT_U_bound, UT_D_bound)

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

    print("3.5) Saving geometric descriptors...")
    LT_shape_descriptors = pd.DataFrame(
        {
            "seed": LT_MPs,
            "seed persistence": pseudo_distributions["lower"]["persistence_of_maximum_points"],
            "L-boundary": [HIoI[0] for HIoI in LT_HIoIs],
            "R-boundary": [HIoI[1] for HIoI in LT_HIoIs],
            "U-boundary": [VIoI[0] for VIoI in LT_VIoIs],
            "D-boundary": [VIoI[1] for VIoI in LT_VIoIs],
        }
    )

    UT_shape_descriptors = pd.DataFrame(
        {
            "seed": UT_MPs,
            "seed persistence": pseudo_distributions["upper"]["persistence_of_maximum_points"],
            "L-boundary": [HIoI[0] for HIoI in UT_HIoIs],
            "R-boundary": [HIoI[1] for HIoI in UT_HIoIs],
            "U-boundary": [VIoI[0] for VIoI in UT_VIoIs],
            "D-boundary": [VIoI[1] for VIoI in UT_VIoIs],
        }
    )

    col_names = ["seed", "seed persistence", "L-boundary", "R_boundary", "U-boundary", "D-boundary"]
    hf["LT/"].create_dataset("geo-descriptors", data=LT_shape_descriptors.values)
    hf["LT/geo-descriptors"].attrs["col_names"] = col_names
    hf["UT/"].create_dataset("geo-descriptors", data=UT_shape_descriptors.values)
    hf["UT/geo-descriptors"].attrs["col_names"] = col_names

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
        plt.clf()
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
        plt.clf()
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
        plt.clf()
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
        plt.clf()


def step_4(
    L,
    U,
    candidate_stripes,
    hf,
    resolution=None,
    thresholds_relative_change=None,
    Iproc_RoI=None,
    RoI=None,
    output_folder=None,
):

    print("4.1) Computing and saving biological descriptors")
    for LT_candidate_stripe in candidate_stripes["lower"]:
        LT_candidate_stripe.compute_biodescriptors(L)
    for UT_candidate_stripe in candidate_stripes["upper"]:
        UT_candidate_stripe.compute_biodescriptors(U)

    LT_biological_descriptors = pd.DataFrame(
        {
            "inner mean": [c_s.inner_mean for c_s in candidate_stripes["lower"]],
            "outer mean": [c_s.outer_mean for c_s in candidate_stripes["lower"]],
            "relative change": [c_s.rel_change for c_s in candidate_stripes["lower"]],
            "standard deviation": [c_s.inner_std for c_s in candidate_stripes["lower"]],
        }
    )

    UT_biological_descriptors = pd.DataFrame(
        {
            "inner mean": [c_s.inner_mean for c_s in candidate_stripes["upper"]],
            "outer mean": [c_s.outer_mean for c_s in candidate_stripes["upper"]],
            "relative change": [c_s.rel_change for c_s in candidate_stripes["upper"]],
            "standard deviation": [c_s.inner_std for c_s in candidate_stripes["upper"]],
        }
    )

    col_names = ["inner mean", "outer mean", "relative change", "standard deviation"]
    hf["LT/"].create_dataset("bio-descriptors", data=LT_biological_descriptors.values)
    hf["LT/bio-descriptors"].attrs["col_names"] = col_names
    hf["UT/"].create_dataset("bio-descriptors", data=UT_biological_descriptors.values)
    hf["UT/bio-descriptors"].attrs["col_names"] = col_names

    if all(param is not None for param in [resolution, thresholds_relative_change, Iproc_RoI, RoI, output_folder]):

        print("4.2) Thresholding...")

        # Retrieve data:
        LT_MPs = [c_s.seed for c_s in candidate_stripes["lower"]]
        UT_MPs = [c_s.seed for c_s in candidate_stripes["upper"]]
        LT_HIoIs = [[c_s.left_bound, c_s.right_bound] for c_s in candidate_stripes["lower"]]
        LT_VIoIs = [[c_s.top_bound, c_s.bottom_bound] for c_s in candidate_stripes["lower"]]
        UT_HIoIs = [[c_s.left_bound, c_s.right_bound] for c_s in candidate_stripes["upper"]]
        UT_VIoIs = [[c_s.top_bound, c_s.bottom_bound] for c_s in candidate_stripes["upper"]]

        for threshold in thresholds_relative_change:

            # Filtration:
            LT_candidates2keep = [
                index
                for index, rel_change in enumerate(LT_biological_descriptors["relative change"])
                if rel_change >= threshold
            ]
            UT_candidates2keep = [
                index
                for index, rel_change in enumerate(UT_biological_descriptors["relative change"])
                if rel_change >= threshold
            ]

            LT_filt_MPs = [LT_MPs[num_cand] for num_cand in LT_candidates2keep]
            LT_filt_HIoIs = [LT_HIoIs[num_cand] for num_cand in LT_candidates2keep]
            LT_filt_VIoIs = [LT_VIoIs[num_cand] for num_cand in LT_candidates2keep]

            UT_filt_MPs = [UT_MPs[num_cand] for num_cand in UT_candidates2keep]
            UT_filt_HIoIs = [UT_HIoIs[num_cand] for num_cand in UT_candidates2keep]
            UT_filt_VIoIs = [UT_VIoIs[num_cand] for num_cand in UT_candidates2keep]

            # # Save to bedpe
            # IO.save_candidates_bedpe(LT_filt_HIoIs, LT_filt_VIoIs, resolution, chr2test,
            #                          output_folder=f"{output_folder}",
            #                          file_name=f"LT_{threshold:.2f}.bedpe")
            # IO.save_candidates_bedpe(UT_filt_HIoIs, UT_filt_VIoIs, resolution, chr2test,
            #                          output_folder=f"{output_folder}",
            #                          file_name=f"UT_{threshold:.2f}.bedpe")

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


# print(f"Of the {len(LT_MPs) + len(UT_MPs)} original candidates, {len(LT_filt_MPs) + len(UT_filt_MPs)} survived")
