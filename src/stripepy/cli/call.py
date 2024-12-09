# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import contextlib
import datetime
import json
import multiprocessing as mp
import pathlib
import time
from importlib.metadata import version
from typing import Any, Dict

import h5py
import hictkpy
import numpy as np
import structlog

from stripepy import IO, others, stripepy

# TODO does this need to be global variable?
MIN_SIZE_CHROMOSOME = 2000000


# TODO can we remove this?
def save_terminal_groups(name, obj):
    if isinstance(obj, h5py.Group):
        has_subgroups = any(isinstance(child_obj, h5py.Group) for _, child_obj in obj.items())
        if not has_subgroups:
            terminal_group_names.append(name)  # TODO avoid global variables


# TODO can we remove this?
def save_all_datasets(name, obj):
    if isinstance(obj, h5py.Dataset):  # check if obj is a group or dataset
        dataset_names.append(name)  # TODO avoid global variables


def print_all_attributes(obj, parent=""):
    if isinstance(obj, h5py.Group) or isinstance(obj, h5py.File):
        for key, val in obj.attrs.items():
            print(f"{parent}/{key}: {val}")
        for key, sub_obj in obj.items():
            print_all_attributes(sub_obj, f"{parent}/{key}")
    elif isinstance(obj, h5py.Dataset):
        for key, val in obj.attrs.items():
            print(f"{parent}/{key}: {val}")


def write_param_summary(
    configs_input: Dict[str, Any],
    configs_thresholds: Dict[str, Any],
    configs_output: Dict[str, Any],
    configs_other: Dict[str, Any],
):
    logger = structlog.get_logger()
    logger.info("Arguments:")
    logger.info(f"--contact-map: {configs_input['contact-map']}")
    logger.info(f"--resolution: {configs_input['resolution']}")
    logger.info(f"--normalization: {configs_input['normalization']}")
    logger.info(f"--genomic-belt: {configs_input['genomic_belt']}")
    logger.info(f"--roi: {configs_input['roi']}")
    logger.info(f"--max-width: {configs_thresholds['max_width']}")
    logger.info(f"--glob-pers-min: {configs_thresholds['glob_pers_min']}")
    logger.info(f"--constrain-heights: {configs_thresholds['constrain_heights']}")
    logger.info(f"--loc-pers-min: {configs_thresholds['loc_pers_min']}")
    logger.info(f"--loc-trend-min: {configs_thresholds['loc_trend_min']}")
    logger.info(f"--output-folder: {configs_output['output_folder']}")
    logger.info(f"--force: {configs_output['force']}")
    logger.info(f"--nproc: {configs_other['nproc']}")


def _init_h5_file(
    dest: pathlib.Path, matrix_file: hictkpy.File, normalization: str, metadata: Dict[str, Any]
) -> h5py.File:
    h5 = h5py.File(dest, "w")
    h5.attrs["assembly"] = matrix_file.attributes().get("assembly", "unknown")
    h5.attrs["bin-size"] = matrix_file.resolution()
    h5.attrs["creation-date"] = datetime.datetime.now().isoformat()
    h5.attrs["format"] = "HDF5::StripePy"
    h5.attrs["format-url"] = "https://github.com/paulsengroup/StripePy"
    h5.attrs["format-version"] = 1
    h5.attrs["generated-by"] = f"StripePy v{version('stripepy')}"
    h5.attrs["metadata"] = json.dumps(metadata, indent=2)
    h5.attrs["normalization"] = normalization

    chroms = matrix_file.chromosomes(include_ALL=False)
    h5.create_group("/chroms")
    h5.create_dataset("/chroms/name", data=list(chroms.keys()))
    h5.create_dataset("/chroms/length", data=list(chroms.values()))

    return h5


def _generate_metadata_attribute(configs_input: Dict[str, Any], configs_thresholds: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "constrain-heights": configs_thresholds["constrain_heights"],
        "genomic-belt": configs_input["genomic_belt"],
        "global-persistence-minimum": configs_thresholds["glob_pers_min"],
        "local-persistence-minimum": configs_thresholds["loc_pers_min"],
        "local-trend-minimum": configs_thresholds["loc_trend_min"],
        "max-width": configs_thresholds["max_width"],
        "min-chromosome-size": MIN_SIZE_CHROMOSOME,
    }


def _create_empty_descriptor_dataset(h5: h5py.File, this_chr: str):
    h5.create_group(f"{this_chr}/stripes/LT/")
    h5.create_group(f"{this_chr}/stripes/UT/")

    # Define empty geo-descriptors:
    col_names = ["seed", "seed persistence", "L-boundary", "R_boundary", "U-boundary", "D-boundary"]
    h5[f"{this_chr}/stripes/LT/"].create_dataset("geo-descriptors", data=np.empty((0, len(col_names))))
    h5[f"{this_chr}/stripes/LT/geo-descriptors"].attrs["col_names"] = col_names
    h5[f"{this_chr}/stripes/UT/"].create_dataset("geo-descriptors", data=np.empty((0, len(col_names))))
    h5[f"{this_chr}/stripes/UT/geo-descriptors"].attrs["col_names"] = col_names

    # Define empty bio-descriptors:
    col_names = ["inner mean", "outer mean", "relative change", "standard deviation"]
    h5[f"{this_chr}/stripes/LT/"].create_dataset("bio-descriptors", data=np.empty((0, len(col_names))))
    h5[f"{this_chr}/stripes/LT/bio-descriptors"].attrs["col_names"] = col_names
    h5[f"{this_chr}/stripes/UT/"].create_dataset("bio-descriptors", data=np.empty((0, len(col_names))))
    h5[f"{this_chr}/stripes/UT/bio-descriptors"].attrs["col_names"] = col_names


def run(
    configs_input: Dict[str, Any],
    configs_thresholds: Dict[str, Any],
    configs_output: Dict[str, Any],
    configs_other: Dict[str, Any],
):
    # How long does stripepy take to analyze the whole Hi-C matrix?
    start_global_time = time.time()

    write_param_summary(configs_input, configs_thresholds, configs_output, configs_other)

    # Data loading:
    f, chr_starts, chr_ends, bp_lengths = others.cmap_loading(configs_input["contact-map"], configs_input["resolution"])

    # Remove existing folders:
    # configs_output["output_folder"] = (
    #     f"{configs_output['output_folder']}/{configs_input['contact-map'].stem}/{configs_input['resolution']}"
    # )

    IO.remove_and_create_folder(configs_output["output_folder"], configs_output["force"])

    # Extract a list of tuples where each tuple is (index, chr), e.g. (2,'chr3'):
    c_pairs = others.chromosomes_to_study(list(f.chromosomes().keys()), bp_lengths, MIN_SIZE_CHROMOSOME)

    with contextlib.ExitStack() as ctx:
        # Create HDF5 file to store candidate stripes:
        h5 = ctx.enter_context(
            _init_h5_file(
                configs_output["output_folder"] / "results.hdf5",
                hictkpy.File(configs_input["contact-map"], configs_input["resolution"]),
                configs_input["normalization"],
                _generate_metadata_attribute(configs_input, configs_thresholds),
            )
        )

        # Set up the process pool when appropriate
        if configs_other["nproc"] > 1:
            pool = ctx.enter_context(mp.Pool(configs_other["nproc"]))
        else:
            pool = None

        # Lopping over all chromosomes:
        for this_chr_idx, this_chr in c_pairs:

            print(f"\n{IO.ANSI.RED}CHROMOSOME {this_chr}{IO.ANSI.ENDC}")
            start_local_time = time.time()

            # Create a group for current chromosome:
            h5.create_group(f"{this_chr}/")

            # Removing and creating folders to store output files:
            # configs_input['roi'] = None
            if configs_input["roi"] is not None:
                IO.create_folders_for_plots(configs_output["output_folder"] / "plots" / this_chr)

            I = f.fetch(this_chr, normalization=configs_input["normalization"]).to_csr("full")

            # RoI:
            RoI = others.define_RoI(
                configs_input["roi"], chr_starts[this_chr_idx], chr_ends[this_chr_idx], configs_input["resolution"]
            )
            print(f"RoI is: {RoI}")

            print(f"{IO.ANSI.YELLOW}Step 1: pre-processing step{IO.ANSI.ENDC}")
            start_time = time.time()
            if all(param is not None for param in [RoI, configs_output["output_folder"]]):
                output_folder_1 = f"{configs_output['output_folder']}/plots/{this_chr}/1_preprocessing/"
                LT_Iproc, UT_Iproc, Iproc_RoI = stripepy.step_1(
                    I,
                    configs_input["genomic_belt"],
                    configs_input["resolution"],
                    RoI=RoI,
                    output_folder=output_folder_1,
                )
            else:
                LT_Iproc, UT_Iproc, _ = stripepy.step_1(I, configs_input["genomic_belt"], configs_input["resolution"])
                Iproc_RoI = None
            print(f"Execution time of step 1: {time.time() - start_time} seconds ---")

            # Find the indices where the sum is zero
            # TODO: DO SOMETHING
            # zero_indices = np.where(np.sum(Iproc_RoI, axis=0) == 0)[0]
            # print(np.min(np.sum(LT_Iproc + UT_Iproc, axis=0)))
            # print(np.max(np.sum(LT_Iproc + UT_Iproc, axis=0)))
            # np.savetxt("trend.txt", np.sum(LT_Iproc + UT_Iproc, axis=0))
            # exit()

            print(f"{IO.ANSI.YELLOW}Step 2: Topological Data Analysis{IO.ANSI.ENDC}")
            # Create the output of this step:
            h5.create_group(f"{this_chr}/global-pseudo-distributions/LT/")
            h5.create_group(f"{this_chr}/global-pseudo-distributions/UT/")
            start_time = time.time()
            if all(param is not None for param in [Iproc_RoI, RoI, configs_output["output_folder"]]):
                output_folder_2 = f"{configs_output['output_folder']}/plots/{this_chr}/2_TDA/"
                pseudo_distributions, candidate_stripes = stripepy.step_2(
                    LT_Iproc,
                    UT_Iproc,
                    configs_input["resolution"],
                    configs_thresholds["glob_pers_min"],
                    h5[f"{this_chr}/global-pseudo-distributions/"],
                    Iproc_RoI=Iproc_RoI,
                    RoI=RoI,
                    output_folder=output_folder_2,
                )
            else:
                pseudo_distributions, candidate_stripes = stripepy.step_2(
                    LT_Iproc,
                    UT_Iproc,
                    configs_input["resolution"],
                    configs_thresholds["glob_pers_min"],
                    h5[f"{this_chr}/global-pseudo-distributions/"],
                )

            # TODO rea1991 Ideally, do not add chromosomes where no seed site is present
            if candidate_stripes is None:
                _create_empty_descriptor_dataset(h5, this_chr)
                print(f"Execution time of step 2: {time.time() - start_time} seconds ---")
                print(f"Chromosome is too sparse, no candidate returned")
                continue
            print(f"Execution time of step 2: {time.time() - start_time} seconds ---")

            print(f"{IO.ANSI.YELLOW}Step 3: Shape analysis{IO.ANSI.ENDC}")
            h5.create_group(f"{this_chr}/stripes/LT/")
            h5.create_group(f"{this_chr}/stripes/UT/")
            start_time = time.time()

            if all(param is not None for param in [Iproc_RoI, RoI, configs_output["output_folder"]]):
                output_folder_3 = f"{configs_output['output_folder']}/plots/{this_chr}/3_shape_analysis/"
                stripepy.step_3(
                    LT_Iproc,
                    UT_Iproc,
                    configs_input["resolution"],
                    configs_input["genomic_belt"],
                    configs_thresholds["max_width"],
                    configs_thresholds["constrain_heights"],
                    configs_thresholds["loc_pers_min"],
                    configs_thresholds["loc_trend_min"],
                    pseudo_distributions,
                    candidate_stripes,
                    h5[f"{this_chr}/stripes/"],
                    Iproc_RoI=Iproc_RoI,
                    RoI=RoI,
                    output_folder=output_folder_3,
                    map=pool.map if pool is not None else map,
                )
            else:
                stripepy.step_3(
                    LT_Iproc,
                    UT_Iproc,
                    configs_input["resolution"],
                    configs_input["genomic_belt"],
                    configs_thresholds["max_width"],
                    configs_thresholds["constrain_heights"],
                    configs_thresholds["loc_pers_min"],
                    configs_thresholds["loc_trend_min"],
                    pseudo_distributions,
                    candidate_stripes,
                    h5[f"{this_chr}/stripes/"],
                    map=pool.map if pool is not None else map,
                )

            print(f"Execution time of step 3: {time.time() - start_time} seconds ---")

            print(f"{IO.ANSI.YELLOW}Step 4: Statistical analysis and post-processing{IO.ANSI.ENDC}")
            start_time = time.time()

            if all(param is not None for param in [Iproc_RoI, RoI, configs_output["output_folder"]]):
                output_folder_4 = f"{configs_output['output_folder']}/plots/{this_chr}/4_biological_analysis/"
                thresholds_relative_change = np.arange(0.0, 15.2, 0.2)
                stripepy.step_4(
                    LT_Iproc,
                    UT_Iproc,
                    candidate_stripes,
                    h5[f"{this_chr}/stripes/"],
                    configs_input["resolution"],
                    thresholds_relative_change,
                    Iproc_RoI=Iproc_RoI,
                    RoI=RoI,
                    output_folder=output_folder_4,
                )
            else:
                stripepy.step_4(LT_Iproc, UT_Iproc, candidate_stripes, h5[f"{this_chr}/stripes/"])

            print(f"Execution time of step 4: {time.time() - start_time} seconds ---")

            print(f"{IO.ANSI.CYAN}This chromosome has taken {(time.time() - start_local_time)} seconds{IO.ANSI.ENDC}")

            # # Get all terminal groups within the file
            # # Recover all terminal group names:
            # terminal_group_names = []
            # h5.visititems(save_terminal_groups)
            # print("List of terminal group names:", terminal_group_names)
            #
            # print_all_attributes(h5)
            #
            # exit()

    print(f"\n\n{IO.ANSI.RED}The code has run for {(time.time() - start_global_time) / 60} minutes{IO.ANSI.ENDC}")
