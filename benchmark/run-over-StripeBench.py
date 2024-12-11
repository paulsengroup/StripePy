#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import pathlib
import subprocess as sp
import sys
import time


# Create a custom formatter to allow multiline and bulleted descriptions
class CustomFormatter(argparse.RawTextHelpFormatter):
    def _fill_text(self, text, width, indent):
        return "".join([indent + line + "\n" for line in text.splitlines()])


def _input_dir_checked(arg: str) -> pathlib.Path:
    input_dir = pathlib.Path(arg)
    if input_dir.exists() and input_dir.is_dir():
        return pathlib.Path(arg)

    raise FileNotFoundError(f'Input folder "{arg}" is not reachable: folder does not exist')


def _output_dir_checked(arg: str) -> pathlib.Path:
    parent = pathlib.Path(arg).parent
    if parent.exists() and parent.is_dir():
        return pathlib.Path(arg)

    raise FileNotFoundError(f'Output folder "{arg}" is not reachable: parent folder does not exist')


def _probability(arg) -> float:
    if 0 <= (n := float(arg)) <= 1:
        return n

    raise ValueError("Not a valid probability")


def _num_cpus(arg: str) -> int:
    try:
        n = int(arg)
        if 0 < n <= mp.cpu_count():
            return n
    except:  # noqa
        pass

    raise ValueError(f"Not a valid number of CPU cores (allowed values are integers between 1 and {mp.cpu_count()})")


def make_cli():
    cli = argparse.ArgumentParser(
        description="stripepy's run over StripeBench ",
        formatter_class=CustomFormatter,
    )

    cli.add_argument("stripepy-exec", type=pathlib.Path)

    cli.add_argument(
        "stripebench-path",
        type=_input_dir_checked,
        help="Path to the StripeBench dataset, which can be downloaded from XX.",
        # TODO complete when benchmark is online
    )

    cli.add_argument(
        "-b",
        "--genomic-belt",
        type=int,
        default=5000000,
        help="Radius of the band, centred around the diagonal, where the search is restricted to "
        "(in bp). The value used for the StripeBench benchmark is here set as default.",
    )

    cli.add_argument(
        "-o",
        "--output-folder",
        type=_output_dir_checked,
        default=pathlib.Path("."),
        help="Path to the folder where the user wants the output to be placed (default: current folder).",
    )

    cli.add_argument(
        "--max-width",
        type=int,
        default=20000,
        help="Maximum stripe width, in bp.",
    )

    cli.add_argument(
        "--glob-pers-min",
        type=_probability,
        default=0.05,
        help="Threshold value between 0 and 1 to filter persistence maxima points and identify loci of interest "
        "(aka seeds). The value used for the StripeBench benchmark is here set as default.",
    )

    cli.add_argument(
        "--constrain-heights",
        action="store_true",
        default=True,
        help="Use peaks in signal to constrain the stripe height. The value used for the StripeBench benchmark "
        "is here set as default.",
    )

    cli.add_argument(
        "--loc-pers-min",
        type=_probability,
        default=0.25,
        help="Threshold value between 0 and 1 to find peaks in signal in a horizontal domain while estimating the "
        "height of a stripe. The value used for the StripeBench benchmark is here set as default.",
    )

    cli.add_argument(
        "--loc-trend-min",
        type=_probability,
        default=0.05,
        help="Threshold value between 0 and 1 to estimate the height of a stripe; the higher this value, the shorter "
        "the stripe; it is always used when --constrain-heights is set to 'False', but could be necessary also "
        "when --constrain-heights is 'True' and no persistent maximum other than the global maximum is found. "
        "The value used for the StripeBench benchmark is here set as default.",
    )

    cli.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing file(s).",
    )

    cli.add_argument(
        "-p",
        "--nproc",
        type=_num_cpus,
        default=1,
        help="Maximum number of parallel processes to use.",
    )

    return cli


def run_stripepy(
    stripepy_exec,
    path_to_mcool,
    resolution,
    genomic_belt,
    output_folder,
    max_width,
    glob_pers_min,
    constrain_heights,
    loc_pers_min,
    loc_trend_min,
    force,
    nproc,
):
    if force:
        sp.check_call(
            [
                stripepy_exec,
                "call",
                path_to_mcool,
                resolution,
                "-b",
                str(genomic_belt),
                "-o",
                output_folder,
                "--max-width",
                str(max_width),
                "--glob-pers-min",
                str(glob_pers_min),
                "--loc-pers-min",
                str(loc_pers_min),
                "--loc-trend-min",
                str(loc_trend_min),
                "-f",
            ]
        )
    else:
        sp.check_call(
            [
                stripepy_exec,
                "call",
                path_to_mcool,
                resolution,
                "-b",
                str(genomic_belt),
                "-o",
                output_folder,
                "--max-width",
                str(max_width),
                "--glob-pers-min",
                str(glob_pers_min),
                "--loc-pers-min",
                str(loc_pers_min),
                "--loc-trend-min",
                str(loc_trend_min),
            ]
        )


def main():
    args = vars(make_cli().parse_args())

    with open(f"{args["output_folder"]}/output.log", "w") as f:
        t0 = time.time()
        resolutions = ["5000", "10000", "25000", "50000"]
        contact_densities = ["1", "5", "10", "15"]
        noise_levels = ["0", "5000", "10000", "15000"]
        for contact_density in contact_densities:
            for noise_level in noise_levels:
                this_contact_map = (
                    args["stripebench-path"]
                    / "data"
                    / f"grch38_h1_rad21_{contact_density}_{noise_level}"
                    / f"grch38_h1_rad21_{contact_density}_{noise_level}.mcool"
                )
                for resolution in resolutions:
                    run_stripepy(
                        args["stripepy-exec"],
                        this_contact_map,
                        resolution,
                        args["genomic_belt"],
                        args["output_folder"],
                        args["max_width"],
                        args["glob_pers_min"],
                        args["constrain_heights"],
                        args["loc_pers_min"],
                        args["loc_trend_min"],
                        args["force"],
                        args["nproc"],
                    )
        delta = time.time() - t0
        print("Total time", file=f)
        print(delta, file=f)


if __name__ == "__main__":
    main()
