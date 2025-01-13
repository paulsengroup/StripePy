# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import argparse
import math
import multiprocessing as mp
import pathlib
from importlib.metadata import version
from typing import Any, Dict, List, Tuple


# Create a custom formatter to allow multiline and bulleted descriptions
class CustomFormatter(argparse.RawTextHelpFormatter):
    def _fill_text(self, text, width, indent):
        return "".join([indent + line + "\n" for line in text.splitlines()])


def _num_cpus(arg: str) -> int:
    try:
        n = int(arg)
        if 0 < n <= mp.cpu_count():
            return n
    except:  # noqa
        pass

    raise argparse.ArgumentTypeError(
        f"Not a valid number of CPU cores (allowed values are integers between 1 and {mp.cpu_count()})"
    )


def _existing_file(arg: str) -> pathlib.Path:
    if (path := pathlib.Path(arg)).is_file():
        return path

    raise argparse.ArgumentTypeError(f'Not an existing file: "{arg}"')


def _output_dir_checked(arg: str) -> pathlib.Path:
    parent = pathlib.Path(arg).parent
    if parent.exists() and parent.is_dir():
        return pathlib.Path(arg)

    raise argparse.ArgumentTypeError(f'Output folder "{arg}" is not reachable: parent folder does not exist')


def _probability(arg) -> float:
    if 0 <= (n := float(arg)) <= 1:
        return n

    raise argparse.ArgumentTypeError("Not a valid probability")


def _positive_float(arg) -> float:
    if (n := float(arg)) > 0:
        return n

    raise argparse.ArgumentTypeError("Not a positive float")


def _positive_int(arg) -> float:
    if (n := int(arg)) > 0:
        return n

    raise argparse.ArgumentTypeError("Not a positive int")


def _make_stripepy_call_subcommand(main_parser) -> argparse.ArgumentParser:
    sc: argparse.ArgumentParser = main_parser.add_parser(
        "call",
        help="stripepy works in four consecutive steps: \n"
        "• Step 1: Pre-processing\n"
        "• Step 2: Recognition of loci of interest (also called 'seeds')\n"
        "• Step 3: Shape analysis (i.e., width and height estimation)\n"
        "• Step 4: Signal analysis\n",
    )

    sc.add_argument(
        "contact-map",
        type=_existing_file,
        help="Path to a .cool, .mcool, or .hic file for input.",
    )

    sc.add_argument(
        "resolution",
        type=int,
        help="Resolution (in bp).",
    )

    sc.add_argument(
        "-n",
        "--normalization",
        type=str,
        help="Normalization to fetch (default: 'None').",
    )

    sc.add_argument(
        "-b",
        "--genomic-belt",
        type=int,
        default=5_000_000,
        help="Radius of the band, centred around the diagonal, where the search is restricted to (in bp, default: 5000000).",
    )

    sc.add_argument(
        "--roi",
        type=str,
        default=None,
        help="Specify 'middle' or input range as 'chr2:10000000-12000000' (default: None)",
    )

    sc.add_argument(
        "-o",
        "--output-folder",
        type=_output_dir_checked,
        default=pathlib.Path("."),
        help="Path to the folder where the user wants the output to be placed (default: current folder).",
    )

    sc.add_argument(
        "--max-width",
        type=int,
        default=100_000,
        help="Maximum stripe width, in bp.",
    )

    sc.add_argument(
        "--glob-pers-min",
        type=_probability,
        default=0.05,
        help="Threshold value between 0 and 1 to filter persistence maxima points and identify loci of interest, "
        "aka seeds (default: 0.2).",
    )

    sc.add_argument(
        "--constrain-heights",
        action="store_true",
        default=False,
        help="Use peaks in signal to constrain the stripe height (default: 'False')",
    )

    sc.add_argument(
        "--loc-pers-min",
        type=_probability,
        default=0.33,
        help="Threshold value between 0 and 1 to find peaks in signal in a horizontal domain while estimating the "
        "height of a stripe; when --constrain-heights is set to 'False', it is not used (default: 0.2).",
    )

    sc.add_argument(
        "--loc-trend-min",
        type=_probability,
        default=0.25,
        help="Threshold value between 0 and 1 to estimate the height of a stripe (default: 0.1); "
        "the higher this value, the shorter the stripe; it is always used when --constrain-heights is set to "
        "'False', but could be necessary also when --constrain-heights is 'True' and no persistent maximum other "
        "than the global maximum is found.",
    )

    sc.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing file(s).",
    )

    sc.add_argument(
        "--verbosity",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set verbosity of output to the console.",
    )

    sc.add_argument(
        "-p",
        "--nproc",
        type=_num_cpus,
        default=1,
        help="Maximum number of parallel processes to use.",
    )

    sc.add_argument(
        "--min-chrom-size",
        type=int,
        default=2_000_000,
        help="Minimum size, in bp, for a chromosome to be analysed (default: 2 Mbp).",
    )

    return sc


def _make_stripepy_download_subcommand(main_parser) -> argparse.ArgumentParser:
    sc: argparse.ArgumentParser = main_parser.add_parser(
        "download",
        help="Helper command to simplify downloading datasets that can be used to test StripePy.",
    )

    def get_avail_ref_genomes():
        from .download import _get_datasets

        return {
            record["assembly"]
            for record in _get_datasets(math.inf, include_private=False).values()
            if "assembly" in record
        }

    grp = sc.add_mutually_exclusive_group(required=False)
    grp.add_argument(
        "--assembly",
        type=str,
        choices=get_avail_ref_genomes(),
        help="Restrict downloads to the given reference genome assembly.",
    )
    grp.add_argument(
        "--name",
        type=str,
        help="Name of the dataset to be downloaded.\n"
        "When not provided, randomly select and download a dataset based on the provided CLI options (if any).",
    )
    grp.add_argument(
        "--list-only",
        action="store_true",
        default=False,
        help="Print the list of available datasets and return.",
    )

    grp_ = sc.add_mutually_exclusive_group(required=False)
    grp_.add_argument(
        "--unit-test",
        action="store_true",
        default=False,
        help="Download the test datasets required by the unit tests.\n"
        "Files will be stored under folder test/data/\n"
        "When specified, all other options are ignored.\n"
        "Existing files will be overwritten.",
    )
    grp_.add_argument(
        "--end2end-test",
        action="store_true",
        default=False,
        help="Download the test datasets required by the end2end tests.\n"
        "Files will be stored under folder test/data/\n"
        "When specified, all other options are ignored.\n"
        "Existing files will be overwritten.",
    )
    sc.add_argument(
        "--include-private",
        action="store_true",
        default=False,
        help="Include datasets used for internal testing.",
    )
    sc.add_argument(
        "--max-size",
        type=_positive_float,
        default=512.0,
        help="Upper bound for the size of the files to be considered when --name is not provided.",
    )
    sc.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        dest="output_path",
        help="Path where to store the downloaded file.",
    )
    sc.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing file(s).",
    )
    sc.add_argument(
        "--verbosity",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set verbosity of output to the console.",
    )

    return sc


def _make_stripepy_plot_subcommand(main_parser) -> argparse.ArgumentParser:
    subparser = main_parser.add_parser(
        "plot",
        help="Generate various static plots useful to visually inspect the output produced by stripepy call.",
    ).add_subparsers(title="plot_subcommands", dest="plot_type", required=True, help="List of available subcommands:")

    def add_common_options(sc, region_is_randomized: bool):
        sc.add_argument(
            "output-name",
            type=pathlib.Path,
            help="Path where to store the generated plot.",
        )

        region_help_msg1 = (
            "Genomic region to be plotted (UCSC format). When not specified, a random 2.5Mb region is plotted."
        )
        region_help_msg2 = (
            "Genomic region to be plotted (UCSC format). When not specified, data for the entire genome is plotted."
        )

        sc.add_argument(
            "--region",
            type=str,
            help=region_help_msg1 if region_is_randomized else region_help_msg2,
        )
        sc.add_argument(
            "--dpi",
            type=_positive_int,
            default=300,
            help="DPI of the generated plot (ignored when the output format is a vector graphic).",
        )
        sc.add_argument(
            "--seed",
            type=int,
            default=7606490399616306585,
            help="Seed for random number generation.",
        )
        sc.add_argument(
            "-n",
            "--normalization",
            type=str,
            help="Normalization to fetch (default: 'None').",
        )
        sc.add_argument(
            "-f",
            "--force",
            action="store_true",
            default=False,
            help="Overwrite existing file(s).",
        )
        sc.add_argument(
            "--verbosity",
            type=str,
            choices=["debug", "info", "warning", "error", "critical"],
            default="info",
            help="Set verbosity of output to the console.",
        )

    def add_stripepy_hdf5_option(sc):
        sc.add_argument(
            "stripepy-hdf5",
            type=_existing_file,
            help="Path to the .hdf5 generated by stripepy call.",
        )

    sc = subparser.add_parser(
        "contact-map",
        help="Plot stripes and other features over the Hi-C matrix.",
        aliases=["cm"],
    )
    sc.add_argument(
        "contact-map",
        type=_existing_file,
        help="Path to the .cool, .mcool, or .hic file used to call stripes.",
    )
    sc.add_argument(
        "resolution",
        type=_positive_int,
        help="Resolution (in bp).",
    )

    sc.add_argument(
        "--stripepy-hdf5",
        type=_existing_file,
        help="Path to the .hdf5 generated by stripepy call. Required when highlighting stripes or seeds.",
    )

    sc.add_argument(
        "--relative-change-threshold",
        type=_positive_float,
        help="Cutoff for the relative change.\n"
        "Only used when highlighting architectural stripes.\n"
        "The relative change is computed as the ratio between the average number of interactions "
        "found inside a stripe and the number of interactions in a neighborhood outside of the stripe.",
    )

    grp = sc.add_mutually_exclusive_group()
    grp.add_argument(
        "--highlight-seeds",
        action="store_true",
        default=False,
        help="Highlight the stripe seeds.",
    )
    grp.add_argument(
        "--highlight-stripes",
        action="store_true",
        default=False,
        help="Highlight the architectural stripes.",
    )
    sc.add_argument(
        "--ignore-stripe-heights",
        action="store_true",
        default=False,
        help="Ignore the stripes height. Has no effect when --highlight-stripes is not specified.",
    )
    sc.add_argument(
        "--cmap",
        type=str,
        default="fruit_punch",
        help="Color map used to plot Hi-C interactions.\n"
        "Can be any of the color maps supported by matplotlib as well as: fall, fruit_punch, "
        "blues, acidblues, and nmeth.",
    )

    grp = sc.add_mutually_exclusive_group()
    grp.add_argument(
        "--linear-scale",
        action="store_false",
        dest="log_scale",
        help="Plot interactions in linear scale.",
    )
    grp.add_argument(
        "--log-scale",
        action="store_true",
        dest="log_scale",
        default=True,
        help="Plot interactions in log scale.",
    )

    add_common_options(sc, region_is_randomized=True)

    sc = subparser.add_parser(
        "pseudodistribution",
        help="Plot the pseudo-distribution over the given region of interest.",
        aliases=["pd"],
    )
    add_stripepy_hdf5_option(sc)
    add_common_options(sc, region_is_randomized=True)

    sc = subparser.add_parser(
        "stripe-hist",
        help="Generate and plot the histograms showing the distribution of the stripe heights and widths.",
        aliases=["hist"],
    )
    add_stripepy_hdf5_option(sc)
    add_common_options(sc, region_is_randomized=False)

    return sc


def _make_stripepy_view_subcommand(main_parser) -> argparse.ArgumentParser:
    sc: argparse.ArgumentParser = main_parser.add_parser(
        "view",
        help="Fetch stripes from the HDF5 file produced by stripepy call.",
    )

    sc.add_argument(
        dest="h5_file",
        metavar="h5-file",
        type=_existing_file,
        help="Path to the HDF5 file generated by stripepy call.",
    )

    sc.add_argument(
        "--relative-change-threshold",
        type=float,
        default=5.0,
        help="Cutoff for the relative change.\n"
        "The relative change is computed as the ratio between the average number of interactions\n"
        "found inside a stripe and the number of interactions in a neighborhood outside of the stripe.",
    )

    sc.add_argument(
        "--transform",
        type=str,
        choices=["transpose_to_ut", "transpose_to_lt", None],
        default=None,
        help="Control if and how stripe coordinates should be transformed.",
    )
    sc.add_argument(
        "--verbosity",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set verbosity of output to the console.",
    )

    return sc


def _make_cli() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        description="stripepy is designed to recognize linear patterns in contact maps (.hic, .mcool, .cool) "
        "through the geometric reasoning, including topological persistence and quasi-interpolation. ",
        formatter_class=CustomFormatter,
    )

    sub_parser = cli.add_subparsers(
        title="subcommands", dest="subcommand", required=True, help="List of available subcommands:"
    )

    _make_stripepy_call_subcommand(sub_parser)
    _make_stripepy_download_subcommand(sub_parser)
    _make_stripepy_plot_subcommand(sub_parser)
    _make_stripepy_view_subcommand(sub_parser)

    cli.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=version("stripepy-hic")),
    )

    return cli


def _process_stripepy_call_args(args: Dict[str, Any]) -> Dict[str, Any]:

    # Gather input parameters in dictionaries:
    configs_input = {key: args[key] for key in ["contact_map", "resolution", "normalization", "genomic_belt", "roi"]}
    configs_thresholds = {
        key: args[key]
        for key in [
            "glob_pers_min",
            "constrain_heights",
            "loc_pers_min",
            "loc_trend_min",
            "max_width",
            "min_chrom_size",
        ]
    }
    configs_output = {key: args[key] for key in ["output_folder", "force"]}

    configs_output["output_folder"] = (
        configs_output["output_folder"] / configs_input["contact_map"].stem / str(configs_input["resolution"])
    )
    configs_other = {"nproc": args["nproc"]}

    return {
        "configs_input": configs_input,
        "configs_thresholds": configs_thresholds,
        "configs_output": configs_output,
        "configs_other": configs_other,
    }


def _normalize_args(args: Dict[str, Any]) -> Dict[str, Any]:
    return {k.replace("-", "_"): v for k, v in args.items()}


def parse_args(cli_args: List[str]) -> Tuple[str, Any, str]:
    # Parse the input parameters:
    args = _normalize_args(vars(_make_cli().parse_args(cli_args)))

    subcommand = args.pop("subcommand")
    verbosity = args.pop("verbosity")
    if subcommand == "call":
        return subcommand, _process_stripepy_call_args(args), verbosity

    return subcommand, args, verbosity
