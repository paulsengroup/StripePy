# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import argparse
import math
import multiprocessing as mp
import pathlib
from importlib.metadata import version
from typing import Any, Dict, List, Tuple, Union


def parse_args(cli_args: List[str]) -> Tuple[str, Any, str]:
    parser = _make_cli()

    # Parse the input parameters:
    args = vars(parser.parse_args(cli_args))
    args = _normalize_args(args)
    args = _define_default_args(parser, args)
    _validate_args(parser, args)

    subcommand = args.pop("subcommand")
    verbosity = args.pop("verbosity")
    return subcommand, args, verbosity


class _CustomFormatter(argparse.RawTextHelpFormatter):
    """
    A custom formatter that enables multiline and bulleted descriptions
    """

    def _fill_text(self, text, width, indent) -> str:
        return text


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


def _directory_is_empty(directory: Union[pathlib.Path, str]) -> bool:
    return not any(pathlib.Path(directory).iterdir())


def _probability(arg) -> float:
    if 0 <= (n := float(arg)) <= 1:
        return n

    raise argparse.ArgumentTypeError("Not a valid probability")


def _positive_float(arg) -> float:
    if (n := float(arg)) > 0:
        return n

    raise argparse.ArgumentTypeError("Not a positive float")


def _positive_int(arg) -> int:
    if (n := int(arg)) > 0:
        return n

    raise argparse.ArgumentTypeError("Not a positive int")


def _make_stripepy_call_subcommand(main_parser) -> argparse.ArgumentParser:
    sc: argparse.ArgumentParser = main_parser.add_parser(
        "call",
        help="stripepy works in four consecutive steps:\n"
        "• Step 1: Pre-processing\n"
        "• Step 2: Recognition of loci of interest (also called 'seeds')\n"
        "• Step 3: Shape analysis (i.e., width and height estimation)\n"
        "• Step 4: Signal analysis\n",
        formatter_class=_CustomFormatter,
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
        help="Normalization to fetch (default: %(default)s).",
    )

    sc.add_argument(
        "-b",
        "--genomic-belt",
        type=int,
        default=5_000_000,
        help="Radius of the band, centred around the diagonal, where the search is restricted to (in bp, default: %(default)s).",
    )

    sc.add_argument(
        "--roi",
        type=str,
        choices={"middle", "start"},
        help="Criterion used to select a region from each chromosome used to generate diagnostic plots (default: %(default)s).\n"
        "Requires --plot-dir.",
    )

    sc.add_argument(
        "-o",
        "--output-file",
        type=pathlib.Path,
        help="Path where to store the output HDF5 file.\n"
        "When not specified, the output file will be saved in the current working directory with a named based on the name of input matrix file.",
    )

    sc.add_argument(
        "--log-file",
        type=pathlib.Path,
        help="Path where to store the log file.",
    )

    sc.add_argument(
        "--plot-dir",
        type=pathlib.Path,
        help="Path where to store the output plots.\n"
        "Required when --roi is specified and ignored otherwise.\n"
        "If the specified folder does not already exist, it will be created.",
    )

    sc.add_argument(
        "--max-width",
        type=int,
        default=100_000,
        help="Maximum stripe width, in bp (default: %(default)s).",
    )

    sc.add_argument(
        "--glob-pers-min",
        type=_probability,
        default=0.04,
        help="Threshold value between 0 and 1 to filter persistence maxima points and identify loci of interest, "
        "aka seeds (default: %(default)s).",
    )

    sc.add_argument(
        "--constrain-heights",
        action="store_true",
        default=False,
        help="Use peaks in signal to constrain the stripe height (default: %(default)s).",
    )

    sc.add_argument(
        "-k",
        "--k-neighbour",
        type=_positive_int,
        dest="k",
        default=3,
        help="k for the k-neighbour, i.e., number of bins adjacent to the stripe boundaries on both sides (default: %(default)s).",
    )

    sc.add_argument(
        "--loc-pers-min",
        type=_probability,
        default=0.33,
        help="Threshold value between 0 and 1 to find peaks in signal in a horizontal domain while estimating the "
        "height of a stripe; when --constrain-heights is set to 'False', it is not used (default: %(default)s).",
    )

    sc.add_argument(
        "--loc-trend-min",
        type=_probability,
        default=0.25,
        help="Threshold value between 0 and 1 to estimate the height of a stripe (default: %(default)s); "
        "the higher this value, the shorter the stripe; it is always used when --constrain-heights is set to "
        "'False', but could be necessary also when --constrain-heights is 'True' and no persistent maximum other "
        "than the global maximum is found.",
    )

    sc.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing file(s) (default: %(default)s).",
    )

    sc.add_argument(
        "--verbosity",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set verbosity of output to the console (default: %(default)s).",
    )

    sc.add_argument(
        "-p",
        "--nproc",
        type=_num_cpus,
        default=1,
        help="Maximum number of parallel processes to use (default: %(default)s).",
    )

    sc.add_argument(
        "--min-chrom-size",
        type=int,
        default=2_000_000,
        help="Minimum size, in bp, for a chromosome to be analysed (default: %(default)s).",
    )

    return sc


def _make_stripepy_download_subcommand(main_parser) -> argparse.ArgumentParser:
    sc: argparse.ArgumentParser = main_parser.add_parser(
        "download",
        help="Helper command to simplify downloading datasets that can be used to test StripePy.",
        formatter_class=_CustomFormatter,
    )

    def get_avail_ref_genomes():
        from stripepy.cli.download import _get_datasets  # noqa

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
        help="Print the list of available datasets and return (default: %(default)s).",
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
        help="Include datasets used for internal testing (default: %(default)s).",
    )
    sc.add_argument(
        "--max-size",
        type=_positive_float,
        default=512.0,
        help="Upper bound for the size of the files to be considered when --name is not provided (default: %(default)s).",
    )
    sc.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        dest="output_path",
        help="Path where to store the downloaded file (default: %(default)s).",
    )
    sc.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing file(s) (default: %(default)s).",
    )
    sc.add_argument(
        "--verbosity",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set verbosity of output to the console (default: %(default)s).",
    )

    return sc


def _make_stripepy_plot_subcommand(main_parser) -> argparse.ArgumentParser:
    subparser = main_parser.add_parser(
        "plot",
        help="Generate various static plots useful to visually inspect the output produced by stripepy call.",
        formatter_class=_CustomFormatter,
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
            help="DPI of the generated plot (default: %(default)s; ignored when the output format is a vector graphic).",
        )
        sc.add_argument(
            "--seed",
            type=int,
            default=7606490399616306585,
            help="Seed for random number generation (default: %(default)s).",
        )
        sc.add_argument(
            "-n",
            "--normalization",
            type=str,
            help="Normalization to fetch (default: %(default)s).",
        )
        sc.add_argument(
            "-f",
            "--force",
            action="store_true",
            default=False,
            help="Overwrite existing file(s) (default: %(default)s).",
        )
        sc.add_argument(
            "--verbosity",
            type=str,
            choices=["debug", "info", "warning", "error", "critical"],
            default="info",
            help="Set verbosity of output to the console (default: %(default)s).",
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
        formatter_class=_CustomFormatter,
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
        help="Path to the .hdf5 generated by stripepy call.\n" "Required when highlighting stripes or seeds.",
    )

    sc.add_argument(
        "--relative-change-threshold",
        type=_positive_float,
        help="Cutoff for the relative change (default: %(default)s).\n"
        "Only used when highlighting architectural stripes.\n"
        "The relative change is computed as the ratio between the average number of interactions "
        "found inside a stripe and the number of interactions in a neighborhood outside of the stripe.",
    )

    grp = sc.add_mutually_exclusive_group()
    grp.add_argument(
        "--highlight-seeds",
        action="store_true",
        default=False,
        help="Highlight the stripe seeds (default: %(default)s).",
    )
    grp.add_argument(
        "--highlight-stripes",
        action="store_true",
        default=False,
        help="Highlight the architectural stripes (default: %(default)s).",
    )
    sc.add_argument(
        "--ignore-stripe-heights",
        action="store_true",
        default=False,
        help="Ignore the stripes height (default: %(default)s).\n"
        "Has no effect when --highlight-stripes is not specified.",
    )
    sc.add_argument(
        "--cmap",
        type=str,
        default="fruit_punch",
        help="Color map used to plot Hi-C interactions (default: %(default)s).\n"
        "Can be any of the color maps supported by matplotlib as well as: fall, fruit_punch, "
        "blues, acidblues, and nmeth.",
    )

    grp = sc.add_mutually_exclusive_group()
    grp.add_argument(
        "--linear-scale",
        action="store_false",
        dest="log_scale",
        help="Plot interactions in linear scale (default: False).",
    )
    grp.add_argument(
        "--log-scale",
        action="store_true",
        dest="log_scale",
        default=True,
        help="Plot interactions in log scale (default: %(default)s).",
    )

    add_common_options(sc, region_is_randomized=True)

    sc = subparser.add_parser(
        "pseudodistribution",
        help="Plot the pseudo-distribution over the given region of interest.",
        aliases=["pd"],
        formatter_class=_CustomFormatter,
    )
    add_stripepy_hdf5_option(sc)
    add_common_options(sc, region_is_randomized=True)

    sc = subparser.add_parser(
        "stripe-hist",
        help="Generate and plot the histograms showing the distribution of the stripe heights and widths.",
        aliases=["hist"],
        formatter_class=_CustomFormatter,
    )
    add_stripepy_hdf5_option(sc)
    add_common_options(sc, region_is_randomized=False)

    return sc


def _make_stripepy_view_subcommand(main_parser) -> argparse.ArgumentParser:
    sc: argparse.ArgumentParser = main_parser.add_parser(
        "view",
        help="Fetch stripes from the HDF5 file produced by stripepy call.",
        formatter_class=_CustomFormatter,
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
        help="Cutoff for the relative change (default: %(default)s).\n"
        "The relative change is computed as the ratio between the average number of interactions\n"
        "found inside a stripe and the number of interactions in a neighborhood outside of the stripe.",
    )

    sc.add_argument(
        "--with-biodescriptors",
        action="store_true",
        default=False,
        help="Include the stripe biodescriptors in the output.",
    )

    sc.add_argument(
        "--with-header",
        action="store_true",
        default=False,
        help="Include column names in the output.",
    )

    sc.add_argument(
        "--transform",
        type=str,
        choices={"transpose_to_ut", "transpose_to_lt", None},
        default=None,
        help="Control if and how stripe coordinates should be transformed (default: %(default)s).",
    )
    sc.add_argument(
        "--verbosity",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set verbosity of output to the console (default: %(default)s).",
    )

    return sc


def _make_cli() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        description="stripepy is designed to recognize linear patterns in contact maps (.hic, .mcool, .cool) "
        "through the geometric reasoning, including topological persistence and quasi-interpolation.",
        formatter_class=_CustomFormatter,
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


def _normalize_args(args: Dict[str, Any]) -> Dict[str, Any]:
    return {k.replace("-", "_"): v for k, v in args.items() if v is not None}


def _define_default_args(parser: argparse.ArgumentParser, args: Dict[str, Any]) -> Dict[str, Any]:
    if args["subcommand"] == "call":
        if "output_file" not in args:
            try:
                args["output_file"] = (
                    pathlib.Path(pathlib.Path(str(args["contact_map"]).partition("::")[0]).stem)
                    .with_suffix(f".{args['resolution']}.hdf5")
                    .absolute()
                )
            except Exception as e:  # noqa
                parser.error(f"failed to infer the output file name: {e}")

    return args


def _validate_stripepy_call_args(parser: argparse.ArgumentParser, args: Dict[str, Any]):
    if "roi" in args and "plot_dir" not in args:
        parser.error("--plot-dir is required when using --roi")

    if not args["force"]:
        path_collisions = []

        output_file = args["output_file"]
        log_file = args.get("log_file")
        plot_dir = args.get("plot_dir")

        if output_file.exists():
            path_collisions.append(f'refusing to overwrite existing file "{output_file}"')
        if log_file is not None and log_file.exists():
            path_collisions.append(f'refusing to overwrite existing file "{log_file}"')
        if plot_dir is not None and plot_dir.is_file():
            path_collisions.append(f'refusing to overwrite existing file "{plot_dir}"')
        elif plot_dir is not None and plot_dir.is_dir() and not _directory_is_empty(plot_dir):
            path_collisions.append(f'refusing to write in the non-empty directory "{plot_dir}"')

        num_collisions = len(path_collisions)
        if num_collisions == 1:
            parser.error(f"{path_collisions[0]}\n" "Pass --force to overwrite.")
        elif num_collisions > 1:
            path_collisions = "\n - ".join(path_collisions)
            parser.error(
                f"encountered the following {num_collisions} path collisions:\n"
                f" - {path_collisions}\n"
                "Pass --force to overwrite."
            )


def _validate_args(parser: argparse.ArgumentParser, args: Dict[str, Any]):
    if args["subcommand"] == "call":
        _validate_stripepy_call_args(parser, args)
