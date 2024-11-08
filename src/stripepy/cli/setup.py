import argparse
import pathlib
from importlib.metadata import version
from typing import Any, Dict, Tuple


# Create a custom formatter to allow multiline and bulleted descriptions
class CustomFormatter(argparse.RawTextHelpFormatter):
    def _fill_text(self, text, width, indent):
        return "".join([indent + line + "\n" for line in text.splitlines()])


def _existing_file(arg: str) -> pathlib.Path:
    if (path := pathlib.Path(arg)).is_file():
        return path

    raise FileNotFoundError(arg)


def _output_dir_checked(arg: str) -> pathlib.Path:
    path = pathlib.Path(arg).parent
    if path.exists() and path.is_dir():
        return path

    raise FileNotFoundError(f'Output folder "{path}" is not reachable: parent folder does not exist')


def _probability(arg) -> float:
    if 0 <= (n := float(arg)) <= 1:
        return n

    raise ValueError("Not a valid probability")


def _make_stripepy_call_subcommand(main_parser) -> argparse.ArgumentParser:
    sc: argparse.ArgumentParser = main_parser.add_parser(
        "call",
        help="stripepy works in four consecutive steps: \n"
        "• Step 1: Pre-processing\n"
        "• Step 2: Recognition of loci of interest (also called 'seeds')\n"
        "• Step 3: Shape analysis (i.e., width and height estimation)\n"
        "• Step 4: Signal analysis and post-processing\n",
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
        default="NONE",
        help="Normalization to fetch (default: 'NONE').",
    )

    sc.add_argument(
        "-b",
        "--genomic-belt",
        type=int,
        default=5000000,
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
        default=100000,
        help="Maximum stripe width, in bp.",
    )

    sc.add_argument(
        "--glob-pers-type",
        type=str,
        choices=["constant", "adaptive"],
        default="constant",
        help="Type of thresholding to filter persistence maxima points and identify loci of interest (aka seeds).",
    )

    sc.add_argument(
        "--glob-pers-min",
        type=_probability,
        default=None,
        help="Threshold value between 0 and 1 to filter persistence maxima points and identify loci of interest "
        "(aka seeds). The default value depends on the option --glob-pers-type (default: 0.2 if --glob-pers-type "
        "set to 'constant', 0.9 if --glob-pers-type set to 'adaptive').",
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
        default=0.2,
        help="Threshold value between 0 and 1 to find peaks in signal in a horizontal domain while estimating the "
        "height of a stripe; when --constrain-heights is set to 'False', it is not used (default: 0.2).",
    )

    sc.add_argument(
        "--loc-trend-min",
        type=_probability,
        default=0.1,
        help="Threshold value between 0 and 1 to estimate the height of a stripe (default: 0.1); "
        "the higher this value, the shorter the stripe; it is always used when --constrain-heights is set to "
        "'False', but could be necessary also when --constrain-heights is 'True' and no persistent maximum other "
        "than the global maximum is found.",
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

    cli.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=version("stripepy")),
    )

    return cli


def _process_stripepy_call_args(args: Dict[str, Any]) -> Dict[str, Any]:
    if args["glob_pers_min"] is None:
        if args["glob_pers_type"] == "constant":
            args["glob_pers_min"] = 0.2
        else:
            args["glob_pers_min"] = 0.9

    # Gather input parameters in dictionaries:
    configs_input = {key: args[key] for key in ["contact-map", "resolution", "normalization", "genomic_belt", "roi"]}
    configs_thresholds = {
        key: args[key]
        for key in [
            "glob_pers_type",
            "glob_pers_min",
            "constrain_heights",
            "loc_pers_min",
            "loc_trend_min",
            "max_width",
        ]
    }
    configs_output = {key: args[key] for key in ["output_folder"]}

    # Print the used parameters (chosen or default-ones):
    print("\nArguments:")
    print(f"--contact-map: {configs_input['contact-map']}")
    print(f"--resolution: {configs_input['resolution']}")
    print(f"--normalization: {configs_input['normalization']}")
    print(f"--genomic-belt: {configs_input['genomic_belt']}")
    print(f"--roi: {configs_input['roi']}")
    print(f"--max-width: {configs_thresholds['max_width']}")
    print(f"--glob-pers-type: {configs_thresholds['glob_pers_type']}")
    print(f"--glob-pers-min: {configs_thresholds['glob_pers_min']}")
    print(f"--constrain-heights: {configs_thresholds['constrain_heights']}")
    print(f"--loc-pers-min: {configs_thresholds['loc_pers_min']}")
    print(f"--loc-trend-min: {configs_thresholds['loc_trend_min']}")
    print(f"--output-folder: {configs_output['output_folder']}")

    return {"configs_input": configs_input, "configs_thresholds": configs_thresholds, "configs_output": configs_output}


def parse_args() -> Tuple[str, Any]:
    # Parse the input parameters:
    args = vars(_make_cli().parse_args())

    subcommand = args.pop("subcommand")
    if subcommand == "call":
        return subcommand, _process_stripepy_call_args(args)

    raise NotImplementedError
