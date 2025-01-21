# Copyright (C) 2024 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import collections.abc
import importlib.util
import pathlib
import platform
import sys
from typing import List, Optional, Union

import hictkpy
import structlog

from .cli import call, download, plot, setup, view


def _setup_matplotlib(subcommand: str, **kwargs):
    if subcommand not in {"call", "plot"}:
        return

    if subcommand == "call" and "roi" not in kwargs:
        return

    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        structlog.get_logger().warning("failed to configure matplotlib")
        return

    # This is very important, as some plotting operations are performed concurrently
    # using multiprocessing.
    # If the wrong backend is selected (e.g. tkinter) this can lead to the whole OS freezing
    matplotlib.use("Agg")
    plt.set_loglevel(level="warning")


class _StructLogPlainStyles(object):
    def __init__(self):
        self.reset = ""
        self.bright = ""
        self.dim = ""

        self.level_critical = ""
        self.level_exception = ""
        self.level_error = ""
        self.level_warn = ""
        self.level_info = ""
        self.level_debug = ""
        self.level_notset = ""

        self.timestamp = ""
        self.chromosome = ""
        self.step = ""
        self.logger_name = ""


class _StructLogColorfulStyles(object):
    @staticmethod
    def _try_get_color(key: str):
        try:
            import colorama

            return eval(f"colorama.{key}")
        except ImportError:
            return ""

    def __init__(self):
        _try_get_color = _StructLogColorfulStyles._try_get_color
        self.reset = _try_get_color("Style.RESET_ALL")
        self.bright = _try_get_color("Style.BRIGHT")
        self.dim = _try_get_color("Style.DIM")

        self.level_critical = _try_get_color("Fore.RED")
        self.level_exception = _try_get_color("Fore.RED")
        self.level_error = _try_get_color("Fore.RED")
        self.level_warn = _try_get_color("Fore.YELLOW")
        self.level_info = _try_get_color("Fore.GREEN")
        self.level_debug = _try_get_color("Fore.GREEN")
        self.level_notset = _try_get_color("Back.RED")

        self.timestamp = self.dim
        self.chromosome = _try_get_color("Fore.BLUE")
        self.step = _try_get_color("Fore.BLUE")
        self.logger_name = _try_get_color("Fore.BLUE")


def _configure_logger_columns(
    colors: bool,
    level_styles: Optional[structlog.dev.Styles] = None,
    event_key: str = "event",
    timestamp_key: str = "timestamp",
    pad_level: bool = True,
    longest_chrom_name: str = "chr22",
    max_step_nest_levels: int = 3,
) -> List:
    """
    The body of this function is an extension of the structlog.dev.ConsoleRenderer:
    https://github.com/hynek/structlog/blob/a60ce7bbb50451ed786ace3c3893fb3a6a01df0a/src/structlog/dev.py#L433
    """

    if colors and importlib.util.find_spec("colorama") is None:
        return _configure_logger_columns(
            False, level_styles, event_key, timestamp_key, pad_level, longest_chrom_name, max_step_nest_levels
        )

    level_to_color = (
        structlog.dev.ConsoleRenderer().get_default_level_styles(colors) if level_styles is None else level_styles
    )

    if hasattr(structlog.dev, "_EVENT_WIDTH"):
        pad_event = structlog.dev._EVENT_WIDTH  # noqa
    else:
        pad_event = 30

    pad_chrom = len(longest_chrom_name)
    pad_step = len("step ") + max_step_nest_levels + (max_step_nest_levels - 1)

    level_width = 0 if not pad_level else None

    styles: structlog.Styles
    if colors:
        if platform.system() == "Windows":
            # Colorama must be init'd on Windows, but must NOT be
            # init'd on other OSes, because it can break colors.
            import colorama

            colorama.init()

        styles = _StructLogColorfulStyles()
    else:
        styles = _StructLogPlainStyles()

    def step_formatter(data):
        if isinstance(data, collections.abc.Sequence):
            return f"step {'.'.join(str(x) for x in data)}"
        return f"step {data}"

    return [
        structlog.dev.Column(
            timestamp_key,
            structlog.dev.KeyValueColumnFormatter(
                key_style=None,
                value_style=styles.timestamp,
                reset_style=styles.reset,
                value_repr=str,
            ),
        ),
        structlog.dev.Column(
            "level",
            structlog.dev.LogLevelColumnFormatter(level_to_color, reset_style=styles.reset, width=level_width),
        ),
        structlog.dev.Column(
            "chrom",
            structlog.dev.KeyValueColumnFormatter(
                key_style=None,
                value_style=styles.chromosome,
                reset_style=styles.reset,
                value_repr=str,
                width=pad_chrom,
                prefix="[",
                postfix="]",
            ),
        ),
        structlog.dev.Column(
            "step",
            structlog.dev.KeyValueColumnFormatter(
                key_style=None,
                value_style=styles.step,
                reset_style=styles.reset,
                value_repr=step_formatter,
                width=pad_step,
                prefix="[",
                postfix="]",
            ),
        ),
        structlog.dev.Column(
            event_key,
            structlog.dev.KeyValueColumnFormatter(
                key_style=None,
                value_style=styles.bright,
                reset_style=styles.reset,
                value_repr=str,
                width=pad_event,
            ),
        ),
        structlog.dev.Column(
            "",
            structlog.dev.KeyValueColumnFormatter(
                key_style=None, value_style=styles.dim, reset_style=styles.reset, value_repr=str
            ),
        ),
    ]


def _get_longest_chrom_name(path: pathlib.Path) -> str:
    try:
        chroms = hictkpy.MultiResFile(path).chromosomes(include_ALL=False).keys()
    except RuntimeError:
        try:
            chroms = hictkpy.File(path).chromosomes(include_ALL=False).keys()
        except RuntimeError:
            chroms = ("chrXX",)

    return max(chroms, key=len)  # noqa


def _setup_logger(
    level: str, file: Optional[pathlib.Path] = None, force: bool = False, matrix_file: Optional[pathlib.Path] = None
):
    # https://www.structlog.org/en/stable/standard-library.html#rendering-using-structlog-based-formatters-within-logging
    timestamper = structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S.%f")
    pre_chain = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.ExtraAdder(),
        timestamper,
    ]

    if matrix_file is not None:
        chrom = _get_longest_chrom_name(matrix_file)
    else:
        chrom = ""

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "plain": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processors": [
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.format_exc_info,
                    structlog.dev.ConsoleRenderer(
                        columns=_configure_logger_columns(colors=False, longest_chrom_name=chrom)
                    ),
                ],
                "foreign_pre_chain": pre_chain,
            },
            "colored": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processors": [
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.dev.ConsoleRenderer(
                        columns=_configure_logger_columns(colors=True, longest_chrom_name=chrom)
                    ),
                ],
                "foreign_pre_chain": pre_chain,
            },
        },
    }

    handlers = {
        "default": {
            "level": level,
            "class": "logging.StreamHandler",
            "formatter": "colored" if sys.stderr.isatty() else "plain",
        },
    }

    exception = None
    if file is not None:
        if file.exists() and not force:
            raise RuntimeError(f"Refusing to overwrite existing file {file}. Pass --force to overwrite.")
        try:
            file.parent.mkdir(parents=True, exist_ok=True)
            if file.exists():
                file.unlink()
            file.touch()

            handlers |= {
                "file": {
                    "level": "DEBUG",
                    "class": "logging.handlers.WatchedFileHandler",
                    "filename": file,
                    "formatter": "plain",
                },
            }

        except Exception as e:  # noqa
            exception = e

    config |= {
        "handlers": handlers,
        "loggers": {
            "": {
                "handlers": list(handlers.keys()),
                "level": "DEBUG",
                "propagate": True,
            },
        },
    }

    import logging.config

    logging.config.dictConfig(config)
    logging.captureWarnings(True)

    structlog.configure(
        cache_logger_on_first_use=True,
        wrapper_class=structlog.make_filtering_bound_logger(0),
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.processors.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

    if exception is not None:
        structlog.get_logger().warn('failed to initialize log file "%s" for writing: %s', file, exception)
    elif file is not None:
        assert file.is_file()
        structlog.get_logger().debug('successfully initialized log file "%s"', file)


def main(args: Union[List[str], None] = None):
    if args is None:
        args = sys.argv[1:]

    _setup_logger("INFO")
    try:
        subcommand, args, verbosity = setup.parse_args(args)
        _setup_matplotlib(subcommand, **args)

        if subcommand == "call":
            _setup_logger(
                verbosity.upper(),
                file=args.get("log_file"),
                force=args["force"],
                matrix_file=args["contact_map"],
            )
            return call.run(**args)
        if subcommand == "download":
            _setup_logger(verbosity.upper())
            return download.run(**args)
        if subcommand == "plot":
            _setup_logger(verbosity.upper())
            return plot.run(**args)
        if subcommand == "view":
            return view.run(**args)

        raise NotImplementedError

    except (RuntimeError, ImportError) as e:
        structlog.get_logger().exception(e)
        if args is not None:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
