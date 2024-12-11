# Copyright (C) 2024 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import collections.abc
import pathlib
import platform
import sys
from typing import List, Optional, Union

import colorama
import structlog

from .cli import call, download, setup, view


def _setup_mpl_backend():
    # This is very important, as some plotting operations are performed concurrently
    # using multiprocessing.
    # If the wrong backend is selected (e.g. tkinter) this can lead to the whole OS freezing
    import matplotlib

    matplotlib.use("Agg")


class _StructLogColorfulStyles:
    reset = colorama.Style.RESET_ALL
    bright = colorama.Style.BRIGHT
    dim = colorama.Style.DIM

    level_critical = colorama.Fore.RED
    level_exception = colorama.Fore.RED
    level_error = colorama.Fore.RED
    level_warn = colorama.Fore.YELLOW
    level_info = colorama.Fore.GREEN
    level_debug = colorama.Fore.GREEN
    level_notset = colorama.Back.RED

    timestamp = dim
    chromosome = colorama.Fore.BLUE
    step = colorama.Fore.BLUE
    logger_name = colorama.Fore.BLUE


class _StructLogPlainStyles:
    reset = ""
    bright = ""
    dim = ""

    level_critical = ""
    level_exception = ""
    level_error = ""
    level_warn = ""
    level_info = ""
    level_debug = ""
    level_notset = ""

    timestamp = ""
    chromosome = ""
    step = ""
    logger_name = ""


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
            colorama.init()

        styles = _StructLogColorfulStyles
    else:
        styles = _StructLogPlainStyles

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


def _setup_logger(level: str, file: Union[pathlib.Path, None] = None):
    # https://www.structlog.org/en/stable/standard-library.html#rendering-using-structlog-based-formatters-within-logging
    timestamper = structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S.%f")
    pre_chain = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.ExtraAdder(),
        timestamper,
    ]

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "plain": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processors": [
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.format_exc_info,
                    structlog.dev.ConsoleRenderer(columns=_configure_logger_columns(colors=False)),
                ],
                "foreign_pre_chain": pre_chain,
            },
            "colored": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processors": [
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.dev.ConsoleRenderer(columns=_configure_logger_columns(colors=True)),
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
        logger = structlog.get_logger()
        logger.warn('failed to initialize log file "%s" for writing: %s', file, exception)


def main(args: Union[List[str], None] = None):
    if args is None:
        args = sys.argv[1:]

    _setup_logger("INFO")
    try:
        subcommand, args, verbosity = setup.parse_args(args)

        if subcommand == "call":
            _setup_mpl_backend()
            _setup_logger(verbosity.upper())
            return call.run(**args)
        if subcommand == "download":
            _setup_logger(verbosity.upper())
            return download.run(**args)
        if subcommand == "view":
            return view.run(**args)

        raise NotImplementedError

    except RuntimeError as e:
        logger = structlog.get_logger()
        logger.exception(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
