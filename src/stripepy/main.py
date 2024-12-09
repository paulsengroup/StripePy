# Copyright (C) 2024 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import logging
import pathlib
import sys
from typing import List, Union

import structlog

from .cli import call, download, setup, view


def _setup_mpl_backend():
    # This is very important, as some plotting operations are performed concurrently
    # using multiprocessing.
    # If the wrong backend is selected (e.g. tkinter) this can lead to the whole OS freezing
    import matplotlib

    matplotlib.use("Agg")


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
                    structlog.dev.ConsoleRenderer(colors=False),
                ],
                "foreign_pre_chain": pre_chain,
            },
            "colored": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processors": [
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.dev.ConsoleRenderer(colors=True),
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
            _setup_logger(verbosity.upper(), args["configs_output"]["output_folder"] / "log.txt")
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
