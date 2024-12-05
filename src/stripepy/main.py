# Copyright (C) 2024 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import logging
import pathlib
from typing import Union

from .cli import call, download, setup, view


def _setup_mpl_backend():
    # This is very important, as some plotting operations are performed concurrently
    # using multiprocessing.
    # If the wrong backend is selected (e.g. tkinter) this can lead to the whole OS freezing
    import matplotlib

    matplotlib.use("Agg")


def _setup_logger(level: str, file: Union[pathlib.Path, None] = None):
    fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    try:
        if file is not None:
            file.parent.mkdir(parents=True, exist_ok=True)

        # TODO is it ok to overwrite existing logs?
        logging.basicConfig(filename=file, level=level, format=fmt)
        logging.getLogger().setLevel(level)
    except Exception as e:  # noqa

        logging.basicConfig(level=level, format=fmt)
        logging.getLogger().setLevel(level)

        if file is not None:
            logging.warning('failed to initialize log file "%s" for writing: %s', file, e)


def main():
    subcommand, args, verbosity = setup.parse_args()

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


if __name__ == "__main__":
    main()
