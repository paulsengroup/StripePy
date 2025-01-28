# Copyright (C) 2024 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import sys
from typing import List, Optional


def _setup_matplotlib(subcommand: str, **kwargs):
    if subcommand not in {"call", "plot"}:
        return

    if subcommand == "call" and "roi" not in kwargs:
        return

    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        import structlog

        structlog.get_logger().warning("failed to configure matplotlib")
        return

    # This is very important, as some plotting operations are performed concurrently
    # using multiprocessing.
    # If the wrong backend is selected (e.g. tkinter) this can lead to the whole OS freezing
    matplotlib.use("Agg")
    plt.set_loglevel(level="warning")


def main(args: Optional[List[str]] = None):
    # It is important that stripepy is not imported in the global namespace to enable coverage
    # collection when using multiprocessing
    from stripepy.cli import call, download, logging, plot, setup, view

    # Parse CLI args
    subcommand, kwargs, verbosity = setup.parse_args(sys.argv[1:] if args is None else args)

    # Set up the main logger
    with logging.ProcessSafeLogger(
        verbosity,
        path=kwargs.get("log_file"),
        force=kwargs.get("force"),
        matrix_file=kwargs.get("contact_map"),
        print_welcome_message=subcommand != "view",
    ) as main_logger:
        try:
            _setup_matplotlib(subcommand, **kwargs)
            kwargs["main_logger"] = main_logger

            # Call the appropriate entrypoint
            if subcommand == "call":
                return call.run(**kwargs, verbosity=verbosity)
            if subcommand == "download":
                return download.run(**kwargs)
            if subcommand == "plot":
                return plot.run(**kwargs)
            if subcommand == "view":
                return view.run(**kwargs)

            raise NotImplementedError

        except FileExistsError as e:
            import structlog

            # Do not print the full stack trace in case of FileExistsError
            # This make it easier to spot the names of the file(s) causing problems
            structlog.get_logger().error(e)

            if args is not None:
                raise
            return 1

        except (RuntimeError, ImportError) as e:
            import structlog

            # Log the exception including its stack trace
            structlog.get_logger().exception(e)
            if args is not None:
                raise
            return 1

        except Exception as e:  # noqa
            # Under normal operating conditions, StripePy should not raise exceptions other than
            # FileExistsError, RuntimeError, and ImportError.
            # Should that happen, log the exception with its stack trace and then re-raise it
            import structlog

            structlog.get_logger().exception(e)

            raise


if __name__ == "__main__":
    sys.exit(main())
