# Copyright (C) 2025 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT


class _DummyProgressBar(object):
    """
    A progress bar class that does nothing.
    """

    def __init__(self):
        pass

    def __enter__(self):
        def callable(*args, **kwargs):  # noqa
            pass

        return callable

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def initialize_progress_bar(*args, **kwargs):
    """
    Attempt to initialize a progress bar using alive_progress.
    In case of failure, return a dummy progress bar that does nothing.
    """
    # TODO switch to Rich
    try:
        import alive_progress

        return alive_progress.alive_bar(*args, **kwargs)
    except ImportError:
        return _DummyProgressBar()
