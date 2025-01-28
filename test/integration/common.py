# Copyright (C) 2025 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT


import json
import pathlib
from typing import Sequence

import pytest

from stripepy.IO import ResultFile
from stripepy.IO import compare_result_files as _compare_result_files


def matplotlib_avail() -> bool:
    try:
        import matplotlib
    except ImportError:
        return False

    return True


def compare_result_files(reference: pathlib.Path, found: pathlib.Path, chroms: Sequence[str]):
    report = _compare_result_files(reference, found, chroms)
    if report["success"]:
        assert True
        return

    pytest.fail(json.dumps(report, indent=2))


def check_results_are_empty(found: pathlib.Path, chroms: Sequence[str]):
    with ResultFile(found) as h5:
        for chrom in chroms:
            assert h5[chrom].empty


def get_avail_cpu_cores() -> int:
    try:
        import multiprocessing
        import os

        if hasattr(os, "process_cpu_count"):
            return os.process_cpu_count()
        return os.cpu_count()
    except ImportError:
        return 1
